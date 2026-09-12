from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable, P2TableAssembler
from dptb.data.interfaces.p23_table import P23VNAFactorAssembler
from dptb.nacf.assembly import NACFTableBank, NACFFeaturePlan, NACFBatchAssemblyPlan


def stores():
    species = {s: {'orbital_norb': 1, 'orbital_cutoff_bohr': 2., 'orbital_shells': [0],
                   'projector_norb': 1, 'projector_max_cutoff_bohr': 1.,
                   'projector_shells': [0], 'projector_cutoffs_bohr': [1.],
                   'vna_cutoff_bohr': 1., 'vna_projector_norb': 1}
               for s in ('X', 'Y')}
    def radial(value, support):
        r = np.linspace(0., support, 31)
        return RadialBlockTable(r, (value * (1 - r / support) ** 2)[:, None, None], (0,), (0,), support)
    p2 = SimpleNamespace(species=species)
    p2.onsite_component = lambda s, k: np.array([[2. if k == 'p2_base' else 1.]])
    p2.base_component = lambda a, b, k: radial(1. if k == 'p2_base' else .4, 4.)
    p2.projector = lambda a, b: radial(.2 if a == 'X' else .3, 3.)
    p2.d_eff = lambda s: np.array([[2. if s == 'X' else 3.]])
    p23 = SimpleNamespace(species=species, manifest_sha256='synthetic')
    p23.has_factor = lambda a, b: True
    p23.factor = lambda a, b: radial(.3 if a == 'X' else .4, 3.)
    p23.epsilon = lambda s: np.array([1.5 if s == 'X' else 2.])
    return p2, p23


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nacf_periodic_third_centres_and_overlap_match_cpu(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p2, p23 = stores()
    symbols = ['X', 'Y']
    positions = np.array([[0., 0., 0.], [1.3, .2, .1]])
    cell = np.eye(3) * 4.5
    edges = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    shifts = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    bank = NACFTableBank(p2, p23, device=device)
    plan = bank.prepare(symbols, positions, cell, edges, shifts)
    actual = plan()
    cpu = P2TableAssembler(p2)
    keys = [(0, 0, 0, 0, 0), (1, 1, 0, 0, 0)] + [(int(i), int(j), *s) for (i, j), s in zip(edges.T, shifts)]
    expected = np.stack([cpu.assemble_block(symbols=symbols, positions_bohr=positions, cell_bohr=cell, i=i, j=j, translation=s) for i, j, *s in keys])
    addition, _, _ = P23VNAFactorAssembler(p23, factor_dtype=np.float64).assemble_graph_addition(
        symbols=symbols, positions_bohr=positions, cell_bohr=cell, edge_index=edges,
        edge_cell_shift=shifts, node_shapes=np.ones((2, 2), dtype=int),
        edge_shapes=np.ones((4, 2), dtype=int), node_pad_shape=(1, 1), edge_pad_shape=(1, 1))
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(), expected[:2] * 13.605698 + addition, atol=2e-8)
    np.testing.assert_allclose(actual['edge_p2_ao_ev'].cpu(), expected[2:] * 13.605698, atol=1e-10)
    np.testing.assert_allclose(actual['node_overlap_ao'].cpu(), np.ones((2, 1, 1)), atol=1e-14)
    for row, ((i, j), shift) in enumerate(zip(edges.T, shifts)):
        ref = p2.base_component(symbols[i], symbols[j], 'overlap').evaluate(positions[j] + shift @ cell - positions[i])
        np.testing.assert_allclose(actual['edge_overlap_ao'][row].cpu(), ref, atol=1e-12)
    torch.testing.assert_close(actual['edge_p2_ao_ev'], actual['edge_p2_ao_ev'][plan.reverse].transpose(-1, -2), atol=0, rtol=0)
    # Evaluation cannot fall back to a file/table reader after preparation.
    p2.projector = lambda *args: pytest.fail('CPU table load during forward')
    p23.factor = p2.projector
    torch.testing.assert_close(plan()['node_p23_ao_ev'], actual['node_p23_ao_ev'])


def test_nacf_rejects_incomplete_graph_and_supports_molecule():
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cpu')
    with pytest.raises(ValueError, match='reverse'):
        bank.prepare(['X', 'Y'], [[0, 0, 0], [1, 0, 0]], np.eye(3), [[0], [1]], [[0, 0, 0]])
    plan = bank.prepare(['X'], [[0, 0, 0]], np.zeros((3, 3)), np.empty((2, 0), dtype=int), np.empty((0, 3), dtype=int), pbc=(False, False, False))
    result = plan()
    assert result['edge_p2_ao_ev'].shape == (0, 1, 1)
    # Self nonlocal projector survives; endpoint VNA addition is excluded.
    assert result['node_p23_ao_ev'].item() == pytest.approx((2 + .2 ** 2 * 2) * 13.605698)


def test_fused_gauge_and_rme_gather_matches_existing_packer():
    from dptb.data.transforms import OrbitalMapper
    from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
    from dptb.data.interfaces.blockwise_tensor import block_tensors_to_feature_tensors
    idp = OrbitalMapper({'H':'1s', 'C':'1s1p'}, method='e3tb')
    assembly = torch.nn.Module()
    assembly.symbols = ('H', 'C')
    assembly.width = 4
    assembly.positions = torch.zeros((2, 3), dtype=torch.float64)
    assembly.edge_index = torch.tensor([[0, 1], [1, 0]])
    assembly.bank = SimpleNamespace(p2=SimpleNamespace(species={'H':{'orbital_shells':[0]}, 'C':{'orbital_shells':[0, 1]}}))
    plan = NACFFeaturePlan(assembly, idp, output_dtype=torch.float64)
    node = torch.randn((2, 4, 4), generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    edge = node.flip(0).clone()
    actual = plan.pack(node, edge)
    converter = OrbAbacus2DeepTB()
    node_ref, edge_ref = torch.zeros_like(node), torch.zeros_like(edge)
    node_ref[0, :1, :1] = node[0, :1, :1]
    node_ref[1] = torch.from_numpy(converter.transform(node[1].numpy(), [0,1], [0,1]))
    edge_ref[0, :1, :] = torch.from_numpy(converter.transform(edge[0, :1, :].numpy(), [0], [0,1]))
    edge_ref[1, :, :1] = torch.from_numpy(converter.transform(edge[1, :, :1].numpy(), [0,1], [0]))
    data = {'atomic_numbers':torch.tensor([[1],[6]]), 'edge_index':assembly.edge_index}
    expected = block_tensors_to_feature_tensors(data, idp, node_blocks=node_ref, edge_blocks=edge_ref)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batch_plan_keeps_cells_queries_and_blocks_separate(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device=device)
    first = bank.prepare(['X','Y'], [[0,0,0],[1.3,.2,.1]], np.eye(3)*4.5,
                         [[0,1],[1,0]], [[0,0,0],[0,0,0]])
    second = bank.prepare(['Y','X'], [[.1,0,0],[1.8,-.2,.1]], np.eye(3)*5.2,
                          [[0,1],[1,0]], [[-1,0,0],[1,0,0]])
    molecule = bank.prepare(['X'], [[0,0,0]], np.zeros((3,3)), np.empty((2,0),dtype=int), np.empty((0,3),dtype=int), pbc=(False,False,False))
    single = [p() for p in (first,second,molecule)]
    merged = NACFBatchAssemblyPlan([first,second,molecule])()
    for key in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(merged[key],torch.cat([s[key] for s in single]),atol=1e-12,rtol=1e-12)
    assert merged['edge_index'].tolist() == [[0,1,2,3],[1,0,3,2]]


def test_geometry_api_ignores_labels_and_adds_nacf_once():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from dptb.data.transforms import OrbitalMapper
    from dptb.nacf.inference import NACFGeometryPredictor
    p2,p23=stores()
    p2.species={s:p2.species['X'] for s in ('H','C')}
    p23.species={s:p23.species['X'] for s in ('H','C')}
    bank=NACFTableBank(p2,p23,device='cpu')
    bank.p2_manifest_sha256='a'*64
    class ResidualModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.zeros(1))
            self.idp=OrbitalMapper({'H':'1s','C':'1s'},method='e3tb')
        def forward(self,data):
            assert not any(k in data for k in ('node_features','edge_features','node_h0','edge_h0','forces','energy'))
            # Mutate prior buffers deliberately: add-back must use a snapshot.
            data['node_features']=torch.full_like(data['node_p23'],2.)
            data['edge_features']=torch.full_like(data['edge_p2'],3.)
            data['node_p23'].zero_();data['edge_p2'].zero_()
            return data
    options={'embedding':{'method':'lem_moe_v3_prior_2b','prior_kind':'na_cf','r_max':2.}}
    model=ResidualModel()
    with pytest.raises(ValueError,match='fingerprint'):
        NACFGeometryPredictor(model,bank,options,target='full_h_minus_nacf',expected_p2_source_fingerprint='b'*64)
    predictor=NACFGeometryPredictor(model,bank,options,target='full_h_minus_nacf',expected_p2_source_fingerprint='a'*64)
    atoms=Atoms('HC',positions=[[0,0,0],[.7,.1,0]])
    atoms.calc=SinglePointCalculator(atoms,energy=-100,forces=np.ones((2,3)))
    atoms.new_array('node_h0',np.ones((2,1))*999)
    prepared=predictor.prepare([atoms,atoms.copy()])
    prior=prepared.plan()
    actual=prepared()
    torch.testing.assert_close(actual['node_features'],prior['node_p23']+2)
    torch.testing.assert_close(actual['edge_features'],prior['edge_p2']+3)
    torch.testing.assert_close(prepared()['node_features'],actual['node_features'])
    assert actual['ptr'].tolist()==[0,2,4]
    assert 'node_h0' not in actual
    with torch.inference_mode():
        actual['pos'].add_(100)
    assert prepared()['pos'].max() < 2


def test_partial_pbc_matches_explicit_vacuum_cell():
    p2,p23=stores()
    bank=NACFTableBank(p2,p23,device='cpu')
    args=(['X','Y'],[[0,0,0],[1.3,.2,.1]])
    edges=[[0,1],[1,0]]
    shifts=[[-1,0,0],[1,0,0]]
    partial=bank.prepare(*args,np.diag([4.5,0,0]),edges,shifts,pbc=(True,False,False))()
    vacuum=bank.prepare(*args,np.diag([4.5,50,50]),edges,shifts)()
    for k in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(partial[k],vacuum[k],atol=1e-12,rtol=1e-12)


def test_batch_pads_different_orbital_widths():
    p2,p23=stores()
    p2.species['Y']={**p2.species['Y'],'orbital_norb':4,'orbital_shells':[0,1]}
    def radial(left,right,value,support):
        distances=np.linspace(0,support,31)
        shape=(sum(2*l+1 for l in left),sum(2*l+1 for l in right))
        values=(1-distances/support)[:,None,None]**2*np.ones(shape)*value
        return RadialBlockTable(distances,values,left,right,support)
    p2.onsite_component=lambda s,k:np.eye(p2.species[s]['orbital_norb'])*(2 if k=='p2_base' else 1)
    p2.base_component=lambda a,b,k:radial(tuple(p2.species[a]['orbital_shells']),tuple(p2.species[b]['orbital_shells']),.2,4.)
    p2.projector=lambda a,b:radial((0,),tuple(p2.species[b]['orbital_shells']),.1,3.)
    p23.factor=lambda a,b:radial((0,),tuple(p2.species[b]['orbital_shells']),.3,3.)
    bank=NACFTableBank(p2,p23,device='cpu')
    first=bank.prepare(['X'],[[0,0,0]],np.zeros((3,3)),np.empty((2,0),int),np.empty((0,3),int),pbc=(False,False,False))
    second=bank.prepare(['Y','X'],[[0,0,0],[1.3,.2,0]],np.zeros((3,3)),[[0,1],[1,0]],[[0,0,0],[0,0,0]],pbc=(False,False,False))
    merged=NACFBatchAssemblyPlan([first,second])()
    singles=[first(),second()]
    for key in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        padded=[torch.nn.functional.pad(s[key],(0,4-s[key].shape[-1],0,4-s[key].shape[-2])) for s in singles]
        torch.testing.assert_close(merged[key],torch.cat(padded),atol=1e-12,rtol=1e-12)
