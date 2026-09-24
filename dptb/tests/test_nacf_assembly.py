"""NACF table-bank assembly: blocks match the independent CPU assemblers, batching, geometry invariances, the
missing-P23 policy, RME packing, fused contraction and the geometry predictor (CUDA parametrizations need the NACF
prebuilt; native-topology ones the topology library)."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.interfaces.p23_table import P23VNAFactorAssembler
from dptb.data.interfaces.p2_table import P2TableAssembler, RadialBlockTable
from dptb.data.transforms import OrbitalMapper
from dptb.nacf import inference
from dptb.nacf.assembly import NACFBatchAssemblyPlan, NACFFeaturePlan, NACFTableBank
from dptb.tests._requires import requires_cuda, requires_nacf_prebuilt, requires_nacf_topology
from dptb.tests.nacf_support import RY_TO_EV, stores

PRIOR_KEYS = ('node_p23_ao_ev', 'edge_p2_ao_ev', 'node_overlap_ao', 'edge_overlap_ao')
DEVICES = ['cpu', pytest.param('cuda', marks=requires_nacf_prebuilt)]
PACKING_ROUTES = [('expanded', 'torch', 'cpu'), ('compact', 'torch', 'cpu'),
                  pytest.param('compact', 'cuda', 'cuda', marks=requires_nacf_prebuilt)]
MOLECULE = (['X'], [[0, 0, 0]], np.zeros((3, 3)), np.empty((2, 0), dtype=int), np.empty((0, 3), dtype=int))


# --------------------------------------------------------------------------- assembled blocks
@pytest.mark.parametrize('device', DEVICES)
def test_periodic_third_centres_and_overlap_match_the_cpu_assemblers(device):
    p2, p23 = stores()
    symbols = ['X', 'Y']
    positions = np.array([[0., 0., 0.], [1.3, .2, .1]])
    cell = np.eye(3) * 4.5
    edges = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    shifts = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    plan = NACFTableBank(p2, p23, device=device).prepare(symbols, positions, cell, edges, shifts)
    actual = plan()
    cpu = P2TableAssembler(p2)
    keys = [(0, 0, 0, 0, 0), (1, 1, 0, 0, 0)] + [(int(i), int(j), *s) for (i, j), s in zip(edges.T, shifts)]
    expected = np.stack([cpu.assemble_block(symbols=symbols, positions_bohr=positions, cell_bohr=cell, i=i, j=j, translation=s)
                         for i, j, *s in keys])
    addition, _, _ = P23VNAFactorAssembler(p23, factor_dtype=np.float64).assemble_graph_addition(
        symbols=symbols, positions_bohr=positions, cell_bohr=cell, edge_index=edges, edge_cell_shift=shifts,
        node_shapes=np.ones((2, 2), dtype=int), edge_shapes=np.ones((4, 2), dtype=int), node_pad_shape=(1, 1), edge_pad_shape=(1, 1))
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(), expected[:2] * RY_TO_EV + addition, atol=2e-8)
    np.testing.assert_allclose(actual['edge_p2_ao_ev'].cpu(), expected[2:] * RY_TO_EV, atol=1e-10)
    np.testing.assert_allclose(actual['node_overlap_ao'].cpu(), np.ones((2, 1, 1)), atol=1e-14)
    for row, ((i, j), shift) in enumerate(zip(edges.T, shifts)):
        ref = p2.base_component(symbols[i], symbols[j], 'overlap').evaluate(positions[j] + shift @ cell - positions[i])
        np.testing.assert_allclose(actual['edge_overlap_ao'][row].cpu(), ref, atol=1e-12)
    torch.testing.assert_close(actual['edge_p2_ao_ev'], actual['edge_p2_ao_ev'][plan.reverse].transpose(-1, -2), atol=0, rtol=0)
    # evaluation cannot fall back to a file/table reader after preparation
    p2.projector = lambda *args: pytest.fail('CPU table load during forward')
    p23.factor = p2.projector
    torch.testing.assert_close(plan()['node_p23_ao_ev'], actual['node_p23_ao_ev'])


def test_rejects_incomplete_graph_and_supports_molecule():
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cpu')
    with pytest.raises(ValueError, match='reverse'):
        bank.prepare(['X', 'Y'], [[0, 0, 0], [1, 0, 0]], np.eye(3), [[0], [1]], [[0, 0, 0]])
    result = bank.prepare(*MOLECULE, pbc=(False, False, False))()
    assert result['edge_p2_ao_ev'].shape == (0, 1, 1)
    # the self nonlocal projector survives; the endpoint VNA addition is excluded
    assert result['node_p23_ao_ev'].item() == pytest.approx((2 + .2 ** 2 * 2) * RY_TO_EV)


@pytest.mark.parametrize('device', DEVICES)
def test_batch_plan_keeps_cells_queries_and_blocks_separate(device):
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device=device)
    first = bank.prepare(['X', 'Y'], [[0, 0, 0], [1.3, .2, .1]], np.eye(3) * 4.5, [[0, 1], [1, 0]], [[0, 0, 0], [0, 0, 0]])
    second = bank.prepare(['Y', 'X'], [[.1, 0, 0], [1.8, -.2, .1]], np.eye(3) * 5.2, [[0, 1], [1, 0]], [[-1, 0, 0], [1, 0, 0]])
    molecule = bank.prepare(*MOLECULE, pbc=(False, False, False))
    single = [p() for p in (first, second, molecule)]
    merged = NACFBatchAssemblyPlan([first, second, molecule])()
    for key in PRIOR_KEYS:
        torch.testing.assert_close(merged[key], torch.cat([s[key] for s in single]), atol=1e-12, rtol=1e-12)
    assert merged['edge_index'].tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]


def test_batch_pads_different_orbital_widths():
    p2, p23 = stores()
    p2.species['Y'] = {**p2.species['Y'], 'orbital_norb': 4, 'orbital_shells': [0, 1]}

    def radial(left, right, value, support):
        distances = np.linspace(0, support, 31)
        shape = (sum(2 * l + 1 for l in left), sum(2 * l + 1 for l in right))
        return RadialBlockTable(distances, (1 - distances / support)[:, None, None] ** 2 * np.ones(shape) * value, left, right, support)
    shells = lambda s: tuple(p2.species[s]['orbital_shells'])
    p2.onsite_component = lambda s, k: np.eye(p2.species[s]['orbital_norb']) * (2 if k == 'p2_base' else 1)
    p2.base_component = lambda a, b, k: radial(shells(a), shells(b), .2, 4.)
    p2.projector = lambda a, b: radial((0,), shells(b), .1, 3.)
    p23.factor = lambda a, b: radial((0,), shells(b), .3, 3.)
    bank = NACFTableBank(p2, p23, device='cpu')
    first = bank.prepare(*MOLECULE, pbc=(False, False, False))
    second = bank.prepare(['Y', 'X'], [[0, 0, 0], [1.3, .2, 0]], np.zeros((3, 3)), [[0, 1], [1, 0]], [[0, 0, 0], [0, 0, 0]],
                          pbc=(False, False, False))
    merged = NACFBatchAssemblyPlan([first, second])()
    singles = [first(), second()]
    for key in PRIOR_KEYS:
        padded = [torch.nn.functional.pad(s[key], (0, 4 - s[key].shape[-1], 0, 4 - s[key].shape[-2])) for s in singles]
        torch.testing.assert_close(merged[key], torch.cat(padded), atol=1e-12, rtol=1e-12)


def test_partial_pbc_matches_explicit_vacuum_cell():
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cpu')
    args = (['X', 'Y'], [[0, 0, 0], [1.3, .2, .1]])
    edges, shifts = [[0, 1], [1, 0]], [[-1, 0, 0], [1, 0, 0]]
    partial = bank.prepare(*args, np.diag([4.5, 0, 0]), edges, shifts, pbc=(True, False, False))()
    vacuum = bank.prepare(*args, np.diag([4.5, 50, 50]), edges, shifts)()
    for key in PRIOR_KEYS:
        torch.testing.assert_close(partial[key], vacuum[key], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('spinor', [False, True])
@pytest.mark.parametrize('topology,device', [
    ('python', 'cpu'),
    pytest.param('native', 'cpu', marks=requires_nacf_topology),
    pytest.param('native', 'cuda', marks=[requires_nacf_topology, requires_nacf_prebuilt]),
])
def test_triclinic_partial_periodic_atom_wrapping_preserves_prior_and_overlap(topology, device, spinor):
    """Re-wrapping atoms by lattice vectors (with the compensating edge shifts) leaves every block unchanged, and the
    native topology reproduces the Python topology on both geometries."""
    p2, p23 = stores()
    soc = SimpleNamespace(manifest={'source_p2_manifest_sha256': None},
                          d_spinor=lambda s: np.array([[2., .3 + .4j], [.3 - .4j, 3.]])) if spinor else None
    bank = NACFTableBank(p2, p23, device=device, soc_store=soc)
    cell = np.array([[3.2, 0., 0.], [1.3, 3.1, 0.], [0., 0., 0.]])
    pos = np.array([[.2, -.1, .3], [1.4, .2, -.1]])
    edges = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    shifts = np.array([[0, 0, 0], [0, 0, 0], [-1, 1, 0], [1, -1, 0]])
    reference = bank.prepare(['X', 'Y'], pos, cell, edges, shifts, pbc=(True, True, False))()
    wrap = np.array([[2, -1, 0], [-1, 2, 0]])
    geometries = [(pos + wrap @ cell, shifts + wrap[edges[0]] - wrap[edges[1]])]
    if topology == 'native':
        geometries.append((pos, shifts))
    for p, s in geometries:
        plan = bank.prepare(['X', 'Y'], p, cell, edges, s, pbc=(True, True, False), topology=topology)
        out = plan()
        for key in PRIOR_KEYS:
            torch.testing.assert_close(out[key], reference[key], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(plan.edge_index.cpu(), edges)


def test_training_ry_conversion_scales_p2_but_not_vna_or_overlap():
    p2, p23 = stores()
    geometry = (['X', 'Y'], [[0., 0., 0.], [1.3, .2, .1]], np.eye(3) * 10, [[0, 1], [1, 0]], np.zeros((2, 3), dtype=int))
    once = NACFTableBank(p2, p23, device='cpu', ry_to_ev=1.).prepare(*geometry)()
    twice = NACFTableBank(p2, p23, device='cpu', ry_to_ev=2.).prepare(*geometry)()
    p23.epsilon = lambda symbol: np.zeros(1)
    p2_only = NACFTableBank(p2, p23, device='cpu', ry_to_ev=1.).prepare(*geometry)()
    for name in ('node_p23_ao_ev', 'edge_p2_ao_ev'):
        torch.testing.assert_close(twice[name] - once[name], p2_only[name], atol=1e-12, rtol=1e-12)
    for name in ('node_overlap_ao', 'edge_overlap_ao'):
        torch.testing.assert_close(twice[name], once[name], atol=0, rtol=0)


# --------------------------------------------------------------------------- missing P23 pairs
PIN = 'a' * 64


def bank_with_missing_pair(device='cpu', **kwargs):
    p2, p23 = stores()
    p23.manifest_sha256 = PIN
    p23.has_factor = lambda a, b: (a, b) != ('X', 'Y')
    return NACFTableBank(p2, p23, device=device, backend='torch', **kwargs)


def pair_geometry(symbols):
    n = len(symbols)
    edges = np.array([[0, 1], [1, 0]]) if n == 2 else np.empty((2, 0), dtype=int)
    return (symbols, np.array([[i * 1.3, .2 * i, .1 * i] for i in range(n)]), np.eye(3) * 4.5, edges,
            np.zeros((edges.shape[1], 3), dtype=int))


def test_missing_p23_pair_requires_an_explicit_pinned_policy():
    with pytest.raises(ValueError, match='P23'):
        bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs')
    with pytest.raises(ValueError, match='fingerprint'):
        bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs', expected_p23_sha256='b' * 64)
    with pytest.raises(KeyError, match='X\\|Y'):
        bank_with_missing_pair().prepare(*pair_geometry(['X', 'Y']))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=requires_cuda)])
def test_pinned_fallback_matches_p2_and_a_mixed_batch_keeps_p23(device):
    bank = bank_with_missing_pair(device, p23_missing_policy='p2_if_missing_pairs', expected_p23_sha256=PIN)
    missing = bank.prepare(*pair_geometry(['X', 'Y']))
    complete = bank.prepare(*pair_geometry(['X', 'X']))
    assert not missing.p23_used and missing.p23_missing == ('X|Y',)
    assert complete.p23_used
    actual = missing()
    symbols, pos, cell, _, _ = pair_geometry(['X', 'Y'])
    cpu = P2TableAssembler(bank.p2)
    expected = np.stack([cpu.assemble_block(symbols=symbols, positions_bohr=pos, cell_bohr=cell, i=i, j=i, translation=(0, 0, 0))
                         for i in range(2)]) * bank.ry_to_ev
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(), expected, atol=2e-8)
    merged = NACFBatchAssemblyPlan([missing, complete])
    assert merged.p23_used == (False, True)
    combined = merged()
    complete_values = complete()
    for key in actual:
        expected = (torch.cat([actual[key], complete_values[key] + missing.natoms], dim=1) if key == 'edge_index'
                    else torch.cat([actual[key], complete_values[key]]))
        torch.testing.assert_close(combined[key], expected, atol=2e-8, rtol=1e-8)


def test_listed_but_missing_payload_does_not_enable_fallback():
    bank = bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs', expected_p23_sha256=PIN)

    def missing_file(*args):
        raise FileNotFoundError('listed P23 payload missing')
    bank.p23.factor = missing_file
    with pytest.raises(FileNotFoundError):
        bank.prepare(*pair_geometry(['X', 'X']))


# --------------------------------------------------------------------------- RME packing
@pytest.mark.parametrize('mapping,backend,device', PACKING_ROUTES)
def test_fused_gauge_and_rme_gather_matches_existing_packer(mapping, backend, device):
    from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
    from dptb.data.interfaces.blockwise_tensor import block_tensors_to_feature_tensors
    idp = OrbitalMapper({'H': '1s', 'C': '1s1p'}, method='e3tb')
    assembly = torch.nn.Module()
    assembly.symbols, assembly.width = ('H', 'C'), 4
    assembly.positions = torch.zeros((2, 3), dtype=torch.float64, device=device)
    assembly.edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    assembly.bank = SimpleNamespace(p2=SimpleNamespace(species={'H': {'orbital_shells': [0]}, 'C': {'orbital_shells': [0, 1]}}))
    plan = NACFFeaturePlan(assembly, idp, output_dtype=torch.float64, mapping=mapping, packing_backend=backend)
    node = torch.randn((2, 4, 4), generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    edge = node.flip(0).clone()
    actual = plan.pack(node.to(device), edge.to(device))
    converter = OrbAbacus2DeepTB()
    node_ref, edge_ref = torch.zeros_like(node), torch.zeros_like(edge)
    node_ref[0, :1, :1] = node[0, :1, :1]
    node_ref[1] = torch.from_numpy(converter.transform(node[1].numpy(), [0, 1], [0, 1]))
    edge_ref[0, :1, :] = torch.from_numpy(converter.transform(edge[0, :1, :].numpy(), [0], [0, 1]))
    edge_ref[1, :, :1] = torch.from_numpy(converter.transform(edge[1, :, :1].numpy(), [0, 1], [0]))
    data = {'atomic_numbers': torch.tensor([[1], [6]]), 'edge_index': assembly.edge_index.cpu()}
    expected = block_tensors_to_feature_tensors(data, idp, node_blocks=node_ref, edge_blocks=edge_ref)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)


@pytest.mark.parametrize('mapping,backend,device', PACKING_ROUTES)
def test_scalar_edge_packing_with_a_soc_bank(mapping, backend, device):
    mapper = OrbitalMapper({'H': '1s1p', 'He': '1s'}, has_soc=True, nextham_uureal_mask=True)
    bank = SimpleNamespace(soc=object(), p2=SimpleNamespace(species={'H': {'orbital_shells': [0, 1]}, 'He': {'orbital_shells': [0]}}))
    plan = SimpleNamespace(bank=bank, spinor_input=False, symbols=('H', 'He'), width=4,
                           positions=torch.zeros((2, 3), dtype=torch.float64, device=device),
                           edge_index=torch.tensor([[0, 1], [1, 0]], device=device))
    packer = NACFFeaturePlan(plan, mapper, mapping=mapping, packing_backend=backend)
    blocks = torch.zeros((2, 4, 4), dtype=torch.float64)
    blocks[0, :, 0] = torch.tensor([1., 2., 3., 4.])
    blocks[1] = blocks[0].T.clone()
    expected = np.zeros((2, mapper.reduced_matrix_element), dtype=np.float32)
    expected[:, mapper.orbpair_maps['1s-1s']] = 1.
    expected[0, mapper.orbpair_maps['1p-1s']] = [-4., 2., -3.]
    expected[1, mapper.orbpair_maps['1s-1p']] = [-4., 2., -3.]
    blocks = blocks.to(device)
    np.testing.assert_array_equal(packer.pack_edges(blocks).cpu().numpy(), expected)
    nodes = torch.zeros((2, 4, 4), dtype=torch.float64, device=device)
    torch.testing.assert_close(packer.pack(nodes, blocks)[1], packer.pack_edges(blocks), rtol=0, atol=0)


def test_old_native_binary_reports_rebuild(monkeypatch):
    from dptb.nacf import _cuda
    monkeypatch.setattr(_cuda, 'check_device', lambda device: None)
    monkeypatch.setattr(_cuda, 'extension', lambda: SimpleNamespace())
    blocks = SimpleNamespace(requires_grad=False, is_cuda=True, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError, match='rebuild'):
        _cuda.pack(blocks, None, None, None, None, torch.float32)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('empty', [False, True])
def test_compact_packing_repeated_types_empty_edges_and_ao_gradients(device, empty):
    e = 0 if empty else 513
    stub = SimpleNamespace(symbols=('H', 'H'), width=1, positions=torch.zeros((2, 3), dtype=torch.float64, device=device),
                           edge_index=torch.zeros((2, e), dtype=torch.long, device=device),
                           bank=SimpleNamespace(p2=SimpleNamespace(species={'H': {'orbital_shells': [0]}})))
    plan = NACFFeaturePlan(stub, OrbitalMapper({'H': '1s'}), mapping='compact',
                           packing_backend='cuda' if device == 'cuda' else 'torch', output_dtype=torch.float64)
    node = torch.tensor([1., -2.], device=device, dtype=torch.float64).reshape(2, 1, 1)
    edge = torch.arange(e, device=device, dtype=torch.float32).reshape(e, 1, 1)
    n, v = plan.pack(node, edge)
    torch.testing.assert_close(n[:, 0], node[:, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(v[:, 0], edge[:, 0, 0].double(), rtol=0, atol=0)
    edge.requires_grad_()
    plan.pack_edges(edge).sum().backward()
    torch.testing.assert_close(edge.grad, torch.ones_like(edge))


# --------------------------------------------------------------------------- fused contraction
@requires_nacf_prebuilt
@pytest.mark.parametrize('diagonal', [False, True])
def test_contraction_scatter_dense_diagonal_and_empty(diagonal):
    from dptb.nacf.fusion import contract_add
    torch.manual_seed(71)
    a = torch.randn(7, 9, 4, device='cuda', dtype=torch.float64)
    b = torch.randn(8, 9, 3, device='cuda', dtype=torch.float64)
    m = torch.randn((9,) if diagonal else (9, 9), device='cuda', dtype=torch.float64)
    rows = torch.tensor([[0, 1, 2], [2, 3, 5], [0, 2, 4], [2, 6, 7]], device='cuda')
    got = torch.zeros(3, 5, 5, device='cuda', dtype=torch.float64)
    ref = got.clone()
    ref[:, :4, :3].index_add_(0, rows[:, 0], a[rows[:, 1]].transpose(-1, -2) @ (torch.diag(m) if diagonal else m) @ b[rows[:, 2]])
    contract_add(a, m, b, rows, got)
    torch.testing.assert_close(got, ref, atol=1e-12, rtol=0)
    old = got.clone()
    contract_add(a, m, b, rows[:0], got)
    torch.testing.assert_close(got, old, atol=0, rtol=0)


@requires_nacf_prebuilt
@requires_nacf_topology
def test_fused_single_merged_and_edge_vna_assembly_equal_unfused():
    from dptb.nacf.fusion import enable_fusion
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cuda')
    g = dict(symbols=['X', 'Y'], positions_bohr=[[0., 0, 0], [1., .2, .1]], cell_bohr=np.eye(3) * 3.,
             edge_index=[[0, 1], [1, 0]], edge_cell_shift=np.zeros((2, 3), int))
    plans = [bank.prepare(**g, topology='native') for _ in range(2)]
    for plan in (plans[0], NACFBatchAssemblyPlan(plans), bank.prepare_edge_vna_batch([g, g])):
        ref = plan()
        enable_fusion(plan, contraction=True)
        got = plan()
        for k in ref:
            if ref[k].is_floating_point():
                torch.testing.assert_close(got[k], ref[k], atol=1e-12, rtol=0)


# --------------------------------------------------------------------------- geometry predictor
def test_geometry_api_ignores_labels_and_adds_nacf_once():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from dptb.nacf.inference import NACFGeometryPredictor
    p2, p23 = stores()
    p2.species = {s: p2.species['X'] for s in ('H', 'C')}
    p23.species = {s: p23.species['X'] for s in ('H', 'C')}
    bank = NACFTableBank(p2, p23, device='cpu')
    bank.p2_manifest_sha256 = 'a' * 64

    class ResidualModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.idp = OrbitalMapper({'H': '1s', 'C': '1s'}, method='e3tb')

        def forward(self, data):
            assert not any(k in data for k in ('node_features', 'edge_features', 'node_h0', 'edge_h0', 'forces', 'energy'))
            # mutate the prior buffers deliberately: the add-back must use a snapshot
            data['node_features'] = torch.full_like(data['node_p23'], 2.)
            data['edge_features'] = torch.full_like(data['edge_p2'], 3.)
            data['node_p23'].zero_()
            data['edge_p2'].zero_()
            return data
    options = {'embedding': {'method': 'lem_moe_v3_prior_2b', 'prior_kind': 'na_cf', 'r_max': 2.}}
    model = ResidualModel()
    with pytest.raises(ValueError, match='fingerprint'):
        NACFGeometryPredictor(model, bank, options, target='full_h_minus_nacf', expected_p2_source_fingerprint='b' * 64)
    predictor = NACFGeometryPredictor(model, bank, options, target='full_h_minus_nacf', expected_p2_source_fingerprint='a' * 64)
    atoms = Atoms('HC', positions=[[0, 0, 0], [.7, .1, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-100, forces=np.ones((2, 3)))
    atoms.new_array('node_h0', np.ones((2, 1)) * 999)
    prepared = predictor.prepare([atoms, atoms.copy()])
    prior = prepared.plan()
    actual = prepared()
    torch.testing.assert_close(actual['node_features'], prior['node_p23'] + 2)
    torch.testing.assert_close(actual['edge_features'], prior['edge_p2'] + 3)
    torch.testing.assert_close(prepared()['node_features'], actual['node_features'])
    assert actual['ptr'].tolist() == [0, 2, 4]
    assert 'node_h0' not in actual
    with torch.inference_mode():
        actual['pos'].add_(100)
    assert prepared()['pos'].max() < 2


@pytest.mark.parametrize('model_backend', ['checkpoint', 'reference'])
def test_loader_keeps_the_checkpoint_model_backend_unless_reference_is_explicit(tmp_path, monkeypatch, model_backend):
    options = {'embedding': {'method': 'lem_moe_v3_prior_2b', 'prior_kind': 'na_cf', 'so2_fusion_mode': 'streamed_m_major_fused_p0',
                             'mole_linear_mode': 'cublas_grouped', 'only2b': False, 'prior_init_scope': 'both'}}
    checkpoint = tmp_path / 'model.pth'
    torch.save({'config': {'model_options': options}, 'model_state_dict': {'weight': torch.ones(1)}}, checkpoint)
    captured = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.idp = SimpleNamespace(has_soc=False)

    def build_model(**kwargs):
        captured['options'] = copy.deepcopy(kwargs['model_options'])
        return Model()

    def table_bank(*args, **kwargs):
        captured['table_backend'] = kwargs['backend']
        return SimpleNamespace(p2_manifest_sha256='pinned', soc=None, _anchor=torch.empty(0, device=kwargs['device']))

    import dptb.nn
    monkeypatch.setattr(dptb.nn, 'build_model', build_model)
    monkeypatch.setattr(inference, 'P2TableStore', lambda path: object())
    monkeypatch.setattr(inference, 'P23VNAFactorTableStore', lambda path: object())
    monkeypatch.setattr(inference, 'OverlapTableStore', lambda path: object())
    monkeypatch.setattr(inference, 'NACFTableBank', table_bank)
    monkeypatch.setattr(inference, 'get_cutoffs_from_model_options', lambda opts: (3., 3., 3.))
    # omitting model_backend exercises the public default
    kwargs = {} if model_backend == 'checkpoint' else {'model_backend': 'reference'}
    predictor = inference.load_predictor(checkpoint, 'p2', 'p23', 'overlap', 'pinned', device='cpu', backend='torch', **kwargs)
    expected = copy.deepcopy(options)
    overrides = {}
    if model_backend == 'reference':
        overrides = {'so2_fusion_mode': 'streamed_m_major_ref', 'mole_linear_mode': 'split_loop'}
        expected['embedding'].update(overrides)
    assert captured['options'] == expected
    assert captured['table_backend'] == 'torch'
    assert predictor.runtime_model_overrides == overrides
    assert not predictor.model.training


def test_loader_rejects_unknown_model_backend_before_loading_checkpoint():
    with pytest.raises(ValueError, match='model_backend'):
        inference.load_predictor('absent.pth', 'p2', 'p23', 'overlap', 'pinned', model_backend='automatic_fallback')
