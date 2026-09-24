"""SOC NACF prior: spin-angular projectors, spinor D, assembly and packing, the CPU reference assembler, uu-real
completion, paired SOC inference and CLI routing."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nacf import cli, spinor_inference
from dptb.nacf.assembly import NACFBatchAssemblyPlan, NACFFeaturePlan, NACFTableBank
from dptb.nacf.soc import spin_angular_projector, spinor_d_matrix
from dptb.nacf.soc_cpu_reference import CPUScalarAlgorithmSOCAssembler, cpu_soc_blocks
from dptb.nacf.spinor_completion import SOCUURealCompletion
from dptb.nacf.spinor_inference import (PreparedSOCInference, SOCResidualPair, _residual_contract,
                                       _training_contract_config)
from dptb.tests._requires import requires_cuda, requires_nacf_prebuilt
from dptb.tests.nacf_support import RY_TO_EV, explicit_soc_blocks, stores

D_SPINOR = np.array([[2., .3 + .4j], [.3 - .4j, 3.]])


def soc_store(d_spinor):
    return SimpleNamespace(manifest={'source_p2_manifest_sha256': None}, d_spinor=d_spinor)


# --------------------------------------------------------------------------- projectors and spinor D
@pytest.mark.parametrize('l', [1, 2, 3])
def test_soc_projector_against_clebsch_gordan_and_sampled_harmonics(l):
    from sympy import S
    from sympy.physics.wigner import clebsch_gordan
    from dptb.data.interfaces.p2_table import real_sph_abacus, abacus_m_order, _complex_sph_harm
    rng = np.random.default_rng(67)
    xyz = rng.normal(size=(100, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    theta, phi = np.arccos(xyz[:, 2]), np.arctan2(xyz[:, 1], xyz[:, 0])
    yc = np.stack([_complex_sph_harm(l, m, theta, phi) for m in range(-l, l + 1)], axis=1)
    yr = np.stack([real_sph_abacus(l, m, xyz) for m in abacus_m_order(l)], axis=1)
    transform = np.kron(np.eye(2), np.linalg.lstsq(yc, yr, rcond=None)[0])
    projectors = []
    for twice_j in (2 * l - 1, 2 * l + 1):
        j = S(twice_j) / 2
        cg = np.array([[float(clebsch_gordan(l, S(1) / 2, j, m, ms, S(mj) / 2)) for mj in range(-twice_j, twice_j + 1, 2)]
                       for ms in (S(1) / 2, -S(1) / 2) for m in range(-l, l + 1)])
        actual = spin_angular_projector(l, float(j))
        np.testing.assert_allclose(actual, transform.conj().T @ (cg @ cg.T) @ transform, atol=2e-14)
        np.testing.assert_allclose(actual @ actual, actual, atol=2e-14)
        projectors.append(actual)
    np.testing.assert_allclose(sum(projectors), np.eye(2 * (2 * l + 1)), atol=1e-14)


def test_soc_d_spin_trace_radial_couplings_and_time_reversal():
    dij = np.array([[2., 0., 0.], [0., 3., .4], [0., .4, 4.]])
    d = spinor_d_matrix([1, 1, 1], [.5, 1.5, 1.5], dij, has_so=True)
    n = 9
    scalar = np.kron(dij * np.array([[1 / 3, 0, 0], [0, 2 / 3, 2 / 3], [0, 2 / 3, 2 / 3]]), np.eye(3))
    np.testing.assert_allclose((d[:n, :n] + d[n:, n:]) / 2, scalar, atol=1e-14)
    # The training prior is scalar D_eff. In real harmonics Lz is imaginary,
    # so real uu/dd reproduce that prior even though the complex blocks differ.
    np.testing.assert_allclose(d[:n, :n].real, scalar, atol=1e-14)
    np.testing.assert_allclose(d[n:, n:].real, scalar, atol=1e-14)
    rng = np.random.default_rng(193)
    left, right = rng.normal(size=(n, 7)), rng.normal(size=(n, 5))
    np.testing.assert_allclose((left.T @ d[:n, :n] @ right).real, left.T @ scalar @ right, atol=1e-13)
    np.testing.assert_allclose(d, d.conj().T, atol=1e-14)
    time_reversal = np.kron(np.array([[0, 1], [-1, 0]]), np.eye(n))
    np.testing.assert_allclose(time_reversal @ d.conj() @ time_reversal.T, d, atol=1e-14)
    assert np.max(abs(d.imag)) > .1 and np.max(abs(d[:n, n:])) > .1


# --------------------------------------------------------------------------- assembly and packing
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=requires_nacf_prebuilt)])
def test_soc_assembly_matches_independent_complex_contractions_and_batch(device):
    from dptb.data.interfaces.p23_table import P23VNAFactorAssembler
    p2, p23 = stores()
    matrices = {s: D_SPINOR * (1 if s == 'X' else 1.4) for s in ('X', 'Y')}
    bank = NACFTableBank(p2, p23, soc_store=soc_store(lambda s: matrices[s]), device=device)
    symbols, pos = ['X', 'Y'], np.array([[0., 0., 0.], [1.3, .2, .1]])
    edges, shifts = np.array([[0, 1], [1, 0]]), np.zeros((2, 3), int)
    plan = bank.prepare(symbols, pos, np.eye(3) * 10, edges, shifts, pbc=(False, False, False))
    actual = plan()
    expected = explicit_soc_blocks(p2, symbols, pos, [(0, 0), (1, 1), (0, 1), (1, 0)], lambda s: matrices[s]) * RY_TO_EV
    vna, _, _ = P23VNAFactorAssembler(p23, factor_dtype=np.float64).assemble_graph_addition(
        symbols=symbols, positions_bohr=pos, cell_bohr=np.eye(3) * 10, edge_index=edges, edge_cell_shift=shifts,
        node_shapes=np.ones((2, 2), int), edge_shapes=np.ones((2, 2), int), node_pad_shape=(1, 1), edge_pad_shape=(1, 1))
    expected[:2] += vna * np.eye(2)
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(), expected[:2], atol=2e-8)
    np.testing.assert_allclose(actual['edge_p2_ao_ev'].cpu(), expected[2:], atol=2e-8)
    second = bank.prepare(symbols, pos * 1.1, np.eye(3) * 11, edges, shifts, pbc=(False, False, False))
    batch = NACFBatchAssemblyPlan([plan, second])()
    for key in ('node_p23_ao_ev', 'edge_p2_ao_ev', 'node_overlap_ao', 'edge_overlap_ao'):
        torch.testing.assert_close(batch[key], torch.cat([actual[key], second()[key]]))


@pytest.mark.parametrize('budget', [0, 64 * 1024 * 1024])
def test_cpu_soc_assembler_retains_imaginary_and_spin_flip_blocks(budget):
    p2, p23 = stores()
    positions = np.array([[0., 0., 0.], [1.3, .2, .1]])
    symbols, cell = ['X', 'Y'], np.eye(3) * 10
    keys = [(0, 0), (1, 1), (0, 1), (1, 0)]
    result = CPUScalarAlgorithmSOCAssembler(p2, soc_store(lambda s: D_SPINOR), projector_overlap_cache_max_bytes=budget) \
        .assemble_sparse_blocks(symbols=symbols, positions_bohr=positions, cell_bohr=cell, block_keys=[(i, j, 0, 0, 0) for i, j in keys])
    expected = explicit_soc_blocks(p2, symbols, positions, keys, lambda s: D_SPINOR)
    for (i, j), block in zip(keys, expected):
        np.testing.assert_allclose(result[(i, j, 0, 0, 0)], block, atol=1e-13, rtol=1e-13)
        assert np.max(abs(result[(i, j, 0, 0, 0)].imag)) > 0
    bank = NACFTableBank(p2, p23, soc_store=soc_store(lambda s: D_SPINOR), device='cpu')
    edges, shifts = np.array([[0, 1], [1, 0]]), np.zeros((2, 3), dtype=int)
    reference = cpu_soc_blocks(bank, symbols, positions, cell, edges, shifts)
    actual = bank.prepare(symbols, positions, cell, edges, shifts)()
    for name in reference:
        torch.testing.assert_close(torch.from_numpy(reference[name]), actual[name].to(torch.complex128), atol=2e-8, rtol=1e-12)


@pytest.mark.parametrize('doubling', [True, False])
@pytest.mark.parametrize('mapping,backend,device', [('expanded', 'torch', 'cpu'), ('compact', 'torch', 'cpu'),
                                                    pytest.param('compact', 'cuda', 'cuda', marks=requires_nacf_prebuilt)])
@pytest.mark.parametrize('input_dtype', [torch.complex64, torch.complex128])
def test_soc_packing_four_spin_blocks_imaginary_and_unequal_widths(doubling, mapping, backend, device, input_dtype):
    from dptb.data.interfaces.ham_to_feature import block_to_feature
    from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
    idp = OrbitalMapper({'H': '1s', 'C': '1s1p'}, method='e3tb', has_soc=True,
                        full_soc_prediction=True, nextham_uureal_mask=False, soc_complex_doubling=doubling)
    assembly = torch.nn.Module()
    assembly.symbols, assembly.width = ('H', 'C'), 4
    assembly.positions = torch.zeros((2, 3), dtype=torch.float64, device=device)
    assembly.edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    assembly.bank = SimpleNamespace(soc=object(), p2=SimpleNamespace(species={'H': {'orbital_shells': [0]}, 'C': {'orbital_shells': [0, 1]}}))
    dtype = torch.float64 if doubling else torch.complex128
    plan = NACFFeaturePlan(assembly, idp, output_dtype=dtype, mapping=mapping, packing_backend=backend)
    rng = torch.Generator().manual_seed(24)
    node = torch.randn((2, 8, 8), generator=rng, dtype=input_dtype).transpose(1, 2)
    edge = torch.randn((2, 8, 8), generator=rng, dtype=input_dtype).transpose(1, 2)
    actual = plan.pack(node.to(device), edge.to(device))
    converter, blocks = OrbAbacus2DeepTB(), {}
    sizes, shells = [1, 4], [[0], [0, 1]]
    for pairs, array in [([(0, 0), (1, 1)], node), ([(0, 1), (1, 0)], edge)]:
        for row, (i, j) in enumerate(pairs):
            ii = np.r_[np.arange(sizes[i]), 4 + np.arange(sizes[i])]
            jj = np.r_[np.arange(sizes[j]), 4 + np.arange(sizes[j])]
            blocks[f'{i}_{j}_0_0_0'] = converter.transform(array[row].numpy()[np.ix_(ii, jj)], shells[i] * 2, shells[j] * 2)
    data = {'atomic_numbers': torch.tensor([[1], [6]]), 'edge_index': assembly.edge_index.cpu(), 'edge_cell_shift': torch.zeros((2, 3))}
    idp(data)
    block_to_feature(data, idp, blocks, output_dtype=dtype)
    torch.testing.assert_close(actual[0].cpu(), data['node_features'], rtol=0, atol=0)
    torch.testing.assert_close(actual[1].cpu(), data['edge_features'], rtol=0, atol=0)


# --------------------------------------------------------------------------- uu-real completion and paired inference
@pytest.mark.parametrize('doubling', [False, True])
def test_uureal_completion_retains_unlearned_soc_and_adds_residual_once(doubling):
    # Unequal orbital-pair sizes and a different dict insertion order catch a global reshape or dict-order
    # assumption. The expected entries are explicit.
    factor = 8 if doubling else 4
    compact = SimpleNamespace(has_soc=True, nextham_uureal_mask=True, basis={'H': ['1s', '1p']}, reduced_matrix_element=4,
                              orbpair_maps={'1s-1p': slice(1, 4), '1s-1s': slice(0, 1)}, get_orbpair_maps=lambda: None)
    full = SimpleNamespace(has_soc=True, nextham_uureal_mask=False, basis=compact.basis, reduced_matrix_element=4 * factor,
                           soc_complex_doubling=doubling, orbpair_maps={'1s-1s': slice(0, factor), '1s-1p': slice(factor, 4 * factor)},
                           get_orbpair_maps=lambda: None)
    mapping = SOCUURealCompletion(compact, full)
    prior = torch.arange(8 * factor, dtype=torch.float64).reshape(2, -1)
    if not doubling:
        prior = prior + 1j * (prior + 100)
    original = prior.clone()
    delta = torch.tensor([[.5, 1., 1.5, 2.], [3., 4., 5., 6.]], dtype=torch.float64)
    uu = [0, factor, factor + 1, factor + 2]
    dd = [3, factor + 9, factor + 10, factor + 11]
    expected = prior.clone()
    expected[:, uu] += delta
    expected[:, dd] += delta
    torch.testing.assert_close(mapping.extract_prior(prior), prior[:, uu].real)
    actual = mapping(prior, delta)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prior, original, rtol=0, atol=0)
    untouched = [i for i in range(4 * factor) if i not in uu + dd]
    torch.testing.assert_close(actual[:, untouched], prior[:, untouched], rtol=0, atol=0)
    if not doubling:
        torch.testing.assert_close(actual.imag, prior.imag, rtol=0, atol=0)
    with pytest.raises(ValueError, match='shape'):
        mapping(prior, delta[:, :3])
    with pytest.raises(ValueError, match='dtype'):
        mapping(prior, delta.to(torch.complex128))


class MutatingArm(torch.nn.Module):
    """A residual arm that mutates its conditioning and geometry, as an adversarial model would."""

    def __init__(self, mapper, node_delta, edge_delta):
        super().__init__()
        self.idp = mapper
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.node_delta, self.edge_delta = node_delta, edge_delta

    def forward(self, data):
        assert 'node_features' not in data and 'edge_features' not in data
        assert torch.count_nonzero(data['node_p23']) > 0
        data['node_p23'].zero_()
        data['pos'].add_(10)
        data['node_overlap'].zero_()
        data['node_features'] = torch.full_like(data['node_p23'], self.node_delta)
        data['edge_features'] = torch.full_like(data['edge_p2'], self.edge_delta)
        return data


@pytest.mark.parametrize('doubling', [True, False])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=requires_cuda)])
def test_paired_residual_full_soc_preserves_input_prior_overlap_and_repeated_calls(doubling, device):
    compact = OrbitalMapper({'H': '1s', 'C': '1s1p'}, method='e3tb', has_soc=True, nextham_uureal_mask=True)
    full = OrbitalMapper(copy.deepcopy(compact.basis), method='e3tb', has_soc=True, full_soc_prediction=True,
                         nextham_uureal_mask=False, soc_complex_doubling=doubling)
    completion = SOCUURealCompletion(compact, full, device=device)
    # the onsite arm contributes only node blocks and the hopping arm only edge blocks; the unused heads are wrong
    pair = SOCResidualPair(MutatingArm(compact, 2., -77.), MutatingArm(compact, -88., 3.)).to(device)
    width = full.reduced_matrix_element
    prior = torch.arange(1, 2 * width + 1, dtype=torch.float64, device=device).reshape(2, width)
    if not doubling:
        prior = prior + 1j * (prior + 100)
    features = {'node_p23': prior.clone(), 'edge_p2': prior.flip(0).clone(),
                'node_overlap': prior.clone() * .01, 'edge_overlap': prior.clone() * .02}
    geometry = {'pos': torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64), 'edge_index': torch.tensor([[0, 1], [1, 0]]),
                'batch': torch.zeros((2, 1), dtype=torch.long), 'node_features': torch.full((2, compact.reduced_matrix_element), 999.)}
    geometry = {k: v.to(device) for k, v in geometry.items()}
    pristine = {k: v.clone() for k, v in features.items()}
    original_geometry = {k: v.clone() for k, v in geometry.items()}
    prepared = PreparedSOCInference(pair, lambda: features, geometry, completion, ('node_p23', 'edge_p2'))
    expected_node, expected_edge = prior.clone(), prior.flip(0).clone()
    factor = 8 if doubling else 4
    # expected slices from the real mapper, independent of the module index buffers
    for span in full.orbpair_maps.values():
        n = (span.stop - span.start) // factor
        for offset in (0, 3 * n):
            expected_node[:, span.start + offset:span.start + offset + n] += 2.
            expected_edge[:, span.start + offset:span.start + offset + n] += 3.
    for _ in range(2):
        actual = prepared()
        torch.testing.assert_close(actual['node_features'], expected_node, rtol=0, atol=0)
        torch.testing.assert_close(actual['edge_features'], expected_edge, rtol=0, atol=0)
        for field in ('node_overlap', 'edge_overlap'):
            torch.testing.assert_close(actual[field], pristine[field], rtol=0, atol=0)
        torch.testing.assert_close(actual['pos'], original_geometry['pos'], rtol=0, atol=0)
        for key in pristine:
            torch.testing.assert_close(features[key], pristine[key], rtol=0, atol=0)
        for key in original_geometry:
            torch.testing.assert_close(geometry[key], original_geometry[key], rtol=0, atol=0)


def test_paired_arms_reject_different_species_type_order():
    first = OrbitalMapper({'H': '1s', 'C': '1s1p'}, method='e3tb', has_soc=True, nextham_uureal_mask=True)
    second = copy.deepcopy(first)
    second.chemical_symbol_to_type = {'C': 1, 'H': 0} if first.chemical_symbol_to_type == {'C': 0, 'H': 1} else {'C': 0, 'H': 1}
    with pytest.raises(ValueError, match='chemical_symbol_to_type'):
        SOCResidualPair(MutatingArm(first, 1., 1.), MutatingArm(second, 1., 1.))


def test_checkpoint_without_dataset_settings_requires_matching_nacf_sidecar(tmp_path):
    embedded = {'common_options': {'basis': {'H': '1s'}, 'has_soc': True, 'nextham_uureal_mask': True, 'full_soc_prediction': False},
                'model_options': {'embedding': {'method': 'lem_moe_v3_edge_h0', 'h0_node_key': 'node_p23', 'h0_edge_key': 'edge_p2'}}}
    with pytest.raises(ValueError, match='sidecar'):
        _training_contract_config(embedded, None)
    sidecar = copy.deepcopy(embedded)
    sidecar['data_options'] = {'train': {'prior_kind': 'na_cf', 'target_kind': 'nacfres', 'get_P2': True}}
    path = tmp_path / 'train_config.json'
    path.write_text(json.dumps(sidecar))
    assert _residual_contract(_training_contract_config(embedded, path)) == ('node_p23', 'edge_p2')
    sidecar['data_options']['train']['target_kind'] = 'h0res'
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='NACF'):
        _residual_contract(_training_contract_config(embedded, path))
    sidecar['model_options']['embedding']['h0_node_key'] = 'node_h0'
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='h0_node_key'):
        _training_contract_config(embedded, path)


# --------------------------------------------------------------------------- CLI routing
def cli_options(**extra):
    args = dict(checkpoint='onsite.pth', hopping_checkpoint='hopping.pth', soc='soc', p2='p2', p23='p23', overlap='overlap',
                expected_p2_sha256='pin', device='cpu', backend='torch', model_backend='checkpoint',
                onsite_config='onsite.json', hopping_config='hopping.json', soc_ry_to_ev=13.605693122994,
                p23_missing_policy='error', expected_p23_sha256=None)
    args.update(extra)
    return SimpleNamespace(**args)


def test_cli_routes_soc_to_the_paired_loader_with_the_training_contract(monkeypatch):
    """SOC must complete both arms through the paired loader and never go through the compact scalar loader."""
    captured, sentinel = {}, object()

    def paired(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return sentinel
    monkeypatch.setattr(spinor_inference, 'load_soc_predictor', paired)
    monkeypatch.setattr(cli, 'load_predictor', lambda *a, **k: pytest.fail('SOC must not use the scalar loader'))
    assert cli.load_cli_predictor(cli_options()) is sentinel
    assert captured['args'] == ('onsite.pth', 'hopping.pth')
    assert {k: captured['kwargs'][k] for k in ('onsite_config', 'hopping_config', 'soc', 'ry_to_ev')} == \
        {'onsite_config': 'onsite.json', 'hopping_config': 'hopping.json', 'soc': 'soc', 'ry_to_ev': 13.605693122994}


@pytest.mark.parametrize('changes,match', [
    ({'hopping_checkpoint': None}, 'hopping-checkpoint'),
    ({'soc': None}, '--soc'),
    ({'model_backend': 'reference'}, 'model backend'),
])
def test_cli_refuses_incomplete_or_changed_soc_contract_before_loading(changes, match):
    with pytest.raises(ValueError, match=match):
        cli.load_cli_predictor(cli_options(**changes))
