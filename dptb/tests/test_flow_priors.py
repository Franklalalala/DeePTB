"""HamiltonianCFM prior families: TE/typewise, basis_onsite, overlap_huckel,
external/dftb-alias, haar_dm; argcheck acceptance; the H0-key guard; and the
seeded-rng scope tied priors and CFM share.

Renamed and trimmed from test_flow_te_prior.py (merged with
test_flow_prior_seed_rng.py). Raw-uureal layout projection tests moved to
test_flow_core.py; the forced-clean-logging config case moved into
test_flow_core.py's accept/reject table.
"""
from __future__ import annotations

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.nnops import prior_physical
from dptb.nnops.flow import (
    HamiltonianCFM,
    _seeded_rng_scope,
    assert_flow_h0_keys_reach_model,
    build_hamiltonian_flow,
)
from dptb.tests._requires import requires_multi_gpu
from dptb.tests.flow_helpers import FakeIDP, FakeIr, FakeIrreps, build_cfm, make_batch
from dptb.utils.argcheck import common_options, flow_options


class _UnsortedFakeIrreps(FakeIrreps):
    def __init__(self):
        super().__init__([(1, FakeIr(1)), (1, FakeIr(0))])

    def sort(self):
        return (FakeIrreps([(1, FakeIr(0)), (1, FakeIr(1))]), None)


class _UnsortedIrrepIDP(FakeIDP):
    def __init__(self, *, device: torch.device):
        super().__init__(device=device)
        self.orbpair_irreps = _UnsortedFakeIrreps()


# ---------------------------------------------------------------------------
# External / TE prior mechanics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_prior_residual_path_is_unchanged(dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("zero", device=device, dtype=dtype)

    t = torch.tensor([0.25, 0.75], device=device, dtype=dtype)
    out, _ref, ctx = flow.prepare_batch(data, ref, t=t)

    node_t = torch.tensor([0.25, 0.25, 0.75], device=device, dtype=dtype).reshape(3, 1)
    edge_t = torch.tensor([0.25, 0.25, 0.25, 0.75], device=device, dtype=dtype).reshape(4, 1)
    expected_node = data[_keys.NODE_H0_KEY] + node_t * (
        ref[_keys.NODE_FEATURES_KEY] - data[_keys.NODE_H0_KEY]
    )
    expected_edge = data[_keys.EDGE_H0_KEY] + edge_t * (
        ref[_keys.EDGE_FEATURES_KEY] - data[_keys.EDGE_H0_KEY]
    )

    assert torch.allclose(ctx.node_prior, torch.zeros_like(ctx.node_prior))
    assert torch.allclose(ctx.edge_prior, torch.zeros_like(ctx.edge_prior))
    assert torch.allclose(out[_keys.NODE_H0_KEY], expected_node)
    assert torch.allclose(out[_keys.EDGE_H0_KEY], expected_edge)


def test_split_external_prior_can_mix_node_overlap_with_edge_h0():
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    node_overlap = data[_keys.NODE_H0_KEY] + 0.25
    data["node_overlap"] = node_overlap
    flow = build_cfm(
        "external", device=device, dtype=dtype, prior_node="external", prior_edge="external",
        prior_node_key="node_overlap", prior_edge_key=_keys.EDGE_H0_KEY,
        external_prior_strict=True,
    )

    model_data, _ref, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    torch.testing.assert_close(ctx.node_current, node_overlap)
    torch.testing.assert_close(ctx.edge_current, data[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(model_data[_keys.NODE_H0_KEY], node_overlap)
    torch.testing.assert_close(model_data[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY])


def test_external_prior_complex_source_requires_explicit_real_projection():
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    complex_source = (data[_keys.NODE_H0_KEY] + 0.25).to(torch.complex64) + 2.0j
    data["node_overlap"] = complex_source
    flow = build_cfm(
        "external", device=device, dtype=dtype, prior_node="external", prior_edge="external",
        prior_node_key="node_overlap", prior_edge_key=_keys.EDGE_H0_KEY,
        external_prior_strict=True,
    )

    with pytest.raises(TypeError, match="complex"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_external_prior_real_projection_requires_explicit_ablation(caplog):
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    complex_source = (data[_keys.NODE_H0_KEY] + 0.25).to(torch.complex64) + 2.0j
    data["node_overlap"] = complex_source
    flow = build_cfm(
        "external", device=device, dtype=dtype, prior_node="external", prior_edge="external",
        prior_node_key="node_overlap", prior_edge_key=_keys.EDGE_H0_KEY,
        external_prior_strict=True, allow_complex_prior_real_projection=True,
    )

    _out, _ref_out, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    torch.testing.assert_close(ctx.node_current, complex_source.real)
    assert "discarding imaginary channels" in caplog.text


@pytest.mark.parametrize("mode", ["irrep", "typewise"])
def test_te_prior_irrep_modes_fail_loud_on_layout_mismatch(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode=mode)
    flow.idp.orbpair_irreps = FakeIrreps([(1, FakeIr(1))])

    with pytest.raises(ValueError, match="orbpair_irreps raw feature spans"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_typewise_te_prior_projects_uureal_raw_layout_to_compressed_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, _ref = make_batch(device=device, dtype=dtype)
    ref = {
        _keys.NODE_FEATURES_KEY: data[_keys.NODE_H0_KEY]
        + torch.tensor(
            [[2.0, 4.0, 3.0, 5.0], [0.0, 0.0, 0.0, 0.0], [6.0, 8.0, 7.0, 9.0]],
            device=device, dtype=dtype,
        ),
        _keys.EDGE_FEATURES_KEY: data[_keys.EDGE_H0_KEY],
    }
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode="typewise")
    flow.idp.orbpair_irreps = FakeIrreps(
        [(1, FakeIr(1)), (1, FakeIr(0)), (1, FakeIr(1)), (1, FakeIr(0))]
    )
    flow.idp.mask_uureal = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    flow.idp.mask_to_nrme = torch.tensor(
        [[1, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], device=device, dtype=torch.bool
    )
    flow.idp.mask_to_erme = torch.zeros((2, 8), device=device, dtype=torch.bool)

    def unit_radius(row_count, active_dim, graph_index, *, device, dtype):
        return active_dim.to(device=device, dtype=dtype).sqrt()

    flow._te_radius = unit_radius

    torch.manual_seed(29)
    _out, _ref_out, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    type0_rows = data[AtomicDataDict.ATOM_TYPE_KEY] == 0
    first_l1_scale = torch.tensor(
        [2.0, 4.0, 6.0, 8.0], device=device, dtype=dtype
    ).square().mean().sqrt()
    l0_scale = torch.tensor([3.0, 7.0], device=device, dtype=dtype).square().mean().sqrt()
    second_l1_scale = torch.tensor([5.0, 9.0], device=device, dtype=dtype).square().mean().sqrt()

    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[0, 0:2]),
        first_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 2:3]), l0_scale)
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 3:4]), second_l1_scale)
    assert torch.all(ctx.node_prior[type0_rows, 0:2] != 0)
    assert torch.all(ctx.node_prior[type0_rows, 2:4] != 0)
    assert torch.all(ctx.node_prior[~type0_rows] == 0)
    assert torch.all(ctx.edge_prior == 0)


def test_typewise_te_prior_uses_unsorted_raw_slices_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode="typewise")
    flow.idp = _UnsortedIrrepIDP(device=device)
    flow.idp.mask_to_nrme = torch.tensor(
        [[1, 1, 0, 1], [0, 0, 0, 0]], device=device, dtype=torch.bool
    )
    flow.idp.mask_to_erme = torch.zeros((2, 4), device=device, dtype=torch.bool)

    def unit_radius(row_count, active_dim, graph_index, *, device, dtype):
        return active_dim.to(device=device, dtype=dtype).sqrt()

    flow._te_radius = unit_radius

    torch.manual_seed(23)
    _out, _ref_out, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    node_res = ref[_keys.NODE_FEATURES_KEY] - data[_keys.NODE_H0_KEY]
    type0_rows = data[AtomicDataDict.ATOM_TYPE_KEY] == 0
    raw_l1_scale = node_res[type0_rows][:, :3][:, [0, 1]].square().mean().sqrt()
    raw_l0_scale = node_res[type0_rows][:, 3].square().mean().sqrt()

    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[0, :3]),
        raw_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 3:4]), raw_l0_scale)
    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[2, :3]),
        raw_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[2, 3:4]), raw_l0_scale)
    assert torch.all(ctx.node_prior[type0_rows, 2] == 0)
    assert torch.all(ctx.node_prior[~type0_rows] == 0)
    assert torch.all(ctx.edge_prior == 0)


def test_block_te_alias_defaults_to_block_mode_unless_explicit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    flow = build_cfm("block-te", device=device, dtype=dtype)
    explicit = build_cfm("block_te", device=device, dtype=dtype, te_prior_mode="irrep")

    assert flow.prior == "block_te"
    assert flow.te_prior_mode == "block"
    assert explicit.te_prior_mode == "irrep"


def test_te_prior_produces_nonzero_residual_noise():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode="irrep")

    torch.manual_seed(7)
    out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert ctx.node_prior.shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert ctx.edge_prior.shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert ctx.node_prior.dtype == dtype
    assert ctx.edge_prior.device == ref[_keys.EDGE_FEATURES_KEY].device
    assert torch.count_nonzero(ctx.node_prior).item() > 0
    assert torch.count_nonzero(ctx.edge_prior).item() > 0
    assert torch.allclose(out[_keys.NODE_H0_KEY], data[_keys.NODE_H0_KEY] + ctx.node_prior)
    assert torch.allclose(out[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY] + ctx.edge_prior)


def test_te_prior_respects_node_edge_masks_and_active_rows():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    data["expert_node_mask"] = torch.tensor([1, 0, 1], device=device, dtype=torch.bool)
    data["expert_edge_mask"] = torch.tensor([1, 1, 0, 1], device=device, dtype=torch.bool)
    flow = build_cfm("structured_te", device=device, dtype=dtype, te_prior_mode="irrep")

    torch.manual_seed(11)
    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert torch.all(ctx.node_prior[1] == 0)
    assert torch.all(ctx.edge_prior[1, [0, 3]] == 0)
    assert torch.count_nonzero(ctx.edge_prior[1, [1, 2]]).item() > 0
    assert torch.all(ctx.edge_prior[2] == 0)


def test_te_prior_mask_alignment_fails_closed_for_present_but_short_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    data[AtomicDataDict.ATOM_TYPE_KEY] = data[AtomicDataDict.ATOM_TYPE_KEY][:2]
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode="irrep")
    flow.idp.mask_to_nrme = flow.idp.mask_to_nrme[:, :2]

    torch.manual_seed(17)
    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert torch.all(ctx.node_prior[2] == 0)
    assert torch.all(ctx.node_prior[:, 2:] == 0)


def test_te_prior_is_reproducible_under_deterministic_seed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm(
        "te", device=device, dtype=dtype, te_prior_mode="irrep", te_prior_per_graph=True
    )
    t = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

    torch.manual_seed(1234)
    _out1, _ref1, ctx1 = flow.prepare_batch(data, ref, t=t)
    torch.manual_seed(1234)
    _out2, _ref2, ctx2 = flow.prepare_batch(data, ref, t=t)
    torch.manual_seed(4321)
    _out3, _ref3, ctx3 = flow.prepare_batch(data, ref, t=t)

    assert torch.allclose(ctx1.node_prior, ctx2.node_prior)
    assert torch.allclose(ctx1.edge_prior, ctx2.edge_prior)
    assert not torch.allclose(ctx1.node_prior, ctx3.node_prior)


def test_te_prior_sampling_interface_keeps_existing_keys():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("te", device=device, dtype=dtype, te_prior_mode="block")

    class _EndpointEcho(torch.nn.Module):
        def forward(self, batch):
            out = batch.copy()
            out[_keys.NODE_FEATURES_KEY] = batch[_keys.NODE_H0_KEY]
            out[_keys.EDGE_FEATURES_KEY] = batch[_keys.EDGE_H0_KEY]
            return out

    torch.manual_seed(99)
    sampled = flow.sample(_EndpointEcho(), data, num_steps=1)

    assert sampled[_keys.NODE_H0_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_H0_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert sampled[_keys.NODE_FEATURES_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_FEATURES_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert torch.allclose(sampled[flow.flow_time_key], torch.ones(2, device=device, dtype=dtype))


def test_full_mode_external_sampling_uses_absolute_prior_once():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, _ref = make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "full", "prior": "external",
            "prior_node_key": _keys.NODE_H0_KEY, "prior_edge_key": _keys.EDGE_H0_KEY,
        },
        idp=FakeIDP(device=device), device=device, dtype=dtype,
    )

    class _EchoCurrentH0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_node_h0 = None
            self.first_edge_h0 = None

        def forward(self, batch):
            if self.first_node_h0 is None:
                self.first_node_h0 = batch[_keys.NODE_H0_KEY].detach().clone()
                self.first_edge_h0 = batch[_keys.EDGE_H0_KEY].detach().clone()
            out = batch.copy()
            out[_keys.NODE_FEATURES_KEY] = batch[_keys.NODE_H0_KEY]
            out[_keys.EDGE_FEATURES_KEY] = batch[_keys.EDGE_H0_KEY]
            return out

    model = _EchoCurrentH0()

    sampled = flow.sample(model, data, num_steps=1)

    torch.testing.assert_close(model.first_node_h0, data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(model.first_edge_h0, data[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(sampled[_keys.NODE_FEATURES_KEY], data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(sampled[_keys.EDGE_FEATURES_KEY], data[_keys.EDGE_H0_KEY])


# ---------------------------------------------------------------------------
# basis_onsite / overlap_huckel physical priors
# ---------------------------------------------------------------------------


def _basis_onsite_case(device: torch.device, dtype: torch.dtype):
    from dptb.data.transforms import OrbitalMapper

    idp = OrbitalMapper({"H": ["1s"], "C": ["2s", "2p"]}, method="e3tb", device=device)
    node = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    edge = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    data = {
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(2, device=device, dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]],
            device=device, dtype=torch.long,
        ),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor(
            [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]], device=device, dtype=torch.long
        ),
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    ref = {_keys.NODE_FEATURES_KEY: node.clone(), _keys.EDGE_FEATURES_KEY: edge.clone()}
    return idp, data, ref


def _onsite_diag_indices(idp, symbol: str, orbital: str, device: torch.device):
    full = idp.basis_to_full_basis[symbol][orbital]
    block = idp.orbpair_maps[f"{full}-{full}"]
    degree = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}[full[-1]]
    width = 2 * degree + 1
    diag = torch.arange(width, device=device, dtype=torch.long)
    return int(block.start) + diag * width + diag


def test_basis_onsite_prior_initializes_without_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "residual", "prior": "basis_onsite",
            "strict_h0": False, "warn_missing_h0": False, "detach_interpolated_h0": False,
        },
        idp=idp, device=device, dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected)
    torch.testing.assert_close(ctx.node_prior, expected)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], torch.zeros_like(ref[_keys.EDGE_FEATURES_KEY]))


def test_basis_onsite_prior_is_residualized_against_existing_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    node_base = torch.full_like(ref[_keys.NODE_FEATURES_KEY], 0.25)
    edge_base = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.5)
    data[_keys.NODE_H0_KEY] = node_base
    data[_keys.EDGE_H0_KEY] = edge_base
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "residual", "prior": "basis_onsite", "detach_interpolated_h0": False},
        idp=idp, device=device, dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected_abs = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_abs[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_abs[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_abs[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_abs)
    torch.testing.assert_close(ctx.node_prior, expected_abs - node_base)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], torch.zeros_like(edge_base))
    torch.testing.assert_close(ctx.edge_prior, -edge_base)


def test_overlap_huckel_prior_uses_overlap_and_basis_without_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    edge_overlap = torch.linspace(
        -0.3, 0.4, steps=ref[_keys.EDGE_FEATURES_KEY].numel(), device=device, dtype=dtype
    ).reshape_as(ref[_keys.EDGE_FEATURES_KEY])
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "residual", "prior": "overlap_huckel",
            "strict_h0": False, "warn_missing_h0": False, "detach_interpolated_h0": False,
            "huckel_k": 2.0,
        },
        idp=idp, device=device, dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    out_again, _ref_again, ctx_again = flow.prepare_batch(
        data, ref, t=torch.zeros(1, device=device, dtype=dtype)
    )

    expected_node = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_node[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]
    h_mean = torch.tensor(onsite_energy_database["H"]["1s"], device=device, dtype=dtype)
    c_mean = torch.tensor(
        (onsite_energy_database["C"]["2s"] + 3.0 * onsite_energy_database["C"]["2p"]) / 4.0,
        device=device, dtype=dtype,
    )
    edge_energy = 0.5 * (h_mean + c_mean)
    expected_edge = 2.0 * edge_energy * edge_overlap
    expected_edge = expected_edge * flow._prior_mask(data, expected_edge, "edge").to(dtype=dtype)

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_node)
    torch.testing.assert_close(ctx.node_prior, expected_node)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], expected_edge)
    torch.testing.assert_close(ctx.edge_prior, expected_edge)
    torch.testing.assert_close(out_again[_keys.NODE_H0_KEY], out[_keys.NODE_H0_KEY])
    torch.testing.assert_close(ctx_again.node_prior, ctx.node_prior)
    torch.testing.assert_close(out_again[_keys.EDGE_H0_KEY], out[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(ctx_again.edge_prior, ctx.edge_prior)


def test_overlap_huckel_prior_is_residualized_against_existing_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    edge_overlap = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    node_base = torch.full_like(ref[_keys.NODE_FEATURES_KEY], 0.25)
    edge_base = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], -0.5)
    data[_keys.NODE_H0_KEY] = node_base
    data[_keys.EDGE_H0_KEY] = edge_base
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "residual", "prior": "overlap_huckel",
            "detach_interpolated_h0": False, "huckel_k": 1.5,
        },
        idp=idp, device=device, dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected_node = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_node[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]
    h_mean = torch.tensor(onsite_energy_database["H"]["1s"], device=device, dtype=dtype)
    c_mean = torch.tensor(
        (onsite_energy_database["C"]["2s"] + 3.0 * onsite_energy_database["C"]["2p"]) / 4.0,
        device=device, dtype=dtype,
    )
    expected_edge = 1.5 * 0.5 * (h_mean + c_mean) * edge_overlap
    expected_edge = expected_edge * flow._prior_mask(data, expected_edge, "edge").to(dtype=dtype)

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_node)
    torch.testing.assert_close(ctx.node_prior, expected_node - node_base)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], expected_edge)
    torch.testing.assert_close(ctx.edge_prior, expected_edge - edge_base)


def test_overlap_huckel_prior_requires_edge_overlap_by_default():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "residual", "prior": "overlap_huckel",
            "strict_h0": False, "warn_missing_h0": False, "detach_interpolated_h0": False,
        },
        idp=idp, device=device, dtype=dtype,
    )

    with pytest.raises(KeyError, match="edge_overlap"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_overlap_huckel_prior_rejects_edge_index_overlap_row_mismatch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    # edge_index carries 3 columns while the edge overlap/feature tensor has 2 rows.
    # Under the default strict basis mode this must fail loud instead of being
    # silently padded/truncated/clamped into a plausible-but-wrong prior.
    data[_keys.EDGE_INDEX_KEY] = torch.tensor(
        [[0, 1, 0], [1, 0, 1]], device=device, dtype=torch.long
    )
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    flow = HamiltonianCFM(
        {
            "enabled": True, "mode": "residual", "prior": "overlap_huckel",
            "strict_h0": False, "warn_missing_h0": False, "detach_interpolated_h0": False,
        },
        idp=idp, device=device, dtype=dtype,
    )

    with pytest.raises(ValueError, match="edge_index"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_prior_physical_huckel_edge_energy_strict_rejects_column_mismatch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    type_mean = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
    atom_types = torch.tensor([0, 1], device=device, dtype=torch.long)
    # 3 edge_index columns but only 2 edge rows: strict mode fails loud, lax
    # mode tolerates it (pad/truncate) -- the same fact test_overlap_huckel_prior_
    # rejects_edge_index_overlap_row_mismatch checks through the flow wrapper.
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]], device=device, dtype=torch.long)

    with pytest.raises(ValueError, match="edge_index"):
        prior_physical.huckel_edge_energy(
            type_mean, edge_index, atom_types, 2, fallback=0.0, strict=True
        )

    out = prior_physical.huckel_edge_energy(
        type_mean, edge_index, atom_types, 2, fallback=0.0, strict=False
    )
    assert out.shape == (2,)


def test_dftb_prior_uses_external_absolute_hamiltonian_keys():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    dftb_node = data[_keys.NODE_H0_KEY] + 3.0
    dftb_edge = data[_keys.EDGE_H0_KEY] - 2.0
    data["dftb_node_h0"] = dftb_node
    data["dftb_edge_h0"] = dftb_edge
    flow = build_cfm("dftb", device=device, dtype=dtype)

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], dftb_node)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], dftb_edge)
    torch.testing.assert_close(ctx.node_prior, dftb_node - data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(ctx.edge_prior, dftb_edge - data[_keys.EDGE_H0_KEY])


def test_dftb_prior_fails_closed_without_matching_keys_or_skdata():
    # Behavior change: prior=dftb/xtb/sk/nnsk with no matching keys and no skdata
    # must fail closed (KeyError), no longer silently falling back to basis_onsite.
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = make_batch(device=device, dtype=dtype)
    # A dropped legacy permutation is present but is no longer consulted.
    data["node_h0_dftb"] = data[_keys.NODE_H0_KEY] + 1.0
    flow = build_cfm("dftb", device=device, dtype=dtype)

    with pytest.raises(KeyError, match="Tried keys"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    # external_prior_strict=False still degrades the whole external family to a
    # zero absolute prior (H_t at t=0 equals the zero Hamiltonian).
    flow_lax = build_cfm("dftb", device=device, dtype=dtype, external_prior_strict=False)
    out, _ref_out, _ctx = flow_lax.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )
    torch.testing.assert_close(out[_keys.NODE_H0_KEY], torch.zeros_like(out[_keys.NODE_H0_KEY]))
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], torch.zeros_like(out[_keys.EDGE_H0_KEY]))

    # physical_prior_fallback governs the generic "physical" alias only: a named
    # alias like "dftb" still fails closed even when it is set to "zero"
    # (formerly test_named_external_prior_alias_ignores_physical_zero_fallback).
    flow_named_alias = build_cfm(
        "dftb", device=device, dtype=dtype, physical_prior_fallback="zero"
    )
    with pytest.raises(KeyError, match="flow_options.prior='dftb'"):
        flow_named_alias.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# haar_dm precomputed-candidate prior
# ---------------------------------------------------------------------------


def test_haar_dm_prior_selects_precomputed_candidate():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm", "haar_candidate_index": 1},
        idp=FakeIDP(device=device), device=device, dtype=dtype,
    )
    node_target = ref[_keys.NODE_FEATURES_KEY]
    edge_target = ref[_keys.EDGE_FEATURES_KEY]
    data[_keys.HAAR_NODE_FEATURES_KEY] = torch.stack(
        [torch.full_like(node_target, 3.0), torch.full_like(node_target, 7.0)], dim=1
    )
    data[_keys.HAAR_EDGE_FEATURES_KEY] = torch.stack(
        [torch.full_like(edge_target, 30.0), torch.full_like(edge_target, 70.0)], dim=1
    )

    _out, _ref_out, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    torch.testing.assert_close(ctx.node_prior, torch.full_like(node_target, 7.0))
    torch.testing.assert_close(ctx.edge_prior, torch.full_like(edge_target, 70.0))
    assert flow.last_haar_candidate_index == 1


def test_haar_dm_prior_requires_precomputed_fields():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm"},
        idp=FakeIDP(device=device), device=device, dtype=dtype,
    )

    with pytest.raises(KeyError, match="haar_dm"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_haar_dm_prior_uses_same_candidate_for_node_and_edge():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm", "detach_interpolated_h0": False},
        idp=FakeIDP(device=device), device=device, dtype=dtype,
    )

    node_target = ref[_keys.NODE_FEATURES_KEY]
    edge_target = ref[_keys.EDGE_FEATURES_KEY]
    # K=2 candidate axis on both node and edge. Candidate 0 vs 1 are far apart,
    # and index i of the node tensor is coherent with index i of the edge tensor
    # (both encode the same Haar density matrix in production), so the node and
    # edge priors must always select the same candidate.
    node_candidates = torch.stack(
        [torch.full_like(node_target, 3.0), torch.full_like(node_target, 7.0)], dim=1
    )
    edge_candidates = torch.stack(
        [torch.full_like(edge_target, 30.0), torch.full_like(edge_target, 70.0)], dim=1
    )
    data[_keys.HAAR_NODE_FEATURES_KEY] = node_candidates
    data[_keys.HAAR_EDGE_FEATURES_KEY] = edge_candidates

    t = torch.zeros(2, device=device, dtype=dtype)
    seen = set()
    torch.manual_seed(0)
    for _ in range(40):
        _out, _ref_out, ctx = flow.prepare_batch(data, ref, t=t)
        node_c0 = bool(torch.all(ctx.node_prior == 3.0))
        node_c1 = bool(torch.all(ctx.node_prior == 7.0))
        edge_c0 = bool(torch.all(ctx.edge_prior == 30.0))
        edge_c1 = bool(torch.all(ctx.edge_prior == 70.0))
        # Each prior is exactly one candidate, fully (no cross-candidate mixing).
        assert node_c0 != node_c1
        assert edge_c0 != edge_c1
        node_idx = 0 if node_c0 else 1
        edge_idx = 0 if edge_c0 else 1
        assert node_idx == edge_idx
        seen.add(node_idx)
    # The shared draw is genuinely random, not pinned to a single candidate.
    assert seen == {0, 1}


def test_haar_dm_prior_rejects_mismatched_node_edge_candidate_counts():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm"},
        idp=FakeIDP(device=device), device=device, dtype=dtype,
    )
    node_target = ref[_keys.NODE_FEATURES_KEY]
    edge_target = ref[_keys.EDGE_FEATURES_KEY]
    # Node exposes K=2 candidates, edge exposes K=3: the coherent-pairing
    # assumption is violated and prepare_batch must refuse to draw an index.
    data[_keys.HAAR_NODE_FEATURES_KEY] = torch.stack(
        [torch.full_like(node_target, float(i)) for i in range(2)], dim=1
    )
    data[_keys.HAAR_EDGE_FEATURES_KEY] = torch.stack(
        [torch.full_like(edge_target, float(i)) for i in range(3)], dim=1
    )

    with pytest.raises(ValueError, match="candidate"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# flow_options / common_options argcheck acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {
            "enabled": True, "mode": "residual", "prior": "te", "te_prior_sigma": 0.5,
            "te_prior_mode": "typewise", "te_prior_per_graph": False,
        },
        {
            "enabled": True, "mode": "residual", "prior": "dftbsk",
            "prior_node_key": "dftb_node_h0", "prior_edge_key": "dftb_edge_h0",
            "prior_key_prefixes": ["dftb", "xtb"], "allow_complex_prior_real_projection": True,
            "prior_skdata": "/tmp/skfiles", "dftb_prior_overlap": True,
            "physical_prior_fallback": "basis_onsite", "basis_onsite_scale": 0.5,
            "physical_prior_jitter_sigma": 0.01, "physical_prior_jitter_reference_scale": True,
            "physical_prior_jitter_edge_decay": 2.0,
        },
        {
            "enabled": True, "mode": "residual", "prior": "overlap_huckel", "huckel_k": 1.75,
            "huckel_node_overlap_key": "node_overlap", "huckel_edge_overlap_key": "edge_overlap",
            "huckel_strict_overlap": True, "huckel_edge_energy_fallback": -2.0,
            "huckel_edge_length_decay": 3.0,
        },
        {
            "enabled": True, "prior": "haar_dm", "haar_node_key": "haar_node_features",
            "haar_edge_key": "haar_edge_features", "haar_candidate_index": 0,
            "haar_dm_strict": True,
        },
    ],
    ids=["te_prior_keys", "physical_prior_keys", "overlap_huckel_keys", "haar_dm_keys"],
)
def test_flow_options_argcheck_accepts_prior_config_keys(config):
    schema = flow_options()
    value = schema.normalize_value(config)
    schema.check_value(value, strict=True)


def test_common_options_argcheck_accepts_nextham_uureal_mask():
    schema = common_options()
    value = schema.normalize_value(
        {"basis": {"Si": ["3s", "3p"]}, "has_soc": True, "nextham_uureal_mask": True}
    )
    schema.check_value(value, strict=True)


# ---------------------------------------------------------------------------
# P0 regression: flow prior must not be silently deactivated by a key mismatch
# between train_options.flow_options.{node,edge}_h0_key (what the flow writes)
# and model_options.embedding.h0_{node,edge}_key (what the H0-init reads).
# ---------------------------------------------------------------------------


def _h0_init_consumer_stub(h0_node_key, h0_edge_key):
    """Minimal stand-in for the H0InitLayer consumer: the build-time guard keys
    purely on a submodule exposing string h0_node_key/h0_edge_key attributes."""
    layer = torch.nn.Module()
    layer.h0_node_key = h0_node_key
    layer.h0_edge_key = h0_edge_key
    model = torch.nn.Module()
    model.init_layer = layer  # registered as a child -> visible via model.modules()
    return model


def _physical_h0_flow(enabled=True):
    return build_hamiltonian_flow(
        {
            "enabled": enabled, "mode": "residual", "prior": "zero",
            "node_h0_key": "node_physical_h0", "edge_h0_key": "edge_physical_h0",
        }
    )


@pytest.mark.parametrize(
    "flow,model,expect_error",
    [
        (
            _physical_h0_flow(enabled=True), _h0_init_consumer_stub("node_h0", "edge_h0"),
            "silently deactivated",
        ),
        (
            _physical_h0_flow(enabled=True),
            _h0_init_consumer_stub("node_physical_h0", "edge_physical_h0"),
            None,
        ),
        (_physical_h0_flow(enabled=False), _h0_init_consumer_stub("node_h0", "edge_h0"), None),
        (_physical_h0_flow(enabled=True), torch.nn.Linear(2, 2), None),
        (
            build_hamiltonian_flow({"enabled": True, "mode": "residual", "prior": "zero"}),
            _h0_init_consumer_stub("node_h0", "edge_h0"),
            None,
        ),
    ],
    ids=[
        "raises_when_embedding_keys_not_repointed",
        "passes_when_embedding_keys_are_aligned",
        "is_noop_when_flow_disabled",
        "is_noop_without_h0_init_consumer",
        "accepts_default_stored_h0_config",
    ],
)
def test_flow_h0_key_guard(flow, model, expect_error):
    if expect_error:
        with pytest.raises(ValueError, match=expect_error) as excinfo:
            assert_flow_h0_keys_reach_model(flow, model)
        message = str(excinfo.value)
        assert "node_physical_h0" in message and "edge_physical_h0" in message
        assert "h0_node_key" in message and "h0_edge_key" in message
    else:
        assert assert_flow_h0_keys_reach_model(flow, model) is None


# ---------------------------------------------------------------------------
# Seeded RNG scope (shared by tied priors and CFM's own prior draws)
# ---------------------------------------------------------------------------


def test_seeded_scope_reproduces_manual_seed_and_restores_the_cpu_generator():
    state = {"node_features": torch.ones(2, 3), "note": "not a tensor"}
    torch.manual_seed(712)
    expected = torch.randn(8)
    torch.manual_seed(5)
    before = torch.random.get_rng_state()
    with _seeded_rng_scope(state, 712):
        drawn = torch.randn(8)
    assert torch.equal(drawn, expected)
    assert torch.equal(torch.random.get_rng_state(), before)


@requires_multi_gpu
def test_seeded_scope_leaves_cuda_devices_outside_the_state_untouched():
    torch.cuda.manual_seed_all(5)
    torch.randn(4, device="cuda:1")  # advance the other device's generator
    other_before = torch.cuda.get_rng_state(1)
    own_before = torch.cuda.get_rng_state(0)
    state = {"node_h0": torch.zeros(2, device="cuda:0")}

    with _seeded_rng_scope(state, 712):
        drawn = torch.randn(8, device="cuda:0")

    assert torch.equal(torch.cuda.get_rng_state(1), other_before)
    assert torch.equal(torch.cuda.get_rng_state(0), own_before)
    torch.cuda.manual_seed(712)
    assert torch.equal(drawn, torch.randn(8, device="cuda:0"))
