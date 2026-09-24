"""H-B0 head: active-edge contract, endpoint conditioning, head-input RMS telemetry,
Hermitian averaging.

All ``LemMoEV3H0``/``LemPair`` cases below run in fp64 with
``torch.use_deterministic_algorithms(True)`` scoped by ``_deterministic()`` — the
conftest.py module-scoped guard fails the whole module if that flag leaks past a test.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import strict_reverse_edge_index
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.late_block_expansion_cg import LateBlockExpansionCGHead
from dptb.nn.embedding.lem_moe_v3 import UpdateNode
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair
from dptb.tests.pair_helpers import ao_wigner, clone_data, fp64_default, model_options, molecule_data, rotate_data


@contextmanager
def _deterministic():
    """Scope ``torch.use_deterministic_algorithms(True)``; never leak it past a test."""
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def _h0_model(*, seed: int = 20260724, **overrides) -> LemMoEV3H0:
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(overrides)
    torch.manual_seed(seed)
    return LemMoEV3H0(**options).eval()


# ---------------------------------------------------------------------------
# Active-edge contract: legacy subset scatter, full-coverage guard, UpdateNode rows
# ---------------------------------------------------------------------------

class _SphericalHarmonics(torch.nn.Module):
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors.new_zeros((vectors.shape[0], 1))


class _NodeOneHot(torch.nn.Module):
    def forward(self, data):
        count = data[_keys.ATOM_TYPE_KEY].numel()
        data[_keys.NODE_ATTRS_KEY] = torch.ones((count, 1), device=data[_keys.POSITIONS_KEY].device)
        return data


class _EdgeOneHot(torch.nn.Module):
    def forward(self, data):
        count = data[_keys.EDGE_INDEX_KEY].shape[1]
        return torch.ones((count, 1), device=data[_keys.POSITIONS_KEY].device)


class _Router(torch.nn.Module):
    def forward(self, global_features: torch.Tensor):
        count = global_features.shape[0]
        coefficients = global_features.new_ones((count, 1))
        return coefficients, global_features.new_ones(()), global_features.new_zeros(())

    def last_topk(self):
        return torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1))


class _ActiveSubsetInit(torch.nn.Module):
    """Small deterministic stand-in for H0InitLayer's active-row contract."""

    def forward(self, data, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot,
                active_edges=None, cutoff_coeffs=None):
        if active_edges is None:
            active_edges = torch.tensor([0, 1], device=edge_length.device)
        else:
            active_edges = active_edges.to(device=edge_length.device, dtype=torch.long)
        if cutoff_coeffs is None:
            cutoff_coeffs = edge_length.new_ones(edge_length.shape[0])
        latents = edge_length.new_zeros((edge_length.shape[0], 1))
        node_features = edge_length.new_zeros((atom_type.numel(), 1))
        edge_features = (
            torch.arange(active_edges.numel(), device=edge_length.device, dtype=edge_length.dtype).add_(1.0).unsqueeze(-1)
        )
        return latents, node_features, edge_features, cutoff_coeffs, active_edges


def _fake_block_heads(self, node_features, edge_features, atom_type, edge_index, active_edges):
    node_blocks = torch.arange(node_features.shape[0], device=node_features.device,
                                dtype=node_features.dtype).add_(11.0).reshape(-1, 1, 1)
    edge_blocks = edge_features.add(20.0).reshape(-1, 1, 1)
    return node_blocks, edge_blocks


def _minimal_hb0_h0_model(idp: OrbitalMapper) -> LemMoEV3H0:
    """Exercise LemMoEV3H0.forward without constructing the expensive MoE stack."""
    model = LemMoEV3H0.__new__(LemMoEV3H0)
    torch.nn.Module.__init__(model)
    model.use_h0_init = True
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    model.idp = idp
    model.sh = _SphericalHarmonics()
    model.onehot = _NodeOneHot()
    model.edge_one_hot = _EdgeOneHot()
    model.router = _Router()
    model.init_layer = _ActiveSubsetInit()
    model.flow_time_conditioner = None
    model.layers = torch.nn.ModuleList()
    model.use_block_native_output = True
    model.require_full_block_edge_coverage = False
    model.out_edge = SimpleNamespace(max_norb=1)
    model._apply_block_native_output_heads = MethodType(_fake_block_heads, model)
    return model


def _minimal_hb0_case():
    idp = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    return idp, _minimal_hb0_h0_model(idp)


def _active_edge_data(idp, active_edges, cutoff_coeffs):
    return {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.long),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float32).unsqueeze(0) * 5.0,
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.zeros((2, 1), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.zeros(4, dtype=torch.long),
        _keys.NODE_H0_KEY: torch.zeros((2, idp.reduced_matrix_element)),
        _keys.EDGE_H0_KEY: torch.zeros((4, idp.reduced_matrix_element)),
        _keys.LEM_ACTIVE_EDGES_KEY: active_edges,
        _keys.LEM_CUTOFF_COEFFS_KEY: cutoff_coeffs,
    }


def _two_graph_data(idp, *, split_sizes=None):
    data = _active_edge_data(idp, torch.arange(4), torch.ones(4))
    data.update({
        _keys.POSITIONS_KEY: torch.zeros((4, 3)),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        _keys.CELL_KEY: torch.eye(3).repeat(2, 1, 1),
        _keys.PBC_KEY: torch.zeros((2, 3), dtype=torch.bool),
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros((4, 3), dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0, 0, 1, 1]),
        _keys.ATOM_TYPE_KEY: torch.zeros((4, 1), dtype=torch.long),
        _keys.NODE_H0_KEY: torch.zeros((4, idp.reduced_matrix_element)),
    })
    if split_sizes is not None:
        data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = torch.as_tensor(split_sizes)
    return data


def test_legacy_hb0_subset_scatter_is_unchanged_when_full_coverage_is_not_required():
    idp, model = _minimal_hb0_case()

    active_edges = torch.tensor([0, 1], dtype=torch.long)
    data = _active_edge_data(idp, active_edges, torch.ones(4))
    data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = torch.tensor([2])

    # Two distinct periodic images, each a complete reverse pair: valid under strict keying.
    assert strict_reverse_edge_index(data, idp=idp).tolist() == [1, 0, 3, 2]

    output = model(data)

    # Legacy ao_block tensors retain the exact zero-canvas/index_copy result.
    torch.testing.assert_close(output[_keys.NODE_HAMILTONIAN_KEY], torch.tensor([[[11.0]], [[12.0]]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[_keys.EDGE_HAMILTONIAN_KEY],
                                torch.tensor([[[21.0]], [[22.0]], [[0.0]], [[0.0]]]), rtol=0.0, atol=0.0)

    # Reusable input-side acceleration metadata still follows its historical cleanup contract.
    assert _keys.LEM_ACTIVE_EDGES_KEY not in output
    assert _keys.LEM_CUTOFF_COEFFS_KEY not in output
    assert _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY not in output


def test_block_ode_coverage_guard_accepts_only_the_actual_ordered_full_head_rows():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True

    assert model.supports_full_block_edge_coverage is True
    output = model(_active_edge_data(idp, torch.arange(4), torch.ones(4)))
    torch.testing.assert_close(output[_keys.EDGE_HAMILTONIAN_KEY],
                                torch.tensor([[[21.0]], [[22.0]], [[23.0]], [[24.0]]]), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("active_edges", "cutoff_coeffs", "match"),
    (
        (torch.tensor([0, 1]), torch.ones(4), "ordered full H-B0"),
        (torch.tensor([1, 0, 2, 3]), torch.ones(4), "ordered full H-B0"),
        (torch.tensor([0.9, 1.2, 2.0, 3.0]), torch.ones(4), "integral active-edge"),
        (torch.arange(4), torch.tensor([1.0, 0.0, 1.0, 1.0]), "strictly positive"),
        (torch.arange(4), torch.tensor([1.0, float("nan"), 1.0, 1.0]), "strictly positive"),
        (torch.arange(4), torch.ones(3), "one finite"),
    ),
)
def test_block_ode_coverage_guard_rejects_uncomputed_or_ambiguous_head_rows(active_edges, cutoff_coeffs, match):
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True
    with pytest.raises(ValueError, match=match):
        model(_active_edge_data(idp, active_edges, cutoff_coeffs))


def test_block_ode_coverage_guard_rejects_stale_graph_split_sizes():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True
    data = _two_graph_data(idp, split_sizes=[1, 3])
    with pytest.raises(ValueError, match="split sizes must exactly match"):
        model(data)


def test_block_ode_rechecks_cross_graph_edges_after_a_valid_forward():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True
    model(_two_graph_data(idp, split_sizes=[2, 2]))

    malformed = _two_graph_data(idp, split_sizes=[2, 2])
    malformed[_keys.EDGE_INDEX_KEY][1, 1] = 2
    with pytest.raises(ValueError, match="stay within one batch graph"):
        model(malformed)


def test_update_node_keeps_rows_for_nodes_without_active_incident_edges():
    class _TensorProduct(torch.nn.Module):
        def forward(self, x, r, mole_globals, latents=None, wigner_D_all=None):
            return torch.ones((x.shape[0], 1), dtype=x.dtype), wigner_D_all

    class _Update:
        irreps_in = o3.Irreps("1x0e")
        irreps_out = o3.Irreps("1x0e")
        edge_irreps_in = o3.Irreps("1x0e")
        tp = _TensorProduct()
        activation = torch.nn.Identity()
        lin_post = torch.nn.Identity()
        focus_gate = torch.nn.Identity()
        post_activation_expert_mixer = None
        node_norm = None
        edge_norm = None
        node_attention = None
        edge_aggregation_gate = None
        env_sum_normalizations = torch.tensor(1.0)
        res_update = False
        use_layer_onehot_tp = False
        edge_message_env_weight = False

    output = UpdateNode.forward(
        _Update(), latents=torch.zeros((2, 1)), node_features=torch.zeros((3, 1)), edge_features=torch.zeros((2, 1)),
        atom_type=torch.zeros((3, 1), dtype=torch.long), node_onehot=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_vector=torch.zeros((2, 3)),
        cutoff_coeffs=torch.ones(2), active_edges=torch.tensor([0, 1], dtype=torch.long),
        wigner_D_all=None, mole_globals=None,
    )

    assert output.shape == (3, 1)
    torch.testing.assert_close(output[:2], torch.ones((2, 1)))
    torch.testing.assert_close(output[2], torch.zeros(1))


# ---------------------------------------------------------------------------
# Condition source / head-input RMS / Hermitian averaging: default equals explicit "off"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("override_key", "override_value", "output_keys"),
    (
        ("condition_source", "edge_0e", (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY, _keys.EDGE_OVERLAP_KEY)),
        ("log_head_input_rms", False, (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY, _keys.EDGE_OVERLAP_KEY)),
        ("hb0_hermitian_average", False, (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY)),
    ),
    ids=["condition_source_edge_0e", "log_head_input_rms_false", "hb0_hermitian_average_false"],
)
def test_default_equals_explicit_off_is_bit_exact(override_key, override_value, output_keys):
    with fp64_default(), _deterministic():
        implicit = _h0_model()
        implicit_rng = torch.random.get_rng_state().clone()
        explicit = _h0_model(**{override_key: override_value})
        explicit_rng = torch.random.get_rng_state().clone()

        assert torch.equal(implicit_rng, explicit_rng)
        assert implicit.state_dict().keys() == explicit.state_dict().keys()
        assert all(torch.equal(implicit.state_dict()[key], explicit.state_dict()[key]) for key in implicit.state_dict())

        implicit_out = implicit(molecule_data(implicit))
        explicit_out = explicit(molecule_data(explicit))
        if override_key == "log_head_input_rms":
            assert "head_input_rms" not in implicit_out
            assert "head_input_rms" not in explicit_out
        for key in output_keys:
            assert torch.equal(implicit_out[key], explicit_out[key])


def test_endpoint_condition_changes_output_and_is_translation_invariant():
    with fp64_default(), _deterministic():
        edge_only = _h0_model()
        endpoint = _h0_model(condition_source="endpoints")
        data = molecule_data(endpoint)

        edge_only_out = edge_only(molecule_data(edge_only))
        reference = endpoint(clone_data(data))
        assert not torch.equal(edge_only_out[_keys.EDGE_HAMILTONIAN_KEY], reference[_keys.EDGE_HAMILTONIAN_KEY])

        translated = clone_data(data)
        translated[_keys.POSITIONS_KEY] += torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64)
        drift = float((endpoint(translated)[_keys.EDGE_HAMILTONIAN_KEY] - reference[_keys.EDGE_HAMILTONIAN_KEY]).abs().max())
        assert drift <= 1.0e-12


def _edge_hermiticity_drift(data, edge_blocks, model) -> float:
    reverse = strict_reverse_edge_index(data, device=edge_blocks.device, idp=model.idp)
    return float((edge_blocks - edge_blocks.index_select(0, reverse).transpose(-1, -2)).abs().max())


@pytest.mark.parametrize(
    ("override_key", "override_value", "check_hermiticity"),
    (
        ("condition_source", "endpoints", False),
        ("hb0_hermitian_average", True, True),
    ),
    ids=["endpoint_condition", "hb0_hermitian_average"],
)
def test_condition_and_hermitian_average_are_rotation_equivariant_fp64(override_key, override_value, check_hermiticity):
    with fp64_default(), _deterministic():
        model = _h0_model(**{override_key: override_value})
        data = molecule_data(model)
        reference = model(clone_data(data))

        torch.manual_seed(31)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = model(rotate_data(data, rotation))

        d_ao = ao_wigner(model, rotation)
        expected_edge = torch.einsum("ij,njk,lk->nil", d_ao, reference[_keys.EDGE_HAMILTONIAN_KEY], d_ao)
        drift = float((rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected_edge).abs().max())
        assert drift <= 1.0e-9

        if check_hermiticity:
            assert _edge_hermiticity_drift(data, rotated[_keys.EDGE_HAMILTONIAN_KEY], model) == 0.0


def test_endpoint_condition_gradient_reaches_conditioner_and_both_node_0e_inputs():
    torch.manual_seed(43)
    head = LateBlockExpansionCGHead("2x0e+1x1o", ["1s", "1p"], symmetrize=False, rank=3, init=0.2,
                                     condition_source="endpoints", node_irreps="2x0e+1x1o", dtype=torch.float64)
    edge_features = torch.randn(1, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    node_features = torch.randn(2, head.node_irreps.dim, dtype=torch.float64, requires_grad=True)
    node_0e = node_features.index_select(-1, head._node_scalar_indices)
    extra_condition = torch.cat([node_0e[0:1], node_0e[1:2]], dim=-1)
    loss = head(edge_features, extra_condition=extra_condition).square().sum()
    loss.backward()

    weight_grad = head.condition_down.weight.grad
    src_grad = node_features.grad[0].index_select(0, head._node_scalar_indices)
    dst_grad = node_features.grad[1].index_select(0, head._node_scalar_indices)
    assert weight_grad is not None
    assert float(weight_grad.norm()) > 0.0
    assert float(src_grad.norm()) > 0.0
    assert float(dst_grad.norm()) > 0.0


def test_invalid_condition_source_fails_before_model_construction():
    with pytest.raises(ValueError, match="condition_source"):
        _h0_model(condition_source="edge_and_endpoints")


def _manual_slice_rms(features, irreps):
    values = []
    for term_slice in irreps.slices():
        block = features[..., term_slice]
        values.append((block.square().sum() / block.numel()).sqrt())
    return torch.stack(values)


def test_head_input_rms_enabled_matches_manual_irreps_slice_calculation():
    with fp64_default(), _deterministic():
        model = _h0_model(log_head_input_rms=True)
        captured = {}

        def capture_node(_module, inputs):
            captured["node"] = inputs[0].detach().clone()

        def capture_edge(_module, inputs):
            captured["edge"] = inputs[0].detach().clone()

        node_handle = model.out_node.register_forward_pre_hook(capture_node)
        edge_handle = model.out_edge.register_forward_pre_hook(capture_edge)
        try:
            output = model(molecule_data(model))
        finally:
            node_handle.remove()
            edge_handle.remove()

        telemetry = output["head_input_rms"]
        expected_node = _manual_slice_rms(captured["node"], model.out_node.irreps_in)
        expected_edge = _manual_slice_rms(captured["edge"], model.out_edge.irreps_in)
        torch.testing.assert_close(telemetry["node"], expected_node, rtol=0.0, atol=1.0e-15)
        torch.testing.assert_close(telemetry["edge"], expected_edge, rtol=0.0, atol=1.0e-15)
        assert torch.equal(telemetry["node_l"], torch.tensor([ir.l for _, ir in model.out_node.irreps_in], dtype=torch.long))
        assert torch.equal(telemetry["edge_l"], torch.tensor([ir.l for _, ir in model.out_edge.irreps_in], dtype=torch.long))
        assert not telemetry["node"].requires_grad
        assert not telemetry["edge"].requires_grad


@pytest.mark.parametrize(
    "overrides",
    (
        dict(condition_source="endpoints"),
        dict(condition_source="endpoints", log_head_input_rms=True),
        dict(pair_refine_enable=True, pair_refine_rank=4, pair_refine_identity_init=True, log_head_input_rms=True),
    ),
    ids=["endpoint_condition", "endpoint_condition_with_head_input_rms", "pair_refine_with_head_input_rms"],
)
def test_lem_pair_feature_combination_smoke(overrides):
    """LemPair (not just LemMoEV3H0) accepts these option combinations end to end."""
    with fp64_default(), _deterministic():
        options = model_options()
        options.update(overrides)
        torch.manual_seed(47)
        model = LemPair(**options).eval()
        output = model(molecule_data(model))
        assert torch.isfinite(output[_keys.NODE_HAMILTONIAN_KEY]).all()
        assert torch.isfinite(output[_keys.EDGE_HAMILTONIAN_KEY]).all()
        if overrides.get("log_head_input_rms"):
            assert torch.isfinite(output["head_input_rms"]["node"]).all()
            assert torch.isfinite(output["head_input_rms"]["edge"]).all()


def test_hb0_hermitian_average_enforces_exact_reverse_transpose_and_preserves_nodes():
    with fp64_default(), _deterministic():
        raw = _h0_model()
        averaged = _h0_model(hb0_hermitian_average=True)
        averaged.load_state_dict(raw.state_dict())

        raw_data = molecule_data(raw)
        averaged_data = molecule_data(averaged)
        raw_out = raw(clone_data(raw_data))
        averaged_out = averaged(clone_data(averaged_data))

        assert torch.equal(raw_out[_keys.NODE_HAMILTONIAN_KEY], averaged_out[_keys.NODE_HAMILTONIAN_KEY])
        assert _edge_hermiticity_drift(averaged_data, averaged_out[_keys.EDGE_HAMILTONIAN_KEY], averaged) == 0.0
        assert not torch.equal(raw_out[_keys.EDGE_HAMILTONIAN_KEY], averaged_out[_keys.EDGE_HAMILTONIAN_KEY])


def test_hb0_hermitian_average_rejects_missing_reverse_active_edge():
    edge_blocks = torch.randn(2, 2, 2, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    active_edges = torch.arange(2, dtype=torch.long)
    with pytest.raises(ValueError, match="missing"):
        LemMoEV3H0._hermitian_average_hb0_edge_blocks(edge_blocks, edge_index, active_edges)
