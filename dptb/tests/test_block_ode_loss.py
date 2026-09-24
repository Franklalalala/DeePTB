"""Block-ODE loss reductions: closed-form component values, the runtime/argcheck weight
contract, finite zero-gradients, empty/zero-edge batches, and the directed-full endpoint
population the block-ODE routes publish to the compatible criterion path."""
from __future__ import annotations

import copy
import math

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_components,
    block_mask_from_shapes,
    infer_block_shapes,
    l1_rmse_from_components,
    strict_reverse_edge_index,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.nnops.flow import HamiltonianCFM, HamiltonianPixelMeanFlow
from dptb.utils.argcheck import validate_flow_loss_contract
from dptb.tests.block_ode_fixtures import (
    FP64_ATOL,
    UUREAL_STATE_KEYS,
    _EndpointSpy,
    _b_flow,
    _b_record,
    _case,
    _flow,
    _fresh,
    _mapper,
    _ref_for,
    _uureal_flow,
    _uureal_mapper,
    _uureal_record,
)

# ---------------------------------------------------------------------------
# Closed-form component reducer values and the weight contract
# ---------------------------------------------------------------------------
def _component_loss(loss_type, reduction, node_values, edge_values, *, node_weight=1.0, edge_weight=1.0):
    flow = HamiltonianCFM(
        {"enabled": True, "loss_type": loss_type, "component_reduction": reduction,
         "node_weight": node_weight, "edge_weight": edge_weight}
    )
    components = []
    for values, weight in ((node_values, node_weight), (edge_values, edge_weight)):
        diff = torch.as_tensor(values, dtype=torch.float64)
        mask = torch.ones_like(diff, dtype=torch.bool)
        components.append((flow._metric_stats(diff, mask, loss_type), weight))
    return flow._reduce_component_stats(tuple(components))


def _build_cfm(overrides):
    return HamiltonianCFM({"enabled": True, **overrides})


def _validate_argcheck_flow(overrides):
    return validate_flow_loss_contract({"train_options": {"flow_options": overrides}})


def _single_atom_block_state(data, mapper, node_shapes, edge_shapes, node_value):
    return project_block_state(
        data, mapper,
        BlockTensorResult(
            node_blocks=torch.full((1, 1, 1), node_value, dtype=torch.float64),
            edge_blocks=torch.empty((0, 1, 1), dtype=torch.float64),
            node_shapes=node_shapes, edge_shapes=edge_shapes,
        ),
    )


@pytest.mark.parametrize(
    ("loss_type", "reduction", "node", "edge", "weights", "expected"),
    [
        ("mse", "global_elements", [0.0], [4.0], {}, 8.0),
        ("mse", "equal_components", [0.0], [4.0], {}, 16.0),
        ("l1_rmse", "global_elements", [0.0], [4.0], {}, 0.5 * (2.0 + math.sqrt(8.0))),
        ("l1_rmse", "equal_components", [0.0], [4.0], {}, 4.0),
        ("mse", "equal_components", [1.0], [3.0], {"node_weight": 2.0, "edge_weight": 0.5}, 6.5),
        ("l1_rmse", "equal_components", [1.0], [3.0], {"node_weight": 2.0, "edge_weight": 0.5}, 3.5),
    ],
)
def test_component_reducer_has_distinct_global_and_component_semantics(loss_type, reduction, node, edge, weights, expected):
    result = _component_loss(loss_type, reduction, node, edge, **weights)
    assert result.item() == pytest.approx(expected, abs=1.0e-12)


@pytest.mark.parametrize("check", [pytest.param(_build_cfm, id="runtime"), pytest.param(_validate_argcheck_flow, id="argcheck")])
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"node_weight": float("nan")}, "finite and non-negative"),
        ({"edge_weight": float("inf")}, "finite and non-negative"),
        ({"node_weight": -1.0}, "finite and non-negative"),
        ({"node_weight": 0.0, "edge_weight": 0.0}, "may not both be zero"),
        ({"node_weight": 2.0}, "global_elements.*requires"),
        ({"edge_weight": 0.5}, "global_elements.*requires"),
    ],
)
def test_cfm_component_weight_contract_fails_closed(check, overrides, match):
    with pytest.raises(ValueError, match=match):
        check(overrides)


def test_pixel_meanflow_explicitly_owns_nonunit_component_weights():
    options = {
        "enabled": True, "objective": "pixel_meanflow", "component_reduction": "global_elements",
        "node_weight": 2.0, "edge_weight": 0.5,
    }
    flow = HamiltonianPixelMeanFlow(options)
    assert (flow.node_weight, flow.edge_weight) == (2.0, 0.5)
    assert _validate_argcheck_flow(options) is None
    with pytest.raises(ValueError, match="finite and non-negative"):
        HamiltonianPixelMeanFlow({**options, "node_weight": math.inf})


def test_l1_rmse_exact_zero_has_zero_value_and_finite_zero_gradient():
    diff = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    stats = HamiltonianCFM._metric_stats(diff, torch.ones_like(diff, dtype=torch.bool), "l1_rmse")
    global_metric = HamiltonianCFM._global_metric(stats[1], stats[2], stats[3], "l1_rmse")
    assert stats[0].item() == 0.0
    assert global_metric.item() == 0.0
    (stats[0] + global_metric).backward()
    assert torch.isfinite(diff.grad).all()
    assert torch.count_nonzero(diff.grad).item() == 0


@pytest.mark.parametrize("loss_type", ["mse", "l1_rmse"])
def test_empty_metric_component_has_zero_count_and_zero_value(loss_type):
    diff = torch.empty((0, 2, 2), dtype=torch.float64)
    stats = HamiltonianCFM._metric_stats(diff, torch.empty_like(diff, dtype=torch.bool), loss_type)
    assert stats[0].item() == 0.0 and stats[1].item() == 0.0 and stats[2].item() == 0.0 and stats[3].item() == 0.0


def test_zero_edge_block_ode_prepare_and_loss_are_finite():
    mapper = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    data = {
        _keys.POSITIONS_KEY: torch.zeros((1, 3), dtype=torch.float64),
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1], dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.tensor([0], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0], dtype=torch.long),
        _keys.PBC_KEY: torch.tensor([False, False, False]),
        _keys.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.empty((0, 3), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.empty((0,), dtype=torch.long),
    }
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    h0 = _single_atom_block_state(data, mapper, node_shapes, edge_shapes, 1.0)
    endpoint = _single_atom_block_state(data, mapper, node_shapes, edge_shapes, 2.0)
    codec = BlockStateCodec(mapper, dtype=torch.float64)
    node_h0, edge_h0 = codec.blocks_to_rme(data, h0)
    node_target, edge_target = codec.blocks_to_rme(data, endpoint)
    data.update(
        {
            _keys.NODE_H0_KEY: node_h0, _keys.EDGE_H0_KEY: edge_h0,
            _keys.NODE_H0_BLOCKS_KEY: h0.node_blocks, _keys.EDGE_H0_BLOCKS_KEY: h0.edge_blocks,
            _keys.NODE_H0_BLOCK_SHAPE_KEY: h0.node_shapes, _keys.EDGE_H0_BLOCK_SHAPE_KEY: h0.edge_shapes,
            _keys.NODE_FEATURES_KEY: node_target, _keys.EDGE_FEATURES_KEY: edge_target,
        }
    )
    ref = dict(data)
    ref.update(
        {
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: endpoint.node_blocks,
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: endpoint.edge_blocks,
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: endpoint.node_shapes,
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: endpoint.edge_shapes,
        }
    )
    flow = HamiltonianCFM(
        {
            "enabled": True, "output_space": "ao_block_ode", "block_ode": True,
            "target_semantics": "absolute_full_h", "time_conditioning_required": True,
            "node_block_target_key": _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            "edge_block_target_key": _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            "node_block_shape_key": _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
            "edge_block_shape_key": _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
            "validation_ode_steps": [1, 3], "loss_type": "l1_rmse", "component_reduction": "global_elements",
        },
        idp=mapper, dtype=torch.float64,
    )
    prepared, prepared_ref, context = flow.prepare_batch(data, ref, t=torch.tensor([0.5]))
    prediction = dict(prepared)
    prediction.update(
        {_keys.NODE_PRED_HAMIL_BLOCKS_KEY: endpoint.node_blocks, _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: endpoint.edge_blocks}
    )
    loss, state = flow.loss(prediction, prepared_ref, context)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert state["train_flow_hopping_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# Generic ao_block_ode route: closed-form training loss + directed-full stats
# ---------------------------------------------------------------------------
_GLOBAL_L1_RMSE_CLOSED_FORM = 0.5 * (5.0 / 3.0 + math.sqrt(11.0 / 3.0))


@pytest.mark.parametrize(
    ("node_values", "edge_value", "expected"),
    [([1.0, 1.0], 3.0, _GLOBAL_L1_RMSE_CLOSED_FORM), ([0.0, 0.0], 0.0, 0.0)],
)
def test_block_ode_global_l1_rmse_matches_closed_form(node_values, edge_value, expected):
    """The training loss on a 2-atom H-H dimer (zero H0, a constant pred) equals the closed-form
    l1_rmse over the canonical population (2 onsite + 1 canonical edge): 0.5*(5/3 + sqrt(11/3))."""
    idp = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    atom_type = idp.chemical_symbol_to_type["H"]
    edge_type = idp.bond_to_type["H-H"]
    data = {
        _keys.POSITIONS_KEY: torch.zeros((2, 3), dtype=torch.float64),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float64).unsqueeze(0),
        _keys.PBC_KEY: torch.tensor([False, False, False]),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros((2, 3), dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.full((2,), atom_type, dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full((2,), edge_type, dtype=torch.long),
    }
    shapes = torch.ones((2, 2), dtype=torch.long)
    h0 = BlockTensorResult(
        torch.zeros((2, 1, 1), dtype=torch.float64), torch.zeros((2, 1, 1), dtype=torch.float64), shapes, shapes.clone()
    )
    codec = BlockStateCodec(idp, dtype=torch.float64)
    node_h0, edge_h0 = codec.blocks_to_rme(data, h0)
    data.update(
        {
            _keys.NODE_H0_KEY: node_h0, _keys.EDGE_H0_KEY: edge_h0,
            _keys.NODE_FEATURES_KEY: node_h0.clone(), _keys.EDGE_FEATURES_KEY: edge_h0.clone(),
            _keys.NODE_H0_BLOCKS_KEY: h0.node_blocks, _keys.EDGE_H0_BLOCKS_KEY: h0.edge_blocks,
            _keys.NODE_H0_BLOCK_SHAPE_KEY: h0.node_shapes, _keys.EDGE_H0_BLOCK_SHAPE_KEY: h0.edge_shapes,
        }
    )
    ref = _fresh(data)
    ref.update(
        {
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: h0.node_blocks.clone(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: h0.edge_blocks.clone(),
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: h0.node_shapes.clone(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: h0.edge_shapes.clone(),
        }
    )
    flow = _flow(idp, loss_type="l1_rmse", component_reduction="global_elements", omit_time_scaling=True)
    prepared, prepared_ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.0], dtype=torch.float64))
    pred = dict(prepared)
    pred.update(
        {
            _keys.NODE_PRED_HAMIL_BLOCKS_KEY: torch.tensor(node_values, dtype=torch.float64).reshape(2, 1, 1),
            _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: torch.full((2, 1, 1), edge_value, dtype=torch.float64),
        }
    )
    actual, _ = flow.loss(pred, prepared_ref, ctx)
    assert actual.item() == pytest.approx(expected, abs=1.0e-12)


def test_new_loss_requires_both_components_exact_shapes_and_physical_target():
    idp, data, codec, _ = _case()
    flow = _flow(idp)
    node_target = data["node_h0"] * 1.4
    edge_target = data["edge_h0"] * 1.4
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, endpoint, node_target, edge_target)
    batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))
    pred = batch.copy()
    pred[flow.node_output_key] = endpoint.node_blocks
    pred[flow.edge_output_key] = endpoint.edge_blocks
    loss, state = flow.loss(pred, ref, ctx)
    assert loss.item() <= FP64_ATOL
    # The compatible-stats population is every shape-active directed AO entry (H 1x1 + C 3x3 = 10
    # onsite, edge(1,3) + edge(3,1) = 6 hopping) -- NOT the canonical independent-freedom subset
    # (onsite upper triangle = 7, one reverse-edge side = 3) train_flow_*_loss uses for training.
    assert state["_compatible_clean_stats"]["onsite_count"].item() == 10
    assert state["_compatible_clean_stats"]["hopping_count"].item() == 6

    polluted_topology = pred.copy()
    polluted_topology[_keys.EDGE_CELL_SHIFT_KEY] = torch.ones_like(pred[_keys.EDGE_CELL_SHIFT_KEY])
    polluted_loss, _ = flow.loss(polluted_topology, ref, ctx)
    assert polluted_loss.item() <= FP64_ATOL

    missing = pred.copy()
    missing.pop(flow.edge_output_key)
    with pytest.raises(KeyError, match="missing required keys"):
        flow.loss(missing, ref, ctx)

    bad_ref = ref.copy()
    bad_ref[flow.node_block_shape_key] = ref[flow.node_block_shape_key].clone()
    bad_ref[flow.node_block_shape_key][0] = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="node_shapes disagrees"):
        flow.loss(pred, bad_ref, ctx)

    padded_ref = ref.copy()
    padded_ref[flow.node_block_target_key] = ref[flow.node_block_target_key].clone()
    padded_ref[flow.node_block_target_key][0, 2, 2] = 1e-3
    with pytest.raises(ValueError, match="target violates"):
        flow.loss(pred, padded_ref, ctx)


def test_block_ode_compatible_stats_match_criterion_directed_population():
    """flow's published block-endpoint stats == criterion's own ``block_components``, and reducing
    them through ``compatible_loss_from_stats`` reproduces the criterion's own reduction -- on a
    Hermitian-symmetric-error sample where projection is provably a no-op on both pred and target."""
    idp, data, _codec, h0_blocks = _case()
    batch = _fresh(data)
    batch["node_full_hamil_target_blocks"] = h0_blocks.node_blocks.clone()
    batch["edge_full_hamil_target_blocks"] = h0_blocks.edge_blocks.clone()
    batch["node_full_hamil_target_block_shape"] = h0_blocks.node_shapes.clone()
    batch["edge_full_hamil_target_block_shape"] = h0_blocks.edge_shapes.clone()

    flow = _flow(idp)  # semantics="absolute_full_h" default
    assert not flow.uureal_block_ode and flow.block_ode

    _prepared, ref, ctx = flow.prepare_batch(_fresh(batch), _fresh(batch), t=torch.tensor([0.0]))
    pred = dict(ref)
    pred[flow.node_output_key] = 2.0 * torch.as_tensor(ref[flow.node_block_target_key])
    pred[flow.edge_output_key] = 2.0 * torch.as_tensor(ref[flow.edge_block_target_key])

    loss, state = flow.loss(pred, ref, ctx)
    assert torch.isfinite(loss)
    stats = state["_compatible_clean_stats"]
    assert stats["metric_space"] == "block"

    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    pred_state = BlockTensorResult(pred[flow.node_output_key], pred[flow.edge_output_key], node_shapes, edge_shapes)
    target_state = BlockTensorResult(
        torch.as_tensor(ref[flow.node_block_target_key]), torch.as_tensor(ref[flow.edge_block_target_key]),
        node_shapes, edge_shapes,
    )
    pred_projected = project_block_state(ref, idp, pred_state)
    target_projected = project_block_state(ref, idp, target_state)
    # Precondition this test relies on: projection is exactly a no-op here.
    torch.testing.assert_close(pred_projected.node_blocks, pred_state.node_blocks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pred_projected.edge_blocks, pred_state.edge_blocks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target_projected.node_blocks, target_state.node_blocks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target_projected.edge_blocks, target_state.edge_blocks, rtol=0.0, atol=0.0)

    node_comp = block_components(pred_projected.node_blocks, target_projected.node_blocks, node_shapes, complex_reduction="modulus")
    edge_comp = block_components(pred_projected.edge_blocks, target_projected.edge_blocks, edge_shapes, complex_reduction="modulus")
    torch.testing.assert_close(stats["onsite_l1_sum"], node_comp.abs_sum, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(stats["onsite_mse_sum"], node_comp.square_sum, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(stats["onsite_count"], node_comp.count, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(stats["hopping_l1_sum"], edge_comp.abs_sum, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(stats["hopping_mse_sum"], edge_comp.square_sum, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(stats["hopping_count"], edge_comp.count, rtol=0.0, atol=1e-12)

    criterion = HamilBlockwiseNexTHamLoss(basis={"H": "1s"})
    assert criterion.endpoint_metric_space == "block"
    total_from_flow, onsite_from_flow, hopping_from_flow = criterion.compatible_loss_from_stats(
        onsite_l1_sum=stats["onsite_l1_sum"], onsite_mse_sum=stats["onsite_mse_sum"], onsite_count=stats["onsite_count"],
        hopping_l1_sum=stats["hopping_l1_sum"], hopping_mse_sum=stats["hopping_mse_sum"], hopping_count=stats["hopping_count"],
    )
    onsite_direct = l1_rmse_from_components(node_comp, eps=criterion.eps)
    hopping_direct = l1_rmse_from_components(edge_comp, eps=criterion.eps)
    total_direct = 0.5 * (onsite_direct + hopping_direct)
    torch.testing.assert_close(onsite_from_flow, onsite_direct, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(hopping_from_flow, hopping_direct, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(total_from_flow, total_direct, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# residual_ao_block_ode: training loss == hand pooled masked MSE; H0 cancels
# between the residual training scalar and the assembled full-H sample score
# ---------------------------------------------------------------------------
def _pooled_masked_mse(ref, idp, pred_state, target_state):
    """Reproduce ``_block_ode_endpoint_loss``'s pooled global_elements MSE."""
    pred = project_block_state(ref, idp, pred_state)
    target = project_block_state(ref, idp, target_state)
    node_diff = pred.node_blocks - target.node_blocks
    edge_diff = pred.edge_blocks - target.edge_blocks

    node_valid = block_mask_from_shapes(pred.node_shapes, tuple(node_diff.shape[-2:]))
    upper = torch.triu(torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool))
    node_mask = node_valid & upper.unsqueeze(0)

    edge_valid = block_mask_from_shapes(pred.edge_shapes, tuple(edge_diff.shape[-2:]))
    rev = strict_reverse_edge_index(ref, idp=idp)
    rows = torch.arange(edge_diff.shape[0])
    canonical_rows = rows <= rev
    edge_mask = edge_valid & canonical_rows.view(-1, 1, 1)
    self_reverse = rows == rev
    if bool(self_reverse.any()):
        edge_mask[self_reverse] &= upper.unsqueeze(0)

    node_f = node_mask.to(node_diff.dtype)
    edge_f = edge_mask.to(edge_diff.dtype)
    square_sum = (node_diff.square() * node_f).sum() + (edge_diff.square() * edge_f).sum()
    count = node_f.sum() + edge_f.sum()
    return square_sum / count.clamp_min(1.0)


def test_residual_training_loss_matches_golden_pooled_masked_mse():
    mapper = _mapper()
    data, _h0, _d1 = _b_record(mapper)
    flow = _b_flow(mapper)
    batch, ref, ctx = flow.prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.4], dtype=torch.float64))
    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    dhat_node = 2.0 * torch.as_tensor(ref[flow.node_block_target_key])
    dhat_edge = 2.0 * torch.as_tensor(ref[flow.edge_block_target_key])

    pred = batch.copy()
    pred[flow.node_output_key] = dhat_node
    pred[flow.edge_output_key] = dhat_edge
    loss, _state = flow.loss(pred, ref, ctx)

    golden = _pooled_masked_mse(
        ref, mapper,
        BlockTensorResult(dhat_node, dhat_edge, node_shapes, edge_shapes),
        BlockTensorResult(
            torch.as_tensor(ref[flow.node_block_target_key]), torch.as_tensor(ref[flow.edge_block_target_key]),
            node_shapes, edge_shapes,
        ),
    )
    assert golden.item() > 0.0
    torch.testing.assert_close(loss, golden, rtol=0.0, atol=1e-12)


def test_residual_h0_cancellation_parity_and_sample_loss_routing():
    """Scoring the assembled full-H sample equals the residual training scalar (H0 cancels), and
    loss_on_sample routes non-uureal block_ode through the compatible path."""
    mapper = _mapper()
    data, _h0, _d1 = _b_record(mapper)
    flow = _b_flow(mapper)
    batch, ref, ctx = flow.prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.4], dtype=torch.float64))
    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    dhat = BlockTensorResult(
        2.0 * torch.as_tensor(ref[flow.node_block_target_key]), 2.0 * torch.as_tensor(ref[flow.edge_block_target_key]),
        node_shapes, edge_shapes,
    )

    train_pred = batch.copy()
    train_pred[flow.node_output_key] = dhat.node_blocks
    train_pred[flow.edge_output_key] = dhat.edge_blocks
    train_loss, _ = flow.loss(train_pred, ref, ctx)

    h0_from_base = flow.block_codec.rme_to_blocks(ref, ctx.node_base, ctx.edge_base, project=True)
    full = flow.block_codec.endpoint_to_full(dhat, h0_from_base)
    sample_pred = batch.copy()
    sample_pred[flow.node_output_key] = full.node_blocks
    sample_pred[flow.edge_output_key] = full.edge_blocks

    compatible_loss, _ = flow.compatible_loss_on_sample(sample_pred, ref, ctx)
    torch.testing.assert_close(compatible_loss, train_loss, rtol=0.0, atol=1e-10)

    routed_loss, _ = flow.loss_on_sample(sample_pred, ref, ctx)
    torch.testing.assert_close(routed_loss, compatible_loss, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# uureal_block_ode: directed-vs-canonical compatible metric, and the
# compatible scorer scoring a route with no block_codec at all
# ---------------------------------------------------------------------------
def test_uureal_directed_compatible_metric_coexists_with_canonical_and_matches_hand_reduction():
    """The historical directed-coordinate metric (train_compatible_directed_*, every stored
    directed AO coordinate) coexists with the canonical training metric (train_flow_*, one
    independent freedom each) and both match an independent hand reduction, on a sample where the
    two disagree on hopping (2 directed edges vs 1 canonical)."""
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    flow = _uureal_flow(mapper)

    _model_data, ref, ctx = flow.prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.0]))
    pred = dict(ref)
    pred[flow.node_output_key] = 2.0 * torch.as_tensor(ref[flow.node_block_target_key])
    pred[flow.edge_output_key] = 2.0 * torch.as_tensor(ref[flow.edge_block_target_key])
    _loss, state = flow.loss_on_sample(pred, ref, ctx)

    for key in ("train_compatible_directed_onsite_loss", "train_compatible_directed_hopping_loss", "train_compatible_directed_loss"):
        assert key in state, key

    node_diff = torch.as_tensor(ref[flow.node_block_target_key], dtype=torch.float32)
    edge_diff = torch.as_tensor(ref[flow.edge_block_target_key], dtype=torch.float32)
    node_valid = block_mask_from_shapes(torch.as_tensor(ref[flow.node_block_shape_key]), tuple(node_diff.shape[-2:]))
    edge_valid = block_mask_from_shapes(torch.as_tensor(ref[flow.edge_block_shape_key]), tuple(edge_diff.shape[-2:]))
    upper = torch.triu(torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool))
    rev = strict_reverse_edge_index(ref, idp=mapper)
    canonical_rows = torch.arange(edge_diff.shape[0]) <= rev
    assert flow.loss_type == "mse"  # hand reduction below assumes the default

    def hand_metric(diff, mask):
        values = diff[mask]
        return values.square().sum() / float(values.numel())

    canonical_onsite = hand_metric(node_diff, node_valid & upper.unsqueeze(0))
    canonical_hopping = hand_metric(edge_diff, edge_valid & canonical_rows.view(-1, 1, 1))
    directed_onsite = hand_metric(node_diff, node_valid)
    directed_hopping = hand_metric(edge_diff, edge_valid)

    torch.testing.assert_close(state["train_flow_onsite_loss"], canonical_onsite, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state["train_flow_hopping_loss"], canonical_hopping, rtol=1e-6, atol=1e-7)
    assert "train_onsite_loss" not in state and "train_hopping_loss" not in state
    torch.testing.assert_close(state["train_compatible_directed_onsite_loss"], directed_onsite, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state["train_compatible_directed_hopping_loss"], directed_hopping, rtol=1e-6, atol=1e-7)
    # The two reductions genuinely differ on this sample's coordinate counts: 2 directed edges
    # vs 1 canonical edge (H-C / C-H Hermitian pair).
    assert int((edge_valid & canonical_rows.view(-1, 1, 1)).sum()) < int(edge_valid.sum())


def test_uureal_compatible_scorer_scores_residual_without_block_codec():
    """uureal_block_ode intentionally has no block_codec (the rollout returns residual-dH blocks
    directly); the compatible scorer must route to the residual-dH scorer rather than the Full-H
    path, and agree with loss_on_sample."""
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    flow = _uureal_flow(mapper, log_validation_flow_euler_loss=False, validation_ode_steps=[1])
    assert flow.block_codec is None

    node = data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY]
    edge = data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY]
    spy = _EndpointSpy([(node, edge)], "node_h0", "edge_h0", state_keys=UUREAL_STATE_KEYS)
    sampled = flow.sample(spy, copy.deepcopy(data), num_steps=1)

    zero_t = torch.zeros(1)
    _flow_batch, flow_ref, flow_ctx = flow.prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=zero_t)
    loss, _state = flow.compatible_loss_on_sample(sampled, flow_ref, flow_ctx)
    assert torch.isfinite(loss)
    reference_loss, _ = flow.loss_on_sample(sampled, flow_ref, flow_ctx)
    torch.testing.assert_close(loss, reference_loss, rtol=0.0, atol=0.0)
