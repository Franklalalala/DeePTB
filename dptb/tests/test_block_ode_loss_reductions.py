from __future__ import annotations

import math

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult, infer_block_shapes
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM, HamiltonianPixelMeanFlow
from dptb.utils.argcheck import validate_flow_loss_contract


def _component_loss(
    loss_type, reduction, node_values, edge_values, *, node_weight=1.0, edge_weight=1.0
):
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "loss_type": loss_type,
            "component_reduction": reduction,
            "node_weight": node_weight,
            "edge_weight": edge_weight,
        }
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
        data,
        mapper,
        BlockTensorResult(
            node_blocks=torch.full((1, 1, 1), node_value, dtype=torch.float64),
            edge_blocks=torch.empty((0, 1, 1), dtype=torch.float64),
            node_shapes=node_shapes,
            edge_shapes=edge_shapes,
        ),
    )


@pytest.mark.parametrize(
    ("loss_type", "reduction", "node", "edge", "weights", "expected"),
    [
        ("mse", "global_elements", [0.0], [4.0], {}, 8.0),
        ("mse", "equal_components", [0.0], [4.0], {}, 16.0),
        ("l1_rmse", "global_elements", [0.0], [4.0], {}, 0.5 * (2.0 + math.sqrt(8.0))),
        ("l1_rmse", "equal_components", [0.0], [4.0], {}, 4.0),
        (
            "mse",
            "equal_components",
            [1.0],
            [3.0],
            {"node_weight": 2.0, "edge_weight": 0.5},
            6.5,
        ),
        (
            "l1_rmse",
            "equal_components",
            [1.0],
            [3.0],
            {"node_weight": 2.0, "edge_weight": 0.5},
            3.5,
        ),
    ],
)
def test_component_reducer_has_distinct_global_and_component_semantics(
    loss_type, reduction, node, edge, weights, expected
):
    result = _component_loss(loss_type, reduction, node, edge, **weights)
    assert result.item() == pytest.approx(expected, abs=1.0e-12)


@pytest.mark.parametrize(
    "check",
    [
        pytest.param(_build_cfm, id="runtime"),
        pytest.param(_validate_argcheck_flow, id="argcheck"),
    ],
)
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
        "enabled": True,
        "objective": "pixel_meanflow",
        "component_reduction": "global_elements",
        "node_weight": 2.0,
        "edge_weight": 0.5,
    }
    flow = HamiltonianPixelMeanFlow(options)
    assert (flow.node_weight, flow.edge_weight) == (2.0, 0.5)
    assert _validate_argcheck_flow(options) is None
    with pytest.raises(ValueError, match="finite and non-negative"):
        HamiltonianPixelMeanFlow({**options, "node_weight": math.inf})


def test_l1_rmse_exact_zero_has_zero_value_and_finite_zero_gradient():
    diff = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    stats = HamiltonianCFM._metric_stats(
        diff, torch.ones_like(diff, dtype=torch.bool), "l1_rmse"
    )
    global_metric = HamiltonianCFM._global_metric(
        stats[1], stats[2], stats[3], "l1_rmse"
    )

    assert stats[0].item() == 0.0
    assert global_metric.item() == 0.0
    (stats[0] + global_metric).backward()
    assert torch.isfinite(diff.grad).all()
    assert torch.count_nonzero(diff.grad).item() == 0


@pytest.mark.parametrize("loss_type", ["mse", "l1_rmse"])
def test_empty_metric_component_has_zero_count_and_zero_value(loss_type):
    diff = torch.empty((0, 2, 2), dtype=torch.float64)
    stats = HamiltonianCFM._metric_stats(
        diff, torch.empty_like(diff, dtype=torch.bool), loss_type
    )
    assert stats[0].item() == 0.0
    assert stats[1].item() == 0.0
    assert stats[2].item() == 0.0
    assert stats[3].item() == 0.0


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
            _keys.NODE_H0_KEY: node_h0,
            _keys.EDGE_H0_KEY: edge_h0,
            _keys.NODE_H0_BLOCKS_KEY: h0.node_blocks,
            _keys.EDGE_H0_BLOCKS_KEY: h0.edge_blocks,
            _keys.NODE_H0_BLOCK_SHAPE_KEY: h0.node_shapes,
            _keys.EDGE_H0_BLOCK_SHAPE_KEY: h0.edge_shapes,
            _keys.NODE_FEATURES_KEY: node_target,
            _keys.EDGE_FEATURES_KEY: edge_target,
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
            "enabled": True,
            "output_space": "ao_block_ode",
            "block_ode": True,
            "target_semantics": "absolute_full_h",
            "time_conditioning_required": True,
            "node_block_target_key": _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            "edge_block_target_key": _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            "node_block_shape_key": _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
            "edge_block_shape_key": _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
            "validation_ode_steps": [1, 3],
            "loss_type": "l1_rmse",
            "component_reduction": "global_elements",
        },
        idp=mapper,
        dtype=torch.float64,
    )
    prepared, prepared_ref, context = flow.prepare_batch(data, ref, t=torch.tensor([0.5]))
    prediction = dict(prepared)
    prediction.update(
        {
            _keys.NODE_PRED_HAMIL_BLOCKS_KEY: endpoint.node_blocks,
            _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: endpoint.edge_blocks,
        }
    )
    loss, state = flow.loss(prediction, prepared_ref, context)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert state["train_flow_hopping_loss"].item() == 0.0
