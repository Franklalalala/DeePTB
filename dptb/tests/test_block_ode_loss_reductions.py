from __future__ import annotations

import math

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult, infer_block_shapes
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM


def test_l1_rmse_global_elements_reduces_once_after_accumulation():
    node = torch.tensor([0.0], dtype=torch.float64)
    edge = torch.tensor([4.0], dtype=torch.float64)
    node_stats = HamiltonianCFM._metric_stats(
        node, torch.ones_like(node, dtype=torch.bool), "l1_rmse"
    )
    edge_stats = HamiltonianCFM._metric_stats(
        edge, torch.ones_like(edge, dtype=torch.bool), "l1_rmse"
    )

    result = HamiltonianCFM._global_metric(
        node_stats[1] + edge_stats[1],
        node_stats[2] + edge_stats[2],
        node_stats[3] + edge_stats[3],
        "l1_rmse",
    )
    expected = 0.5 * (2.0 + math.sqrt(8.0))
    assert result.item() == pytest.approx(expected, abs=1.0e-12)
    assert result.item() != pytest.approx(2.0, abs=1.0e-6)


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


def test_empty_metric_component_has_zero_count_and_zero_value():
    diff = torch.empty((0, 2, 2), dtype=torch.float64)
    stats = HamiltonianCFM._metric_stats(
        diff, torch.empty_like(diff, dtype=torch.bool), "l1_rmse"
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
    h0 = project_block_state(
        data,
        mapper,
        BlockTensorResult(
            node_blocks=torch.tensor([[[1.0]]], dtype=torch.float64),
            edge_blocks=torch.empty((0, 1, 1), dtype=torch.float64),
            node_shapes=node_shapes,
            edge_shapes=edge_shapes,
        ),
    )
    endpoint = project_block_state(
        data,
        mapper,
        BlockTensorResult(
            node_blocks=torch.tensor([[[2.0]]], dtype=torch.float64),
            edge_blocks=torch.empty((0, 1, 1), dtype=torch.float64),
            node_shapes=node_shapes,
            edge_shapes=edge_shapes,
        ),
    )
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
            "mode": "residual",
            "prior": "zero",
            "output_space": "ao_block_ode",
            "block_ode": True,
            "target_semantics": "absolute_full_h",
            "prediction_add_h0": False,
            "time_conditioning_required": True,
            "strict_h0": True,
            "block_inverse_mode": "strict",
            "block_inverse_atol": 1.0e-10,
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
            _keys.BLOCK_PRED_ACTIVE_EDGES_KEY: torch.empty((0,), dtype=torch.long),
        }
    )
    loss, state = flow.loss(prediction, prepared_ref, context)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert state["train_flow_hopping_loss"].item() == 0.0
