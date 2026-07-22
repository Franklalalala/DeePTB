from __future__ import annotations

"""Characterization test for the ``block_ode/endpoint_stats.py`` extraction.

``HamiltonianCFM._block_ode_endpoint_loss`` was moved verbatim (``self.`` ->
``owner.``) to :func:`dptb.nnops.block_ode.endpoint_stats.block_ode_endpoint_loss`;
``HamiltonianCFM`` keeps the original method name as a 1-line delegator. This
test pins two invariants on one fixed-seed ``uureal_block_ode`` fixture and one
fixed-seed ``residual_ao_block_ode`` fixture:

  1. Calling the production entry point (``flow.loss_on_sample(...)``, which
     resolves through the delegator) and calling the extracted module function
     directly (``endpoint_stats.block_ode_endpoint_loss(flow, ...)``) with the
     same route-appropriate ``prediction_is_full_h`` produce bit-identical
     ``train_flow_*``/``train_compatible_directed_*`` state entries -- proving
     the delegator is a transparent forward, not just "returns something".
  2. The numeric values themselves are locked against a hardcoded snapshot, so
     any future change to the moved body's arithmetic (not just its location)
     is caught here even if every other block-ode suite happens not to exercise
     that particular code path.

Fixtures are self-contained (seeded ``torch.Generator``, no cross-import of
other test files' private helpers) and structurally mirror the existing
``_record``/``_flow`` (uureal) and ``_b_record``/``_b_flow`` (residual)
builders in ``test_uureal_block_ode.py``/``test_residual_ao_block_ode.py``,
with hand-picked constants replaced by seeded draws.
"""

import copy

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_tensors_to_feature_tensors,
    infer_block_shapes,
    mapper_max_norb,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import project_block_state
from dptb.nnops.block_ode import endpoint_stats
from dptb.nnops.flow import HamiltonianCFM

_STATE_KEY_PREFIXES = ("train_flow_", "train_compatible_directed_")


def _state_keys_of_interest(state):
    return {key for key in state if key.startswith(_STATE_KEY_PREFIXES)}


def _assert_state_dicts_match_bitwise(state_a, state_b):
    keys_a = _state_keys_of_interest(state_a)
    keys_b = _state_keys_of_interest(state_b)
    assert keys_a == keys_b, (keys_a, keys_b)
    assert keys_a, "expected at least one train_flow_*/train_compatible_directed_* key"
    for key in sorted(keys_a):
        assert torch.equal(state_a[key], state_b[key]), key


# ---------------------------------------------------------------------------
# uureal_block_ode fixture (compact-uu-real SOC route)
# ---------------------------------------------------------------------------
_UUREAL_SEED = 20260722


def _uureal_mapper():
    mapper = OrbitalMapper(
        {"H": "1s", "C": "1s1p"},
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


def _uureal_data(mapper, *, seed):
    data = {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor(
            [[mapper.chemical_symbol_to_type["H"]], [mapper.chemical_symbol_to_type["C"]]]
        ),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3),
        "edge_type": torch.tensor(
            [[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]
        ),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": torch.eye(3) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
    }
    generator = torch.Generator().manual_seed(seed)
    node = torch.zeros(2, 4, 4)
    node[0, 0, 0] = torch.randn((), generator=generator).item()
    c = torch.randn(4, 4, generator=generator)
    node[1] = 0.5 * (c + c.T)
    edge = torch.zeros(2, 4, 4)
    edge_row = torch.randn(1, 4, generator=generator)
    edge[0, :1, :4] = edge_row
    edge[1, :4, :1] = edge_row.T
    packed = BlockTensorResult(
        node, edge, torch.tensor([[1, 1], [4, 4]]), torch.tensor([[1, 4], [4, 1]])
    )
    node_h0, edge_h0 = block_tensors_to_feature_tensors(
        data, mapper, node_blocks=node * 0.5, edge_blocks=edge * 0.5
    )
    keep = int(mapper.reduced_matrix_element)
    data.update(
        {
            "node_h0": node_h0,
            "edge_h0": edge_h0,
            "node_delta_hamil_blocks": packed.node_blocks,
            "edge_delta_hamil_blocks": packed.edge_blocks,
            "node_delta_hamil_block_shape": packed.node_shapes,
            "edge_delta_hamil_block_shape": packed.edge_shapes,
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "blockwise_source_target_feature_width": keep,
            "blockwise_source_h0_feature_width": keep,
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
    )
    return data


def _uureal_flow(mapper):
    options = {
        "enabled": True,
        "mode": "residual",
        "prior": "zero",
        "output_space": "uureal_block_ode",
        "block_ode": True,
        "state_space": "residual_ao_block",
        "target_semantics": "residual_dh",
        "block_input_adapter": "direct_cg",
        "h0_condition_space": "compact_uureal_rme",
        "block_export_final_full_h": False,
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1, 3],
    }
    return HamiltonianCFM(options, idp=mapper, dtype=torch.float32)


def _uureal_case():
    """Build one fixed-seed ``uureal_block_ode`` (flow, pred, ref, ctx) tuple.

    ``loss_on_sample`` routes ``uureal_block_ode`` directly through
    ``_block_ode_endpoint_loss(..., prediction_is_full_h=False)`` (the compact
    residual-dH rollout never assembles physical Full-H) -- see
    ``HamiltonianCFM.loss_on_sample``.
    """
    mapper = _uureal_mapper()
    data = _uureal_data(mapper, seed=_UUREAL_SEED)
    flow = _uureal_flow(mapper)
    _model_data, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.3])
    )
    target_node = torch.as_tensor(ref[flow.node_block_target_key])
    target_edge = torch.as_tensor(ref[flow.edge_block_target_key])
    generator = torch.Generator().manual_seed(_UUREAL_SEED + 1)
    pred = dict(ref)
    pred[flow.node_output_key] = target_node + 0.1 * torch.randn(
        tuple(target_node.shape), generator=generator, dtype=target_node.dtype
    )
    pred[flow.edge_output_key] = target_edge + 0.1 * torch.randn(
        tuple(target_edge.shape), generator=generator, dtype=target_edge.dtype
    )
    return flow, pred, ref, ctx, False


# ---------------------------------------------------------------------------
# residual_ao_block_ode fixture (plain non-SOC direct-residual route)
# ---------------------------------------------------------------------------
_RESIDUAL_SEED = 20260722


def _residual_mapper():
    mapper = OrbitalMapper({"H": "1s", "C": "1s1p"}, method="e3tb")
    mapper.get_irreps()
    return mapper


def _residual_graph(mapper, *, dtype=torch.float64):
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]
    return {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor([[t_h], [t_c]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3, dtype=dtype),
        "edge_type": torch.tensor(
            [[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]
        ),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
        # Seeded prior draws (unused here, prior="zero") require a stable uid;
        # kept for structural parity with the production data contract.
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }


def _residual_projected_state(mapper, data, *, canvas, n, e, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return project_block_state(
        data,
        mapper,
        BlockTensorResult(
            torch.randn(n, canvas, canvas, generator=generator, dtype=dtype),
            torch.randn(e, canvas, canvas, generator=generator, dtype=dtype),
            *infer_block_shapes(data, mapper),
        ),
    )


def _residual_record(mapper, *, dtype=torch.float64, seed):
    data = _residual_graph(mapper, dtype=dtype)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    n = int(node_shapes.shape[0])
    e = int(edge_shapes.shape[0])
    h0 = _residual_projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed)
    d1 = _residual_projected_state(
        mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed + 1000
    )
    data[_keys.NODE_H0_BLOCKS_KEY] = h0.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0.edge_shapes.clone()
    data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY] = d1.node_blocks.clone()
    data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY] = d1.edge_blocks.clone()
    data[_keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = d1.node_shapes.clone()
    data[_keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = d1.edge_shapes.clone()
    return data


def _residual_flow(mapper, *, dtype=torch.float64):
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "zero",
        "output_space": "residual_ao_block_ode",
        "block_ode": True,
        "state_space": "residual_ao_block",
        "block_input_adapter": "direct_cg",
        "h0_condition_space": "spatial_h0_rme",
        "block_export_final_full_h": True,
        "target_semantics": "residual_dh",
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "strict_h0": True,
        "t0_probability": 0.15,
        "block_inverse_mode": "strict",
        "block_inverse_atol": 1e-10 if dtype == torch.float64 else 2e-5,
        "strict_certification": "always",
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1],
    }
    return HamiltonianCFM(options, idp=mapper, dtype=dtype)


def _residual_case():
    """Build one fixed-seed ``residual_ao_block_ode`` (flow, pred, ref, ctx) tuple.

    ``loss_on_sample`` routes non-``uureal`` ``block_ode`` through
    ``compatible_loss_on_sample``, which scores a physical Full-H sample via
    ``_block_ode_endpoint_loss(..., prediction_is_full_h=True)`` -- see
    ``HamiltonianCFM.loss_on_sample``/``compatible_loss_on_sample``.
    """
    mapper = _residual_mapper()
    data = _residual_record(mapper, seed=_RESIDUAL_SEED)
    flow = _residual_flow(mapper)
    batch, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.4], dtype=torch.float64)
    )
    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    target_node = torch.as_tensor(ref[flow.node_block_target_key])
    target_edge = torch.as_tensor(ref[flow.edge_block_target_key])
    generator = torch.Generator().manual_seed(_RESIDUAL_SEED + 2000)
    dhat = BlockTensorResult(
        target_node
        + 0.1 * torch.randn(tuple(target_node.shape), generator=generator, dtype=target_node.dtype),
        target_edge
        + 0.1 * torch.randn(tuple(target_edge.shape), generator=generator, dtype=target_edge.dtype),
        node_shapes,
        edge_shapes,
    )
    h0_from_base = flow.block_codec.rme_to_blocks(
        ref, ctx.node_base, ctx.edge_base, project=True
    )
    full = flow.block_codec.endpoint_to_full(dhat, h0_from_base)
    sample_pred = batch.copy()
    sample_pred[flow.node_output_key] = full.node_blocks
    sample_pred[flow.edge_output_key] = full.edge_blocks
    return flow, sample_pred, ref, ctx, True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_fn", [_uureal_case, _residual_case], ids=["uureal", "residual"])
def test_delegator_matches_module_function_bitwise(case_fn):
    flow, pred, ref, ctx, prediction_is_full_h = case_fn()

    # (a) Production entry point: resolves through HamiltonianCFM's delegator.
    delegator_loss, delegator_state = flow.loss_on_sample(
        copy.deepcopy(pred), copy.deepcopy(ref), ctx
    )

    # (b) Same computation, calling the extracted module function directly.
    direct_loss, direct_state = endpoint_stats.block_ode_endpoint_loss(
        flow,
        copy.deepcopy(pred),
        copy.deepcopy(ref),
        ctx,
        prediction_is_full_h=prediction_is_full_h,
    )

    assert torch.equal(delegator_loss, direct_loss)
    _assert_state_dicts_match_bitwise(delegator_state, direct_state)


def test_uureal_snapshot_locked_to_fixed_seed_values():
    flow, pred, ref, ctx, _prediction_is_full_h = _uureal_case()
    loss, state = flow.loss_on_sample(pred, ref, ctx)

    assert loss.item() == pytest.approx(0.009417356923222542, abs=1e-6)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(0.01225984375923872, abs=1e-6)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(0.0016005175421014428, abs=1e-6)
    assert state["train_flow_loss"].item() == pytest.approx(0.009417356923222542, abs=1e-6)
    assert state["train_compatible_directed_onsite_loss"].item() == pytest.approx(
        0.011145579628646374, abs=1e-6
    )
    assert state["train_compatible_directed_hopping_loss"].item() == pytest.approx(
        0.0016005175421014428, abs=1e-6
    )
    assert state["train_compatible_directed_loss"].item() == pytest.approx(
        0.008091159164905548, abs=1e-6
    )


def test_residual_snapshot_locked_to_fixed_seed_values():
    flow, sample_pred, ref, ctx, _prediction_is_full_h = _residual_case()
    loss, state = flow.loss_on_sample(sample_pred, ref, ctx)

    assert "train_compatible_directed_onsite_loss" not in state
    assert loss.item() == pytest.approx(0.006981479957376853, abs=1e-9)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(0.0057569340393279845, abs=1e-9)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(0.01034898123201124, abs=1e-9)
    assert state["train_flow_loss"].item() == pytest.approx(0.006981479957376853, abs=1e-9)
