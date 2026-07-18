from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dptb.data.interfaces.blockwise_tensor import BlockTensorResult, block_mask_from_shapes
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import (
    HamiltonianCFM,
    assert_flow_h0_keys_reach_model,
)
from dptb.utils.argcheck import validate_block_ode_contract


ATOL = 1e-10


def _case():
    idp = OrbitalMapper({"H": ["1s"], "C": ["2p"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    atom_types = torch.tensor(
        [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]]
    )
    edge_types = torch.tensor(
        [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]]
    )
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]], dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64) * 5.0,
        "batch": torch.zeros(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "atom_types": atom_types,
        "edge_types": edge_types,
    }
    codec = BlockStateCodec(idp, dtype=torch.float64)
    raw = BlockTensorResult(
        torch.randn(2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(11)),
        torch.randn(2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(12)),
        torch.tensor([[1, 1], [3, 3]]),
        torch.tensor([[1, 3], [3, 1]]),
    )
    h0_blocks = project_block_state(data, idp, raw)
    node_h0, edge_h0 = codec.blocks_to_rme(data, h0_blocks)
    data["node_h0"] = node_h0
    data["edge_h0"] = edge_h0
    data["node_features"] = node_h0.clone()
    data["edge_features"] = edge_h0.clone()
    return idp, data, codec, h0_blocks


def _fresh(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def _flow(idp, semantics="absolute_full_h", **updates):
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "zero",
        "output_space": "ao_block_ode",
        "block_ode": True,
        "target_semantics": semantics,
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "block_inverse_mode": "strict",
        "block_inverse_atol": ATOL,
        "validation_ode_steps": [1, 3],
    }
    options.update(updates)
    return HamiltonianCFM(options, idp=idp, dtype=torch.float64)


def _scaled_endpoint(codec, data, node_h0, edge_h0, scale):
    return codec.rme_to_blocks(data, node_h0 * scale, edge_h0 * scale, project=True)


class EndpointSequence(torch.nn.Module):
    def __init__(self, endpoints):
        super().__init__()
        self.endpoints = list(endpoints)
        self.inputs = []
        self.times = []

    def forward(self, data):
        index = len(self.inputs)
        self.inputs.append((data["node_h0"].clone(), data["edge_h0"].clone()))
        self.times.append(data["flow_time"].clone())
        endpoint = self.endpoints[min(index, len(self.endpoints) - 1)]
        out = data.copy()
        out["node_hamil_blocks"] = endpoint.node_blocks
        out["edge_hamil_blocks"] = endpoint.edge_blocks
        return out


def _blend(data, idp, current, endpoint, alpha):
    return project_block_state(
        data,
        idp,
        BlockTensorResult(
            (1.0 - alpha) * current.node_blocks + alpha * endpoint.node_blocks,
            (1.0 - alpha) * current.edge_blocks + alpha * endpoint.edge_blocks,
            current.node_shapes,
            current.edge_shapes,
        ),
    )


def _assert_state_invariants(data, state):
    assert (state.node_blocks - state.node_blocks.transpose(-1, -2)).abs().max() <= ATOL
    assert (state.edge_blocks[0] - state.edge_blocks[1].T).abs().max() <= ATOL
    node_mask = block_mask_from_shapes(state.node_shapes, tuple(state.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(state.edge_shapes, tuple(state.edge_blocks.shape[-2:]))
    assert torch.count_nonzero(state.node_blocks[~node_mask]) == 0
    assert torch.count_nonzero(state.edge_blocks[~edge_mask]) == 0


def test_spy_second_forward_receives_exact_first_blended_state():
    idp, data, codec, h0 = _case()
    endpoints = [
        _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], scale)
        for scale in (2.0, -0.5, 1.25)
    ]
    model = EndpointSequence(endpoints)
    result = _flow(idp).sample(model, _fresh(data), num_steps=3)

    first_blend = _blend(data, idp, h0, endpoints[0], 1.0 / 3.0)
    expected_node, expected_edge = codec.blocks_to_rme(data, first_blend)
    assert (model.inputs[1][0] - expected_node).abs().max().item() <= ATOL
    assert (model.inputs[1][1] - expected_edge).abs().max().item() <= ATOL

    second_blend = _blend(data, idp, first_blend, endpoints[1], 0.5)
    expected_final = _blend(data, idp, second_blend, endpoints[2], 1.0)
    assert (result["node_hamil_blocks"] - expected_final.node_blocks).abs().max() <= ATOL
    assert (result["edge_hamil_blocks"] - expected_final.edge_blocks).abs().max() <= ATOL


def test_n1_matches_frozen_one_step_adapter_for_physical_endpoint():
    idp, data, codec, _ = _case()
    endpoint = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.7)
    block_model = EndpointSequence([endpoint])
    adapter_model = EndpointSequence([endpoint])
    block_result = _flow(idp).sample(block_model, _fresh(data), num_steps=1)
    adapter = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "zero",
            "output_space": "ao_block",
            "validation_ode_steps": [1],
        },
        idp=idp,
        dtype=torch.float64,
    )
    adapter_result = adapter.sample(adapter_model, _fresh(data), num_steps=1)
    assert torch.equal(block_model.times[0], adapter_model.times[0])
    assert (block_model.inputs[0][0] - adapter_model.inputs[0][0]).abs().max() <= ATOL
    assert (block_model.inputs[0][1] - adapter_model.inputs[0][1]).abs().max() <= ATOL
    assert (block_result["node_hamil_blocks"] - adapter_result["node_hamil_blocks"]).abs().max() <= ATOL
    assert (block_result["edge_hamil_blocks"] - adapter_result["edge_hamil_blocks"]).abs().max() <= ATOL


@pytest.mark.parametrize("num_steps", [2, 3])
def test_two_and_three_step_rollouts_match_manual_endpoint_blends(num_steps):
    idp, data, codec, current = _case()
    raw_endpoints = []
    generator = torch.Generator().manual_seed(100 + num_steps)
    for _ in range(num_steps):
        raw_endpoints.append(
            BlockTensorResult(
                torch.randn(2, 3, 3, dtype=torch.float64, generator=generator),
                torch.randn(2, 3, 3, dtype=torch.float64, generator=generator),
                current.node_shapes,
                current.edge_shapes,
            )
        )
    model = EndpointSequence(raw_endpoints)
    result = _flow(idp).sample(model, _fresh(data), num_steps=num_steps)
    for step, raw in enumerate(raw_endpoints):
        endpoint = project_block_state(data, idp, raw)
        alpha = (1.0 / num_steps) / (1.0 - step / num_steps)
        current = _blend(data, idp, current, endpoint, alpha)
        _assert_state_invariants(data, current)
    assert (result["node_hamil_blocks"] - current.node_blocks).abs().max() <= ATOL
    assert (result["edge_hamil_blocks"] - current.edge_blocks).abs().max() <= ATOL


def test_every_forward_state_and_final_state_are_projected():
    idp, data, codec, _ = _case()
    endpoint = BlockTensorResult(
        torch.randn(2, 3, 3, dtype=torch.float64),
        torch.randn(2, 3, 3, dtype=torch.float64),
        torch.tensor([[1, 1], [3, 3]]),
        torch.tensor([[1, 3], [3, 1]]),
    )
    model = EndpointSequence([endpoint] * 3)
    result = _flow(idp).sample(model, _fresh(data), num_steps=3)
    for node_rme, edge_rme in model.inputs:
        state = codec.rme_to_blocks(data, node_rme, edge_rme)
        _assert_state_invariants(data, state)
    final = BlockTensorResult(
        result["node_hamil_blocks"], result["edge_hamil_blocks"],
        result["node_hamil_block_shape"], result["edge_hamil_block_shape"],
    )
    _assert_state_invariants(data, final)


def test_full_contract_never_adds_h0_and_residual_contract_adds_exactly_once():
    idp, data, codec, h0 = _case()
    delta = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 0.25)
    absolute = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.25)
    full_result = _flow(idp, "absolute_full_h").sample(
        EndpointSequence([absolute]), _fresh(data), num_steps=1
    )
    residual_result = _flow(idp, "residual_dh").sample(
        EndpointSequence([delta]), _fresh(data), num_steps=1
    )
    expected_residual = _blend(data, idp, h0, project_block_state(data, idp, BlockTensorResult(
        h0.node_blocks + delta.node_blocks,
        h0.edge_blocks + delta.edge_blocks,
        h0.node_shapes,
        h0.edge_shapes,
    )), 1.0)
    assert (full_result["node_hamil_blocks"] - absolute.node_blocks).abs().max() <= ATOL
    assert (full_result["node_hamil_blocks"] - (absolute.node_blocks + h0.node_blocks)).abs().max() > 1e-3
    assert (residual_result["node_hamil_blocks"] - expected_residual.node_blocks).abs().max() <= ATOL
    assert (residual_result["edge_hamil_blocks"] - expected_residual.edge_blocks).abs().max() <= ATOL


def test_same_block_state_at_different_t_is_a_different_model_input():
    idp, data, _, h0 = _case()
    model = EndpointSequence([h0, h0])
    _flow(idp).sample(model, _fresh(data), num_steps=2)
    assert (model.inputs[0][0] - model.inputs[1][0]).abs().max() <= ATOL
    assert (model.inputs[0][1] - model.inputs[1][1]).abs().max() <= ATOL
    assert model.times[0].item() == 0.0
    assert model.times[1].item() == 0.5


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"prediction_add_h0": True}, "add_h0=false"),
        ({"prior": "te"}, "prior='zero'"),
        ({"target_semantics": ""}, "explicit target_semantics"),
        ({"time_conditioning_required": False}, "time_conditioning_required=true"),
        ({"validation_ode_steps": [1, 5]}, r"drawn from \[1, 3\]"),
    ],
)
def test_block_ode_constructor_guards(updates, match):
    idp, _, _, _ = _case()
    with pytest.raises(ValueError, match=match):
        _flow(idp, **updates)


def test_block_inverse_default_tolerance_is_dtype_aware():
    idp, _, _, _ = _case()
    base = {
        "enabled": True,
        "mode": "residual",
        "prior": "zero",
        "output_space": "ao_block_ode",
        "block_ode": True,
        "target_semantics": "absolute_full_h",
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "block_inverse_mode": "strict",
        "validation_ode_steps": [1, 3],
    }
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float64).block_inverse_atol == 1e-10
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float32).block_inverse_atol == 2e-5


class _H0Consumer(torch.nn.Module):
    h0_node_key = "node_h0"
    h0_edge_key = "edge_h0"
    merge_mode = "replace"


class _TimeConsumer(torch.nn.Module):
    use_flow_time_embedding = True
    flow_time_condition_edges = True

    def __init__(self, allow_missing=False):
        super().__init__()
        self.flow_time_conditioner = SimpleNamespace(
            flow_time_keys=(), flow_time_key="flow_time", allow_missing_time=allow_missing
        )


class _GuardModel(torch.nn.Module):
    def __init__(self, *, add_h0=False, allow_missing=False):
        super().__init__()
        self.block_native_add_h0 = add_h0
        self.h0 = _H0Consumer()
        self.time = _TimeConsumer(allow_missing=allow_missing)


def test_model_gate_requires_no_add_h0_and_fail_closed_node_edge_time_embedding():
    idp, _, _, _ = _case()
    flow = _flow(idp)
    assert assert_flow_h0_keys_reach_model(flow, _GuardModel()) is None
    with pytest.raises(ValueError, match="prediction.add_h0=false"):
        assert_flow_h0_keys_reach_model(flow, _GuardModel(add_h0=True))
    with pytest.raises(ValueError, match="flow_time_allow_missing=false"):
        assert_flow_h0_keys_reach_model(flow, _GuardModel(allow_missing=True))


def _ref_for(flow, data, codec, h0, endpoint, endpoint_node_rme, endpoint_edge_rme):
    ref = _fresh(data)
    ref["node_features"] = endpoint_node_rme
    ref["edge_features"] = endpoint_edge_rme
    ref[flow.node_block_target_key] = endpoint.node_blocks
    ref[flow.edge_block_target_key] = endpoint.edge_blocks
    ref[flow.node_block_shape_key] = endpoint.node_shapes
    ref[flow.edge_block_shape_key] = endpoint.edge_shapes
    return ref


def test_new_loss_requires_both_components_exact_shapes_and_physical_target():
    idp, data, codec, _ = _case()
    flow = _flow(idp)
    node_target = data["node_h0"] * 1.4
    edge_target = data["edge_h0"] * 1.4
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, None, endpoint, node_target, edge_target)
    batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))
    pred = batch.copy()
    pred[flow.node_output_key] = endpoint.node_blocks
    pred[flow.edge_output_key] = endpoint.edge_blocks
    loss, state = flow.loss(pred, ref, ctx)
    assert loss.item() <= ATOL
    assert state["_compatible_clean_stats"]["onsite_count"].item() == 7
    assert state["_compatible_clean_stats"]["hopping_count"].item() == 3

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


def test_residual_sample_scores_against_full_target_after_one_h0_add():
    idp, data, codec, h0 = _case()
    flow = _flow(idp, "residual_dh")
    delta_node = data["node_h0"] * 0.2
    delta_edge = data["edge_h0"] * 0.2
    delta = codec.rme_to_blocks(data, delta_node, delta_edge, project=True)
    full_node = data["node_h0"] + delta_node
    full_edge = data["edge_h0"] + delta_edge
    ref = _ref_for(flow, data, codec, h0, delta, full_node, full_edge)
    _, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.0]))
    sampled = flow.sample(EndpointSequence([delta]), _fresh(data), num_steps=1)
    loss, _ = flow.loss(sampled, ref, ctx)
    assert loss.item() <= ATOL


def test_argcheck_cross_contract_rejects_add_h0_and_missing_time_conditioning():
    base = {
        "train_options": {"flow_options": {
            "enabled": True, "mode": "residual", "prior": "zero",
            "output_space": "ao_block_ode", "block_ode": True,
            "target_semantics": "absolute_full_h", "prediction_add_h0": False,
            "time_conditioning_required": True, "block_inverse_mode": "strict",
            "validation_ode_steps": [1, 3],
        }},
        "model_options": {
            "embedding": {
                "use_flow_time_embedding": True, "flow_time_condition_edges": True,
                "flow_time_allow_missing": False, "flow_time_key": "flow_time",
                "h0_merge_mode": "replace", "use_h0_node_init": True,
                "use_h0_edge_init": True,
            },
            "prediction": {"add_h0": False},
        },
        "data_options": {"train": {
            "get_Hamiltonian": True, "residual_hamiltonian": False,
        }},
    }
    assert validate_block_ode_contract(base) is None
    bad_add = {**base, "model_options": {**base["model_options"], "prediction": {"add_h0": True}}}
    with pytest.raises(ValueError, match="prediction.add_h0=false"):
        validate_block_ode_contract(bad_add)
    bad_time = {**base, "model_options": {**base["model_options"], "embedding": {
        **base["model_options"]["embedding"], "flow_time_allow_missing": True,
    }}}
    with pytest.raises(ValueError, match="flow_time_allow_missing=false"):
        validate_block_ode_contract(bad_time)
