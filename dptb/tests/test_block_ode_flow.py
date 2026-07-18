from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    canonical_block_tensors_to_feature_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM
from dptb.nnops.multi_trainer import MultiTrainer


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
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        "atom_types": atom_types,
        _keys.EDGE_TYPE_KEY: edge_types,
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
    data[_keys.NODE_H0_BLOCKS_KEY] = h0_blocks.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0_blocks.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0_blocks.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0_blocks.edge_shapes.clone()
    data["node_features"] = node_h0.clone()
    data["edge_features"] = edge_h0.clone()
    return idp, data, codec, h0_blocks


def _pd_legacy_product_h0_case():
    """Return p/d product features plus authoritative physical H0 blocks."""
    idp = OrbitalMapper(
        {"C": ["2p"], "Si": ["3d"]}, method="e3tb", device="cpu"
    )
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    atom_types = torch.tensor(
        [idp.chemical_symbol_to_type["C"], idp.chemical_symbol_to_type["Si"]]
    )
    edge_types = torch.tensor(
        [idp.bond_to_type["C-Si"], idp.bond_to_type["Si-C"]]
    )
    data = {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.2, 0.1]], dtype=torch.float64
        ),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float64).unsqueeze(0) * 6.0,
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[1, 0, 0], [-1, 0, 0]], dtype=torch.long
        ),
        _keys.ATOM_TYPE_KEY: atom_types,
        _keys.EDGE_TYPE_KEY: edge_types,
    }
    generator = torch.Generator().manual_seed(20260719)
    raw = BlockTensorResult(
        node_blocks=torch.randn(2, 5, 5, dtype=torch.float64, generator=generator),
        edge_blocks=torch.randn(2, 5, 5, dtype=torch.float64, generator=generator),
        node_shapes=torch.tensor([[3, 3], [5, 5]], dtype=torch.long),
        edge_shapes=torch.tensor([[3, 5], [5, 3]], dtype=torch.long),
    )
    h0_blocks = project_block_state(data, idp, raw)
    codec = BlockStateCodec(idp, dtype=torch.float64)
    canonical_node_h0, canonical_edge_h0 = codec.blocks_to_rme(data, h0_blocks)
    gathered = canonical_block_tensors_to_feature_tensors(
        data,
        idp,
        node_blocks=h0_blocks.node_blocks,
        edge_blocks=h0_blocks.edge_blocks,
        node_shapes=h0_blocks.node_shapes,
        edge_shapes=h0_blocks.edge_shapes,
        mode="strict",
        atol=ATOL,
    )
    assert gathered.node_features is not None
    assert gathered.edge_features is not None
    # These are historical uncoupled/product coordinates.  The p/d CG change
    # of basis is nontrivial, so reading them as coupled RME changes physical B0.
    assert (gathered.node_features - canonical_node_h0).abs().max().item() > 1.0e-3
    assert (gathered.edge_features - canonical_edge_h0).abs().max().item() > 1.0e-3
    data[_keys.NODE_H0_KEY] = gathered.node_features.clone()
    data[_keys.EDGE_H0_KEY] = gathered.edge_features.clone()
    data[_keys.NODE_FEATURES_KEY] = gathered.node_features.clone()
    data[_keys.EDGE_FEATURES_KEY] = gathered.edge_features.clone()
    data[_keys.NODE_H0_BLOCKS_KEY] = h0_blocks.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0_blocks.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0_blocks.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0_blocks.edge_shapes.clone()
    return idp, data, codec, h0_blocks, canonical_node_h0, canonical_edge_h0


def _fresh(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def _flow(idp, semantics="absolute_full_h", **updates):
    target_fields = (
        {
            "node_block_target_key": "node_full_hamil_target_blocks",
            "edge_block_target_key": "edge_full_hamil_target_blocks",
            "node_block_shape_key": "node_full_hamil_target_block_shape",
            "edge_block_shape_key": "edge_full_hamil_target_block_shape",
        }
        if semantics == "absolute_full_h"
        else {
            "node_block_target_key": "node_delta_hamil_blocks",
            "edge_block_target_key": "edge_delta_hamil_blocks",
            "node_block_shape_key": "node_delta_hamil_block_shape",
            "edge_block_shape_key": "edge_delta_hamil_block_shape",
        }
    )
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
        **target_fields,
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


class LinearRMEEndpoint(torch.nn.Module):
    """Input-sensitive endpoint for adapter versus one-step ODE comparison."""

    def __init__(self, codec: BlockStateCodec, gain: float):
        super().__init__()
        self.codec = codec
        self.gain = float(gain)
        self.inputs = []
        self.times = []

    def forward(self, data):
        node = data[_keys.NODE_H0_KEY]
        edge = data[_keys.EDGE_H0_KEY]
        self.inputs.append((node.clone(), edge.clone()))
        self.times.append(data["flow_time"].clone())
        endpoint = self.codec.rme_to_blocks(
            data, node * self.gain, edge * self.gain, project=True
        )
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = endpoint.node_blocks
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = endpoint.edge_blocks
        return out


class MetadataConsumingEndpointSequence(EndpointSequence):
    """Mimic LemMoEV3H0 consuming its precomputed cutoff sidecar."""

    metadata_keys = (
        _keys.LEM_ACTIVE_EDGES_KEY,
        _keys.LEM_CUTOFF_COEFFS_KEY,
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY,
    )

    def __init__(self, endpoints):
        super().__init__(endpoints)
        self.seen_metadata = []

    def forward(self, data):
        snapshot = {key: data.pop(key) for key in self.metadata_keys}
        self.seen_metadata.append(snapshot)
        return super().forward(data)


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


def test_multitrainer_residual_euler_validation_scores_full_sample_via_flow():
    idp, data, codec, _ = _case()
    delta = _scaled_endpoint(
        codec,
        data,
        data[_keys.NODE_H0_KEY],
        data[_keys.EDGE_H0_KEY],
        scale=0.25,
    )
    batch = _fresh(data)
    batch[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY] = delta.node_blocks.clone()
    batch[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY] = delta.edge_blocks.clone()
    batch[_keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = delta.node_shapes.clone()
    batch[_keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = delta.edge_shapes.clone()

    class WrongSpaceCriterion:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("Residual full-H samples must not be scored as dH")

    trainer = object.__new__(MultiTrainer)
    trainer.iter = 1
    trainer.dtype = torch.float64
    trainer.device = torch.device("cpu")
    trainer._tagger = SimpleNamespace(
        tag=lambda *_args, **_kwargs: nullcontext()
    )
    trainer.model = EndpointSequence([delta])
    trainer.flow_cfm = _flow(idp, semantics="residual_dh")
    trainer._prepare_expert_masks = lambda batch_dict, *_args: (
        torch.ones(batch_dict[_keys.EDGE_H0_KEY].shape[0], dtype=torch.bool),
        torch.ones(batch_dict[_keys.NODE_H0_KEY].shape[0], dtype=torch.bool),
    )

    payload = trainer._build_validation_euler_payload(
        batch_dict=batch,
        batch_info={},
        criterion=WrongSpaceCriterion(),
        expert_idx=0,
        range_dis=(0.0, 1.0),
        num_steps=1,
    )

    assert payload["loss"].item() <= ATOL**2
    assert payload["onsite_l1_sum"].item() <= ATOL
    assert payload["hopping_l1_sum"].item() <= ATOL


def test_block_ode_reinjects_lem_sidecar_each_step_without_leaking_it():
    idp, data, _, h0 = _case()
    metadata = {
        _keys.LEM_ACTIVE_EDGES_KEY: torch.tensor([0, 1], dtype=torch.long),
        _keys.LEM_CUTOFF_COEFFS_KEY: torch.tensor([0.75, 0.5], dtype=torch.float64),
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([2], dtype=torch.long),
    }
    rollout_input = {**_fresh(data), **metadata}
    model = MetadataConsumingEndpointSequence([h0, h0])

    result = _flow(idp).sample(model, rollout_input, num_steps=2)

    assert len(model.seen_metadata) == 2
    for step_metadata in model.seen_metadata:
        for key, expected in metadata.items():
            assert torch.equal(step_metadata[key], expected)
    for key in metadata:
        assert model.seen_metadata[0][key].data_ptr() != model.seen_metadata[1][key].data_ptr()
        assert key not in result


def test_block_ode_sample_ignores_model_topology_and_isolates_input_tensors():
    idp, data, _, h0 = _case()

    class TopologyMutatingModel(EndpointSequence):
        def forward(self, batch):
            batch[_keys.EDGE_INDEX_KEY].copy_(
                batch[_keys.EDGE_INDEX_KEY].flip(1)
            )
            out = super().forward(batch)
            out[_keys.EDGE_CELL_SHIFT_KEY] = torch.ones_like(
                out[_keys.EDGE_CELL_SHIFT_KEY]
            )
            return out

    original_edge_index = data[_keys.EDGE_INDEX_KEY].clone()
    original_shift = data[_keys.EDGE_CELL_SHIFT_KEY].clone()
    result = _flow(idp).sample(
        TopologyMutatingModel([h0]), _fresh(data), num_steps=1
    )

    torch.testing.assert_close(result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY], h0.edge_blocks)
    assert torch.equal(result[_keys.EDGE_INDEX_KEY], original_edge_index)
    assert torch.equal(result[_keys.EDGE_CELL_SHIFT_KEY], original_shift)
    assert torch.equal(data[_keys.EDGE_INDEX_KEY], original_edge_index)


def test_n1_is_tolerance_equivalent_to_adapter_for_input_sensitive_endpoint():
    idp, data, codec, _ = _case()
    block_model = LinearRMEEndpoint(codec, gain=1.7)
    adapter_model = LinearRMEEndpoint(codec, gain=1.7)
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
    torch.testing.assert_close(
        block_model.inputs[0][0], adapter_model.inputs[0][0], rtol=0.0, atol=ATOL
    )
    torch.testing.assert_close(
        block_model.inputs[0][1], adapter_model.inputs[0][1], rtol=0.0, atol=ATOL
    )
    torch.testing.assert_close(
        block_result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        adapter_result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        rtol=0.0,
        atol=ATOL,
    )
    torch.testing.assert_close(
        block_result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
        adapter_result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
        rtol=0.0,
        atol=ATOL,
    )
    # Bitwise equality or inequality is deliberately not asserted: the public
    # contract is tolerance-level equivalence after the codec/projector loop.


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
        expected_node, expected_edge = codec.blocks_to_rme(data, current)
        torch.testing.assert_close(model.inputs[step][0], expected_node, rtol=0, atol=ATOL)
        torch.testing.assert_close(model.inputs[step][1], expected_edge, rtol=0, atol=ATOL)
        _assert_state_invariants(data, current)
        endpoint = project_block_state(data, idp, raw)
        alpha = (1.0 / num_steps) / (1.0 - step / num_steps)
        current = _blend(data, idp, current, endpoint, alpha)
        _assert_state_invariants(data, current)
    assert (result["node_hamil_blocks"] - current.node_blocks).abs().max() <= ATOL
    assert (result["edge_hamil_blocks"] - current.edge_blocks).abs().max() <= ATOL


@pytest.mark.parametrize(
    "missing_key",
    [_keys.NODE_PRED_HAMIL_BLOCKS_KEY, _keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
)
def test_block_ode_multistep_requires_fresh_endpoint_outputs(missing_key):
    idp, data, codec, _ = _case()
    endpoint = _scaled_endpoint(
        codec, data, data[_keys.NODE_H0_KEY], data[_keys.EDGE_H0_KEY], 1.1
    )

    class FirstStepCompleteSecondStepIncomplete(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            out = batch.copy()
            outputs = {
                _keys.NODE_PRED_HAMIL_BLOCKS_KEY: endpoint.node_blocks,
                _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: endpoint.edge_blocks,
            }
            for key, value in outputs.items():
                if self.calls == 1 or key != missing_key:
                    out[key] = value
            return out

    with pytest.raises(ValueError, match=rf"step 2.*{missing_key}"):
        _flow(idp).sample(
            FirstStepCompleteSecondStepIncomplete(),
            _fresh(data),
            num_steps=2,
        )


@pytest.mark.parametrize(
    ("node_values", "edge_value", "expected"),
    [
        ([1.0, 1.0], 3.0, 1.790760441090),
        ([0.0, 0.0], 0.0, 0.0),
    ],
)
def test_block_ode_global_l1_rmse_frozen_golden(
    node_values, edge_value, expected
):
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
        torch.zeros((2, 1, 1), dtype=torch.float64),
        torch.zeros((2, 1, 1), dtype=torch.float64),
        shapes,
        shapes.clone(),
    )
    codec = BlockStateCodec(idp, dtype=torch.float64)
    node_h0, edge_h0 = codec.blocks_to_rme(data, h0)
    data.update(
        {
            _keys.NODE_H0_KEY: node_h0,
            _keys.EDGE_H0_KEY: edge_h0,
            _keys.NODE_FEATURES_KEY: node_h0.clone(),
            _keys.EDGE_FEATURES_KEY: edge_h0.clone(),
            _keys.NODE_H0_BLOCKS_KEY: h0.node_blocks,
            _keys.EDGE_H0_BLOCKS_KEY: h0.edge_blocks,
            _keys.NODE_H0_BLOCK_SHAPE_KEY: h0.node_shapes,
            _keys.EDGE_H0_BLOCK_SHAPE_KEY: h0.edge_shapes,
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
    flow = _flow(
        idp,
        loss_type="l1_rmse",
        component_reduction="global_elements",
        omit_time_scaling=True,
    )
    prepared, prepared_ref, ctx = flow.prepare_batch(
        _fresh(data), ref, t=torch.tensor([0.0], dtype=torch.float64)
    )
    pred = dict(prepared)
    pred.update(
        {
            _keys.NODE_PRED_HAMIL_BLOCKS_KEY: torch.tensor(
                node_values, dtype=torch.float64
            ).reshape(2, 1, 1),
            _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: torch.full(
                (2, 1, 1), edge_value, dtype=torch.float64
            ),
        }
    )

    actual, _ = flow.loss(pred, prepared_ref, ctx)

    assert actual.item() == pytest.approx(expected, abs=1.0e-12)


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
        "node_block_target_key": "node_full_hamil_target_blocks",
        "edge_block_target_key": "edge_full_hamil_target_blocks",
        "node_block_shape_key": "node_full_hamil_target_block_shape",
        "edge_block_shape_key": "edge_full_hamil_target_block_shape",
        "validation_ode_steps": [1, 3],
    }
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float64).block_inverse_atol == 1e-10
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float32).block_inverse_atol == 2e-5


def _ref_for(flow, data, codec, h0, endpoint, endpoint_node_rme, endpoint_edge_rme):
    ref = _fresh(data)
    ref["node_features"] = endpoint_node_rme
    ref["edge_features"] = endpoint_edge_rme
    ref[flow.node_block_target_key] = endpoint.node_blocks
    ref[flow.edge_block_target_key] = endpoint.edge_blocks
    ref[flow.node_block_shape_key] = endpoint.node_shapes
    ref[flow.edge_block_shape_key] = endpoint.edge_shapes
    return ref


def test_prepare_is_block_authoritative_over_legacy_product_sidechannels():
    (
        idp,
        data,
        codec,
        h0,
        canonical_node_h0,
        canonical_edge_h0,
    ) = _pd_legacy_product_h0_case()
    flow = _flow(idp)
    node_target = canonical_node_h0 * 1.4
    edge_target = canonical_edge_h0 * 1.4
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, h0, endpoint, node_target, edge_target)

    # Keep the genuine p/d AO-product legacy side channel in `data`; make the
    # independent main-target feature side channel stale as well.  Neither may
    # override the authoritative H0/endpoint blocks.
    stale_data = _fresh(data)
    ref["node_features"] = torch.full_like(ref["node_features"], 123.0)
    ref["edge_features"] = torch.full_like(ref["edge_features"], -456.0)
    batch, _, _ = flow.prepare_batch(
        stale_data, ref, t=torch.tensor([0.4], dtype=torch.float64)
    )
    expected_blocks = _blend(data, idp, h0, endpoint, 0.4)
    expected_node, expected_edge = codec.blocks_to_rme(data, expected_blocks)
    torch.testing.assert_close(batch["node_h0"], expected_node, rtol=0.0, atol=ATOL)
    torch.testing.assert_close(batch["edge_h0"], expected_edge, rtol=0.0, atol=ATOL)
    assert (stale_data["node_h0"] - canonical_node_h0).abs().max().item() > 1.0e-3
    assert (stale_data["edge_h0"] - canonical_edge_h0).abs().max().item() > 1.0e-3

    # A different but still physical endpoint block must also be accepted even
    # when the legacy main feature tensor still describes the old endpoint.
    stale_blocks = _ref_for(flow, data, codec, h0, endpoint, node_target, edge_target)
    stale_blocks[flow.node_block_target_key] = (
        stale_blocks[flow.node_block_target_key] * 1.1
    )
    changed, _, _ = flow.prepare_batch(
        _fresh(data), stale_blocks, t=torch.tensor([0.4], dtype=torch.float64)
    )
    changed_endpoint = BlockTensorResult(
        endpoint.node_blocks * 1.1,
        endpoint.edge_blocks,
        endpoint.node_shapes,
        endpoint.edge_shapes,
    )
    changed_expected = _blend(data, idp, h0, changed_endpoint, 0.4)
    changed_node, changed_edge = codec.blocks_to_rme(data, changed_expected)
    torch.testing.assert_close(changed["node_h0"], changed_node, rtol=0.0, atol=ATOL)
    torch.testing.assert_close(changed["edge_h0"], changed_edge, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("topology_key", [_keys.EDGE_INDEX_KEY, _keys.EDGE_CELL_SHIFT_KEY])
def test_prepare_rejects_data_ref_topology_row_mismatch(topology_key):
    idp, data, codec, h0 = _case()
    flow = _flow(idp)
    node_target = data["node_h0"] * 1.4
    edge_target = data["edge_h0"] * 1.4
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, h0, endpoint, node_target, edge_target)
    ref[topology_key] = ref[topology_key].flip(1 if topology_key == _keys.EDGE_INDEX_KEY else 0)

    with pytest.raises(ValueError, match=rf"topology mismatch at key '{topology_key}'"):
        flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_prepare_rejects_nonfinite_block_target_before_projection(bad_value):
    idp, data, codec, h0 = _case()
    flow = _flow(idp)
    node_target = data["node_h0"] * 1.4
    edge_target = data["edge_h0"] * 1.4
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, h0, endpoint, node_target, edge_target)
    ref[flow.node_block_target_key] = ref[flow.node_block_target_key].clone()
    ref[flow.node_block_target_key][0, 0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="absolute_full_h endpoint node blocks contains NaN or Inf",
    ):
        flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_block_ode_sample_rejects_nonfinite_h0_blocks_before_projection(bad_value):
    idp, data, _, h0 = _case()
    data = _fresh(data)
    data[_keys.NODE_H0_BLOCKS_KEY][0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="physical H0 node blocks contains NaN or Inf"):
        _flow(idp).sample(EndpointSequence([h0]), data, num_steps=1)


def test_block_ode_sample_fails_closed_without_h0_block_sidechannel():
    idp, data, _, h0 = _case()
    data = _fresh(data)
    data.pop(_keys.EDGE_H0_BLOCKS_KEY)

    with pytest.raises(KeyError, match="physical H0 blocks and shapes"):
        _flow(idp).sample(EndpointSequence([h0]), data, num_steps=1)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_block_ode_sample_rejects_nonfinite_endpoint_before_cast(bad_value):
    idp, data, _, h0 = _case()
    bad_node = h0.node_blocks.clone()
    bad_node[0, 0, 0] = bad_value
    endpoint = BlockTensorResult(
        bad_node,
        h0.edge_blocks,
        h0.node_shapes,
        h0.edge_shapes,
    )

    with pytest.raises(ValueError, match="node endpoint prediction contains NaN or Inf"):
        _flow(idp).sample(EndpointSequence([endpoint]), _fresh(data), num_steps=1)


def test_block_ode_sample_rejects_complex_endpoint_before_real_dtype_cast():
    idp, data, _, h0 = _case()
    endpoint = BlockTensorResult(
        h0.node_blocks.to(torch.complex128) + 1.0j,
        h0.edge_blocks.to(torch.complex128),
        h0.node_shapes,
        h0.edge_shapes,
    )

    with pytest.raises(ValueError, match="node endpoint prediction must be real"):
        _flow(idp).sample(EndpointSequence([endpoint]), _fresh(data), num_steps=1)


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

    polluted_topology = pred.copy()
    polluted_topology[_keys.EDGE_CELL_SHIFT_KEY] = torch.ones_like(
        pred[_keys.EDGE_CELL_SHIFT_KEY]
    )
    polluted_loss, _ = flow.loss(polluted_topology, ref, ctx)
    assert polluted_loss.item() <= ATOL

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


def test_residual_training_and_sample_scoring_have_flow_owned_semantics():
    idp, data, codec, h0 = _case()
    flow = _flow(idp, "residual_dh")
    delta_node = data["node_h0"] * 0.2
    delta_edge = data["edge_h0"] * 0.2
    delta = codec.rme_to_blocks(data, delta_node, delta_edge, project=True)
    # The authoritative residual block side channel is dH; legacy feature
    # coordinates are irrelevant to block-ODE scoring.
    # The sampled rollout is converted back to Full H only for endpoint scoring.
    record = _ref_for(flow, data, codec, h0, delta, delta_node, delta_edge)
    sample_record = _fresh(record)
    # Match Trainer's shallow input/reference copies so this also guards the
    # original model-to-label storage alias.
    prepared, ref, ctx = flow.prepare_batch(record, record.copy(), t=torch.tensor([0.0]))
    authority_keys = (
        flow.node_block_target_key,
        flow.edge_block_target_key,
        flow.node_block_shape_key,
        flow.edge_block_shape_key,
        flow.node_h0_block_key,
        flow.edge_h0_block_key,
        flow.node_h0_block_shape_key,
        flow.edge_h0_block_shape_key,
    )
    assert all(key in ref for key in authority_keys)
    assert all(key not in prepared for key in authority_keys)

    prediction = prepared.copy()
    prediction[flow.node_output_key] = delta.node_blocks
    prediction[flow.edge_output_key] = delta.edge_blocks
    # A model-owned/stale key must be inert: training predictions always use
    # the configured endpoint semantics.
    prediction["_block_ode_sample_is_full"] = torch.tensor(True)
    training_loss, _ = flow.loss(prediction, ref, ctx)
    assert training_loss.item() <= ATOL

    class AuthorityRejectingEndpoint(EndpointSequence):
        def forward(self, batch):
            assert all(key not in batch for key in authority_keys)
            return super().forward(batch)

    sampled = flow.sample(AuthorityRejectingEndpoint([delta]), sample_record, num_steps=1)
    assert all(key not in sampled for key in authority_keys)
    assert "_block_ode_sample_is_full" not in sampled
    sample_loss, _ = flow.loss_on_sample(sampled, ref, ctx)
    assert sample_loss.item() <= ATOL
