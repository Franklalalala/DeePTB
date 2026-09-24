"""Block-ODE sampler and training bridge: rollouts vs manual endpoint blends, H0 bookkeeping, input
isolation, seeded priors, image certification cadence and fail-closed inputs."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces import blockwise_tensor as blockwise_module
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_flow_codec import project_block_state
from dptb.nnops.flow import HamiltonianCFM
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.tests._requires import requires_multi_gpu
from dptb.tests.block_ode_fixtures import (
    DELTA_LABEL_KEYS,
    FP64_ATOL,
    _EndpointSequence,
    _assert_state_invariants,
    _b_flow,
    _b_record,
    _blend,
    _case,
    _flow,
    _fresh,
    _mapper,
    _pd_legacy_product_h0_case,
    _ref_for,
    _scaled_endpoint,
    _uureal_flow,
    _uureal_mapper,
    _uureal_record,
)

NODE_PRED = _keys.NODE_PRED_HAMIL_BLOCKS_KEY
EDGE_PRED = _keys.EDGE_PRED_HAMIL_BLOCKS_KEY
T04 = torch.tensor([0.4], dtype=torch.float64)
_TE = dict(prior="projected_te", te_prior_mode="irrep", node_sigma=0.25, edge_sigma=0.25)


class _ConstantEndpoint(torch.nn.Module):
    """Return the same endpoint blocks every step; optionally stop emitting one key after step 1."""

    def __init__(self, node, edge, omit_after_first=None):
        super().__init__()
        self.outputs = {NODE_PRED: node, EDGE_PRED: edge}
        self.omit = omit_after_first
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        out = batch.copy()
        for key, value in self.outputs.items():
            if self.calls == 1 or key != self.omit:
                out[key] = value.clone()
        return out


class _LinearRMEEndpoint(torch.nn.Module):
    """Input-sensitive endpoint: gain * (H0 RME input) expanded to projected blocks."""

    def __init__(self, codec, gain):
        super().__init__()
        self.codec = codec
        self.gain = float(gain)
        self.inputs = []
        self.times = []

    def forward(self, data):
        node, edge = data[_keys.NODE_H0_KEY], data[_keys.EDGE_H0_KEY]
        self.inputs.append((node.clone(), edge.clone()))
        self.times.append(data["flow_time"].clone())
        endpoint = self.codec.rme_to_blocks(data, node * self.gain, edge * self.gain, project=True)
        out = data.copy()
        out[NODE_PRED] = endpoint.node_blocks
        out[EDGE_PRED] = endpoint.edge_blocks
        return out


_ROUTES = ("full_h", "uureal", "residual")


def _route_case(route):
    """(flow, record, endpoint node blocks, endpoint edge blocks, final blocks of a rollout on that endpoint)."""
    if route == "full_h":
        idp, data, codec, _ = _case()
        endpoint = _scaled_endpoint(codec, data, data[_keys.NODE_H0_KEY], data[_keys.EDGE_H0_KEY], 1.1)
        final = (endpoint.node_blocks, endpoint.edge_blocks)
        return _flow(idp), data, endpoint.node_blocks, endpoint.edge_blocks, final
    if route == "uureal":
        mapper = _uureal_mapper()
        data = _uureal_record(mapper)
        node, edge = data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY], data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY]
        return _uureal_flow(mapper), data, node, edge, (node, edge)
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    final = (h0.node_blocks + d1.node_blocks, h0.edge_blocks + d1.edge_blocks)
    return _b_flow(mapper), data, d1.node_blocks, d1.edge_blocks, final


def _labelled(data, flow, endpoint, scale):
    """Reference record whose endpoint blocks are ``endpoint`` and legacy features ``scale * H0``."""
    return _ref_for(flow, data, endpoint, data[_keys.NODE_H0_KEY] * scale, data[_keys.EDGE_H0_KEY] * scale)


# ---------------------------------------------------------------------------
# Rollout arithmetic and H0 bookkeeping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_steps", [2, 3])
@pytest.mark.parametrize("prior", ["zero", "projected_te"])
def test_multistep_rollouts_match_manual_endpoint_blends(num_steps, prior):
    """Every step sees blocks_to_rme(current) and the state advances by the endpoint blend
    alpha = (1/N) / (1 - k/N), staying in the invariant block subspace throughout."""
    idp, data, codec, h0 = _case()
    flow = _flow(idp, **(_TE if prior == "projected_te" else {}))
    prior_seed = 20260719 if prior == "projected_te" else None
    current, _, _ = flow._block_initial_state(_fresh(data), h0, prior_seed=prior_seed)
    generator = torch.Generator().manual_seed(100 + num_steps)
    raw_endpoints = [
        BlockTensorResult(
            torch.randn(2, 3, 3, dtype=torch.float64, generator=generator),
            torch.randn(2, 3, 3, dtype=torch.float64, generator=generator),
            current.node_shapes,
            current.edge_shapes,
        )
        for _ in range(num_steps)
    ]
    model = _EndpointSequence(raw_endpoints)
    result = flow.sample(model, _fresh(data), num_steps=num_steps, prior_seed=prior_seed)
    for step, raw in enumerate(raw_endpoints):
        expected_node, expected_edge = codec.blocks_to_rme(data, current)
        torch.testing.assert_close(model.inputs[step][0], expected_node, rtol=0, atol=FP64_ATOL)
        torch.testing.assert_close(model.inputs[step][1], expected_edge, rtol=0, atol=FP64_ATOL)
        alpha = (1.0 / num_steps) / (1.0 - step / num_steps)
        current = _blend(data, idp, current, project_block_state(data, idp, raw), alpha)
        _assert_state_invariants(current)
    assert (result[NODE_PRED] - current.node_blocks).abs().max() <= FP64_ATOL
    assert (result[EDGE_PRED] - current.edge_blocks).abs().max() <= FP64_ATOL


def test_one_step_rollout_matches_the_block_adapter_for_an_input_sensitive_endpoint():
    idp, data, codec, _ = _case()
    block_model = _LinearRMEEndpoint(codec, gain=1.7)
    adapter_model = _LinearRMEEndpoint(codec, gain=1.7)
    block_result = _flow(idp).sample(block_model, _fresh(data), num_steps=1)
    adapter = HamiltonianCFM(
        {"enabled": True, "mode": "residual", "prior": "zero", "output_space": "ao_block", "validation_ode_steps": [1]},
        idp=idp,
        dtype=torch.float64,
    )
    adapter_result = adapter.sample(adapter_model, _fresh(data), num_steps=1)
    assert torch.equal(block_model.times[0], adapter_model.times[0])
    for got, expected in (
        (block_model.inputs[0][0], adapter_model.inputs[0][0]),
        (block_model.inputs[0][1], adapter_model.inputs[0][1]),
        (block_result[NODE_PRED], adapter_result[NODE_PRED]),
        (block_result[EDGE_PRED], adapter_result[EDGE_PRED]),
    ):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=FP64_ATOL)


def test_full_h_endpoint_is_used_as_is_and_residual_endpoint_gets_h0_added_once():
    idp, data, codec, h0 = _case()
    delta = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 0.25)
    absolute = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.25)
    full = _flow(idp, "absolute_full_h").sample(_EndpointSequence([absolute]), _fresh(data), num_steps=1)
    residual = _flow(idp, "residual_dh").sample(_EndpointSequence([delta]), _fresh(data), num_steps=1)
    expected = project_block_state(
        data,
        idp,
        BlockTensorResult(
            h0.node_blocks + delta.node_blocks, h0.edge_blocks + delta.edge_blocks, h0.node_shapes, h0.edge_shapes
        ),
    )
    assert (full[NODE_PRED] - absolute.node_blocks).abs().max() <= FP64_ATOL
    assert (full[NODE_PRED] - (absolute.node_blocks + h0.node_blocks)).abs().max() > 1e-3
    assert (residual[NODE_PRED] - expected.node_blocks).abs().max() <= FP64_ATOL
    assert (residual[EDGE_PRED] - expected.edge_blocks).abs().max() <= FP64_ATOL


def test_sampler_conditions_on_physical_h0_blocks_not_legacy_product_features():
    idp, data, codec, _, node_h0, edge_h0 = _pd_legacy_product_h0_case()
    endpoint = _scaled_endpoint(codec, data, node_h0, edge_h0, 1.2)
    model = _EndpointSequence([endpoint])
    result = _flow(idp).sample(model, _fresh(data), num_steps=1)
    for got, expected in (
        (model.inputs[0][0], node_h0),
        (model.inputs[0][1], edge_h0),
        (result[NODE_PRED], endpoint.node_blocks),
        (result[EDGE_PRED], endpoint.edge_blocks),
    ):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=FP64_ATOL)


def test_float32_rollout_grid_never_evaluates_t1_and_lands_on_the_endpoint():
    idp, data64, codec64, _ = _case()
    data = {
        key: value.to(torch.float32) if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in _fresh(data64).items()
    }
    endpoint64 = _scaled_endpoint(codec64, data64, data64["node_h0"], data64["edge_h0"], 1.3)
    endpoint = BlockTensorResult(
        endpoint64.node_blocks.float(), endpoint64.edge_blocks.float(), endpoint64.node_shapes, endpoint64.edge_shapes
    )
    model = _EndpointSequence([endpoint])
    flow = HamiltonianCFM(dict(_flow(idp).options, block_inverse_atol=2e-5), idp=idp, dtype=torch.float32)
    result = flow.sample(model, data, num_steps=3)
    times = torch.stack([value.reshape(()) for value in model.times])
    torch.testing.assert_close(
        times, torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0]), rtol=0.0, atol=torch.finfo(torch.float32).eps
    )
    assert bool((times < 1.0).all())
    train_t = flow._sample_t(num_graphs=4096, device=torch.device("cpu"), dtype=torch.float32)
    assert train_t.min().item() >= flow.t_min
    assert train_t.max().item() <= min(flow.t_max, 1.0 - flow.t_eps)
    torch.testing.assert_close(result[NODE_PRED], endpoint.node_blocks, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(result[EDGE_PRED], endpoint.edge_blocks, rtol=0.0, atol=2e-5)


# ---------------------------------------------------------------------------
# Model-owned state cannot leak between steps or into the caller
# ---------------------------------------------------------------------------
def test_rollout_reinjects_lem_sidecar_each_step_without_leaking_it():
    """Precomputed LEM cutoff metadata reaches every step as a fresh copy and is not returned."""
    idp, data, _, h0 = _case()
    metadata = {
        _keys.LEM_ACTIVE_EDGES_KEY: torch.tensor([0, 1], dtype=torch.long),
        _keys.LEM_CUTOFF_COEFFS_KEY: torch.tensor([0.75, 0.5], dtype=torch.float64),
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([2], dtype=torch.long),
    }

    class MetadataConsumingModel(_EndpointSequence):
        def __init__(self, endpoints):
            super().__init__(endpoints)
            self.seen = []

        def forward(self, batch):
            self.seen.append({key: batch.pop(key) for key in metadata})
            return super().forward(batch)

    model = MetadataConsumingModel([h0, h0])
    result = _flow(idp).sample(model, {**_fresh(data), **metadata}, num_steps=2)
    assert len(model.seen) == 2
    for key, expected in metadata.items():
        assert all(torch.equal(step[key], expected) for step in model.seen)
        assert model.seen[0][key].data_ptr() != model.seen[1][key].data_ptr()
        assert key not in result


def test_rollout_ignores_model_topology_writes_and_isolates_input_tensors():
    idp, data, _, h0 = _case()

    class TopologyMutatingModel(_EndpointSequence):
        def forward(self, batch):
            batch[_keys.EDGE_INDEX_KEY].copy_(batch[_keys.EDGE_INDEX_KEY].flip(1))
            out = super().forward(batch)
            out[_keys.EDGE_CELL_SHIFT_KEY] = torch.ones_like(out[_keys.EDGE_CELL_SHIFT_KEY])
            return out

    passed = _fresh(data)
    result = _flow(idp).sample(TopologyMutatingModel([h0]), passed, num_steps=1)
    torch.testing.assert_close(result[EDGE_PRED], h0.edge_blocks)
    for key in (_keys.EDGE_INDEX_KEY, _keys.EDGE_CELL_SHIFT_KEY):
        assert torch.equal(result[key], data[key])
        assert torch.equal(passed[key], data[key])


def test_next_step_h0_input_survives_in_place_feature_overwrite():
    """A model that overwrites node/edge features in place (as E3Hamiltonian does) cannot pollute
    the H0 conditioning of the following step."""
    idp, data, _, h0 = _case()

    class FeatureOverwritingModel(_EndpointSequence):
        def forward(self, batch):
            out = super().forward(batch)
            batch["node_features"].add_(1234.0)
            batch["edge_features"].sub_(567.0)
            return out

    model = FeatureOverwritingModel([h0, h0])
    _flow(idp).sample(model, _fresh(data), num_steps=2)
    torch.testing.assert_close(model.inputs[1][0], model.inputs[0][0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(model.inputs[1][1], model.inputs[0][1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("missing_key", [NODE_PRED, EDGE_PRED])
@pytest.mark.parametrize("route", _ROUTES)
def test_multistep_rollout_requires_fresh_endpoint_outputs(route, missing_key):
    """A model that stops emitting an endpoint key at step 2 fails closed instead of reusing step 1's."""
    flow, data, node, edge, _ = _route_case(route)
    with pytest.raises(ValueError, match=missing_key):
        flow.sample(_ConstantEndpoint(node, edge, omit_after_first=missing_key), _fresh(data), num_steps=2)


@pytest.mark.parametrize("route", ["uureal", "residual"])
def test_sampler_is_label_free_while_training_requires_labels(route):
    """Sampling a record without residual endpoint labels works; prepare_batch still demands them."""
    flow, data, node, edge, (final_node, final_edge) = _route_case(route)
    unlabeled = {key: value for key, value in data.items() if key not in DELTA_LABEL_KEYS}
    result = flow.sample(_ConstantEndpoint(node, edge), _fresh(unlabeled), num_steps=2)
    atol = 1e-12 if node.dtype == torch.float64 else 1e-6
    torch.testing.assert_close(result[NODE_PRED], final_node, rtol=0.0, atol=atol)
    torch.testing.assert_close(result[EDGE_PRED], final_edge, rtol=0.0, atol=atol)
    with pytest.raises(KeyError, match=_keys.NODE_DELTA_HAMIL_BLOCKS_KEY):
        flow.prepare_batch(_fresh(unlabeled), _fresh(unlabeled), t=torch.tensor([0.5], dtype=node.dtype))


def test_residual_target_authority_fields_stay_out_of_model_input_and_scoring_is_flow_owned():
    """Endpoint/H0 authority blocks live only in the reference; a model-written marker key cannot switch
    the endpoint semantics; the rollout is scored as full H."""
    idp, data, codec, _ = _case()
    flow = _flow(idp, "residual_dh")
    delta = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 0.2)
    record = _labelled(data, flow, delta, 0.2)
    sample_record = _fresh(record)
    # Trainer's shallow input/reference copies share storage; the model input must still be clean.
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
    prediction["_block_ode_sample_is_full"] = torch.tensor(True)
    assert flow.loss(prediction, ref, ctx)[0].item() <= FP64_ATOL

    class AuthorityRejectingEndpoint(_EndpointSequence):
        def forward(self, batch):
            assert all(key not in batch for key in authority_keys)
            return super().forward(batch)

    sampled = flow.sample(AuthorityRejectingEndpoint([delta]), sample_record, num_steps=1)
    assert all(key not in sampled for key in authority_keys)
    assert "_block_ode_sample_is_full" not in sampled
    assert flow.loss_on_sample(sampled, ref, ctx)[0].item() <= FP64_ATOL


# ---------------------------------------------------------------------------
# Seeded stochastic priors
# ---------------------------------------------------------------------------
def test_absolute_tied_irrep_start_is_independent_of_physical_h0():
    """An absolute-mode tied-irrep start is a nonzero draw that does not move with H0."""
    idp, data, _, h0 = _case()
    flow = _flow(
        idp,
        mode="absolute",
        prior="tied_irrep_gaussian",
        tied_irrep_mode="so3_tied",
        tied_irrep_irreps="3x0e + 2x1e + 1x2e",
        tied_irrep_sigma=1.0,
        tied_irrep_validation_seed=20260725,
    )
    scaled_h0 = BlockTensorResult(h0.node_blocks * 7.0, h0.edge_blocks * 7.0, h0.node_shapes, h0.edge_shapes)
    start_a, node_a, edge_a = flow._block_initial_state(data, h0, prior_seed=20260725)
    start_b, node_b, edge_b = flow._block_initial_state(data, scaled_h0, prior_seed=20260725)
    for a, b in ((start_a.node_blocks, start_b.node_blocks), (start_a.edge_blocks, start_b.edge_blocks),
                 (node_a, node_b), (edge_a, edge_b)):
        torch.testing.assert_close(a, b)
    for blocks in (start_a.node_blocks, start_a.edge_blocks):
        assert torch.isfinite(blocks).all() and blocks.abs().max().item() > 0.0


def test_projected_te_t0_state_is_endpoint_independent_and_matches_the_sampling_start():
    idp, data, codec, h0 = _case()
    flow = _flow(idp, te_prior_sigma=1.0, **_TE)
    zero_t = torch.zeros(1, dtype=torch.float64)
    starts = []
    for scale in (1.2, 1.8):
        record = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], scale), scale)
        # Label-bearing model input (Trainer's shallow-copy path) must not leak the endpoint into the prior.
        batch, _, ctx = flow.prepare_batch(_fresh(record), _fresh(record), t=zero_t, prior_seed=20260719)
        starts.append((batch[_keys.NODE_H0_KEY], batch[_keys.EDGE_H0_KEY]))
        assert torch.count_nonzero(ctx.node_prior) > 0 and torch.count_nonzero(ctx.edge_prior) > 0
    torch.testing.assert_close(starts[0], starts[1])

    model = _EndpointSequence([h0])
    flow.sample(model, _fresh(data), num_steps=1, prior_seed=20260719)
    torch.testing.assert_close(model.inputs[0], starts[0])
    other_seed = _EndpointSequence([h0])
    flow.sample(other_seed, _fresh(data), num_steps=1, prior_seed=20260720)
    assert not torch.equal(other_seed.inputs[0][0], model.inputs[0][0])
    assert not torch.equal(other_seed.inputs[0][1], model.inputs[0][1])


def test_projected_te_seed_replays_the_start_without_touching_the_global_rng():
    """A prior_seed gives the same start for N=1 and N=3 rollouts, seeded prepare_batch calls replay
    their time and state, the global RNG is not advanced, and the start lies in the codec image."""
    idp, data, codec, h0 = _case()
    flow = _flow(idp, **_TE)
    model_n1 = _EndpointSequence([h0])
    model_n3 = _EndpointSequence([h0])
    record = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.2), 1.2)
    seeds = dict(prior_seed=flow.validation_seed(0, "prior"), time_seed=flow.validation_seed(0, "time"))
    torch.manual_seed(91)
    expected_draw = torch.randn(4)
    torch.manual_seed(91)
    flow.sample(model_n1, _fresh(data), num_steps=1, prior_seed=20260719)
    flow.sample(model_n3, _fresh(data), num_steps=3, prior_seed=20260719)
    prepared = [flow.prepare_batch(_fresh(record), _fresh(record), **seeds)[0] for _ in range(2)]
    assert torch.equal(torch.randn(4), expected_draw)

    assert torch.equal(model_n1.inputs[0][0], model_n3.inputs[0][0])
    assert torch.equal(model_n1.inputs[0][1], model_n3.inputs[0][1])
    for key in (flow.flow_time_key, _keys.NODE_H0_KEY, _keys.EDGE_H0_KEY):
        assert torch.equal(prepared[0][key], prepared[1][key])
    assert flow.validation_seed(0, "prior") != flow.validation_seed(0, "time")
    assert flow.validation_seed(0, "prior") != flow.validation_seed(1, "prior")

    start = flow.block_codec.rme_to_blocks(data, *model_n1.inputs[0], project=True)
    _assert_state_invariants(start)
    node, edge = flow.block_codec.blocks_to_rme(data, start)
    torch.testing.assert_close(node, model_n1.inputs[0][0], rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(edge, model_n1.inputs[0][1], rtol=0.0, atol=FP64_ATOL)


@requires_multi_gpu
def test_projected_te_sampling_leaves_every_cuda_rng_untouched():
    idp, data, _, h0 = _case()
    flow = _flow(idp, **_TE)
    before = torch.cuda.get_rng_state_all()
    flow.sample(_EndpointSequence([h0]), _fresh(data), num_steps=1, prior_seed=20260719)
    after = torch.cuda.get_rng_state_all()
    assert len(before) == torch.cuda.device_count() >= 2
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_projected_te_max_validation_seed_gives_distinct_in_range_substreams():
    idp, _, _, _ = _case()
    flow = _flow(idp, prior="projected_te", te_prior_validation_seed=(1 << 64) - 1)
    seeds = {flow.validation_seed(batch, purpose) for batch in (0, 1) for purpose in ("prior", "time")}
    assert len(seeds) == 4
    assert all(0 <= seed <= (1 << 64) - 1 for seed in seeds)


# ---------------------------------------------------------------------------
# prepare_batch: authority, certification cadence, topology
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prior", ["zero", "projected_te"])
def test_prepare_is_block_authoritative_over_legacy_product_sidechannels(prior):
    """The bridge state is built from the physical H0/endpoint blocks: stale AO-product H0 features and
    stale endpoint features are ignored, and a changed endpoint block is honoured."""
    idp, data, codec, h0, node_h0, edge_h0 = _pd_legacy_product_h0_case()
    flow = _flow(idp, **(_TE if prior == "projected_te" else {}))
    seed = {"prior_seed": 20260719} if prior == "projected_te" else {}
    start, _, _ = flow._block_initial_state(_fresh(data), h0, prior_seed=seed.get("prior_seed"))
    endpoint = codec.rme_to_blocks(data, node_h0 * 1.4, edge_h0 * 1.4, project=True)
    for node_scale in (1.0, 1.1):
        target = BlockTensorResult(
            endpoint.node_blocks * node_scale, endpoint.edge_blocks, endpoint.node_shapes, endpoint.edge_shapes
        )
        ref = _ref_for(flow, data, target, torch.full_like(node_h0, 123.0), torch.full_like(edge_h0, -456.0))
        batch, _, _ = flow.prepare_batch(_fresh(data), ref, t=T04, **seed)
        expected_node, expected_edge = codec.blocks_to_rme(data, _blend(data, idp, start, target, 0.4))
        torch.testing.assert_close(batch[_keys.NODE_H0_KEY], expected_node, rtol=0.0, atol=FP64_ATOL)
        torch.testing.assert_close(batch[_keys.EDGE_H0_KEY], expected_edge, rtol=0.0, atol=FP64_ATOL)


def _corrupt_repack(original):
    def corrupt(*args, **kwargs):
        packed = original(*args, **kwargs)
        node = packed.node_blocks.clone()
        node[0, 0, 0] += 1.0e-3
        return BlockTensorResult(node, packed.edge_blocks, packed.node_shapes, packed.edge_shapes)
    return corrupt


@pytest.mark.parametrize(
    ("cadence", "certified"),
    [("always", [True, True, True]), ("first_batch", [True, False, False]), ("every_n(2)", [True, False, True])],
)
def test_certification_cadence_decides_which_batches_catch_a_corrupt_repack(cadence, certified, monkeypatch):
    """With the repack self-check corrupted at batch k, only batches the cadence certifies are rejected."""
    rejected = []
    for batch_index in range(len(certified)):
        idp, data, codec, _ = _case()
        flow = _flow(idp, strict_certification=cadence)
        ref = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.25), 1.25)
        for _ in range(batch_index):
            flow.prepare_batch(_fresh(data), _fresh(ref), t=T04)
        with monkeypatch.context() as patch:
            patch.setattr(
                blockwise_module,
                "feature_tensors_to_block_tensors",
                _corrupt_repack(blockwise_module.feature_tensors_to_block_tensors),
            )
            try:
                flow.prepare_batch(_fresh(data), _fresh(ref), t=T04)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
    assert rejected == certified


def test_first_batch_cadence_never_skips_the_topology_gate():
    idp, data, codec, _ = _case()
    flow = _flow(idp, strict_certification="first_batch")
    ref = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.25), 1.25)
    flow.prepare_batch(_fresh(data), _fresh(ref), t=T04)
    for batch in (data, ref):
        batch["edge_index"] = batch["edge_index"][:, :1]
        batch["edge_cell_shift"] = batch["edge_cell_shift"][:1]
        batch[_keys.EDGE_TYPE_KEY] = batch[_keys.EDGE_TYPE_KEY][:1]
    with pytest.raises(ValueError, match="reverse"):
        flow.prepare_batch(_fresh(data), _fresh(ref), t=T04)


@pytest.mark.parametrize("topology_key", [_keys.EDGE_INDEX_KEY, _keys.EDGE_CELL_SHIFT_KEY])
def test_prepare_rejects_data_ref_topology_mismatch(topology_key):
    idp, data, codec, _ = _case()
    flow = _flow(idp)
    ref = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.4), 1.4)
    ref[topology_key] = ref[topology_key].flip(1 if topology_key == _keys.EDGE_INDEX_KEY else 0)
    with pytest.raises(ValueError, match=topology_key):
        flow.prepare_batch(_fresh(data), ref, t=T04)


@pytest.mark.parametrize("missing_key", [
    _keys.NODE_H0_BLOCKS_KEY, _keys.EDGE_H0_BLOCKS_KEY, _keys.NODE_H0_BLOCK_SHAPE_KEY, _keys.EDGE_H0_BLOCK_SHAPE_KEY,
])
@pytest.mark.parametrize("entry", ["sample", "prepare_batch"])
def test_missing_physical_h0_blocks_or_shapes_fail_closed(entry, missing_key):
    idp, data, codec, h0 = _case()
    flow = _flow(idp)
    ref = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.1), 1.1)
    bad = _fresh(data)
    bad.pop(missing_key)
    with pytest.raises(KeyError, match=missing_key):
        if entry == "sample":
            flow.sample(_EndpointSequence([h0]), bad, num_steps=1)
        else:
            flow.prepare_batch(bad, ref, t=T04)


_NAN, _INF = float("nan"), float("inf")


@pytest.mark.parametrize(("where", "bad"), [
    ("endpoint_target", _NAN), ("endpoint_target", _INF),
    ("physical_h0", _NAN), ("physical_h0", _INF),
    ("endpoint_prediction", _NAN), ("endpoint_prediction", _INF), ("endpoint_prediction", 1j),
])
def test_nonfinite_or_complex_blocks_fail_closed(where, bad):
    idp, data, codec, h0 = _case()
    flow = _flow(idp)

    def corrupted(blocks):
        out = blocks.clone()
        out[0, 0, 0] = bad
        return out

    if where == "endpoint_target":
        ref = _labelled(data, flow, _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.4), 1.4)
        ref[flow.node_block_target_key] = corrupted(ref[flow.node_block_target_key])
        with pytest.raises(ValueError, match="endpoint node blocks"):
            flow.prepare_batch(_fresh(data), ref, t=T04)
    elif where == "physical_h0":
        record = _fresh(data)
        record[_keys.NODE_H0_BLOCKS_KEY] = corrupted(record[_keys.NODE_H0_BLOCKS_KEY])
        with pytest.raises(ValueError, match="H0 node blocks"):
            flow.sample(_EndpointSequence([h0]), record, num_steps=1)
    else:
        if isinstance(bad, complex):
            node, edge = h0.node_blocks.to(torch.complex128) + bad, h0.edge_blocks.to(torch.complex128)
        else:
            node, edge = corrupted(h0.node_blocks), h0.edge_blocks
        endpoint = BlockTensorResult(node, edge, h0.node_shapes, h0.edge_shapes)
        with pytest.raises(ValueError, match="endpoint prediction"):
            flow.sample(_EndpointSequence([endpoint]), _fresh(data), num_steps=1)


# ---------------------------------------------------------------------------
# MultiTrainer validation routes block-ODE scoring through the flow
# ---------------------------------------------------------------------------
def _validation_trainer(model, flow):
    """A MultiTrainer shell carrying only what the Euler validation payload reads."""
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 1
    trainer.dtype = torch.float64
    trainer.device = torch.device("cpu")
    trainer._tagger = SimpleNamespace(tag=lambda *_args, **_kwargs: nullcontext())
    trainer.model = model
    trainer.flow_cfm = flow
    flow.log_validation_flow_euler_loss = False
    trainer._prepare_expert_masks = lambda batch, *_args: (
        torch.ones(batch[_keys.EDGE_H0_KEY].shape[0], dtype=torch.bool),
        torch.ones(batch[_keys.NODE_H0_KEY].shape[0], dtype=torch.bool),
    )
    return trainer


@pytest.mark.parametrize("criterion_space", ["block", "rme"])
def test_multitrainer_euler_validation_scores_the_flow_rollout(criterion_space):
    """Residual-target validation scores the full-H rollout through the flow (never by calling the
    criterion); a criterion whose endpoint metric space is not 'block' is rejected."""
    idp, data, codec, _ = _case()
    delta = _scaled_endpoint(codec, data, data[_keys.NODE_H0_KEY], data[_keys.EDGE_H0_KEY], 0.25)
    batch = _fresh(data)
    for key, value in zip(DELTA_LABEL_KEYS, (delta.node_blocks, delta.edge_blocks, delta.node_shapes, delta.edge_shapes)):
        batch[key] = value.clone()

    class Criterion:
        endpoint_metric_space = criterion_space

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("block-ODE validation must score through the flow")

        @staticmethod
        def compatible_loss_from_stats(**stats):
            zero = stats["onsite_l1_sum"] * 0.0
            return zero, zero, zero

    model = _EndpointSequence([delta])
    trainer = _validation_trainer(model, _flow(idp, "residual_dh", **_TE))
    kwargs = dict(batch_dict=batch, batch_info={}, criterion=Criterion(), expert_idx=0, range_dis=(0.0, 1.0),
                  prior_seed=20260719)
    if criterion_space != "block":
        with pytest.raises(ValueError, match="endpoint_metric_space"):
            trainer._build_validation_euler_payload(num_steps=1, **kwargs)
        return
    for num_steps in (1, 3):
        payload = trainer._build_validation_euler_payload(num_steps=num_steps, **kwargs)
        assert payload["loss"].item() <= FP64_ATOL**2
        assert payload["onsite_l1_sum"].item() <= FP64_ATOL
        assert payload["hopping_l1_sum"].item() <= FP64_ATOL
    # N=1 and N=3 start from the same seeded prior state.
    assert torch.equal(model.inputs[0][0], model.inputs[1][0])
    assert torch.equal(model.inputs[0][1], model.inputs[1][1])
