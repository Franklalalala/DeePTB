# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the single-embedding multi-head committee feature.

Covers: (1) committee_num_heads=1 (default) is bit-exact with the
pre-committee single-head architecture, including old-checkpoint state_dict
key naming; (2) multi-head forward shapes/finiteness and that heads with
different seeds actually produce different parameters AND different
predictions; (3) fail-closed checkpoint loading on architecture mismatch;
(4) the committee auxiliary training loss
(dptb.nnops.block_ode.committee_endpoint_stats.committee_auxiliary_loss) is a
strict no-op unless both the embedding wrote committee keys AND
flow_options.committee_aux_loss_weight > 0, and when active it actually
produces gradient for the extra heads' predictions through the real
block-ODE endpoint loss path used by train_options.flow_options in this repo.

Reuses existing fixture modules rather than re-deriving minimal
model/data/flow construction from scratch:
  - test_lem_pair_common (model_options/molecule_data/fp64_default) for the
    embedding-level tests, the same fixtures test_hb0_head_input_rms.py uses.
  - test_block_ode_flow (_case/_flow/_fresh/_ref_for) for the loss-level
    tests, the same fixtures that module's own block-ODE endpoint-loss tests
    use.
"""

from __future__ import annotations

import pytest
import torch

from dptb.data import _keys
from dptb.nn.embedding.lem_moe_v3 import (
    COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY,
    COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY,
    unpack_block_native_head_outputs,
)
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nnops.block_ode.committee_endpoint_stats import committee_auxiliary_loss

from test_lem_pair_common import fp64_default, model_options, molecule_data
from test_block_ode_flow import _case, _flow, _fresh, _ref_for


# ===========================================================================
# Embedding-level: construction, bit-exact-off contract, multi-head forward.
# ===========================================================================

def _h0_model(*, seed: int = 20260803, **overrides) -> LemMoEV3H0:
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(overrides)
    torch.manual_seed(seed)
    return LemMoEV3H0(**options).eval()


def test_committee_default_off_is_bit_exact():
    """committee_num_heads absent from the constructor call vs explicitly
    passed 1: identical RNG consumption, state_dict, and forward output, no
    committee keys, no ModuleList wrapping. Mirrors the existing
    test_head_input_rms_default_off_has_no_key_and_is_bit_exact contract."""
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        implicit = _h0_model()
        implicit_rng = torch.random.get_rng_state().clone()
        explicit = _h0_model(committee_num_heads=1)
        explicit_rng = torch.random.get_rng_state().clone()

        assert torch.equal(implicit_rng, explicit_rng)
        assert implicit.state_dict().keys() == explicit.state_dict().keys()
        assert all(
            torch.equal(implicit.state_dict()[key], explicit.state_dict()[key])
            for key in implicit.state_dict()
        )
        assert implicit.committee_enabled is False
        assert not isinstance(implicit.out_node, torch.nn.ModuleList)
        assert not isinstance(implicit.out_edge, torch.nn.ModuleList)

        implicit_out = implicit(molecule_data(implicit))
        explicit_out = explicit(molecule_data(explicit))
        assert COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY not in implicit_out
        assert COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY not in implicit_out
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(implicit_out[key], explicit_out[key])


def test_committee_off_state_dict_uses_pre_feature_key_naming():
    """Old-checkpoint-compat contract: with the flag off, out_node/out_edge
    parameter keys must be `out_node.<param>`, never `out_node.0.<param>` --
    no accidental ModuleList wrapping leaking into the state_dict even when
    committee_num_heads is explicitly 1."""
    with fp64_default():
        model = _h0_model(committee_num_heads=1)
        keys = list(model.state_dict().keys())
        node_keys = [k for k in keys if k.startswith("out_node.")]
        edge_keys = [k for k in keys if k.startswith("out_edge.")]
        assert node_keys and edge_keys
        for k in node_keys + edge_keys:
            second_segment = k.split(".")[1]
            assert not second_segment.isdigit(), f"unexpected list-indexed key {k}"


def test_committee_invalid_num_heads_raises():
    with fp64_default():
        with pytest.raises(ValueError, match="committee_num_heads"):
            _h0_model(committee_num_heads=0)
        with pytest.raises(ValueError, match="committee_num_heads"):
            _h0_model(committee_num_heads=-2)


def test_committee_rejects_non_ao_block_route():
    with fp64_default():
        with pytest.raises(ValueError, match="output_contract"):
            _h0_model(committee_num_heads=3, output_route="h_a0")


def test_committee_multi_head_forward_shapes_and_finiteness():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        k = 4
        model = _h0_model(committee_num_heads=k, committee_head_seed=1000)
        assert isinstance(model.out_node, torch.nn.ModuleList)
        assert isinstance(model.out_edge, torch.nn.ModuleList)
        assert len(model.out_node) == k
        assert len(model.out_edge) == k

        out = model(molecule_data(model))
        primary_node = out[_keys.NODE_HAMILTONIAN_KEY]
        primary_edge = out[_keys.EDGE_HAMILTONIAN_KEY]
        extra_node = out[COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY]
        extra_edge = out[COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY]

        assert extra_node.shape == (k - 1,) + tuple(primary_node.shape)
        assert extra_edge.shape == (k - 1,) + tuple(primary_edge.shape)
        assert torch.isfinite(primary_node).all()
        assert torch.isfinite(primary_edge).all()
        assert torch.isfinite(extra_node).all()
        assert torch.isfinite(extra_edge).all()


def test_committee_heads_are_actually_different():
    """Gate-2a precondition at unit-test scale: different-seed heads must
    have different parameters AND different forward predictions on the same
    shared trunk features (not just different params that happen to cancel)."""
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model = _h0_model(committee_num_heads=4, committee_head_seed=7)
        data = molecule_data(model)

        head_state_dicts = [head.state_dict() for head in model.out_node]
        for i in range(len(head_state_dicts)):
            for j in range(i + 1, len(head_state_dicts)):
                any_diff = any(
                    not torch.equal(head_state_dicts[i][name], head_state_dicts[j][name])
                    for name in head_state_dicts[i]
                )
                assert any_diff, f"heads {i} and {j} have identical out_node params"

        out = model(data)
        extra_edge = out[COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY]
        primary_edge = out[_keys.EDGE_HAMILTONIAN_KEY]
        all_edge_preds = torch.cat([primary_edge.unsqueeze(0), extra_edge], dim=0)
        for i in range(all_edge_preds.shape[0]):
            for j in range(i + 1, all_edge_preds.shape[0]):
                assert not torch.equal(all_edge_preds[i], all_edge_preds[j]), (
                    f"head {i} and head {j} produced identical edge blocks"
                )


def test_committee_head_seed_controls_reproducibility_and_diversity():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model_a = _h0_model(committee_num_heads=3, committee_head_seed=42)
        model_b = _h0_model(committee_num_heads=3, committee_head_seed=42)
        model_c = _h0_model(committee_num_heads=3, committee_head_seed=99)

        # Same committee_head_seed (and same outer torch.manual_seed inside
        # _h0_model) -> identical parameters for every head.
        for head_a, head_b in zip(model_a.out_node, model_b.out_node):
            for name in head_a.state_dict():
                assert torch.equal(head_a.state_dict()[name], head_b.state_dict()[name])

        # Different committee_head_seed -> different extra-head (index 1..K-1)
        # parameters (head 0 is unaffected by committee_head_seed by design).
        differs = False
        for head_a, head_c in zip(list(model_a.out_node)[1:], list(model_c.out_node)[1:]):
            for name in head_a.state_dict():
                if not torch.equal(head_a.state_dict()[name], head_c.state_dict()[name]):
                    differs = True
        assert differs


def test_committee_log_head_input_rms_combination():
    """Exercises the exact flag combination the S2 water-molecule config uses
    (committee_num_heads>1 AND log_head_input_rms=True): the 5-tuple unpack
    path in unpack_block_native_head_outputs."""
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model = _h0_model(committee_num_heads=3, log_head_input_rms=True)
        out = model(molecule_data(model))
        assert "head_input_rms" in out
        assert torch.isfinite(out["head_input_rms"]["node"]).all()
        assert torch.isfinite(out["head_input_rms"]["edge"]).all()
        assert COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY in out
        assert COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY in out


def test_committee_endpoint_condition_source_configures_all_heads():
    """Regression for a real bug this test suite caught: LemMoEV3H0.__init__
    reconfigures the H-B0 head's conditioning via
    `self.out_edge.configure_condition_source("endpoints", ...)` when
    condition_source='endpoints'. With committee heads enabled, self.out_edge
    is a ModuleList, so the old unconditional call raised AttributeError (no
    such method on ModuleList) -- and a naive fix of only reconfiguring head
    0 would have been a SILENT correctness bug instead: heads 1..K-1 would
    keep the 'edge_0e' default while head 0 used 'endpoints', breaking the
    committee's single-architecture-different-seed premise without ever
    raising. Assert every head actually got reconfigured, and that forward()
    runs end to end producing finite committee outputs."""
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model = _h0_model(committee_num_heads=3, condition_source="endpoints")
        assert isinstance(model.out_edge, torch.nn.ModuleList)
        for head in model.out_edge:
            assert head.condition_source == "endpoints"

        out = model(molecule_data(model))
        assert torch.isfinite(out[_keys.EDGE_HAMILTONIAN_KEY]).all()
        assert torch.isfinite(out[COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY]).all()


@pytest.mark.parametrize(
    "committee_enabled,has_rms",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_unpack_block_native_head_outputs_contract(committee_enabled, has_rms):
    node = torch.zeros(2, 3, 3)
    edge = torch.zeros(5, 3, 3)
    rms = {"node": torch.zeros(1), "edge": torch.zeros(1)}
    if committee_enabled:
        extra_node = torch.zeros(2, 2, 3, 3)
        extra_edge = torch.zeros(2, 5, 3, 3)
        head_outputs = (
            (node, edge, extra_node, extra_edge, rms)
            if has_rms
            else (node, edge, extra_node, extra_edge)
        )
    else:
        head_outputs = (node, edge, rms) if has_rms else (node, edge)

    out_node, out_edge, out_extra_node, out_extra_edge, out_rms = (
        unpack_block_native_head_outputs(head_outputs, committee_enabled)
    )
    assert out_node is node
    assert out_edge is edge
    if committee_enabled:
        assert out_extra_node is extra_node
        assert out_extra_edge is extra_edge
    else:
        assert out_extra_node is None
        assert out_extra_edge is None
    if has_rms:
        assert out_rms is rms
    else:
        assert out_rms is None


# ===========================================================================
# Checkpoint backward-compat: fail-closed on architecture mismatch.
# ===========================================================================

def test_committee_off_checkpoint_round_trips():
    with fp64_default():
        source = _h0_model(seed=1, committee_num_heads=1)
        target = _h0_model(seed=2, committee_num_heads=1)
        target.load_state_dict(source.state_dict())  # default strict=True; must not raise


def test_committee_on_checkpoint_round_trips_same_k():
    with fp64_default():
        source = _h0_model(seed=1, committee_num_heads=4)
        target = _h0_model(seed=2, committee_num_heads=4)
        target.load_state_dict(source.state_dict())  # default strict=True; must not raise


def test_committee_checkpoint_shape_mismatch_fails_closed():
    with fp64_default():
        single_head = _h0_model(seed=1, committee_num_heads=1)
        committee = _h0_model(seed=2, committee_num_heads=4)

        with pytest.raises(RuntimeError):
            committee.load_state_dict(single_head.state_dict())
        with pytest.raises(RuntimeError):
            single_head.load_state_dict(committee.state_dict())


# ===========================================================================
# argcheck schema: new flags present, defaults inert.
# ===========================================================================

def test_argcheck_schema_includes_committee_flags_with_inert_defaults():
    from dptb.utils.argcheck import flow_options, slem

    # slem() returns a bare List[Argument] (embedded into a larger schema by
    # its caller), so a dict-comprehension over it works directly.
    slem_args = {arg.name: arg for arg in slem()}
    assert slem_args["committee_num_heads"].optional
    assert slem_args["committee_num_heads"].default == 1
    assert slem_args["committee_head_seed"].optional
    assert slem_args["committee_head_seed"].default == 0

    # flow_options() is different: it returns a single self-contained dargs
    # Argument("flow_options", dict, [...]) node (it is used directly as a
    # schema node elsewhere, not spliced into a parent's sub_fields list), so
    # iterating over it does NOT yield its child Arguments -- Argument has no
    # __iter__, only __getitem__(str), so a bare `for arg in flow_options()`
    # falls back to the legacy int-index iteration protocol and blows up
    # inside dargs (`key.lstrip("/")` on an int). The child arguments live in
    # `.sub_fields`, already a Dict[str, Argument] keyed by name.
    flow_args = flow_options().sub_fields
    assert flow_args["committee_aux_loss_weight"].optional
    assert flow_args["committee_aux_loss_weight"].default == 0.0


# ===========================================================================
# Loss-level: committee_auxiliary_loss no-op guarantee + real gradient flow
# through dptb.nnops.block_ode.endpoint_stats.block_ode_endpoint_loss (the
# ACTUAL training-gradient path for output_space=residual_ao_block_ode, not
# just the HamilBlockwiseNexTHamLoss validation/compatibility criterion).
# ===========================================================================

def test_committee_auxiliary_loss_function_returns_same_object_when_inactive():
    """Isolated no-op check on the helper itself, no model/flow construction."""
    total = torch.tensor(1.23)
    state = {"foo": "bar"}
    owner = type("Owner", (), {"committee_aux_loss_weight": 0.0})()
    out_total, out_state = committee_auxiliary_loss(
        owner,
        {},
        {},
        None,
        total=total,
        state=state,
        target_projected=None,
        node_shapes=None,
        edge_shapes=None,
        node_mask=None,
        edge_mask=None,
    )
    assert out_total is total
    assert out_state is state
    assert "train_committee_aux_loss" not in out_state


def test_committee_auxiliary_loss_is_noop_when_keys_absent():
    idp, data, codec, _ = _case()
    flow_with_weight = _flow(idp, "residual_dh", committee_aux_loss_weight=1.0)
    flow_no_weight = _flow(idp, "residual_dh", committee_aux_loss_weight=0.0)
    node_target = data["node_h0"] * 0.3
    edge_target = data["edge_h0"] * 0.3
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)

    losses = []
    for flow in (flow_with_weight, flow_no_weight):
        ref = _ref_for(flow, data, codec, None, endpoint, node_target, edge_target)
        batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))
        pred = batch.copy()
        pred[flow.node_output_key] = endpoint.node_blocks
        pred[flow.edge_output_key] = endpoint.edge_blocks
        loss, state = flow.loss(pred, ref, ctx)
        assert "train_committee_aux_loss" not in state
        losses.append(loss)

    # A configured nonzero weight with NO committee keys in pred_data must be
    # numerically identical to weight=0.0: presence of keys gates the term,
    # not the weight alone.
    assert torch.allclose(losses[0], losses[1], atol=1e-12)


def test_committee_auxiliary_loss_zero_weight_is_noop_even_with_keys_present():
    idp, data, codec, _ = _case()
    flow = _flow(idp, "residual_dh", committee_aux_loss_weight=0.0)
    node_target = data["node_h0"] * 0.3
    edge_target = data["edge_h0"] * 0.3
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, None, endpoint, node_target, edge_target)
    batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))
    pred = batch.copy()
    pred[flow.node_output_key] = endpoint.node_blocks
    pred[flow.edge_output_key] = endpoint.edge_blocks
    # A badly-wrong extra head that WOULD blow up the loss if the weight gate
    # were broken.
    pred["committee_extra_node_hamiltonian_blocks"] = (
        (endpoint.node_blocks + 5.0).unsqueeze(0)
    )
    pred["committee_extra_edge_hamiltonian_blocks"] = (
        (endpoint.edge_blocks + 5.0).unsqueeze(0)
    )
    loss, state = flow.loss(pred, ref, ctx)
    assert loss.item() <= 1e-10
    assert "train_committee_aux_loss" not in state


def test_committee_auxiliary_loss_active_changes_loss_and_flows_gradient():
    """End-to-end through the REAL training-gradient path: a deliberately
    wrong extra committee head must (a) raise the total loss and (b) receive
    a nonzero, finite gradient -- i.e. it would actually learn, not sit
    frozen at random init."""
    idp, data, codec, _ = _case()
    flow = _flow(idp, "residual_dh", committee_aux_loss_weight=1.0)
    node_target = data["node_h0"] * 0.3
    edge_target = data["edge_h0"] * 0.3
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, None, endpoint, node_target, edge_target)
    batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))

    pred = batch.copy()
    pred[flow.node_output_key] = endpoint.node_blocks
    pred[flow.edge_output_key] = endpoint.edge_blocks
    baseline_loss, _ = flow.loss(pred, ref, ctx)
    assert baseline_loss.item() <= 1e-10  # exact endpoint prediction -> ~0 primary loss

    extra_node = (endpoint.node_blocks + 0.5).unsqueeze(0).clone().detach().requires_grad_(True)
    extra_edge = (endpoint.edge_blocks + 0.5).unsqueeze(0).clone().detach().requires_grad_(True)
    pred_committee = pred.copy()
    pred_committee["committee_extra_node_hamiltonian_blocks"] = extra_node
    pred_committee["committee_extra_edge_hamiltonian_blocks"] = extra_edge

    committee_loss, state = flow.loss(pred_committee, ref, ctx)
    assert committee_loss.item() > 1e-3
    assert state["train_committee_aux_loss"].item() > 1e-3
    assert int(state["train_committee_num_extra_heads"].item()) == 1

    committee_loss.backward()
    assert extra_node.grad is not None
    assert torch.isfinite(extra_node.grad).all()
    assert extra_node.grad.abs().sum().item() > 0.0
    assert extra_edge.grad is not None
    assert torch.isfinite(extra_edge.grad).all()
    assert extra_edge.grad.abs().sum().item() > 0.0


def test_committee_auxiliary_loss_mismatched_head_counts_raises():
    idp, data, codec, _ = _case()
    flow = _flow(idp, "residual_dh", committee_aux_loss_weight=1.0)
    node_target = data["node_h0"] * 0.3
    edge_target = data["edge_h0"] * 0.3
    endpoint = codec.rme_to_blocks(data, node_target, edge_target, project=True)
    ref = _ref_for(flow, data, codec, None, endpoint, node_target, edge_target)
    batch, ref, ctx = flow.prepare_batch(_fresh(data), ref, t=torch.tensor([0.4]))
    pred = batch.copy()
    pred[flow.node_output_key] = endpoint.node_blocks
    pred[flow.edge_output_key] = endpoint.edge_blocks
    pred["committee_extra_node_hamiltonian_blocks"] = endpoint.node_blocks.unsqueeze(0)
    pred["committee_extra_edge_hamiltonian_blocks"] = torch.stack(
        [endpoint.edge_blocks, endpoint.edge_blocks], dim=0
    )
    with pytest.raises(ValueError, match="disagree"):
        flow.loss(pred, ref, ctx)
