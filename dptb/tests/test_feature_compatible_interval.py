# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for HamilBlockwiseNexTHamLoss.log_feature_compatible_interval.

The interval throttles ONLY the logging-only feature-compatible onsite/hopping
metric (the host-sync-heavy path: cpu()/tolist(), Python species/pair grouping,
two collective all-reduces).  It never touches the optimization/gradient loss.

Design choice for non-firing steps: the feature side-effect attributes are set to
None exactly as they are when log_feature_compatible=false (last_feature_compat_loss,
last_onsite_loss, last_hopping_loss, last_feature_count) and the feature_* raw
component sums are simply absent from last_component_stats.  Every trainer/multi_trainer
consumer reads these through getattr(..., default)/dict.get(...) (trainer.py:401-402
via _loss_component_state only inserts when not None; multi_trainer.py:2003-2004 via
_as_scalar_tensor(..., default=0.0) which maps None->default), so absence/None can
never KeyError a consumer and no carry-forward is needed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss


BASIS = {"H": "1s", "O": "1s1p"}
# The feature-compatible count is the mapper's reduced-matrix-element feature
# slice count (RME-derived), which is deterministic but distinct from the block
# active-entry count.  Tests capture it dynamically from a reference criterion
# rather than hardcoding it, so they assert gating behavior + cross-criterion
# byte-identity instead of a brittle magic number.


def _data() -> dict:
    """Fresh synthetic block payload (mirrors test_blockwise_qhflow_lr)."""
    max_norb = 4  # 1s1p union
    pred_node = torch.zeros(2, max_norb, max_norb)
    target_node = torch.zeros(2, max_norb, max_norb)
    target_node[0, 0, 0] = 0.5  # H onsite 1x1
    target_node[1, :4, :4] = 0.25  # O onsite 4x4
    pred_edge = torch.zeros(2, max_norb, max_norb)
    target_edge = torch.zeros(2, max_norb, max_norb)
    target_edge[0, :1, :4] = 1.0  # H->O 1x4
    target_edge[1, :4, :1] = -1.0  # O->H 4x1
    return {
        "node_hamil_blocks": pred_node,
        "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]),
        "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node,
        "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }


def _criterion(**kwargs) -> HamilBlockwiseNexTHamLoss:
    opts = dict(basis=BASIS, optimization="block_mae", block_reduction="global")
    opts.update(kwargs)
    return HamilBlockwiseNexTHamLoss(**opts)


def _feature_state(crit: HamilBlockwiseNexTHamLoss):
    return (
        crit.last_feature_compat_loss,
        crit.last_onsite_loss,
        crit.last_hopping_loss,
        crit.last_feature_count,
    )


def _feature_absent(crit: HamilBlockwiseNexTHamLoss) -> bool:
    compat, onsite, hopping, count = _feature_state(crit)
    return compat is None and onsite is None and hopping is None and count is None


def _feature_present(crit: HamilBlockwiseNexTHamLoss) -> bool:
    return all(v is not None for v in _feature_state(crit))


# ---------------------------------------------------------------------------
# (a) default interval == 1: present every call, byte-identical to an explicit
#     log_feature_compatible=true reference on the same inputs.
# ---------------------------------------------------------------------------
def test_default_interval_is_one_and_matches_explicit_reference():
    default = _criterion()
    assert default.log_feature_compatible_interval == 1

    reference = _criterion(log_feature_compatible=True)
    reference(_data())
    ref_compat, ref_onsite, ref_hopping, ref_count = _feature_state(reference)

    # Reference anchors: a real, non-trivial metric was computed.
    assert ref_count.item() > 0.0
    assert ref_count.item() == pytest.approx(round(ref_count.item()))  # integral count
    assert torch.isfinite(ref_compat) and ref_compat.item() > 0.0
    assert torch.isfinite(ref_onsite) and ref_onsite.item() > 0.0
    assert torch.isfinite(ref_hopping) and ref_hopping.item() > 0.0

    # Default criterion (interval=1) fires on EVERY call, byte-identical values.
    for _ in range(3):
        default(_data())
        assert _feature_present(default)
        compat, onsite, hopping, count = _feature_state(default)
        assert torch.equal(compat, ref_compat)
        assert torch.equal(onsite, ref_onsite)
        assert torch.equal(hopping, ref_hopping)
        assert torch.equal(count, ref_count)


# ---------------------------------------------------------------------------
# (b) interval == 3: fires on calls 1 and 4 (call_index 0 and 3); absent on 2,3.
# ---------------------------------------------------------------------------
def test_interval_three_fires_on_calls_one_and_four():
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)

    fired = []
    for _ in range(4):
        crit(_data())
        fired.append(_feature_present(crit))
        # Block-level logging state is NEVER gated -- present on every call.
        assert crit.last_block_loss is not None
        assert crit.last_block_onsite_loss is not None
        assert crit.last_block_hopping_loss is not None

    assert fired == [True, False, False, True]

    # Re-run to confirm the counter keeps advancing: call 5 (index 4) no, call 6
    # (index 5) no, call 7 (index 6) fires again.
    more = []
    for _ in range(3):
        crit(_data())
        more.append(_feature_present(crit))
    assert more == [False, False, True]


def test_non_firing_step_clears_feature_state_and_stats_like_disabled():
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)
    crit(_data())  # call 1 fires
    assert _feature_present(crit)
    assert "feature_onsite_abs_sum" in crit.last_component_stats
    assert "feature_total_count" in crit.last_component_stats

    crit(_data())  # call 2 does not fire
    assert _feature_absent(crit)
    # Feature raw component sums are simply absent (same as disabled path);
    # block_* raw sums remain present.
    assert "feature_onsite_abs_sum" not in crit.last_component_stats
    assert "feature_hopping_abs_sum" not in crit.last_component_stats
    assert "feature_total_count" not in crit.last_component_stats
    assert "block_onsite_abs_sum" in crit.last_component_stats


# ---------------------------------------------------------------------------
# (c) firing values under interval=3 equal an interval=1 criterion (same inputs).
# ---------------------------------------------------------------------------
def test_firing_values_identical_across_intervals():
    ref = _criterion(log_feature_compatible=True, log_feature_compatible_interval=1)
    ref(_data())
    ref_state = _feature_state(ref)

    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)
    firing_calls = (0, 3)  # call_index values that fire
    for call_index in range(4):
        crit(_data())
        if call_index in firing_calls:
            for got, expected in zip(_feature_state(crit), ref_state):
                assert torch.equal(got, expected)
        else:
            assert _feature_absent(crit)


# ---------------------------------------------------------------------------
# (d) invalid interval -> fail-closed ValueError (0, negative, bool, non-int).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -5, True, False, 2.5])
def test_invalid_interval_raises(bad):
    with pytest.raises(ValueError, match="log_feature_compatible_interval"):
        _criterion(log_feature_compatible_interval=bad)


# ---------------------------------------------------------------------------
# (e) optimization-mode feature requirement is NOT gated by the interval.
# ---------------------------------------------------------------------------
def test_optimization_feature_mode_not_gated_by_interval():
    crit = _criterion(
        optimization="feature_compatible",
        log_feature_compatible=False,  # logging trigger off ...
        log_feature_compatible_interval=5,  # ... and a large cadence
    )
    # Every call must still compute features because the optimization loss needs
    # them; the interval only ever gates the logging trigger.
    counts = []
    for _ in range(6):
        loss = crit(_data())
        assert torch.isfinite(loss)
        assert _feature_present(crit)
        counts.append(crit.last_feature_count.item())
    assert all(c > 0.0 for c in counts)
    assert len(set(counts)) == 1  # feature count stable across calls (never gated)


# ---------------------------------------------------------------------------
# (f) argcheck accepts the new key (both variants) and defaults to 1.
# ---------------------------------------------------------------------------
def test_argcheck_accepts_interval_and_defaults_to_one():
    from dptb.utils.argcheck import train_options

    def _cfg(method, loss_train):
        return {
            "num_epoch": 1,
            "batch_size": 1,
            "optimizer": {"type": "AdamW", "lr": 1e-3},
            "lr_scheduler": {"type": "rop"},
            "loss_options": {"train": {"method": method, **loss_train}},
        }

    # explicit value, hamil_blockwise_nextham
    cfg = _cfg("hamil_blockwise_nextham", {"log_feature_compatible_interval": 25})
    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    assert normalized["loss_options"]["train"]["log_feature_compatible_interval"] == 25

    # default when omitted
    cfg_default = _cfg("hamil_blockwise_nextham", {})
    normalized_default = train_options().normalize_value(cfg_default)
    train_options().check_value(normalized_default, strict=True)
    assert normalized_default["loss_options"]["train"]["log_feature_compatible_interval"] == 1

    # sibling variant hamil_block_abs shares the same schema
    cfg_abs = _cfg("hamil_block_abs", {"log_feature_compatible_interval": 7})
    normalized_abs = train_options().normalize_value(cfg_abs)
    train_options().check_value(normalized_abs, strict=True)
    assert normalized_abs["loss_options"]["train"]["log_feature_compatible_interval"] == 7


# ---------------------------------------------------------------------------
# (g) log_feature_compatible=false still skips the feature path entirely
#     (frozen behavior; the interval counter has no effect).
# ---------------------------------------------------------------------------
def test_log_feature_compatible_false_skips_feature_path():
    crit = _criterion(log_feature_compatible=False)  # default interval == 1
    for _ in range(3):
        loss = crit(_data())
        assert torch.isfinite(loss)
        assert _feature_absent(crit)
        assert "feature_onsite_abs_sum" not in crit.last_component_stats
        # optimization/block loss is untouched and always present
        assert crit.last_block_loss is not None
        assert crit.last_opt_loss is not None


# ===========================================================================
# FINDING B: the throttle cadence must not dilute the MultiTrainer epoch
# onsite/hopping metric.
#
# _snapshot_loss_metrics used to coerce a throttled (None) last_onsite_loss /
# last_hopping_loss to 0.0, and _build_train_payload weighted it by the FULL
# active node/edge count, so a skipped batch added 0 to the numerator but the
# full count to the denominator -- averaging 2.0,skip toward 1.0.  The fix marks
# the snapshot entry invalid (None) and gates BOTH the numerator AND the weight
# (denominator/count) via _gated_metric_weighted_sum, so a skipped batch drops
# out of the average entirely; the raw active_nodes/active_edges telemetry stays
# untouched, and interval=1 (which never throttles) stays byte-identical.
# ===========================================================================
from dptb.nnops.multi_trainer import MultiTrainer


_ACTIVE_NODES = 4.0
_ACTIVE_EDGES = 6.0
_TRUE_METRIC = 2.0


def _minimal_multitrainer() -> MultiTrainer:
    mt = MultiTrainer.__new__(MultiTrainer)
    mt.dtype = torch.float64
    mt.device = torch.device("cpu")
    return mt


class _StubCriterion:
    """Bare loss-module stub exposing only the feature-compatible side effects.

    Mirrors HamilBlockwiseNexTHamLoss's snapshot surface: a firing step exposes
    last_onsite_loss/last_hopping_loss as tensors, a throttled step leaves them
    None.  It deliberately has NO last_onsite_l1_sum/last_onsite_count so the
    only onsite/hopping signal flows through the weighted-sum path FINDING B is
    about (the l1_sum/count stats path is separately guarded already).
    """

    def __init__(self, onsite, hopping):
        self.last_onsite_loss = onsite
        self.last_hopping_loss = hopping
        self.last_z_loss = None
        self.expert_load_cv = None


def _payload(mt: MultiTrainer, onsite_val, hopping_val):
    """Drive the real _snapshot_loss_metrics + _build_train_payload seam.

    Only the model-dependent _run_one_expert_loss is stubbed; the snapshot and
    the weighted-sum/weight aggregation under test run for real.
    """
    stub = _StubCriterion(
        None if onsite_val is None else torch.tensor(onsite_val, dtype=torch.float64),
        None if hopping_val is None else torch.tensor(hopping_val, dtype=torch.float64),
    )

    def _fake_run_one_expert_loss(**kwargs):
        out = {
            "loss": torch.zeros((), dtype=torch.float64),
            "active_nodes": torch.tensor(_ACTIVE_NODES, dtype=torch.float64),
            "active_edges": torch.tensor(_ACTIVE_EDGES, dtype=torch.float64),
        }
        out.update(mt._snapshot_loss_metrics(stub))
        return out

    mt._run_one_expert_loss = _fake_run_one_expert_loss
    mt.train_lossfunc = stub
    return mt._build_train_payload(
        batch_dict=None, batch_info=None, expert_idx=0, range_dis=None
    )


def _aggregate_onsite_hopping(mt: MultiTrainer, payloads):
    pack = torch.zeros(MultiTrainer._PACK_LEN, dtype=mt.dtype, device=mt.device)
    for p in payloads:
        pack = pack + mt._make_step_pack(p)
    onsite = (
        pack[MultiTrainer._P_ONSITE_WEIGHTED_SUM]
        / pack[MultiTrainer._P_ACTIVE_NODES_SUM].clamp_min(1.0)
    ).item()
    hopping = (
        pack[MultiTrainer._P_HOPPING_WEIGHTED_SUM]
        / pack[MultiTrainer._P_ACTIVE_EDGES_SUM].clamp_min(1.0)
    ).item()
    return onsite, hopping, pack


# ---------------------------------------------------------------------------
# (h) The real criterion under interval=2: snapshot fires then marks None.
# ---------------------------------------------------------------------------
def test_snapshot_marks_throttled_metrics_none_with_real_criterion():
    mt = _minimal_multitrainer()
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=2)

    crit(_data())  # call 1 fires
    fired = mt._snapshot_loss_metrics(crit)
    assert fired["onsite"] is not None and torch.isfinite(fired["onsite"])
    assert fired["hopping"] is not None and torch.isfinite(fired["hopping"])
    # the snapshot resolved the same module the criterion wrote to
    assert torch.equal(fired["onsite"], crit.last_onsite_loss)
    assert torch.equal(fired["hopping"], crit.last_hopping_loss)

    crit(_data())  # call 2 throttled -> attrs None -> snapshot invalid
    skipped = mt._snapshot_loss_metrics(crit)
    assert skipped["onsite"] is None
    assert skipped["hopping"] is None


# ---------------------------------------------------------------------------
# (i) interval=1 and interval=2 both aggregate to exactly the true metric.
# ---------------------------------------------------------------------------
def test_aggregation_not_diluted_by_throttled_batch():
    mt = _minimal_multitrainer()

    # interval=1 analog: both batches fire.
    onsite1, hopping1, _ = _aggregate_onsite_hopping(
        mt, [_payload(mt, _TRUE_METRIC, _TRUE_METRIC), _payload(mt, _TRUE_METRIC, _TRUE_METRIC)]
    )
    assert onsite1 == pytest.approx(_TRUE_METRIC)
    assert hopping1 == pytest.approx(_TRUE_METRIC)

    # interval=2 analog: batch 1 fires, batch 2 throttled.
    p_fire = _payload(mt, _TRUE_METRIC, _TRUE_METRIC)
    p_skip = _payload(mt, None, None)
    onsite2, hopping2, pack2 = _aggregate_onsite_hopping(mt, [p_fire, p_skip])

    # Not diluted toward 1.0 -- identical to interval=1; only the update cadence
    # differs.
    assert onsite2 == pytest.approx(_TRUE_METRIC)
    assert hopping2 == pytest.approx(_TRUE_METRIC)

    # The throttled batch contributes ZERO numerator and ZERO weight (count) ...
    assert p_skip["onsite_weighted_sum"].item() == 0.0
    assert p_skip["hopping_weighted_sum"].item() == 0.0
    assert p_skip["onsite_weight"].item() == 0.0
    assert p_skip["hopping_weight"].item() == 0.0
    # ... while its raw active_nodes/active_edges telemetry is untouched.
    assert p_skip["active_nodes"].item() == pytest.approx(_ACTIVE_NODES)
    assert p_skip["active_edges"].item() == pytest.approx(_ACTIVE_EDGES)

    # A firing batch's gated weight equals the raw active count (byte-identical
    # seam for the interval=1 / always-firing case).
    assert p_fire["onsite_weight"].item() == pytest.approx(_ACTIVE_NODES)
    assert p_fire["hopping_weight"].item() == pytest.approx(_ACTIVE_EDGES)

    # The aggregated pack denominators hold only the firing batch's count.
    assert pack2[MultiTrainer._P_ACTIVE_NODES_SUM].item() == pytest.approx(_ACTIVE_NODES)
    assert pack2[MultiTrainer._P_ACTIVE_EDGES_SUM].item() == pytest.approx(_ACTIVE_EDGES)
    assert pack2[MultiTrainer._P_ONSITE_WEIGHTED_SUM].item() == pytest.approx(
        _TRUE_METRIC * _ACTIVE_NODES
    )


# ---------------------------------------------------------------------------
# (j) Explicit dilution contrast: the pre-fix raw-count denominator WOULD halve
#     the metric; the gated weight denominator does not.
# ---------------------------------------------------------------------------
def test_raw_denominator_would_dilute_but_gated_weight_does_not():
    mt = _minimal_multitrainer()
    p_fire = _payload(mt, _TRUE_METRIC, _TRUE_METRIC)
    p_skip = _payload(mt, None, None)

    numerator = p_fire["onsite_weighted_sum"] + p_skip["onsite_weighted_sum"]

    # Pre-fix behavior: numerator gated to 0 for the skip, but the FULL active
    # count still counted in the denominator (4 + 4 = 8) -> 8/8*... = 1.0.
    raw_denominator = p_fire["active_nodes"] + p_skip["active_nodes"]
    assert (numerator / raw_denominator).item() == pytest.approx(_TRUE_METRIC / 2.0)

    # Fixed behavior: the gated weight denominator (4 + 0 = 4) -> the true metric.
    gated_denominator = p_fire["onsite_weight"] + p_skip["onsite_weight"]
    assert (numerator / gated_denominator).item() == pytest.approx(_TRUE_METRIC)


# ---------------------------------------------------------------------------
# (k) The real metric-producing aggregation function agrees.
# ---------------------------------------------------------------------------
def test_compute_compatible_state_from_pack_reports_true_metric_when_throttled():
    mt = _minimal_multitrainer()
    pack = mt._make_step_pack(_payload(mt, _TRUE_METRIC, _TRUE_METRIC)) + mt._make_step_pack(
        _payload(mt, None, None)
    )

    state = mt._compute_compatible_state_from_pack(
        pack, criterion=_StubCriterion(None, None), prefix="train"
    )

    assert state is not None
    assert state["train_onsite_loss"].item() == pytest.approx(_TRUE_METRIC)
    assert state["train_hopping_loss"].item() == pytest.approx(_TRUE_METRIC)


# ===========================================================================
# H10: standard Trainer.validation per-key valid-batch counts (P2 trainer.py).
#
# The reviewer's repro: with a throttleable feature-compatible onsite/hopping
# metric, a non-firing batch reports None (omitted), so dividing the accumulated
# metric sum by the UNIFORM num_batches would dilute it (2.0 over one firing batch
# of two -> 1.0).  The fix accumulates a PER-KEY contributing-batch count in
# _accumulate_metric_state and validation() divides each metric by its own count.
# This drives the real static accumulator seam and the exact divisor formula from
# Trainer.validation (trainer.py ~line 929).
# ===========================================================================
from dptb.nnops.trainer import Trainer


def _resolve_per_key(metric_sums, counts, num_batches):
    """The exact divisor rule Trainer.validation applies to its metric sums."""
    divisor = max(num_batches, 1)
    return {key: value / counts.get(key, divisor) for key, value in metric_sums.items()}


def test_h10_two_batch_validation_per_key_count_not_diluted():
    """interval=2 analog: batch 1 fires (loss 3, onsite 2), batch 2 throttles the
    onsite metric (None) but still reports loss 3.  validation_loss averages to 3.0
    (present on both batches) while validation_onsite_loss stays 2.0 (NOT 1.0)."""
    metric_sums, counts = {}, {}
    Trainer._accumulate_metric_state(
        metric_sums,
        {
            "validation_loss": torch.tensor(3.0),
            "validation_onsite_loss": torch.tensor(2.0),
            "validation_hopping_loss": torch.tensor(2.0),
        },
        counts,
    )
    Trainer._accumulate_metric_state(
        metric_sums,
        {
            "validation_loss": torch.tensor(3.0),
            "validation_onsite_loss": None,  # throttled this batch
            "validation_hopping_loss": None,
        },
        counts,
    )
    # per-key counts: loss on both batches, onsite/hopping on only the firing one.
    assert counts["validation_loss"] == 2
    assert counts["validation_onsite_loss"] == 1
    assert counts["validation_hopping_loss"] == 1

    resolved = _resolve_per_key(metric_sums, counts, num_batches=2)
    assert resolved["validation_loss"].item() == pytest.approx(3.0)
    assert resolved["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert resolved["validation_hopping_loss"].item() == pytest.approx(2.0)

    # Contrast: the pre-fix uniform num_batches divisor WOULD dilute onsite to 1.0.
    assert (metric_sums["validation_onsite_loss"] / 2).item() == pytest.approx(1.0)


def test_h10_interval_one_twin_is_byte_identical_to_uniform_divisor():
    """interval=1 analog: every key present on every batch => per-key count ==
    num_batches, so the per-key divisor is byte-identical to the uniform divisor."""
    metric_sums, counts = {}, {}
    for _ in range(2):
        Trainer._accumulate_metric_state(
            metric_sums,
            {"validation_loss": torch.tensor(3.0), "validation_onsite_loss": torch.tensor(2.0)},
            counts,
        )
    per_key = _resolve_per_key(metric_sums, counts, num_batches=2)
    uniform = {key: value / 2 for key, value in metric_sums.items()}
    assert set(per_key) == set(uniform)
    for key in per_key:
        assert torch.equal(per_key[key], uniform[key])


# ===========================================================================
# H10b: the SAME per-key-count fix, now on MultiTrainer.validation (FINDING C /
# review F2).  MultiTrainer.validation divided every accumulated metric by a
# UNIFORM num_batches and never threaded a per-key count, so a metric omitted on
# some batches was diluted (the standard Trainer was fixed by H10; this was not).
# Drives the real validation() loop (non-distributed full-forward path) over a
# 2-batch loader whose second batch omits onsite/hopping.
# ===========================================================================
class _NullTagger:
    def tag(self, *args, **kwargs):
        return contextlib.nullcontext()


def test_h10b_multitrainer_validation_per_key_count_not_diluted(monkeypatch):
    """MultiTrainer.validation twin of H10: onsite present on 1 of 2 batches must
    report 2.0 (divided by its OWN contributing-batch count), not 1.0 (value /
    num_batches).  Exercises the real validation() accumulation + divisor."""
    mt = _minimal_multitrainer()
    mt.model = SimpleNamespace(eval=lambda: None)
    mt.validation_loader = [object(), object()]  # two batches
    mt.validation_loader_generator = None
    mt.distributed_expert = False
    mt.flow_cfm = None
    mt.log_single_model_compatible_loss = False
    mt.iter = 0
    mt._tagger = _NullTagger()
    mt.validation_lossfunc = object()

    monkeypatch.setattr(
        mt, "_prepare_batch_bundle", lambda batch, with_lengths=True: (None, None)
    )
    monkeypatch.setattr(
        mt, "_run_full_batch_loss", lambda *a, **k: torch.zeros((), dtype=mt.dtype)
    )
    snapshots = iter([
        {"onsite": torch.tensor(2.0, dtype=mt.dtype), "hopping": torch.tensor(2.0, dtype=mt.dtype)},
        {"onsite": None, "hopping": None},  # throttled/omitted this batch
    ])
    monkeypatch.setattr(mt, "_snapshot_loss_metrics", lambda crit: next(snapshots))

    mt.validation(fast=False)

    state = mt._last_flow_validation_state
    # Un-diluted: divided by its own count (1 firing batch), not num_batches (2).
    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)


# ===========================================================================
# H11: MultiTrainer per-expert display metric + fired-step window (P2
# multi_trainer.py).  Same FINDING B mechanism, now on the expert-display seam:
# the per-expert onsite/hopping ratio and the display-window average must use the
# GATED weight (which drops a throttled step) so a throttled batch does not halve
# the displayed metric.  Raw active_nodes/active_edges telemetry stays untouched.
# ===========================================================================
def _expert_run_output(mt, stub, active_nodes, active_edges):
    """One _run_one_expert_loss result: model-independent loss/active + the real
    _snapshot_loss_metrics of the stub criterion (onsite/hopping possibly None)."""
    out = {
        "loss": torch.zeros((), dtype=torch.float64),
        "active_nodes": torch.tensor(active_nodes, dtype=torch.float64),
        "active_edges": torch.tensor(active_edges, dtype=torch.float64),
    }
    out.update(mt._snapshot_loss_metrics(stub))
    return out


def _stub(value):
    return _StubCriterion(
        None if value is None else torch.tensor(value, dtype=torch.float64),
        None if value is None else torch.tensor(value, dtype=torch.float64),
    )


def test_h11_expert_display_metric_uses_gated_denominator_across_main_and_ref():
    """main fires (metric 2, active 4/6) + ref throttled (None, raw active 4/6 but
    GATED weight 0) => expert_onsite/hopping == 2 (not 1): the gated denominator
    (4 from main + 0 from ref), not the raw 4+4."""
    mt = _minimal_multitrainer()
    mt.distributed_expert = False

    outputs = iter([_expert_run_output(mt, _stub(2.0), 4.0, 6.0),
                    _expert_run_output(mt, _stub(None), 4.0, 6.0)])

    def _fake_run(**kwargs):
        return next(outputs)

    mt._run_one_expert_loss = _fake_run
    mt.train_lossfunc = _stub(2.0)
    payload = mt._build_train_payload(
        batch_dict=None, batch_info=None, expert_idx=0, range_dis=None,
        ref_batch_dict={"stub": True}, ref_batch_info=None,
    )

    assert payload["expert_onsite"].item() == pytest.approx(2.0)
    assert payload["expert_hopping"].item() == pytest.approx(2.0)
    # gated denominators drop the throttled ref (weight 0), not raw 4+4.
    assert payload["onsite_weight"].item() == pytest.approx(4.0)
    assert payload["hopping_weight"].item() == pytest.approx(6.0)
    # raw active telemetry is the FULL main+ref count, untouched by gating.
    assert payload["active_nodes"].item() == pytest.approx(8.0)
    assert payload["active_edges"].item() == pytest.approx(12.0)


def _init_display_window(mt):
    mt._display_window_pack_local = torch.zeros((mt._PACK_LEN,), dtype=mt.dtype)
    mt._display_window_dynamic_batch_pack_local = torch.zeros((mt._DB_PACK_LEN,), dtype=mt.dtype)
    for attr in (
        "_display_window_expert_onsite_sum_local",
        "_display_window_expert_hopping_sum_local",
        "_display_window_expert_active_nodes_sum_local",
        "_display_window_expert_active_edges_sum_local",
        "_display_window_expert_onsite_steps_local",
        "_display_window_expert_hopping_steps_local",
    ):
        setattr(mt, attr, torch.zeros((), dtype=mt.dtype))
    mt._display_window_last_lr_local = 0.0


def _single_expert_payload(mt, value):
    outputs = iter([_expert_run_output(mt, _stub(value), 4.0, 6.0)])

    def _fake_run(**kwargs):
        return next(outputs)

    mt._run_one_expert_loss = _fake_run
    mt.train_lossfunc = _stub(value)
    return mt._build_train_payload(
        batch_dict=None, batch_info=None, expert_idx=0, range_dis=None
    )


def test_h11_display_window_expert_metric_averages_only_over_fired_steps():
    """A window with one firing step (expert 2, gated weight 4) and one throttled
    step (gated weight 0) reports the window expert metric averaged over the FIRED
    step count (== 2, not diluted to 1), while the raw active_nodes/active_edges
    telemetry keeps the plain step-count mean (untouched)."""
    mt = _minimal_multitrainer()
    mt.distributed_expert = False
    _init_display_window(mt)

    mt._update_display_window_local(_single_expert_payload(mt, 2.0), current_local_lr=1e-3)
    mt._update_display_window_local(_single_expert_payload(mt, None), current_local_lr=1e-3)

    onsite_sum, onsite_fired, hopping_sum, hopping_fired, lr, active_nodes, active_edges = (
        mt._gather_display_window_expert_metrics()[0]
    )
    # Gathered layout now carries RAW sums + fired counts (the per-expert merge does
    # Σsum/Σfired downstream); the single-rank pooled mean is still 2, averaged over
    # the fired step (1), not the 2 window steps.
    assert (onsite_sum / onsite_fired.clamp_min(1.0)).item() == pytest.approx(2.0)
    assert (hopping_sum / hopping_fired.clamp_min(1.0)).item() == pytest.approx(2.0)
    assert onsite_fired.item() == pytest.approx(1.0)
    assert hopping_fired.item() == pytest.approx(1.0)
    # fired-step counters recorded exactly one contributing step per metric.
    assert mt._display_window_expert_onsite_steps_local.item() == pytest.approx(1.0)
    assert mt._display_window_expert_hopping_steps_local.item() == pytest.approx(1.0)
    # raw active telemetry: mean over BOTH window steps (unchanged by throttling).
    assert active_nodes.item() == pytest.approx(_ACTIVE_NODES)
    assert active_edges.item() == pytest.approx(_ACTIVE_EDGES)


def test_h11b_expert_display_pools_fired_counts_across_dp_ranks():
    """Two DP ranks mapped to the SAME expert with UNEQUAL fired counts: the merged
    onsite/hopping must be the pooled Σsum/Σfired (fired-count-weighted global mean),
    NOT the equal-weight average of the two rank-local means (FINDING D / review F3).

    rank0: onsite_sum=10, fired=1  (rank-local mean 10)
    rank1: onsite_sum=9,  fired=9  (rank-local mean 1)
    pooled = (10+9)/(1+9) = 1.9    (average of rank means would be the wrong 5.5)
    """
    mt = _minimal_multitrainer()
    mt.num_experts = 1
    mt._rank_to_expert_idx = lambda rank: 0  # both ranks -> expert 0

    def _vec(onsite_sum, onsite_fired, hopping_sum, hopping_fired, lr, an, ae):
        return torch.tensor(
            [onsite_sum, onsite_fired, hopping_sum, hopping_fired, lr, an, ae],
            dtype=mt.dtype,
        )

    gathered = [
        _vec(10.0, 1.0, 3.0, 1.0, 1e-3, 4.0, 6.0),
        _vec(9.0, 9.0, 27.0, 9.0, 1e-3, 4.0, 6.0),
    ]
    merged = mt._expert_display_metrics_from_gathered(gathered)

    assert merged["expert_0_onsite"] == pytest.approx((10.0 + 9.0) / (1.0 + 9.0))    # 1.9
    assert merged["expert_0_hopping"] == pytest.approx((3.0 + 27.0) / (1.0 + 9.0))   # 3.0
    # The old equal-weight average of rank-local means would have been 5.5 (wrong).
    assert merged["expert_0_onsite"] != pytest.approx((10.0 / 1.0 + 9.0 / 9.0) / 2)
    # lr and active telemetry keep the per-rank mean.
    assert merged["expert_0_lr"] == pytest.approx(1e-3)
    assert merged["expert_0_active_nodes"] == pytest.approx(4.0)
    assert merged["expert_0_active_edges"] == pytest.approx(6.0)


def test_h11c_payload_metrics_never_source_onsite_from_flow_namespace():
    """FINDING E: the compatible onsite/hopping payload must come from the compatible
    metric, never the flow-namespaced train_flow_* value.  Since the flow objective no
    longer aliases its value into the bare train_onsite_loss, a flow-only state must
    not report the flow value as the compatible onsite/hopping -- the namespaces are
    disjoint (in a real run the forced compatible pass supplies the compatible value)."""
    mt = _minimal_multitrainer()
    flow_only = {
        "train_flow_onsite_loss": torch.tensor(7.0, dtype=mt.dtype),
        "train_flow_hopping_loss": torch.tensor(9.0, dtype=mt.dtype),
        # no train_compatible_*, no bare train_onsite_loss, no _compatible_clean_stats
    }
    metrics = mt._payload_metrics_from_flow_state(flow_only, prefix="train")
    # Not the flow value; falls back to the neutral default, not train_flow_*.
    assert metrics["onsite"].item() != pytest.approx(7.0)
    assert metrics["hopping"].item() != pytest.approx(9.0)
    assert metrics["onsite"].item() == pytest.approx(0.0)
    assert metrics["hopping"].item() == pytest.approx(0.0)
