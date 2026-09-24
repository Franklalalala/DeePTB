"""WS4 phase C smoke tests plus the sc_guard ABACUS-log/residual-guard unit
tests (absorbs test_sc_guard.py).

The self-consistency tests use a fake ``repair_fn`` (no ABACUS dependency) to
lock down the mechanism: loss gradient only flows through ``h_pred``, the
submit/consume double-buffering schedule fires on the right cadence and
respects ``staleness_steps``, and a failing/refused repair degrades to "no
loss this round" instead of raising. The real ABACUS-backed accuracy claims
(does L_sc actually improve generalization) need a full training run and are
out of scope here -- see the WS4 report.

The sc_guard tests exercise dptb.postprocess.hrebuild's calibration helpers;
that module only needs the optional dftio package for its ABACUS/dftio CSR
conversion path (not for these functions), but they are still guarded so they
report skipped rather than silently meaning less on an environment where
dftio is genuinely absent.
"""
import threading
import time

import numpy as np
import pytest
import torch

from dptb.nnops.self_consistency import (
    SelfConsistencyScheduler,
    SelfConsistencySchedulerConfig,
    compute_self_consistency_loss,
    compute_self_consistency_payload_loss,
)
from dptb.tests._requires import requires_module


def test_loss_gradient_only_flows_through_h_pred():
    h_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h_repaired = torch.tensor([1.5, 2.5, 2.5], requires_grad=True)  # would-be leaf if not detached

    loss = compute_self_consistency_loss(h_pred, h_repaired)
    loss.backward()

    assert h_pred.grad is not None
    assert h_repaired.grad is None  # detached inside compute_self_consistency_loss


@pytest.mark.parametrize(
    "h_pred,h_repaired,mask,expected",
    [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]),
         torch.tensor([1.0, 1.0, 0.0]), 2.5),
        (torch.ones(2, 3), torch.zeros(2, 3), torch.tensor([1.0, 0.0]), 1.0),
    ],
    ids=["1d_mask_only_first_two_elements", "per_sample_mask_broadcasts_over_elements"],
)
def test_loss_masking_and_broadcast_denominator(h_pred, h_repaired, mask, expected):
    """The denominator counts active ELEMENTS after broadcast, not active
    samples: a per-sample mask (leading dims) broadcasts against per-element
    diffs, so masked and unmasked paths agree on scale (per-element mean)."""
    loss = compute_self_consistency_loss(h_pred, h_repaired, mask=mask)
    assert torch.isclose(loss, torch.tensor(expected))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_self_consistency_loss(torch.zeros(3), torch.zeros(4))


def test_loss_coerces_numpy_repair_to_pred_device_dtype():
    """Real repair endpoints (ABACUS subprocess / hrebuild server) return CPU
    numpy float64 arrays; the loss must not require the caller to convert."""
    h_pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    h_repaired = np.array([1.0, 2.0, 3.0], dtype=np.float64) + 0.5

    loss = compute_self_consistency_loss(h_pred, h_repaired)
    assert loss.dtype == torch.float32
    loss.backward()
    assert h_pred.grad is not None


def test_payload_loss_averages_configured_feature_tensors():
    h_pred = {
        "node_features": torch.tensor([1.0, 3.0], requires_grad=True),
        "edge_features": torch.tensor([2.0], requires_grad=True),
        "metadata": "kept out of loss",
    }
    h_repaired = {
        "node_features": torch.tensor([2.0, 1.0]),
        "edge_features": torch.tensor([4.0]),
    }

    loss = compute_self_consistency_payload_loss(
        h_pred, h_repaired, tensor_keys=("node_features", "edge_features")
    )
    loss.backward()

    assert loss.item() == pytest.approx(((1.0 + 4.0) / 2.0 + 4.0) / 2.0)
    assert h_pred["node_features"].grad is not None
    assert h_pred["edge_features"].grad is not None


def _fake_repair_fn_success(sample_id, h_pred_snapshot):
    time.sleep(0.05)
    return h_pred_snapshot + 0.01  # pretend ABACUS nudged it slightly


def _fake_repair_fn_refuses(sample_id, h_pred_snapshot):
    return None  # e.g. gap-threshold guard refused


def test_scheduler_round_trips_mapping_payloads():
    def repair_payload(_sample_id, snapshot):
        return {
            "node_features": snapshot["node_features"] + 1.0,
            "edge_features": snapshot["edge_features"] + 2.0,
        }

    cfg = SelfConsistencySchedulerConfig(
        every_n_steps=1, sample_frac=1.0, staleness_steps=1, warmup_epochs=0
    )
    sched = SelfConsistencyScheduler(repair_fn=repair_payload, config=cfg)
    try:
        sample = {
            "node_features": torch.tensor([1.0, 2.0]),
            "edge_features": torch.tensor([3.0]),
            "meta": "not a tensor",
        }
        assert sched.maybe_submit(step=0, epoch=0, samples=[("batch", sample)])
        pairs = sched.maybe_consume(step=1, current_samples={"batch": sample}, timeout=1.0)
    finally:
        sched.shutdown()

    assert len(pairs) == 1
    current, repaired = pairs[0]
    assert current is sample
    assert torch.allclose(repaired["node_features"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(repaired["edge_features"], torch.tensor([5.0]))


def test_submit_consume_cadence_and_staleness():
    cfg = SelfConsistencySchedulerConfig(
        every_n_steps=2, sample_frac=1.0, staleness_steps=1, warmup_epochs=0
    )
    sched = SelfConsistencyScheduler(repair_fn=_fake_repair_fn_success, config=cfg)
    try:
        samples_step0 = [("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0, 4.0]))]

        # step 0 is on-cadence (0 % 2 == 0) -> submits, due at step 0+1=1
        submitted = sched.maybe_submit(step=0, epoch=0, samples=samples_step0)
        assert submitted is True

        # step 1 is off-cadence for submission, but its due requests haven't
        # necessarily finished yet (0.05s sleep) -- consuming with timeout=0
        # is allowed to see nothing yet; poll with a real timeout instead so
        # the test is deterministic rather than racing the worker thread.
        current = {"a": torch.tensor([1.01, 2.01]), "b": torch.tensor([3.01, 4.01])}
        pairs = sched.maybe_consume(step=1, current_samples=current, timeout=2.0)
        assert len(pairs) == 2
        for h_pred_now, h_repaired in pairs:
            assert h_repaired.requires_grad is False

        # nothing was submitted at step 1 (off-cadence), so step 2's due
        # bucket (1+1) is empty -- consuming twice must not double-count
        assert sched.maybe_consume(step=1, current_samples=current, timeout=0.1) == []

        # step 2 is on-cadence again
        submitted2 = sched.maybe_submit(step=2, epoch=0, samples=samples_step0)
        assert submitted2 is True
        pairs2 = sched.maybe_consume(step=3, current_samples=current, timeout=2.0)
        assert len(pairs2) == 2
    finally:
        sched.shutdown()


def test_warmup_epochs_suppresses_submission():
    cfg = SelfConsistencySchedulerConfig(every_n_steps=1, sample_frac=1.0, warmup_epochs=5)
    sched = SelfConsistencyScheduler(repair_fn=_fake_repair_fn_success, config=cfg)
    try:
        submitted = sched.maybe_submit(step=0, epoch=0, samples=[("a", torch.zeros(2))])
        assert submitted is False
        submitted = sched.maybe_submit(step=0, epoch=5, samples=[("a", torch.zeros(2))])
        assert submitted is True
    finally:
        sched.shutdown()


def test_refused_repair_drops_sample_without_raising():
    cfg = SelfConsistencySchedulerConfig(every_n_steps=1, sample_frac=1.0, staleness_steps=0)
    sched = SelfConsistencyScheduler(repair_fn=_fake_repair_fn_refuses, config=cfg)
    try:
        sched.maybe_submit(step=0, epoch=0, samples=[("a", torch.zeros(2))])
        pairs = sched.maybe_consume(step=0, current_samples={"a": torch.zeros(2)}, timeout=2.0)
        assert pairs == []  # refused (None) repair is dropped, not raised
    finally:
        sched.shutdown()


def test_unfinished_repair_is_requeued_not_dropped():
    """A due-but-unfinished repair must survive into later consume calls
    (whatever their step -- the overdue pool is not keyed on step), so slow
    ABACUS jobs still contribute L_sc instead of silently becoming no-ops."""
    release = threading.Event()

    def slow_repair(sample_id, snapshot):
        release.wait(timeout=10.0)
        return snapshot + 1.0

    cfg = SelfConsistencySchedulerConfig(every_n_steps=1, sample_frac=1.0, staleness_steps=1)
    sched = SelfConsistencyScheduler(repair_fn=slow_repair, config=cfg)
    try:
        sched.maybe_submit(step=0, epoch=0, samples=[("a", torch.zeros(2))])
        current = {"a": torch.zeros(2)}

        # due at step 1, repair still blocked -> nothing yet, but requeued
        assert sched.maybe_consume(step=1, current_samples=current, timeout=0.0) == []

        release.set()
        # picked up on a later consume call at an unrelated step
        deadline = time.time() + 10.0
        pairs = []
        while not pairs and time.time() < deadline:
            pairs = sched.maybe_consume(step=7, current_samples=current, timeout=1.0)
        assert len(pairs) == 1
        _, h_repaired = pairs[0]
        assert torch.allclose(h_repaired, torch.ones(2))
    finally:
        release.set()
        sched.shutdown()


def test_retry_unfinished_false_restores_drop_behavior():
    release = threading.Event()

    def slow_repair(sample_id, snapshot):
        release.wait(timeout=10.0)
        return snapshot

    cfg = SelfConsistencySchedulerConfig(
        every_n_steps=1, sample_frac=1.0, staleness_steps=1, retry_unfinished=False
    )
    sched = SelfConsistencyScheduler(repair_fn=slow_repair, config=cfg)
    try:
        sched.maybe_submit(step=0, epoch=0, samples=[("a", torch.zeros(2))])
        current = {"a": torch.zeros(2)}
        assert sched.maybe_consume(step=1, current_samples=current, timeout=0.0) == []
        release.set()
        # dropped for good: later consume calls never see it
        assert sched.maybe_consume(step=1, current_samples=current, timeout=1.0) == []
        assert sched.maybe_consume(step=2, current_samples=current, timeout=1.0) == []
    finally:
        release.set()
        sched.shutdown()


def test_trainer_requires_explicit_self_consistency_repair_fn_for_now():
    """Until ABACUS block serialization is wired into Trainer, enabling the
    hook from JSON-only config must still fail fast instead of pretending to
    run a real SCF repair path."""
    from dptb.nnops.trainer import Trainer

    with pytest.raises(NotImplementedError, match="self_consistency"):
        Trainer(
            train_options={"self_consistency": {"enabled": True}},
            common_options={}, model=None, train_datasets=None,
        )


def test_self_consistency_argcheck_accepts_payload_hook_options():
    from dptb.utils.argcheck import self_consistency_options

    arg = self_consistency_options()
    normalized = arg.normalize_value(
        {
            "enabled": True, "sample_mode": "payload",
            "tensor_keys": ["node_features", "edge_features"],
            "consume_timeout": 0.25, "max_workers": 1, "retry_unfinished": False,
        }
    )
    arg.check_value(normalized, strict=True)


class _FakeSelfConsistencyScheduler:
    def __init__(self):
        self.submitted = None

    def maybe_consume(self, step, current_samples, timeout=0.0):
        node = current_samples["node_features"]
        return [(node, node.detach() + 1.0)]

    def maybe_submit(self, step, epoch, samples):
        self.submitted = (step, epoch, [(key, value.detach().clone()) for key, value in samples])
        return True


class _FakePayloadSelfConsistencyScheduler:
    def __init__(self):
        self.submitted = None

    def maybe_consume(self, step, current_samples, timeout=0.0):
        payload = current_samples["batch"]
        repaired = {
            "node_features": payload["node_features"].detach() + 1.0,
            "edge_features": payload["edge_features"].detach() + 2.0,
        }
        return [(payload, repaired)]

    def maybe_submit(self, step, epoch, samples):
        self.submitted = (step, epoch, samples)
        return True


def _per_tensor_keys_case():
    return dict(
        scheduler=_FakeSelfConsistencyScheduler(),
        weight=0.25,
        sample_mode=None,
        iteration=11,
        epoch=3,
        pred={
            "node_features": torch.tensor([[1.0, 2.0]], requires_grad=True),
            "edge_features": torch.tensor([[3.0]], requires_grad=True),
        },
        # The fake scheduler's maybe_consume only round-trips node_features
        # (a minimal stand-in, not a real per-key iteration), so only it gets
        # a gradient here; the whole-payload case below repairs both keys.
        grad_keys=("node_features",),
        expected_got=2.25,
        expected_state={
            "train_self_consistency_loss": 1.0,
            "train_self_consistency_weighted_loss": 0.25,
            "train_self_consistency_pairs": 1.0,
            "train_self_consistency_submitted": 1.0,
        },
        check_submitted=lambda submitted: [key for key, _v in submitted]
        == ["node_features", "edge_features"],
    )


def _whole_payload_case():
    return dict(
        scheduler=_FakePayloadSelfConsistencyScheduler(),
        weight=0.5,
        sample_mode="payload",
        iteration=12,
        epoch=4,
        pred={
            "node_features": torch.tensor([1.0, 3.0], requires_grad=True),
            "edge_features": torch.tensor([2.0], requires_grad=True),
            "atomic_numbers": torch.tensor([6]),
        },
        grad_keys=("node_features", "edge_features"),
        expected_got=3.25,
        expected_state={"train_self_consistency_loss": 2.5},
        check_submitted=lambda submitted: submitted[0][0] == "batch",
    )


@pytest.mark.parametrize(
    "make_case", [_per_tensor_keys_case, _whole_payload_case],
    ids=["per_tensor_keys", "whole_payload"],
)
def test_trainer_self_consistency_hook_adds_weighted_loss_and_resubmits(make_case):
    from dptb.nnops.trainer import Trainer

    case = make_case()
    trainer = Trainer.__new__(Trainer)
    trainer.self_consistency_enabled = True
    trainer.self_consistency_weight = case["weight"]
    trainer.self_consistency_tensor_keys = ("node_features", "edge_features")
    trainer.self_consistency_consume_timeout = 0.0
    trainer.self_consistency_scheduler = case["scheduler"]
    if case["sample_mode"]:
        trainer.self_consistency_sample_mode = case["sample_mode"]
    trainer.iter = case["iteration"]
    trainer.ep = case["epoch"]
    trainer._last_self_consistency_state = {}

    base_loss = torch.tensor(2.0, requires_grad=True)
    pred = case["pred"]
    got = trainer._apply_self_consistency_loss(base_loss, pred)
    got.backward()

    assert got.item() == pytest.approx(case["expected_got"])
    for key in case["grad_keys"]:
        assert pred[key].grad is not None
    for state_key, expected_value in case["expected_state"].items():
        assert trainer._last_self_consistency_state[state_key].item() == pytest.approx(
            expected_value
        )
    step, epoch, submitted = trainer.self_consistency_scheduler.submitted
    assert (step, epoch) == (case["iteration"], case["epoch"])
    assert case["check_submitted"](submitted)


# ---------------------------------------------------------------------------
# sc_guard: ABACUS log parsing, residual units, and the accept/reject guard
# (formerly test_sc_guard.py; absorbed here). Calibration fixtures are the
# measured values from the 2026-07-03 production repair-line test (998933 CFM
# iter100000, three large-gap SOC cases):
#   case_0154 residual_mean 2.7e-4 eV (healthy, floor-limited)  -> reject (below gain floor)
#   case_0193 residual_mean 1.9e-3 eV (sick, inside basin)      -> accept
#   case_0008 residual_mean 7.0e-2 eV (outside basin, blew up)  -> reject
# ---------------------------------------------------------------------------

# Verbatim row shapes from ABACUS running_scf.log final-energy tables; each
# row carries the SAME energy in (Ry, eV) columns -- the parser must take the
# eV column and derive the Harris/KS gap across rows, not across columns.
_LOG_0008 = """
  E_KohnSham     -380.3861913450      -5175.4196428107
  E_Harris       -355.5499559313      -4837.5053243140
  E_Fermi        -0.1106552797        -1.5055423183
"""

_LOG_0154 = """
  E_KohnSham     -286.9922200589      -3904.7294744705
  E_Harris       -286.9760728400      -3904.5097802873
  E_Fermi        0.3681094976         5.0083866547
"""


@requires_module("dftio", reason="hrebuild's guard fixtures exercise dptb.postprocess.hrebuild")
@pytest.mark.parametrize(
    "text,expected_gap",
    [
        (_LOG_0008, pytest.approx(337.914, abs=1e-2)),
        (_LOG_0154, pytest.approx(0.2197, abs=1e-3)),
        ("no energies here", None),
        (_LOG_0154 + "\n" + _LOG_0008, pytest.approx(337.914, abs=1e-2)),  # takes the LAST occurrence
    ],
    ids=["ev_column_and_row_gap", "second_case_values", "missing_rows", "takes_last_occurrence"],
)
def test_parse_scf_energies_ev(text, expected_gap):
    from dptb.postprocess.hrebuild import parse_scf_energies_ev

    energies = parse_scf_energies_ev(text)
    assert energies["harris_ks_gap_ev"] == expected_gap
    if text is _LOG_0008:
        assert energies["e_kohnsham_ev"] == pytest.approx(-5175.4196428107)
        assert energies["e_harris_ev"] == pytest.approx(-4837.5053243140)
    elif expected_gap is None:
        assert energies["e_kohnsham_ev"] is None


@requires_module("dftio", reason="hrebuild's guard fixtures exercise dptb.postprocess.hrebuild")
def test_self_consistency_residual_units_and_common_keys():
    from dptb.postprocess.hrebuild import self_consistency_residual

    a = {"0_0_0_0_0": np.eye(2), "0_1_0_0_0": np.zeros((2, 2))}
    b = {"0_0_0_0_0": np.eye(2) + 0.01, "1_1_0_0_0": np.eye(2)}
    r = self_consistency_residual(a, b, unit="eV")
    assert r["n_common"] == 1
    assert r["residual_mean_ev"] == pytest.approx(0.01)
    # Hartree-unit blocks are converted to eV
    r_ha = self_consistency_residual(a, b, unit="Ha")
    assert r_ha["residual_mean_ev"] == pytest.approx(0.01 * 27.211386245988)

    r_none = self_consistency_residual({"a": np.eye(1)}, {"b": np.eye(1)})
    assert r_none["n_common"] == 0 and np.isnan(r_none["residual_mean_ev"])


@requires_module("dftio", reason="hrebuild's guard fixtures exercise dptb.postprocess.hrebuild")
@pytest.mark.parametrize(
    "residual,expect_ok",
    [
        (2.7e-4, False),  # case_0154: repair gain below floor
        (1.9e-3, True),   # case_0193: inside basin -> accept
        (7.0e-2, False),  # case_0008: outside basin -> reject
        (float("nan"), False),
    ],
    ids=["below_gain_floor", "inside_basin_accepts", "outside_basin_rejects", "nan_rejects"],
)
def test_guard_calibration_accepts_and_rejects(residual, expect_ok):
    from dptb.postprocess.hrebuild import SCGuardConfig, evaluate_sc_guard

    ok, reason = evaluate_sc_guard(residual, SCGuardConfig())
    assert ok is expect_ok
    if not expect_ok:
        assert reason


@requires_module("dftio", reason="hrebuild's guard fixtures exercise dptb.postprocess.hrebuild")
def test_coerce_sc_guard_forms():
    from dptb.postprocess.hrebuild import SCGuardConfig, _coerce_sc_guard

    assert _coerce_sc_guard(None) is None
    assert _coerce_sc_guard(False) is None
    assert isinstance(_coerce_sc_guard(True), SCGuardConfig)
    cfg = _coerce_sc_guard({"max_residual_mean_ev": 0.5})
    assert cfg.max_residual_mean_ev == 0.5
    with pytest.raises(TypeError):
        _coerce_sc_guard(3.14)
