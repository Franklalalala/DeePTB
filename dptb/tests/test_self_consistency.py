"""WS4 phase C smoke tests. Uses a fake ``repair_fn`` (no ABACUS
dependency) to lock down the mechanism: loss gradient only flows through
``h_pred``, the submit/consume double-buffering schedule fires on the
right cadence and respects ``staleness_steps``, and a failing/refused
repair degrades to "no loss this round" instead of raising. The real
ABACUS-backed accuracy claims (does L_sc actually improve generalization)
need a full training run and are out of scope here -- see WS4 report.
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
)


def test_loss_gradient_only_flows_through_h_pred():
    h_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h_repaired = torch.tensor([1.5, 2.5, 2.5], requires_grad=True)  # would-be leaf if not detached

    loss = compute_self_consistency_loss(h_pred, h_repaired)
    loss.backward()

    assert h_pred.grad is not None
    assert h_repaired.grad is None  # detached inside compute_self_consistency_loss


def test_loss_masking():
    h_pred = torch.tensor([1.0, 2.0, 3.0])
    h_repaired = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss = compute_self_consistency_loss(h_pred, h_repaired, mask=mask)
    # only the first two elements contribute: (1^2+2^2)/2 = 2.5
    assert torch.isclose(loss, torch.tensor(2.5))


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


def test_loss_mask_broadcasts_with_per_element_denominator():
    """A per-sample mask (leading dims) must broadcast against per-element
    diffs, and the denominator must count active ELEMENTS after broadcast so
    masked and unmasked paths agree on scale (per-element mean)."""
    h_pred = torch.ones(2, 3)
    h_repaired = torch.zeros(2, 3)
    mask = torch.tensor([1.0, 0.0])  # sample 0 active, sample 1 masked out

    loss = compute_self_consistency_loss(h_pred, h_repaired, mask=mask)
    # active region = 3 elements, each with squared error 1 -> mean = 1.0
    # (a per-sample denominator would wrongly give 3.0)
    assert torch.isclose(loss, torch.tensor(1.0))


def _fake_repair_fn_success(sample_id, h_pred_snapshot):
    time.sleep(0.05)
    return h_pred_snapshot + 0.01  # pretend ABACUS nudged it slightly


def _fake_repair_fn_refuses(sample_id, h_pred_snapshot):
    return None  # e.g. gap-threshold guard refused


def test_submit_consume_cadence_and_staleness():
    cfg = SelfConsistencySchedulerConfig(every_n_steps=2, sample_frac=1.0, staleness_steps=1, warmup_epochs=0)
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


def test_trainer_fails_fast_on_unwired_self_consistency():
    """train_options.self_consistency is in the schema but not wired into the
    Trainer loss assembly; enabling it must raise instead of silently training
    without L_sc. The check runs before any model/optimizer construction, so
    placeholder arguments never get touched."""
    from dptb.nnops.trainer import Trainer

    with pytest.raises(NotImplementedError, match="self_consistency"):
        Trainer(
            train_options={"self_consistency": {"enabled": True}},
            common_options={},
            model=None,
            train_datasets=None,
        )
