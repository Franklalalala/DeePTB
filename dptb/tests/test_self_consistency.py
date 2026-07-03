"""WS4 phase C smoke tests. Uses a fake ``repair_fn`` (no ABACUS
dependency) to lock down the mechanism: loss gradient only flows through
``h_pred``, the submit/consume double-buffering schedule fires on the
right cadence and respects ``staleness_steps``, and a failing/refused
repair degrades to "no loss this round" instead of raising. The real
ABACUS-backed accuracy claims (does L_sc actually improve generalization)
need a full training run and are out of scope here -- see WS4 report.
"""
import time

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
    import pytest
    with pytest.raises(ValueError):
        compute_self_consistency_loss(torch.zeros(3), torch.zeros(4))


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
