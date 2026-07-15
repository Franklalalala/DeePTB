"""OOM-decision unit tests for :class:`DynamicBatchController`.

These exercise the *decision* surface of the runtime dynamic-batch OOM fallback
in isolation (no CUDA, no distributed, no display flush): OOM detection, the
expert-DP consensus gate, the ``can_skip_after_oom`` boolean, config-time
auto-disable, and the runtime ``max_cost`` shrink arithmetic. A lightweight fake
trainer stands in for ``MultiTrainer`` so the controller's ``self._t`` reads are
fully observable.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.dynamic_batch_controller import DynamicBatchController


class _FakeSampler:
    def __init__(self, max_cost):
        self.max_cost = max_cost


class _FakeLoader:
    def __init__(self, max_cost):
        self.batch_sampler = _FakeSampler(max_cost)
        self.dynamic_batch_options = {"max_cost": max_cost}
        self.invalidated = 0

    def invalidate_dynamic_batch_cache(self, clear_costs=False):
        self.invalidated += 1


def _fake_trainer(**overrides):
    t = SimpleNamespace()
    t.dynamic_batch_enabled = True
    t.dynamic_batch_oom_fallback = True
    t.dynamic_batch_oom_shrink_factor = 0.8
    t.dynamic_batch_oom_skipped_iters = 0
    t.dynamic_batch_oom_skipped_since_display = 0
    t.dynamic_batch_options = {"oom_fallback": True, "oom_shrink_factor": 0.8}
    t.distributed_rank0_prepare_batch = False
    t.distributed_expert = False
    t.expert_data_parallel_size = 1
    t.optimizers = []
    t.train_loader = _FakeLoader(100)
    t._is_cuda_device = lambda: False
    # only used by shrink logging; return a benign log-values dict
    t._dynamic_batch_oom_log_values = lambda batch: {
        "batch_cost": 42, "batch_max_item_cost": 5, "num_graphs": 3,
    }
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def _ctl(**overrides):
    return DynamicBatchController(_fake_trainer(**overrides))


# --------------------------------------------------------------------------
# OOM detection
# --------------------------------------------------------------------------

def test_is_cuda_oom_matches_message_and_type():
    assert DynamicBatchController.is_cuda_oom(RuntimeError("CUDA out of memory. Tried..."))
    assert DynamicBatchController.is_cuda_oom(RuntimeError("out Of Memory"))
    assert not DynamicBatchController.is_cuda_oom(RuntimeError("some other error"))
    assert not DynamicBatchController.is_cuda_oom(ValueError("boom"))


# --------------------------------------------------------------------------
# consensus gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "distributed_expert, dp_size, expected",
    [(False, 1, False), (False, 4, False), (True, 1, False), (True, 2, True)],
)
def test_requires_expert_dp_consensus(distributed_expert, dp_size, expected):
    ctl = _ctl(distributed_expert=distributed_expert, expert_data_parallel_size=dp_size)
    assert ctl.requires_expert_dp_consensus() is expected


# --------------------------------------------------------------------------
# can_skip_after_oom gate (every falsifying condition)
# --------------------------------------------------------------------------

def test_can_skip_after_oom_all_conditions_met():
    assert _ctl().can_skip_after_oom() is True


def test_can_skip_after_oom_blocked_by_each_condition():
    assert _ctl(dynamic_batch_enabled=False).can_skip_after_oom() is False
    assert _ctl(dynamic_batch_oom_fallback=False).can_skip_after_oom() is False
    assert _ctl(distributed_rank0_prepare_batch=True).can_skip_after_oom() is False
    # expert-DP consensus required -> cannot skip
    assert _ctl(distributed_expert=True, expert_data_parallel_size=2).can_skip_after_oom() is False
    # a reference batch present -> cannot skip
    assert _ctl().can_skip_after_oom(ref_batch=object()) is False
    # optimizer step already started -> cannot skip
    assert _ctl().can_skip_after_oom(optimizer_step_started=True) is False


def test_maybe_skip_short_circuits_when_not_local_oom():
    ctl = _ctl()
    # gate passes but local_oom False -> no skip, no state mutation
    assert ctl.maybe_skip_after_oom(object(), local_oom=False, where="probe") is False
    assert ctl._t.dynamic_batch_oom_skipped_iters == 0


def test_maybe_skip_returns_false_when_gate_blocks():
    ctl = _ctl(dynamic_batch_oom_fallback=False)
    assert ctl.maybe_skip_after_oom(object(), local_oom=True, where="probe") is False


# --------------------------------------------------------------------------
# config-time auto-disable
# --------------------------------------------------------------------------

def test_configure_disables_for_bad_shrink_factor():
    t = _fake_trainer(dynamic_batch_options={"oom_fallback": True, "oom_shrink_factor": 1.5})
    DynamicBatchController(t).configure_oom_fallback()
    assert t.dynamic_batch_oom_fallback is False
    assert t.dynamic_batch_options["oom_fallback"] is False


def test_configure_disables_for_rank0_prepare_batch():
    t = _fake_trainer(distributed_rank0_prepare_batch=True)
    DynamicBatchController(t).configure_oom_fallback()
    assert t.dynamic_batch_oom_fallback is False


def test_configure_disables_for_expert_dp_consensus():
    t = _fake_trainer(distributed_expert=True, expert_data_parallel_size=2)
    DynamicBatchController(t).configure_oom_fallback()
    assert t.dynamic_batch_oom_fallback is False


def test_configure_keeps_enabled_in_single_process():
    t = _fake_trainer()
    DynamicBatchController(t).configure_oom_fallback()
    assert t.dynamic_batch_oom_fallback is True
    assert t.dynamic_batch_oom_shrink_factor == pytest.approx(0.8)


# --------------------------------------------------------------------------
# runtime max_cost shrink arithmetic
# --------------------------------------------------------------------------

def test_shrink_after_oom_progresses_and_invalidates():
    ctl = _ctl()
    old, new = ctl.shrink_after_oom(object())
    assert old == 100
    assert new == 80  # floor(100 * 0.8)
    assert ctl._t.train_loader.batch_sampler.max_cost == 80
    assert ctl._t.train_loader.dynamic_batch_options["max_cost"] == 80
    assert ctl._t.train_loader.invalidated == 1
    # fallback stays enabled while progress is possible
    assert ctl._t.dynamic_batch_oom_fallback is True


def test_shrink_after_oom_disables_when_cannot_progress():
    ctl = _ctl(train_loader=_FakeLoader(1))
    old, new = ctl.shrink_after_oom(object())
    assert old == 1 and new == 1
    assert ctl._t.dynamic_batch_oom_fallback is False


def test_shrink_after_oom_disables_when_floor_would_not_progress():
    # shrink_factor so close to 1 that floor(old*factor) == old
    ctl = _ctl(dynamic_batch_oom_shrink_factor=0.999)
    old, new = ctl.shrink_after_oom(object())  # floor(100*0.999)=99 < 100 -> progresses
    assert old == 100 and new == 99
    # but with max_cost=2 and factor 0.999 -> floor(1.998)=1 -> progresses to 1
    ctl2 = _ctl(dynamic_batch_oom_shrink_factor=0.6, train_loader=_FakeLoader(1))
    old2, new2 = ctl2.shrink_after_oom(object())
    assert old2 == 1 and new2 == 1
    assert ctl2._t.dynamic_batch_oom_fallback is False


def test_shrink_after_oom_no_sampler_returns_none():
    loader = SimpleNamespace(batch_sampler=None)
    ctl = _ctl(train_loader=loader)
    assert ctl.shrink_after_oom(object()) == (None, None)


def test_disable_oom_fallback_sets_flags(caplog):
    ctl = _ctl()
    with caplog.at_level(logging.WARNING):
        ctl.disable_oom_fallback("disabled because %s", "reason")
    assert ctl._t.dynamic_batch_oom_fallback is False
    assert ctl._t.dynamic_batch_options["oom_fallback"] is False


def test_record_oom_skip_increments_and_bumps_pack():
    pack = torch.zeros(7)
    ctl = _ctl(_display_window_dynamic_batch_pack_local=pack)
    ctl.record_oom_skip()
    ctl.record_oom_skip()
    assert ctl._t.dynamic_batch_oom_skipped_iters == 2
    assert ctl._t.dynamic_batch_oom_skipped_since_display == 2
    from dptb.nnops.metric_pack import DynamicBatchStat
    assert pack[DynamicBatchStat.index("oom_skipped_count")].item() == 2.0
