"""Dynamic-batch OOM fallback: the DynamicBatchController decision surface
(OOM detection, expert-DP consensus gate, can_skip_after_oom, config-time
auto-disable, runtime max_cost shrink) plus MultiTrainer's real iteration()
wiring of that fallback and the expert-DP dynamic-batch loader sharding rule.
"""
import logging
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.dynamic_batch_controller import DynamicBatchController
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.tests._trainer_probes import DistPathProbeTrainer, StubBatch


# ===========================================================================
# DynamicBatchController: the OOM-fallback decision surface in isolation (no
# CUDA, no distributed, no display flush). A lightweight fake trainer stands
# in for MultiTrainer so the controller's self._t reads are fully observable.
# ===========================================================================
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
    t._dynamic_batch_oom_log_values = lambda batch: {"batch_cost": 42, "batch_max_item_cost": 5, "num_graphs": 3}
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def _ctl(**overrides):
    return DynamicBatchController(_fake_trainer(**overrides))


def test_is_cuda_oom_matches_message_and_type():
    assert DynamicBatchController.is_cuda_oom(RuntimeError("CUDA out of memory. Tried..."))
    assert DynamicBatchController.is_cuda_oom(RuntimeError("out Of Memory"))
    assert not DynamicBatchController.is_cuda_oom(RuntimeError("some other error"))
    assert not DynamicBatchController.is_cuda_oom(ValueError("boom"))


@pytest.mark.parametrize(
    "distributed_expert, dp_size, expected",
    [(False, 1, False), (False, 4, False), (True, 1, False), (True, 2, True)],
)
def test_requires_expert_dp_consensus(distributed_expert, dp_size, expected):
    ctl = _ctl(distributed_expert=distributed_expert, expert_data_parallel_size=dp_size)
    assert ctl.requires_expert_dp_consensus() is expected


@pytest.mark.parametrize(
    ("overrides", "ref_batch", "optimizer_step_started", "expected"),
    [
        ({}, None, False, True),
        ({"dynamic_batch_enabled": False}, None, False, False),
        ({"dynamic_batch_oom_fallback": False}, None, False, False),
        ({"distributed_rank0_prepare_batch": True}, None, False, False),
        ({"distributed_expert": True, "expert_data_parallel_size": 2}, None, False, False),
        ({}, "some_ref_batch", False, False),
        ({}, None, True, False),
    ],
    ids=["all_conditions_met", "dynamic_batch_disabled", "oom_fallback_disabled", "rank0_prepare_batch",
        "expert_dp_consensus_required", "reference_batch_present", "optimizer_step_already_started"],
)
def test_can_skip_after_oom_gate(overrides, ref_batch, optimizer_step_started, expected):
    ctl = _ctl(**overrides)
    if ref_batch == "some_ref_batch":
        ref_batch = object()
    assert ctl.can_skip_after_oom(ref_batch=ref_batch, optimizer_step_started=optimizer_step_started) is expected


def test_maybe_skip_short_circuits_when_not_local_oom():
    ctl = _ctl()
    # gate passes but local_oom False -> no skip, no state mutation
    assert ctl.maybe_skip_after_oom(object(), local_oom=False, where="probe") is False
    assert ctl._t.dynamic_batch_oom_skipped_iters == 0


def test_maybe_skip_returns_false_when_gate_blocks():
    ctl = _ctl(dynamic_batch_oom_fallback=False)
    assert ctl.maybe_skip_after_oom(object(), local_oom=True, where="probe") is False


@pytest.mark.parametrize(
    ("overrides", "expect_enabled", "expect_shrink_factor"),
    [
        ({"dynamic_batch_options": {"oom_fallback": True, "oom_shrink_factor": 1.5}}, False, None),
        ({"distributed_rank0_prepare_batch": True}, False, None),
        ({"distributed_expert": True, "expert_data_parallel_size": 2}, False, None),
        ({}, True, 0.8),
    ],
    ids=["bad_shrink_factor", "rank0_prepare_batch", "expert_dp_consensus", "single_process_stays_enabled"],
)
def test_configure_oom_fallback_auto_disables(overrides, expect_enabled, expect_shrink_factor):
    t = _fake_trainer(**overrides)
    DynamicBatchController(t).configure_oom_fallback()
    assert t.dynamic_batch_oom_fallback is expect_enabled
    assert t.dynamic_batch_options["oom_fallback"] is expect_enabled
    if expect_shrink_factor is not None:
        assert t.dynamic_batch_oom_shrink_factor == pytest.approx(expect_shrink_factor)


@pytest.mark.parametrize(
    ("overrides", "expect_new_cost", "expect_fallback_after"),
    [
        ({}, 80, True),  # floor(100 * 0.8) progresses
        ({"train_loader": None}, 1, False),  # placeholder, replaced below (max_cost already at the floor)
        ({"train_loader": None, "dynamic_batch_oom_shrink_factor": 0.6}, 1, False),  # tiny cost still floors to 1
    ],
    ids=["progresses_and_invalidates", "disables_when_cannot_progress", "disables_when_floor_stays_at_one"],
)
def test_shrink_after_oom_progresses_or_disables(overrides, expect_new_cost, expect_fallback_after):
    if "train_loader" in overrides and overrides["train_loader"] is None:
        overrides["train_loader"] = _FakeLoader(1)
    ctl = _ctl(**overrides)
    old_cost = ctl._t.train_loader.batch_sampler.max_cost
    old, new = ctl.shrink_after_oom(object())
    assert old == old_cost
    assert new == expect_new_cost
    assert ctl._t.dynamic_batch_oom_fallback is expect_fallback_after
    if expect_fallback_after:
        assert ctl._t.train_loader.batch_sampler.max_cost == expect_new_cost
        assert ctl._t.train_loader.dynamic_batch_options["max_cost"] == expect_new_cost
        assert ctl._t.train_loader.invalidated == 1


def test_shrink_after_oom_no_sampler_returns_none():
    loader = SimpleNamespace(batch_sampler=None)
    ctl = _ctl(train_loader=loader)
    assert ctl.shrink_after_oom(object()) == (None, None)


def test_record_oom_skip_increments_and_bumps_pack():
    pack = torch.zeros(7)
    ctl = _ctl(_display_window_dynamic_batch_pack_local=pack)
    ctl.record_oom_skip()
    ctl.record_oom_skip()
    assert ctl._t.dynamic_batch_oom_skipped_iters == 2
    assert ctl._t.dynamic_batch_oom_skipped_since_display == 2
    from dptb.nnops.metric_pack import DynamicBatchStat
    assert pack[DynamicBatchStat.index("oom_skipped_count")].item() == 2.0


# ===========================================================================
# MultiTrainer.iteration(): the real dynamic-batch OOM fallback wiring
# ===========================================================================
def _oom_probe_trainer(*, max_cost=100, num_experts=1, distributed_expert=False):
    trainer = DistPathProbeTrainer(num_experts=num_experts)
    trainer.distributed_expert = distributed_expert
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True
    trainer.dynamic_batch_oom_shrink_factor = 0.8
    trainer.dynamic_batch_oom_skipped_iters = 0
    trainer.train_loader = SimpleNamespace(
        batch_sampler=SimpleNamespace(max_cost=max_cost),
        dynamic_batch_options={"max_cost": max_cost},
        invalidate_dynamic_batch_cache=lambda clear_costs=False: None,
    )
    return trainer


def _cost_payload(weight, cost):
    loss = weight * cost
    return {
        "loss": loss, "expert_onsite": loss.detach(), "expert_hopping": loss.detach(),
        "onsite_weighted_sum": loss.detach(), "hopping_weighted_sum": loss.detach(),
        "active_nodes": torch.tensor(1.0), "active_edges": torch.tensor(1.0),
        "onsite_l1_sum": None, "onsite_mse_sum": None, "onsite_cnt": None,
        "hopping_l1_sum": None, "hopping_mse_sum": None, "hopping_cnt": None,
        "z_values": [], "load_cv_values": [],
    }


def _cost_batch(cost):
    return StubBatch(cost)  # __dptb_batch_cost__ == 10 + cost; only the attribute's presence matters here


@pytest.mark.parametrize(
    ("distributed_expert", "failure_stage", "expect_iter_advances"),
    [(False, "prepare", False), (False, "forward", False), (True, "forward", True)],
    ids=["single_process_prepare_failure", "single_process_forward_failure", "distributed_forward_failure_no_sync"],
)
def test_oom_fallback_skips_the_batch_without_a_step_or_a_collective(monkeypatch, distributed_expert, failure_stage,
                                                                      expect_iter_advances):
    trainer = _oom_probe_trainer(distributed_expert=distributed_expert)
    weight = trainer.model.experts[0].weight
    batch = _cost_batch(0)

    def _unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("the local OOM fallback path must not call all_reduce")

    monkeypatch.setattr("dptb.nnops.multi_trainer.dist.all_reduce", _unexpected_all_reduce)

    if failure_stage == "prepare":
        calls = {"count": 0}

        def _prepare(batch, with_lengths=True):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("CUDA out of memory")
            return {"cost": torch.tensor(1.0)}, {}

        trainer._prepare_batch_bundle = _prepare
        trainer._build_train_payload = lambda *, batch_dict, batch_info, expert_idx, range_dis, **kw: (
            _cost_payload(weight, batch_dict["cost"])
        )
    else:
        trainer._prepare_batch_bundle = lambda batch, with_lengths=True: ({"cost": torch.tensor(1.0)}, {})
        trainer._build_train_payload = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory during forward")
        )

    before_iter = trainer.iter
    loss = trainer.iteration(batch)

    assert loss is None
    assert trainer.optimizers[0].step_calls == 0
    # Single process: the skipped batch never advances the optimizer clock.
    # Distributed-expert: all ranks already consumed the batch in lockstep
    # before the local OOM, so the shared iteration counter still advances.
    assert trainer.iter == (before_iter + 1 if expect_iter_advances else before_iter)
    assert trainer.train_loader.batch_sampler.max_cost == 80
    assert trainer.dynamic_batch_oom_skipped_iters == 1


def test_oom_fallback_disabled_reraises_and_never_retries_after_a_step():
    trainer = _oom_probe_trainer()
    trainer.dynamic_batch_oom_fallback = False
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (
        (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        trainer.iteration(_cost_batch(0))
    assert trainer.optimizers[0].step_calls == 0


def test_oom_after_optimizer_step_reraises_instead_of_retrying():
    trainer = _oom_probe_trainer()
    weight = trainer.model.experts[0].weight
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: ({"cost": torch.tensor(1.0)}, {})
    trainer._build_train_payload = lambda *, batch_dict, batch_info, expert_idx, range_dis, **kw: (
        _cost_payload(weight, batch_dict["cost"])
    )
    trainer._local_scheduler_step = lambda metric: (
        (_ for _ in ()).throw(RuntimeError("CUDA out of memory after optimizer"))
    )

    with pytest.raises(RuntimeError, match="out of memory after optimizer"):
        trainer.iteration(_cost_batch(0))
    # the optimizer step already committed before the failure -> must not be retried
    assert trainer.optimizers[0].step_calls == 1


def test_distributed_oom_fallback_disabled_for_expert_dp_replicas():
    trainer = _oom_probe_trainer(distributed_expert=True)
    trainer.expert_data_parallel_size = 2
    assert trainer._can_skip_dynamic_batch_after_oom() is False


# ---------------------------------------------------------------------------
# dynamic_batch_cfg_for_train_loader: expert-DP loader sharding
# ---------------------------------------------------------------------------
def _dynamic_batch_cfg_probe(**overrides):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.dynamic_batch_enabled = overrides.pop("dynamic_batch_enabled", True)
    trainer.dynamic_batch_options = {"enabled": True, "max_cost": 100, "rank": 99, "world_size": 99}
    trainer.dynamic_batch_options.update(overrides.pop("dynamic_batch_options", {}))
    trainer.distributed_expert = overrides.pop("distributed_expert", False)
    trainer.distributed_rank0_prepare_batch = overrides.pop("distributed_rank0_prepare_batch", False)
    trainer.expert_data_parallel_size = overrides.pop("expert_data_parallel_size", 1)
    trainer.expert_dp_rank = overrides.pop("expert_dp_rank", 0)
    assert not overrides
    return trainer._dynamic_batch_cfg_for_train_loader()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"distributed_expert": True, "expert_data_parallel_size": 1}, {"rank": 0, "world_size": 1}),
        ({"distributed_expert": True, "expert_data_parallel_size": 2, "expert_dp_rank": 1},
         {"rank": 1, "world_size": 2}),
        ({"distributed_expert": True, "distributed_rank0_prepare_batch": True,
          "expert_data_parallel_size": 2, "expert_dp_rank": 1}, {"rank": 0, "world_size": 1}),
        ({"dynamic_batch_enabled": False}, None),
    ],
    ids=["expert_parallel_unsharded", "shards_expert_dp_replicas", "rank0_prepare_stays_unsharded", "disabled_returns_none"],
)
def test_dynamic_batch_cfg_for_expert_parallel_layouts(overrides, expected):
    cfg = _dynamic_batch_cfg_probe(**overrides)
    if expected is None:
        assert cfg is None
    else:
        assert cfg["rank"] == expected["rank"]
        assert cfg["world_size"] == expected["world_size"]


def test_dynamic_batch_cfg_ignores_global_dist_by_default(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="dptb.nnops.multi_trainer")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    cfg = _dynamic_batch_cfg_probe(distributed_expert=False)

    assert cfg["rank"] == 0
    assert cfg["world_size"] == 1
    assert any("use_global_dist" in record.message for record in caplog.records)


def test_dynamic_batch_cfg_can_opt_into_global_dist(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    cfg = _dynamic_batch_cfg_probe(distributed_expert=False, dynamic_batch_options={"use_global_dist": True})

    assert cfg["rank"] == 2
    assert cfg["world_size"] == 4
