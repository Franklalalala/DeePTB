"""Training monitors: loss/validation statistics, parameter dynamics, CUDA memory probes."""
from __future__ import annotations

import csv
import logging

import pytest
import torch
from torch import nn

from dptb.plugins.base_plugin import PluginUser
from dptb.plugins.monitor import (
    CUDAMemoryMonitor,
    Monitor,
    ParamDynamicsMonitor,
    PreTPBlockMonitor,
    Validationer,
    _is_tensor_product_module,
    _is_torchscript_module,
)
from dptb.plugins.training_monitor import register_core_training_monitors


# --------------------------------------------------------------------------
# loss and validation statistics
# --------------------------------------------------------------------------
def test_validationer_records_endpoint_components_and_epoch_statistics():
    class ValidatingTrainer:
        def __init__(self):
            self.stats = {}
            self.iter = 1
            self.ep = 1
            self.validation_loss = 3.0
            self._last_flow_validation_state = {}

        def validation(self, fast=True):
            self._last_flow_validation_state = {
                "validation_loss": torch.tensor(self.validation_loss),
                "validation_onsite_loss": torch.tensor(1.0),
                "validation_hopping_loss": torch.tensor(5.0),
            }
            return torch.tensor(self.validation_loss)

    trainer = ValidatingTrainer()
    monitor = Validationer(interval=[(1, "iteration"), (1, "epoch")])
    monitor.register(trainer)
    monitor.iteration(field="iteration", time=1)

    assert trainer.stats["validation_loss"]["epoch_stats"] == (3.0, 1)
    assert trainer.stats["validation_onsite_loss"]["epoch_stats"] == (1.0, 1)
    assert trainer.stats["validation_hopping_loss"]["epoch_stats"] == (5.0, 1)

    trainer.validation_loss = 7.0
    monitor.epoch(time=2)
    stats = trainer.stats["validation_loss"]
    assert stats["last"] == 7.0
    assert stats["last_updated"] == 2
    # exponential running average (factor 0.7) over 3.0 then 7.0
    assert stats["running_avg"] == pytest.approx(0.7 * (0.3 * 3.0) + 0.3 * 7.0)
    assert stats["epoch_mean"] == 7.0


@pytest.mark.parametrize("endpoint_capable", [True, False])
def test_core_training_monitors_share_the_full_iteration_population(endpoint_capable):
    class RegisteringTrainer:
        def __init__(self):
            self.plugins = []

        def register_plugin(self, plugin):
            self.plugins.append(plugin)

    trainer = RegisteringTrainer()
    register_core_training_monitors(
        trainer, train_endpoint_capable=endpoint_capable, sliding_win_size=7, avg_per_iter=False
    )

    names = {plugin.stat_name for plugin in trainer.plugins}
    assert "train_loss" in names
    assert ("train_onsite_loss" in names) is endpoint_capable
    assert ("train_hopping_loss" in names) is endpoint_capable
    for plugin in trainer.plugins:
        assert plugin.trigger_interval == [(1, "iteration"), (1, "epoch")]


def test_restored_monitor_statistics_survive_reregistration():
    class StatsOwner:
        def __init__(self):
            self.stats = {}
            self.iter = 1

    class LossMonitor(Monitor):
        stat_name = "train_loss"

        def _get_value(self, **kwargs):
            return kwargs.get("val")

    owner = StatsOwner()
    restored = {"last": 5.0, "running_avg": 4.2, "epoch_mean": 3.9, "epoch_stats": (11.0, 3)}
    owner.stats["train_loss"] = dict(restored)  # stats restored from a checkpoint
    LossMonitor(running_average=True, epoch_average=True).register(owner)
    assert {key: owner.stats["train_loss"][key] for key in restored} == restored


def test_monitor_csv_is_appended_across_restarts(tmp_path):
    log_dir = str(tmp_path / "mon")
    first = PreTPBlockMonitor(log_dir=log_dir)
    with open(first.csv_path, "a", newline="") as handle:
        csv.writer(handle).writerow([7, "blk", "comp", 0.1, 0.2, 0.5])

    PreTPBlockMonitor(log_dir=log_dir)  # constructed again, as a restart would
    with open(first.csv_path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    assert lines.count(",".join(first.header)) == 1
    assert lines[-1].startswith("7,blk")


# --------------------------------------------------------------------------
# parameter dynamics
# --------------------------------------------------------------------------
class _DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.out_node = nn.Linear(2, 1, bias=False)


class _ModelOwner(PluginUser):
    def __init__(self, model=None):
        super().__init__()
        self.model = model if model is not None else _DynamicsModel()
        self.rank = 0
        self.world_size = 1
        self.num_experts = 0
        self.is_main_process = True
        self.distributed_expert = False


def _rows(path, group="model"):
    with open(path / "param_dynamics.csv", newline="") as handle:
        return [row for row in csv.DictReader(handle) if group is None or row["group"] == group]


def _set_grads(model, value):
    for param in model.parameters():
        param.grad = None if value is None else torch.full_like(param, value)


def _dynamics_monitor(path, **kwargs):
    return ParamDynamicsMonitor(str(path), interval=[(1, "iteration")], tensorboard=False, **kwargs)


def test_param_dynamics_records_a_baseline_then_the_weight_delta(tmp_path):
    owner = _ModelOwner()
    monitor = _dynamics_monitor(tmp_path)
    monitor.register(owner)

    _set_grads(owner.model, 1.0)
    monitor.iteration(time=1)
    baseline = _rows(tmp_path)[-1]
    assert baseline["baseline"] == "1"
    assert float(baseline["delta_norm"]) == 0.0
    assert float(baseline["delta_ratio"]) == 0.0

    with torch.no_grad():
        owner.model.layers[0].weight.add_(0.5)
    _set_grads(owner.model, 1.0)
    monitor.iteration(time=2)
    update = _rows(tmp_path)[-1]
    assert update["baseline"] == "0"
    assert update["status"] == "ACTIVE"
    assert float(update["delta_norm"]) > 0.0
    assert float(update["delta_ratio"]) > 0.0
    assert float(update["delta_nonzero_fraction"]) > 0.0
    assert float(update["grad_norm"]) > 0.0


@pytest.mark.parametrize("weights_move", [False, True])
def test_param_dynamics_marks_missing_gradients_dead_after_patience(tmp_path, weights_move):
    """DEAD is gradient-based: moving weights without gradients are still dead."""
    owner = _ModelOwner()
    monitor = _dynamics_monitor(tmp_path, dead_patience=2)
    monitor.register(owner)

    for step in (1, 2, 3):
        if weights_move:
            with torch.no_grad():
                owner.model.layers[0].weight.add_(0.5)
        monitor.iteration(time=step)

    dead = _rows(tmp_path)[-1]
    assert dead["baseline"] == "0"
    assert dead["status"] == "DEAD"
    assert dead["dead"] == "1"
    assert dead["dead_streak"] == "2"
    assert float(dead["grad_norm"]) == 0.0
    assert (float(dead["delta_norm"]) > 0.0) is weights_move


def test_param_dynamics_logs_only_the_local_expert(tmp_path):
    model = nn.Module()
    model.experts = nn.ModuleList([_DynamicsModel(), _DynamicsModel()])
    owner = _ModelOwner(model)
    owner.world_size = 2
    owner.num_experts = 2
    owner.local_expert_idx = 0
    owner.distributed_expert = True
    monitor = _dynamics_monitor(tmp_path)
    monitor.register(owner)
    monitor.iteration(time=1)

    groups = [row["group"] for row in _rows(tmp_path, group=None)]
    assert len(groups) == len(set(groups))
    assert "experts.0" in groups
    assert not any(name.startswith("experts.1") for name in groups)


def test_param_dynamics_restart_rebaselines_before_resuming_dead_tracking(tmp_path):
    """register() always re-baselines weight tracking, so the first post-restart row
    reports BASELINE; the raw dead_streak state_dict round trip is covered separately
    in test_restart_resume.py."""
    owner = _ModelOwner()
    original = _dynamics_monitor(tmp_path / "before", dead_patience=3)
    owner.register_plugin(original)
    for step in (1, 2, 3):  # baseline, then two gradient-free samples
        owner.call_plugins(queue_name="iteration", time=step)
    assert _rows(tmp_path / "before")[-1]["dead_streak"] == "2"

    restarted = _ModelOwner(owner.model)
    restarted._restored_plugin_state = owner.harvest_plugin_states()  # what restart() installs
    restarted.register_plugin(_dynamics_monitor(tmp_path / "after", dead_patience=3))
    restarted.call_plugins(queue_name="iteration", time=4)

    after = _rows(tmp_path / "after")[-1]
    assert after["status"] == "BASELINE"
    assert after["dead_streak"] == "0"


# --------------------------------------------------------------------------
# CUDA memory monitors
# --------------------------------------------------------------------------
def test_cuda_memory_monitor_tracks_global_and_expert_peaks():
    owner = PluginUser()
    owner.num_experts = 2
    monitor = CUDAMemoryMonitor(interval=[(1, "iteration"), (1, "epoch")])
    monitor.register(owner)

    monitor.iteration(cuda_peak_allocated_mb=100.0, cuda_peak_reserved_mb=120.0,
                      expert_0_cuda_peak_allocated_mb=90.0, expert_1_cuda_peak_allocated_mb=80.0,
                      train_loss=1.5)
    monitor.iteration(cuda_peak_allocated_mb=85.0, cuda_peak_reserved_mb=130.0,
                      expert_0_cuda_peak_allocated_mb=95.0)
    monitor.epoch()

    stats = owner.stats
    assert stats["cuda_peak_allocated_mb"]["last"] == 85.0
    assert stats["cuda_peak_allocated_mb"]["max"] == 100.0
    assert stats["cuda_peak_allocated_mb"]["epoch_max"] == 100.0
    assert stats["cuda_peak_reserved_mb"]["max"] == 130.0
    assert stats["expert_0_cuda_peak_allocated_mb"]["max"] == 95.0
    assert stats["expert_1_cuda_peak_allocated_mb"]["max"] == 80.0
    assert "train_loss" not in stats  # non-memory fields are ignored
    assert stats["cuda_peak_allocated_mb"]["log_unit"] == "MB"

    monitor.iteration(cuda_peak_allocated_mb=70.0)
    monitor.epoch()
    assert stats["cuda_peak_allocated_mb"]["epoch_max"] == 70.0
    assert stats["cuda_peak_allocated_mb"]["max"] == 100.0


def test_cuda_memory_epoch_peak_survives_restart():
    owner = PluginUser()
    owner.register_plugin(CUDAMemoryMonitor())
    owner.call_plugins(queue_name="iteration", time=1, cuda_peak_allocated_mb=321.0)

    restarted = PluginUser()
    restarted._restored_plugin_state = owner.harvest_plugin_states()  # what restart() installs
    restarted.register_plugin(CUDAMemoryMonitor())
    restarted.call_plugins(queue_name="iteration", time=2, cuda_peak_allocated_mb=100.0)
    restarted.call_plugins(queue_name="epoch", time=1)

    assert restarted.stats["cuda_peak_allocated_mb"]["epoch_max"] == pytest.approx(321.0)


def _module_type(name, module_name):
    return type(name, (nn.Module,), {
        "__module__": module_name,
        "__init__": lambda self: nn.Module.__init__(self),
    })


def test_module_memory_hooks_select_tensor_products_but_not_torchscript():
    e3nn_tp = _module_type("FullyConnectedTensorProduct", "e3nn.o3._tensor_product._tensor_product")()
    scripted = _module_type("RecursiveScriptModule", "torch.jit._script")()

    assert _is_tensor_product_module(e3nn_tp)
    assert _is_torchscript_module(scripted)
    assert not _is_tensor_product_module(scripted)


@pytest.fixture
def cache_probe(monkeypatch):
    """The process-local cache-memory probe with its configuration restored afterwards."""
    from dptb.utils import cuda_cache_memory as probe

    monkeypatch.setattr(probe, "_CONFIG", dict(probe._CONFIG))
    probe.reset_cuda_cache_event_stats()
    yield probe
    probe.reset_cuda_cache_event_stats()


def _no_snapshot(device=None):
    raise AssertionError("the CUDA allocator must not be queried")


def test_disabled_cache_memory_probe_takes_no_snapshot(cache_probe, monkeypatch):
    cache_probe.configure_cuda_cache_memory_monitor(enabled=False)
    monkeypatch.setattr(cache_probe, "snapshot_cuda_memory", _no_snapshot)

    with cache_probe.cuda_cache_memory_probe("cueq_indexed_linear", ("key",), device="cuda:0"):
        pass


def test_enabled_cache_memory_probe_logs_memory_deltas(cache_probe, monkeypatch, caplog):
    cache_probe.configure_cuda_cache_memory_monitor(enabled=True, min_delta_mb=0.0)
    snapshots = iter([
        {"allocated_mb": 10.0, "reserved_mb": 20.0, "peak_allocated_mb": 30.0,
         "peak_reserved_mb": 40.0, "free_mb": 1000.0, "total_mb": 2000.0},
        {"allocated_mb": 15.5, "reserved_mb": 28.0, "peak_allocated_mb": 35.0,
         "peak_reserved_mb": 48.0, "free_mb": 990.0, "total_mb": 2000.0},
    ])
    monkeypatch.setattr(cache_probe, "snapshot_cuda_memory", lambda device=None: next(snapshots))

    with caplog.at_level(logging.INFO):
        with cache_probe.cuda_cache_memory_context(iteration=42, stage="expert/model_forward", expert=7):
            with cache_probe.cuda_cache_memory_probe(
                "cueq_indexed_linear", (16, "torch.float32", "cuda:1"),
                device="cuda:1", metadata={"local_entries": 2},
            ):
                pass

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[CUDA_CACHE_MEMORY]" in messages
    assert "allocated_delta_mb=5.5" in messages
    assert "reserved_delta_mb=8.0" in messages
    assert "free_delta_mb=-10.0" in messages


def test_cache_event_monitor_counts_hits_and_misses_without_cuda(cache_probe, monkeypatch, caplog):
    cache_probe.configure_cuda_cache_memory_monitor(enabled=False, event_enabled=True,
                                                    event_summary_interval=2)
    monkeypatch.setattr(cache_probe, "snapshot_cuda_memory", _no_snapshot)
    metadata = {"num_graphs": 16, "in_features": 64, "out_features": 64}

    with caplog.at_level(logging.INFO):
        for event in ("miss", "hit"):
            cache_probe.record_cuda_cache_event(
                "cueq_indexed_linear", (16, "torch.float32", "cuda:0", 64, 64), event, metadata=metadata
            )

    assert any("[CUDA_CACHE_EVENT]" in record.getMessage() for record in caplog.records)
    [stats] = cache_probe.cuda_cache_event_stats_snapshot().values()
    assert (stats["cache"], stats["total"], stats["hits"], stats["misses"]) == (
        "cueq_indexed_linear", 2, 1, 1,
    )
