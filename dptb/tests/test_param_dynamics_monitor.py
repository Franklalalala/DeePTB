import csv

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dptb.plugins.monitor import ParamDynamicsMonitor


class TinyDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.out_node = nn.Linear(2, 1, bias=False)


class DummyTrainer:
    def __init__(self):
        self.model = TinyDynamicsModel()
        self.stats = {}
        self.rank = 0
        self.world_size = 1
        self.num_experts = 0
        self.is_main_process = True
        self.distributed_expert = False


class DistributedExpertTrainer(DummyTrainer):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.experts = nn.ModuleList([TinyDynamicsModel(), TinyDynamicsModel()])
        self.rank = 0
        self.world_size = 2
        self.num_experts = 2
        self.local_expert_idx = 0
        self.distributed_expert = True


def _read_model_rows(path):
    with open(path / "param_dynamics.csv", newline="") as f:
        return [row for row in csv.DictReader(f) if row["group"] == "model"]


def test_param_dynamics_monitor_records_baseline_then_weight_delta(tmp_path):
    trainer = DummyTrainer()
    monitor = ParamDynamicsMonitor(
        str(tmp_path),
        interval=[(1, "iteration")],
        tensorboard=False,
    )
    monitor.register(trainer)

    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    monitor.iteration(time=1)
    baseline_row = _read_model_rows(tmp_path)[-1]

    assert baseline_row["baseline"] == "1"
    assert float(baseline_row["delta_norm"]) == 0.0
    assert float(baseline_row["delta_ratio"]) == 0.0

    with torch.no_grad():
        trainer.model.layers[0].weight.add_(0.5)
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    monitor.iteration(time=2)
    update_row = _read_model_rows(tmp_path)[-1]

    assert update_row["baseline"] == "0"
    assert update_row["status"] == "ACTIVE"
    assert float(update_row["delta_norm"]) > 0.0
    assert float(update_row["delta_ratio"]) > 0.0
    assert float(update_row["delta_nonzero_fraction"]) > 0.0
    assert float(update_row["grad_norm"]) > 0.0


def test_param_dynamics_monitor_marks_dead_after_patience(tmp_path):
    trainer = DummyTrainer()
    monitor = ParamDynamicsMonitor(
        str(tmp_path),
        interval=[(1, "iteration")],
        tensorboard=False,
        dead_patience=2,
    )
    monitor.register(trainer)

    monitor.iteration(time=1)
    monitor.iteration(time=2)
    monitor.iteration(time=3)

    dead_row = _read_model_rows(tmp_path)[-1]

    assert dead_row["baseline"] == "0"
    assert dead_row["status"] == "DEAD"
    assert dead_row["dead"] == "1"
    assert dead_row["dead_streak"] == "2"
    assert float(dead_row["grad_norm"]) == 0.0
    assert float(dead_row["delta_norm"]) == 0.0


def test_param_dynamics_monitor_uses_unique_local_expert_groups(tmp_path):
    trainer = DistributedExpertTrainer()
    monitor = ParamDynamicsMonitor(
        str(tmp_path),
        interval=[(1, "iteration")],
        tensorboard=False,
    )
    monitor.register(trainer)

    group_names = [group["name"] for group in monitor._groups]

    assert len(group_names) == len(set(group_names))
    assert "experts.0" in group_names
    assert all(not name.startswith("experts.1") for name in group_names)
