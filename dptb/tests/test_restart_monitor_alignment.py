from __future__ import annotations

import pytest
import torch


def test_single_trainer_restart_builds_with_merged_common_options(monkeypatch):
    import dptb.nnops.trainer as trainer_mod

    checkpoint = {
        "config": {
            "train_options": {},
            "model_options": {"embedding": {}, "prediction": {}},
            "common_options": {
                "device": "cuda:7",
                "dtype": "float32",
                "basis": {"H": "1s"},
                "overlap": False,
                "seed": 17,
            },
        },
        "model_state_dict": {},
        "epoch": 2,
        "iteration": 11,
        "stats": {},
    }
    captured = {}

    monkeypatch.setattr(trainer_mod.torch, "load", lambda *args, **kwargs: checkpoint)

    def fake_build_model(checkpoint_path, model_options, common_options, **kwargs):
        captured["checkpoint"] = checkpoint_path
        captured["common_options"] = common_options
        return object()

    monkeypatch.setattr(trainer_mod, "build_model", fake_build_model)

    class ProbeTrainer(trainer_mod.Trainer):
        def __init__(self, *, common_options, **kwargs):
            self.received_common_options = common_options
            self.plugin_queues = {}

    trainer = ProbeTrainer.restart(
        "dummy.pth",
        train_datasets=object(),
        train_options={},
        common_options={"device": "cpu", "dtype": "float64"},
    )

    assert captured["common_options"]["device"] == "cpu"
    assert captured["common_options"]["dtype"] == "float64"
    assert captured["common_options"]["basis"] == {"H": "1s"}
    assert captured["common_options"]["seed"] == 17
    assert trainer.received_common_options == captured["common_options"]


def test_nnenv_from_reference_preserves_partial_model_and_runtime_overrides(monkeypatch):
    import dptb.nn.deeptb as deeptb_mod

    checkpoint = {
        "config": {
            "model_options": {
                "embedding": {"method": "checkpoint_embedding"},
                "prediction": {"method": "e3tb"},
            },
            "common_options": {
                "device": "cuda:3",
                "dtype": "float32",
                "basis": {"C": "2s2p"},
                "overlap": True,
                "has_soc": False,
            },
        },
        "model_state_dict": {},
    }
    monkeypatch.setattr(deeptb_mod.torch, "load", lambda *args, **kwargs: checkpoint)

    class ProbeNNENV(deeptb_mod.NNENV):
        def __init__(self, **kwargs):
            torch.nn.Module.__init__(self)
            self.received = kwargs

    model = ProbeNNENV.from_reference(
        "dummy.pth",
        embedding={"method": "explicit_embedding"},
        prediction={},
        device="cpu",
        dtype="float64",
        basis={"H": "1s"},
        overlap=False,
        has_soc=True,
    )

    assert model.received["embedding"] == {"method": "explicit_embedding"}
    assert model.received["prediction"] == {"method": "e3tb"}
    assert model.received["device"] == "cpu"
    assert model.received["dtype"] == "float64"
    assert model.received["basis"] == {"H": "1s"}
    assert model.received["overlap"] is False
    assert model.received["has_soc"] is True


def test_validationer_records_primary_endpoint_once_and_preserves_epoch_metadata():
    from dptb.plugins.monitor import Validationer

    class DummyTrainer:
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

    trainer = DummyTrainer()
    monitor = Validationer(interval=[(1, "iteration"), (1, "epoch")])
    monitor.register(trainer)
    monitor.iteration(field="iteration", time=1)

    assert trainer.stats["validation_loss"]["epoch_stats"] == (3.0, 1)
    assert trainer.stats["validation_onsite_loss"]["epoch_stats"] == (1.0, 1)
    assert trainer.stats["validation_hopping_loss"]["epoch_stats"] == (5.0, 1)

    trainer.validation_loss = 7.0
    monitor.epoch(time=2)
    assert trainer.stats["validation_loss"]["last"] == 7.0
    assert trainer.stats["validation_loss"]["last_updated"] == 2
    assert trainer.stats["validation_loss"]["running_avg"] == pytest.approx(2.73)
    assert trainer.stats["validation_loss"]["epoch_mean"] == 7.0


def test_core_training_monitors_share_the_full_iteration_population():
    from dptb.plugins.training_monitor import register_core_training_monitors

    class DummyTrainer:
        def __init__(self):
            self.plugins = []

        def register_plugin(self, plugin):
            self.plugins.append(plugin)

    trainer = DummyTrainer()
    register_core_training_monitors(
        trainer,
        train_endpoint_capable=True,
        sliding_win_size=7,
        avg_per_iter=False,
    )

    by_name = {plugin.stat_name: plugin for plugin in trainer.plugins}
    assert set(by_name) == {
        "train_loss",
        "lr",
        "train_loss_opt",
        "total_grad_norm",
        "train_onsite_loss",
        "train_hopping_loss",
        "mean_max_prob",
        "expert_load_cv",
    }
    for plugin in by_name.values():
        assert plugin.trigger_interval == [(1, "iteration"), (1, "epoch")]
