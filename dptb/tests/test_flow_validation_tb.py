from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dptb.plugins.monitor import CleanCompatibleTensorBoardMonitor, TensorBoardMonitor, Validationer


class _FlowValidationTrainer:
    def __init__(self):
        self.stats = {}
        self.ep = 7
        self.iter = 70
        self._last_flow_validation_state = {}

    def validation(self, fast=True):
        self._last_flow_validation_state = {
            "validation_loss": torch.tensor(2.0),
            "validation_onsite_loss": torch.tensor(1.5),
            "validation_hopping_loss": torch.tensor(0.5),
            "validation_flow_random_t_loss": torch.tensor(9.0),
        }
        return torch.tensor(9.0)


def test_flow_validation_legacy_total_uses_compatible_loss_in_stats_and_tensorboard():
    trainer = _FlowValidationTrainer()
    validationer = Validationer(interval=[(1, "epoch")], fast_mode=False)
    validationer.register(trainer)

    validationer.epoch(time=trainer.ep)

    assert trainer.stats["validation_loss"]["epoch_mean"] == pytest.approx(2.0)
    assert trainer.stats["validation_flow_random_t_loss"]["epoch_mean"] == pytest.approx(9.0)

    recorded = {}
    tensorboard = object.__new__(TensorBoardMonitor)
    tensorboard.trainer = trainer
    tensorboard.writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: recorded.setdefault(tag, (value, step)),
        flush=lambda: None,
    )

    tensorboard.epoch(time=trainer.ep)

    assert recorded["validation_loss_iter"] == pytest.approx((2.0, trainer.iter))
    assert recorded["validation_onsite_loss"] == pytest.approx((1.5, trainer.iter))
    assert recorded["validation_hopping_loss"] == pytest.approx((0.5, trainer.iter))
    assert recorded["validation_flow_random_t_loss_epoch"] == pytest.approx(
        (9.0, trainer.ep)
    )


def test_tensorboard_iteration_uses_canonical_tag_names_without_suffix_groups():
    trainer = SimpleNamespace(
        iter=11,
        ep=3,
        num_experts=1,
        stats={
            "train_loss": {"last": 1.0},
            "train_onsite_loss": {"last": 2.0},
            "train_hopping_loss": {"last": 3.0},
            "train_flow_loss": {"last": 4.0},
            "expert_0_lr": {"last": 0.125},
        },
    )
    recorded = {}
    tensorboard = object.__new__(TensorBoardMonitor)
    tensorboard.trainer = trainer
    tensorboard.flush_every = 0
    tensorboard.writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: recorded.setdefault(tag, (value, step)),
        flush=lambda: None,
    )

    tensorboard.iteration(time=trainer.iter)

    assert recorded["train_loss_iter"] == pytest.approx((1.0, trainer.iter))
    assert recorded["train_onsite_loss_iter"] == pytest.approx((2.0, trainer.iter))
    assert recorded["train_hopping_loss_iter"] == pytest.approx((3.0, trainer.iter))
    assert recorded["train_flow_loss_iter"] == pytest.approx((4.0, trainer.iter))
    assert recorded["Expert_LR_Iter/Expert_0"] == pytest.approx((0.125, trainer.iter))
    assert "train_loss_iter/iteration" not in recorded


def test_clean_compatible_tensorboard_writes_train_and_validation_canonical_tags():
    trainer = SimpleNamespace(
        iter=20,
        ep=4,
        num_experts=0,
        stats={
            "lr": {"last": 0.01},
            "train_loss": {"last": 1.0},
            "train_onsite_loss": {"last": 2.0},
            "train_hopping_loss": {"last": 3.0},
            "validation_loss": {"epoch_mean": 4.0, "epoch_last_updated": 4},
            "validation_onsite_loss": {"epoch_mean": 5.0, "epoch_last_updated": 4},
            "validation_hopping_loss": {"epoch_mean": 6.0, "epoch_last_updated": 4},
        },
    )
    recorded = {}
    tensorboard = object.__new__(CleanCompatibleTensorBoardMonitor)
    tensorboard.trainer = trainer
    tensorboard.flush_every = 0
    tensorboard.writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: recorded.setdefault(tag, (value, step)),
        flush=lambda: None,
    )

    tensorboard.iteration(time=trainer.iter)
    tensorboard.epoch(time=trainer.ep)

    assert recorded["lr_iter"] == pytest.approx((0.01, trainer.iter))
    assert recorded["train_loss_iter"] == pytest.approx((1.0, trainer.iter))
    assert recorded["train_onsite_loss_iter"] == pytest.approx((2.0, trainer.iter))
    assert recorded["train_hopping_loss_iter"] == pytest.approx((3.0, trainer.iter))
    assert recorded["validation_loss_iter"] == pytest.approx((4.0, trainer.iter))
    assert recorded["validation_onsite_loss"] == pytest.approx((5.0, trainer.iter))
    assert recorded["validation_hopping_loss"] == pytest.approx((6.0, trainer.iter))
