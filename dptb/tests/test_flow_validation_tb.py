from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dptb.plugins.monitor import TensorBoardMonitor, Validationer


class _FlowValidationTrainer:
    def __init__(self):
        self.stats = {}
        self.ep = 7
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

    assert recorded["validation_loss_mean/epoch"] == pytest.approx((2.0, trainer.ep))
    assert recorded["validation_flow_random_t_loss_mean/epoch"] == pytest.approx(
        (9.0, trainer.ep)
    )
