"""train_options.epoch_checkpoint: epoch saves by default, iteration-only checkpoints on request."""
import pytest

from dptb.plugins.saver import checkpoint_intervals
from dptb.utils.argcheck import train_options


def _normalized(**kw):
    cfg = {
        "num_epoch": 1,
        "batch_size": 1,
        "save_freq": 1000,
        "optimizer": {"type": "AdamW", "lr": 1e-3},
        "lr_scheduler": {"type": "rop"},
        "loss_options": {"train": {"method": "hamil_blockwise_nextham"}},
        **kw,
    }
    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    return normalized


@pytest.mark.parametrize(
    "option, expected",
    [
        ({}, [(1000, "iteration"), (1, "epoch")]),
        ({"epoch_checkpoint": True}, [(1000, "iteration"), (1, "epoch")]),
        ({"epoch_checkpoint": False}, [(1000, "iteration")]),
    ],
)
def test_epoch_checkpoint_selects_saver_triggers(option, expected):
    assert checkpoint_intervals(_normalized(**option)) == expected


def test_no_save_freq_registers_no_trigger():
    assert checkpoint_intervals({"save_freq": 0, "epoch_checkpoint": True}) is None
