"""The documented quick-start config must work after ONE normalize() pass.

Regression: `dptb train input.json -o output` crashed from scratch because
train_options.optimizer / lr_scheduler are optional dict Arguments whose
sub_variants dargs never walks when the key is missing, so they normalized to
`{}` and Trainer.__init__ raised
`get_lr_scheduler() missing 1 required positional argument: 'type'`.
The restart/init-model paths hid it by normalizing a second time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from dptb.utils.argcheck import normalize
from dptb.utils.tools import get_optimizer, get_lr_scheduler


DOCS_INPUT = Path(__file__).resolve().parents[2] / "docs" / "quick_start" / "input.md"


def _load_documented_config():
    text = DOCS_INPUT.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.S)
    assert blocks, f"no json block found in {DOCS_INPUT}"
    return json.loads(blocks[0])


def _build_optimizer_and_scheduler(train_options):
    params = [("w", torch.nn.Parameter(torch.zeros(2)))]
    optimizer = get_optimizer(
        model_param=iter(params), **train_options["optimizer"]
    )
    scheduler = get_lr_scheduler(
        optimizer=optimizer, **train_options["lr_scheduler"]
    )
    return optimizer, scheduler


def test_documented_quickstart_config_is_trainable_after_one_normalize():
    jdata = normalize(_load_documented_config())
    train_options = jdata["train_options"]

    assert train_options["optimizer"]["type"] == "Adam"
    assert train_options["optimizer"]["lr"] == pytest.approx(1e-3)
    assert train_options["lr_scheduler"]["type"] == "exp"

    optimizer, scheduler = _build_optimizer_and_scheduler(train_options)
    assert optimizer.__class__.__name__ == "Adam"
    assert scheduler.__class__.__name__ == "ExponentialLR"


def test_omitting_both_optimizer_and_lr_scheduler_still_normalizes_to_defaults():
    config = _load_documented_config()
    config["train_options"].pop("optimizer", None)
    config["train_options"].pop("lr_scheduler", None)

    train_options = normalize(config)["train_options"]

    assert train_options["optimizer"]["type"] == "Adam"
    assert train_options["lr_scheduler"]["type"] == "exp"
    assert train_options["lr_scheduler"]["gamma"] == pytest.approx(0.999)
    _build_optimizer_and_scheduler(train_options)


def test_normalize_is_idempotent_for_optimizer_defaults():
    once = normalize(_load_documented_config())
    twice = normalize(once)
    assert twice["train_options"]["optimizer"] == once["train_options"]["optimizer"]
    assert twice["train_options"]["lr_scheduler"] == once["train_options"]["lr_scheduler"]


def test_explicit_optimizer_choice_survives_normalization():
    config = _load_documented_config()
    config["train_options"]["optimizer"] = {"type": "SGD", "lr": 0.05}
    config["train_options"]["lr_scheduler"] = {"type": "rop", "factor": 0.5}

    train_options = normalize(config)["train_options"]

    assert train_options["optimizer"]["type"] == "SGD"
    assert train_options["optimizer"]["lr"] == pytest.approx(0.05)
    assert train_options["lr_scheduler"]["type"] == "rop"
    assert train_options["lr_scheduler"]["factor"] == pytest.approx(0.5)
