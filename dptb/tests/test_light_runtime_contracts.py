from __future__ import annotations

import pytest

from dptb.checkpoint_config import merge_checkpoint_common_options
from dptb.entrypoints.main import parse_args
from dptb.utils.argcheck import common_options


def test_partial_common_override_is_filled_from_checkpoint_without_mutation():
    checkpoint = {
        "basis": {"H": ["1s"]},
        "has_soc": True,
        "dtype": "float64",
        "device": "cuda:0",
    }
    explicit = {"device": "cpu"}

    merged = merge_checkpoint_common_options(explicit, checkpoint, explicit)

    assert merged == {
        "basis": {"H": ["1s"]},
        "has_soc": True,
        "dtype": "float64",
        "device": "cpu",
    }
    assert checkpoint["device"] == "cuda:0"
    assert explicit == {"device": "cpu"}


def test_schema_defaults_do_not_replace_soc_checkpoint_architecture():
    normalized = {
        "basis": {"H": ["1s"]},
        "overlap": False,
        "has_soc": False,
        "nextham_uureal_mask": False,
        "full_soc_prediction": False,
        "dtype": "float32",
        "device": "cpu",
        "seed": 123,
    }
    checkpoint = {
        "basis": {"H": ["1s"]},
        "overlap": False,
        "has_soc": True,
        "nextham_uureal_mask": True,
        "full_soc_prediction": False,
        "dtype": "float64",
        "device": "cuda:0",
        "seed": 999,
    }
    explicit = {"basis": {"H": ["1s"]}}

    merged = merge_checkpoint_common_options(
        normalized,
        checkpoint,
        explicit,
        preserve_runtime_defaults=True,
    )

    assert merged["has_soc"] is True
    assert merged["nextham_uureal_mask"] is True
    assert merged["dtype"] == "float64"
    assert merged["device"] == "cpu"
    assert merged["seed"] == 123


def test_explicit_dtype_override_is_preserved():
    normalized = {"basis": {"H": ["1s"]}, "dtype": "float32"}
    checkpoint = {"basis": {"H": ["1s"]}, "dtype": "float64"}
    merged = merge_checkpoint_common_options(
        normalized, checkpoint, {"dtype": "float32"}
    )
    assert merged["dtype"] == "float32"


def test_explicit_soc_override_against_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="has_soc.*conflicts with checkpoint"):
        merge_checkpoint_common_options(
            {"has_soc": False},
            {"has_soc": True},
            {"has_soc": False},
        )


def test_common_options_rejects_removed_overlap_prediction():
    schema = common_options()
    normalized = schema.normalize_value(
        {"basis": {"H": ["1s"]}, "overlap": True}
    )
    with pytest.raises(ValueError, match="common_options.overlap must be false"):
        schema.check_value(normalized, strict=True)


@pytest.mark.parametrize("command", ["test", "run"])
def test_checkpoint_is_required_by_cli_parser(command):
    with pytest.raises(SystemExit):
        parse_args([command, "input.yaml"])


@pytest.mark.parametrize("command", ["test", "run"])
def test_checkpoint_cli_parser_accepts_explicit_model(command):
    parsed = parse_args([command, "input.yaml", "--init-model", "model.pth"])
    assert parsed.init_model == "model.pth"
