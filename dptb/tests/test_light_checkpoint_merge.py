"""Checkpoint/common_options merge contract across every CLI entrypoint.

The regression these lock down: a minimal user config normalizes to a full set
of schema defaults (has_soc=False, dtype=float32, ...), and if those defaults
are treated as user intent they silently override the architecture baked into
the checkpoint.
"""

from __future__ import annotations

import importlib
import json

import pytest
import torch

from dptb.checkpoint_config import merge_checkpoint_common_options


SOC_CHECKPOINT_COMMON_OPTIONS = {
    "basis": {"Si": ["3s", "3p"]},
    "overlap": False,
    "has_soc": True,
    "nextham_uureal_mask": True,
    "full_soc_prediction": True,
    "device": "cuda:0",
    "dtype": "float64",
    "seed": 999,
}

MODEL_OPTIONS = {
    "embedding": {
        "method": "lem_moe_v3",
        "r_max": 6.0,
        "irreps_hidden": "32x0e+16x1o+8x2e",
        "avg_num_neighbors": 50.0,
        "n_layers": 2,
    },
    "prediction": {"method": "e3tb", "neurons": [16, 16]},
}

TRAIN_OPTIONS = {
    "num_epoch": 1,
    "batch_size": 1,
    "loss_options": {"train": {"method": "hamil_abs"}},
}


class _Captured(Exception):
    """Raised by the stubs to stop the entrypoint once we have what we need."""

    def __init__(self, payload):
        super().__init__("captured")
        self.payload = payload


def _raise_captured(payload):
    raise _Captured(payload)


def _write_checkpoint(path, common_options, train_options=None, with_state_dict=True):
    payload = {
        "config": {
            "common_options": dict(common_options),
            "model_options": json.loads(json.dumps(MODEL_OPTIONS)),
            "train_options": dict(train_options or {}),
        },
    }
    if with_state_dict:
        payload["model_state_dict"] = {}
    torch.save(payload, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# helper-level contracts for the two parameters added on top of the bundle
# ---------------------------------------------------------------------------
def test_empty_explicit_lets_checkpoint_architecture_win():
    normalized = {
        "basis": {"Si": ["3s"]},
        "has_soc": False,
        "dtype": "float32",
        "device": "cpu",
        "seed": 1,
    }
    merged = merge_checkpoint_common_options(
        normalized,
        SOC_CHECKPOINT_COMMON_OPTIONS,
        {},
        preserve_runtime_defaults=True,
    )
    assert merged["has_soc"] is True
    assert merged["nextham_uureal_mask"] is True
    assert merged["full_soc_prediction"] is True
    assert merged["dtype"] == "float64"
    assert merged["basis"] == {"Si": ["3s", "3p"]}
    # runtime knobs still come from the caller
    assert merged["device"] == "cpu"
    assert merged["seed"] == 1


def test_weights_inferred_override_bypasses_the_architecture_conflict_gate():
    normalized = {"basis": {"Si": ["3s"]}, "overlap": False}
    merged = merge_checkpoint_common_options(
        normalized,
        {"basis": {"Si": ["3s"]}, "overlap": False},
        normalized,
        weights_inferred_overrides={"overlap": True},
    )
    assert merged["overlap"] is True


def test_weights_inferred_override_wins_over_explicit_and_checkpoint():
    merged = merge_checkpoint_common_options(
        {"overlap": False},
        {"overlap": True},
        {"overlap": False},
        weights_inferred_overrides={"overlap": True},
    )
    assert merged["overlap"] is True


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
def test_build_model_partial_common_options_builds_distance_ensemble(tmp_path, monkeypatch):
    """CONFIRMED crash: a non-empty partial dict used to discard the whole
    checkpoint common_options, so the ensemble branch lost `basis` and raised
    'Either basis or idp should be provided'."""
    from dptb.nn import build as build_mod

    ckpt = _write_checkpoint(
        tmp_path / "ensemble.pth",
        {
            "basis": {"Si": ["3s", "3p"]},
            "overlap": False,
            "has_soc": True,
            "device": "cuda:0",
            "dtype": "float64",
            "seed": 7,
        },
        train_options={"distance_ranges": [[0.0, 3.0], [3.0, 6.0]]},
        with_state_dict=False,
    )

    seen = {}

    def fake_construct_single_model(model_options, common_options):
        seen["common_options"] = dict(common_options)
        raise _Captured(dict(common_options))

    monkeypatch.setattr(
        build_mod, "_construct_single_model", fake_construct_single_model
    )

    with pytest.raises(_Captured):
        build_mod.build_model(
            checkpoint=ckpt,
            common_options={"device": "cpu"},
        )

    merged = seen["common_options"]
    assert merged["basis"] == {"Si": ["3s", "3p"]}
    assert merged["has_soc"] is True
    assert merged["dtype"] == "float64"
    assert merged["device"] == "cpu"


def test_build_model_schema_defaults_do_not_override_checkpoint(tmp_path, monkeypatch):
    from dptb.nn import build as build_mod

    ckpt = _write_checkpoint(tmp_path / "soc.pth", SOC_CHECKPOINT_COMMON_OPTIONS)

    seen = {}

    def fake_from_reference(checkpoint, **kwargs):
        seen.update(kwargs)
        raise _Captured(kwargs)

    monkeypatch.setattr(build_mod.NNENV, "from_reference", staticmethod(fake_from_reference))

    normalized_defaults = {
        "basis": {"Si": ["3s", "3p"]},
        "overlap": False,
        "has_soc": False,
        "nextham_uureal_mask": False,
        "full_soc_prediction": False,
        "device": "cpu",
        "dtype": "float32",
        "seed": 3982377700,
    }
    with pytest.raises(_Captured):
        build_mod.build_model(
            checkpoint=ckpt,
            common_options=normalized_defaults,
            explicit_common_options={"basis": {"Si": ["3s", "3p"]}},
        )

    assert seen["has_soc"] is True
    assert seen["nextham_uureal_mask"] is True
    assert seen["full_soc_prediction"] is True
    assert seen["dtype"] == "float64"
    # device/seed are runtime knobs and stay with the caller
    assert seen["device"] == "cpu"
    assert seen["seed"] == 3982377700


# ---------------------------------------------------------------------------
# dptb test entrypoint
# ---------------------------------------------------------------------------
def test_test_entrypoint_keeps_soc_and_float64_from_checkpoint(tmp_path, monkeypatch):
    # `dptb.entrypoints.test` is shadowed by the re-exported `test` function.
    test_mod = importlib.import_module("dptb.entrypoints.test")

    ckpt = _write_checkpoint(tmp_path / "soc.pth", SOC_CHECKPOINT_COMMON_OPTIONS)

    input_path = tmp_path / "test.json"
    input_path.write_text(
        json.dumps(
            {
                "common_options": {"basis": {"Si": ["3s", "3p"]}},
                "data_options": {
                    "test": {
                        "root": str(tmp_path),
                        "prefix": "t",
                        "type": "LMDBDataset",
                    }
                },
                "test_options": {
                    "batch_size": 1,
                    "loss_options": {"test": {"method": "hamil_abs"}},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        test_mod, "build_dataset", lambda **kwargs: _raise_captured(kwargs)
    )

    with pytest.raises(_Captured) as excinfo:
        test_mod._test(
            INPUT=str(input_path),
            init_model=ckpt,
            output=None,
            log_level=2,
            log_path=None,
        )

    common = excinfo.value.payload
    assert common["has_soc"] is True
    assert common["nextham_uureal_mask"] is True
    assert common["full_soc_prediction"] is True
    assert common["dtype"] == "float64"
    assert common["device"] == "cpu"


# ---------------------------------------------------------------------------
# dptb multi-train restart entrypoint
# ---------------------------------------------------------------------------
def test_multi_train_restart_keeps_soc_and_float64_from_checkpoint(tmp_path, monkeypatch):
    multi_train_mod = importlib.import_module("dptb.entrypoints.multi_train")

    train_options = json.loads(json.dumps(TRAIN_OPTIONS))
    ckpt = _write_checkpoint(
        tmp_path / "soc.pth", SOC_CHECKPOINT_COMMON_OPTIONS, train_options
    )

    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "common_options": {"basis": {"Si": ["3s", "3p"]}},
                "model_options": MODEL_OPTIONS,
                "train_options": train_options,
                "data_options": {
                    "train": {
                        "root": str(tmp_path),
                        "prefix": "t",
                        "type": "LMDBDataset",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        multi_train_mod, "build_dataset", lambda **kwargs: _raise_captured(kwargs)
    )

    previous_dtype = torch.get_default_dtype()
    try:
        with pytest.raises(_Captured) as excinfo:
            multi_train_mod._multi_train_impl(
                INPUT=str(input_path),
                init_model=None,
                restart=ckpt,
                output=None,
                log_level=2,
                log_path=None,
            )
        common = excinfo.value.payload
        assert common["has_soc"] is True
        assert common["nextham_uureal_mask"] is True
        assert common["full_soc_prediction"] is True
        assert common["dtype"] == "float64"
        assert common["device"] == "cpu"
        # set_default_dtype must run AFTER the checkpoint merge, not before.
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(previous_dtype)
