"""0721-stable checkpoints must still load after the 0726-light schema cleanup.

The cleanup deleted 6 optional-with-default Arguments. dargs had already baked
their defaults into every saved checkpoint config, so strict validation on
--restart / --init-model rejected the checkpoint's own config.
"""

from __future__ import annotations

import logging

import pytest

from dptb.configuration import (
    REMOVED_EMBEDDING_OPTION_DEFAULTS,
    REMOVED_FLOW_OPTION_DEFAULTS,
    migrate_legacy_checkpoint_model_options,
    migrate_legacy_checkpoint_train_options,
)
from dptb.utils.argcheck import normalize


LEGACY_EMBEDDING = {
    "method": "lem_moe_v3",
    "r_max": 6.0,
    "irreps_hidden": "32x0e+16x1o",
    "avg_num_neighbors": 50.0,
    "n_layers": 2,
    # removed by the cleanup, present in every 0721-stable checkpoint
    "edge_message_value_gate": False,
    "edge_message_value_gate_hidden_dim": 0,
}

LEGACY_FLOW_OPTIONS = {
    "enabled": False,
    # removed by the cleanup, present in every 0721-stable checkpoint
    "prior_skdata": "",
    "dftb_prior_overlap": False,
    "dftb_prior_strict": True,
    "dftb_prior_require_geometry": True,
}


def test_removed_key_tables_cover_exactly_the_six_deleted_arguments():
    names = {name for name, _ in REMOVED_EMBEDDING_OPTION_DEFAULTS}
    names |= {name for name, _ in REMOVED_FLOW_OPTION_DEFAULTS}
    assert names == {
        "edge_message_value_gate",
        "edge_message_value_gate_hidden_dim",
        "prior_skdata",
        "dftb_prior_overlap",
        "dftb_prior_strict",
        "dftb_prior_require_geometry",
    }


def test_model_options_migration_strips_removed_embedding_keys():
    migrated = migrate_legacy_checkpoint_model_options(
        {
            "embedding": dict(LEGACY_EMBEDDING),
            "prediction": {"method": "e3tb", "neurons": [16, 16]},
        }
    )
    assert "edge_message_value_gate" not in migrated["embedding"]
    assert "edge_message_value_gate_hidden_dim" not in migrated["embedding"]
    # untouched keys survive
    assert migrated["embedding"]["irreps_hidden"] == "32x0e+16x1o"
    assert migrated["embedding"]["n_layers"] == 2


def test_train_options_migration_strips_removed_flow_keys():
    migrated = migrate_legacy_checkpoint_train_options(
        {"num_epoch": 3, "flow_options": dict(LEGACY_FLOW_OPTIONS)}
    )
    for key, _default in REMOVED_FLOW_OPTION_DEFAULTS:
        assert key not in migrated["flow_options"]
    assert migrated["num_epoch"] == 3
    assert migrated["flow_options"]["enabled"] is False


def test_non_default_removed_value_is_warned_about(caplog):
    embedding = dict(LEGACY_EMBEDDING)
    embedding["edge_message_value_gate"] = True
    with caplog.at_level(logging.WARNING, logger="dptb.configuration"):
        migrate_legacy_checkpoint_model_options({"embedding": embedding})
    assert any(
        "edge_message_value_gate=True" in record.getMessage()
        for record in caplog.records
    )


def test_migrated_legacy_checkpoint_config_passes_strict_normalize():
    """End-to-end shape of the restart path: the checkpoint's own saved config
    goes through the migrations and then through strict normalize()."""

    saved_model_options = {
        "embedding": dict(LEGACY_EMBEDDING),
        "prediction": {"method": "e3tb", "neurons": [16, 16]},
    }
    saved_train_options = {
        "num_epoch": 1,
        "batch_size": 1,
        "loss_options": {"train": {"method": "hamil_abs"}},
        "flow_options": dict(LEGACY_FLOW_OPTIONS),
    }

    jdata = {
        "common_options": {"basis": {"Si": ["3s", "3p"]}},
        "model_options": migrate_legacy_checkpoint_model_options(saved_model_options),
        "train_options": migrate_legacy_checkpoint_train_options(saved_train_options),
        "data_options": {
            "train": {"root": "./x", "prefix": "t", "type": "LMDBDataset"}
        },
    }

    normalized = normalize(jdata)
    assert normalized["model_options"]["embedding"]["method"] == "lem_moe_v3"


def test_unmigrated_legacy_checkpoint_config_is_still_rejected():
    """Guards the premise: without the migration these keys really do fail."""

    jdata = {
        "common_options": {"basis": {"Si": ["3s", "3p"]}},
        "model_options": {
            "embedding": dict(LEGACY_EMBEDDING),
            "prediction": {"method": "e3tb", "neurons": [16, 16]},
        },
        "train_options": {
            "num_epoch": 1,
            "batch_size": 1,
            "loss_options": {"train": {"method": "hamil_abs"}},
        },
        "data_options": {
            "train": {"root": "./x", "prefix": "t", "type": "LMDBDataset"}
        },
    }
    with pytest.raises(Exception, match="edge_message_value_gate"):
        normalize(jdata)
