from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dptb.configuration import (
    canonicalize_embedding_options,
    canonicalize_flow_options,
    canonicalize_prediction_options,
    canonicalize_training_config,
    migrate_legacy_checkpoint_model_options,
    migrate_legacy_checkpoint_train_options,
    resolve_init_scope,
)
from dptb.nnops.ddp_utils import merge_restart_train_options
from dptb.nnops.flow import HamiltonianCFM
from dptb.nnops.flow_priors import OverlapHuckelFamily
from dptb.utils.argcheck import (
    flow_options,
    reference_data_sub,
    train_data_sub,
    train_options,
    validation_data_sub,
)
# Aliased so pytest does not collect the schema builder as a test case.
from dptb.utils.argcheck import test_data_sub as _test_data_sub

REPO_ROOT = Path(__file__).resolve().parents[2]


def _schema_then_runtime(raw):
    canonical = canonicalize_training_config(
        {"train_options": {"flow_options": raw}}, warn_deprecated=False
    )["train_options"]["flow_options"]
    schema = flow_options()
    normalized = schema.normalize_value(canonical)
    schema.check_value(normalized, strict=True)
    return normalized, HamiltonianCFM(normalized)


def test_flow_aliases_survive_schema_defaults_and_reach_runtime():
    normalized, flow = _schema_then_runtime(
        {
            "enabled": True,
            "prior": "overlap_huckel",
            "overlap_huckel_k": 9.0,
            "overlap_huckel_edge_channel_scale": [2.0],
            "prior_jitter_sigma": 0.25,
            "dftb_skdata": "sentinel-skdata",
        }
    )

    assert "overlap_huckel_k" not in normalized
    assert "prior_jitter_sigma" not in normalized
    assert normalized["huckel_k"] == pytest.approx(9.0)
    assert normalized["physical_prior_jitter_sigma"] == pytest.approx(0.25)
    assert normalized["prior_skdata"] == "sentinel-skdata"
    family = flow._families[OverlapHuckelFamily]
    assert family.huckel_k == pytest.approx(9.0)
    assert family.huckel_edge_channel_scale == [2.0]
    assert flow.physical_prior_jitter_sigma == pytest.approx(0.25)


def test_conflicting_flow_alias_is_fail_closed():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        canonicalize_flow_options(
            {"huckel_k": 2.0, "overlap_huckel_k": 3.0},
            warn_deprecated=False,
        )


def test_alias_conflict_is_strict_even_when_canonical_equals_old_default():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        canonicalize_flow_options(
            {"huckel_k": 1.75, "overlap_huckel_k": 9.0},
            warn_deprecated=False,
        )


@pytest.mark.parametrize(
    ("canonical", "alias", "old_default", "custom"),
    [
        ("huckel_k", "overlap_huckel_k", 1.75, 9.0),
        (
            "huckel_edge_channel_scale",
            "overlap_huckel_edge_channel_scale",
            None,
            [2.0],
        ),
        ("prior_skdata", "dftb_skdata", "", "sentinel-skdata"),
        (
            "physical_prior_jitter_sigma",
            "prior_jitter_sigma",
            0.0,
            0.25,
        ),
    ],
)
def test_legacy_checkpoint_migration_resolves_only_schema_default_alias_collisions(
    canonical, alias, old_default, custom
):
    alias_wins = migrate_legacy_checkpoint_train_options(
        {
            "flow_options": {
                canonical: old_default,
                alias: custom,
            }
        },
        warn_deprecated=False,
    )["flow_options"]
    assert alias_wins[canonical] == custom
    assert alias not in alias_wins

    canonical_wins = migrate_legacy_checkpoint_train_options(
        {
            "flow_options": {
                canonical: custom,
                alias: old_default,
            }
        },
        warn_deprecated=False,
    )["flow_options"]
    assert canonical_wins[canonical] == custom
    assert alias not in canonical_wins


def test_legacy_checkpoint_migration_keeps_distinct_custom_values_fail_closed():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        migrate_legacy_checkpoint_train_options(
            {
                "flow_options": {
                    "huckel_k": 2.0,
                    "overlap_huckel_k": 3.0,
                }
            },
            warn_deprecated=False,
        )


def test_legacy_checkpoint_migration_canonicalizes_endpoint_logging_options():
    migrated = migrate_legacy_checkpoint_train_options(
        {
            "log_single_model_compatible_loss": False,
            "log_single_model_compatible_loss_mode": "full_forward",
        },
        warn_deprecated=False,
    )
    assert "log_single_model_compatible_loss" not in migrated
    assert "log_single_model_compatible_loss_mode" not in migrated
    assert migrated["endpoint_loss_mode"] == "full_forward"


@pytest.mark.parametrize(
    ("legacy_enabled", "legacy_mode", "expected"),
    [
        (True, "reduce", "reduce"),
        (True, "full_forward", "full_forward"),
        (False, "reduce", "full_forward"),
        (False, "full_forward", "full_forward"),
    ],
)
def test_endpoint_legacy_boolean_mode_truth_table_is_preserved_everywhere(
    legacy_enabled, legacy_mode, expected
):
    legacy = {
        "log_single_model_compatible_loss": legacy_enabled,
        "log_single_model_compatible_loss_mode": legacy_mode,
    }

    canonical = canonicalize_training_config(
        {"train_options": legacy}, warn_deprecated=False
    )["train_options"]
    migrated = migrate_legacy_checkpoint_train_options(
        legacy, warn_deprecated=False
    )

    for options in (canonical, migrated):
        assert options["endpoint_loss_mode"] == expected
        assert "log_single_model_compatible_loss" not in options
        assert "log_single_model_compatible_loss_mode" not in options


def test_conflicting_meanflow_aggressive_aliases_are_fail_closed():
    with pytest.raises(ValueError, match="meanflow_aggressive"):
        canonicalize_flow_options(
            {
                "meanflow_aggressive": True,
                "meanflow": {"aggressive": False},
            },
            warn_deprecated=False,
        )


def test_block_te_schema_default_resolves_to_block_mode():
    normalized, flow = _schema_then_runtime(
        {"enabled": True, "prior": "block_te"}
    )
    assert normalized["te_prior_mode"] == "auto"
    assert flow.te_prior_mode == "block"


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"strict_h0": True, "warn_missing_h0": False}, "error"),
        ({"strict_h0": False, "warn_missing_h0": True}, "warn_zero"),
        ({"strict_h0": False, "warn_missing_h0": False}, "zero"),
    ],
)
def test_missing_h0_flags_collapse_to_policy(legacy, expected):
    canonical = canonicalize_flow_options(legacy, warn_deprecated=False)
    assert canonical["missing_h0_policy"] == expected
    assert "strict_h0" not in canonical
    assert "warn_missing_h0" not in canonical


def test_dead_logging_flags_are_removed_but_runtime_contract_stays_on():
    canonical = canonicalize_flow_options(
        {
            "enabled": True,
            "log_compatible_loss": False,
            "log_train_compatible_loss": False,
            "log_validation_compatible_loss": False,
            "compatible_loss_to_legacy_keys": False,
        },
        warn_deprecated=False,
    )
    assert not any(key.startswith("log_") for key in canonical)
    flow = HamiltonianCFM(canonical)
    assert flow.log_train_compatible_loss is True
    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True


def test_dead_multi_compatible_boolean_is_removed_and_mode_is_canonicalized():
    canonical = canonicalize_training_config(
        {
            "train_options": {
                "log_single_model_compatible_loss": False,
                "log_single_model_compatible_loss_mode": "full_forward",
            }
        },
        warn_deprecated=False,
    )
    train = canonical["train_options"]
    assert "log_single_model_compatible_loss" not in train
    assert "log_single_model_compatible_loss_mode" not in train
    assert train["endpoint_loss_mode"] == "full_forward"


def test_conflicting_endpoint_loss_mode_alias_is_fail_closed():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        canonicalize_training_config(
            {
                "train_options": {
                    "endpoint_loss_mode": "reduce",
                    "log_single_model_compatible_loss_mode": "full_forward",
                }
            },
            warn_deprecated=False,
        )


def test_unknown_endpoint_loss_mode_is_fail_closed():
    with pytest.raises(ValueError, match="endpoint_loss_mode"):
        canonicalize_training_config(
            {"train_options": {"endpoint_loss_mode": "sometimes"}},
            warn_deprecated=False,
        )


def test_validation_flow_logging_booleans_collapse_to_one_metric_list():
    canonical = canonicalize_flow_options(
        {
            "log_validation_random_t_loss": False,
            "log_validation_t0_loss": True,
            "log_validation_flow_euler_loss": False,
        },
        warn_deprecated=False,
    )

    assert canonical["validation_flow_metrics"] == ["one_step"]
    assert not any(key.startswith("log_validation_") for key in canonical)


def test_validation_flow_metric_list_conflict_with_legacy_flags_fails_closed():
    with pytest.raises(ValueError, match="validation_flow_metrics conflicts"):
        canonicalize_flow_options(
            {
                "validation_flow_metrics": ["random_t"],
                "log_validation_random_t_loss": False,
            },
            warn_deprecated=False,
        )


def test_h0_and_p2_init_boolean_combinations_collapse_to_scopes():
    h0 = canonicalize_embedding_options(
        {
            "method": "lem_moe_v3_h0",
            "use_h0_init": True,
            "use_h0_node_init": False,
            "use_h0_edge_init": True,
            "h0_fallback_to_hamiltonian": False,
        },
        warn_deprecated=False,
    )
    assert h0["h0_init_scope"] == "edge"
    assert h0["fallback_to_hamiltonian"] is False

    p2 = canonicalize_embedding_options(
        {
            "method": "lem_moe_v3_prior",
            "prior_kind": "p2",
            "use_prior_init": True,
            "use_prior_node_init": True,
            "use_prior_edge_init": False,
        },
        warn_deprecated=False,
    )
    assert p2["prior_init_scope"] == "node"
    assert p2["prior_kind"] == "p2"


@pytest.mark.parametrize(
    ("canonical", "legacy_alias"),
    [(True, False), (False, True)],
)
def test_checkpoint_h0_fallback_alias_preserves_legacy_runtime_precedence(
    canonical, legacy_alias
):
    migrated = migrate_legacy_checkpoint_model_options(
        {
            "embedding": {
                "method": "lem_moe_v3_h0",
                "fallback_to_hamiltonian": canonical,
                "h0_fallback_to_hamiltonian": legacy_alias,
            }
        },
        warn_deprecated=False,
    )

    embedding = migrated["embedding"]
    assert embedding["fallback_to_hamiltonian"] is legacy_alias
    assert "h0_fallback_to_hamiltonian" not in embedding


def test_raw_h0_fallback_alias_conflict_remains_fail_closed():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        canonicalize_embedding_options(
            {
                "method": "lem_moe_v3_h0",
                "fallback_to_hamiltonian": True,
                "h0_fallback_to_hamiltonian": False,
            },
            warn_deprecated=False,
        )


def test_restart_merge_uses_checkpoint_as_base_and_locks_optimizer_contract():
    checkpoint = {
        "endpoint_loss_mode": "full_forward",
        "flow_options": {
            "enabled": True,
            "objective": "cfm",
            "prior": "block_te",
            "sigma_data": 0.5,
        },
        "optimizer": {"type": "AdamW", "lr": 1.0e-3},
        "lr_scheduler": {"type": "rop", "factor": 0.5},
    }

    untouched = merge_restart_train_options({}, checkpoint)
    assert untouched == checkpoint

    merged = merge_restart_train_options(
        {
            "display_freq": 7,
            "flow_options": {"sigma_data": 0.25},
            "optimizer": {"type": "SGD", "lr": 0.1},
        },
        checkpoint,
    )
    assert merged["endpoint_loss_mode"] == "full_forward"
    assert merged["flow_options"] == {
        "enabled": True,
        "objective": "cfm",
        "prior": "block_te",
        "sigma_data": 0.25,
    }
    assert merged["display_freq"] == 7
    assert merged["optimizer"] == checkpoint["optimizer"]
    assert merged["lr_scheduler"] == checkpoint["lr_scheduler"]


@pytest.mark.parametrize(
    ("enabled", "node", "edge", "expected"),
    [
        (False, False, False, "none"),
        (False, False, True, "none"),
        (False, True, False, "none"),
        (False, True, True, "none"),
        (True, False, False, "auxiliary"),
        (True, False, True, "edge"),
        (True, True, False, "node"),
        (True, True, True, "both"),
    ],
)
def test_init_scope_preserves_all_legacy_boolean_semantics(
    enabled, node, edge, expected
):
    scope, wrapper_enabled, use_node, use_edge = resolve_init_scope(
        None,
        enabled=enabled,
        node=node,
        edge=edge,
    )
    assert scope == expected
    assert wrapper_enabled is (expected != "none")
    assert use_node is (expected in {"both", "node"})
    assert use_edge is (expected in {"both", "edge"})


@pytest.mark.parametrize("scope", ["NONE", "off", "disabled"])
def test_canonical_scope_aliases_are_normalized_before_contract_validation(scope):
    embedding = canonicalize_embedding_options(
        {
            "method": "lem_moe_v3_h0",
            "h0_init_scope": scope,
        },
        warn_deprecated=False,
    )
    assert embedding["h0_init_scope"] == "none"


@pytest.mark.parametrize("field", ["h0_init_scope", "prior_init_scope"])
def test_invalid_canonical_scope_fails_during_normalization(field):
    method = "lem_moe_v3_prior" if field.startswith("prior") else "lem_moe_v3_h0"
    with pytest.raises(ValueError, match=field):
        canonicalize_embedding_options(
            {"method": method, field: "bogus"},
            warn_deprecated=False,
        )


def test_p2_disabled_scope_rejects_default_enabled_memory_during_normalization():
    with pytest.raises(ValueError, match="soft_edge_memory.enabled=true"):
        canonicalize_embedding_options(
            {
                "method": "lem_moe_v3_prior",
                "prior_init_scope": "none",
            },
            warn_deprecated=False,
        )


def test_p2_auxiliary_scope_preserves_memory_only_legacy_route():
    embedding = canonicalize_embedding_options(
        {
            "method": "lem_moe_v3_prior",
            "use_prior_init": True,
            "use_prior_node_init": False,
            "use_prior_edge_init": False,
            "use_soft_edge_memory": True,
        },
        warn_deprecated=False,
    )
    assert embedding["prior_init_scope"] == "auxiliary"
    assert embedding["soft_edge_memory"]["enabled"] is True


def test_edge_h0_embedding_consumes_canonical_scope(monkeypatch):
    from dptb.nn.embedding import lem_moe_v3_edge as edge_module

    base_init = object()

    def fake_base_init(self, **kwargs):
        self.init_layer = base_init
        self.dtype = None
        self.device = None

    class CapturedH0Init:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(edge_module.LemMoEV3Edge, "__init__", fake_base_init)
    monkeypatch.setattr(edge_module, "H0InitLayer", CapturedH0Init)

    model = edge_module.LemMoEV3EdgeH0(h0_init_scope="edge")

    assert model.h0_init_scope == "edge"
    assert model.use_h0_init is True
    assert model.init_layer.kwargs["base_init"] is base_init
    assert model.init_layer.kwargs["use_h0_node_init"] is False
    assert model.init_layer.kwargs["use_h0_edge_init"] is True


def test_soft_edge_memory_flat_flags_collapse_to_one_subconfig():
    embedding = canonicalize_embedding_options(
        {
            "method": "lem_moe_v3_prior",
            "use_soft_edge_memory": True,
            "soft_edge_memory_num_slots": 8,
            "soft_edge_memory_diagnostics_mode": "sampled",
        },
        warn_deprecated=False,
    )
    assert embedding["soft_edge_memory"] == {
        "enabled": True,
        "num_slots": 8,
        "diagnostics_mode": "sampled",
    }


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"add_h0": True, "add_prior": False}, "h0_residual"),
        ({"add_h0": False, "add_prior": True}, "prior_residual"),
        ({"add_h0": False, "add_prior": False}, "direct"),
    ],
)
def test_prediction_reconstruction_replaces_mutually_exclusive_flags(legacy, expected):
    prediction = canonicalize_prediction_options(legacy, warn_deprecated=False)
    assert prediction == {"reconstruction": expected}


def test_meanflow_export_is_order_independent():
    from dptb.nnops.flow import HamiltonianPixelMeanFlow as compatibility_alias
    from dptb.nnops.flow_meanflow import HamiltonianPixelMeanFlow

    assert compatibility_alias is HamiltonianPixelMeanFlow


# --------------------------------------------------------------------------
# (3) DEV-only prior-fingerprint escape hatch: schema default must stay False.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "split_builder",
    [train_data_sub, validation_data_sub, reference_data_sub, _test_data_sub],
)
def test_allow_unbound_prior_source_fingerprint_schema_default_is_false(split_builder):
    field = split_builder().sub_fields["allow_unbound_prior_source_fingerprint"]
    assert field.default is False


def test_smoke_config_keeps_dev_only_flag_and_loud_warning_comment():
    smoke = REPO_ROOT / "configs" / "p2_prior_non_soc_full_h_smoke.yaml"
    text = smoke.read_text(encoding="utf-8")

    # The dev-only override is still present (the P2 smoke fixture omits the
    # source fingerprint on purpose) ...
    payload = yaml.safe_load(text)
    assert (
        payload["data_options"]["train"]["allow_unbound_prior_source_fingerprint"]
        is True
    )
    # ... but it must be guarded by a loud DEV-ONLY comment so nobody copies
    # it into a production config.
    assert "DEV-ONLY ESCAPE HATCH" in text
    flag_line = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("allow_unbound_prior_source_fingerprint:")
    )
    assert "DEV-ONLY" in flag_line


# --------------------------------------------------------------------------
# (2) Typed nested train_options groups mirror + normalize to the flat keys.
# --------------------------------------------------------------------------
_KNOWN_GOOD_LOSS = {
    "train": {
        "method": "hamil_blockwise_nextham",
        "optimization": "block_mae",
        "block_reduction": "global",
    }
}


def _normalized_train_options(train_opts):
    """Canonicalize then run the real train_options schema, as normalize() does."""

    canonical = canonicalize_training_config(
        {"train_options": train_opts}, warn_deprecated=False
    )["train_options"]
    schema = train_options()
    normalized = schema.normalize_value(canonical)
    schema.check_value(normalized, strict=True)
    return normalized


def test_train_option_group_names_match_schema():
    from dptb.configuration import _TRAIN_OPTION_GROUP_NAMES
    from dptb.utils.argcheck import TRAIN_OPTION_GROUP_MEMBERS

    assert set(_TRAIN_OPTION_GROUP_NAMES) == set(TRAIN_OPTION_GROUP_MEMBERS)


def test_every_group_member_is_a_real_flat_train_option_key():
    from dptb.utils.argcheck import TRAIN_OPTION_GROUP_MEMBERS

    schema = train_options()
    flat_keys = set(schema.sub_fields.keys())
    for group, members in TRAIN_OPTION_GROUP_MEMBERS.items():
        for member in members:
            assert member in flat_keys, f"{group}.{member} is not a flat key"
        # The group name itself must not shadow a flat key.
        assert group in flat_keys  # declared as an optional group
        for member in members:
            assert member != group


def test_nested_groups_normalize_identically_to_flat():
    flat = {
        "num_epoch": 3,
        "use_ddp": True,
        "ddp_backend": "gloo",
        "expert_data_parallel_size": 2,
        "save_freq": 25,
        "max_ckpt": 8,
        "use_tensorboard": True,
        "monitor_flag": True,
        "debug_profile": True,
        "cudnn_benchmark": True,
        "allow_tf32": False,
        "train_num_workers": 4,
        "flow_options": {"enabled": True, "prior": "zero"},
        "self_consistency": {"enabled": True, "weight": 0.2},
        "loss_options": _KNOWN_GOOD_LOSS,
    }
    nested = {
        "num_epoch": 3,
        "distributed": {
            "use_ddp": True,
            "ddp_backend": "gloo",
            "expert_data_parallel_size": 2,
        },
        "checkpoint": {"save_freq": 25, "max_ckpt": 8},
        "observers": {
            "use_tensorboard": True,
            "monitor_flag": True,
            "debug_profile": True,
        },
        "runtime": {
            "cudnn_benchmark": True,
            "allow_tf32": False,
            "train_num_workers": 4,
        },
        "physical_prior": {
            "flow_options": {"enabled": True, "prior": "zero"},
            "self_consistency": {"enabled": True, "weight": 0.2},
        },
        "loss_options": _KNOWN_GOOD_LOSS,
    }

    normalized_flat = _normalized_train_options(flat)
    normalized_nested = _normalized_train_options(nested)

    # The trainer reads flat keys; no group survives normalization.
    for group in ("runtime", "distributed", "checkpoint", "observers", "physical_prior"):
        assert group not in normalized_flat
        assert group not in normalized_nested

    assert normalized_nested == normalized_flat
    assert normalized_nested["use_ddp"] is True
    assert normalized_nested["save_freq"] == 25
    assert normalized_nested["flow_options"]["enabled"] is True
    assert normalized_nested["self_consistency"]["weight"] == pytest.approx(0.2)


def test_nested_group_conflicting_with_flat_key_fails_closed():
    with pytest.raises(ValueError, match="Conflicting configuration values"):
        canonicalize_training_config(
            {
                "train_options": {
                    "use_ddp": False,
                    "distributed": {"use_ddp": True},
                }
            },
            warn_deprecated=False,
        )


def test_nested_group_matching_flat_key_is_accepted():
    canonical = canonicalize_training_config(
        {
            "train_options": {
                "save_freq": 25,
                "checkpoint": {"save_freq": 25, "max_ckpt": 8},
            }
        },
        warn_deprecated=False,
    )["train_options"]
    assert canonical["save_freq"] == 25
    assert canonical["max_ckpt"] == 8
    assert "checkpoint" not in canonical


def test_flatten_train_option_groups_is_idempotent():
    once = canonicalize_training_config(
        {"train_options": {"distributed": {"use_ddp": True}}},
        warn_deprecated=False,
    )
    twice = canonicalize_training_config(once, warn_deprecated=False)
    assert twice == once
    assert twice["train_options"]["use_ddp"] is True
    assert "distributed" not in twice["train_options"]


# --------------------------------------------------------------------------
# (1) Unified alias-migration registry + one FutureWarning per used legacy key.
# --------------------------------------------------------------------------
def test_alias_registry_engine_applies_rename_rows_generically():
    from dptb.configuration import _apply_alias_registry, _AliasHit, _Rename

    container = {"legacy_a": 1, "legacy_b": 2, "canon_c": 3}
    changes: list = []
    _apply_alias_registry(
        container,
        [
            _Rename("canon_a", ("legacy_a",), "9.9"),
            _Rename("canon_b", "legacy_b", "9.9"),
        ],
        changes=changes,
    )
    assert container == {"canon_a": 1, "canon_b": 2, "canon_c": 3}
    assert {hit.legacy for hit in changes} == {"legacy_a", "legacy_b"}
    assert all(isinstance(hit, _AliasHit) for hit in changes)
    assert all(hit.removal_version == "9.9" for hit in changes)


def test_every_canonicalizer_is_backed_by_a_registry_of_alias_rules():
    from dptb.configuration import (
        _AliasRule,
        _EMBEDDING_REGISTRY,
        _FLOW_REGISTRY,
        _PREDICTION_REGISTRY,
        _TRAIN_ENDPOINT_REGISTRY,
    )

    for registry in (
        _FLOW_REGISTRY,
        _EMBEDDING_REGISTRY,
        _PREDICTION_REGISTRY,
        _TRAIN_ENDPOINT_REGISTRY,
    ):
        assert len(registry) >= 1
        assert all(isinstance(rule, _AliasRule) for rule in registry)


def test_emit_alias_warnings_is_one_per_legacy_key():
    from dptb.configuration import _AliasHit, _emit_alias_warnings

    hits = [
        _AliasHit("dup_key", "canon", "2.3"),
        _AliasHit("dup_key", "canon", "2.3"),
        _AliasHit("other_key", "canon", "2.3"),
    ]
    with pytest.warns(FutureWarning) as record:
        _emit_alias_warnings(hits, enabled=True)
    messages = [str(w.message) for w in record]
    assert len(messages) == 2
    assert sum("'dup_key'" in m for m in messages) == 1
    assert sum("'other_key'" in m for m in messages) == 1


def test_legacy_flow_aliases_resolve_and_warn_once_per_key():
    with pytest.warns(FutureWarning) as record:
        out = canonicalize_flow_options(
            {"overlap_huckel_k": 9.0, "dftb_skdata": "sentinel"},
            warn_deprecated=True,
        )
    assert out["huckel_k"] == pytest.approx(9.0)
    assert out["prior_skdata"] == "sentinel"
    assert "overlap_huckel_k" not in out
    assert "dftb_skdata" not in out

    messages = [str(w.message) for w in record if w.category is FutureWarning]
    assert sum("'overlap_huckel_k'" in m for m in messages) == 1
    assert sum("'dftb_skdata'" in m for m in messages) == 1
    # Each deprecation names the removal version.
    assert all("2.3" in m for m in messages)


def test_endpoint_truth_table_alias_warns_once_per_key():
    with pytest.warns(FutureWarning) as record:
        canonical = canonicalize_training_config(
            {
                "train_options": {
                    "log_single_model_compatible_loss": False,
                    "log_single_model_compatible_loss_mode": "reduce",
                }
            },
            warn_deprecated=True,
        )["train_options"]
    # Truth table preserved: enabled=False forces full_forward regardless of mode.
    assert canonical["endpoint_loss_mode"] == "full_forward"

    messages = [str(w.message) for w in record if w.category is FutureWarning]
    assert sum("'log_single_model_compatible_loss'" in m for m in messages) == 1
    assert sum("'log_single_model_compatible_loss_mode'" in m for m in messages) == 1


def test_warn_deprecated_false_emits_no_warnings():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning becomes an error
        out = canonicalize_flow_options(
            {"overlap_huckel_k": 9.0, "strict_h0": True},
            warn_deprecated=False,
        )
    assert out["huckel_k"] == pytest.approx(9.0)
    assert out["missing_h0_policy"] == "error"
