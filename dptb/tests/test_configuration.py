"""Configuration canonicalization, schema acceptance and CLI parsing.

Old option names must resolve all the way to the runtime objects, conflicting
spellings are rejected, legacy boolean combinations keep their meaning, and
nested option groups normalize exactly like their flat keys.
"""
from __future__ import annotations

import copy
import warnings

import pytest

from dptb.configuration import (
    DEPRECATED_TRAIN_OPTION_KEYS,
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
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.utils.argcheck import (
    TRAIN_OPTION_GROUP_MEMBERS,
    flow_options,
    reference_data_sub,
    train_data_sub,
    train_options,
    validation_data_sub,
)
# Aliased so pytest does not collect the schema builder as a test case.
from dptb.utils.argcheck import test_data_sub as _test_data_sub


def _schema_then_runtime(raw):
    canonical = canonicalize_training_config(
        {"train_options": {"flow_options": raw}}, warn_deprecated=False
    )["train_options"]["flow_options"]
    schema = flow_options()
    normalized = schema.normalize_value(canonical)
    schema.check_value(normalized, strict=True)
    return normalized, HamiltonianCFM(normalized)


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


# --------------------------------------------------------------------------
# flow options
# --------------------------------------------------------------------------
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


@pytest.mark.parametrize(
    ("canonical", "alias", "old_default", "custom"),
    [
        ("huckel_k", "overlap_huckel_k", 1.75, 9.0),
        ("huckel_edge_channel_scale", "overlap_huckel_edge_channel_scale", None, [2.0]),
        ("prior_skdata", "dftb_skdata", "", "sentinel-skdata"),
        ("physical_prior_jitter_sigma", "prior_jitter_sigma", 0.0, 0.25),
    ],
)
def test_legacy_checkpoint_migration_resolves_only_schema_default_alias_collisions(
    canonical, alias, old_default, custom
):
    for stored in ({canonical: old_default, alias: custom}, {canonical: custom, alias: old_default}):
        migrated = migrate_legacy_checkpoint_train_options(
            {"flow_options": stored}, warn_deprecated=False
        )["flow_options"]
        assert migrated[canonical] == custom
        assert alias not in migrated


@pytest.mark.parametrize(
    ("canonicalize", "raw", "key"),
    [
        (canonicalize_flow_options, {"huckel_k": 2.0, "overlap_huckel_k": 3.0}, "huckel_k"),
        # a canonical value equal to the old schema default still conflicts
        (canonicalize_flow_options, {"huckel_k": 1.75, "overlap_huckel_k": 9.0}, "huckel_k"),
        (
            migrate_legacy_checkpoint_train_options,
            {"flow_options": {"huckel_k": 2.0, "overlap_huckel_k": 3.0}},
            "huckel_k",
        ),
        (
            canonicalize_flow_options,
            {"meanflow_aggressive": True, "meanflow": {"aggressive": False}},
            "meanflow_aggressive",
        ),
        (
            canonicalize_training_config,
            {"train_options": {"endpoint_loss_mode": "reduce",
                               "log_single_model_compatible_loss_mode": "full_forward"}},
            "endpoint_loss_mode",
        ),
        (
            canonicalize_training_config,
            {"train_options": {"endpoint_loss_mode": "sometimes"}},
            "endpoint_loss_mode",
        ),
        (
            canonicalize_flow_options,
            {"validation_flow_metrics": ["random_t"], "log_validation_random_t_loss": False},
            "validation_flow_metrics",
        ),
        (
            canonicalize_embedding_options,
            {"method": "lem_moe_v3_h0", "fallback_to_hamiltonian": True,
             "h0_fallback_to_hamiltonian": False},
            "fallback_to_hamiltonian",
        ),
        (
            canonicalize_embedding_options,
            {"method": "lem_moe_v3_h0", "h0_init_scope": "bogus"},
            "h0_init_scope",
        ),
        (
            canonicalize_embedding_options,
            {"method": "lem_moe_v3_prior", "prior_init_scope": "bogus"},
            "prior_init_scope",
        ),
        # the P2 embedding enables soft-edge memory by default, which needs a prior scope
        (
            canonicalize_embedding_options,
            {"method": "lem_moe_v3_prior", "prior_init_scope": "none"},
            "soft_edge_memory",
        ),
        (
            canonicalize_training_config,
            {"train_options": {"use_ddp": False, "distributed": {"use_ddp": True}}},
            "use_ddp",
        ),
    ],
)
def test_conflicting_or_invalid_options_fail_closed(canonicalize, raw, key):
    with pytest.raises(ValueError, match=key):
        canonicalize(raw, warn_deprecated=False)


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
    migrated = migrate_legacy_checkpoint_train_options(legacy, warn_deprecated=False)

    for options in (canonical, migrated):
        assert options["endpoint_loss_mode"] == expected
        assert "log_single_model_compatible_loss" not in options
        assert "log_single_model_compatible_loss_mode" not in options


def test_block_te_schema_default_resolves_to_block_mode():
    normalized, flow = _schema_then_runtime({"enabled": True, "prior": "block_te"})
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


# --------------------------------------------------------------------------
# embedding / prediction options
# --------------------------------------------------------------------------
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

    pair_default = canonicalize_embedding_options({"method": "lem_pair"}, warn_deprecated=False)
    assert pair_default["h0_init_scope"] == "both"

    pair = canonicalize_embedding_options(
        {
            "method": "lem_pair",
            "use_h0_init": True,
            "use_h0_node_init": False,
            "use_h0_edge_init": True,
        },
        warn_deprecated=False,
    )
    assert pair["h0_init_scope"] == "edge"
    assert "use_h0_init" not in pair
    assert "use_h0_node_init" not in pair
    assert "use_h0_edge_init" not in pair

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


@pytest.mark.parametrize(("canonical", "legacy_alias"), [(True, False), (False, True)])
def test_checkpoint_h0_fallback_alias_preserves_legacy_runtime_precedence(canonical, legacy_alias):
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
def test_init_scope_preserves_all_legacy_boolean_semantics(enabled, node, edge, expected):
    scope, wrapper_enabled, use_node, use_edge = resolve_init_scope(
        None, enabled=enabled, node=node, edge=edge
    )
    assert scope == expected
    assert wrapper_enabled is (expected != "none")
    assert use_node is (expected in {"both", "node"})
    assert use_edge is (expected in {"both", "edge"})


@pytest.mark.parametrize("scope", ["NONE", "off", "disabled"])
def test_canonical_scope_aliases_are_normalized_before_contract_validation(scope):
    embedding = canonicalize_embedding_options(
        {"method": "lem_moe_v3_h0", "h0_init_scope": scope},
        warn_deprecated=False,
    )
    assert embedding["h0_init_scope"] == "none"


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


@pytest.mark.parametrize(
    ("mode", "accepted"),
    [(None, True), ("", True), ("standard", True), ("triton_complex_exact_grouped_linear", False)],
)
def test_stable_route_mode_guard_accepts_only_standard(mode, accepted):
    from dptb.nn.embedding.lem_moe_v3 import _normalize_stable_standard_compat_mode

    if accepted:
        assert _normalize_stable_standard_compat_mode("so2_m_linear_mode", mode) == "standard"
    else:
        with pytest.raises(ValueError, match="so2_m_linear_mode"):
            _normalize_stable_standard_compat_mode("so2_m_linear_mode", mode)


# --------------------------------------------------------------------------
# restart merge of train options
# --------------------------------------------------------------------------
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

    assert merge_restart_train_options({}, checkpoint) == checkpoint

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


def test_restart_merge_strips_runtime_ddp_keys():
    checkpoint = {
        "optimizer": {"type": "AdamW", "lr": 1.0e-3},
        "ddp_world_size": 2,
        "ddp_rank": 1,
    }
    merged = merge_restart_train_options({}, checkpoint)
    assert "ddp_world_size" not in merged
    assert "ddp_rank" not in merged
    assert merged["optimizer"]["lr"] == 1.0e-3


# --------------------------------------------------------------------------
# train_options schema
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "split_builder",
    [train_data_sub, validation_data_sub, reference_data_sub, _test_data_sub],
)
def test_allow_unbound_prior_source_fingerprint_schema_default_is_false(split_builder):
    field = split_builder().sub_fields["allow_unbound_prior_source_fingerprint"]
    assert field.default is False


@pytest.mark.parametrize(
    ("method", "loss_options", "expected"),
    [
        ("hamil_blockwise_nextham", {"log_feature_compatible_interval": 25},
         {"log_feature_compatible_interval": 25}),
        ("hamil_blockwise_nextham", {}, {"log_feature_compatible_interval": 1}),
        ("hamil_block_abs", {"log_feature_compatible_interval": 7},
         {"log_feature_compatible_interval": 7}),
        (
            "hamil_blockwise_nextham",
            {
                "optimization": "block_mae",
                "block_reduction": "global",
                "complex_reduction": "modulus",
                "log_feature_compatible": True,
                "feature_log_no_grad": True,
                "distributed_log_reduce": True,
                "loss_weight": 10.0,
            },
            {"method": "hamil_blockwise_nextham", "loss_weight": 10.0},
        ),
    ],
)
def test_blockwise_loss_options_are_accepted_with_defaults(method, loss_options, expected):
    cfg = {
        "num_epoch": 1,
        "batch_size": 1,
        "optimizer": {"type": "AdamW", "lr": 1e-3},
        "lr_scheduler": {"type": "rop"},
        "loss_options": {"train": {"method": method, **loss_options}},
    }
    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    train_loss = normalized["loss_options"]["train"]
    for key, value in expected.items():
        assert train_loss[key] == value


@pytest.mark.parametrize(
    ("train_opts", "accepted"),
    [
        # a Muon routing key mentions "force" but is not a geometry loss
        ({"precompute_lem_cutoff_coeffs": True, "muon_force_name_patterns": [],
          "loss_options": {"train": {"method": "hamil_abs"}}}, True),
        ({"precompute_lem_cutoff_coeffs": True,
          "loss_options": {"train": {"method": "hamil_abs", "force_weight": 1.0}}}, False),
    ],
)
def test_cutoff_precompute_rejects_geometry_gradient_losses(train_opts, accepted):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.train_options = train_opts
    if accepted:
        trainer._validate_lem_cutoff_precompute_options()
    else:
        with pytest.raises(ValueError, match="precompute_lem_cutoff_coeffs"):
            trainer._validate_lem_cutoff_precompute_options()


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
        "distributed": {"use_ddp": True, "ddp_backend": "gloo", "expert_data_parallel_size": 2},
        "checkpoint": {"save_freq": 25, "max_ckpt": 8},
        "observers": {"use_tensorboard": True, "monitor_flag": True, "debug_profile": True},
        "runtime": {"cudnn_benchmark": True, "allow_tf32": False, "train_num_workers": 4},
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


@pytest.mark.parametrize("group", sorted(TRAIN_OPTION_GROUP_MEMBERS))
def test_every_declared_group_normalizes_like_its_flat_keys(group):
    """Each schema group, filled with all of its members, flattens onto real flat keys."""
    schema_fields = train_options().sub_fields
    members = {name: copy.deepcopy(schema_fields[name].default)
               for name in TRAIN_OPTION_GROUP_MEMBERS[group]}
    base = {"num_epoch": 1, "loss_options": _KNOWN_GOOD_LOSS}

    flat = _normalized_train_options({**base, **copy.deepcopy(members)})
    nested = _normalized_train_options({**base, group: copy.deepcopy(members)})

    assert group not in nested
    assert nested == flat


def test_nested_group_matching_flat_key_is_accepted():
    canonical = canonicalize_training_config(
        {"train_options": {"save_freq": 25, "checkpoint": {"save_freq": 25, "max_ckpt": 8}}},
        warn_deprecated=False,
    )["train_options"]
    assert canonical["save_freq"] == 25
    assert canonical["max_ckpt"] == 8
    assert "checkpoint" not in canonical


def test_flatten_train_option_groups_is_idempotent():
    once = canonicalize_training_config(
        {"train_options": {"distributed": {"use_ddp": True}}}, warn_deprecated=False
    )
    twice = canonicalize_training_config(once, warn_deprecated=False)
    assert twice == once
    assert twice["train_options"]["use_ddp"] is True
    assert "distributed" not in twice["train_options"]


# --------------------------------------------------------------------------
# deprecation warnings
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("section", "legacy", "expected"),
    [
        ("flow", {"overlap_huckel_k": 9.0, "dftb_skdata": "sentinel"},
         {"huckel_k": 9.0, "prior_skdata": "sentinel"}),
        # enabled=False forces full_forward regardless of the legacy mode
        ("train", {"log_single_model_compatible_loss": False,
                   "log_single_model_compatible_loss_mode": "reduce"},
         {"endpoint_loss_mode": "full_forward"}),
    ],
)
def test_legacy_aliases_resolve_and_warn_once_per_key(section, legacy, expected):
    with pytest.warns(FutureWarning) as record:
        if section == "flow":
            out = canonicalize_flow_options(legacy, warn_deprecated=True)
        else:
            out = canonicalize_training_config(
                {"train_options": legacy}, warn_deprecated=True
            )["train_options"]
    assert {key: out[key] for key in expected} == expected
    messages = [str(w.message) for w in record if w.category is FutureWarning]
    for key in legacy:
        assert key not in out
        assert sum(f"'{key}'" in m for m in messages) == 1


def test_warn_deprecated_false_emits_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        out = canonicalize_flow_options(
            {"overlap_huckel_k": 9.0, "strict_h0": True},
            warn_deprecated=False,
        )
    assert out["huckel_k"] == pytest.approx(9.0)
    assert out["missing_h0_policy"] == "error"


def test_deprecated_train_option_keys_dropped_with_warning():
    cfg = {"train_options": {
        "num_epoch": 1,
        "shared_scheduler_metric": True,
        "independent_expert_scheduler": False,
        "distributed_global_reduce_every": 1,
    }}
    with pytest.warns(FutureWarning):
        out = canonicalize_training_config(cfg)
    for key in DEPRECATED_TRAIN_OPTION_KEYS:
        assert key not in out["train_options"]
    # warn_deprecated=False stays silent but still drops them
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = canonicalize_training_config(cfg, warn_deprecated=False)
    for key in DEPRECATED_TRAIN_OPTION_KEYS:
        assert key not in quiet["train_options"]


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["multi-train", "s2.json", "-i", "s1.pth", "-o", "out", "-lp", "train.log"],
            {"command": "multi-train", "INPUT": "s2.json", "init_model": "s1.pth",
             "output": "out", "log_path": "train.log"},
        ),
        (["train", "input.json"], {"command": "train", "INPUT": "input.json"}),
    ],
)
def test_training_subcommands_parse_input_init_and_logging(argv, expected):
    from dptb.entrypoints.main import main_parser

    args = main_parser().parse_args(argv)
    assert {key: getattr(args, key) for key in expected} == expected
