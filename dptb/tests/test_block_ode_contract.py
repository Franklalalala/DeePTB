"""Block-ODE configuration contract: argcheck hard gates, flow-constructor guards and the
documented topology-authority subclass extension point.

Covers ``validate_block_ode_contract``/``validate_flow_loss_contract`` (the generic
``ao_block_ode`` route, the ``residual_ao_block_ode`` B/te arms and the ``uureal_block_ode``
route) plus the matching ``HamiltonianCFM`` constructor-time guards.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import yaml

from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nnops.flow import HamiltonianCFM
from dptb.nnops.blockwise_metric_space import (
    FEATURE_ENDPOINT_OPTIMIZATION_MODES,
    endpoint_metric_space_for_options,
)
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.utils.argcheck import flow_options, validate_block_ode_contract
from dptb.tests.block_ode_fixtures import (
    CONFIG_DIR,
    FULL_H_TARGET_FIELDS,
    RESIDUAL_TARGET_FIELDS,
    _b_flow,
    _b_te_flow,
    _case,
    _flow,
    _load_b_config,
    _load_te_config,
    _mapper,
    _mutate,
    _uureal_config,
    _uureal_mapper,
    _valid_contract,
)
from dptb.tests.pair_helpers import model_options

_H_B0_BLOCK_ODE_CONFIGS = sorted(CONFIG_DIR.glob("h_b0_block_ode_*.yaml"))


# ---------------------------------------------------------------------------
# Absolute-mode "no H0 current state" marker (tied-irrep start with no H0 init)
# ---------------------------------------------------------------------------
def test_absolute_tied_irrep_allows_no_h0_feature_initialization():
    config = _valid_contract()
    config["train_options"]["flow_options"].update(
        {
            "mode": "absolute",
            "prior": "tied_irrep_gaussian",
            "tied_irrep_mode": "so3_tied",
            "tied_irrep_irreps": "3x0e + 2x1e + 1x2e",
            "tied_irrep_sigma": 1.0,
            "tied_irrep_validation_seed": 20260725,
        }
    )
    config["model_options"]["embedding"]["h0_init_scope"] = "none"
    config["model_options"]["embedding"]["allow_no_h0_current_state"] = True
    assert validate_block_ode_contract(config) is None


def test_residual_mode_still_rejects_no_h0_feature_initialization():
    config = _valid_contract()
    config["model_options"]["embedding"]["h0_init_scope"] = "none"
    with pytest.raises(ValueError, match=r"h0_init_scope='both'.*node and edge H0 initialization"):
        validate_block_ode_contract(config)


def test_absolute_mode_does_not_enter_residual_ao_block_ode_route():
    config = _valid_contract()
    config["train_options"]["flow_options"].update(
        {"mode": "absolute", "output_space": "residual_ao_block_ode", "prior": "tied_irrep_gaussian"}
    )
    with pytest.raises(ValueError, match="requires flow_options.mode='residual'"):
        validate_block_ode_contract(config)


def test_absolute_current_state_marker_builds_no_h0_late_pair_model():
    options = model_options()
    options.update(
        h0_init_scope="none",
        use_h0_init=False,
        use_h0_node_init=False,
        use_h0_edge_init=False,
        allow_no_h0_current_state=True,
        two_stage_pair_enable=True,
    )
    model = LemMoEV3H0(**options)
    assert model.use_h0_init is False
    assert model.allow_no_h0_current_state is True
    assert model.two_stage_pair is not None


def test_no_h0_late_pair_model_still_rejects_without_absolute_marker():
    options = model_options()
    options.update(
        h0_init_scope="none", use_h0_init=False, use_h0_node_init=False, use_h0_edge_init=False,
        two_stage_pair_enable=True,
    )
    with pytest.raises(ValueError, match="allow_no_h0_current_state=true"):
        LemMoEV3H0(**options)


# ---------------------------------------------------------------------------
# strict_certification schema (default + allowed values, contract-level rejection)
# ---------------------------------------------------------------------------
def test_strict_certification_argcheck_default_and_allowed_values():
    schema = flow_options()
    default = schema.normalize_value({"enabled": False})
    schema.check_value(default, strict=True)
    assert default["strict_certification"] == "always"
    for cadence in ("always", "first_batch", "every_n(7)"):
        value = schema.normalize_value({"enabled": False, "strict_certification": cadence})
        schema.check_value(value, strict=True)
        assert value["strict_certification"] == cadence


@pytest.mark.parametrize("cadence", ["", "sometimes", "every_n", "every_n(0)", "every_n(-1)"])
def test_strict_certification_contract_rejects_invalid_cadence(cadence):
    config = _valid_contract()
    config["train_options"]["flow_options"]["strict_certification"] = cadence
    with pytest.raises(ValueError, match="strict_certification"):
        validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# Generic ao_block_ode: acceptance + the shipped configs + the hard-gate /
# cross-product rejection table
# ---------------------------------------------------------------------------
def test_argcheck_accepts_base_and_projected_te_variants():
    base = _valid_contract()
    assert validate_block_ode_contract(base) is None
    projected_te = deepcopy(base)
    projected_te["train_options"]["flow_options"].update(
        {
            "prior": "projected_te", "te_prior_mode": "irrep",
            "node_sigma": 1.0, "edge_sigma": 1.0, "te_prior_sigma": 1.0,
            "te_prior_validation_seed": 20260719,
        }
    )
    assert validate_block_ode_contract(projected_te) is None
    max_seed = deepcopy(projected_te)
    max_seed["train_options"]["flow_options"]["te_prior_validation_seed"] = (1 << 64) - 1
    assert validate_block_ode_contract(max_seed) is None


@pytest.mark.parametrize("path", _H_B0_BLOCK_ODE_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_h_b0_block_ode_config_validates(path):
    """Every shipped block-ODE overlay (generic, B and te arms, tied-irrep) validates as-is."""
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_block_ode_contract(overlay) is None


def _hard_gate_and_cross_product_cases():
    base = _valid_contract()
    projected_te = deepcopy(base)
    projected_te["train_options"]["flow_options"].update(
        {
            "prior": "projected_te", "te_prior_mode": "irrep",
            "node_sigma": 1.0, "edge_sigma": 1.0, "te_prior_sigma": 1.0,
            "te_prior_validation_seed": 20260719,
        }
    )
    cases = []
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["prior"] = "te"
    cases.append((bad, "only prior='zero'.*prior='projected_te'"))
    for option, value, match in (
        ("te_prior_mode", "typewise", "requires explicit te_prior_mode='irrep'"),
        ("node_sigma", 0.0, "finite positive scales"),
        ("edge_sigma", float("nan"), "finite positive scales"),
        ("te_prior_sigma", float("inf"), "finite positive scales"),
        ("node_sigma", True, "finite positive scales"),
        ("te_prior_validation_seed", True, r"integer.*\[0"),
        ("te_prior_validation_seed", 1 << 64, r"integer.*\[0"),
    ):
        bad = deepcopy(projected_te)
        bad["train_options"]["flow_options"][option] = value
        cases.append((bad, match))
    bad = deepcopy(projected_te)
    bad["train_options"]["flow_options"].pop("te_prior_validation_seed")
    cases.append((bad, r"integer.*\[0"))
    for option, value in (("node_sigma", 1.0e-50), ("node_sigma", 1.0e30)):
        bad = deepcopy(projected_te)
        bad["common_options"]["dtype"] = "float32"
        bad["train_options"]["flow_options"][option] = value
        if value > 1.0:
            bad["train_options"]["flow_options"]["te_prior_sigma"] = value
        cases.append((bad, "effective scales"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["target_semantics"] = ""
    cases.append((bad, "explicit absolute_full_h/residual_dh"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["time_conditioning_required"] = False
    cases.append((bad, "time_conditioning_required=true"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["use_flow_time_embedding"] = False
    cases.append((bad, "use_flow_time_embedding=true"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["flow_time_condition_edges"] = False
    cases.append((bad, "both nodes and edges"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["flow_time_allow_missing"] = True
    cases.append((bad, "flow_time_allow_missing=false"))
    bad = deepcopy(base)
    bad["model_options"]["prediction"]["reconstruction"] = "h0_residual"
    cases.append((bad, "prediction.reconstruction='direct'"))
    for section, option, value, match in (
        ("embedding", "method", "lem_moe_v3", "embedding.method='lem_moe_v3_h0'"),
        ("embedding", "output_route", "h_b1", "embedding.output_route='h_b0'"),
        ("embedding", "require_full_block_edge_coverage", False, "full_block_edge_coverage=true"),
        ("prediction", "method", "e3tb", "prediction.method='block_native'"),
        ("prediction", "block_decoder", "cartesian_projector", "block_decoder='expansion_cg'"),
        ("prediction", "blockwise_hamiltonian", False, "blockwise_hamiltonian=true"),
    ):
        bad = deepcopy(base)
        bad["model_options"][section][option] = value
        cases.append((bad, match))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["block_ode"] = False
    cases.append((bad, "distinct mode"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["output_space"] = "ao_block"
    cases.append((bad, "distinct mode"))
    bad = deepcopy(base)
    bad["data_options"]["train"]["residual_hamiltonian"] = True
    cases.append((bad, "conflicts"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["target_semantics"] = "residual_dh"
    cases.append((bad, "target_semantics"))
    return cases


@pytest.mark.parametrize(("bad", "match"), _hard_gate_and_cross_product_cases())
def test_argcheck_hard_gates_and_cross_products_reject(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_block_ode_contract(bad)


@pytest.mark.parametrize("distance_ranges", ([[0.0, 2.0], [2.0, 6.0]], [[1.0, 6.0]], []))
def test_argcheck_rejects_distance_partitioned_block_ode(distance_ranges):
    config = _valid_contract()
    config["train_options"]["distance_ranges"] = distance_ranges
    with pytest.raises(ValueError, match="distance-partitioned experts"):
        validate_block_ode_contract(config)
    config["train_options"]["distance_ranges"] = [[0.0, 6.0]]
    assert validate_block_ode_contract(config) is None


def test_argcheck_rejects_legacy_full_h_names_and_cross_semantic_fields():
    absolute = _valid_contract()
    legacy_fields = {
        "node_block_target_key": "node_full_hamil_blocks",
        "edge_block_target_key": "edge_full_hamil_blocks",
        "node_block_shape_key": "node_full_hamil_block_shape",
        "edge_block_shape_key": "edge_full_hamil_block_shape",
    }
    for option in FULL_H_TARGET_FIELDS:
        for wrong_value in (legacy_fields[option], RESIDUAL_TARGET_FIELDS[option]):
            bad = deepcopy(absolute)
            bad["train_options"]["flow_options"][option] = wrong_value
            with pytest.raises(ValueError, match=option):
                validate_block_ode_contract(bad)

    residual = deepcopy(absolute)
    residual_flow = residual["train_options"]["flow_options"]
    residual_flow["target_semantics"] = "residual_dh"
    residual_flow.update(RESIDUAL_TARGET_FIELDS)
    residual_split = residual["data_options"]["train"]
    residual_split.update(residual_hamiltonian=True, require_full_h_target=False, require_residual_h_target=True)
    assert validate_block_ode_contract(residual) is None
    for option, wrong_value in FULL_H_TARGET_FIELDS.items():
        bad = deepcopy(residual)
        bad["train_options"]["flow_options"][option] = wrong_value
        with pytest.raises(ValueError, match=option):
            validate_block_ode_contract(bad)


def test_argcheck_requires_physical_h0_contract_on_every_split():
    base = _valid_contract()
    base["data_options"]["validation"] = deepcopy(base["data_options"]["train"])
    assert validate_block_ode_contract(base) is None

    for split in ("train", "validation"):
        for option in ("get_Hamiltonian", "get_H0", "require_full_h_target"):
            bad = deepcopy(base)
            bad["data_options"][split].pop(option)
            with pytest.raises(ValueError, match=rf"data_options\.{split}\.{option}"):
                validate_block_ode_contract(bad)

    for unsupported_type in ("DefaultDataset", "HDF5Dataset"):
        bad = deepcopy(base)
        bad["data_options"]["validation"]["type"] = unsupported_type
        with pytest.raises(ValueError, match=r"data_options\.validation\.type"):
            validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["missing_h0_policy"] = "zero"
    with pytest.raises(ValueError, match="missing_h0_policy='error'"):
        validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["data_options"].pop("train")
    with pytest.raises(ValueError, match="configured data_options.train"):
        validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["common_options"]["has_soc"] = True
    with pytest.raises(ValueError, match="non-SOC only"):
        validate_block_ode_contract(bad)

    residual = deepcopy(base)
    residual_flow = residual["train_options"]["flow_options"]
    residual_flow["target_semantics"] = "residual_dh"
    residual_flow.update(RESIDUAL_TARGET_FIELDS)
    for split_options in residual["data_options"].values():
        split_options.update(residual_hamiltonian=True, require_full_h_target=False, require_residual_h_target=True)
    assert validate_block_ode_contract(residual) is None

    for split in ("train", "validation"):
        bad = deepcopy(residual)
        bad["data_options"][split].pop("require_residual_h_target")
        with pytest.raises(ValueError, match=rf"data_options\.{split}\.require_residual_h_target"):
            validate_block_ode_contract(bad)
    residual["data_options"]["validation"]["require_full_h_target"] = True
    with pytest.raises(ValueError, match="require_full_h_target must be false"):
        validate_block_ode_contract(residual)


@pytest.mark.parametrize(("dtype", "maximum_atol"), (("float32", 2.0e-5), ("float64", 1.0e-10)))
def test_argcheck_caps_strict_inverse_tolerance_by_dtype(dtype, maximum_atol):
    base = _valid_contract()
    base["common_options"]["dtype"] = dtype
    base["train_options"]["flow_options"]["block_inverse_atol"] = maximum_atol
    assert validate_block_ode_contract(base) is None
    for invalid_atol in (maximum_atol * (1.0 + 1.0e-6), 1.0e6):
        bad = deepcopy(base)
        bad["train_options"]["flow_options"]["block_inverse_atol"] = invalid_atol
        with pytest.raises(ValueError, match=rf"{dtype} maximum"):
            validate_block_ode_contract(bad)
    bad_dtype = deepcopy(base)
    bad_dtype["common_options"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="dtype='float32' or 'float64'"):
        validate_block_ode_contract(bad_dtype)


def test_argcheck_and_ctor_reject_fractional_or_empty_steps_and_nan_atol():
    idp, _data, _codec, _h0 = _case()
    base = _flow(idp).options
    for bad_steps in ([1.5], []):
        options = dict(base)
        options["validation_ode_steps"] = bad_steps
        with pytest.raises(ValueError, match="validation_ode_steps"):
            HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    options = dict(base)
    options["block_inverse_atol"] = float("nan")
    with pytest.raises(ValueError, match="block_inverse_atol"):
        HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    options = dict(base)
    options["block_ode"] = False
    with pytest.raises(ValueError, match="distinct mode"):
        HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    config = _valid_contract()
    config["train_options"]["flow_options"]["validation_ode_steps"] = [1.5]
    with pytest.raises(ValueError, match="validation_ode_steps"):
        validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# HamiltonianCFM constructor guards (generic route)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"prediction_add_h0": True}, "prediction_add_h0=true.*reconstruction"),
        ({"prior": "te"}, "only prior='zero'.*prior='projected_te'"),
        ({"prior": "projected_te", "te_prior_mode": "typewise"}, "requires te_prior_mode='irrep'"),
        ({"prior": "projected_te", "te_prior_sigma": 0.0}, "finite positive scales"),
        ({"prior": "projected_te", "node_sigma": True}, "finite positive scales"),
        ({"prior": "projected_te", "te_prior_validation_seed": 1 << 64}, "validation_seed.*\\[0"),
        ({"target_semantics": ""}, "explicit target_semantics"),
        ({"time_conditioning_required": False}, "time_conditioning_required=true"),
        ({"validation_ode_steps": [1, 5]}, r"drawn from \[1, 3\]"),
    ],
)
def test_block_ode_constructor_guards(updates, match):
    idp, _, _, _ = _case()
    with pytest.raises(ValueError, match=match):
        options = _flow(idp).options.copy()
        options.update(updates)
        HamiltonianCFM(options, idp=idp, dtype=torch.float64)


@pytest.mark.parametrize("node_sigma,te_prior_sigma", [(1.0e-50, 1.0), (1.0e30, 1.0e30)])
def test_projected_te_effective_scale_must_be_representable_in_dtype(node_sigma, te_prior_sigma):
    idp, _, _, _ = _case()
    options = _flow(idp).options.copy()
    options.update(prior="projected_te", node_sigma=node_sigma, te_prior_sigma=te_prior_sigma)
    with pytest.raises(ValueError, match="effective scales"):
        HamiltonianCFM(options, idp=idp, dtype=torch.float32)


def test_block_inverse_default_tolerance_is_dtype_aware():
    idp, _, _, _ = _case()
    base = dict(_flow(idp).options, block_inverse_atol=None)
    base.pop("block_inverse_atol")
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float64).block_inverse_atol == 1e-10
    assert HamiltonianCFM(base, idp=idp, dtype=torch.float32).block_inverse_atol == 2e-5


# ---------------------------------------------------------------------------
# Endpoint metric space: block_ode + a feature/rme-space criterion must fail
# closed at configuration time (both the generic and residual_ao_block_ode
# routes), and must not misfire for block-space losses or non-block-ode flows.
# ---------------------------------------------------------------------------
_BLOCK_SPACE_OPTIMIZATIONS = ["block_mae", "block_l1_rmse", "block_mae_mse"]
_FEATURE_SPACE_OPTIMIZATIONS = sorted(FEATURE_ENDPOINT_OPTIMIZATION_MODES)
_BLOCKWISE_LOSS_METHODS = ["hamil_blockwise_nextham", "hamil_block_abs"]
_METRIC_MATCH = "endpoint_metric_space"


@pytest.mark.parametrize("log_feature_compatible", [False, True])
@pytest.mark.parametrize("optimization", _BLOCK_SPACE_OPTIMIZATIONS + _FEATURE_SPACE_OPTIMIZATIONS)
def test_endpoint_metric_space_pure_function_matches_real_loss_class(optimization, log_feature_compatible):
    expected = endpoint_metric_space_for_options(
        log_feature_compatible=log_feature_compatible, optimization=optimization
    )
    loss = HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"}, optimization=optimization, log_feature_compatible=log_feature_compatible
    )
    assert loss.endpoint_metric_space == expected


def _with_loss(config, *, split="train", method="hamil_blockwise_nextham", **loss_fields):
    config = deepcopy(config)
    loss_options = config["train_options"].setdefault("loss_options", {})
    loss_options[split] = {"method": method, **loss_fields}
    return config


def _non_block_ode_config(**loss_fields):
    return {
        "train_options": {
            "flow_options": {"enabled": True, "prior": "zero", "output_space": "rme"},
            "loss_options": {"train": {"method": "hamil_blockwise_nextham", **loss_fields}},
        },
    }


def _endpoint_metric_space_cases():
    cases = []
    for method in _BLOCKWISE_LOSS_METHODS:
        for split in ("train", "validation"):
            cases.append(
                (_with_loss(_valid_contract(), split=split, method=method, log_feature_compatible=True), False)
            )
    for split in ("train", "validation"):
        cases.append((_with_loss(_load_b_config(), split=split, log_feature_compatible=True), False))
    for optimization in _FEATURE_SPACE_OPTIMIZATIONS:
        cases.append((_with_loss(_valid_contract(), optimization=optimization), False))
        cases.append((_with_loss(_load_b_config(), optimization=optimization), False))
    for optimization in _BLOCK_SPACE_OPTIMIZATIONS:
        cases.append((_with_loss(_valid_contract(), optimization=optimization, log_feature_compatible=False), True))
        cases.append((_with_loss(_load_b_config(), optimization=optimization, log_feature_compatible=False), True))
    cases.append((_valid_contract(), True))
    cases.append((_with_loss(_valid_contract(), method="hamil_abs", log_feature_compatible=True), True))
    cases.append((_non_block_ode_config(log_feature_compatible=True), True))
    for optimization in _FEATURE_SPACE_OPTIMIZATIONS:
        cases.append((_non_block_ode_config(optimization=optimization), True))
    cases.append(
        (
            {
                "train_options": {
                    "loss_options": {
                        "train": {"method": "hamil_blockwise_nextham", "log_feature_compatible": True}
                    }
                }
            },
            True,
        )
    )
    return cases


@pytest.mark.parametrize(("config", "accepted"), _endpoint_metric_space_cases())
def test_block_ode_rejects_non_block_endpoint_metric_space(config, accepted):
    """block_ode (generic or residual_ao_block_ode) + a loss whose endpoint metric space is not
    'block' fails closed; block-space losses, missing loss_options, non-blockwise loss methods
    and non-block-ode flows are all unaffected."""
    if accepted:
        assert validate_block_ode_contract(config) is None
    else:
        with pytest.raises(ValueError, match=_METRIC_MATCH):
            validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# residual_ao_block_ode: B/te arm contract + flow-constructor guards
# ---------------------------------------------------------------------------
_B_ARM_REJECTIONS = [
    (("common_options", "has_soc"), True, "has_soc"),
    (("common_options", "nextham_uureal_mask"), True, "nextham_uureal_mask"),
    (("train_options", "flow_options", "state_space"), "wrong", "state_space"),
    (("train_options", "flow_options", "block_input_adapter"), "wrong", "block_input_adapter"),
    (("train_options", "flow_options", "h0_condition_space"), "wrong", "h0_condition_space"),
    (("train_options", "flow_options", "block_export_final_full_h"), False, "block_export"),
    (("train_options", "flow_options", "t0_probability"), 0.0, "t0_probability"),
    (("train_options", "flow_options", "prediction_add_h0"), True, "prediction_add_h0"),
    (("model_options", "prediction", "add_h0"), True, "add_h0"),
    (("train_options", "flow_options", "node_block_target_key"), "node_full_hamil_target_blocks", "node_block_target_key"),
    (("model_options", "embedding", "use_spatial_residual_block_input"), False, "use_spatial_residual_block_input"),
    (("model_options", "embedding", "use_uureal_residual_block_input"), True, "use_uureal_residual_block_input"),
    (("data_options", "train", "residual_hamiltonian"), False, "residual_hamiltonian"),
    (("data_options", "train", "require_full_h_target"), True, "require_full_h_target"),
    (("data_options", "train", "require_residual_h_target"), True, "require_residual_h_target"),
    (("data_options", "train", "require_residual_from_full_h_target"), False, "require_residual_from_full_h_target"),
    (("data_options", "train", "require_uureal_block_ode"), True, "require_uureal_block_ode"),
    (("train_options", "flow_options", "output_space"), "spatial_residual_block_ode", "output_space"),
]


def test_b_arm_yaml_hyphenated_output_space_alias_normalizes():
    cfg = _mutate(_load_b_config(), ("train_options", "flow_options", "output_space"), "residual-ao-block-ode")
    assert validate_block_ode_contract(cfg) is None


@pytest.mark.parametrize("path,value,match", _B_ARM_REJECTIONS)
def test_b_arm_rejection_matrix(path, value, match):
    cfg = _mutate(_load_b_config(), path, value)
    with pytest.raises(ValueError, match=match):
        validate_block_ode_contract(cfg)


def test_te_arm_rejects_missing_validation_seed():
    cfg = deepcopy(_load_te_config())
    del cfg["train_options"]["flow_options"]["te_prior_validation_seed"]
    with pytest.raises(ValueError, match="te_prior_validation_seed"):
        validate_block_ode_contract(cfg)


def test_residual_flow_ctor_rejects_soc_mapper():
    with pytest.raises((ValueError, NotImplementedError), match="non-SOC"):
        _b_flow(_uureal_mapper({"C": "1s1p"}))


def test_residual_flow_ctor_rejects_absolute_full_h_semantics():
    with pytest.raises(ValueError, match="target_semantics"):
        _b_flow(_mapper(), target_semantics="absolute_full_h")


def test_residual_flow_ctor_accepts_projected_te_prior_with_full_te_options():
    flow = _b_te_flow(_mapper())
    assert flow.prior == "projected_te"
    assert flow.residual_ao_block_ode is True
    assert flow.te_prior_mode == "irrep"
    with pytest.raises(ValueError, match="te_prior_validation_seed"):
        _b_flow(_mapper(), prior="projected_te", te_prior_mode="irrep", node_sigma=1.0, edge_sigma=1.0, te_prior_sigma=1.0)
    with pytest.raises(ValueError, match="projected_te"):
        _b_flow(_mapper(), prior="gaussian")


def test_residual_flow_ctor_requires_block_export_final_full_h():
    with pytest.raises(ValueError, match="exactly once outside"):
        _b_flow(_mapper(), block_export_final_full_h=False)


# ---------------------------------------------------------------------------
# uureal_block_ode: argcheck exception, alias sealing, v2 vocabulary aliases
# ---------------------------------------------------------------------------
def test_uureal_argcheck_exception_is_explicit_and_t0_bound():
    config = _uureal_config()
    assert validate_block_ode_contract(config) is None

    config["train_options"]["flow_options"]["t0_probability"] = 0.0
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(config)
    config["train_options"]["flow_options"]["t0_probability"] = 0.2
    assert validate_block_ode_contract(config) is None
    del config["train_options"]["flow_options"]["t0_probability"]

    config["common_options"]["has_soc"] = False
    with pytest.raises(ValueError, match="has_soc=true"):
        validate_block_ode_contract(config)


def test_uureal_residual_block_ode_alias_cannot_bypass_argcheck_interlocks():
    """``output_space='uureal_residual_block_ode'`` is a runtime alias flow.py normalizes into the
    uureal mode; it must hit the identical argcheck interlocks, sealed to the canonical spelling."""
    config = _uureal_config(output_space="uureal_residual_block_ode")
    del config["train_options"]["flow_options"]["block_ode"]
    with pytest.raises(ValueError, match="output_space"):
        validate_block_ode_contract(config)
    config["train_options"]["flow_options"]["block_ode"] = True
    with pytest.raises(ValueError, match="output_space"):
        validate_block_ode_contract(config)
    config["train_options"]["flow_options"]["output_space"] = "uureal_block_ode"
    assert validate_block_ode_contract(config) is None
    config["common_options"]["has_soc"] = False
    with pytest.raises(ValueError, match="has_soc=true"):
        validate_block_ode_contract(config)


def test_uureal_argcheck_accepts_v2_contract_aliases():
    """``state_space=nextham_uureal_delta_block``/``h0_condition_space=nextham_uureal_rme`` are the
    V2 vocabulary for the canonical ``residual_ao_block``/``compact_uureal_rme`` markers."""
    config = _uureal_config(
        state_space="nextham_uureal_delta_block", h0_condition_space="nextham_uureal_rme"
    )
    assert validate_block_ode_contract(config) is None
    config["train_options"]["flow_options"]["state_space"] = "not_a_marker"
    with pytest.raises(ValueError, match="state_space"):
        validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# H9: projected_te effective-scale working-interval gates (argcheck + flow ctor).
# Rejects effective scales (node_sigma*te_prior_sigma / edge_sigma*te_prior_sigma) that collapse
# to exact zero (subnormal-adjacent) or overflow (near dtype-max) once the Gaussian radius
# multiplies in, at both layers, for the residual_ao_block_ode te arm.
# ---------------------------------------------------------------------------
_FMAX_FP32 = float(torch.finfo(torch.float32).max)
FP32_OUT_OF_INTERVAL = [2.0**-149, 2.0**-120, _FMAX_FP32 / 2.0]
FP32_IN_INTERVAL = [1.0, 1e-6, 1e6]
_SCALE_MATCH = "working interval|effective scale"


def _te_config_scale(te_prior_sigma, *, dtype="float32"):
    cfg = _mutate(_load_te_config(), ("train_options", "flow_options", "te_prior_sigma"), float(te_prior_sigma))
    if dtype != "float32":
        cfg = _mutate(cfg, ("common_options", "dtype"), dtype)
    return cfg


@pytest.mark.parametrize("scale", FP32_OUT_OF_INTERVAL)
def test_h9_flow_ctor_rejects_fp32_effective_scale_outside_interval(scale):
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        _b_te_flow(_mapper(), dtype=torch.float32, te_prior_sigma=float(scale))


@pytest.mark.parametrize("scale", FP32_IN_INTERVAL)
def test_h9_flow_ctor_accepts_fp32_effective_scale_inside_interval(scale):
    flow = _b_te_flow(_mapper(), dtype=torch.float32, te_prior_sigma=float(scale))
    assert flow.te_prior_sigma == pytest.approx(scale)


@pytest.mark.parametrize("scale", FP32_OUT_OF_INTERVAL)
def test_h9_argcheck_rejects_fp32_effective_scale_outside_interval(scale):
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        validate_block_ode_contract(_te_config_scale(scale))


@pytest.mark.parametrize("scale", FP32_IN_INTERVAL)
def test_h9_argcheck_accepts_fp32_effective_scale_inside_interval(scale):
    assert validate_block_ode_contract(_te_config_scale(scale)) is None


def test_h9_fp64_working_interval_spot_check_both_layers():
    """fp64 uses the wider interval [2**-996, fmax/2**12]: a 2**-1040 subnormal-adjacent scale is
    rejected at both layers, while 1.0 is accepted at both."""
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        _b_te_flow(_mapper(), dtype=torch.float64, te_prior_sigma=2.0**-1040)
    assert _b_te_flow(_mapper(), dtype=torch.float64, te_prior_sigma=1.0).te_prior_sigma == pytest.approx(1.0)
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        validate_block_ode_contract(_te_config_scale(2.0**-1040, dtype="float64"))


# ---------------------------------------------------------------------------
# Documented subclass extension point: a HamiltonianCFM subclass may widen the
# immutable block-topology authority set, and the widened key must be honoured
# end to end (snapshot on entry, restored after a model step).
# ---------------------------------------------------------------------------
_CUSTOM_KEY = "custom_row_identity"


class _TopoExtendedCFM(HamiltonianCFM):
    @staticmethod
    def _block_primary_topology_keys():
        return (*HamiltonianCFM._block_primary_topology_keys(), _CUSTOM_KEY)


def test_subclass_topology_authority_override_is_restored_after_model_overwrite():
    source = {"edge_index": torch.tensor([[0, 1], [1, 0]]), _CUSTOM_KEY: torch.tensor([17])}
    snapshot = _TopoExtendedCFM._snapshot_block_topology(source)
    assert _CUSTOM_KEY in snapshot and int(snapshot[_CUSTOM_KEY].item()) == 17

    state = {"edge_index": torch.tensor([[0, 1], [1, 0]]), _CUSTOM_KEY: torch.tensor([99])}
    _TopoExtendedCFM._restore_block_topology(state, snapshot)
    assert int(state[_CUSTOM_KEY].item()) == 17
