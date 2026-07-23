# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-field, post-dargs-schema validators for the block-ODE flow contract.

Split out of :mod:`dptb.utils.argcheck` (which re-exports both names below,
by name, at their original line position -- see that module's own import
block) to keep the single 500+-line ``validate_block_ode_contract`` function
out of the already-4000+-line ``argcheck.py``. Kept as a plain sibling module
rather than folding ``argcheck.py`` into a package: smaller blast radius for
the same size win, matching the precedent already set by
:mod:`dptb.nnops.blockwise_metric_space` and
:mod:`dptb.nnops.tied_irrep_constants` (both imported below).

Deliberately import-light -- stdlib, :mod:`dptb.configuration`, and the two
dependency-free sibling modules below, nothing else -- for exactly the reason
those two modules' own docstrings give: ``dptb.data``'s package ``__init__``
imports ``dptb.utils.argcheck`` (via ``dptb.data.build``), and ``argcheck.py``
imports the two functions defined here, so anything this module imports at
module scope rides along on every ``dptb.data`` import. Neither
``dptb.nnops.flow`` nor anything that imports ``torch``/e3nn at module scope
may be imported here; e3nn is imported lazily inside
``validate_block_ode_contract``'s ``tied_irrep_gaussian`` branch instead, for
the same reason ``argcheck.py`` always did that.
"""

import math
import re
from numbers import Integral, Number

from dptb.configuration import canonicalize_training_config
# Dependency-free (no torch, no dptb.data): dptb.data's package __init__
# imports dptb.utils.argcheck (via dptb.data.build), so importing anything
# from dptb.nnops.blockwise_nextham_loss or dptb.nnops.loss here would be a
# circular import.  See dptb/nnops/blockwise_metric_space.py's module
# docstring for the full explanation.
from dptb.nnops.blockwise_metric_space import endpoint_metric_space_for_options
# Also dependency-free (no torch, no e3nn): see
# dptb/nnops/tied_irrep_constants.py's module docstring. e3nn/torch
# themselves are only imported lazily, inside validate_block_ode_contract's
# tied_irrep_gaussian branch, to compare tied_irrep_irreps by o3.Irreps
# equivalence without paying that import weight for every argcheck caller.
from dptb.nnops.tied_irrep_constants import TIED_IRREP_CANONICAL_IRREPS


def validate_flow_loss_contract(data):
    """Fail early on ambiguous or non-finite flow component weighting."""
    # Direct callers (tests and downstream launchers) deserve the same
    # pre-schema alias semantics as normalize().  This is idempotent for an
    # already-normalized configuration and keeps every subsequent check on the
    # 0715 canonical vocabulary.
    data = canonicalize_training_config(data, warn_deprecated=False)
    train = dict(data.get("train_options", {}) or {})
    flow = dict(train.get("flow_options", {}) or {})
    node_weight = float(flow.get("node_weight", 1.0))
    edge_weight = float(flow.get("edge_weight", 1.0))
    for name, value in (("node_weight", node_weight), ("edge_weight", edge_weight)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"flow_options.{name} must be finite and non-negative, got {value!r}"
            )
    if node_weight == 0.0 and edge_weight == 0.0:
        raise ValueError("flow_options.node_weight and edge_weight may not both be zero")

    objective = str(flow.get("objective", flow.get("type", "cfm"))).lower().replace(
        "-", "_"
    )
    pixel_meanflow = objective in {
        "pixel_meanflow",
        "pixel_mean_flow",
        "pmf",
        "meanflow",
        "mean_flow",
    }
    reduction = str(flow.get("component_reduction", "global_elements")).lower()
    if (
        reduction == "global_elements"
        and not pixel_meanflow
        and (node_weight != 1.0 or edge_weight != 1.0)
    ):
        raise ValueError(
            "flow_options.component_reduction='global_elements' requires "
            "node_weight=edge_weight=1 for CFM; use equal_components for "
            "node/edge loss multipliers"
        )


# loss_options.<split>.method values that instantiate HamilBlockwiseNexTHamLoss
# (dptb/nnops/blockwise_nextham_loss.py registers the same class under both
# names) and therefore expose an ``endpoint_metric_space``.
_BLOCKWISE_NEXTHAM_LOSS_METHODS = frozenset({"hamil_blockwise_nextham", "hamil_block_abs"})


def _validate_block_ode_loss_endpoint_metric_space(data):
    """block_ode flows only ever publish block-space endpoint statistics.

    ``HamilBlockwiseNexTHamLoss`` declares ``endpoint_metric_space="rme"`` (not
    "block") whenever ``log_feature_compatible=true`` or ``optimization`` is
    ``feature``/``feature_compatible``/``compat`` (see
    ``endpoint_metric_space_for_options``, the single source of truth shared
    with the loss module).  A block-ODE flow's runtime stats are always in
    "block" space, and ``Trainer`` explicitly refuses the direct-criterion
    recompute fallback for block_ode (``and not block_ode`` guards in
    ``_loss_on_batch``/the validation loop), so a "block"/"rme" mismatch is not
    a soft warning: ``compatible_state`` comes back ``None`` and the one-step
    validation endpoint-triplet construction raises.  Catch it here, at
    configuration time, instead of at the first validation step.

    This is schema-legal and was even the documented way to opt into
    cross-representation comparison (see docs/advanced/pixel_meanflow.md), so
    dargs' per-field schema cannot reject it -- only this cross-tree check,
    which sees both flow_options.block_ode and loss_options together, can.
    """

    loss_options_value = dict(data.get("train_options", {}).get("loss_options", {}) or {})
    for split in ("train", "validation"):
        split_loss = loss_options_value.get(split)
        if not isinstance(split_loss, dict):
            continue
        if split_loss.get("method") not in _BLOCKWISE_NEXTHAM_LOSS_METHODS:
            continue
        metric_space = endpoint_metric_space_for_options(
            log_feature_compatible=split_loss.get("log_feature_compatible", False),
            optimization=split_loss.get("optimization", "block_mae"),
        )
        if metric_space != "block":
            raise ValueError(
                f"train_options.loss_options.{split} configures a blockwise "
                f"Hamiltonian loss (method={split_loss.get('method')!r}) whose "
                f"endpoint_metric_space would be {metric_space!r}, not 'block' "
                "(log_feature_compatible=true or optimization in "
                "{'feature', 'feature_compatible', 'compat'}). block_ode flows "
                "only publish block-space endpoint statistics and validation "
                "refuses to fall back to a direct criterion recompute for "
                "block_ode, so this combination crashes at the first "
                "validation step. Set "
                f"train_options.loss_options.{split}.log_feature_compatible=false "
                "and use a block-space optimization (block_mae, "
                "block_l1_rmse, or block_mae_mse), or disable block_ode."
            )


def validate_block_ode_contract(data):
    """Cross-check the full model/data contract that a field schema cannot see."""
    # normalize() already performs this pass before dargs.  Repeat it here so
    # the public validator cannot be bypassed by calling it directly with
    # deprecated aliases (strict_h0, add_h0, use_h0_* or hyphenated enums).
    data = canonicalize_training_config(data, warn_deprecated=False)
    train = dict(data.get("train_options", {}) or {})
    flow = dict(train.get("flow_options", {}) or {})
    output_space = str(flow.get("output_space", "rme")).lower().replace("-", "_")
    requested = bool(flow.get("block_ode", False)) or output_space in {
        "ao_block_ode",
        "block_ode",
        "ao_blocks_ode",
        "uureal_block_ode",
        "spatial_uureal_residual_block_ode",
        "uureal_residual_block_ode",
        "residual_ao_block_ode",
    }
    if not requested:
        return
    if not bool(flow.get("enabled", False)):
        raise ValueError("block_ode requires train_options.flow_options.enabled=true")
    _validate_block_ode_loss_endpoint_metric_space(data)
    # Every runtime alias flow.py normalizes to uureal_block_ode MUST appear in
    # both sets above/below: an alias missing here while block_ode is omitted
    # slips past this whole contract validation yet still activates the mode at
    # runtime (interlock bypass).
    uureal_mode = output_space in {
        "uureal_block_ode",
        "spatial_uureal_residual_block_ode",
        "uureal_residual_block_ode",
    }
    # Non-SOC direct-residual mode: canonical spelling only (no runtime aliases).
    residual_spatial_mode = output_space == "residual_ao_block_ode"
    if uureal_mode:
        expected_output_space = "uureal_block_ode"
    elif residual_spatial_mode:
        expected_output_space = "residual_ao_block_ode"
    else:
        expected_output_space = "ao_block_ode"
    if output_space != expected_output_space or not bool(flow.get("block_ode", False)):
        raise ValueError(
            "block_ode is a distinct mode: set block_ode=true and "
            "output_space='ao_block_ode'; do not reuse the frozen ao_block adapter"
        )
    if str(flow.get("mode", "")).lower() != "residual":
        raise ValueError("block_ode v1 requires flow_options.mode='residual'")
    prior = str(flow.get("prior", "")).lower().replace("-", "_")
    if uureal_mode and prior != "zero":
        raise ValueError("uureal_block_ode requires prior='zero'")
    if residual_spatial_mode and prior not in {
        "zero",
        "projected_te",
        "tied_irrep_gaussian",
    }:
        raise ValueError(
            "residual_ao_block_ode supports only prior='zero', "
            "prior='projected_te', or prior='tied_irrep_gaussian'"
        )
    if not residual_spatial_mode and prior not in {"zero", "projected_te"}:
        raise ValueError(
            "block_ode supports only prior='zero' or explicit prior='projected_te'"
        )
    effective_scale_products = None
    effective_scale_label = None
    if prior == "projected_te":
        # Do not let the general-flow schema default (auto) or an implicit
        # validator default select the block prior.  This route must declare
        # both the irrep sampler and its deterministic validation substream.
        mode = str(flow.get("te_prior_mode", "")).lower().replace("-", "_")
        if mode != "irrep":
            raise ValueError(
                "projected_te block_ode requires explicit te_prior_mode='irrep'; "
                "typewise/auto modes can read target residual scales"
            )
        raw_scales = {
            "node_sigma": flow.get("node_sigma", 1.0),
            "edge_sigma": flow.get("edge_sigma", 1.0),
            "te_prior_sigma": flow.get("te_prior_sigma", 1.0),
        }
        invalid = [
            name
            for name, value in raw_scales.items()
            if isinstance(value, bool)
            or not isinstance(value, Number)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "projected_te block_ode requires finite positive scales; "
                f"invalid options={invalid}"
            )
        projected_scales = {
            name: float(value) for name, value in raw_scales.items()
        }
        effective_scale_products = {
            "node_sigma*te_prior_sigma": projected_scales["node_sigma"]
            * projected_scales["te_prior_sigma"],
            "edge_sigma*te_prior_sigma": projected_scales["edge_sigma"]
            * projected_scales["te_prior_sigma"],
        }
        effective_scale_label = "projected_te block_ode"
        validation_seed = flow.get("te_prior_validation_seed", None)
        if (
            isinstance(validation_seed, bool)
            or not isinstance(validation_seed, int)
            or validation_seed < 0
            or validation_seed > (1 << 64) - 1
        ):
            raise ValueError(
                "projected_te block_ode requires an explicit integer "
                f"te_prior_validation_seed in [0, {(1 << 64) - 1}]"
            )
    if prior == "tied_irrep_gaussian":
        mode = str(flow.get("tied_irrep_mode", "")).lower().replace("-", "_")
        if mode != "so3_tied":
            raise ValueError(
                "tied_irrep_gaussian requires tied_irrep_mode='so3_tied'"
            )
        configured_irreps = str(flow.get("tied_irrep_irreps", ""))
        # Compare via o3.Irreps equivalence, not a literal string diff, so
        # this schema-time gate agrees exactly with flow.py's runtime
        # validate_tied_irrep_options (tied_irrep_gaussian_prior.py): e.g.
        # "3x0e + 2x1e + 2e" and "3x0e + 2x1e + 1x2e" parse to the same
        # o3.Irreps object (bare "2e" is e3nn shorthand for "1x2e") but
        # differ as strings -- PR#31 review finding P2-1. e3nn/torch are
        # imported lazily here rather than at module scope: see
        # dptb/nnops/tied_irrep_constants.py's module docstring for why
        # dptb.utils.argcheck must stay importable without that weight for
        # callers who never validate a tied_irrep_gaussian config.
        from e3nn import o3 as _tied_irrep_o3

        try:
            irreps_match = _tied_irrep_o3.Irreps(
                configured_irreps
            ) == _tied_irrep_o3.Irreps(TIED_IRREP_CANONICAL_IRREPS)
        except Exception as exc:
            raise ValueError(
                "tied_irrep_gaussian requires a valid tied_irrep_irreps "
                f"string; got {configured_irreps!r}"
            ) from exc
        if not irreps_match:
            raise ValueError(
                "tied_irrep_gaussian currently supports exactly "
                f"tied_irrep_irreps={TIED_IRREP_CANONICAL_IRREPS!r}"
            )
        raw_scales = {
            "node_sigma": flow.get("node_sigma", 1.0),
            "edge_sigma": flow.get("edge_sigma", 1.0),
            "tied_irrep_sigma": flow.get("tied_irrep_sigma", 1.0),
        }
        invalid = [
            name
            for name, value in raw_scales.items()
            if isinstance(value, bool)
            or not isinstance(value, Number)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "tied_irrep_gaussian requires finite positive scales; "
                f"invalid options={invalid}"
            )
        tied_scales = {name: float(value) for name, value in raw_scales.items()}
        effective_scale_products = {
            "node_sigma*tied_irrep_sigma": tied_scales["node_sigma"]
            * tied_scales["tied_irrep_sigma"],
            "edge_sigma*tied_irrep_sigma": tied_scales["edge_sigma"]
            * tied_scales["tied_irrep_sigma"],
        }
        effective_scale_label = "tied_irrep_gaussian"
        validation_seed = flow.get("tied_irrep_validation_seed", None)
        if (
            isinstance(validation_seed, bool)
            or not isinstance(validation_seed, int)
            or validation_seed < 0
            or validation_seed > (1 << 64) - 1
        ):
            raise ValueError(
                "tied_irrep_gaussian requires an explicit integer "
                f"tied_irrep_validation_seed in [0, {(1 << 64) - 1}]"
            )
    semantics = str(flow.get("target_semantics", "")).lower().replace("-", "_")
    if semantics not in {"absolute_full_h", "residual_dh"}:
        raise ValueError("block_ode requires explicit absolute_full_h/residual_dh semantics")
    if uureal_mode and semantics != "residual_dh":
        raise ValueError("uureal_block_ode requires target_semantics='residual_dh'")
    if residual_spatial_mode and semantics != "residual_dh":
        raise ValueError("residual_ao_block_ode requires target_semantics='residual_dh'")
    if uureal_mode:
        # These three options are declarative contract markers (validated only;
        # runtime behavior is driven solely by output_space=uureal_block_ode).
        # Accept both the F4 canonical names and the authoritative V2 dataset
        # contract aliases so a config written to either vocabulary validates:
        #   state_space:       residual_ao_block   <-> nextham_uureal_delta_block
        #   h0_condition_space: compact_uureal_rme <-> nextham_uureal_rme
        #   block_input_adapter: direct_cg (shared)
        accepted_mode_options = {
            "state_space": ("residual_ao_block", "nextham_uureal_delta_block"),
            "block_input_adapter": ("direct_cg",),
            "h0_condition_space": ("compact_uureal_rme", "nextham_uureal_rme"),
        }
        for option, accepted in accepted_mode_options.items():
            if str(flow.get(option, "")).lower().replace("-", "_") not in accepted:
                raise ValueError(
                    f"uureal_block_ode requires flow_options.{option} in {accepted!r}"
                )
        if bool(flow.get("block_export_final_full_h", False)):
            raise ValueError(
                "uureal_block_ode keeps full-H export outside the ODE hot path"
            )
        # The rollout always starts at exactly t=0, D=0; with D_t = t*D1 the
        # interior schedule alone never trains that boundary.  An explicitly
        # configured non-positive t0_probability is therefore a
        # misconfiguration; omitting it lets the runtime default (0.15) apply.
        t0_probability = flow.get("t0_probability", None)
        if t0_probability is not None:
            if (
                isinstance(t0_probability, bool)
                or not isinstance(t0_probability, Number)
                or not math.isfinite(float(t0_probability))
                or float(t0_probability) <= 0.0
                or float(t0_probability) >= 1.0
            ):
                raise ValueError(
                    "uureal_block_ode requires 0 < t0_probability < 1 (recommended "
                    "0.1-0.25): the t=0, D=0 inference boundary needs training mass, "
                    "and t0_probability >= 1 collapses every training time to t=0"
                )
    if residual_spatial_mode:
        # Declarative contract markers (validated only; runtime behavior is driven
        # solely by output_space=residual_ao_block_ode). The non-SOC direct-residual
        # mode conditions on the physical H0 RME (spatial_h0_rme), distinct from the
        # uu_real compact_uureal_rme channel.
        accepted_mode_options = {
            "state_space": ("residual_ao_block",),
            "block_input_adapter": ("direct_cg",),
            "h0_condition_space": ("spatial_h0_rme",),
        }
        for option, accepted in accepted_mode_options.items():
            if str(flow.get(option, "")).lower().replace("-", "_") not in accepted:
                raise ValueError(
                    f"residual_ao_block_ode requires flow_options.{option} in {accepted!r}"
                )
        # Unlike uureal_block_ode (which keeps H0 out of its hot path and rejects
        # this flag), the direct-residual mode assembles the final full H = H0 + D
        # exactly once outside the ODE and therefore REQUIRES the flag.
        if not bool(flow.get("block_export_final_full_h", False)):
            raise ValueError(
                "residual_ao_block_ode assembles final full H exactly once outside "
                "the ODE; set block_export_final_full_h=true"
            )
        # The rollout always starts at exactly t=0, D=0; with D_t = t*D1 the
        # interior schedule alone never trains that boundary.  An explicitly
        # configured non-positive t0_probability is therefore a misconfiguration;
        # omitting it lets the runtime default (0.15) apply.
        t0_probability = flow.get("t0_probability", None)
        if t0_probability is not None:
            if (
                isinstance(t0_probability, bool)
                or not isinstance(t0_probability, Number)
                or not math.isfinite(float(t0_probability))
                or float(t0_probability) <= 0.0
                or float(t0_probability) >= 1.0
            ):
                raise ValueError(
                    "residual_ao_block_ode requires 0 < t0_probability < 1 (recommended "
                    "0.1-0.25): the t=0, D=0 inference boundary needs training mass, "
                    "and t0_probability >= 1 collapses every training time to t=0"
                )
    target_fields = {
        "absolute_full_h": {
            "node_block_target_key": "node_full_hamil_target_blocks",
            "edge_block_target_key": "edge_full_hamil_target_blocks",
            "node_block_shape_key": "node_full_hamil_target_block_shape",
            "edge_block_shape_key": "edge_full_hamil_target_block_shape",
        },
        "residual_dh": {
            "node_block_target_key": "node_delta_hamil_blocks",
            "edge_block_target_key": "edge_delta_hamil_blocks",
            "node_block_shape_key": "node_delta_hamil_block_shape",
            "edge_block_shape_key": "edge_delta_hamil_block_shape",
        },
    }[semantics]
    for option, expected in target_fields.items():
        actual = flow.get(option)
        if actual != expected:
            raise ValueError(
                f"block_ode target_semantics={semantics!r} requires "
                f"flow_options.{option}={expected!r}; got {actual!r}"
            )
    if not bool(flow.get("time_conditioning_required", False)):
        raise ValueError("block_ode requires flow_options.time_conditioning_required=true")
    missing_h0_policy = str(flow.get("missing_h0_policy", "error")).lower().replace(
        "-", "_"
    )
    if missing_h0_policy != "error":
        raise ValueError(
            "block_ode requires flow_options.missing_h0_policy='error' "
            "(legacy strict_h0=true); physical H0 may not fall back to a zero base"
        )
    if str(flow.get("block_inverse_mode", "strict")).lower() != "strict":
        raise ValueError("block_ode requires block_inverse_mode='strict'")
    strict_certification = str(
        flow.get("strict_certification", "always")
    ).strip().lower()
    if strict_certification not in {"always", "first_batch"} and re.fullmatch(
        r"every_n\(([1-9][0-9]*)\)", strict_certification
    ) is None:
        raise ValueError(
            "block_ode strict_certification must be 'always', 'first_batch', "
            "or 'every_n(N)' with N >= 1"
        )
    common = dict(data.get("common_options", {}) or {})
    configured_dtype = str(common.get("dtype", "float32")).lower().replace(
        "torch.", ""
    )
    inverse_atol_caps = {"float32": 2.0e-5, "float64": 1.0e-10}
    if configured_dtype not in inverse_atol_caps:
        raise ValueError(
            "block_ode requires common_options.dtype='float32' or 'float64'; "
            f"got {configured_dtype!r}"
        )
    if effective_scale_products is not None:
        # Working interval, NOT the raw representable interval: a scale near the
        # subnormal floor lets the direction*radius product round to exact zero
        # (silently reviving the deterministic zero prior), and a scale near
        # dtype-max overflows once the unbounded Gaussian radius multiplies in.
        # Lower bound = smallest NORMAL positive value x 2**26 headroom; upper
        # bound = dtype max / 2**12 radius headroom.
        working_interval = {
            "float32": (2.0 ** -100, (2.0 - 2.0 ** -23) * 2.0 ** 115),
            "float64": (2.0 ** -996, (2.0 - 2.0 ** -52) * 2.0 ** 1011),
        }
        minimum, maximum = working_interval[configured_dtype]
        invalid_effective = [
            name
            for name, value in effective_scale_products.items()
            if not math.isfinite(value) or not minimum <= abs(value) <= maximum
        ]
        if invalid_effective:
            raise ValueError(
                f"{effective_scale_label} effective scales must be finite and "
                f"inside the safe {configured_dtype} working interval "
                f"[{minimum:.3g}, {maximum:.3g}] (subnormal-adjacent scales can "
                "collapse draws to exact zero; near-max scales can overflow under "
                f"the Gaussian radius); invalid products={invalid_effective}"
            )
    maximum_inverse_atol = inverse_atol_caps[configured_dtype]
    raw_steps = flow.get("validation_ode_steps", [])
    if not raw_steps or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw_steps
    ):
        raise ValueError("block_ode validation_ode_steps must contain integers drawn from [1, 3]")
    steps = {int(value) for value in raw_steps}
    if not steps or not steps.issubset({1, 3}):
        raise ValueError("block_ode validation_ode_steps must be a non-empty subset of [1, 3]")
    configured_atol = flow.get("block_inverse_atol", None)
    if configured_atol is not None:
        configured_atol = float(configured_atol)
        if not math.isfinite(configured_atol) or configured_atol < 0:
            raise ValueError("block_ode block_inverse_atol must be finite and non-negative")
        if configured_atol > maximum_inverse_atol:
            raise ValueError(
                "block_ode block_inverse_atol exceeds the certified "
                f"{configured_dtype} maximum {maximum_inverse_atol:.6g}; "
                f"got {configured_atol:.6g}"
            )

    if uureal_mode:
        if not bool(common.get("has_soc", False)):
            raise ValueError("uureal_block_ode requires common_options.has_soc=true")
        if not bool(common.get("nextham_uureal_mask", False)):
            raise ValueError("uureal_block_ode requires common_options.nextham_uureal_mask=true")
        if bool(common.get("full_soc_prediction", False)):
            raise ValueError("uureal_block_ode requires full_soc_prediction=false")
    elif residual_spatial_mode:
        if bool(common.get("has_soc", False)):
            raise ValueError("residual_ao_block_ode requires common_options.has_soc=false")
        if bool(common.get("nextham_uureal_mask", False)):
            raise ValueError(
                "residual_ao_block_ode requires common_options.nextham_uureal_mask=false"
            )
        if bool(common.get("full_soc_prediction", False)):
            raise ValueError("residual_ao_block_ode requires full_soc_prediction=false")
    elif bool(common.get("has_soc", False)):
        raise ValueError("block_ode v1 is non-SOC only; set common_options.has_soc=false")

    distance_ranges = train.get("distance_ranges", None)
    if distance_ranges is not None:
        valid_full_graph_range = (
            isinstance(distance_ranges, list)
            and len(distance_ranges) == 1
            and isinstance(distance_ranges[0], (list, tuple))
            and len(distance_ranges[0]) == 2
            and not isinstance(distance_ranges[0][0], bool)
            and isinstance(distance_ranges[0][0], Number)
            and math.isfinite(float(distance_ranges[0][0]))
            and float(distance_ranges[0][0]) <= 0.0
        )
        if not valid_full_graph_range:
            raise ValueError(
                "block_ode v1 cannot use distance-partitioned experts because "
                "H-B0 must predict every graph edge; omit distance_ranges or use "
                "one full-graph range whose lower bound is <= 0"
            )

    model = dict(data.get("model_options", {}) or {})
    prediction = dict(model.get("prediction", {}) or {})
    embedding = dict(model.get("embedding", {}) or {})
    embedding_method = str(embedding.get("method", "")).lower()
    if embedding_method not in {"lem_moe_v3_h0", "lem_pair"}:
        raise ValueError(
            "block_ode requires embedding.method='lem_moe_v3_h0' or 'lem_pair'"
        )
    if str(embedding.get("output_route", "")).lower() != "h_b0":
        raise ValueError("block_ode requires embedding.output_route='h_b0'")
    if not bool(embedding.get("require_full_block_edge_coverage", False)):
        raise ValueError(
            "block_ode requires embedding.require_full_block_edge_coverage=true"
        )
    if str(prediction.get("method", "")).lower() != "block_native":
        raise ValueError("block_ode requires prediction.method='block_native'")
    if str(prediction.get("block_decoder", "")).lower() != "expansion_cg":
        raise ValueError("block_ode requires prediction.block_decoder='expansion_cg'")
    if not bool(prediction.get("blockwise_hamiltonian", False)):
        raise ValueError("block_ode requires prediction.blockwise_hamiltonian=true")
    reconstruction = str(prediction.get("reconstruction", "direct")).lower().replace(
        "-", "_"
    )
    if reconstruction != "direct":
        raise ValueError(
            "block_ode requires model_options.prediction.reconstruction='direct' "
            "(legacy model_options.prediction.add_h0=false) to prevent double "
            "reconstruction"
        )
    if not bool(embedding.get("use_flow_time_embedding", False)):
        raise ValueError("block_ode requires embedding.use_flow_time_embedding=true")
    if not bool(embedding.get("flow_time_condition_edges", False)):
        raise ValueError("block_ode requires flow-time conditioning on both nodes and edges")
    if bool(embedding.get("flow_time_allow_missing", True)):
        raise ValueError("block_ode requires embedding.flow_time_allow_missing=false")
    if str(embedding.get("flow_time_key", "flow_time")) != str(
        flow.get("flow_time_key", "flow_time")
    ):
        raise ValueError("block_ode flow and embedding flow_time_key values must match")
    if str(embedding.get("h0_merge_mode", "replace")).lower() != "replace":
        raise ValueError("block_ode requires embedding.h0_merge_mode='replace'")
    h0_init_scope = str(embedding.get("h0_init_scope", "both")).lower().replace(
        "-", "_"
    )
    if h0_init_scope != "both":
        raise ValueError(
            "block_ode requires embedding.h0_init_scope='both' for both node and "
            "edge H0 initialization"
        )
    if bool(embedding.get("use_uureal_residual_block_input", False)) != uureal_mode:
        raise ValueError(
            "uureal_block_ode and embedding.use_uureal_residual_block_input must be enabled together"
        )
    if bool(embedding.get("use_spatial_residual_block_input", False)) != residual_spatial_mode:
        raise ValueError(
            "residual_ao_block_ode and embedding.use_spatial_residual_block_input "
            "must be enabled together"
        )

    data_options_value = data.get("data_options", {}) or {}
    if not isinstance(data_options_value.get("train"), dict):
        raise ValueError("block_ode requires a configured data_options.train split")
    configured_splits = {
        split: data_options_value[split]
        for split in ("train", "validation", "reference", "test")
        if isinstance(data_options_value.get(split), dict)
    }
    expected_residual = semantics == "residual_dh" and not uureal_mode
    expected_full_h_target = semantics == "absolute_full_h"
    # require_residual_h_target gates the h0_residual-semantics records; the
    # residual_spatial mode reads absolute_full_h records and instead opts into
    # require_residual_from_full_h_target. Decouple both from expected_residual,
    # which still gates residual_hamiltonian for the generic AND residual_spatial
    # modes (both materialize D1 = H - H0 on the fly). expected_residual_h_target
    # is identical to expected_residual for every non-spatial mode.
    expected_residual_h_target = expected_residual and not residual_spatial_mode
    expected_residual_from_full_h = residual_spatial_mode
    for split, split_options in configured_splits.items():
        path = f"data_options.{split}"
        dataset_type = str(split_options.get("type", "DefaultDataset"))
        if dataset_type != "LMDBDataset":
            raise ValueError(
                f"{path}.type must be 'LMDBDataset' for block_ode; "
                "other dataset backends do not implement the physical-H0 and "
                "versioned AO-block target contract"
            )
        if not bool(split_options.get("get_Hamiltonian", False)):
            raise ValueError(f"{path}.get_Hamiltonian must be true for block_ode")
        if not bool(split_options.get("get_H0", False)):
            raise ValueError(
                f"{path}.get_H0 must be true for block_ode physical-H0 initialization"
            )
        actual_residual = bool(split_options.get("residual_hamiltonian", False))
        if actual_residual != expected_residual:
            raise ValueError(
                f"{path}.residual_hamiltonian={actual_residual} conflicts with "
                f"block_ode target_semantics={semantics!r}"
            )
        actual_full_h_target = bool(
            split_options.get("require_full_h_target", False)
        )
        if actual_full_h_target != expected_full_h_target:
            raise ValueError(
                f"{path}.require_full_h_target must be "
                f"{str(expected_full_h_target).lower()} for block_ode "
                f"target_semantics={semantics!r}"
            )
        actual_residual_h_target = bool(
            split_options.get("require_residual_h_target", False)
        )
        if actual_residual_h_target != expected_residual_h_target:
            raise ValueError(
                f"{path}.require_residual_h_target must be "
                f"{str(expected_residual_h_target).lower()} for block_ode "
                f"target_semantics={semantics!r}"
            )
        actual_residual_from_full_h = bool(
            split_options.get("require_residual_from_full_h_target", False)
        )
        if actual_residual_from_full_h != expected_residual_from_full_h:
            raise ValueError(
                f"{path}.require_residual_from_full_h_target must be "
                f"{str(expected_residual_from_full_h).lower()} for block_ode "
                f"target_semantics={semantics!r}"
            )
        actual_uureal = bool(split_options.get("require_uureal_block_ode", False))
        if actual_uureal != uureal_mode:
            raise ValueError(
                f"{path}.require_uureal_block_ode must be {str(uureal_mode).lower()}"
            )
