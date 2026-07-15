"""Canonical configuration adapters for recently evolved DeePTB routes.

The flow, H0, and P2 implementations accumulated compatibility aliases faster
than the dargs schema could safely resolve them.  In particular, dargs inserts
canonical defaults before runtime constructors inspect aliases, so a valid
legacy value can be silently hidden by a default.  This module canonicalizes
raw input *before* schema defaults are applied and is also used by constructors
that are called directly in tests or downstream code.

Every legacy alias is expressed as one row of a small migration registry and
applied through the generic :func:`_apply_alias_registry` engine, which records
a per-key deprecation so exactly one :class:`FutureWarning` fires for each used
legacy key.  Keep this module dependency-free: configuration normalization
happens before Torch/model imports in several entrypoints, and
``dptb.utils.argcheck`` imports this module (so it must not import argcheck).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import warnings


_MISSING = object()

# Default DeePTB version in which the deprecated aliases below are scheduled to
# be removed.  Individual registry rows may override this if their timeline
# diverges.
_ALIAS_REMOVAL_VERSION = "2.3"


# The 0711 schema normalized both sides of these alias pairs into saved
# checkpoints.  Consequently a user-provided alias could be stored next to an
# injected canonical default (and vice versa).  Keep this table deliberately
# narrow: raw/new configuration canonicalization remains strictly fail-closed.
_LEGACY_CHECKPOINT_FLOW_ALIAS_DEFAULTS = (
    ("huckel_k", "overlap_huckel_k", 1.75),
    (
        "huckel_edge_channel_scale",
        "overlap_huckel_edge_channel_scale",
        None,
    ),
    ("prior_skdata", "dftb_skdata", ""),
    ("physical_prior_jitter_sigma", "prior_jitter_sigma", 0.0),
)


@dataclass(frozen=True)
class _AliasHit:
    """One recorded use of a deprecated key during canonicalization."""

    legacy: str
    canonical: str
    removal_version: str


def _same_value(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def _merge_aliases(
    values: MutableMapping[str, Any],
    canonical: str,
    aliases: Iterable[str],
    removal_version: str,
    *,
    changes: List[_AliasHit],
) -> None:
    """Move aliases onto one key and reject conflicting explicit values."""

    for alias in aliases:
        if alias not in values:
            continue
        alias_value = values.pop(alias)
        if canonical not in values:
            values[canonical] = alias_value
        else:
            canonical_value = values[canonical]
            if _same_value(canonical_value, alias_value):
                pass
            else:
                raise ValueError(
                    f"Conflicting configuration values for {canonical!r} "
                    f"and deprecated alias {alias!r}: "
                    f"{canonical_value!r} != {alias_value!r}."
                )
        changes.append(_AliasHit(alias, canonical, removal_version))


# ---------------------------------------------------------------------------
# Generic alias-migration registry.
#
# The five historical canonicalization branches (flow options, endpoint loss
# mode, legacy-checkpoint flow defaults, embedding switches, prediction
# reconstruction) are all expressed as ordered lists of rows and applied by the
# same engine.  A row is either a simple :class:`_Rename` (a fail-closed alias
# merge onto one canonical key) or a :class:`_Transform` wrapping a semantic
# collapse that cannot be reduced to a rename (e.g. the endpoint-loss-mode truth
# table or the init-scope boolean fan-in).  Every row carries a removal_version
# so the emitted FutureWarnings can name it.
# ---------------------------------------------------------------------------
class _AliasRule:
    """Base row of the alias-migration registry."""

    removal_version: str

    def apply(self, container: MutableMapping[str, Any], changes: List[_AliasHit]) -> None:
        raise NotImplementedError


class _Rename(_AliasRule):
    """Merge one or more legacy keys onto a canonical key (fail-closed)."""

    def __init__(
        self,
        canonical: str,
        legacy: Union[str, Sequence[str]],
        removal_version: str = _ALIAS_REMOVAL_VERSION,
    ) -> None:
        self.canonical = canonical
        self.legacy: Tuple[str, ...] = (
            (legacy,) if isinstance(legacy, str) else tuple(legacy)
        )
        self.removal_version = removal_version

    def apply(self, container: MutableMapping[str, Any], changes: List[_AliasHit]) -> None:
        _merge_aliases(
            container, self.canonical, self.legacy, self.removal_version, changes=changes
        )


class _Transform(_AliasRule):
    """Wrap a semantic collapse that records its own consumed legacy keys."""

    def __init__(
        self,
        fn: Callable[[MutableMapping[str, Any], List[_AliasHit], str], None],
        *,
        removal_version: str = _ALIAS_REMOVAL_VERSION,
    ) -> None:
        self.fn = fn
        self.removal_version = removal_version

    def apply(self, container: MutableMapping[str, Any], changes: List[_AliasHit]) -> None:
        self.fn(container, changes, self.removal_version)


def _apply_alias_registry(
    container: MutableMapping[str, Any],
    rules: Iterable[_AliasRule],
    *,
    changes: List[_AliasHit],
) -> None:
    """Apply every registry row to ``container`` in order."""

    for rule in rules:
        rule.apply(container, changes)


def _emit_alias_warnings(changes: Iterable[_AliasHit], *, enabled: bool) -> None:
    """Emit exactly one FutureWarning per used legacy key."""

    if not enabled:
        return
    seen: set[str] = set()
    for hit in changes:
        if hit.legacy in seen:
            continue
        seen.add(hit.legacy)
        warnings.warn(
            f"Deprecated configuration key {hit.legacy!r} was canonicalized to "
            f"{hit.canonical!r}; it will be removed in DeePTB {hit.removal_version}.",
            FutureWarning,
            stacklevel=3,
        )


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _scope_from_legacy(
    *,
    enabled: Optional[bool],
    node: Optional[bool],
    edge: Optional[bool],
) -> str:
    master = True if enabled is None else bool(enabled)
    use_node = True if node is None else bool(node)
    use_edge = True if edge is None else bool(edge)
    if not master:
        return "none"
    if use_node and use_edge:
        return "both"
    if use_node:
        return "node"
    if use_edge:
        return "edge"
    # The legacy wrapper could stay enabled with both tensor injections off.
    # H0 uses this for flow-time conditioning; P2 can use it for edge memory.
    return "auxiliary"


def resolve_init_scope(
    scope: Optional[str],
    *,
    enabled: Optional[bool] = None,
    node: Optional[bool] = None,
    edge: Optional[bool] = None,
    option_name: str = "init_scope",
) -> Tuple[str, bool, bool, bool]:
    """Resolve one scope enum and optional legacy booleans.

    Returns ``(scope, enabled, use_node, use_edge)``.
    """

    legacy_present = any(value is not None for value in (enabled, node, edge))
    legacy_scope = _scope_from_legacy(enabled=enabled, node=node, edge=edge)
    resolved = legacy_scope if scope is None else _normalized_name(scope)
    aliases = {
        "all": "both",
        "off": "none",
        "disabled": "none",
        "wrapper_only": "auxiliary",
        "conditioning_only": "auxiliary",
        "memory_only": "auxiliary",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"both", "node", "edge", "auxiliary", "none"}:
        raise ValueError(
            f"{option_name} must be 'both', 'node', 'edge', 'auxiliary', "
            "or 'none'; "
            f"got {scope!r}."
        )
    if scope is not None and legacy_present and resolved != legacy_scope:
        raise ValueError(
            f"Conflicting {option_name}={resolved!r} and legacy init flags "
            f"(equivalent scope {legacy_scope!r})."
        )
    return resolved, resolved != "none", resolved in {"both", "node"}, resolved in {
        "both",
        "edge",
    }


# ---------------------------------------------------------------------------
# flow_options registry rows.
# ---------------------------------------------------------------------------
def _flow_meanflow_subtree(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    meanflow = out.get("meanflow", {}) or {}
    if not isinstance(meanflow, Mapping):
        raise TypeError("flow_options.meanflow must be a mapping.")
    meanflow = deepcopy(dict(meanflow))
    _merge_aliases(meanflow, "du_dt_backend", ("jvp_backend",), rv, changes=changes)
    _merge_aliases(
        meanflow, "jvp_fallback", ("jvp_fallback_to_finite_difference",), rv, changes=changes
    )
    _merge_aliases(meanflow, "objective", ("loss_objective",), rv, changes=changes)
    for dead_flag in (
        "log_compatible_loss",
        "log_train_compatible_loss",
        "log_validation_compatible_loss",
        "compatible_loss_to_legacy_keys",
    ):
        if dead_flag in meanflow:
            meanflow.pop(dead_flag)
            changes.append(_AliasHit(f"meanflow.{dead_flag}", "always_on", rv))

    top_profile = out.pop("meanflow_profile", _MISSING)
    if top_profile is not _MISSING:
        if "profile" in meanflow and _normalized_name(meanflow["profile"]) != _normalized_name(
            top_profile
        ):
            raise ValueError("Conflicting flow_options.meanflow.profile and meanflow_profile.")
        meanflow["profile"] = top_profile
        changes.append(_AliasHit("meanflow_profile", "meanflow.profile", rv))
    aggressive_values = []
    if "meanflow_aggressive" in out:
        aggressive_values.append(("meanflow_aggressive", bool(out.pop("meanflow_aggressive"))))
    if "aggressive" in meanflow:
        aggressive_values.append(("meanflow.aggressive", bool(meanflow.pop("aggressive"))))
    if len({value for _name, value in aggressive_values}) > 1:
        raise ValueError(
            "Conflicting flow_options.meanflow_aggressive and "
            "flow_options.meanflow.aggressive."
        )
    if any(value for _name, value in aggressive_values):
        if "profile" in meanflow and _normalized_name(meanflow["profile"]) != "aggressive":
            raise ValueError("Aggressive MeanFlow flag conflicts with meanflow.profile.")
        meanflow["profile"] = "aggressive"
    for name, _value in aggressive_values:
        changes.append(_AliasHit(name, "meanflow.profile", rv))
    out["meanflow"] = meanflow


def _flow_missing_h0_policy(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    strict = out.pop("strict_h0", _MISSING)
    warn_missing = out.pop("warn_missing_h0", _MISSING)
    if strict is not _MISSING or warn_missing is not _MISSING:
        strict_value = True if strict is _MISSING else bool(strict)
        warn_value = True if warn_missing is _MISSING else bool(warn_missing)
        legacy_policy = "error" if strict_value else "warn_zero" if warn_value else "zero"
        if "missing_h0_policy" in out and _normalized_name(out["missing_h0_policy"]) != legacy_policy:
            raise ValueError(
                "flow_options.missing_h0_policy conflicts with strict_h0/warn_missing_h0."
            )
        out["missing_h0_policy"] = legacy_policy
        if strict is not _MISSING:
            changes.append(_AliasHit("strict_h0", "missing_h0_policy", rv))
        if warn_missing is not _MISSING:
            changes.append(_AliasHit("warn_missing_h0", "missing_h0_policy", rv))
    if "missing_h0_policy" in out:
        policy = _normalized_name(out["missing_h0_policy"])
        policy = {"warn": "warn_zero", "silent_zero": "zero"}.get(policy, policy)
        if policy not in {"error", "warn_zero", "zero"}:
            raise ValueError(
                "flow_options.missing_h0_policy must be 'error', 'warn_zero', or 'zero'."
            )
        out["missing_h0_policy"] = policy


def _flow_validation_metrics(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    metric_order = ("random_t", "one_step", "trajectory")
    metric_aliases = {"t0": "one_step", "euler": "trajectory"}
    configured_metrics = out.get("validation_flow_metrics", _MISSING)
    if configured_metrics is not _MISSING:
        if not isinstance(configured_metrics, (list, tuple, set)):
            raise TypeError("flow_options.validation_flow_metrics must be a list.")
        normalized_metrics = {
            metric_aliases.get(_normalized_name(value), _normalized_name(value))
            for value in configured_metrics
        }
        unknown = normalized_metrics.difference(metric_order)
        if unknown:
            raise ValueError(
                "flow_options.validation_flow_metrics contains unknown values: "
                f"{sorted(unknown)}."
            )
    else:
        normalized_metrics = set(metric_order)

    legacy_metric_flags = {
        "log_validation_random_t_loss": "random_t",
        "log_validation_t0_loss": "one_step",
        "log_validation_flow_euler_loss": "trajectory",
    }
    present_legacy_metrics = {
        metric: bool(out.pop(flag))
        for flag, metric in legacy_metric_flags.items()
        if flag in out
    }
    if present_legacy_metrics:
        legacy_metrics = {
            metric
            for metric in metric_order
            if present_legacy_metrics.get(metric, True)
        }
        if configured_metrics is not _MISSING and normalized_metrics != legacy_metrics:
            raise ValueError(
                "flow_options.validation_flow_metrics conflicts with deprecated "
                "validation logging flags."
            )
        normalized_metrics = legacy_metrics
        for flag, metric in legacy_metric_flags.items():
            if metric in present_legacy_metrics:
                changes.append(_AliasHit(flag, "validation_flow_metrics", rv))
    if configured_metrics is not _MISSING or present_legacy_metrics:
        out["validation_flow_metrics"] = [
            metric for metric in metric_order if metric in normalized_metrics
        ]


def _flow_validation_ode_steps(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    if "validation_ode_steps" in out:
        steps = {int(value) for value in out["validation_ode_steps"] if int(value) > 0}
        steps.add(1)
        out["validation_ode_steps"] = sorted(steps)


def _flow_omit_time_scaling(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    if "omit_time_scaling" in out:
        omit = bool(out.pop("omit_time_scaling"))
        if omit:
            # This reproduces the legacy short-circuit exactly, including when
            # endpoint_weight_power was configured to a non-zero value.
            out["endpoint_weight_power"] = 0.0
        changes.append(_AliasHit("omit_time_scaling", "endpoint_weight_power", rv))


def _flow_dead_top_flags(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    # These switches have been hard invariants since the endpoint-compatible
    # logging fix.  Accept and discard them so old configs keep loading without
    # pretending that ``false`` changes runtime behavior.
    for dead_flag in (
        "log_compatible_loss",
        "log_train_compatible_loss",
        "log_validation_compatible_loss",
        "compatible_loss_to_legacy_keys",
    ):
        if dead_flag in out:
            out.pop(dead_flag)
            changes.append(_AliasHit(dead_flag, "always_on", rv))


def _flow_final_normalize(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    if "objective" in out:
        out["objective"] = _normalized_name(out["objective"])
    if "prior" in out and isinstance(out["prior"], str):
        out["prior"] = _normalized_name(out["prior"])
    if "te_prior_mode" in out:
        out["te_prior_mode"] = _normalized_name(out["te_prior_mode"])


# Ordered registry for ``flow_options``.  Order is load-bearing: pixel_meanflow
# must be merged onto ``meanflow`` before the meanflow subtree is processed, and
# the final normalization must run last.
_FLOW_REGISTRY: Tuple[_AliasRule, ...] = (
    _Rename("objective", ("type",)),
    _Rename("meanflow", ("pixel_meanflow",)),
    _Rename("huckel_k", ("overlap_huckel_k",)),
    _Rename("huckel_edge_channel_scale", ("overlap_huckel_edge_channel_scale",)),
    _Rename("prior_skdata", ("dftb_skdata", "skdata")),
    _Rename("physical_prior_jitter_sigma", ("prior_jitter_sigma",)),
    _Transform(_flow_meanflow_subtree),
    _Transform(_flow_missing_h0_policy),
    _Transform(_flow_validation_metrics),
    _Transform(_flow_validation_ode_steps),
    _Transform(_flow_omit_time_scaling),
    _Transform(_flow_dead_top_flags),
    _Transform(_flow_final_normalize),
)


def canonicalize_flow_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Return one fail-closed representation of ``train_options.flow_options``."""

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("train_options.flow_options must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    _apply_alias_registry(out, _FLOW_REGISTRY, changes=changes)
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return out


def _canonicalize_endpoint_loss_mode(
    options: MutableMapping[str, Any],
    changes: List[_AliasHit],
    removal_version: str,
) -> None:
    """Preserve the legacy boolean+mode execution truth table.

    Before 0714, ``log_single_model_compatible_loss`` was not merely a logging
    switch in MultiTrainer: ``false`` selected the stitched full-forward path
    even when the companion mode was ``reduce``.  Treating the boolean as dead
    would therefore silently change the training algorithm on restart.
    """

    legacy_enabled = options.pop("log_single_model_compatible_loss", _MISSING)
    legacy_mode = options.pop("log_single_model_compatible_loss_mode", _MISSING)
    canonical_mode = options.get("endpoint_loss_mode", _MISSING)

    def _validate(value: Any) -> str:
        mode = _normalized_name(value)
        if mode not in {"reduce", "full_forward"}:
            raise ValueError(
                "train_options.endpoint_loss_mode must be 'reduce' or "
                "'full_forward'."
            )
        return mode

    if canonical_mode is not _MISSING:
        canonical_mode = _validate(canonical_mode)

    legacy_effective = _MISSING
    if legacy_enabled is not _MISSING or legacy_mode is not _MISSING:
        normalized_legacy_mode = (
            "reduce" if legacy_mode is _MISSING else _validate(legacy_mode)
        )
        legacy_effective = (
            normalized_legacy_mode
            if legacy_enabled is _MISSING or bool(legacy_enabled)
            else "full_forward"
        )
        if canonical_mode is not _MISSING and canonical_mode != legacy_effective:
            raise ValueError(
                "Conflicting configuration values: "
                "train_options.endpoint_loss_mode conflicts with deprecated "
                "log_single_model_compatible_loss/mode (legacy effective mode "
                f"{legacy_effective!r})."
            )
        canonical_mode = legacy_effective

    if canonical_mode is not _MISSING:
        options["endpoint_loss_mode"] = canonical_mode
    if legacy_enabled is not _MISSING:
        changes.append(
            _AliasHit("log_single_model_compatible_loss", "endpoint_loss_mode", removal_version)
        )
    if legacy_mode is not _MISSING:
        changes.append(
            _AliasHit(
                "log_single_model_compatible_loss_mode", "endpoint_loss_mode", removal_version
            )
        )


# Registry for the top-level train_options endpoint truth table.
_TRAIN_ENDPOINT_REGISTRY: Tuple[_AliasRule, ...] = (
    _Transform(_canonicalize_endpoint_loss_mode),
)


def migrate_legacy_checkpoint_flow_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Canonicalize flow options saved by the pre-0714 checkpoint schema.

    The old schema inserted defaults for both a canonical key and its alias.
    Only that recognizable default/custom collision is relaxed here.  Two
    distinct non-default values are passed to the normal canonicalizer and
    therefore still fail closed.
    """

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("checkpoint train_options.flow_options must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    for canonical, alias, old_default in _LEGACY_CHECKPOINT_FLOW_ALIAS_DEFAULTS:
        if canonical not in out or alias not in out:
            continue
        canonical_value = out[canonical]
        alias_value = out[alias]
        canonical_is_default = _same_value(canonical_value, old_default)
        alias_is_default = _same_value(alias_value, old_default)
        if canonical_is_default and not alias_is_default:
            out[canonical] = alias_value
            out.pop(alias)
            changes.append(_AliasHit(alias, canonical, _ALIAS_REMOVAL_VERSION))
        elif alias_is_default or _same_value(canonical_value, alias_value):
            out.pop(alias)
            changes.append(_AliasHit(alias, canonical, _ALIAS_REMOVAL_VERSION))

    # This retains the strict conflict check for all unrecognized collisions
    # and also migrates aliases that were not populated by the old schema.
    migrated = canonicalize_flow_options(out, warn_deprecated=warn_deprecated)
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return migrated


def migrate_legacy_checkpoint_train_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Migrate only checkpoint-sourced train options before restart."""

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("checkpoint train_options must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    _apply_alias_registry(out, _TRAIN_ENDPOINT_REGISTRY, changes=changes)
    if "flow_options" in out:
        out["flow_options"] = migrate_legacy_checkpoint_flow_options(
            out.get("flow_options"), warn_deprecated=warn_deprecated
        )
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return out


# ---------------------------------------------------------------------------
# model_options.embedding registry rows.
# ---------------------------------------------------------------------------
_H0_METHODS = {"lem_moe_v3_h0", "lem_moe_v3_edge_h0", "lem_non_linear_h0"}

_SOFT_EDGE_MEMORY_ALIASES = {
    "use_soft_edge_memory": "enabled",
    "soft_edge_memory_num_slots": "num_slots",
    "soft_edge_memory_num_heads": "num_heads",
    "soft_edge_memory_head_dim": "head_dim",
    "soft_edge_memory_temperature": "temperature",
    "soft_edge_memory_dropout": "dropout",
    "soft_edge_memory_gate_mode": "gate_mode",
    "soft_edge_memory_gate_bias": "gate_bias",
    "soft_edge_memory_gate_eps": "gate_eps",
    "soft_edge_memory_zero_init_output": "zero_init_output",
    "soft_edge_memory_input_norm": "input_norm",
    "soft_edge_memory_diagnostics_mode": "diagnostics_mode",
    "soft_edge_memory_diagnostics_sample_size": "diagnostics_sample_size",
}


def _embedding_h0_scope(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    method = _normalized_name(out.get("method", ""))
    if method not in _H0_METHODS and not any(
        key in out
        for key in (
            "h0_init_scope",
            "use_h0_init",
            "use_h0_node_init",
            "use_h0_edge_init",
        )
    ):
        return
    legacy = (
        out.pop("use_h0_init", None),
        out.pop("use_h0_node_init", None),
        out.pop("use_h0_edge_init", None),
    )
    scope, *_ = resolve_init_scope(
        out.get("h0_init_scope"),
        enabled=legacy[0],
        node=legacy[1],
        edge=legacy[2],
        option_name="model_options.embedding.h0_init_scope",
    )
    out["h0_init_scope"] = scope
    for name, value in zip(
        ("use_h0_init", "use_h0_node_init", "use_h0_edge_init"), legacy
    ):
        if value is not None:
            changes.append(_AliasHit(name, "h0_init_scope", rv))


def _embedding_prior(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    method = _normalized_name(out.get("method", ""))
    if method != "lem_moe_v3_prior":
        return
    prior_kind = _normalized_name(out.get("prior_kind", "p2"))
    if prior_kind not in {"p2", "p23"}:
        raise ValueError(
            "lem_moe_v3_prior supports prior_kind='p2' or 'p23'; "
            f"got {prior_kind!r}."
        )
    out["prior_kind"] = prior_kind

    legacy = (
        out.pop("use_prior_init", None),
        out.pop("use_prior_node_init", None),
        out.pop("use_prior_edge_init", None),
    )
    scope, *_ = resolve_init_scope(
        out.get("prior_init_scope"),
        enabled=legacy[0],
        node=legacy[1],
        edge=legacy[2],
        option_name="model_options.embedding.prior_init_scope",
    )
    out["prior_init_scope"] = scope
    for name, value in zip(
        ("use_prior_init", "use_prior_node_init", "use_prior_edge_init"), legacy
    ):
        if value is not None:
            changes.append(_AliasHit(name, "prior_init_scope", rv))

    memory = out.get("soft_edge_memory", {}) or {}
    if not isinstance(memory, Mapping):
        raise TypeError("embedding.soft_edge_memory must be a mapping.")
    memory = deepcopy(dict(memory))
    for old, new in _SOFT_EDGE_MEMORY_ALIASES.items():
        if old not in out:
            continue
        value = out.pop(old)
        if new in memory and not _same_value(memory[new], value):
            raise ValueError(
                f"Conflicting embedding.soft_edge_memory.{new} and {old}."
            )
        memory[new] = value
        changes.append(_AliasHit(old, f"soft_edge_memory.{new}", rv))
    if memory or "soft_edge_memory" in out:
        out["soft_edge_memory"] = memory
    if scope == "none" and bool(memory.get("enabled", True)):
        raise ValueError(
            "embedding.soft_edge_memory.enabled=true requires "
            "prior_init_scope != 'none'."
        )


_EMBEDDING_REGISTRY: Tuple[_AliasRule, ...] = (
    _Rename("fallback_to_hamiltonian", ("h0_fallback_to_hamiltonian",)),
    _Transform(_embedding_h0_scope),
    _Transform(_embedding_prior),
)


def canonicalize_embedding_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Canonicalize H0/P2 embedding switches without changing model semantics."""

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("model_options.embedding must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    _apply_alias_registry(out, _EMBEDDING_REGISTRY, changes=changes)
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return out


_RECONSTRUCTION_ALIASES = {
    "none": "direct",
    "full": "direct",
    "h0": "h0_residual",
    "prior": "prior_residual",
    "physical_prior": "prior_residual",
}


def resolve_reconstruction_mode(
    mode: Optional[str],
    *,
    add_h0: Optional[bool] = None,
    add_prior: Optional[bool] = None,
) -> str:
    legacy_h0 = bool(add_h0) if add_h0 is not None else False
    legacy_prior = bool(add_prior) if add_prior is not None else False
    if legacy_h0 and legacy_prior:
        raise ValueError("prediction.add_h0 and prediction.add_prior are mutually exclusive.")
    legacy_mode = "h0_residual" if legacy_h0 else "prior_residual" if legacy_prior else "direct"
    resolved = legacy_mode if mode is None else _normalized_name(mode)
    resolved = _RECONSTRUCTION_ALIASES.get(resolved, resolved)
    if resolved not in {"direct", "h0_residual", "prior_residual"}:
        raise ValueError(
            "prediction.reconstruction must be 'direct', 'h0_residual', or "
            f"'prior_residual'; got {mode!r}."
        )
    if mode is not None and (add_h0 is not None or add_prior is not None) and resolved != legacy_mode:
        raise ValueError(
            f"prediction.reconstruction={resolved!r} conflicts with legacy "
            f"add_h0/add_prior (equivalent mode {legacy_mode!r})."
        )
    return resolved


def _prediction_reconstruction(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    add_h0 = out.pop("add_h0", None)
    add_prior = out.pop("add_prior", None)
    if add_h0 is not None or add_prior is not None:
        out["reconstruction"] = resolve_reconstruction_mode(
            out.get("reconstruction"), add_h0=add_h0, add_prior=add_prior
        )
        if add_h0 is not None:
            changes.append(_AliasHit("add_h0", "reconstruction", rv))
        if add_prior is not None:
            changes.append(_AliasHit("add_prior", "reconstruction", rv))
    elif "reconstruction" in out:
        out["reconstruction"] = resolve_reconstruction_mode(out["reconstruction"])


_PREDICTION_REGISTRY: Tuple[_AliasRule, ...] = (
    _Transform(_prediction_reconstruction),
)


def canonicalize_prediction_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("model_options.prediction must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    _apply_alias_registry(out, _PREDICTION_REGISTRY, changes=changes)
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return out


def migrate_legacy_checkpoint_model_options(
    options: Optional[Mapping[str, Any]],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Migrate model options saved after legacy schema default injection.

    The old H0 constructors let ``h0_fallback_to_hamiltonian`` override
    ``fallback_to_hamiltonian`` whenever the alias was present.  Old dargs
    normalization saved both keys, so a legitimate checkpoint can contain
    distinct values.  Preserve the value the old runtime actually used, but
    keep ordinary raw/new configuration conflict checks strict.
    """

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("checkpoint model_options must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(options))
    changes: List[_AliasHit] = []
    embedding = out.get("embedding")
    if isinstance(embedding, Mapping):
        embedding = deepcopy(dict(embedding))
        alias = "h0_fallback_to_hamiltonian"
        canonical = "fallback_to_hamiltonian"
        if alias in embedding:
            # Old runtime precedence: the alias value always wins, even when it
            # disagrees with the canonical key persisted alongside it.
            alias_value = embedding.pop(alias)
            embedding[canonical] = alias_value
            changes.append(_AliasHit(alias, canonical, _ALIAS_REMOVAL_VERSION))
        out["embedding"] = canonicalize_embedding_options(
            embedding, warn_deprecated=warn_deprecated
        )
    prediction = out.get("prediction")
    if isinstance(prediction, Mapping):
        out["prediction"] = canonicalize_prediction_options(
            prediction, warn_deprecated=warn_deprecated
        )
    _emit_alias_warnings(changes, enabled=warn_deprecated)
    return out


# Typed nested train_options groups (PR-F). These names must stay in sync with
# dptb.utils.argcheck.TRAIN_OPTION_GROUP_MEMBERS; configuration.py deliberately
# does not import argcheck (argcheck imports this module), so the coupling is
# asserted by test instead. Unlike the alias registry above, using a group is
# NOT deprecated: it is the new preferred input form, so no FutureWarning fires.
_TRAIN_OPTION_GROUP_NAMES = (
    "runtime",
    "distributed",
    "checkpoint",
    "observers",
    "physical_prior",
)


def _flatten_train_option_groups(train: MutableMapping[str, Any]) -> None:
    """Hoist typed nested train_options groups onto their flat keys in place.

    The trainer reads flat train_options keys, so a nested group such as
    ``distributed: {use_ddp: true}`` is flattened to ``use_ddp: true`` here,
    before dargs inserts defaults.  Flattening is idempotent (a second pass sees
    no groups) and fail-closed: a nested key that disagrees with an explicit
    flat key of the same name is rejected rather than silently overriding.
    """

    for group in _TRAIN_OPTION_GROUP_NAMES:
        if group not in train:
            continue
        block = train.pop(group)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise TypeError(f"train_options.{group} must be a mapping.")
        for key, value in dict(block).items():
            if key in train and not _same_value(train[key], value):
                raise ValueError(
                    f"Conflicting configuration values: nested "
                    f"train_options.{group}.{key} ({value!r}) and flat "
                    f"train_options.{key} ({train[key]!r})."
                )
            train[key] = value


def canonicalize_training_config(
    data: Mapping[str, Any],
    *,
    warn_deprecated: bool = True,
) -> Dict[str, Any]:
    """Canonicalize the route-bearing sections of a full training config."""

    if not isinstance(data, Mapping):
        raise TypeError("DeePTB configuration must be a mapping.")
    out: Dict[str, Any] = deepcopy(dict(data))
    train = out.get("train_options")
    if isinstance(train, Mapping):
        train = dict(train)
        # Flatten typed nested groups onto flat keys first so that, e.g., a
        # physical_prior.flow_options block is canonicalized by the flow path
        # below exactly like a top-level flow_options block.
        _flatten_train_option_groups(train)
        train_changes: List[_AliasHit] = []
        _apply_alias_registry(train, _TRAIN_ENDPOINT_REGISTRY, changes=train_changes)
        if "flow_options" in train:
            train["flow_options"] = canonicalize_flow_options(
                train.get("flow_options"), warn_deprecated=warn_deprecated
            )
        _emit_alias_warnings(train_changes, enabled=warn_deprecated)
        out["train_options"] = train
    model = out.get("model_options")
    if isinstance(model, Mapping):
        model = dict(model)
        if "embedding" in model:
            model["embedding"] = canonicalize_embedding_options(
                model.get("embedding"), warn_deprecated=warn_deprecated
            )
        if "prediction" in model:
            model["prediction"] = canonicalize_prediction_options(
                model.get("prediction"), warn_deprecated=warn_deprecated
            )
        out["model_options"] = model
    return out


__all__ = [
    "canonicalize_embedding_options",
    "canonicalize_flow_options",
    "canonicalize_prediction_options",
    "canonicalize_training_config",
    "migrate_legacy_checkpoint_flow_options",
    "migrate_legacy_checkpoint_model_options",
    "migrate_legacy_checkpoint_train_options",
    "resolve_init_scope",
    "resolve_reconstruction_mode",
]
