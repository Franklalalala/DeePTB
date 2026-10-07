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
from numbers import Integral
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


def _require_bool(value: Any, *, option_name: str) -> bool:
    """Reject truthy strings/numbers before legacy aliases bypass dargs."""

    if not isinstance(value, bool):
        raise TypeError(f"{option_name} must be a boolean.")
    return value


def _require_string(value: Any, *, option_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{option_name} must be a string.")
    return value


def _scope_from_legacy(
    *,
    enabled: Optional[bool],
    node: Optional[bool],
    edge: Optional[bool],
) -> str:
    master = (
        True
        if enabled is None
        else _require_bool(enabled, option_name="legacy init enabled flag")
    )
    use_node = (
        True
        if node is None
        else _require_bool(node, option_name="legacy node init flag")
    )
    use_edge = (
        True
        if edge is None
        else _require_bool(edge, option_name="legacy edge init flag")
    )
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
    resolved = (
        legacy_scope
        if scope is None
        else _normalized_name(_require_string(scope, option_name=option_name))
    )
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


# Inactive historical fields are discarded when loading supervised checkpoints.
_ARCHIVED_FLOW_OPTIONS = frozenset({
    'log_compatible_loss',
    'allow_complex_prior_real_projection', 'apply_to_reference', 'basis_onsite_edge_value',
    'basis_onsite_missing_value', 'basis_onsite_mode', 'basis_onsite_scale',
    'block_export_final_full_h', 'block_input_adapter', 'block_inverse_atol', 'block_inverse_mode',
    'block_ode', 'compatible_loss_to_legacy_keys', 'component_reduction', 'dftb_prior_overlap',
    'dftb_prior_require_geometry', 'dftb_prior_strict', 'dftb_skdata', 'edge_block_shape_key',
    'edge_block_target_key', 'edge_output_key', 'edge_weight', 'endpoint_weight_cap',
    'endpoint_weight_power', 'external_prior_strict', 'flow_time_h_key', 'flow_time_r_key',
    'flow_time_t_key', 'h0_condition_space', 'haar_candidate_index', 'haar_dm_strict',
    'haar_edge_key', 'haar_node_key', 'huckel_edge_channel_scale', 'huckel_edge_energy_fallback',
    'huckel_edge_length_decay', 'huckel_edge_overlap_key', 'huckel_energy_mode', 'huckel_k',
    'huckel_node_overlap_key', 'huckel_scale_global', 'huckel_scale_mode', 'huckel_strict_basis',
    'huckel_strict_overlap', 'log_train_compatible_loss', 'log_validation_compatible_loss',
    'log_validation_flow_euler_loss', 'log_validation_random_t_loss', 'log_validation_t0_loss',
    'loss_type', 'meanflow', 'meanflow_aggressive', 'meanflow_profile', 'node_block_shape_key',
    'node_block_target_key', 'node_output_key', 'node_weight', 'objective', 'omit_time_scaling',
    'overlap_huckel_edge_channel_scale', 'overlap_huckel_k', 'overwrite_feature_keys',
    'physical_prior_fallback', 'physical_prior_jitter_edge_decay',
    'physical_prior_jitter_reference_scale', 'physical_prior_jitter_sigma', 'pixel_meanflow',
    'prediction_add_h0', 'prior_calibration', 'prior_edge', 'prior_edge_key', 'prior_jitter_sigma',
    'prior_key_prefixes', 'prior_node', 'prior_node_key', 'prior_skdata', 'sample_prior_scale',
    'skdata', 'state_space', 'strict_certification', 't_eps', 'target_semantics',
    'te_prior_validation_seed', 'tied_irrep_irreps', 'tied_irrep_mode', 'tied_irrep_sigma',
    'tied_irrep_validation_seed', 'time_conditioning_required', 'time_logit_mean', 'time_logit_std',
    'time_sampling', 'type', 'validation_flow_metrics', 'validation_ode_steps', 'z_loss_coef',
})


def _flow_missing_h0_policy(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    strict = out.pop("strict_h0", _MISSING)
    warn_missing = out.pop("warn_missing_h0", _MISSING)
    if strict is not _MISSING or warn_missing is not _MISSING:
        strict_value = (
            True
            if strict is _MISSING
            else _require_bool(strict, option_name="flow_options.strict_h0")
        )
        warn_value = (
            True
            if warn_missing is _MISSING
            else _require_bool(
                warn_missing, option_name="flow_options.warn_missing_h0"
            )
        )
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
        policy = _normalized_name(
            _require_string(
                out["missing_h0_policy"],
                option_name="flow_options.missing_h0_policy",
            )
        )
        policy = {"warn": "warn_zero", "silent_zero": "zero"}.get(policy, policy)
        if policy not in {"error", "warn_zero", "zero"}:
            raise ValueError(
                "flow_options.missing_h0_policy must be 'error', 'warn_zero', or 'zero'."
            )
        out["missing_h0_policy"] = policy


def canonicalize_flow_options(options, *, warn_deprecated=True):
    """Resolve prior-noise options saved under the historical flow name."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("train_options.flow_options must be a mapping.")
    out = deepcopy(dict(options))
    if _require_bool(out.get("enabled", False), option_name="flow_options.enabled"):
        raise ValueError("Flow training is available only in archived models")
    if out.get("block_ode", False):
        raise ValueError("Block ODE objectives are available only in archived models")
    if "prediction_add_h0" in out and _require_bool(
        out["prediction_add_h0"], option_name="flow_options.prediction_add_h0"
    ):
        raise ValueError("flow_options.prediction_add_h0=true is not a valid canonical option")
    changes = []
    _flow_missing_h0_policy(out, changes, _ALIAS_REMOVAL_VERSION)
    for key in _ARCHIVED_FLOW_OPTIONS:
        out.pop(key, None)
    for key in ("mode", "prior", "te_prior_mode", "output_space"):
        if key in out and isinstance(out[key], str):
            out[key] = _normalized_name(out[key])
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
        mode = _normalized_name(
            _require_string(value, option_name="train_options.endpoint_loss_mode")
        )
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
        if legacy_enabled is not _MISSING:
            legacy_enabled = _require_bool(
                legacy_enabled,
                option_name="train_options.log_single_model_compatible_loss",
            )
        normalized_legacy_mode = (
            "reduce" if legacy_mode is _MISSING else _validate(legacy_mode)
        )
        legacy_effective = (
            normalized_legacy_mode
            if legacy_enabled is _MISSING or legacy_enabled
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


def migrate_legacy_checkpoint_flow_options(options, *, warn_deprecated=True):
    """Load inactive flow dictionaries and retain prior-noise sampling keys."""
    return canonicalize_flow_options(options, warn_deprecated=warn_deprecated)


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
_H0_METHODS = {"lem_moe_v3_h0", "lem_moe_v3_edge_h0"}

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


def _embedding_prior_2b(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    method = _normalized_name(out.get("method", ""))
    if method not in {"lem_moe_v3_prior_2b", "lem_moe_v3_edge_prior_2b"}:
        return
    prior_kind = _normalized_name(out.get("prior_kind", "na_cf"))
    if prior_kind not in {"p2", "p23", "na_cf", "h0"}:
        raise ValueError(
            "lem_moe_v3_prior_2b supports prior_kind='p2', 'p23', 'na_cf', or 'h0'; "
            f"got {prior_kind!r}."
        )
    out["prior_kind"] = prior_kind
    merge = _normalized_name(out.get("prior_merge_mode", "concat"))
    if merge != "concat":
        raise ValueError(
            "lem_moe_v3_prior_2b requires prior_merge_mode='concat'; "
            f"got {out.get('prior_merge_mode')!r}."
        )
    out["prior_merge_mode"] = "concat"
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
    if scope != "both":
        raise ValueError(
            "lem_moe_v3_prior_2b requires prior_init_scope='both'; "
            f"got {scope!r}."
        )
    out["prior_init_scope"] = scope
    for name, value in zip(
        ("use_prior_init", "use_prior_node_init", "use_prior_edge_init"), legacy
    ):
        if value is not None:
            changes.append(_AliasHit(name, "prior_init_scope", rv))


def _embedding_prior(
    out: MutableMapping[str, Any], changes: List[_AliasHit], rv: str
) -> None:
    method = _normalized_name(out.get("method", ""))
    if method != "lem_moe_v3_prior":
        return
    prior_kind = _normalized_name(out.get("prior_kind", "p2"))
    if prior_kind not in {"p2", "p23", "na_cf", "h0"}:
        raise ValueError(
            "lem_moe_v3_prior supports prior_kind='p2', 'p23', 'na_cf', or 'h0'; "
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
    memory_enabled = memory.get("enabled", True)
    if "enabled" in memory:
        memory_enabled = _require_bool(
            memory_enabled, option_name="embedding.soft_edge_memory.enabled"
        )
    if scope == "none" and memory_enabled:
        raise ValueError(
            "embedding.soft_edge_memory.enabled=true requires "
            "prior_init_scope != 'none'."
        )


_EMBEDDING_REGISTRY: Tuple[_AliasRule, ...] = (
    _Rename("fallback_to_hamiltonian", ("h0_fallback_to_hamiltonian",)),
    _Transform(_embedding_h0_scope),
    _Transform(_embedding_prior),
    _Transform(_embedding_prior_2b),
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
    legacy_h0 = (
        _require_bool(add_h0, option_name="prediction.add_h0")
        if add_h0 is not None
        else False
    )
    legacy_prior = (
        _require_bool(add_prior, option_name="prediction.add_prior")
        if add_prior is not None
        else False
    )
    if legacy_h0 and legacy_prior:
        raise ValueError("prediction.add_h0 and prediction.add_prior are mutually exclusive.")
    legacy_mode = "h0_residual" if legacy_h0 else "prior_residual" if legacy_prior else "direct"
    resolved = (
        legacy_mode
        if mode is None
        else _normalized_name(
            _require_string(mode, option_name="prediction.reconstruction")
        )
    )
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


# Deprecated train_options keys dropped during canonicalization. The DDP loader
# (dptb.nnops.ddp_utils) imports this same tuple, so there is a single source of
# truth for which keys are stripped and one place that records their removal.
DEPRECATED_TRAIN_OPTION_KEYS = (
    "shared_scheduler_metric",
    "independent_expert_scheduler",
    "distributed_global_reduce_every",
)


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
        # Drop deprecated train_options keys (the DDP loader's strip shares this
        # same tuple) with a FutureWarning, so a config still carrying them
        # canonicalizes cleanly instead of tripping strict argcheck later.
        for _dep_key in DEPRECATED_TRAIN_OPTION_KEYS:
            if _dep_key in train:
                train.pop(_dep_key, None)
                if warn_deprecated:
                    warnings.warn(
                        f"train_options.{_dep_key} is deprecated and ignored "
                        f"(scheduled for removal in 2.3).",
                        FutureWarning,
                        stacklevel=2,
                    )
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
    "DEPRECATED_TRAIN_OPTION_KEYS",
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
