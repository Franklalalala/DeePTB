"""Checkpoint/common-option merge contracts shared by CLI entrypoints."""

from __future__ import annotations

import copy
from typing import Optional


ARCHITECTURE_COMMON_KEYS = frozenset(
    {
        "overlap",
        "has_soc",
        "nextham_uureal_mask",
        "full_soc_prediction",
    }
)
RUNTIME_DEFAULT_COMMON_KEYS = frozenset({"device", "seed"})


def merge_checkpoint_common_options(
    normalized_common_options: dict,
    checkpoint_common_options: dict,
    explicit_common_options: Optional[dict] = None,
    *,
    preserve_runtime_defaults: bool = False,
) -> dict:
    """Merge checkpoint architecture with explicit runtime/config overrides.

    ``normalized_common_options`` may contain schema defaults that were never
    written by the user. Existing checkpoint values win over those defaults.
    Only keys present in ``explicit_common_options`` are restored afterwards;
    device/seed defaults may also be preserved for CLI evaluation/training.
    """

    normalized = copy.deepcopy(normalized_common_options or {})
    checkpoint = copy.deepcopy(checkpoint_common_options or {})
    explicit = copy.deepcopy(
        normalized if explicit_common_options is None else explicit_common_options
    )

    for key in ARCHITECTURE_COMMON_KEYS:
        if (
            key in explicit
            and key in checkpoint
            and normalized.get(key) != checkpoint.get(key)
        ):
            raise ValueError(
                f"common_options.{key}={normalized.get(key)!r} conflicts with "
                f"checkpoint value {checkpoint.get(key)!r}. Architecture-sensitive "
                "checkpoint options cannot be changed while loading that checkpoint."
            )

    merged = copy.deepcopy(normalized)
    merged.update(checkpoint)
    override_keys = set(explicit)
    if preserve_runtime_defaults:
        override_keys.update(RUNTIME_DEFAULT_COMMON_KEYS)
    for key in override_keys:
        if key == "basis":
            continue
        if key in normalized:
            merged[key] = copy.deepcopy(normalized[key])
    if "basis" in checkpoint:
        merged["basis"] = copy.deepcopy(checkpoint["basis"])
    return merged
