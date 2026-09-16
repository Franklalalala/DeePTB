from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from .models import BlockKey


def energy_squared_weighted_r2(
    predicted: Mapping[BlockKey, np.ndarray],
    target: Mapping[BlockKey, np.ndarray],
    *,
    onsite: bool | None = None,
    complex_mode: Literal["magnitude", "stacked_components"] = "magnitude",
    strict_keys: bool = True,
) -> float:
    """Energy-squared-weighted R² over common matrix elements.

    With ``complex_mode='magnitude'`` (recommended), each complex target entry
    has weight ``|H_ref|²`` and residual ``|H_pred-H_ref|²``; the weighted mean
    is complex. ``stacked_components`` reproduces the alternative convention of
    treating real and imaginary components as separate scalar observations.
    The exact acceptance metric should be fixed against the data-generation
    pipeline before comparing published numbers.  ``strict_keys=True`` is the
    default: silently scoring only the intersection can make a pruned or
    misaligned prediction look artificially good.
    """
    selected_target = {
        key: ref
        for key, ref in target.items()
        if onsite is None or (key.i == key.j and key.R == (0, 0, 0)) == onsite
    }
    if strict_keys:
        missing = sorted(
            (key.as_tuple() for key in selected_target if key not in predicted),
        )
        if missing:
            raise KeyError(
                f"Prediction is missing {len(missing)} target blocks; first={missing[0]}"
            )
        extra = sorted(
            key.as_tuple()
            for key in predicted
            if (onsite is None or (key.i == key.j and key.R == (0, 0, 0)) == onsite)
            and key not in selected_target
        )
        if extra:
            raise KeyError(
                f"Prediction has {len(extra)} extra blocks; first={extra[0]}"
            )

    xs, ys = [], []
    for key, ref in selected_target.items():
        if key not in predicted:
            continue
        pred = predicted[key]
        if pred.shape != ref.shape:
            raise ValueError(f"Shape mismatch for {key}: {pred.shape} vs {ref.shape}")
        xs.append(np.asarray(ref, dtype=complex).ravel())
        ys.append(np.asarray(pred, dtype=complex).ravel())
    if not xs:
        raise ValueError("No common blocks")
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if complex_mode == "magnitude":
        w = np.abs(x) ** 2
        if np.sum(w) <= 0:
            raise ValueError("Target has zero energy-squared weight")
        mean = np.sum(w * x) / np.sum(w)
        denom = np.sum(w * np.abs(x - mean) ** 2)
        numer = np.sum(w * np.abs(y - x) ** 2)
    elif complex_mode == "stacked_components":
        xr = np.concatenate([x.real, x.imag])
        yr = np.concatenate([y.real, y.imag])
        w = xr**2
        if np.sum(w) <= 0:
            raise ValueError("Target has zero energy-squared weight")
        mean = np.sum(w * xr) / np.sum(w)
        denom = np.sum(w * (xr - mean) ** 2)
        numer = np.sum(w * (yr - xr) ** 2)
    else:
        raise ValueError(f"Unknown complex_mode={complex_mode!r}")
    if denom <= 0:
        raise ValueError("Degenerate weighted target variance")
    return float(1.0 - numer / denom)
