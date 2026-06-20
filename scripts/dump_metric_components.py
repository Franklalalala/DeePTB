#!/usr/bin/env python3
"""Dump additive metric components instead of only a scalar loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_components(value: Any, default_name: str) -> Dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {default_name: value}
    if isinstance(value, Mapping):
        result = {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        if not result:
            raise ValueError("Mapping contains no tensor components")
        return result
    raise TypeError(f"Expected tensor or tensor mapping, got {type(value)!r}")


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    try:
        return torch.broadcast_to(mask, like.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is not broadcastable to {tuple(like.shape)}"
        ) from exc


def component_stats(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    symmetrize_last_two: bool = False,
) -> Dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    prediction = prediction.detach()
    target = target.detach().to(dtype=prediction.dtype)
    if symmetrize_last_two:
        if prediction.ndim < 2 or prediction.shape[-1] != prediction.shape[-2]:
            raise ValueError("Symmetrization requires square final two dimensions")
        prediction = 0.5 * (prediction + prediction.transpose(-1, -2).conj())
        target = 0.5 * (target + target.transpose(-1, -2).conj())

    if mask is None:
        active = torch.ones(prediction.shape, dtype=torch.bool)
    else:
        active = _broadcast_mask(mask.detach(), prediction)

    diff = prediction - target
    abs_diff = diff.abs()
    sq_diff = abs_diff.square()
    active_abs = abs_diff.masked_select(active)
    active_sq = sq_diff.masked_select(active)
    count = int(active.sum().item())
    total = active.numel()

    abs_sum = active_abs.sum(dtype=torch.float64).item()
    sq_sum = active_sq.sum(dtype=torch.float64).item()
    return {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "is_complex": bool(prediction.is_complex()),
        "symmetrize_last_two": bool(symmetrize_last_two),
        "abs_sum": abs_sum,
        "sq_sum": sq_sum,
        "count": count,
        "active_mask_count": count,
        "active_mask_total": total,
        "active_mask_fraction": (count / total) if total else 0.0,
        "mae_from_components": (abs_sum / count) if count else None,
        "mse_from_components": (sq_sum / count) if count else None,
        "rmse_from_components": (sq_sum / count) ** 0.5 if count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--name", default="component")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--symmetrize-last-two", action="store_true")
    args = parser.parse_args()

    predictions = _as_components(_load(args.pred), args.name)
    targets = _as_components(_load(args.target), args.name)
    masks = _as_components(_load(args.mask), args.name) if args.mask else {}

    shared_mask = None
    if len(masks) == 1 and args.name in masks:
        shared_mask = masks[args.name]

    output = {}
    for name, prediction in predictions.items():
        if name not in targets:
            raise KeyError(f"Target does not contain component {name!r}")
        mask = masks.get(name, shared_mask)
        output[name] = component_stats(
            prediction,
            targets[name],
            mask,
            symmetrize_last_two=args.symmetrize_last_two,
        )

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
