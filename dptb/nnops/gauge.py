"""Gauge-alignment utilities for non-orthogonal Hamiltonians.

The physical gauge freedom is H -> H + mu S.  This module provides a
closed-form L2-optimal mu and reports MAE after applying that global shift.
It is intentionally dependency-light so it can be used by both training and
offline evaluation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class GaugeResult:
    """Result returned by :func:`gauge_mae`."""

    mae: Tensor
    mu: Tensor
    residual: Tensor


def _broadcast_mask(mask: Optional[Tensor], ref: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    mask = mask.to(device=ref.device)
    while mask.ndim < ref.ndim:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=ref.real.dtype)


def _real_inner(a: Tensor, b: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Return Re(sum(conj(a) * b)) with optional mask.

    The result keeps all leading dimensions that are not part of the final two
    matrix axes.  For vectors or flat tensors, it reduces the last axis.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    m = _broadcast_mask(mask, a)
    prod = torch.conj(a) * b
    if m is not None:
        prod = prod * m
    dims = tuple(range(-(a.ndim if a.ndim < 2 else 2), 0)) if a.ndim <= 2 else (-2, -1)
    return prod.real.sum(dim=dims)


def _sq_norm(a: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if mask is not None:
        m = _broadcast_mask(mask, a)
        a = a * m
    dims = tuple(range(-(a.ndim if a.ndim < 2 else 2), 0)) if a.ndim <= 2 else (-2, -1)
    return (torch.conj(a) * a).real.sum(dim=dims)


def solve_mu(diff: Tensor, overlap: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-30) -> Tensor:
    """Solve the global gauge shift ``mu`` for ``diff = H_pred - H_ref``.

    We minimize ``||diff - mu * S||_2^2`` over a real scalar ``mu`` for each
    leading batch item.  This corresponds to comparing ``H_pred`` against
    ``H_ref + mu S``.

    Parameters
    ----------
    diff:
        Difference tensor ``H_pred - H_ref``.  Supports real or complex dtype.
    overlap:
        Overlap tensor ``S`` with the same shape as ``diff``.
    mask:
        Optional boolean/float mask for valid entries.
    eps:
        Denominator floor.
    """
    numerator = _real_inner(diff, overlap, mask=mask)
    denominator = _sq_norm(overlap, mask=mask).clamp_min(eps)
    return numerator / denominator


def apply_mu_to_target(target: Tensor, overlap: Tensor, mu: Tensor) -> Tensor:
    """Return ``target + mu * overlap`` with batch-safe broadcasting."""
    while mu.ndim < target.ndim:
        mu = mu.unsqueeze(-1)
    return target + mu.to(dtype=target.real.dtype if not target.is_complex() else target.dtype) * overlap


def gauge_residual(pred: Tensor, target: Tensor, overlap: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Return ``(residual, mu)`` after global gauge alignment.

    The residual is ``pred - (target + mu S)``.
    """
    if pred.shape != target.shape or pred.shape != overlap.shape:
        raise ValueError(
            f"pred/target/overlap shapes must match, got {tuple(pred.shape)}, "
            f"{tuple(target.shape)}, {tuple(overlap.shape)}"
        )
    diff = pred - target
    mu = solve_mu(diff, overlap, mask=mask)
    aligned_target = apply_mu_to_target(target, overlap, mu)
    residual = pred - aligned_target
    if mask is not None:
        residual = residual * _broadcast_mask(mask, residual)
    return residual, mu


def masked_mae(x: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-30) -> Tensor:
    """Mean absolute value with optional mask over the final two axes."""
    if mask is None:
        dims = (-2, -1) if x.ndim >= 2 else (-1,)
        return x.abs().mean(dim=dims)
    m = _broadcast_mask(mask, x)
    dims = (-2, -1) if x.ndim >= 2 else (-1,)
    denom = m.sum(dim=dims).clamp_min(eps)
    return (x.abs() * m).sum(dim=dims) / denom


def gauge_mae(pred: Tensor, target: Tensor, overlap: Tensor, mask: Optional[Tensor] = None) -> GaugeResult:
    """Compute MAE after L2-optimal global gauge alignment.

    Returns a :class:`GaugeResult` containing the MAE, the real scalar ``mu``
    per batch item, and the aligned residual.
    """
    residual, mu = gauge_residual(pred, target, overlap, mask=mask)
    return GaugeResult(mae=masked_mae(residual, mask=mask), mu=mu, residual=residual)
