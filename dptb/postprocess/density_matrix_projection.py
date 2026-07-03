"""Density-matrix physical projections.

Closed-shell projection implements the EMolES-I style constraint in a
non-orthogonal AO basis:
    Tr(D S) = Ne,   D S D = 2 D.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class DMProjectionResult:
    density: Tensor
    occupations: Tensor
    electron_count: Tensor
    idempotency_error: Tensor


def hermitian_part(x: Tensor) -> Tensor:
    return 0.5 * (x + x.mH)


def _s_sqrt_and_inv_sqrt(s: Tensor, eig_floor: float) -> tuple[Tensor, Tensor]:
    se, su = torch.linalg.eigh(hermitian_part(s))
    se = se.clamp_min(eig_floor)
    s_sqrt = (su * se.sqrt().unsqueeze(-2)) @ su.mH
    s_inv_sqrt = (su * se.rsqrt().unsqueeze(-2)) @ su.mH
    return s_sqrt, s_inv_sqrt


def density_diagnostics(d: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    electron_count = torch.trace((d @ s).real)
    idem = d @ s @ d - 2.0 * d
    denom = d.norm().clamp_min(1e-30)
    return electron_count, idem.norm() / denom


def project_closed_shell_density(
    d_pred: Tensor,
    s: Tensor,
    n_electrons: int,
    *,
    eig_floor: float = 1e-10,
) -> DMProjectionResult:
    """Project a spin-summed AO density matrix onto the closed-shell manifold.

    Parameters
    ----------
    d_pred:
        Predicted AO-basis spin-summed 1-RDM.
    s:
        AO overlap matrix.
    n_electrons:
        Total electron count. Must be even for this closed-shell projection.
    eig_floor:
        Floor for S eigenvalues before symmetric orthogonalization.
    """
    if n_electrons % 2 != 0:
        raise ValueError("closed-shell projection requires an even electron count")
    if d_pred.shape != s.shape or d_pred.ndim != 2:
        raise ValueError("d_pred and s must be single dense square matrices with identical shape")
    n_occ = n_electrons // 2
    n = d_pred.shape[-1]
    if n_occ < 0 or n_occ > n:
        raise ValueError(f"invalid n_electrons={n_electrons} for matrix size {n}")
    s_sqrt, s_inv_sqrt = _s_sqrt_and_inv_sqrt(s, eig_floor=eig_floor)
    d_orth = hermitian_part(s_sqrt @ hermitian_part(d_pred) @ s_sqrt)
    occ, u = torch.linalg.eigh(d_orth)
    order = torch.argsort(occ, descending=True)
    u = u[:, order]
    occ_proj = torch.zeros_like(occ)
    occ_proj[:n_occ] = 2.0
    d_orth_proj = (u * occ_proj.unsqueeze(-2)) @ u.mH
    d_proj = hermitian_part(s_inv_sqrt @ d_orth_proj @ s_inv_sqrt)
    ne, idem = density_diagnostics(d_proj, s)
    return DMProjectionResult(density=d_proj, occupations=occ_proj, electron_count=ne, idempotency_error=idem)


def project_occupations_capped_simplex(occ: Tensor, n_electrons: float, max_occ: float = 2.0, iters: int = 80) -> Tensor:
    """Project occupation numbers to ``0 <= n_i <= max_occ`` and ``sum n_i = Ne``.

    This is the soft fallback for metallic / finite-temperature cases where hard
    idempotent occupations are not appropriate.
    """
    if n_electrons < 0 or n_electrons > max_occ * occ.numel():
        raise ValueError("requested electron count is outside capped simplex")
    lo = (occ - max_occ).min() - abs(max_occ) - 1.0
    hi = occ.max() + abs(max_occ) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        projected = (occ - mid).clamp(0.0, max_occ)
        if projected.sum() > n_electrons:
            lo = mid
        else:
            hi = mid
    return (occ - hi).clamp(0.0, max_occ)
