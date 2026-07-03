"""Dense core for NextHAM-style k-space P/Q/PQ loss.

This module is deliberately independent of DeePTB's AtomicDataDict and HR2HK
plumbing.  It operates on dense H(k), H_ref(k), S(k) matrices and can be used
in unit tests, diagnostic scripts, or as the numerical kernel behind a registered
``Loss`` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

Tensor = torch.Tensor
EV_PER_HA = 27.211386245988


@dataclass(frozen=True)
class PQLossResult:
    loss: Tensor
    loss_p: Tensor
    loss_q: Tensor
    loss_pq: Tensor
    mu: Tensor
    n_p: int
    n_q: int
    fermi: Tensor
    eigvals_ref: Tensor


def hermitian_part(x: Tensor) -> Tensor:
    return 0.5 * (x + x.mH)


def regularize_overlap(s: Tensor, eig_floor: float = 1e-10) -> Tensor:
    s_h = hermitian_part(s)
    evals, evecs = torch.linalg.eigh(s_h)
    evals = evals.clamp_min(eig_floor)
    return (evecs * evals.unsqueeze(-2)) @ evecs.mH


def generalized_eigh(h: Tensor, s: Tensor, eig_floor: float = 1e-10) -> Tuple[Tensor, Tensor]:
    """Solve ``H C = S C eps`` for a Hermitian pair.

    Returns eigenvalues and S-orthonormal eigenvectors ``C`` satisfying
    ``Cᴴ S C = I``.  The operation is differentiable, but in the k-space loss
    this should normally be called under ``torch.no_grad()`` on the reference
    Hamiltonian only.
    """
    if h.ndim != 2 or s.ndim != 2:
        raise ValueError("generalized_eigh expects single dense matrices")
    if h.shape != s.shape or h.shape[0] != h.shape[1]:
        raise ValueError(f"expected square matrices with same shape, got {h.shape} and {s.shape}")
    h_h = hermitian_part(h)
    s_h = regularize_overlap(s, eig_floor=eig_floor)
    se, su = torch.linalg.eigh(s_h)
    inv_sqrt = (su * se.clamp_min(eig_floor).rsqrt().unsqueeze(-2)) @ su.mH
    h_orth = hermitian_part(inv_sqrt.mH @ h_h @ inv_sqrt)
    eps, q = torch.linalg.eigh(h_orth)
    c = inv_sqrt @ q
    return eps, c


def fermi_from_n_occ(eigvals: Tensor, n_occ: int) -> Tensor:
    if n_occ <= 0 or n_occ >= eigvals.numel():
        raise ValueError(f"n_occ must be in [1, n_bands-1], got {n_occ} for {eigvals.numel()} bands")
    return 0.5 * (eigvals[n_occ - 1] + eigvals[n_occ])


def choose_p_subspace(eigvals: Tensor, n_occ: int, e_cut_above_fermi: Optional[float] = None) -> Tuple[int, Tensor]:
    """Choose P as states up to ``E_F + e_cut`` and at least occupied states."""
    fermi = fermi_from_n_occ(eigvals, n_occ)
    if e_cut_above_fermi is None:
        n_p = n_occ
    else:
        n_p = int((eigvals <= fermi + e_cut_above_fermi).sum().item())
        n_p = max(n_p, n_occ)
        n_p = min(n_p, eigvals.numel() - 1)
    return n_p, fermi


def project_to_ref_basis(h: Tensor, ref_vecs: Tensor) -> Tensor:
    """Project dense H into reference generalized eigenbasis U: Uᴴ H U."""
    return ref_vecs.mH @ h @ ref_vecs


def _mse(x: Tensor) -> Tensor:
    return (x.abs() ** 2).mean()


def _solve_kspace_mu(diff_tilde: Tensor, n_p: int, lambda_p: float, lambda_q: float, eps: float = 1e-30) -> Tensor:
    """Solve k-only gauge mu in the reference eigenbasis.

    Since Uᴴ S U = I, a gauge shift changes only the diagonal of PP and QQ.
    The PQ block is independent of mu.
    """
    n = diff_tilde.shape[-1]
    diag = diff_tilde.diagonal().real
    num = lambda_p * diag[:n_p].sum() + lambda_q * diag[n_p:].sum()
    den = lambda_p * n_p + lambda_q * (n - n_p)
    return num / max(float(den), eps)


def dense_kspace_pq_loss_single(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Tensor,
    *,
    n_occ: int,
    e_cut_above_fermi: Optional[float] = None,
    lambda_p: float = 1.0,
    lambda_q: float = 0.1,
    lambda_pq: float = 1.0,
    gauge_mu: bool = True,
    overlap_eig_floor: float = 1e-10,
) -> PQLossResult:
    """Compute NextHAM-style dense P/Q/PQ loss for one k point."""
    with torch.no_grad():
        eps_ref, u_ref = generalized_eigh(h_ref.detach(), s.detach(), eig_floor=overlap_eig_floor)
        n_p, fermi = choose_p_subspace(eps_ref, n_occ=n_occ, e_cut_above_fermi=e_cut_above_fermi)
    h_pred_t = project_to_ref_basis(hermitian_part(h_pred), u_ref)
    h_ref_t = project_to_ref_basis(hermitian_part(h_ref), u_ref).detach()
    diff_t = h_pred_t - h_ref_t
    mu = _solve_kspace_mu(diff_t, n_p=n_p, lambda_p=lambda_p, lambda_q=lambda_q) if gauge_mu else diff_t.new_zeros(())
    eye = torch.eye(diff_t.shape[-1], dtype=diff_t.dtype, device=diff_t.device)
    diff_t = diff_t - mu.to(diff_t.real.dtype) * eye
    pp = diff_t[:n_p, :n_p]
    qq = diff_t[n_p:, n_p:]
    pq = diff_t[:n_p, n_p:]
    loss_p = _mse(pp)
    loss_q = _mse(qq) if qq.numel() else diff_t.real.new_zeros(())
    loss_pq = _mse(pq) if pq.numel() else diff_t.real.new_zeros(())
    loss = lambda_p * loss_p + lambda_q * loss_q + lambda_pq * loss_pq
    return PQLossResult(
        loss=loss,
        loss_p=loss_p,
        loss_q=loss_q,
        loss_pq=loss_pq,
        mu=mu,
        n_p=n_p,
        n_q=diff_t.shape[-1] - n_p,
        fermi=fermi,
        eigvals_ref=eps_ref,
    )


def dense_kspace_pq_loss(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Tensor,
    *,
    n_occ: int,
    e_cut_above_fermi: Optional[float] = None,
    lambda_p: float = 1.0,
    lambda_q: float = 0.1,
    lambda_pq: float = 1.0,
    gauge_mu: bool = True,
    overlap_eig_floor: float = 1e-10,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Batch wrapper for :func:`dense_kspace_pq_loss_single`.

    Inputs may be ``[N,N]`` or ``[...,N,N]``.  Returns scalar mean loss and a
    diagnostics dictionary containing mean components.
    """
    if h_pred.shape != h_ref.shape or h_pred.shape != s.shape:
        raise ValueError("h_pred, h_ref and s must have identical shapes")
    if h_pred.ndim == 2:
        r = dense_kspace_pq_loss_single(
            h_pred,
            h_ref,
            s,
            n_occ=n_occ,
            e_cut_above_fermi=e_cut_above_fermi,
            lambda_p=lambda_p,
            lambda_q=lambda_q,
            lambda_pq=lambda_pq,
            gauge_mu=gauge_mu,
            overlap_eig_floor=overlap_eig_floor,
        )
        return r.loss, {
            "loss_p": r.loss_p.detach(),
            "loss_q": r.loss_q.detach(),
            "loss_pq": r.loss_pq.detach(),
            "mu": r.mu.detach(),
            "n_p": torch.as_tensor(r.n_p, device=h_pred.device),
            "n_q": torch.as_tensor(r.n_q, device=h_pred.device),
            "fermi": r.fermi.detach(),
        }
    flat_pred = h_pred.reshape((-1,) + h_pred.shape[-2:])
    flat_ref = h_ref.reshape((-1,) + h_ref.shape[-2:])
    flat_s = s.reshape((-1,) + s.shape[-2:])
    results = [
        dense_kspace_pq_loss_single(
            hp,
            hr,
            ss,
            n_occ=n_occ,
            e_cut_above_fermi=e_cut_above_fermi,
            lambda_p=lambda_p,
            lambda_q=lambda_q,
            lambda_pq=lambda_pq,
            gauge_mu=gauge_mu,
            overlap_eig_floor=overlap_eig_floor,
        )
        for hp, hr, ss in zip(flat_pred, flat_ref, flat_s)
    ]
    losses = torch.stack([r.loss for r in results])
    return losses.mean(), {
        "loss_p": torch.stack([r.loss_p.detach() for r in results]).mean(),
        "loss_q": torch.stack([r.loss_q.detach() for r in results]).mean(),
        "loss_pq": torch.stack([r.loss_pq.detach() for r in results]).mean(),
        "mu": torch.stack([r.mu.detach() for r in results]).mean(),
        "n_p": torch.as_tensor(sum(r.n_p for r in results) / len(results), device=h_pred.device),
        "n_q": torch.as_tensor(sum(r.n_q for r in results) / len(results), device=h_pred.device),
        "fermi": torch.stack([r.fermi.detach() for r in results]).mean(),
    }
