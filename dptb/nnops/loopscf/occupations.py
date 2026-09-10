"""Overlap factorization and electron-conserving global occupations."""

from __future__ import annotations

import math
import torch


def factor_overlap_robust(
    S_k: torch.Tensor, ill_threshold: float = 1e-5, work_dtype=None
):
    """Batched overlap factorization with positive-subspace projection fallback.

    Occupy keeps the default complex128/float64 work dtype. The fw10 loss path
    must pass the model dtype (complex64 for float32) or every [n_k,N,N]
    tensor doubles and bs>1 OOMs for a non-physical reason.
    """
    n_k, dim, _ = S_k.shape
    if work_dtype is None:
        work_dtype = torch.complex128 if S_k.is_complex() else torch.float64
    S_k64 = S_k.to(work_dtype)
    S_k64 = 0.5 * (S_k64 + S_k64.mH)

    L, info = torch.linalg.cholesky_ex(S_k64)
    bad = (info != 0).nonzero().reshape(-1)

    proj_cache = {}
    if len(bad) > 0:
        L = L.clone()
        min_w_overall = 1e9
        n_dropped_max = 0
        for ik in bad.tolist():
            w, V = torch.linalg.eigh(S_k64[ik])
            healthy = w > ill_threshold
            if not bool(healthy.any()):
                raise ValueError("overlap has an empty retained subspace at k=%d" % ik)
            V_h = V[:, healthy]
            w_h = w[healthy]
            M_h = V_h * (1.0 / torch.sqrt(w_h.to(work_dtype))).unsqueeze(0)
            proj_cache[ik] = (M_h, int(healthy.sum().item()), w.min().item())
            L[ik] = torch.eye(dim, device=S_k.device, dtype=work_dtype)
            min_w_overall = min(min_w_overall, w.min().item())
            n_dropped_max = max(n_dropped_max, dim - int(healthy.sum().item()))
        print(
            "[WM-TrueDiag-WARN] ill-conditioned S(k) on %d/%d k-points (min_w=%.4f), applied positive-subspace projection (dropped %d modes)"
            % (len(bad), n_k, min_w_overall, n_dropped_max),
            flush=True,
        )

    eye = (
        torch.eye(dim, device=S_k.device, dtype=work_dtype)
        .unsqueeze(0)
        .expand(n_k, dim, dim)
    )
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
    return S_k64, L_inv, bad, proj_cache


def _global_occupations(eigenvalues, nelec, k_weights, valid):
    """Zero-temperature, spin-degenerate occupations on a weighted k sample.

    One common Fermi level; a partially occupied degenerate shell receives the
    same occupation at every k. This conserves electrons without rescaling q.
    The caller must supply integration points, not a plotting band path.
    """
    if not math.isfinite(float(nelec)) or nelec < 0:
        raise ValueError("nelec must be finite and nonnegative")
    nk = eigenvalues.shape[0]
    weights = torch.as_tensor(k_weights, device=eigenvalues.device, dtype=torch.float64)
    if (
        weights.shape != (nk,)
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or not bool(weights.sum() > 0)
    ):
        raise ValueError(
            "k_weights must be finite, nonnegative, length n_k, with positive sum"
        )
    weights = weights / weights.sum()
    active = valid & (weights[:, None] > 0)
    energies = eigenvalues[active]
    if not bool(torch.isfinite(energies).all()):
        raise ValueError("nonfinite eigenvalues in retained subspace")
    capacity = (2.0 * weights[:, None].expand_as(eigenvalues))[active]
    total_capacity = float(capacity.sum())
    if float(nelec) > total_capacity + 1e-8:
        raise ValueError(
            "electron count %.12g exceeds retained capacity %.12g"
            % (nelec, total_capacity)
        )
    occupation = torch.zeros_like(eigenvalues)
    if not energies.numel() or nelec == 0:
        return occupation, weights
    order = torch.argsort(energies)
    ev = energies[order]
    cap = capacity[order]
    # Group numerical degeneracies without selecting an arbitrary eigenvector.
    new_shell = torch.ones_like(ev, dtype=torch.bool)
    new_shell[1:] = (ev[1:] - ev[:-1]).abs() > 1e-8
    group = new_shell.cumsum(0) - 1
    shell_cap = torch.zeros_like(cap).scatter_add_(0, group, cap)
    filled_before = shell_cap.cumsum(0) - shell_cap
    shell_fraction = (
        (float(nelec) - filled_before) / shell_cap.clamp_min(1e-30)
    ).clamp(0, 1)
    sorted_occ = 2.0 * shell_fraction[group]
    occ = torch.empty_like(sorted_occ)
    occ[order] = sorted_occ
    occupation[active] = occ.to(occupation.dtype)
    return occupation, weights


def compute_mulliken_fast(
    H_k: torch.Tensor,
    S_k64: torch.Tensor,
    L_inv: torch.Tensor,
    bad: torch.Tensor,
    proj_cache: dict,
    nelec: float,
    idp,
    atom_types: torch.Tensor,
    k_weights=None,
) -> torch.Tensor:
    """Atomic electron populations for a non-SOC spin-degenerate k sample.

    Default weights are equal. Odd/fractional N and metals use a common Fermi
    level, including partial filling. Not a spin-polarized density solver.
    """
    c128 = S_k64.dtype
    H_k64 = H_k.to(c128)
    if H_k64.shape != S_k64.shape or H_k64.ndim != 3 or H_k64.shape[0] == 0:
        raise ValueError("H and S must have matching nonempty (n_k,n_orb,n_orb) shapes")
    norb_list = idp.atom_norb[atom_types.reshape(-1)]
    if int(norb_list.sum()) != H_k.shape[-1]:
        raise ValueError("atomic orbital partition does not cover H/S")

    # Batched solve for healthy k-points
    H_eff = L_inv @ H_k64 @ L_inv.mH
    H_eff = 0.5 * (H_eff + H_eff.mH)
    evals, evecs = torch.linalg.eigh(H_eff)
    C = L_inv.mH @ evecs
    valid = torch.ones_like(evals, dtype=torch.bool)

    # Override ill-conditioned k-points with exact projection
    if len(bad) > 0:
        C, evals = C.clone(), evals.clone()
        for ik in bad.tolist():
            M_h, n_h, min_w = proj_cache[ik]
            if n_h == 0:
                raise ValueError("overlap has an empty retained subspace at k=%d" % ik)
            H_eff_bad = M_h.mH @ H_k64[ik] @ M_h
            H_eff_bad = 0.5 * (H_eff_bad + H_eff_bad.mH)
            evals_bad, evecs_bad = torch.linalg.eigh(H_eff_bad)
            C[ik] = 0
            C[ik, :, :n_h] = M_h @ evecs_bad
            evals[ik, :n_h] = evals_bad
            valid[ik, n_h:] = False

    if k_weights is None:
        k_weights = torch.ones(H_k.shape[0], device=H_k.device, dtype=torch.float64)
    occupation, weights = _global_occupations(evals, nelec, k_weights, valid)
    P = (C * occupation.unsqueeze(-2)) @ C.mH
    PS = P @ S_k64
    mulliken_diag = (torch.diagonal(PS, dim1=-2, dim2=-1).real * weights[:, None]).sum(
        dim=0
    )
    if not bool(torch.isfinite(mulliken_diag).all()) or not math.isclose(
        float(mulliken_diag.sum()), float(nelec), rel_tol=1e-7, abs_tol=1e-6
    ):
        raise RuntimeError(
            "Mulliken electron conservation failed: expected %.12g got %.12g"
            % (nelec, float(mulliken_diag.sum()))
        )

    starts = torch.cat(
        [torch.tensor([0], device=H_k.device), torch.cumsum(norb_list, dim=0)[:-1]]
    )
    ends = torch.cumsum(norb_list, dim=0)
    q_atom = torch.zeros(len(norb_list), device=H_k.device, dtype=torch.float32)
    for i, (st, en) in enumerate(zip(starts, ends)):
        q_atom[i] = mulliken_diag[st:en].sum().float()
    return q_atom


def _eval_occupation_kpoints(n_graph, n_k, device):
    """Fixed uniform-BZ quadrature, independent of a dataset's plotting path.

    A scrambled Sobol sample is deterministic and does not consume the training
    RNG. Small n_k is an approximation; converge it before physical claims.
    """
    if n_graph < 1 or n_k < 1:
        raise ValueError("occupation sampling requires positive graph and k counts")
    pts = torch.quasirandom.SobolEngine(3, scramble=True, seed=20260910).draw(n_k)
    return pts.to(device).unsqueeze(0).expand(n_graph, -1, -1).contiguous()
