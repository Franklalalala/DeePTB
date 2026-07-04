# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grassmann-manifold toolkit and occupied-projector (P) regression for DeePTB.

Motivation
----------
The 2026-07-04 audit showed that a Hamiltonian predictor's element-wise error is
organized in the spectrum: the harmful part lives in high-energy virtual states,
while the near-Fermi / occupied subspace is what band structure actually reads.
The occupied projector ``P`` (a.k.a. the S-metric density matrix) is the object
that naturally lives on a curved manifold: the set of rank-``n_occ`` idempotent
projectors is the **Grassmann manifold** ``Gr(n_occ, N)``.  This module implements
the Grassmann geometry and a training loss that regresses ``P`` instead of raw
matrix elements, so model capacity is spent on the occupied subspace.

Design
------
* All geometry is done in the **symmetrically orthogonalized basis** ``S^{1/2}``
  so the overlap becomes identity and standard (orthonormal) Grassmann formulas
  apply.  AO <-> orthogonalized transport uses ``X = S^{1/2}`` / ``X^{-1}``.
* The occupied projector of ``H`` is a **smooth** function of ``H`` exactly when a
  HOMO-LUMO gap is present (insulators/semiconductors), so differentiating through
  ``eigh`` is well conditioned for the intended non-SOC band-regression task.
* The reference Grassmann point is built under ``no_grad``; gradients flow only
  through the predicted Hamiltonian, exactly mirroring ``riemannian_alignment``.
* The default training distance is the **chordal** (Frobenius) distance
  ``||P_pred - P_ref||_F^2``: it is smooth everywhere *including at the optimum*.
  The **geodesic** (principal-angle) distance is provided for diagnostics and for
  the product-manifold flow story, but its gradient is singular at zero distance,
  so it is not the default loss.

This module reuses the numerically-hardened helpers from
``dptb.nnops.riemannian_alignment`` (the manifold-alignment "legacy") and adds the
Grassmann layer on top of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# --- reuse the legacy manifold-alignment helpers when available ---------------
try:
    from dptb.nnops.riemannian_alignment import (
        hermitian_part,
        regularize_overlap,
        generalized_eigh,
    )
except Exception:  # pragma: no cover - keep the math core importable standalone
    def hermitian_part(x: "torch.Tensor") -> "torch.Tensor":
        return 0.5 * (x + x.mH)

    def regularize_overlap(s: "torch.Tensor", eig_floor: float = 1.0e-10) -> "torch.Tensor":
        s_h = hermitian_part(s)
        evals, evecs = torch.linalg.eigh(s_h)
        evals = evals.clamp_min(float(eig_floor))
        return (evecs * evals.unsqueeze(-2)) @ evecs.mH

    def generalized_eigh(h, s=None, eig_floor: float = 1.0e-10):
        h_h = hermitian_part(h)
        if s is None:
            return torch.linalg.eigh(h_h)
        s_h = regularize_overlap(s, eig_floor=eig_floor)
        se, su = torch.linalg.eigh(s_h)
        inv_sqrt = (su * se.clamp_min(float(eig_floor)).rsqrt().unsqueeze(-2)) @ su.mH
        h_orth = hermitian_part(inv_sqrt.mH @ h_h @ inv_sqrt)
        eps, q = torch.linalg.eigh(h_orth)
        return eps, inv_sqrt @ q

try:
    from dptb.nnops.loss import Loss
except Exception:  # pragma: no cover
    class _DummyLoss:
        @staticmethod
        def register(_name):
            def deco(cls):
                return cls
            return deco
    Loss = _DummyLoss()

try:
    from dptb.data import AtomicDataDict, _keys
except Exception:  # pragma: no cover
    class _Keys:
        HAMILTONIAN_KEY = "hamiltonian"
        OVERLAP_KEY = "overlap"
        KPOINT_KEY = "kpoint"
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"
        NODE_OVERLAP_KEY = "node_overlap"
        EDGE_OVERLAP_KEY = "edge_overlap"
    _keys = _Keys()

    class AtomicDataDict:  # type: ignore[no-redef]
        HAMILTONIAN_KEY = _keys.HAMILTONIAN_KEY
        OVERLAP_KEY = _keys.OVERLAP_KEY
        KPOINT_KEY = _keys.KPOINT_KEY
        NODE_FEATURES_KEY = _keys.NODE_FEATURES_KEY
        EDGE_FEATURES_KEY = _keys.EDGE_FEATURES_KEY
        NODE_OVERLAP_KEY = _keys.NODE_OVERLAP_KEY
        EDGE_OVERLAP_KEY = _keys.EDGE_OVERLAP_KEY

try:
    from dptb.nn.hr2hk import HR2HK
except Exception:  # pragma: no cover
    HR2HK = None

Tensor = torch.Tensor


# ============================================================================
# Orthogonalization (S-metric handling)
# ============================================================================
def _stable_eigh(a: Tensor) -> Tuple[Tensor, Tensor]:
    """Hermitian eigendecomposition done in double precision, cast back.

    float32 ``torch.linalg.eigh`` frequently fails to converge on ill-conditioned
    inputs (near-linearly-dependent NAO overlaps, random-init predicted H on crystals).
    Solving in float64/complex128 is far more robust; the dtype casts are autograd-
    transparent so gradients still flow to the float32 model.
    """
    hi = torch.complex128 if a.is_complex() else torch.float64
    w, v = torch.linalg.eigh(a if a.dtype == hi else a.to(hi))
    w_dtype = torch.float64 if a.dtype in (torch.float64, torch.complex128) else torch.float32
    return w.to(w_dtype), v.to(a.dtype)


def s_half_and_inv(s: Tensor, eig_floor: float = 1.0e-10) -> Tuple[Tensor, Tensor]:
    """Return ``(S^{1/2}, S^{-1/2})`` for a (regularized) overlap matrix.

    Working in the ``S^{1/2}`` basis turns the generalized eigenproblem into an
    ordinary one and makes the standard Grassmann formulas apply.
    """
    s_h = hermitian_part(s)
    evals, evecs = _stable_eigh(s_h)
    evals = evals.clamp_min(float(eig_floor))
    root = evals.sqrt()
    x = (evecs * root.unsqueeze(-2)) @ evecs.mH
    x_inv = (evecs * root.reciprocal().unsqueeze(-2)) @ evecs.mH
    return hermitian_part(x), hermitian_part(x_inv)


def occupied_projector(
    h: Tensor,
    s: Optional[Tensor],
    n_occ: int,
    *,
    eig_floor: float = 1.0e-10,
    return_frame: bool = False,
    from_density: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Occupied projector ``P`` (orthonormal, symmetric idempotent, rank ``n_occ``).

    The generalized eigenproblem ``M C = S C eps`` is solved in the ``S^{1/2}``
    basis; ``P = U Uᴴ`` with ``U`` an orthonormal frame for the occupied subspace.

    * ``from_density=False`` (Hamiltonian source): occupied = the ``n_occ`` **lowest**
      eigenvectors of ``H``.  ``P`` is smooth in ``H`` when the HOMO-LUMO gap is open.
    * ``from_density=True`` (density-matrix source): ``M`` is a predicted/label
      **density matrix** whose eigenvalues are occupations; occupied = the ``n_occ``
      **highest**-occupation eigenvectors.  This is the retraction of a regressed
      density onto the Grassmann manifold -- the DM-regression route (get_DM).

    With ``return_frame``: also returns the frame ``U`` (N x n_occ) and the full
    eigenvalue/occupation vector ``eps`` (ascending).
    """
    if h.ndim != 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError(f"expected a single square dense matrix, got {tuple(h.shape)}")
    n = h.shape[-1]
    if n_occ <= 0 or n_occ >= n:
        raise ValueError(f"n_occ must be in [1, N-1], got {n_occ} for N={n}")
    h_h = hermitian_part(h)
    if s is None:
        eps, q = _stable_eigh(h_h)
    else:
        x, x_inv = s_half_and_inv(s, eig_floor=eig_floor)
        # Hamiltonians transform as X^{-1} H X^{-1}.  An AO density kernel D obeys
        # D S D = D and its orthonormal projector is X D X (X = S^{1/2}); using
        # X^{-1} for a density selects the WRONG subspace in a non-orthogonal basis.
        transport = x if from_density else x_inv
        h_orth = hermitian_part(transport @ h_h @ transport)
        eps, q = _stable_eigh(h_orth)
    u = q[:, -n_occ:] if from_density else q[:, :n_occ]  # top occupations vs lowest energies
    p = u @ u.mH
    if return_frame:
        return p, u, eps
    return p


# ============================================================================
# Distances on the Grassmann manifold
# ============================================================================
def chordal_distance_sq(p1: Tensor, p2: Tensor) -> Tensor:
    """Squared chordal (projector Frobenius) distance ``||P1 - P2||_F^2 / 2``.

    Equal to ``sum_i sin^2(theta_i)`` over principal angles.  Smooth everywhere,
    including at ``P1 == P2`` -- this is the recommended *loss*.
    """
    diff = p1 - p2
    return 0.5 * (diff.abs() ** 2).sum()


def principal_angles(u1: Tensor, u2: Tensor, eps: float = 1.0e-7) -> Tensor:
    """Principal angles between the column spaces of two orthonormal frames."""
    m = u1.mH @ u2
    sv = torch.linalg.svdvals(m).clamp(0.0, 1.0 - eps)
    return torch.arccos(sv)


def geodesic_distance(u1: Tensor, u2: Tensor) -> Tensor:
    """Grassmann geodesic distance ``||theta||_2`` (diagnostic only).

    Gradient is singular as the distance -> 0, so prefer ``chordal_distance_sq``
    as a training objective.
    """
    theta = principal_angles(u1, u2)
    return (theta ** 2).sum().sqrt()


# ============================================================================
# Exp / Log maps (for geodesic transport; product-manifold flow story)
# ============================================================================
def grassmann_exp(u0: Tensor, delta: Tensor) -> Tensor:
    """Grassmann exponential: geodesic from frame ``u0`` along tangent ``delta``.

    ``delta`` must be horizontal (``u0ᴴ delta = 0``).  Returns an orthonormal frame
    for the endpoint subspace (Edelman-Arias-Smith 1998).
    """
    q, sv, vh = torch.linalg.svd(delta, full_matrices=False)
    v = vh.mH
    cos_s = torch.cos(sv)
    sin_s = torch.sin(sv)
    term1 = ((u0 @ v) * cos_s.unsqueeze(-2)) @ vh
    term2 = (q * sin_s.unsqueeze(-2)) @ vh
    ut = term1 + term2
    # numerical re-orthonormalization
    qq, _ = torch.linalg.qr(ut)
    return qq


def grassmann_log(u0: Tensor, u1: Tensor) -> Tensor:
    """Grassmann logarithm: horizontal tangent at ``u0`` pointing to ``u1``.

    Requires principal angles ``< pi/2`` (``u0ᴴ u1`` invertible), which holds for
    nearby subspaces such as ``P(H0) -> P(H_ref)``.
    """
    m = u0.mH @ u1
    u1_perp = u1 - u0 @ m
    m_inv = torch.linalg.inv(m)
    b = u1_perp @ m_inv
    q, sv, vh = torch.linalg.svd(b, full_matrices=False)
    theta = torch.arctan(sv)
    return (q * theta.unsqueeze(-2)) @ vh


# ============================================================================
# McWeeny purification (differentiable retraction to the manifold)
# ============================================================================
def mcweeny_purify(p: Tensor, n_iter: int = 30, tol: float = 1.0e-10) -> Tensor:
    """Grand-canonical McWeeny purification ``P <- 3P^2 - 2P^3``.

    Retracts an approximate projector (eigenvalues in ``(0,1)`` with the correct
    count above ``0.5``) onto the nearest idempotent.  Matrix-multiplies only, so
    it is a stable differentiable retraction with no eigensolver.
    """
    p = hermitian_part(p)
    for _ in range(int(n_iter)):
        p2 = p @ p
        p_next = hermitian_part(3.0 * p2 - 2.0 * (p2 @ p))
        if float((p_next - p).abs().max()) < tol:
            p = p_next
            break
        p = p_next
    return p


def idempotency_error(p: Tensor) -> Tensor:
    """``||P^2 - P||_F / max(||P||_F, eps)`` -- a scale-free non-idempotency."""
    num = (p @ p - p).norm()
    return num / p.norm().clamp_min(1.0e-30)


# ============================================================================
# Dense P-regression loss core (unit-testable, framework-free)
# ============================================================================
@dataclass(frozen=True)
class GrassmannPResult:
    loss: Tensor
    loss_chordal: Tensor
    loss_eps: Tensor
    geo_dist: Tensor
    n_occ: int
    n_bands: int
    gap: Tensor
    pred_gap: Tensor


def grassmann_p_loss_single(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: int,
    lambda_chordal: float = 1.0,
    lambda_eps: float = 0.0,
    eps_window: Optional[int] = None,
    gauge_mu: bool = True,
    eig_floor: float = 1.0e-10,
    from_density: bool = False,
    min_gap: float = 0.0,
    check_pred_gap: bool = False,
) -> GrassmannPResult:
    """Single-k Grassmann P-regression loss.

    * ``lambda_chordal``: weight on the occupied-subspace chordal distance
      (constrains *which states are occupied* -- the Grassmann object).
    * ``lambda_eps``: optional weight on occupied (+ small virtual window)
      eigenvalue matching, which supplies band *energies* that the pure subspace
      distance does not carry.  A scalar gauge ``mu`` (Fermi shift) is removed.
      Not meaningful when ``from_density=True`` (occupations, not energies).
    * ``from_density``: treat ``h_pred``/``h_ref`` as density matrices and take the
      ``n_occ`` highest-occupation eigenvectors (DM-regression retraction).
    """
    x = x_inv = None
    if s is not None:
        with torch.no_grad():
            x, x_inv = s_half_and_inv(s.detach(), eig_floor=eig_floor)
    sel = slice(-n_occ, None) if from_density else slice(0, n_occ)

    def _project(h, grad: bool):
        h_h = hermitian_part(h)
        if x_inv is None:
            h_orth = h_h
        else:
            transport = x if from_density else x_inv  # X D X for density, X^{-1} H X^{-1} for H
            h_orth = hermitian_part(transport @ h_h @ transport)
        if grad:
            eps, q = _stable_eigh(h_orth)
        else:
            with torch.no_grad():
                eps, q = _stable_eigh(h_orth)
        u = q[:, sel]
        return u @ u.mH, u, eps

    def _boundary_gap(e: Tensor) -> Tensor:
        b = e.shape[-1] - n_occ if from_density else n_occ  # occ/vir boundary index
        return e[b] - e[b - 1]

    with torch.no_grad():
        p_ref, u_ref, eps_ref = _project(h_ref.detach(), grad=False)
    p_pred, u_pred, eps_pred = _project(h_pred, grad=True)

    loss_chordal = chordal_distance_sq(p_pred, p_ref)

    loss_eps = h_pred.real.new_zeros(())
    if lambda_eps != 0.0 and not from_density:
        n = eps_ref.shape[-1]
        if eps_window is None:
            hi = n
        else:
            hi = min(n, n_occ + int(eps_window))
        lo = 0
        de = eps_pred[lo:hi] - eps_ref[lo:hi].detach()
        if gauge_mu:
            de = de - de.mean()
        loss_eps = (de ** 2).mean()

    loss = lambda_chordal * loss_chordal + lambda_eps * loss_eps
    with torch.no_grad():
        geo = geodesic_distance(u_pred, u_ref)
        gap = _boundary_gap(eps_ref).detach()
        pred_gap = _boundary_gap(eps_pred).detach()

    if min_gap and float(min_gap) > 0.0:
        kind = "occupation" if from_density else "HOMO-LUMO"
        if (not torch.isfinite(gap)) or float(gap.abs()) < float(min_gap):
            raise ValueError(f"reference {kind} gap {float(gap):.3g} < min_gap {float(min_gap):.3g} "
                             "(metallic/near-degenerate: projector is ill-conditioned)")
        if check_pred_gap and ((not torch.isfinite(pred_gap)) or float(pred_gap.abs()) < float(min_gap)):
            raise ValueError(f"predicted {kind} gap {float(pred_gap):.3g} < min_gap {float(min_gap):.3g}")

    return GrassmannPResult(
        loss=loss,
        loss_chordal=loss_chordal.detach(),
        loss_eps=loss_eps.detach() if torch.is_tensor(loss_eps) else loss_eps,
        geo_dist=geo,
        n_occ=int(n_occ),
        n_bands=int(eps_ref.shape[-1]),
        gap=gap,
        pred_gap=pred_gap,
    )


def _flatten_mats(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x.reshape((1,) + tuple(x.shape))
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected dense matrices [..., n, n], got {tuple(x.shape)}")
    return x.reshape((-1,) + tuple(x.shape[-2:]))


def dense_grassmann_p_loss(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: Union[int, Sequence[int], Tensor],
    lambda_chordal: float = 1.0,
    lambda_eps: float = 0.0,
    eps_window: Optional[int] = None,
    gauge_mu: bool = True,
    eig_floor: float = 1.0e-10,
    max_kpoints: Optional[int] = None,
    random_kpoints: bool = False,
    from_density: bool = False,
    min_gap: float = 0.0,
    check_pred_gap: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Batched (over k / matrices) Grassmann P-regression loss."""
    if h_pred.shape != h_ref.shape:
        raise ValueError(f"h_pred/h_ref shape mismatch: {tuple(h_pred.shape)} vs {tuple(h_ref.shape)}")
    fp = _flatten_mats(h_pred)
    fr = _flatten_mats(h_ref)
    fs = None
    if s is not None:
        fs = _flatten_mats(s) if s.ndim > 2 else s.reshape((1,) + tuple(s.shape)).expand(fp.shape)
        if fs.shape != fp.shape:
            raise ValueError(f"S shape mismatch after flatten: {tuple(fs.shape)} vs {tuple(fp.shape)}")

    n_mats = fp.shape[0]
    if max_kpoints is not None and 0 < int(max_kpoints) < n_mats:
        if random_kpoints:
            idx = torch.randperm(n_mats, device=fp.device)[: int(max_kpoints)]
        else:
            idx = torch.linspace(0, n_mats - 1, steps=int(max_kpoints), device=fp.device).round().long().unique()
    else:
        idx = torch.arange(n_mats, device=fp.device)

    def _n_occ_at(i: int) -> int:
        if torch.is_tensor(n_occ):
            fl = n_occ.reshape(-1)
            return int(fl[min(i, fl.numel() - 1)].item())
        if isinstance(n_occ, (list, tuple)):
            return int(n_occ[min(i, len(n_occ) - 1)])
        return int(n_occ)

    results = []
    for j in idx.tolist():
        ss = None if fs is None else fs[int(j)]
        results.append(
            grassmann_p_loss_single(
                fp[int(j)], fr[int(j)], ss,
                n_occ=_n_occ_at(int(j)),
                lambda_chordal=lambda_chordal,
                lambda_eps=lambda_eps,
                eps_window=eps_window,
                gauge_mu=gauge_mu,
                eig_floor=eig_floor,
                from_density=from_density,
                min_gap=min_gap,
                check_pred_gap=check_pred_gap,
            )
        )
    if not results:
        z = fp.real.new_zeros(())
        return z, {"grassmann_skipped": fp.real.new_ones(())}

    loss = torch.stack([r.loss for r in results]).mean()
    stats = {
        "grassmann_loss": loss.detach(),
        "grassmann_chordal": torch.stack([r.loss_chordal for r in results]).mean(),
        "grassmann_eps": torch.stack([torch.as_tensor(r.loss_eps) for r in results]).mean(),
        "grassmann_pred_gap": torch.stack([r.pred_gap for r in results]).mean(),
        "grassmann_geo_dist": torch.stack([r.geo_dist for r in results]).mean(),
        "grassmann_gap": torch.stack([r.gap for r in results]).mean(),
        "grassmann_rank": fp.real.new_tensor(float(sum(r.n_occ for r in results) / len(results))),
        "grassmann_skipped": fp.real.new_zeros(()),
    }
    return loss, stats


# ============================================================================
# Registered DeePTB loss
# ============================================================================
def _valence_electron_table() -> Dict[int, int]:
    """Minimal pseudopotential valence-electron count fallback (extend as needed)."""
    return {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
        11: 9, 12: 10, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
        19: 9, 20: 10, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7,
        49: 3, 50: 4, 51: 5, 52: 6, 53: 7,
    }


@Loss.register("grassmann_p_align")
@Loss.register("p_regression")
class GrassmannPAlignLoss(nn.Module):
    """Occupied-projector (Grassmann) regression loss for non-SOC band prediction.

    Assembles dense ``H(k)``/``S(k)`` from predicted vs. reference features via
    ``HR2HK`` at a fixed k-set, forms the occupied projectors, and penalizes their
    chordal distance (plus an optional occupied-eigenvalue term for band energies).

    Use as a normal ``loss_options.train`` method::

        loss_options:
          train:
            method: grassmann_p_align
            coeff_align: 1.0
            lambda_chordal: 1.0
            lambda_eps: 0.1
            base_loss_options: {method: hamil_abs, coeff_base: 0.0}
    """

    def __init__(
        self,
        basis: Optional[dict] = None,
        idp: Any = None,
        overlap: bool = True,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        n_occ: Optional[int] = None,
        n_occ_key: str = "n_occ",
        kpoints: Optional[Sequence[Sequence[float]]] = None,
        lambda_chordal: float = 1.0,
        lambda_eps: float = 0.1,
        eps_window: Optional[int] = 8,
        gauge_mu: bool = True,
        eig_floor: float = 1.0e-10,
        max_kpoints: Optional[int] = None,
        coeff_align: float = 1.0,
        coeff_base: float = 0.0,
        base_loss_options: Optional[Mapping[str, Any]] = None,
        valence_fallback: bool = True,
        all_electron: bool = False,
        skip_on_error: bool = False,
        from_density: bool = False,
        density_key: str = "density_matrix",
        min_gap: float = 0.0,
        check_pred_gap: bool = False,
        require_overlap: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.from_density = bool(from_density)
        self.density_key = str(density_key)
        self.min_gap = float(min_gap)
        self.check_pred_gap = bool(check_pred_gap)
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.idp = idp
        self.overlap = bool(overlap)
        self.n_occ = None if n_occ is None else int(n_occ)
        self.n_occ_key = str(n_occ_key)
        self.kpoints = [[0.0, 0.0, 0.0]] if kpoints is None else [list(map(float, k)) for k in kpoints]
        self.lambda_chordal = float(lambda_chordal)
        self.lambda_eps = float(lambda_eps)
        self.eps_window = None if eps_window is None else int(eps_window)
        self.gauge_mu = bool(gauge_mu)
        self.eig_floor = float(eig_floor)
        self.max_kpoints = None if max_kpoints is None else int(max_kpoints)
        self.coeff_align = float(coeff_align)
        self.coeff_base = float(coeff_base)
        self.valence_fallback = bool(valence_fallback)
        self.all_electron = bool(all_electron)
        self.skip_on_error = bool(skip_on_error)
        self.require_overlap = bool(self.overlap if require_overlap is None else require_overlap)
        self.last_scalar_state: Dict[str, Tensor] = {}
        self._vtab = _valence_electron_table()

        self.base_loss = None
        if base_loss_options:
            opts = dict(base_loss_options)
            opts.pop("enabled", None)
            if "method" in opts:
                self.base_loss = Loss(**opts, idp=idp, basis=basis, dtype=dtype, device=device, overlap=overlap, **kwargs)

        self.h2k = self.s2k = None
        if HR2HK is not None and idp is not None:
            self.h2k = HR2HK(
                idp=idp,
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.HAMILTONIAN_KEY,
                dtype=dtype, device=device,
            )
            # Build the S(k) assembler whenever an overlap is wanted. `self.overlap`
            # reflects whether the *model* predicts overlap; the reference S we need
            # for the Grassmann projector comes from the dataset (get_overlap) and is
            # gated by require_overlap -- so decouple from the model-overlap flag.
            if self.overlap or self.require_overlap:
                self.s2k = HR2HK(
                    idp=idp, overlap=True,
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    out_field=AtomicDataDict.OVERLAP_KEY,
                    dtype=dtype, device=device,
                )

    @staticmethod
    def _get(data: Mapping[str, Any], key: str, default=None):
        try:
            return data[key]
        except Exception:
            return default

    def _atomic_numbers(self, data, ref_data):
        """Return per-atom Z as a list[int], or None.

        Prefers an explicit ``atomic_numbers`` field; a processed AtomicData batch
        usually carries only ``atom_types`` (basis indices), so fall back to mapping
        those through the idp's ``type_names`` (element symbols) -> Z.
        """
        akey = AtomicDataDict.ATOMIC_NUMBERS_KEY if hasattr(AtomicDataDict, "ATOMIC_NUMBERS_KEY") else "atomic_numbers"
        for src in (ref_data, data):
            z = self._get(src, akey, None)
            if z is not None:
                return [int(v) for v in torch.as_tensor(z).reshape(-1).tolist()]
        tkey = AtomicDataDict.ATOM_TYPE_KEY if hasattr(AtomicDataDict, "ATOM_TYPE_KEY") else "atom_types"
        type_names = getattr(self.idp, "type_names", None)
        if type_names is not None:
            try:
                from ase.data import atomic_numbers as _AN
                for src in (ref_data, data):
                    t = self._get(src, tkey, None)
                    if t is not None:
                        return [int(_AN[type_names[int(ti)]]) for ti in torch.as_tensor(t).reshape(-1).tolist()]
            except Exception:
                return None
        return None

    def _resolve_n_occ(self, data, ref_data):
        for src in (ref_data, data):
            v = self._get(src, self.n_occ_key, None)
            if v is not None:
                return int(v.reshape(-1)[0].item()) if torch.is_tensor(v) else int(v)
        if self.n_occ is not None:
            return self.n_occ
        zs = self._atomic_numbers(data, ref_data)
        if self.all_electron and zs:
            # All-electron, neutral, closed-shell: n_occ = (sum of Z) / 2 exactly.
            # Correct for all-electron GTO/def2 data where core states are occupied
            # (the pseudopotential-valence table below would undercount).
            nel = sum(zs)
            if nel > 0 and nel % 2 == 0:
                return nel // 2
        if self.valence_fallback and zs:
            if any(zi not in self._vtab for zi in zs):
                return None  # unknown species -> force explicit n_occ, never a silent guess
            nel = sum(self._vtab[zi] for zi in zs)
            if nel > 0 and nel % 2 == 0:
                return nel // 2
        return None

    def _num_structures(self, data: Mapping[str, Any]) -> int:
        ptr = self._get(data, AtomicDataDict.BATCH_PTR_KEY if hasattr(AtomicDataDict, "BATCH_PTR_KEY") else "ptr", None)
        if torch.is_tensor(ptr) and ptr.numel() >= 2:
            return int(ptr.numel() - 1)
        batch = self._get(data, AtomicDataDict.BATCH_KEY if hasattr(AtomicDataDict, "BATCH_KEY") else "batch", None)
        if torch.is_tensor(batch) and batch.numel() > 0:
            return int(batch.detach().max().item()) + 1
        return 1

    def _build_hk(self, data, module, out_key) -> Tensor:
        kpts = torch.as_tensor(self.kpoints, device=self.device, dtype=self.dtype)
        d = dict(data)
        d[AtomicDataDict.KPOINT_KEY] = kpts
        d = module(d)
        return d[out_key]

    def _zero(self, like, skipped=1.0):
        z = (like.real.new_zeros(()) if torch.is_tensor(like) else torch.zeros((), device=self.device, dtype=self.dtype))
        self.last_scalar_state = {"grassmann_loss": z.detach(), "grassmann_skipped": z.detach() + float(skipped)}
        return z

    def forward(self, data: Mapping[str, Any], ref_data: Optional[Mapping[str, Any]] = None) -> Tensor:
        if ref_data is None:
            ref_data = data
        if self.h2k is None:
            raise RuntimeError("GrassmannPAlignLoss requires HR2HK and an idp.")
        h_pred = h_ref = None
        try:
            # This dense, single-structure path assumes one system per loss call.
            # Reject PyG batches until per-structure HR2HK/n_occ handling exists.
            if self._num_structures(data) != 1 or self._num_structures(ref_data) != 1:
                raise NotImplementedError(
                    "grassmann_p_align currently assumes one structure per loss call; use "
                    "batch_size=1 (or implement per-structure HR2HK/n_occ before batch>1).")
            n_occ = self._resolve_n_occ(data, ref_data)
            if n_occ is None:
                if self.skip_on_error:
                    return self._zero(None)
                raise KeyError(f"grassmann_p_align needs `{self.n_occ_key}`, an explicit n_occ, or a known-species valence fallback")
            if self.from_density:
                # DM route (dataset get_DM). Prefer an explicit dense density field;
                # otherwise the node/edge feature slots hold density blocks under
                # get_DM and HR2HK assembles the density the same way it assembles H.
                dp, dr = self._get(data, self.density_key), self._get(ref_data, self.density_key)
                if torch.is_tensor(dp) and torch.is_tensor(dr):
                    h_pred = dp.to(device=self.device)
                    h_ref = dr.to(device=self.device, dtype=h_pred.dtype)
                else:
                    h_pred = self._build_hk(data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
                    h_ref = self._build_hk(ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            else:
                h_pred = self._build_hk(data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
                h_ref = self._build_hk(ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            s = None
            if self.s2k is not None and self._get(ref_data, AtomicDataDict.EDGE_OVERLAP_KEY) is not None:
                s = self._build_hk(ref_data, self.s2k, AtomicDataDict.OVERLAP_KEY)
            if self.require_overlap and s is None:
                raise KeyError("grassmann_p_align: require_overlap is set but no overlap was assembled "
                               "(a non-orthogonal NAO basis must not be treated as orthonormal)")
            loss, stats = dense_grassmann_p_loss(
                h_pred, h_ref, s,
                n_occ=n_occ,
                lambda_chordal=self.lambda_chordal,
                lambda_eps=self.lambda_eps,
                eps_window=self.eps_window,
                gauge_mu=self.gauge_mu,
                eig_floor=self.eig_floor,
                max_kpoints=self.max_kpoints,
                from_density=self.from_density,
                min_gap=self.min_gap,
                check_pred_gap=self.check_pred_gap,
            )
        except Exception:
            if self.skip_on_error:
                return self._zero(h_pred if torch.is_tensor(h_pred) else h_ref)
            raise
        self.last_scalar_state = {k: (v.detach() if torch.is_tensor(v) else torch.as_tensor(float(v))) for k, v in stats.items()}
        total = self.coeff_align * loss
        if self.base_loss is not None and self.coeff_base != 0.0:
            base = self.base_loss(data, ref_data)
            total = total + self.coeff_base * base
            self.last_scalar_state["grassmann_base_loss"] = base.detach() if torch.is_tensor(base) else torch.as_tensor(float(base))
        return total
