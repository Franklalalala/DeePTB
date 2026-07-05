# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-free numerical core for the manifold-alignment losses.

This is the **single** source of the pure linear-algebra / Grassmann geometry used
by both ``grassmann.py`` (occupied-projector P regression) and
``riemannian_alignment.py`` (near-Fermi subspace alignment).  It deliberately imports
**only** ``torch`` -- no ``dptb`` / ``torch_geometric`` -- so it is unit-testable
standalone (no ``importorskip``) and cannot drift into a second copy behind an import
fallback (the previous arrangement redefined ``hermitian_part`` / ``generalized_eigh``
in a ``try/except`` shim, which could silently diverge from the real one).

Numerical hygiene lives here too: eigensolves run in double precision
(:func:`stable_eigh`), ``arccos`` near the singular ``+/-1`` endpoint is replaced by a
linearly-extrapolated variant (:func:`acos_linear_extrapolation`) so principal-angle
gradients stay finite, and the Grassmann log uses ``solve`` rather than an explicit
inverse.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

Tensor = torch.Tensor

# Angle/So2 tolerance keyed by (real) precision -- float32 needs a looser clamp than
# float64; complex dtypes reuse their real-part precision.
_EPS_BY_DTYPE = {
    torch.float32: 1.0e-4,
    torch.float64: 1.0e-7,
    torch.complex64: 1.0e-4,
    torch.complex128: 1.0e-7,
}


def angle_eps(dtype: torch.dtype) -> float:
    """Dtype-keyed epsilon for angle/orthogonality tolerances."""
    return _EPS_BY_DTYPE.get(dtype, 1.0e-7)


def scale_like(scale: "float | Tensor", like: Tensor) -> Tensor:
    """Reshape a scalar / leading-batch time (or interval) so it broadcasts over ``like``.

    A time tensor of shape ``[B]`` cannot directly multiply a tangent of shape
    ``[B, N, K]`` (``[B]`` broadcasts against the *trailing* axis).  This appends the
    missing singleton axes so ``scale_like(t, delta) * delta`` works whether ``t`` is a
    python float, a 0-d tensor, or a ``[B]`` batch.  ``scale`` is cast to ``like``'s real
    dtype (time is real; a real * complex tangent stays complex).
    """
    if not torch.is_tensor(scale):
        return like.real.new_tensor(float(scale))
    scale = scale.to(device=like.device, dtype=like.real.dtype)
    if scale.ndim == 0:
        return scale
    if scale.ndim > like.ndim:
        raise ValueError(
            f"time tensor rank {scale.ndim} cannot broadcast over tangent rank {like.ndim}")
    return scale.reshape(tuple(scale.shape) + (1,) * (like.ndim - scale.ndim))


# --------------------------------------------------------------------------- #
# Hermitian helpers + eigensolvers
# --------------------------------------------------------------------------- #
def hermitian_part(x: Tensor) -> Tensor:
    """Return ``(x + xᴴ) / 2`` over the last two dimensions."""
    return 0.5 * (x + x.mH)


def stable_eigh(a: Tensor) -> Tuple[Tensor, Tensor]:
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


def regularize_overlap(s: Tensor, eig_floor: float = 1.0e-10) -> Tensor:
    """Project an overlap matrix onto the positive-definite cone."""
    s_h = hermitian_part(s)
    evals, evecs = stable_eigh(s_h)
    evals = evals.clamp_min(float(eig_floor)).to(evecs.real.dtype)
    return hermitian_part((evecs * evals.unsqueeze(-2)) @ evecs.mH)


def s_half_and_inv(s: Tensor, eig_floor: float = 1.0e-10) -> Tuple[Tensor, Tensor]:
    """Return ``(S^{1/2}, S^{-1/2})`` for a (regularized) overlap matrix.

    Working in the ``S^{1/2}`` basis turns the generalized eigenproblem into an
    ordinary one and makes the standard (orthonormal) Grassmann formulas apply.
    """
    s_h = hermitian_part(s)
    evals, evecs = stable_eigh(s_h)
    evals = evals.clamp_min(float(eig_floor))
    root = evals.sqrt().to(evecs.real.dtype)
    x = (evecs * root.unsqueeze(-2)) @ evecs.mH
    x_inv = (evecs * root.reciprocal().unsqueeze(-2)) @ evecs.mH
    return hermitian_part(x), hermitian_part(x_inv)


def generalized_eigh(h: Tensor, s: Optional[Tensor] = None, eig_floor: float = 1.0e-10) -> Tuple[Tensor, Tensor]:
    """Solve ``H C = S C eps`` and return S-orthonormal eigenvectors.

    ``s=None`` falls back to the ordinary Hermitian eigenproblem.
    """
    if h.ndim != 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError(f"expected a single square dense matrix, got {tuple(h.shape)}")
    h_h = hermitian_part(h)
    if s is None:
        return stable_eigh(h_h)
    if s.shape != h.shape:
        raise ValueError(f"H/S shape mismatch: {tuple(h.shape)} vs {tuple(s.shape)}")
    x, x_inv = s_half_and_inv(s, eig_floor=eig_floor)
    h_orth = hermitian_part(x_inv @ h_h @ x_inv)
    eps, q = stable_eigh(h_orth)
    return eps, x_inv @ q


def idempotency_error(p: Tensor) -> Tensor:
    """``||P^2 - P||_F / max(||P||_F, eps)`` -- a scale-free non-idempotency."""
    num = (p @ p - p).norm()
    return num / p.norm().clamp_min(1.0e-30)


# --------------------------------------------------------------------------- #
# Occupied projector
# --------------------------------------------------------------------------- #
def occupied_projector(
    h: Tensor,
    s: Optional[Tensor],
    n_occ: int,
    *,
    eig_floor: float = 1.0e-10,
    return_frame: bool = False,
    from_density: bool = False,
) -> "Tensor | Tuple[Tensor, Tensor, Tensor]":
    """Occupied projector ``P`` (orthonormal, symmetric idempotent, rank ``n_occ``).

    The generalized eigenproblem is solved in the ``S^{1/2}`` basis; ``P = U Uᴴ`` with
    ``U`` an orthonormal frame for the occupied subspace.

    * ``from_density=False`` (Hamiltonian source): occupied = the ``n_occ`` **lowest**
      eigenvectors of ``H``.  ``P`` is smooth in ``H`` while the HOMO-LUMO gap is open.
    * ``from_density=True`` (density-matrix source): ``h`` is a predicted/label density
      matrix whose eigenvalues are occupations; occupied = the ``n_occ`` **highest**-
      occupation eigenvectors (the retraction of a regressed density onto Grassmann).
      An AO density kernel ``D`` obeys ``D S D = D`` and its orthonormal projector is
      ``X D X`` (X=S^{1/2}); using ``X^{-1}`` for a density picks the WRONG subspace in
      a non-orthogonal basis, hence the ``transport`` switch below.
    """
    if h.ndim != 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError(f"expected a single square dense matrix, got {tuple(h.shape)}")
    n = h.shape[-1]
    if n_occ <= 0 or n_occ >= n:
        raise ValueError(f"n_occ must be in [1, N-1], got {n_occ} for N={n}")
    h_h = hermitian_part(h)
    if s is None:
        eps, q = stable_eigh(h_h)
    else:
        x, x_inv = s_half_and_inv(s, eig_floor=eig_floor)
        transport = x if from_density else x_inv
        h_orth = hermitian_part(transport @ h_h @ transport)
        eps, q = stable_eigh(h_orth)
    u = q[:, -n_occ:] if from_density else q[:, :n_occ]  # top occupations vs lowest energies
    p = u @ u.mH
    if return_frame:
        return p, u, eps
    return p


# --------------------------------------------------------------------------- #
# Distances on the Grassmann manifold
# --------------------------------------------------------------------------- #
def chordal_distance_sq(p1: Tensor, p2: Tensor) -> Tensor:
    """Squared chordal (projector Frobenius) distance ``||P1 - P2||_F^2 / 2``.

    Equal to ``sum_i sin^2(theta_i)`` over principal angles.  Smooth everywhere,
    including at ``P1 == P2`` -- this is the recommended *loss*.  Only the matrix axes
    are reduced, so leading batch dimensions are preserved (a bare ``[N, N]`` pair still
    returns a 0-d scalar).
    """
    diff = p1 - p2
    return 0.5 * (diff.abs() ** 2).sum(dim=(-2, -1))


def acos_linear_extrapolation(x: Tensor, bound: float = 1.0 - 1.0e-4) -> Tensor:
    """``arccos`` with a linear extension beyond ``|x| > bound``.

    ``arccos`` has an infinite derivative at ``x = +/-1``; when principal-angle cosines
    approach 1 (coincident subspaces -- the optimum) a bare ``arccos`` yields infinite
    gradients.  Beyond ``bound`` we replace ``arccos`` by its first-order Taylor
    expansion, so the gradient stays bounded (following the standard PyTorch3D trick).
    """
    x_mid = x.clamp(-bound, bound)
    acos_mid = torch.arccos(x_mid)
    # d/dx arccos(x) = -1/sqrt(1-x^2)
    slope = -1.0 / torch.sqrt(1.0 - torch.as_tensor(bound, dtype=x.dtype, device=x.device) ** 2)
    return torch.where(x.abs() <= bound, acos_mid, acos_mid + slope * (x - x_mid))


def principal_angles(u1: Tensor, u2: Tensor, eps: Optional[float] = None) -> Tensor:
    """Principal angles between the column spaces of two orthonormal frames."""
    if eps is None:
        eps = angle_eps(u1.dtype)
    m = u1.mH @ u2
    sv = torch.linalg.svdvals(m).clamp(0.0, 1.0)
    return acos_linear_extrapolation(sv, bound=1.0 - float(eps))


def geodesic_distance(u1: Tensor, u2: Tensor) -> Tensor:
    """Grassmann geodesic distance ``||theta||_2`` (diagnostic only).

    Gradient is singular as the distance -> 0, so prefer :func:`chordal_distance_sq`
    as a training objective.  Reduces only the principal-angle axis, so leading batch
    dimensions are preserved (a single frame pair still returns a 0-d scalar).
    """
    theta = principal_angles(u1, u2)
    return (theta ** 2).sum(-1).clamp_min(0.0).sqrt()


# --------------------------------------------------------------------------- #
# Exp / Log maps (geodesic transport; product-manifold flow story)
# --------------------------------------------------------------------------- #
def grassmann_exp(u0: Tensor, delta: Tensor) -> Tensor:
    """Grassmann exponential: geodesic from frame ``u0`` along horizontal tangent ``delta``.

    ``delta`` must be horizontal (``u0ᴴ delta = 0``).  Returns an orthonormal frame for
    the endpoint subspace (Edelman-Arias-Smith 1998).
    """
    q, sv, vh = torch.linalg.svd(delta, full_matrices=False)
    v = vh.mH
    cos_s = torch.cos(sv)
    sin_s = torch.sin(sv)
    term1 = ((u0 @ v) * cos_s.to(v.dtype).unsqueeze(-2)) @ vh
    term2 = (q * sin_s.to(q.dtype).unsqueeze(-2)) @ vh
    ut = term1 + term2
    qq, _ = torch.linalg.qr(ut)  # numerical re-orthonormalization
    return qq


def grassmann_log(u0: Tensor, u1: Tensor) -> Tensor:
    """Grassmann logarithm: horizontal tangent at ``u0`` pointing to ``u1``.

    Well-defined for principal angles ``<= pi/2``.  Computed from the SVD of the
    overlap ``m = u0ᴴ u1 = A cos(Θ) Bᴴ`` and the perpendicular component
    ``u1_perp = u1 - u0 m = Ξ sin(Θ) Bᴴ``, giving the tangent ``delta = Ξ Θ Aᴴ`` with
    ``Θ = atan2(sin Θ, cos Θ)``.  Unlike the earlier ``u1_perp @ m^{-1}`` form, this
    never inverts a singular overlap, so a principal angle of exactly ``pi/2`` (an
    occupied direction going orthogonal, e.g. symmetry-forced or an avoided crossing)
    yields ``Θ = pi/2`` instead of raising ``LinAlgError`` / returning a silent NaN.
    The gradient is still singular exactly at ``pi/2`` (the injectivity-radius boundary,
    where the geodesic is genuinely non-unique).
    """
    tiny = angle_eps(u0.dtype)
    m = u0.mH @ u1
    u1_perp = u1 - u0 @ m
    a, cos_theta, bh = torch.linalg.svd(m)               # m = a diag(cos_theta) bh
    cos_theta = cos_theta.clamp(0.0, 1.0)
    perp_aligned = u1_perp @ bh.mH                        # columns: sin(Θ_i) · Ξ_i
    sin_theta = perp_aligned.norm(dim=-2)                 # real, per principal direction
    xi = perp_aligned / sin_theta.clamp_min(tiny).unsqueeze(-2)   # Ξ (zero-angle cols -> 0·Θ)
    theta = torch.atan2(sin_theta, cos_theta)            # pi/2-safe principal angles
    return (xi * theta.to(xi.dtype).unsqueeze(-2)) @ a.mH


# --------------------------------------------------------------------------- #
# McWeeny purification (differentiable retraction to the manifold)
# --------------------------------------------------------------------------- #
def mcweeny_purify(p: Tensor, n_iter: int = 30, tol: float = 1.0e-10,
                   clamp_spectrum: bool = True, n_occ: Optional[int] = None) -> Tensor:
    """Grand-canonical McWeeny purification ``P <- 3P^2 - 2P^3``.

    The cubic map drives eigenvalues toward ``{0, 1}`` but its fixed point is fixed by
    the eigenvalue **count above 0.5**, not by any target rank -- eigenvalues outside
    ``[0,1]`` diverge, and a spectrum whose count above 0.5 differs from the intended
    occupation converges to the *wrong-rank* idempotent.

    * ``n_occ`` given: the exact, rank-correct retraction.  One eigendecomposition sets
      the top-``n_occ`` occupations to 1 and the rest to 0, returning the nearest rank-
      ``n_occ`` idempotent regardless of where the eigenvalues sit relative to 0.5.
      Prefer this (or :func:`occupied_projector` with ``from_density=True``) for density
      retraction.
    * ``n_occ=None``, ``clamp_spectrum=True`` (default): clamp the spectrum to ``[0,1]``
      then iterate.  This guarantees convergence to *an* idempotent, namely the one whose
      rank equals ``#{eig > 0.5}`` -- the caller must ensure that count is the intended
      occupation; clamping does **not** enforce it.
    * ``n_occ=None``, ``clamp_spectrum=False``: the pure matmul-only path, valid only when
      the input spectrum is already in band with the correct count above 0.5.
    """
    p = hermitian_part(p)
    if n_occ is not None:
        if not 0 < int(n_occ) < p.shape[-1]:
            raise ValueError(f"n_occ must be in [1, N-1], got {n_occ} for N={p.shape[-1]}")
        w, v = stable_eigh(p)  # ascending; top-n_occ occupations are the highest
        w_new = torch.zeros_like(w)
        w_new[..., -int(n_occ):] = 1.0
        return hermitian_part((v * w_new.to(v.dtype).unsqueeze(-2)) @ v.mH)
    if clamp_spectrum:
        w, v = stable_eigh(p)
        w = w.clamp(0.0, 1.0).to(v.dtype)
        p = hermitian_part((v * w.unsqueeze(-2)) @ v.mH)
    for _ in range(int(n_iter)):
        p2 = p @ p
        p_next = hermitian_part(3.0 * p2 - 2.0 * (p2 @ p))
        if float((p_next - p).abs().max()) < tol:
            p = p_next
            break
        p = p_next
    return p


__all__ = [
    "Tensor",
    "angle_eps",
    "scale_like",
    "hermitian_part",
    "stable_eigh",
    "regularize_overlap",
    "s_half_and_inv",
    "generalized_eigh",
    "idempotency_error",
    "occupied_projector",
    "chordal_distance_sq",
    "acos_linear_extrapolation",
    "principal_angles",
    "geodesic_distance",
    "grassmann_exp",
    "grassmann_log",
    "mcweeny_purify",
]
