# SPDX-License-Identifier: LGPL-3.0-or-later
"""Riemannian ``Manifold`` abstraction + ``GrassmannManifold``.

This is the geometry *interface* on top of the framework-free primitives in
:mod:`dptb.nnops._manifold_math`.  It mirrors the ``Manifold`` ABC used by the
Riemannian-MeanFlow references so a future ``ProductManifold`` (Euclidean delta-H
(+) Grassmann P) and the MeanFlow / split-flow losses can plug in with no glue.

Design choices (following the review):
* A Grassmann point is carried as an **orthonormal frame** ``U`` (N x n_occ, possibly
  complex), *not* the projector ``P = U Uᴴ``.  exp/log/proju/metric are cheapest on the
  frame; the *training loss* still uses the chordal distance on ``P = to_projector(U)``.
* ``geodesic_with_tangent`` returns ``(x_t, dx_t/dt)`` in one shot via ``torch.func.jvp``
  -- the du/dt mechanism the MeanFlow identity needs, in native PyTorch.

Nothing here is imported by the shipping P-regression loss; it is additive scaffolding
for the product-manifold flow story.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dptb.nnops._manifold_math import (
    Tensor,
    angle_eps,
    scale_like,
    grassmann_exp,
    grassmann_log,
    geodesic_distance,
    chordal_distance_sq,
    occupied_projector,
)


class Manifold(ABC):
    """Minimal Riemannian manifold interface (exp/log/proj/metric/transport)."""

    # --- abstract primitives ------------------------------------------------ #
    @abstractmethod
    def exp_map(self, x: Tensor, u: Tensor) -> Tensor:
        """Move from point ``x`` along tangent ``u`` (assumed horizontal)."""

    @abstractmethod
    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Horizontal tangent at ``x`` pointing to ``y``."""

    @abstractmethod
    def projx(self, x: Tensor) -> Tensor:
        """Project an ambient array back onto the manifold (retraction anchor)."""

    @abstractmethod
    def proju(self, x: Tensor, v: Tensor) -> Tensor:
        """Project an ambient vector ``v`` onto the (horizontal) tangent space at ``x``."""

    @abstractmethod
    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Riemannian inner product ``<u, v>_x`` (real scalar, batched over leading dims)."""

    @abstractmethod
    def geodesic_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance between two points."""

    @abstractmethod
    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel-transport tangent ``v`` at ``x`` along the geodesic to ``y``."""

    # --- tangent / point algebra (tensor defaults; ProductManifold overrides) --- #
    # These route the raw arithmetic the flow losses used to do inline (``c * v``,
    # ``a - b``, ``x.detach()``) through the manifold, so a ``ProductManifold`` whose
    # points/tangents are *tuples* can dispatch component-wise while the single-tensor
    # manifolds keep byte-for-byte identical behavior.
    def scale_tangent(self, v: Tensor, c: "float | Tensor") -> Tensor:
        """Scalar (or leading-batch time) times a tangent, broadcast-safe."""
        return scale_like(c, v) * v

    def tangent_sub(self, a: Tensor, b: Tensor) -> Tensor:
        """Tangent difference ``a - b`` (same base point)."""
        return a - b

    def detach_point(self, x: Tensor) -> Tensor:
        return x.detach()

    def detach_tangent(self, v: Tensor) -> Tensor:
        return v.detach()

    def distance_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """Squared distance used as a *loss* (smooth default: geodesic^2; overridden)."""
        return self.geodesic_distance(x, y) ** 2

    # --- provided for free -------------------------------------------------- #
    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Point on the geodesic from ``x0`` to ``x1`` at time ``t`` in [0, 1].

        ``t`` may be a scalar or a leading-batch ``[B]`` tensor; it is reshaped to
        broadcast over the tangent (a bare ``[B]`` cannot multiply a ``[B, N, K]``
        tangent otherwise).  Uses ``scale_tangent`` so a product manifold dispatches
        the scaling component-wise (identical to ``scale_like(t, delta) * delta`` for a
        single-tensor manifold).
        """
        delta = self.log_map(x0, x1)
        return self.projx(self.exp_map(x0, self.scale_tangent(delta, t)))

    def geodesic_with_tangent(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(x_t, dx_t/dt)`` in one forward pass via ``torch.func.jvp``.

        This is the du/dt mechanism the Riemannian-MeanFlow objective consumes: the
        forward-mode derivative of the geodesic interpolant w.r.t. time.  The velocity is
        projected onto the tangent at ``x_t`` so differentiating through the ``projx`` QR
        cannot leak a non-horizontal (gauge) component.
        """
        def gamma(_t: Tensor) -> Tensor:
            return self.geodesic_interpolant(x0, x1, _t)

        x_t, dx_t = torch.func.jvp(gamma, (t,), (torch.ones_like(t),))
        return x_t, self.proju(x_t, dx_t)

    def square_norm_at(self, x: Tensor, v: Tensor) -> Tensor:
        return self.metric(x, v, v)

    def eps(self, dtype: torch.dtype) -> float:
        return angle_eps(dtype)


class GrassmannManifold(Manifold):
    """``Gr(n_occ, N)`` carried as an orthonormal frame ``U`` (N x n_occ).

    Wraps the ``_manifold_math`` geometry and adds the two primitives it did not yet
    expose: horizontal projection ``proju`` and parallel transport.
    """

    def exp_map(self, x: Tensor, u: Tensor) -> Tensor:
        return grassmann_exp(x, u)

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        return grassmann_log(x, y)

    def projx(self, x: Tensor) -> Tensor:
        """Nearest orthonormal frame (QR); fixes the gauge deterministically."""
        q, r = torch.linalg.qr(x)
        # make the QR gauge deterministic (positive diagonal of R)
        sign = torch.sgn(torch.diagonal(r, dim1=-2, dim2=-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q * sign.unsqueeze(-2)

    def proju(self, x: Tensor, v: Tensor) -> Tensor:
        """Horizontal component: remove the part of ``v`` inside the column space of ``x``."""
        return v - x @ (x.mH @ v)

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Canonical Grassmann metric ``Re tr(uᴴ v)`` for horizontal tangents."""
        return (u.mH @ v).diagonal(dim1=-2, dim2=-1).sum(-1).real

    def geodesic_distance(self, x: Tensor, y: Tensor) -> Tensor:
        return geodesic_distance(x, y)

    def chordal_distance_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """Chordal distance on the derived projectors -- the smooth training distance."""
        return chordal_distance_sq(self.to_projector(x), self.to_projector(y))

    def distance_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """The Grassmann *loss* distance is chordal (smooth at the optimum)."""
        return self.chordal_distance_sq(x, y)

    def geodesic_with_tangent(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Closed-form ``(x_t, dx_t/dt)`` along the Grassmann geodesic (NaN-safe at t=0/1).

        Overrides the ABC's generic ``torch.func.jvp`` route, which differentiates
        through ``svd`` of the *zero* tangent at the endpoints (``t in {0, 1}``) and
        yields a NaN velocity.  Here ``svd(delta)`` is taken once (t-independent) and
        both the point and its exact time-derivative are assembled analytically:
        ``x_t = (x0 V cos(tΣ) + Q sin(tΣ)) Vᴴ`` with ``delta = Q Σ Vᴴ``, and
        ``dx_t = (-x0 V Σ sin(tΣ) + Q Σ cos(tΣ)) Vᴴ`` -- finite for every ``t`` (at
        ``t=0`` it reduces to ``delta`` itself).  Also cheaper (no forward-AD trace).
        """
        delta = self.log_map(x0, x1)
        q, sv, vh = torch.linalg.svd(delta, full_matrices=False)
        v = vh.mH
        angle = scale_like(t, sv) * sv                      # t * Σ (per principal mode)
        x0v = x0 @ v
        cos_a = torch.cos(angle); sin_a = torch.sin(angle)
        x_t = (x0v * cos_a.to(v.dtype).unsqueeze(-2)) @ vh + (q * sin_a.to(q.dtype).unsqueeze(-2)) @ vh
        d_cos = (-sv * sin_a).to(v.dtype)                   # d/dt cos(tΣ)
        d_sin = (sv * cos_a).to(q.dtype)                    # d/dt sin(tΣ)
        dx_t = (x0v * d_cos.unsqueeze(-2)) @ vh + (q * d_sin.unsqueeze(-2)) @ vh
        return x_t, self.proju(x_t, dx_t)

    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport ``v`` (tangent at ``x``) to ``y`` (Edelman-Arias-Smith eq. 2.4).

        With the compact SVD of the connecting tangent ``delta = log(x, y) = Q S Wᴴ``,
        transport to ``t = 1`` acts by
        ``tau = (-x W sin(S) Qᴴ + Q cos(S) Qᴴ + (I - Q Qᴴ))`` applied to ``v``.
        """
        v = self.proju(x, v)                         # accept only the horizontal part of v
        delta = self.log_map(x, y)
        q, s, wh = torch.linalg.svd(delta, full_matrices=False)
        w = wh.mH
        sin_s = torch.sin(s).to(q.dtype)
        cos_s = torch.cos(s).to(q.dtype)
        # rotate the components of v that live in span(Q) / span(x W)
        qhv = q.mH @ v
        term_rot = (x @ w) @ (-sin_s.unsqueeze(-1) * qhv) + q @ (cos_s.unsqueeze(-1) * qhv)
        term_perp = v - q @ qhv                      # component orthogonal to both frames
        # exact arithmetic already lands horizontal at y; re-project to clear fp/QR drift.
        return self.proju(y, term_rot + term_perp)

    # --- constructors / conversions ---------------------------------------- #
    @classmethod
    def from_hamiltonian(cls, h: Tensor, s: Optional[Tensor] = None, *, n_occ: int,
                         eig_floor: float = 1.0e-10, from_density: bool = False) -> Tensor:
        """Occupied frame ``U`` (N x n_occ) from a Hamiltonian (or density) matrix."""
        _p, u, _eps = occupied_projector(
            h, s, n_occ, eig_floor=eig_floor, return_frame=True, from_density=from_density)
        return u

    @staticmethod
    def to_projector(u: Tensor) -> Tensor:
        """``P = U Uᴴ`` (gauge-invariant Grassmann point)."""
        return u @ u.mH


class EuclideanManifold(Manifold):
    """Flat ``R^d`` (or a block of Hamiltonian RMEs) -- the delta-H factor of the product.

    Straight-line geodesics, trivial transport, identity projections.  ``event_ndim``
    is the number of trailing axes that form one point (1 for a feature vector, 2 for a
    dense ``[N, N]`` block); everything to the left is a preserved batch axis, matching
    the Grassmann convention (distances/metrics reduce only the event axes).
    """

    def __init__(self, event_ndim: int = 1):
        self.event_ndim = int(event_ndim)

    def _event_dims(self) -> Tuple[int, ...]:
        return tuple(range(-self.event_ndim, 0))

    def exp_map(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        return x

    def proju(self, x: Tensor, v: Tensor) -> Tensor:
        return v

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return (u.conj() * v).real.sum(dim=self._event_dims())

    def geodesic_distance(self, x: Tensor, y: Tensor) -> Tensor:
        return self.distance_sq(x, y).clamp_min(0.0).sqrt()

    def distance_sq(self, x: Tensor, y: Tensor) -> Tensor:
        return ((x - y).abs() ** 2).sum(dim=self._event_dims())

    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        return v  # flat space: transport is the identity

    def geodesic_with_tangent(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        delta = x1 - x0
        x_t = x0 + scale_like(t, delta) * delta
        # velocity of a straight line is constant (= delta); broadcast to x_t's shape so a
        # batched time still returns a per-item velocity.
        return x_t, delta.expand_as(x_t) if delta.shape != x_t.shape else delta


class ProductManifold(Manifold):
    """Cartesian product ``M_1 x ... x M_k`` -- e.g. Euclidean(delta-H) (+) Grassmann(P).

    A point (and a tangent) is a **tuple** aligned with ``manifolds``; every primitive
    dispatches component-wise.  The Riemannian metric is the **sum** of the component
    metrics (so ``square_norm_at`` and the geodesic distance combine in quadrature), and
    the loss distance is the sum of component ``distance_sq`` (Euclidean L2 on delta-H +
    chordal on P).  ``torch.func.jvp`` treats the tuple as a pytree, so the MeanFlow du/dt
    machinery in :mod:`manifold_flow` works unchanged once the velocity head emits a tuple.
    """

    def __init__(self, manifolds: Sequence[Manifold]):
        self.manifolds = tuple(manifolds)
        if not self.manifolds:
            raise ValueError("ProductManifold needs at least one factor")

    def _check(self, x) -> None:
        if not isinstance(x, (tuple, list)) or len(x) != len(self.manifolds):
            raise ValueError(
                f"product point/tangent must be a {len(self.manifolds)}-tuple, got {type(x).__name__}")

    def exp_map(self, x, u):
        self._check(x); self._check(u)
        return tuple(m.exp_map(xi, ui) for m, xi, ui in zip(self.manifolds, x, u))

    def log_map(self, x, y):
        self._check(x); self._check(y)
        return tuple(m.log_map(xi, yi) for m, xi, yi in zip(self.manifolds, x, y))

    def projx(self, x):
        self._check(x)
        return tuple(m.projx(xi) for m, xi in zip(self.manifolds, x))

    def proju(self, x, v):
        self._check(x); self._check(v)
        return tuple(m.proju(xi, vi) for m, xi, vi in zip(self.manifolds, x, v))

    def metric(self, x, u, v):
        self._check(x)
        parts = [m.metric(xi, ui, vi) for m, xi, ui, vi in zip(self.manifolds, x, u, v)]
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def geodesic_distance(self, x, y):
        self._check(x); self._check(y)
        d2 = [m.geodesic_distance(xi, yi) ** 2 for m, xi, yi in zip(self.manifolds, x, y)]
        out = d2[0]
        for p in d2[1:]:
            out = out + p
        return out.clamp_min(0.0).sqrt()

    def distance_sq(self, x, y):
        self._check(x); self._check(y)
        parts = [m.distance_sq(xi, yi) for m, xi, yi in zip(self.manifolds, x, y)]
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def parallel_transport(self, x, y, v):
        self._check(x); self._check(y); self._check(v)
        return tuple(m.parallel_transport(xi, yi, vi)
                     for m, xi, yi, vi in zip(self.manifolds, x, y, v))

    def geodesic_with_tangent(self, x0, x1, t):
        self._check(x0); self._check(x1)
        pts, tans = [], []
        for m, a, b in zip(self.manifolds, x0, x1):
            p, d = m.geodesic_with_tangent(a, b, t)
            pts.append(p)
            tans.append(d)
        return tuple(pts), tuple(tans)

    # tuple-aware tangent/point algebra (used by the manifold-generic flow losses)
    def scale_tangent(self, v, c):
        return tuple(m.scale_tangent(vi, c) for m, vi in zip(self.manifolds, v))

    def tangent_sub(self, a, b):
        return tuple(m.tangent_sub(ai, bi) for m, ai, bi in zip(self.manifolds, a, b))

    def detach_point(self, x):
        return tuple(m.detach_point(xi) for m, xi in zip(self.manifolds, x))

    def detach_tangent(self, v):
        return tuple(m.detach_tangent(vi) for m, vi in zip(self.manifolds, v))


class GrassmannTangentWrapper(nn.Module):
    """Wrap a flow head so its raw output is a *horizontal* Grassmann tangent at ``U``.

    A learned average-velocity head emits an ambient ``N x n_occ`` array; the geodesic
    formulas assume ``Uᴴ delta = 0``.  This projects to the horizontal space (and can
    cap the tangent norm at the injectivity radius pi/2 so ``exp`` stays a diffeomorphism).
    """

    def __init__(self, head: nn.Module, manifold: Optional[GrassmannManifold] = None,
                 max_angle: Optional[float] = None):
        super().__init__()
        self.head = head
        self.manifold = manifold or GrassmannManifold()
        self.max_angle = None if max_angle is None else float(max_angle)

    def forward(self, u: Tensor, *args, **kwargs) -> Tensor:
        delta = self.head(u, *args, **kwargs)
        delta = self.manifold.proju(u, delta)
        if self.max_angle is not None:
            sv = torch.linalg.svdvals(delta)
            scale = (self.max_angle / sv.amax(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
            delta = delta * scale.unsqueeze(-1)
        return delta


__all__ = ["Manifold", "GrassmannManifold", "EuclideanManifold", "ProductManifold",
           "GrassmannTangentWrapper"]
