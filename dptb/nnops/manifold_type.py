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
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dptb.nnops._manifold_math import (
    Tensor,
    grassmann_exp,
    grassmann_log,
    geodesic_distance,
    chordal_distance_sq,
    occupied_projector,
)


class Manifold(ABC):
    """Minimal Riemannian manifold interface (exp/log/proj/metric/transport)."""

    # dtype-keyed tolerance (real + complex), shared convention with _manifold_math
    EPS = {
        torch.float32: 1.0e-4,
        torch.float64: 1.0e-7,
        torch.complex64: 1.0e-4,
        torch.complex128: 1.0e-7,
    }

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

    # --- provided for free -------------------------------------------------- #
    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Point on the geodesic from ``x0`` to ``x1`` at time ``t`` in [0, 1]."""
        return self.projx(self.exp_map(x0, t * self.log_map(x0, x1)))

    def geodesic_with_tangent(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(x_t, dx_t/dt)`` in one forward pass via ``torch.func.jvp``.

        This is the du/dt mechanism the Riemannian-MeanFlow objective consumes: the
        forward-mode derivative of the geodesic interpolant w.r.t. time.
        """
        def gamma(_t: Tensor) -> Tensor:
            return self.geodesic_interpolant(x0, x1, _t)

        return torch.func.jvp(gamma, (t,), (torch.ones_like(t),))

    def square_norm_at(self, x: Tensor, v: Tensor) -> Tensor:
        return self.metric(x, v, v)

    def eps(self, dtype: torch.dtype) -> float:
        return self.EPS.get(dtype, 1.0e-7)


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

    def parallel_transport(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport ``v`` (tangent at ``x``) to ``y`` (Edelman-Arias-Smith eq. 2.4).

        With the compact SVD of the connecting tangent ``delta = log(x, y) = Q S Wᴴ``,
        transport to ``t = 1`` acts by
        ``tau = (-x W sin(S) Qᴴ + Q cos(S) Qᴴ + (I - Q Qᴴ))`` applied to ``v``.
        """
        delta = self.log_map(x, y)
        q, s, wh = torch.linalg.svd(delta, full_matrices=False)
        w = wh.mH
        sin_s = torch.sin(s).to(q.dtype)
        cos_s = torch.cos(s).to(q.dtype)
        # rotate the components of v that live in span(Q) / span(x W)
        qhv = q.mH @ v
        term_rot = (x @ w) @ (-sin_s.unsqueeze(-1) * qhv) + q @ (cos_s.unsqueeze(-1) * qhv)
        term_perp = v - q @ qhv                      # component orthogonal to both frames
        return term_rot + term_perp

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


__all__ = ["Manifold", "GrassmannManifold", "GrassmannTangentWrapper"]
