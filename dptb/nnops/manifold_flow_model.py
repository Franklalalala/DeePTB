# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learnable velocity head + product-manifold flow-loss assembly (the "接 flow" layer).

:mod:`dptb.nnops.manifold_flow` holds the *pure geometry* of the flow (samplers, split-flow,
MeanFlow identity, euler sampling).  This module adds the two things a trainer needs to
actually optimise a flow on the product manifold Euclidean(delta-H) (+) Grassmann(occupied P):

* :class:`EndpointProductHead` -- a minimal, **endpoint-parameterised** average-velocity head.
  It carries a learnable product endpoint ``x1_hat`` and returns the exact average velocity of
  the conditional geodesic toward it, ``u(x, t, s) = log(x, x1_hat) / (1 - t)`` (which for a
  geodesic is independent of ``s``).  This mirrors the CFM endpoint parameterisation the
  production e3tb head uses, and -- being exactly expressive -- lets the whole flow assembly be
  **trained offline** (loss -> 0 by gradient descent through exp/log/proju/QR/distance) before
  the real RME head is wired into ``flow_cfm`` (handoff Part 4-3).

* :func:`product_flow_loss` -- the balanced training objective
  ``L = L_H(Euclidean) + lambda * L_P(chordal, tangent metric) + mu * consistency``, ``lambda``
  acting as the implicit-metric / shaping-prior knob, ``mu`` weighting the physical coupling
  ``P_grassmann vs occupied_projector(H_euclidean)`` (:func:`projector_consistency_loss`).

* :func:`conduction_window_loss` -- the "分而治之" companion for states at/above the Fermi
  level that the occupied projector cannot see: a fixed-reference-window Euclidean block
  penalty ``||C_w^H (H_pred - H_ref) C_w||_F^2`` (no eig-through-gradient, no manifold
  ill-conditioning at the window edge -- handoff Part 2).

Nothing here is imported by the shipping P-regression loss; it is the flow-training layer.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from dptb.nnops._manifold_math import Tensor, chordal_distance_sq, occupied_projector
from dptb.nnops.manifold_type import GrassmannManifold, EuclideanManifold, ProductManifold
from dptb.nnops.manifold_flow import (
    forward_flow,
    meanflow_average_velocity_target,
    validation_endpoint_loss,
)


class EndpointProductHead(nn.Module):
    """Endpoint-parameterised average-velocity head on a 2-factor product (Euclidean, Grassmann).

    Holds a learnable Euclidean endpoint ``h1`` and a raw Grassmann-frame parameter (re-
    orthonormalised by ``projx`` each call so it always represents a point on ``Gr``).
    ``forward(x, t, s) = log(x, x1_hat) / (1 - t)`` is the exact average velocity of the
    conditional geodesic from ``x`` toward the predicted endpoint (``forward_flow`` projects the
    Grassmann part horizontal).  A toy reference head to exercise/train the flow assembly
    offline; production swaps in the e3tb RME head.
    """

    def __init__(self, manifold: ProductManifold, h_shape: Tuple[int, ...], n: int, k: int,
                 *, dtype: torch.dtype = torch.float64, seed: int = 0):
        super().__init__()
        if len(manifold.manifolds) != 2:
            raise ValueError("EndpointProductHead expects a 2-factor (Euclidean, Grassmann) product")
        self.manifold = manifold
        g = torch.Generator().manual_seed(int(seed))
        self.h1 = nn.Parameter(torch.randn(*h_shape, generator=g, dtype=dtype))
        self.raw_u = nn.Parameter(torch.randn(n, k, generator=g, dtype=dtype))

    @property
    def grassmann(self) -> GrassmannManifold:
        return self.manifold.manifolds[1]

    def endpoint(self):
        """Current predicted product endpoint ``(h1, projx(raw_u))`` (frame on Gr)."""
        return (self.h1, self.grassmann.projx(self.raw_u))

    def forward(self, x, t, s):
        delta = self.manifold.log_map(x, self.endpoint())          # tuple tangent toward endpoint
        return self.manifold.scale_tangent(delta, 1.0 / (1.0 - t))  # avg velocity over [t, 1]


def projector_consistency_loss(u_frame: Tensor, h: Tensor, s: Optional[Tensor] = None,
                               *, n_occ: int, eig_floor: float = 1.0e-10) -> Tensor:
    """Chordal distance between the Grassmann point ``P = U Uᴴ`` and the occupied projector of ``H``.

    The physical coupling (the ``mu`` term): keeps the flowed occupied subspace ``P`` consistent
    with the occupied subspace *implied* by the flowed Hamiltonian ``H``.  ``0`` iff ``U`` spans
    the ``n_occ`` lowest eigenstates of ``H`` (in the ``S`` metric).  Gradients flow to both
    ``U`` and ``H`` as passed by the caller.
    """
    p_grass = u_frame @ u_frame.mH
    p_h = occupied_projector(h, s, int(n_occ), eig_floor=eig_floor)
    return chordal_distance_sq(p_grass, p_h)


def conduction_window_loss(h_pred: Tensor, h_ref: Tensor, c_window: Tensor) -> Tensor:
    """Fixed-reference-window Euclidean block penalty ``||C_wᴴ (H_pred - H_ref) C_w||_F^2``.

    For states at/above the Fermi level that the occupied projector cannot reach (they live in
    ``Q = I - P``), penalise the Hamiltonian directly inside a **fixed** reference window frame
    ``C_w`` (occupied + a conduction margin).  Because ``C_w`` is a fixed/detached reference, the
    gradient never differentiates an eig through a moving window edge (no ``1/gap`` blow-up, no
    rank jump) -- the clean "divide and conquer" partner to the occupied Grassmann-P factor
    (handoff Part 2).  Batch axes are preserved.
    """
    proj = c_window.mH @ (h_pred - h_ref) @ c_window
    return (proj.abs() ** 2).sum(dim=(-2, -1))


def product_flow_loss(manifold: ProductManifold, head: Callable, x0, x1,
                      t: Tensor, s: Tensor, *, lam: float = 1.0, mu: float = 0.0,
                      consistency_fn: Optional[Callable] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Balanced product-manifold MeanFlow loss ``L = L_H + lambda*L_P (+ mu*consistency)``.

    Splits the MeanFlow tangent residual into its Euclidean (delta-H) and Grassmann (P) parts so
    ``lambda`` can weight the occupied-subspace factor independently (the implicit-metric knob),
    and optionally adds ``mu * consistency_fn(x_t)`` (e.g. :func:`projector_consistency_loss`
    tying the flowed ``P`` to the flowed ``H``).  Returns ``(loss, parts)`` with detached
    ``L_H``/``L_P``/``consistency`` for telemetry.
    """
    if len(manifold.manifolds) != 2:
        raise ValueError("product_flow_loss expects a 2-factor (Euclidean, Grassmann) product")
    euclid, grass = manifold.manifolds
    x_t, dx_t = manifold.geodesic_with_tangent(x0, x1, t)
    u, u_tgt = meanflow_average_velocity_target(
        head, x_t, t, s, path_velocity=dx_t, instantaneous_velocity=dx_t, manifold=manifold)
    diff = manifold.proju(x_t, manifold.tangent_sub(u, u_tgt))
    l_h = euclid.square_norm_at(x_t[0], diff[0]).mean()
    l_p = grass.square_norm_at(x_t[1], diff[1]).mean()
    loss = l_h + float(lam) * l_p
    parts = {"L_H": l_h.detach(), "L_P": l_p.detach(),
             "consistency": l_h.detach().new_zeros(())}
    if mu != 0.0 and consistency_fn is not None:
        cons = consistency_fn(x_t)
        cons = cons.mean() if cons.ndim else cons
        loss = loss + float(mu) * cons
        parts["consistency"] = cons.detach()
    return loss, parts


__all__ = [
    "EndpointProductHead",
    "projector_consistency_loss",
    "conduction_window_loss",
    "product_flow_loss",
    "validation_endpoint_loss",
]
