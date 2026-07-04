# SPDX-License-Identifier: LGPL-3.0-or-later
"""Riemannian flow-matching scaffolding on a manifold (samplers + split-flow + MeanFlow).

This is the "flow story" layer that sits on top of :class:`GrassmannManifold`.  It is
**not** used by the shipping P-regression loss -- it is the machinery for turning the
static occupied-projector regression into a product-manifold Riemannian-MeanFlow, as
laid out in the review's Part A.

Pieces (in increasing order of what they demand):
* :class:`OrderedIntervalSampler` -- the two-time ``(t, s)`` sampler with a boundary
  mass that collapses ``s -> t`` (degenerate = ordinary flow matching), the stabiliser
  the MeanFlow references rely on.
* :func:`geodesic_conditional` -- the conditional geodesic path point + velocity.
* :func:`split_flow_loss` -- integration-free **semigroup** consistency (needs only
  ``exp`` + a geodesic distance); the review's recommended *first* flow milestone.
* :func:`meanflow_average_velocity_target` / :func:`meanflow_loss` -- the average-velocity
  identity via ``torch.func.jvp`` (the du/dt mechanism), stop-grad self-labelled, no EMA
  teacher.

The velocity fields are passed in as callables so the geometry can be exercised (and
unit-tested against analytic geodesics) before any network is wired in.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from dptb.nnops._manifold_math import Tensor
from dptb.nnops.manifold_type import GrassmannManifold, Manifold


class OrderedIntervalSampler:
    """Sample two times in ``[0, 1]`` and order them to ``(t, s)`` with ``t <= s``.

    With probability ``boundary_ratio`` the interval collapses (``s = t``), which reduces
    the MeanFlow objective to ordinary (instantaneous) flow matching -- the stabiliser
    the references force ~75% of the time.
    """

    def __init__(self, boundary_ratio: float = 0.75):
        self.boundary_ratio = float(boundary_ratio)

    def sample(self, n: int, *, device=None, dtype=torch.float64,
               generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        a = torch.rand(n, device=device, dtype=dtype, generator=generator)
        b = torch.rand(n, device=device, dtype=dtype, generator=generator)
        t = torch.minimum(a, b)
        s = torch.maximum(a, b)
        collapse = torch.rand(n, device=device, dtype=dtype, generator=generator) < self.boundary_ratio
        s = torch.where(collapse, t, s)
        return t, s


def geodesic_conditional(manifold: Manifold, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Return ``(x_t, dx_t/dt)`` on the geodesic from ``x0`` to ``x1`` (the FM target path)."""
    return manifold.geodesic_with_tangent(x0, x1, t)


AvgVel = Callable[[Tensor, Tensor, Tensor], Tensor]  # (x, t, s) -> horizontal tangent at x


def forward_flow(manifold: GrassmannManifold, avg_vel: AvgVel, x: Tensor,
                 t: Tensor, s: Tensor) -> Tensor:
    """Step from ``x`` at time ``t`` to time ``s`` with the average velocity: ``exp(x, (s-t) v)``."""
    v = manifold.proju(x, avg_vel(x, t, s))
    return manifold.exp_map(x, (s - t).reshape(*([-1] + [1] * (v.ndim - 1))) * v)


def split_flow_loss(manifold: GrassmannManifold, avg_vel: AvgVel,
                    x0: Tensor, x1: Tensor, t: Tensor, r: Tensor) -> Tensor:
    """Integration-free semigroup consistency (Grassmann split-flow).

    Stepping ``x_t -> r`` in one shot with the learned average velocity must land on the
    geodesic point ``x_r``.  Uses only ``exp`` + a (squared chordal) distance -- no
    forward-mode autodiff, no eigh-in-jvp.  Recommended as the first flow milestone.
    """
    x_t, _ = manifold.geodesic_with_tangent(x0, x1, t)
    x_r, _ = manifold.geodesic_with_tangent(x0, x1, r)
    stepped = forward_flow(manifold, avg_vel, x_t, t, r)
    return manifold.chordal_distance_sq(stepped, x_r.detach())


def meanflow_average_velocity_target(vel_fn: AvgVel, x_t: Tensor, t: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    """MeanFlow target ``u_tgt = u - (t - s) * du/dt`` via one ``torch.func.jvp``.

    ``vel_fn(x, t, s)`` is the average-velocity field; its total time derivative
    ``du/dt`` (holding ``x`` fixed here -- the point-trajectory term is added by the
    caller when training a trajectory) is obtained forward-mode in a single pass.  The
    label is stop-grad (``.detach()``) so no EMA teacher network is needed.
    """
    def u_of_t(tt: Tensor) -> Tensor:
        return vel_fn(x_t, tt, s)

    u, dudt = torch.func.jvp(u_of_t, (t,), (torch.ones_like(t),))
    dt = (t - s).reshape(*([-1] + [1] * (u.ndim - 1)))
    return u, (u - dt * dudt).detach()


def meanflow_loss(manifold: GrassmannManifold, vel_fn: AvgVel, x_t: Tensor,
                  t: Tensor, s: Tensor) -> Tensor:
    """Squared tangent-metric residual between the predicted and target average velocity."""
    u, u_tgt = meanflow_average_velocity_target(vel_fn, x_t, t, s)
    diff = manifold.proju(x_t, u - u_tgt)
    return manifold.square_norm_at(x_t, diff).mean()


__all__ = [
    "OrderedIntervalSampler",
    "geodesic_conditional",
    "forward_flow",
    "split_flow_loss",
    "meanflow_average_velocity_target",
    "meanflow_loss",
]
