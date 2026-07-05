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

from dptb.nnops._manifold_math import Tensor, scale_like
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
    return manifold.exp_map(x, scale_like(s - t, v) * v)


def split_flow_loss(manifold: GrassmannManifold, avg_vel: AvgVel,
                    x0: Tensor, x1: Tensor, t: Tensor, r: Tensor) -> Tensor:
    """Integration-free semigroup consistency (Grassmann split-flow).

    Stepping ``x_t -> r`` in one shot with the learned average velocity must land on the
    geodesic point ``x_r``.  Uses only ``exp`` + a (squared chordal) distance -- no
    forward-mode autodiff, no eigh-in-jvp.  Recommended as the first flow milestone.
    The chordal distance keeps its batch axis, so the per-sample residuals are averaged.
    """
    x_t, _ = manifold.geodesic_with_tangent(x0, x1, t)
    x_r, _ = manifold.geodesic_with_tangent(x0, x1, r)
    stepped = forward_flow(manifold, avg_vel, x_t, t, r)
    return manifold.chordal_distance_sq(stepped, x_r.detach()).mean()


def meanflow_time_derivative(
    vel_fn: AvgVel,
    x_t: Tensor,
    t: Tensor,
    s: Tensor,
    *,
    path_velocity: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return ``(u, du/dt)`` for the average-velocity field ``vel_fn(x, t, s)``.

    With ``path_velocity`` (= ``dx_t/dt`` along the training trajectory) this is the
    **total** time derivative required by MeanFlow, computed as one forward-mode JVP of
    ``vel_fn`` in the direction ``(dx_t/dt, 1, 0)``: it adds the state term
    ``(du/dx) . (dx_t/dt)`` that the point moving along the geodesic contributes.  Without
    it, only the partial ``du/dt`` (``x`` held fixed) is returned -- a diagnostic, not a
    complete MeanFlow target.
    """
    if path_velocity is None:
        def u_of_t(tt: Tensor) -> Tensor:
            return vel_fn(x_t, tt, s)

        return torch.func.jvp(u_of_t, (t,), (torch.ones_like(t),))

    def u_of_xt(xt: Tensor, tt: Tensor) -> Tensor:
        return vel_fn(xt, tt, s)

    return torch.func.jvp(u_of_xt, (x_t, t), (path_velocity, torch.ones_like(t)))


def meanflow_average_velocity_target(
    vel_fn: AvgVel,
    x_t: Tensor,
    t: Tensor,
    s: Tensor,
    *,
    path_velocity: Optional[Tensor] = None,
    instantaneous_velocity: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Full MeanFlow target ``u_tgt = v - (t - s) * D_t u`` (stop-grad label).

    ``D_t u`` is the **total** time derivative along the trajectory, so ``path_velocity``
    (``dx_t/dt``) is required -- omitting the state term (as an earlier version did) is
    only correct for velocity fields with no ``x``-dependence and silently biases the
    target otherwise.  ``instantaneous_velocity`` is the ground-truth ``v`` in the
    identity; on a geodesic conditional path it equals ``dx_t/dt`` and so defaults to
    ``path_velocity``.  The label is detached, so no EMA teacher network is needed.
    """
    if path_velocity is None:
        raise ValueError(
            "MeanFlow needs the total derivative along the trajectory: pass "
            "path_velocity=dx_t/dt (use meanflow_time_derivative for the partial-only diagnostic)")
    u, dudt = meanflow_time_derivative(vel_fn, x_t, t, s, path_velocity=path_velocity)
    v = path_velocity if instantaneous_velocity is None else instantaneous_velocity
    return u, (v - scale_like(t - s, u) * dudt).detach()


def meanflow_loss(manifold: GrassmannManifold, vel_fn: AvgVel, x_t: Tensor,
                  t: Tensor, s: Tensor, *,
                  path_velocity: Optional[Tensor] = None,
                  instantaneous_velocity: Optional[Tensor] = None) -> Tensor:
    """Squared tangent-metric residual between the predicted and target average velocity."""
    u, u_tgt = meanflow_average_velocity_target(
        vel_fn, x_t, t, s,
        path_velocity=path_velocity,
        instantaneous_velocity=instantaneous_velocity,
    )
    diff = manifold.proju(x_t, u - u_tgt)
    return manifold.square_norm_at(x_t, diff).mean()


__all__ = [
    "OrderedIntervalSampler",
    "geodesic_conditional",
    "forward_flow",
    "split_flow_loss",
    "meanflow_time_derivative",
    "meanflow_average_velocity_target",
    "meanflow_loss",
]
