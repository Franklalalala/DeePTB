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


def forward_flow(manifold: Manifold, avg_vel: AvgVel, x, t: Tensor, s: Tensor):
    """Step from ``x`` at time ``t`` to time ``s`` with the average velocity: ``exp(x, (s-t) v)``.

    Routed through ``manifold.scale_tangent`` / ``exp_map`` so a ``ProductManifold`` (whose
    point/tangent are tuples) dispatches component-wise; identical to
    ``scale_like(s-t, v) * v`` for a single-tensor manifold.
    """
    v = manifold.proju(x, avg_vel(x, t, s))
    return manifold.exp_map(x, manifold.scale_tangent(v, s - t))


def split_flow_loss(manifold: Manifold, avg_vel: AvgVel, x0, x1, t: Tensor, r: Tensor) -> Tensor:
    """Integration-free semigroup consistency (split-flow), any manifold.

    Stepping ``x_t -> r`` in one shot with the learned average velocity must land on the
    geodesic point ``x_r``.  Uses only ``exp`` + a squared distance -- no forward-mode
    autodiff, no eigh-in-jvp.  Recommended as the first flow milestone.  The distance
    keeps its batch axis, so the per-sample residuals are averaged; on a product manifold
    it is the sum of the Euclidean (delta-H) and chordal (P) squared distances.
    """
    x_t, _ = manifold.geodesic_with_tangent(x0, x1, t)
    x_r, _ = manifold.geodesic_with_tangent(x0, x1, r)
    stepped = forward_flow(manifold, avg_vel, x_t, t, r)
    return manifold.distance_sq(stepped, manifold.detach_point(x_r)).mean()


class ThreePointSampler:
    """Sample an ordered triple ``t <= s <= r`` in ``[0, 1]`` for a *true* semigroup check.

    Unlike :class:`OrderedIntervalSampler` (two times), this draws three and sorts them, so
    the middle time ``s`` is a genuine interior split of ``[t, r]``.  With probability
    ``boundary_ratio`` the split collapses (``s = t``), degenerating to the two-point
    consistency of :func:`split_flow_loss`.
    """

    def __init__(self, boundary_ratio: float = 0.0):
        self.boundary_ratio = float(boundary_ratio)

    def sample(self, n: int, *, device=None, dtype=torch.float64,
               generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor]:
        u = torch.rand(n, 3, device=device, dtype=dtype, generator=generator)
        srt, _ = torch.sort(u, dim=-1)
        t, s, r = srt[:, 0], srt[:, 1], srt[:, 2]
        if self.boundary_ratio > 0.0:
            collapse = torch.rand(n, device=device, dtype=dtype, generator=generator) < self.boundary_ratio
            s = torch.where(collapse, t, s)
        return t, s, r


def semigroup_consistency_loss(manifold: Manifold, avg_vel: AvgVel, x0, x1,
                               t: Tensor, s: Tensor, r: Tensor) -> Tensor:
    """True multi-step semigroup residual: ``step(t->r) == step(s->r) . step(t->s)``.

    The average-velocity flow must be *path-composable*: one big jump ``t->r`` has to equal
    stepping ``t->s`` then ``s->r``.  This is a stronger, integration-free constraint than
    :func:`split_flow_loss` (which only checks a one-shot jump against the geodesic point)
    and does not reference the geodesic target of the middle time at all -- it is a pure
    self-consistency of the learned field.  For the exact geodesic average velocity both
    routes land on ``x_r`` and the residual is ~0.
    """
    x_t, _ = manifold.geodesic_with_tangent(x0, x1, t)
    direct = forward_flow(manifold, avg_vel, x_t, t, r)
    x_s = forward_flow(manifold, avg_vel, x_t, t, s)
    composed = forward_flow(manifold, avg_vel, x_s, s, r)
    return manifold.distance_sq(direct, manifold.detach_point(composed)).mean()


def _leaf_tensor(x):
    """First tensor leaf of a point (tensor, or tuple/list of them) -- for dtype/device."""
    if isinstance(x, (tuple, list)):
        for xi in x:
            t = _leaf_tensor(xi)
            if t is not None:
                return t
        return None
    return x if torch.is_tensor(x) else None


def euler_sample(manifold: Manifold, avg_vel: AvgVel, x0, num_steps: int,
                 *, t0: float = 0.0, t1: float = 1.0):
    """Integrate the average-velocity field from ``x0`` (at ``t0``) to ``t1`` in Euler steps.

    Each sub-step advances with the average velocity over its own sub-interval
    (``x <- forward_flow(x, t_i, t_{i+1})``).  ``num_steps=1`` is the one-shot endpoint jump
    used for endpoint-aligned validation (the euler-1 sample of 0703-Flow's CFM).  Works on
    any manifold, including the product tuple.
    """
    ref = _leaf_tensor(x0)
    ts = torch.linspace(float(t0), float(t1), int(num_steps) + 1,
                        dtype=ref.real.dtype, device=ref.device)
    x = x0
    for i in range(int(num_steps)):
        x = forward_flow(manifold, avg_vel, x, ts[i], ts[i + 1])
    return x


def validation_endpoint_loss(manifold: Manifold, avg_vel: AvgVel, x0, x1,
                             *, num_steps: int = 1) -> Tensor:
    """Endpoint-aligned validation score -- the euler-1 fix of commits 58eb6b6 / a2b7b7b.

    A MeanFlow head trains on a *random-t* average-velocity regression, so its train loss is
    not on the same scale as a data-space endpoint error and train/val curves are otherwise
    incomparable.  For a comparable validation number, euler-integrate the learned field from
    ``x0`` to the ``t=1`` endpoint (``num_steps`` steps; 1 = one-shot jump) and score it against
    the data endpoint ``x1`` with the **same** distance the training loss uses.  For the exact
    average velocity the field is path-consistent, so this is ~0 for any ``num_steps``.  When the
    Grassmann-P flow is wired into ``flow_cfm``, this is what the CFM validation branch
    (``trainer.py:657``) must call, mapped to the legacy ``validation_*`` keys.
    """
    reached = euler_sample(manifold, avg_vel, x0, num_steps, t0=0.0, t1=1.0)
    return manifold.distance_sq(reached, manifold.detach_point(x1)).mean()


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
    manifold: Optional[Manifold] = None,
) -> Tuple[Tensor, Tensor]:
    """Full MeanFlow target ``u_tgt = v - (t - s) * D_t u`` (stop-grad label).

    ``D_t u`` is the **total** time derivative along the trajectory, so ``path_velocity``
    (``dx_t/dt``) is required -- omitting the state term (as an earlier version did) is
    only correct for velocity fields with no ``x``-dependence and silently biases the
    target otherwise.  ``instantaneous_velocity`` is the ground-truth ``v`` in the
    identity; on a geodesic conditional path it equals ``dx_t/dt`` and so defaults to
    ``path_velocity``.  The label is detached, so no EMA teacher network is needed.

    ``manifold`` (optional) routes the ``v - (t-s) D_t u`` combination through tuple-aware
    tangent algebra so a ``ProductManifold`` works; for a single-tensor manifold (or
    ``None``) it is byte-identical to the plain ``(v - scale_like(t-s, u) * dudt).detach()``.
    """
    if path_velocity is None:
        raise ValueError(
            "MeanFlow needs the total derivative along the trajectory: pass "
            "path_velocity=dx_t/dt (use meanflow_time_derivative for the partial-only diagnostic)")
    u, dudt = meanflow_time_derivative(vel_fn, x_t, t, s, path_velocity=path_velocity)
    v = path_velocity if instantaneous_velocity is None else instantaneous_velocity
    if manifold is None:
        return u, (v - scale_like(t - s, u) * dudt).detach()
    u_tgt = manifold.detach_tangent(
        manifold.tangent_sub(v, manifold.scale_tangent(dudt, t - s)))
    return u, u_tgt


def meanflow_loss(manifold: Manifold, vel_fn: AvgVel, x_t,
                  t: Tensor, s: Tensor, *,
                  path_velocity=None,
                  instantaneous_velocity=None) -> Tensor:
    """Squared tangent-metric residual between the predicted and target average velocity.

    Works on any :class:`Manifold`; on a :class:`ProductManifold` the residual is measured
    in the product metric (sum of the component tangent norms).
    """
    u, u_tgt = meanflow_average_velocity_target(
        vel_fn, x_t, t, s,
        path_velocity=path_velocity,
        instantaneous_velocity=instantaneous_velocity,
        manifold=manifold,
    )
    diff = manifold.proju(x_t, manifold.tangent_sub(u, u_tgt))
    return manifold.square_norm_at(x_t, diff).mean()


__all__ = [
    "OrderedIntervalSampler",
    "ThreePointSampler",
    "geodesic_conditional",
    "forward_flow",
    "split_flow_loss",
    "semigroup_consistency_loss",
    "euler_sample",
    "validation_endpoint_loss",
    "meanflow_time_derivative",
    "meanflow_average_velocity_target",
    "meanflow_loss",
]
