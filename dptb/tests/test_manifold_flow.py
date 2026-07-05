"""Tests for the Grassmann flow scaffolding (sampler + split-flow + MeanFlow JVP)."""
import torch

from dptb.nnops.manifold_type import GrassmannManifold
from dptb.nnops.manifold_flow import (
    OrderedIntervalSampler,
    split_flow_loss,
    forward_flow,
    meanflow_time_derivative,
    meanflow_average_velocity_target,
    meanflow_loss,
)

torch.manual_seed(0)
M = GrassmannManifold()


def _frame(n, k):
    q, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    return q[:, :k]


def _nearby(u0, k):
    u1 = _frame(u0.shape[0], k)
    if float(torch.linalg.svdvals(u0.mH @ u1).min()) < 0.3:
        u1, _ = torch.linalg.qr(u0 + 0.3 * u1)
        u1 = u1[:, :k]
    return u1


def _cframe(n, k):
    q, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.complex128))
    return q[:, :k]


def _cnearby(u0, k):
    u1 = _cframe(u0.shape[0], k)
    if float(torch.linalg.svdvals(u0.mH @ u1).min()) < 0.3:
        u1, _ = torch.linalg.qr(u0 + 0.3 * u1)
        u1 = u1[:, :k]
    return u1


# --------------------------------------------------------------------------- #
# time sampler
# --------------------------------------------------------------------------- #
def test_sampler_ordering_and_range():
    t, s = OrderedIntervalSampler(boundary_ratio=0.0).sample(2000)
    assert torch.all(t <= s + 1e-12)
    assert torch.all((t >= 0) & (s <= 1))


def test_sampler_boundary_mass():
    ratio = 0.75
    t, s = OrderedIntervalSampler(boundary_ratio=ratio).sample(20000)
    frac_collapsed = float((s == t).double().mean())
    # collapsed whenever the Bernoulli fires (plus the measure-zero a==b case)
    assert abs(frac_collapsed - ratio) < 0.03


# --------------------------------------------------------------------------- #
# split-flow (semigroup) -- exact average velocity => zero loss
# --------------------------------------------------------------------------- #
def test_split_flow_zero_for_true_average_velocity():
    x0 = _frame(8, 2)
    x1 = _nearby(x0, 2)

    def true_avg_vel(x, t, s):
        # exact average velocity from x at time t to the geodesic point at time s
        x_s, _ = M.geodesic_with_tangent(x0, x1, s)
        return M.log_map(x, x_s) / (s - t)

    t = torch.tensor(0.3, dtype=torch.float64)
    r = torch.tensor(0.7, dtype=torch.float64)
    loss = split_flow_loss(M, true_avg_vel, x0, x1, t, r)
    assert float(loss) < 1e-10


def test_forward_flow_reaches_endpoint():
    x0 = _frame(7, 2); x1 = _nearby(x0, 2)
    t = torch.tensor(0.0, dtype=torch.float64)
    s = torch.tensor(1.0, dtype=torch.float64)

    def avg_vel(x, tt, ss):
        return M.log_map(x, x1) / (ss - tt)

    reached = forward_flow(M, avg_vel, x0, t, s)
    assert float(M.chordal_distance_sq(reached, x1)) < 1e-9


# --------------------------------------------------------------------------- #
# MeanFlow average-velocity identity via jvp
# --------------------------------------------------------------------------- #
def test_meanflow_target_constant_velocity():
    # u independent of t and x => D_t u = 0 => u_tgt == v == u
    x_t = _frame(8, 3)
    V0 = M.proju(x_t, torch.randn(8, 3, dtype=torch.float64))

    def vel(x, t, s):
        return V0

    t = torch.tensor(0.4, dtype=torch.float64)
    s = torch.tensor(0.9, dtype=torch.float64)
    u, u_tgt = meanflow_average_velocity_target(vel, x_t, t, s, path_velocity=V0, instantaneous_velocity=V0)
    assert float((u - V0).abs().max()) < 1e-12
    assert float((u_tgt - V0).abs().max()) < 1e-10
    assert float(meanflow_loss(M, vel, x_t, t, s, path_velocity=V0, instantaneous_velocity=V0)) < 1e-12


def test_meanflow_target_linear_velocity():
    # u = t * V0 => D_t u = V0 => u_tgt = v - (t-s) V0 = t V0 - (t-s) V0 = s V0
    x_t = _frame(8, 2)
    V0 = M.proju(x_t, torch.randn(8, 2, dtype=torch.float64))

    def vel(x, t, s):
        return t * V0

    t = torch.tensor(0.4, dtype=torch.float64)
    s = torch.tensor(0.9, dtype=torch.float64)
    u, u_tgt = meanflow_average_velocity_target(
        vel, x_t, t, s, path_velocity=V0, instantaneous_velocity=t * V0)
    assert float((u - t * V0).abs().max()) < 1e-12
    assert float((u_tgt - s * V0).abs().max()) < 1e-9


def test_meanflow_requires_path_velocity():
    # The state term is mandatory: omitting path_velocity must fail loudly, not bias silently.
    x_t = _frame(6, 2)
    import pytest
    with pytest.raises(ValueError):
        meanflow_average_velocity_target(lambda x, t, s: x, x_t,
                                         torch.tensor(0.3, dtype=torch.float64),
                                         torch.tensor(0.8, dtype=torch.float64))


def test_meanflow_identity_zero_for_true_geodesic_average_velocity():
    # The strongest check: for the exact average velocity of a geodesic, the MeanFlow
    # identity u = v - (t-s) D_t u holds, so the residual is ~0 (real AND complex).
    for complex_ in (False, True):
        x0 = _cframe(8, 2) if complex_ else _frame(8, 2)
        x1 = _cnearby(x0, 2) if complex_ else _nearby(x0, 2)
        t = torch.tensor(0.35, dtype=torch.float64)
        s = torch.tensor(0.80, dtype=torch.float64)
        x_t, dx_t = M.geodesic_with_tangent(x0, x1, t)      # point + trajectory velocity

        def true_avg(x, tt, ss):
            x_s = M.geodesic_interpolant(x0, x1, ss)
            return M.log_map(x, x_s) / (ss - tt)

        loss = meanflow_loss(M, true_avg, x_t, t, s, path_velocity=dx_t, instantaneous_velocity=dx_t)
        assert float(loss) < 1e-18, (complex_, float(loss))


def test_meanflow_total_derivative_includes_state_jvp():
    # For an x-dependent field the total derivative must add (du/dx).(dx/dt); the
    # partial-only mode misses exactly x_dot @ A.
    x_t = _frame(7, 2)
    x_dot = M.proju(x_t, torch.randn(7, 2, dtype=torch.float64))
    A = torch.randn(2, 2, dtype=torch.float64)
    B = M.proju(x_t, torch.randn(7, 2, dtype=torch.float64))

    def vel(x, t, s):
        return x @ A + t * B

    t = torch.tensor(0.25, dtype=torch.float64)
    s = torch.tensor(0.75, dtype=torch.float64)
    _, partial = meanflow_time_derivative(vel, x_t, t, s)
    _, total = meanflow_time_derivative(vel, x_t, t, s, path_velocity=x_dot)
    assert float((total - partial - x_dot @ A).abs().max()) < 1e-10
    assert float((total - partial).norm()) > 1e-8


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception as e:  # noqa
            fail += 1; print("FAIL", fn.__name__, ":", type(e).__name__, e)
    print(f"\n{len(fns)-fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)
