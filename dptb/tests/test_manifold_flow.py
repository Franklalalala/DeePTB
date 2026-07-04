"""Tests for the Grassmann flow scaffolding (sampler + split-flow + MeanFlow JVP)."""
import torch

from dptb.nnops.manifold_type import GrassmannManifold
from dptb.nnops.manifold_flow import (
    OrderedIntervalSampler,
    split_flow_loss,
    forward_flow,
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
    # u independent of t => du/dt = 0 => u_tgt == u
    x_t = _frame(8, 3)
    V0 = M.proju(x_t, torch.randn(8, 3, dtype=torch.float64))

    def vel(x, t, s):
        return V0

    t = torch.tensor(0.4, dtype=torch.float64)
    s = torch.tensor(0.9, dtype=torch.float64)
    u, u_tgt = meanflow_average_velocity_target(vel, x_t, t, s)
    assert float((u - V0).abs().max()) < 1e-12
    assert float((u_tgt - V0).abs().max()) < 1e-10
    assert float(meanflow_loss(M, vel, x_t, t, s)) < 1e-12


def test_meanflow_target_linear_velocity():
    # u = t * V0 => du/dt = V0 => u_tgt = t V0 - (t-s) V0 = s V0
    x_t = _frame(8, 2)
    V0 = M.proju(x_t, torch.randn(8, 2, dtype=torch.float64))

    def vel(x, t, s):
        return t * V0

    t = torch.tensor(0.4, dtype=torch.float64)
    s = torch.tensor(0.9, dtype=torch.float64)
    u, u_tgt = meanflow_average_velocity_target(vel, x_t, t, s)
    assert float((u - t * V0).abs().max()) < 1e-12
    assert float((u_tgt - s * V0).abs().max()) < 1e-9


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
