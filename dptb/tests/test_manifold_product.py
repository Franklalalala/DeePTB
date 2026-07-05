"""Tests for EuclideanManifold, ProductManifold, and the manifold-generic flow losses.

The product manifold is the geometry the M-b Riemannian-MeanFlow rides on: the Euclidean
delta-H factor (+) the Grassmann occupied-projector P factor.  These checks validate the
geometry (component-wise exp/log/metric/transport, additive distance) and, crucially, that
the *same* split-flow / semigroup / MeanFlow losses that work on a single Grassmann factor
also work on the product tuple -- i.e. ``torch.func.jvp`` propagates through the tuple
pytree so the average-velocity identity closes to ~0 on the product.
"""
import math

import torch

from dptb.nnops.manifold_type import (
    GrassmannManifold,
    EuclideanManifold,
    ProductManifold,
)
from dptb.nnops.manifold_flow import (
    ThreePointSampler,
    forward_flow,
    split_flow_loss,
    semigroup_consistency_loss,
    meanflow_loss,
)

torch.manual_seed(0)
MG = GrassmannManifold()


def _frame(n, k, complex_=False):
    a = torch.randn(n, k, dtype=torch.complex128 if complex_ else torch.float64)
    q, _ = torch.linalg.qr(a)
    return q[:, :k]


def _nearby(u0, k, scale=0.3):
    u1 = _frame(u0.shape[0], k, complex_=u0.is_complex())
    if float(torch.linalg.svdvals(u0.mH @ u1).min()) < 0.2:
        u1, _ = torch.linalg.qr(u0 + scale * u1)
        u1 = u1[:, :k]
    return u1


# --------------------------------------------------------------------------- #
# EuclideanManifold
# --------------------------------------------------------------------------- #
def test_euclidean_exp_log_and_geodesic():
    E = EuclideanManifold(event_ndim=1)
    x0 = torch.randn(5, dtype=torch.float64)
    x1 = torch.randn(5, dtype=torch.float64)
    # log then exp round-trips
    assert float((E.exp_map(x0, E.log_map(x0, x1)) - x1).abs().max()) < 1e-12
    # straight-line geodesic + constant velocity, checked at t=0.4 vs finite difference
    t = torch.tensor(0.4, dtype=torch.float64)
    x_t, dx_t = E.geodesic_with_tangent(x0, x1, t)
    assert float((x_t - (x0 + 0.4 * (x1 - x0))).abs().max()) < 1e-12
    assert float((dx_t - (x1 - x0)).abs().max()) < 1e-12
    # distance_sq / metric reduce only the event axis => batch preserved
    E2 = EuclideanManifold(event_ndim=2)
    a = torch.randn(3, 4, 4, dtype=torch.float64)
    b = torch.randn(3, 4, 4, dtype=torch.float64)
    d = E2.distance_sq(a, b)
    assert d.shape == (3,)
    assert torch.allclose(d, ((a - b) ** 2).sum(dim=(-2, -1)))
    assert E2.distance_sq(a[0], b[0]).ndim == 0     # single point -> 0-d scalar


def test_euclidean_complex_metric_is_real():
    E = EuclideanManifold(event_ndim=1)
    x = torch.zeros(4, dtype=torch.complex128)
    v = torch.randn(4, dtype=torch.complex128)
    m = E.metric(x, v, v)
    assert m.dtype == torch.float64
    assert abs(float(m) - float((v.conj() * v).real.sum())) < 1e-12


# --------------------------------------------------------------------------- #
# ProductManifold geometry
# --------------------------------------------------------------------------- #
def _product():
    return ProductManifold([EuclideanManifold(event_ndim=1), GrassmannManifold()])


def _prod_point():
    return (torch.randn(5, dtype=torch.float64), _frame(8, 2))


def test_product_requires_tuple():
    P = _product()
    import pytest
    with pytest.raises(ValueError):
        P.exp_map(torch.zeros(5), torch.zeros(5))   # not a tuple -> zip over a bare tensor


def test_product_geodesic_endpoints_and_distance():
    P = _product()
    xe0 = torch.randn(5, dtype=torch.float64); xg0 = _frame(8, 2)
    xe1 = torch.randn(5, dtype=torch.float64); xg1 = _nearby(xg0, 2)
    x0, x1 = (xe0, xg0), (xe1, xg1)
    g0, _ = P.geodesic_with_tangent(x0, x1, torch.tensor(0.0, dtype=torch.float64))
    g1, _ = P.geodesic_with_tangent(x0, x1, torch.tensor(1.0, dtype=torch.float64))
    assert float(P.distance_sq(g0, x0)) < 1e-16
    assert float(P.distance_sq(g1, x1)) < 1e-14
    # product distance_sq is the SUM of the Euclidean L2 and the Grassmann chordal terms
    euc = ((xe0 - xe1) ** 2).sum()
    cho = MG.chordal_distance_sq(xg0, xg1)
    assert abs(float(P.distance_sq(x0, x1)) - float(euc + cho)) < 1e-10


def test_product_metric_is_additive():
    P = _product()
    x = _prod_point()
    ve = torch.randn(5, dtype=torch.float64)
    vg = MG.proju(x[1], torch.randn(8, 2, dtype=torch.float64))
    v = (ve, vg)
    total = float(P.square_norm_at(x, v))
    parts = float((ve * ve).sum()) + float(MG.square_norm_at(x[1], vg))
    assert abs(total - parts) < 1e-10


def test_product_parallel_transport_isometry():
    P = _product()
    xe0 = torch.randn(5, dtype=torch.float64); xg0 = _frame(9, 3)
    xe1 = torch.randn(5, dtype=torch.float64); xg1 = _nearby(xg0, 3)
    x, y = (xe0, xg0), (xe1, xg1)
    v = (torch.randn(5, dtype=torch.float64), MG.proju(xg0, torch.randn(9, 3, dtype=torch.float64)))
    tv = P.parallel_transport(x, y, v)
    n0 = float(P.square_norm_at(x, v))
    n1 = float(P.square_norm_at(y, tv))
    assert abs(n0 - n1) / max(n0, 1e-12) < 1e-6           # product norm preserved
    assert float((y[1].mH @ tv[1]).abs().max()) < 1e-7    # P-component stays horizontal


# --------------------------------------------------------------------------- #
# Three-point sampler
# --------------------------------------------------------------------------- #
def test_three_point_sampler_ordering():
    t, s, r = ThreePointSampler(boundary_ratio=0.0).sample(5000)
    assert torch.all(t <= s + 1e-12) and torch.all(s <= r + 1e-12)
    assert torch.all((t >= 0) & (r <= 1))


def test_three_point_sampler_boundary_mass():
    ratio = 0.5
    t, s, r = ThreePointSampler(boundary_ratio=ratio).sample(20000)
    frac = float((s == t).double().mean())
    assert abs(frac - ratio) < 0.03


# --------------------------------------------------------------------------- #
# Flow losses on the product manifold (the real L3 validation)
# --------------------------------------------------------------------------- #
def _true_product_avg(P, x0, x1):
    E, G = P.manifolds
    xe0, xg0 = x0
    xe1, xg1 = x1

    def true_avg(x, t, s):
        xe, xg = x
        e_s = xe0 + float(s) * (xe1 - xe0)                 # Euclidean geodesic point at s
        g_s = G.geodesic_interpolant(xg0, xg1, s)          # Grassmann geodesic point at s
        ve = (e_s - xe) / (s - t)
        vg = G.log_map(xg, g_s) / (s - t)
        return (ve, vg)

    return true_avg


def test_product_split_flow_zero_for_true_average_velocity():
    P = _product()
    x0 = (torch.randn(5, dtype=torch.float64), _frame(8, 2))
    x1 = (torch.randn(5, dtype=torch.float64), _nearby(x0[1], 2))
    avg = _true_product_avg(P, x0, x1)
    t = torch.tensor(0.3, dtype=torch.float64)
    r = torch.tensor(0.7, dtype=torch.float64)
    loss = split_flow_loss(P, avg, x0, x1, t, r)
    assert float(loss) < 1e-14, float(loss)


def test_product_semigroup_zero_for_true_average_velocity():
    P = _product()
    x0 = (torch.randn(5, dtype=torch.float64), _frame(8, 2))
    x1 = (torch.randn(5, dtype=torch.float64), _nearby(x0[1], 2))
    avg = _true_product_avg(P, x0, x1)
    t = torch.tensor(0.2, dtype=torch.float64)
    s = torch.tensor(0.55, dtype=torch.float64)
    r = torch.tensor(0.9, dtype=torch.float64)
    loss = semigroup_consistency_loss(P, avg, x0, x1, t, s, r)
    assert float(loss) < 1e-14, float(loss)


def test_grassmann_semigroup_zero_single_manifold():
    # the semigroup loss must also work on a single Grassmann factor
    x0 = _frame(8, 2); x1 = _nearby(x0, 2)

    def avg(x, t, s):
        x_s = MG.geodesic_interpolant(x0, x1, s)
        return MG.log_map(x, x_s) / (s - t)

    t = torch.tensor(0.2, dtype=torch.float64)
    s = torch.tensor(0.5, dtype=torch.float64)
    r = torch.tensor(0.9, dtype=torch.float64)
    assert float(semigroup_consistency_loss(MG, avg, x0, x1, t, s, r)) < 1e-12


def test_product_meanflow_identity_zero_for_true_geodesic_average_velocity():
    # The strongest check: MeanFlow identity u = v - (t-s) D_t u on the PRODUCT manifold.
    # This exercises torch.func.jvp through the (Euclidean, Grassmann) tuple pytree.
    P = _product()
    x0 = (torch.randn(5, dtype=torch.float64), _frame(8, 2))
    x1 = (torch.randn(5, dtype=torch.float64), _nearby(x0[1], 2))
    t = torch.tensor(0.35, dtype=torch.float64)
    s = torch.tensor(0.80, dtype=torch.float64)
    x_t, dx_t = P.geodesic_with_tangent(x0, x1, t)         # tuple point + tuple trajectory velocity
    avg = _true_product_avg(P, x0, x1)
    loss = meanflow_loss(P, avg, x_t, t, s, path_velocity=dx_t, instantaneous_velocity=dx_t)
    assert float(loss) < 1e-16, float(loss)


def test_product_forward_flow_reaches_endpoint():
    P = _product()
    x0 = (torch.randn(4, dtype=torch.float64), _frame(7, 2))
    x1 = (torch.randn(4, dtype=torch.float64), _nearby(x0[1], 2))
    avg = _true_product_avg(P, x0, x1)
    t = torch.tensor(0.0, dtype=torch.float64)
    s = torch.tensor(1.0, dtype=torch.float64)
    reached = forward_flow(P, avg, x0, t, s)
    assert float(P.distance_sq(reached, x1)) < 1e-14


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
