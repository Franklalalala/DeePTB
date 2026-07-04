"""Tests for the GrassmannManifold interface (exp/log/proju/transport/jvp)."""
import torch

from dptb.nnops.manifold_type import GrassmannManifold, GrassmannTangentWrapper

torch.manual_seed(0)
M = GrassmannManifold()


def _frame(n, k, complex_=False):
    a = torch.randn(n, k, dtype=torch.complex128 if complex_ else torch.float64)
    q, _ = torch.linalg.qr(a)
    return q[:, :k]


def _nearby(u0, k, scale=0.3):
    """A frame within principal angle < pi/2 of u0 so log is well-defined."""
    u1 = _frame(u0.shape[0], k, complex_=u0.is_complex())
    m = u0.mH @ u1
    if float(torch.linalg.svdvals(m).min()) < 0.2:
        u1, _ = torch.linalg.qr(u0 + scale * u1)
        u1 = u1[:, :k]
    return u1


def test_projx_returns_orthonormal_frame():
    a = torch.randn(7, 3, dtype=torch.float64)
    u = M.projx(a)
    assert torch.allclose(u.mH @ u, torch.eye(3, dtype=torch.float64), atol=1e-10)


def test_proju_is_horizontal():
    u = _frame(8, 3)
    v = torch.randn(8, 3, dtype=torch.float64)
    h = M.proju(u, v)
    assert float((u.mH @ h).abs().max()) < 1e-10          # Uᴴ h = 0


def test_exp_log_roundtrip():
    u0 = _frame(9, 3)
    u1 = _nearby(u0, 3)
    delta = M.log_map(u0, u1)
    assert float((u0.mH @ delta).abs().max()) < 1e-8       # horizontal
    u1_rec = M.exp_map(u0, delta)
    assert float(M.chordal_distance_sq(u1_rec, u1)) < 1e-9


def test_geodesic_endpoints():
    u0 = _frame(8, 2); u1 = _nearby(u0, 2)
    g0 = M.geodesic_interpolant(u0, u1, torch.tensor(0.0, dtype=torch.float64))
    g1 = M.geodesic_interpolant(u0, u1, torch.tensor(1.0, dtype=torch.float64))
    assert float(M.chordal_distance_sq(g0, u0)) < 1e-9
    assert float(M.chordal_distance_sq(g1, u1)) < 1e-8


def test_geodesic_with_tangent_matches_finite_difference():
    u0 = _frame(8, 2); u1 = _nearby(u0, 2)
    t = torch.tensor(0.4, dtype=torch.float64)
    x_t, dx_t = M.geodesic_with_tangent(u0, u1, t)
    h = 1e-6
    x_p = M.geodesic_interpolant(u0, u1, t + h)
    x_m = M.geodesic_interpolant(u0, u1, t - h)
    fd = (x_p - x_m) / (2 * h)
    # QR gauge can differ; compare on the projector (gauge-invariant) time-derivative.
    P = lambda u: u @ u.mH
    dP_ad = dx_t @ x_t.mH + x_t @ dx_t.mH
    dP_fd = (P(x_p) - P(x_m)) / (2 * h)
    assert float((dP_ad - dP_fd).abs().max()) < 1e-5


def test_parallel_transport_is_isometry():
    u0 = _frame(9, 3); u1 = _nearby(u0, 3)
    v = M.proju(u0, torch.randn(9, 3, dtype=torch.float64))     # horizontal tangent at u0
    tv = M.parallel_transport(u0, u1, v)
    assert float((u1.mH @ tv).abs().max()) < 1e-7               # still horizontal at u1
    n0 = float(M.square_norm_at(u0, v))
    n1 = float(M.square_norm_at(u1, tv))
    assert abs(n0 - n1) / max(n0, 1e-12) < 1e-6                 # norm preserved


def test_tangent_wrapper_outputs_horizontal():
    class RawHead(torch.nn.Module):
        def forward(self, u):
            return torch.randn_like(u)
    u = _frame(8, 2)
    wrap = GrassmannTangentWrapper(RawHead(), M, max_angle=1.0)
    d = wrap(u)
    assert float((u.mH @ d).abs().max()) < 1e-10               # horizontal
    assert float(torch.linalg.svdvals(d).amax()) <= 1.0 + 1e-9  # angle cap


def test_complex_exp_log_roundtrip():
    u0 = _frame(7, 2, complex_=True)
    u1 = _nearby(u0, 2)
    delta = M.log_map(u0, u1)
    u1_rec = M.exp_map(u0, delta)
    assert float(M.chordal_distance_sq(u1_rec, u1)) < 1e-7


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
