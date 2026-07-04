"""Dense-core unit tests for the Grassmann P-regression toolkit.

Run standalone (no full DeePTB runtime needed for the math core):
    pytest tests/test_grassmann_core.py -q
"""
import math

import torch

# pure math lives in the framework-free core; the loss layer lives in grassmann.py
from dptb.nnops._manifold_math import (
    occupied_projector,
    chordal_distance_sq,
    principal_angles,
    geodesic_distance,
    grassmann_exp,
    grassmann_log,
    mcweeny_purify,
    idempotency_error,
    s_half_and_inv,
)
from dptb.nnops.grassmann import (
    grassmann_p_loss_single,
    dense_grassmann_p_loss,
)

torch.manual_seed(0)


def _rand_herm(n, complex_=False):
    if complex_:
        a = torch.randn(n, n, dtype=torch.complex128)
    else:
        a = torch.randn(n, n, dtype=torch.float64)
    return 0.5 * (a + a.mH)


def _rand_spd(n):
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.mT + n * torch.eye(n, dtype=torch.float64)


def _rand_frame(n, k, complex_=False):
    a = torch.randn(n, k, dtype=torch.complex128 if complex_ else torch.float64)
    q, _ = torch.linalg.qr(a)
    return q[:, :k]


# --------------------------------------------------------------------------- #
# occupied projector
# --------------------------------------------------------------------------- #
def test_occupied_projector_is_idempotent_rank_k():
    h = _rand_herm(8)
    p, u, eps = occupied_projector(h, None, n_occ=3, return_frame=True)
    assert torch.allclose(p, p @ p, atol=1e-9)          # idempotent
    assert torch.allclose(p, p.mH, atol=1e-12)          # hermitian
    assert abs(float(torch.trace(p).real) - 3.0) < 1e-9  # rank = n_occ
    # eigenvalues ascending, occupied = lowest 3
    assert torch.all(eps[:-1] <= eps[1:] + 1e-9)


def test_occupied_projector_smetric_orthonormal():
    n, k = 7, 2
    h = _rand_herm(n)
    s = _rand_spd(n)
    p = occupied_projector(h, s, n_occ=k)
    assert torch.allclose(p, p @ p, atol=1e-8)
    assert abs(float(torch.trace(p).real) - k) < 1e-8


def test_occupied_projector_complex():
    h = _rand_herm(6, complex_=True)
    p = occupied_projector(h, None, n_occ=2)
    assert torch.allclose(p, p @ p, atol=1e-9)
    assert torch.allclose(p, p.mH, atol=1e-12)


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def test_chordal_zero_for_same_subspace():
    u = _rand_frame(6, 2)
    p = u @ u.mH
    assert float(chordal_distance_sq(p, p)) < 1e-20


def test_chordal_equals_sum_sin2():
    u1 = _rand_frame(8, 3)
    u2 = _rand_frame(8, 3)
    p1, p2 = u1 @ u1.mH, u2 @ u2.mH
    theta = principal_angles(u1, u2)
    lhs = float(chordal_distance_sq(p1, p2))
    rhs = float((torch.sin(theta) ** 2).sum())
    assert abs(lhs - rhs) < 1e-6


def test_geodesic_zero_for_same():
    # geodesic uses arccos(singular values), which has an O(sqrt(clamp_eps)) floor
    # at zero distance (the very reason chordal, not geodesic, is the loss).
    u = _rand_frame(5, 2)
    assert float(geodesic_distance(u, u)) < 2e-3


# --------------------------------------------------------------------------- #
# exp / log round trip
# --------------------------------------------------------------------------- #
def test_exp_log_roundtrip():
    n, k = 9, 3
    u0 = _rand_frame(n, k)
    u1 = _rand_frame(n, k)
    # ensure principal angles < pi/2 so log is well-defined: nudge u1 toward u0
    u1 = _rand_frame(n, k)
    m = u0.mH @ u1
    if float(torch.linalg.svdvals(m).min()) < 0.2:
        u1, _ = torch.linalg.qr(u0 + 0.3 * u1)
        u1 = u1[:, :k]
    delta = grassmann_log(u0, u1)
    # horizontal: u0^H delta = 0
    assert float((u0.mH @ delta).abs().max()) < 1e-8
    u1_rec = grassmann_exp(u0, delta)
    d = float(chordal_distance_sq(u1_rec @ u1_rec.mH, u1 @ u1.mH))
    assert d < 1e-8, d


def test_exp_from_zero_tangent_is_identity_subspace():
    u0 = _rand_frame(6, 2)
    delta = torch.zeros_like(u0)
    u1 = grassmann_exp(u0, delta)
    assert float(chordal_distance_sq(u1 @ u1.mH, u0 @ u0.mH)) < 1e-8


# --------------------------------------------------------------------------- #
# McWeeny purification
# --------------------------------------------------------------------------- #
def test_mcweeny_purifies_noisy_projector():
    n, k = 8, 3
    u = _rand_frame(n, k)
    p_clean = u @ u.mH
    noise = 0.02 * _rand_herm(n)
    p_noisy = p_clean + noise
    assert float(idempotency_error(p_noisy)) > 1e-3
    p_pure = mcweeny_purify(p_noisy, n_iter=40)
    assert float(idempotency_error(p_pure)) < 1e-6
    assert abs(float(torch.trace(p_pure).real) - k) < 1e-2


# --------------------------------------------------------------------------- #
# P-regression loss
# --------------------------------------------------------------------------- #
def test_p_loss_zero_for_identical():
    h = _rand_herm(8)
    res = grassmann_p_loss_single(h, h.clone(), None, n_occ=3, lambda_chordal=1.0, lambda_eps=1.0)
    assert float(res.loss) < 1e-16


def test_p_loss_positive_for_perturbed():
    h_ref = _rand_herm(8)
    h_pred = h_ref + 0.1 * _rand_herm(8)
    res = grassmann_p_loss_single(h_pred, h_ref, None, n_occ=3, lambda_chordal=1.0, lambda_eps=0.0)
    assert float(res.loss) > 0.0


def test_p_loss_gradient_flows_to_h_pred():
    h_ref = _rand_herm(8)
    h_pred = (h_ref + 0.1 * _rand_herm(8)).clone().requires_grad_(True)
    res = grassmann_p_loss_single(h_pred, h_ref, None, n_occ=3, lambda_chordal=1.0, lambda_eps=0.5)
    res.loss.backward()
    assert h_pred.grad is not None
    assert torch.isfinite(h_pred.grad).all()
    assert float(h_pred.grad.abs().max()) > 0.0


def test_p_loss_smetric_gradient():
    n, k = 7, 2
    h_ref = _rand_herm(n)
    s = _rand_spd(n)
    h_pred = (h_ref + 0.05 * _rand_herm(n)).clone().requires_grad_(True)
    res = grassmann_p_loss_single(h_pred, h_ref, s, n_occ=k, lambda_chordal=1.0, lambda_eps=0.1)
    res.loss.backward()
    assert torch.isfinite(h_pred.grad).all()
    assert float(res.gap) != 0.0


def test_from_density_picks_top_occupation():
    # A density D = 2 P has occupations {2 (x k), 0 (x n-k)}; from_density must
    # recover exactly the projector's subspace from the top-k occupations.
    u = _rand_frame(8, 3)
    p = u @ u.mH
    d = 2.0 * p
    p_from_d = occupied_projector(d, None, n_occ=3, from_density=True)
    assert float(chordal_distance_sq(p_from_d, p)) < 1e-16


def test_from_density_loss_gradient():
    # Regress a density: pred/ref are density matrices, retract via from_density.
    u = _rand_frame(8, 3)
    d_ref = 2.0 * (u @ u.mH)
    d_pred = (d_ref + 0.05 * _rand_herm(8)).clone().requires_grad_(True)
    res = grassmann_p_loss_single(d_pred, d_ref, None, n_occ=3, lambda_chordal=1.0, from_density=True)
    res.loss.backward()
    assert torch.isfinite(d_pred.grad).all()
    assert float(d_pred.grad.abs().max()) > 0.0


def test_from_density_smetric_uses_ao_density_kernel():
    # For an AO density kernel D in a non-orthogonal basis, D S D = D and the
    # orthonormal Grassmann projector is X D X (X = S^{1/2}), NOT X^{-1} D X^{-1}.
    # from_density must transport with X so it recovers the same subspace as H.
    n, k = 7, 3
    h = _rand_herm(n)
    s = _rand_spd(n)
    p_orth = occupied_projector(h, s, n_occ=k)          # H-route projector (orthonormal basis)
    _x, x_inv = s_half_and_inv(s)
    d_ao = x_inv @ p_orth @ x_inv                        # AO density kernel D = X^{-1} P X^{-1}
    p_from_d = occupied_projector(d_ao, s, n_occ=k, from_density=True)
    assert float(chordal_distance_sq(p_from_d, p_orth)) < 1e-8


def test_dense_batched_over_k():
    hp = torch.stack([_rand_herm(6, complex_=True) for _ in range(4)])
    hr = hp + 0.05 * torch.stack([_rand_herm(6, complex_=True) for _ in range(4)])
    loss, stats = dense_grassmann_p_loss(hp, hr, None, n_occ=2, lambda_chordal=1.0)
    assert float(loss) > 0.0
    assert int(stats["grassmann_skipped"]) == 0
    assert float(stats["grassmann_rank"]) == 2.0


def test_s_half_inverse_consistency():
    s = _rand_spd(6)
    x, xi = s_half_and_inv(s)
    assert torch.allclose(x @ x, s, atol=1e-8)
    assert torch.allclose(x @ xi, torch.eye(6, dtype=torch.float64), atol=1e-8)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa
            fail += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)
