# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equivalence tests for the batched per-k eigh fast-path in ``dense_grassmann_p_loss``.

When ``n_occ`` is constant across the selected k-indices and more than one matrix is
selected, ``dense_grassmann_p_loss`` batches the two float64 eighs (reference + predicted)
into a single ``[K, N, N]`` ``stable_eigh`` each and vectorizes every downstream quantity,
instead of the per-k Python loop.  These tests lock that the fast-path is a
behavior-preserving optimization:

* loss + every stats key agree with the loop to ~1e-9 (the batched eigh/matmul/chordal
  paths are byte-identical; only the diagnostic ``grassmann_geo_dist``, computed under
  ``no_grad`` from a *batched vs unbatched SVD*, can differ at the ~1e-14 float rounding
  level -- it never touches the loss or the gradient);
* gradients w.r.t. ``h_pred`` agree to ~1e-7;
* a metallic k (reference gap < ``min_gap``) raises ``SkippableRecord`` in BOTH paths.

Run standalone (self-running ``__main__``; prints PASS/FAIL, exits nonzero on failure):
    python dptb/tests/test_grassmann_batched_eigh.py
or under pytest:
    pytest dptb/tests/test_grassmann_batched_eigh.py -q
"""
import torch

from dptb.nnops.grassmann import dense_grassmann_p_loss, SkippableRecord
from dptb.nnops._manifold_math import occupied_projector, s_half_and_inv

torch.manual_seed(0)

# stats keys whose loop/batched agreement we assert bit-for-bit (~1e-9).  geo_dist is
# checked separately at a looser tolerance because batched vs per-k SVD is not required to
# be bitwise identical (it is a no_grad diagnostic, not part of the loss/gradient).
_TIGHT_STATS = (
    "grassmann_loss", "grassmann_chordal", "grassmann_eps", "grassmann_pred_gap",
    "grassmann_gap_weight", "grassmann_gap", "grassmann_rank", "grassmann_skipped",
)


# --------------------------------------------------------------------------- #
# synthetic insulating Hamiltonians
# --------------------------------------------------------------------------- #
def _gapped_stack(K, n, n_occ, gaps, complex_=False, seed=0):
    """A ``[K, N, N]`` stack of Hermitian Hamiltonians with a clear occ/vir gap.

    Eigenvalues: occupied block in ``[-2, -1]`` (``n_occ`` of them), virtual block starting
    ``gap`` above the HOMO up to ``+1``.  ``gaps`` is a scalar (same gap for every k) or a
    per-k list, letting us make a chosen k metallic.
    """
    if isinstance(gaps, (int, float)):
        gaps = [float(gaps)] * K
    dt = torch.complex128 if complex_ else torch.float64
    g = torch.Generator().manual_seed(seed)
    mats = []
    for k in range(K):
        w = torch.cat([
            torch.linspace(-2.0, -1.0, n_occ, dtype=torch.float64),
            torch.linspace(-1.0 + gaps[k], 1.0, n - n_occ, dtype=torch.float64),
        ])
        a = torch.randn(n, n, generator=g, dtype=dt)
        q, _ = torch.linalg.qr(a)
        mats.append((q * w.to(dt)) @ q.mH)
    return torch.stack(mats)


def _spd_stack(K, n, complex_=False, seed=0):
    """A ``[K, N, N]`` stack of SPD overlaps (per-k, all distinct)."""
    dt = torch.complex128 if complex_ else torch.float64
    g = torch.Generator().manual_seed(seed)
    mats = []
    for k in range(K):
        a = torch.randn(n, n, generator=g, dtype=dt)
        mats.append(a @ a.mH + n * torch.eye(n, dtype=dt))
    return torch.stack(mats)


def _density_stack_from_H(hr, S, n_occ):
    """AO density kernels ``D = X^{-1} P X^{-1}`` matching the H-route occupied subspace.

    Gives a from_density input whose top-``n_occ`` occupations reproduce the same subspace,
    so the from_density boundary/transport branch is genuinely exercised.
    """
    out = []
    for k in range(hr.shape[0]):
        p = occupied_projector(hr[k], S[k], n_occ=n_occ)
        _x, x_inv = s_half_and_inv(S[k])
        out.append(x_inv @ p @ x_inv)
    return torch.stack(out)


# --------------------------------------------------------------------------- #
# equivalence helpers
# --------------------------------------------------------------------------- #
def _assert_stats_agree(s_loop, s_bat, tol=1e-9, geo_tol=1e-6):
    assert set(s_loop) == set(s_bat), (set(s_loop) ^ set(s_bat))
    for key in _TIGHT_STATS:
        d = abs(float(s_loop[key]) - float(s_bat[key]))
        assert d < tol, f"stat {key} diverged: |loop-batched| = {d:.3e} >= {tol:.0e}"
    # geo_dist: batched vs per-k SVD -> tiny float difference is allowed (no_grad diagnostic)
    dg = abs(float(s_loop["grassmann_geo_dist"]) - float(s_bat["grassmann_geo_dist"]))
    assert dg < geo_tol, f"grassmann_geo_dist diverged: {dg:.3e} >= {geo_tol:.0e}"


def _loop_vs_batched(hp, hr, s, n_occ, **kw):
    """Run both paths on detached-but-grad copies; return (loss_loop, stats_loop, loss_bat,
    stats_bat, grad_loop, grad_bat)."""
    hp_loop = hp.detach().clone().requires_grad_(True)
    hp_bat = hp.detach().clone().requires_grad_(True)
    l_loop, st_loop = dense_grassmann_p_loss(hp_loop, hr, s, n_occ=n_occ, _force_loop=True, **kw)
    l_bat, st_bat = dense_grassmann_p_loss(hp_bat, hr, s, n_occ=n_occ, _force_loop=False, **kw)
    l_loop.backward()
    l_bat.backward()
    return l_loop, st_loop, l_bat, st_bat, hp_loop.grad, hp_bat.grad


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_batched_matches_loop_real_no_overlap():
    K, n, n_occ = 6, 8, 3
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=False, seed=1)
    hp = hr + 0.05 * _gapped_stack(K, n, n_occ, 1.0, complex_=False, seed=2)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.3, eps_window=4, min_gap=0.05)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(hp, hr, None, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-9
    _assert_stats_agree(sl, sb)
    assert float((gl - gb).abs().max()) < 1e-7


def test_batched_matches_loop_complex_no_overlap():
    K, n, n_occ = 5, 7, 2
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=3)
    hp = hr + 0.05 * _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=4)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.2, eps_window=None, min_gap=0.05,
              chordal_normalize=True)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(hp, hr, None, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-9
    _assert_stats_agree(sl, sb)
    assert float((gl - gb).abs().max()) < 1e-7


def test_batched_matches_loop_with_overlap():
    K, n, n_occ = 4, 7, 3
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=5)
    S = _spd_stack(K, n, complex_=True, seed=6)
    hp = hr + 0.05 * _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=7)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.2, eps_window=3, min_gap=0.05, gauge_mu=False)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(hp, hr, S, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-9
    _assert_stats_agree(sl, sb)
    assert float((gl - gb).abs().max()) < 1e-7


def test_batched_matches_loop_from_density_with_overlap():
    K, n, n_occ = 4, 7, 3
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=8)
    S = _spd_stack(K, n, complex_=True, seed=9)
    dr = _density_stack_from_H(hr, S, n_occ)
    dp = dr + 0.02 * _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=10)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.0, from_density=True, min_gap=0.02)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(dp, dr, S, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-9
    _assert_stats_agree(sl, sb)
    assert float((gl - gb).abs().max()) < 1e-7


def test_batched_matches_loop_soft_pred_gap_downweight():
    # Some predicted k are transiently metallic (pred_gap < min_gap) while the REFERENCE is
    # gapped -> soft_pred_gap must apply the same per-k (gap/min_gap)^2 weight in both paths.
    K, n, n_occ = 5, 8, 3
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=11)
    # predicted spectrum: k=1 and k=3 nearly metallic, others healthy
    hp = _gapped_stack(K, n, n_occ, [1.0, 0.005, 1.0, 0.002, 1.0], complex_=True, seed=12)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.0, min_gap=0.05, soft_pred_gap=True,
              check_pred_gap=False)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(hp, hr, None, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-9
    _assert_stats_agree(sl, sb)
    assert float((gl - gb).abs().max()) < 1e-7
    # the down-weight is genuinely active (< 1), so this test actually exercises the branch
    assert float(sb["grassmann_gap_weight"]) < 1.0


def test_metallic_reference_k_raises_in_both_paths():
    # k=2 has a vanishing reference gap -> both the loop and the batched path must raise
    # SkippableRecord (the whole record is skipped, identically).
    K, n, n_occ = 5, 8, 3
    hr = _gapped_stack(K, n, n_occ, [1.0, 1.0, 0.001, 1.0, 1.0], complex_=True, seed=13)
    hp = hr + 0.05 * _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=14)
    for force in (True, False):
        raised = False
        try:
            dense_grassmann_p_loss(hp, hr, None, n_occ=n_occ, min_gap=0.05, _force_loop=force)
        except SkippableRecord:
            raised = True
        assert raised, f"metallic reference k did not raise (force_loop={force})"


def test_check_pred_gap_metallic_prediction_raises_in_both_paths():
    # With check_pred_gap, a metallic PREDICTED k hard-skips in both paths.
    K, n, n_occ = 5, 8, 3
    hr = _gapped_stack(K, n, n_occ, 1.0, complex_=True, seed=15)
    hp = _gapped_stack(K, n, n_occ, [1.0, 1.0, 0.001, 1.0, 1.0], complex_=True, seed=16)
    for force in (True, False):
        raised = False
        try:
            dense_grassmann_p_loss(hp, hr, None, n_occ=n_occ, min_gap=0.05,
                                   check_pred_gap=True, _force_loop=force)
        except SkippableRecord:
            raised = True
        assert raised, f"metallic predicted k did not hard-skip (force_loop={force})"


def test_single_matrix_uses_loop_and_matches():
    # K == 1 must fall back to the loop (the fast-path requires > 1 selected matrix);
    # the public call (batched switch on) must still equal the forced loop.
    n, n_occ = 8, 3
    hr = _gapped_stack(1, n, n_occ, 1.0, complex_=True, seed=17)
    hp = hr + 0.05 * _gapped_stack(1, n, n_occ, 1.0, complex_=True, seed=18)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.3, eps_window=4, min_gap=0.05)
    ll, sl, lb, sb, gl, gb = _loop_vs_batched(hp, hr, None, n_occ, **kw)
    assert abs(float(ll) - float(lb)) < 1e-12   # same code path -> exact
    _assert_stats_agree(sl, sb, geo_tol=1e-12)
    assert float((gl - gb).abs().max()) < 1e-12


def test_varying_n_occ_falls_back_to_loop_and_matches():
    # A per-k n_occ list (not constant) must NOT take the batched fast-path; verify the
    # public call still equals the forced loop.
    K, n = 4, 9
    n_occ = [3, 3, 4, 3]  # not constant -> loop
    hr = _gapped_stack(K, n, 3, 1.0, complex_=True, seed=19)  # gap position ok for n_occ 3/4
    hp = hr + 0.03 * _gapped_stack(K, n, 3, 1.0, complex_=True, seed=20)
    kw = dict(lambda_chordal=1.0, lambda_eps=0.1, eps_window=3, min_gap=0.0)
    hp_loop = hp.detach().clone().requires_grad_(True)
    hp_pub = hp.detach().clone().requires_grad_(True)
    l_loop, s_loop = dense_grassmann_p_loss(hp_loop, hr, None, n_occ=n_occ, _force_loop=True, **kw)
    l_pub, s_pub = dense_grassmann_p_loss(hp_pub, hr, None, n_occ=n_occ, _force_loop=False, **kw)
    l_loop.backward()
    l_pub.backward()
    assert abs(float(l_loop) - float(l_pub)) < 1e-12   # same (loop) code path -> exact
    _assert_stats_agree(s_loop, s_pub, geo_tol=1e-12)
    assert float((hp_loop.grad - hp_pub.grad).abs().max()) < 1e-12


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)
