"""Spectral clipping fast path of ``Eigenvalues`` (dptb/nn/band_fast.py, dptb/nn/cusolver_batched.py).

Small periodic toy systems (Si / O basis with a missing 2s on O, a full spin-orbital idp and a non-SOC idp); every expected number comes from
the original ``HR2HK`` assembly plus an independent NumPy float64 implementation of the production rule (remove the directions of S with
lambda <= ill_threshold, pad with 1e4 eV) and of the clipping (generalized eigenvalues of (R, S) clipped to [-b, b]).
The CUDA tests (cuSOLVER batched solver, streams, the cuSOLVER backend of ``Eigenvalues``) are skipped without a GPU.
"""
import itertools
import subprocess
import sys

import numpy as np
import pytest
import torch

from dptb.data import AtomicDataDict as A
from dptb.data.transforms import OrbitalMapper
from dptb.nn import band_fast
from dptb.nn.energy import Eigenvalues
from dptb.nn.hr2hk import HR2HK
from dptb.tests._requires import requires_cuda

PAD = 1.0e4
KPTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.13, 0.31, -0.27], [0.5, 0.5, 0.5], [-0.21, 0.07, 0.44], [0.33, 0.33, 0.0]])


# ------------------------------------------------------------------ toy systems
def make_idp(soc):
    basis = {"O": ["1s", "1p"], "Si": ["1s", "2s", "1p"]}  # a fresh dict: OrbitalMapper sorts it in place
    return OrbitalMapper(basis=basis, method="e3tb", has_soc=soc, soc_complex_doubling=True)


def make_graph():
    types = torch.tensor([0, 1, 0])  # O, Si, O  (types follow the atomic number)
    edges, shifts = [], []
    for u, v in itertools.product(range(3), repeat=2):
        for s in itertools.product((-1, 0, 1), repeat=3):
            if u == v and s == (0, 0, 0):
                continue
            edges.append((u, v))
            shifts.append(s)
    return dict(types=types, ei=torch.tensor(edges).T.contiguous(), esh=torch.tensor(shifts, dtype=torch.float64))


def pair_slices(idp):
    """(feature column, shell dimension, shell pair) of every orbital-pair slice."""
    dims, _ = band_fast._shell_dims(idp.full_basis)
    out = []
    for i, io in enumerate(idp.full_basis):
        for j, jo in enumerate(idp.full_basis):
            sl = idp.orbpair_maps.get(io + "-" + jo)
            if sl is not None:
                out.append((sl.start, dims[i], dims[j], io == jo))
    return out


def random_features(idp, graph, rng, scale, dtype):
    rme = idp.reduced_matrix_element
    n_node, n_edge = len(graph["types"]), graph["ei"].shape[1]
    f = lambda n: torch.tensor(scale * rng.normal(size=(n, rme)), dtype=dtype)
    return f(n_node), f(n_edge)


def spin_diagonal(idp, feats):
    """Make full spin-orbital features spin diagonal with uu = dd (a copy)."""
    out = []
    for F in feats:
        F = F.clone()
        for start, di, dj, _ in pair_slices(idp):
            w = di * dj
            blk = F[:, start:start + 8 * w].view(F.shape[0], 2, 4, w)
            blk[:, :, 3] = blk[:, :, 0]
            blk[:, :, 1:3] = 0
        out.append(F)
    return tuple(out)


def overlap_features(idp, graph, rng, scale, dtype):
    """Real overlap features (the uu real part of every pair block): identity on the onsite shell diagonals plus small noise, so S(k) > 0."""
    rme = idp.reduced_matrix_element
    out = []
    for n, onsite in ((len(graph["types"]), True), (graph["ei"].shape[1], False)):
        F = torch.zeros(n, rme, dtype=torch.float64)
        for start, di, dj, same in pair_slices(idp):
            F[:, start:start + di * dj] = torch.tensor(scale * rng.normal(size=(n, di * dj)))
            if onsite and same:
                F[:, start:start + di * dj] = (torch.eye(di, dtype=torch.float64).flatten())[None, :].expand(n, -1) + F[:, start:start + di * dj] * 0.2
        out.append(F.to(dtype))
    return tuple(out)


def shift_overlap(idp, S, delta):
    """S(k) -> S(k) - delta * 1 for every k (the onsite shell-diagonal entries)."""
    Sn, Se = S[0].clone(), S[1]
    for start, di, dj, same in pair_slices(idp):
        if same:
            for a in range(di):
                Sn[:, start + a * di + a] -= delta
    return Sn, Se


def dense(idp, graph, feats, kpts, overlap, dtype=torch.float64):
    """The original HR2HK assembly of one feature pair at the k points (complex128).

    Non-SOC features are handed over as complex128: HR2HK's scalar path casts real features to complex64
    (``recover_complex_tensor`` -> ``cfloat``), which would limit the reference to float32 precision."""
    module = HR2HK(idp=idp, overlap=overlap, edge_field="e", node_field="n", out_field="o", dtype=dtype)
    n, e = feats
    if not getattr(idp, "has_soc", False):
        n, e = n.to(torch.complex128), e.to(torch.complex128)
    data = {A.ATOM_TYPE_KEY: graph["types"], A.EDGE_INDEX_KEY: graph["ei"], A.EDGE_CELL_SHIFT_KEY: graph["esh"],
            A.KPOINT_KEY: torch.as_tensor(kpts, dtype=dtype), "n": n, "e": e}
    return module(data)["o"].to(torch.complex128).numpy()


def lambda_min(idp, graph, S, kpts):
    return np.array([np.linalg.eigvalsh((s + s.conj().T) / 2)[0] for s in dense(idp, graph, S, kpts, True)])


def reference(S, H0, R, b, threshold):
    """Independent float64 NumPy levels of one k point: production rule (+ padding) and spectral clipping.  -> (levels, n_kept, n_clipped)"""
    N, D = S.shape[0], H0.shape[0]
    lam, U = np.linalg.eigh((S + S.conj().T) / 2)
    keep = lam > (0.0 if threshold is None else threshold)
    X = U[:, keep] / np.sqrt(lam[keep])
    Xb = X if D == N else np.kron(np.eye(2), X)
    Ht, Rt = Xb.conj().T @ H0 @ Xb, Xb.conj().T @ R @ Xb
    n_clipped = 0
    if b is not None:
        mu, W = np.linalg.eigh((Rt + Rt.conj().T) / 2)
        n_clipped = int((np.abs(mu) > b).sum())
        Rt = (W * np.clip(mu, -b, b)[None, :]) @ W.conj().T
    else:
        Rt = (Rt + Rt.conj().T) / 2
    lev = np.linalg.eigvalsh(Ht + Rt)
    return np.concatenate([lev, np.full(D - lev.size, PAD)]), int(keep.sum()), n_clipped


class Toy:
    """One toy structure: H0, residual R, overlap S as features, the data dict for ``Eigenvalues`` and the dense reference matrices."""

    def __init__(self, soc=True, dtype=torch.float64, seed=0, r_scale=0.05, spin="diag", target=None):
        rng = np.random.default_rng(seed)
        self.idp, self.graph, self.soc, self.dtype = make_idp(soc), make_graph(), soc, dtype
        self.H0 = random_features(self.idp, self.graph, rng, 0.3, dtype)
        R = random_features(self.idp, self.graph, rng, r_scale, dtype)
        self.R = spin_diagonal(self.idp, R) if (soc and spin == "diag") else R
        self.S = overlap_features(self.idp, self.graph, rng, 0.012, dtype)
        self.lam0 = lambda_min(self.idp, self.graph, self.S, KPTS)
        if target is not None:  # push the smallest S eigenvalue over the k points down to `target`
            self.S = shift_overlap(self.idp, self.S, float(self.lam0.min()) - target)
        self.H = tuple(h0 + r for h0, r in zip(self.H0, self.R))
        self.Rrec = tuple(h - h0 for h, h0 in zip(self.H, self.H0))  # what the solver sees as the residual (same rounding)
        self.k = KPTS
        self.S_k = dense(self.idp, self.graph, self.S, KPTS, True)
        self.H0_k = dense(self.idp, self.graph, self.H0, KPTS, False if soc else False)
        self.R_k = dense(self.idp, self.graph, self.Rrec, KPTS, False)
        self.lam = lambda_min(self.idp, self.graph, self.S, KPTS)

    def data(self, h0=True, requires_grad=False):
        d = {A.ATOM_TYPE_KEY: self.graph["types"], A.EDGE_INDEX_KEY: self.graph["ei"], A.EDGE_CELL_SHIFT_KEY: self.graph["esh"].to(self.dtype),
             A.NODE_FEATURES_KEY: self.H[0].clone().requires_grad_(requires_grad), A.EDGE_FEATURES_KEY: self.H[1].clone().requires_grad_(requires_grad),
             A.NODE_OVERLAP_KEY: self.S[0], A.EDGE_OVERLAP_KEY: self.S[1], A.KPOINT_KEY: torch.as_tensor(self.k, dtype=self.dtype)}
        if h0:
            d[A.NODE_H0_KEY], d[A.EDGE_H0_KEY] = self.H0[0], self.H0[1]
        return d

    def module(self, device="cpu", **kw):
        return Eigenvalues(idp=self.idp, device=device, dtype=self.dtype, s_edge_field=A.EDGE_OVERLAP_KEY, s_node_field=A.NODE_OVERLAP_KEY,
                           s_out_field=A.OVERLAP_KEY, **kw)

    def expected(self, b=10.0, threshold=1e-5):
        rows = [reference(S, H0, R, b, threshold) for S, H0, R in zip(self.S_k, self.H0_k, self.R_k)]
        return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def run(module, data, **kw):
    with torch.no_grad():
        out = module(data, **kw)
    return out[A.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()


def window(levels, ref, width=60.0):
    """Mask of the levels in the physical window (the levels far above are amplified by 1/lambda and carry float32 noise)."""
    return (np.abs(ref) < width) & (ref < PAD / 2)


# ------------------------------------------------------------------ assembly and layout
@pytest.mark.parametrize("soc", [True, False])
def test_dense_assembler_matches_hr2hk(soc):
    toy = Toy(soc=soc, spin="full")
    g = toy.graph
    for mode, feats, overlap in (("overlap", toy.S, True), ("soc" if soc else "overlap", toy.H0, False)):
        asm = band_fast.DenseRAssembler(toy.idp, g["types"], g["ei"], g["esh"], mode, "cpu")
        got = asm.hk(asm.real_space(*feats, torch.float64), torch.as_tensor(KPTS)).numpy()
        np.testing.assert_allclose(got, dense(toy.idp, g, feats, KPTS, overlap), atol=1e-12)
    if soc:  # the uu-only assembler is the uu block of the full one
        full = band_fast.DenseRAssembler(toy.idp, g["types"], g["ei"], g["esh"], "soc", "cpu")
        uu = band_fast.DenseRAssembler(toy.idp, g["types"], g["ei"], g["esh"], "soc", "cpu", uu_only=True)
        N = full.N
        a = full.hk(full.real_space(*toy.R, torch.float64), torch.as_tensor(KPTS))[:, :N, :N]
        np.testing.assert_allclose(uu.hk(uu.real_space(*toy.R, torch.float64), torch.as_tensor(KPTS)).numpy(), a.numpy(), atol=1e-12)


def test_residual_spin_structure_detection():
    toy_diag, toy_gen = Toy(spin="diag"), Toy(spin="full")
    eps = torch.finfo(torch.float64).eps
    assert band_fast.detect_kind(toy_diag.idp, toy_diag.Rrec, 3.0, eps)[0] == "uu"
    assert band_fast.detect_kind(toy_gen.idp, toy_gen.Rrec, 3.0, eps)[0] == "general"
    assert band_fast.detect_kind(Toy(soc=False).idp, Toy(soc=False).Rrec, 3.0, eps)[0] == "scalar"
    # float32 features: H - H0 of a spin-diagonal residual differs between the two spin blocks by rounding only
    t32 = Toy(dtype=torch.float32, spin="diag")
    assert band_fast.detect_kind(t32.idp, t32.Rrec, 3.0, torch.finfo(torch.float32).eps)[0] == "uu"


def test_importing_the_modules_does_not_load_cusolver():
    code = ("import dptb.nn.energy, dptb.nn.band_fast, dptb.nn.cusolver_batched as m; "
            "assert m._load_library.cache_info().currsize == 0")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


# ------------------------------------------------------------------ the legacy path is untouched
def test_legacy_path_is_bitwise_unchanged_outside_the_fast_path():
    toy = Toy(spin="diag")
    mod = toy.module()
    ev_off = run(mod, toy.data(), clip_b=None)  # clip_b switched off
    assert mod.last_clip_stats is None
    ev_no_h0 = run(mod, toy.data(h0=False))  # data without H0
    assert mod.last_clip_stats is None
    with torch.enable_grad():  # differentiation: H0 present, grad mode on
        ev_grad = mod(toy.data(requires_grad=True))[A.ENERGY_EIGENVALUE_KEY][0].detach().numpy()
    assert mod.last_clip_stats is None
    with torch.no_grad():  # H requires grad even though grad mode is off
        ev_req = mod(toy.data(requires_grad=True))[A.ENERGY_EIGENVALUE_KEY][0].detach().numpy()
    assert mod.last_clip_stats is None
    assert np.array_equal(ev_off, ev_no_h0)
    # with gradients in play the legacy eigvalsh also computes eigenvectors (for the backward pass): another LAPACK route,
    # equal up to rounding
    for other in (ev_grad, ev_req):
        np.testing.assert_allclose(ev_off, other, rtol=0, atol=1e-12)
    # the same numbers as a plain generalized solve (healthy S: nothing is removed)
    ref = np.stack([reference(S, H0 + R, np.zeros_like(R), None, None)[0] for S, H0, R in zip(toy.S_k, toy.H0_k, toy.R_k)])
    np.testing.assert_allclose(ev_off, ref, atol=1e-9)


def test_legacy_path_with_ill_threshold_none_is_plain_cholesky():
    toy = Toy(spin="diag")
    ev = run(toy.module(), toy.data(), clip_b=None, ill_threshold=None)
    for k, (S, H0, R) in enumerate(zip(toy.S_k, toy.H0_k, toy.R_k)):
        Sb = np.kron(np.eye(2), S)
        L = np.linalg.cholesky(Sb)
        Li = np.linalg.inv(L)
        np.testing.assert_allclose(ev[k], np.linalg.eigvalsh(Li @ (H0 + R) @ Li.conj().T), atol=1e-9)


def test_output_layout_matches_the_legacy_path():
    for dtype in (torch.float64, torch.float32):
        toy = Toy(dtype=dtype)
        mod = toy.module()
        for nested in (False, True):
            d = toy.data()
            if nested:
                d[A.KPOINT_KEY] = torch.nested.as_nested_tensor([d[A.KPOINT_KEY]])
            with torch.no_grad():
                fast = mod(dict(d))
                legacy = mod(dict(d), clip_b=None)
            assert mod.last_clip_stats is None  # reset by the legacy call; the fast call filled it
            a, b = fast[A.ENERGY_EIGENVALUE_KEY], legacy[A.ENERGY_EIGENVALUE_KEY]
            assert a.is_nested and b.is_nested and a[0].shape == b[0].shape == (len(KPTS), 2 * toy.S_k.shape[-1])
            assert a[0].dtype == b[0].dtype == dtype
            assert fast[A.KPOINT_KEY].is_nested == nested
            assert bool((a[0].diff(dim=1) >= 0).all())  # ascending


# ------------------------------------------------------------------ the fast path against the reference
@pytest.mark.parametrize("soc", [True, False])
def test_in_bound_residual_equals_the_unclipped_solve(soc):
    toy = Toy(soc=soc, r_scale=1e-3)
    mod = toy.module()
    clipped = run(mod, toy.data())
    stats = mod.last_clip_stats
    assert stats["path"] == ("uu" if soc else "scalar") and stats["k_clipped"] == 0 and stats["k_fp64"] == 0 and stats["k_total"] == len(KPTS)
    # the legacy scalar (non-SOC) assembly casts real features to complex64 (HR2HK), so it agrees to float32 precision only
    np.testing.assert_allclose(clipped, run(mod, toy.data(), clip_b=None), atol=1e-9 if soc else 1e-6)
    np.testing.assert_allclose(clipped, toy.expected()[0], atol=1e-9)


@pytest.mark.parametrize("soc,spin", [(True, "diag"), (True, "full"), (False, "diag")])
@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-8), (torch.float32, 3e-3)])
def test_out_of_bound_component_is_clipped_like_the_reference(soc, spin, dtype, atol):
    toy = Toy(soc=soc, dtype=dtype, spin=spin, r_scale=0.05, target=2e-3)  # lambda_min(S) = 2e-3 at one k: R amplified 500x there
    mod = toy.module()
    got = run(mod, toy.data())
    ref, _, n_clip = toy.expected()
    stats = mod.last_clip_stats
    assert stats["path"] == ("scalar" if not soc else ("uu" if spin == "diag" else "general"))
    assert n_clip.sum() > 0 and stats["n_clipped"] > 0 and stats["k_clipped"] > 0, "the toy has no out-of-bound component"
    assert stats["k_fp64"] == 0
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=atol)
    unclipped = run(mod, toy.data(), clip_b=None)
    assert np.abs(unclipped - ref)[win].max() > 100 * atol, "clipping changed nothing: the test would not see a missing clip"


def test_uu_path_equals_general_path_on_a_spin_diagonal_residual():
    toy = Toy(spin="diag", r_scale=0.05, target=2e-3)
    g = toy.graph
    out = {}
    for kind in ("uu", "general"):
        solver = band_fast.SpectralBandSolver(toy.idp, g["types"], g["ei"], g["esh"], toy.S, toy.H0, toy.Rrec, "cpu", "fp64", kind=kind)
        out[kind] = solver.solve(toy.k, b=10.0).numpy()
        assert solver.stats["n_clipped"] > 0
    ref = toy.expected()[0]
    win = window(out["uu"], ref)
    np.testing.assert_allclose(out["uu"][win], out["general"][win], atol=1e-8)
    np.testing.assert_allclose(out["uu"][win], ref[win], atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("target,threshold", [(1e-6, 1e-5), (7e-6, 1e-5), (5e-5, 1e-5), (1e-6, None), (3e-3, 1e-5)])
def test_near_singular_overlap_follows_the_production_rule(dtype, target, threshold):
    toy = Toy(dtype=dtype, spin="diag", r_scale=0.01, target=target)
    mod = toy.module()
    got = run(mod, toy.data(), ill_threshold=threshold)
    ref, kept, _ = toy.expected(threshold=threshold)
    stats = mod.last_clip_stats
    N = toy.S_k.shape[-1]
    n_removed = int((N - kept).sum())
    assert stats["n_dropped"] == 2 * n_removed and stats["k_dropped"] == int((kept < N).sum())
    assert stats["n_effective_min"] == 2 * int(kept.min())
    # padding = levels at exactly 1e4 eV (a genuine level of a near-null direction of H0 can lie far above the window too)
    pad_got, pad_ref = np.isclose(got, PAD, rtol=0, atol=1e-3), np.isclose(ref, PAD, rtol=0, atol=1e-3)
    assert int(pad_got.sum()) == int(pad_ref.sum()) == 2 * n_removed
    np.testing.assert_array_equal(pad_got, pad_ref)
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=1e-8 if dtype == torch.float64 else 3e-3)
    if dtype == torch.float32 and threshold is not None and target < 1e-4:
        assert stats["k_fp64"] >= 1  # near-singular k points leave float32
        rows = np.nonzero(toy.lam < 1e-4)[0]
        g, r = got[rows], ref[rows]
        w = window(g, r)
        np.testing.assert_allclose(g[w], r[w], rtol=2e-7, atol=1e-6)  # solved in float64, returned in float32
    if dtype == torch.float32 and target > 1e-3:
        assert stats["k_fp64"] == 0


def test_mixed_batch_healthy_rows_do_not_depend_on_the_near_singular_ones():
    toy = Toy(dtype=torch.float32, spin="diag", r_scale=0.01, target=1e-7)
    mod = toy.module()
    got = run(mod, toy.data(), ill_threshold=1e-5)
    stats = mod.last_clip_stats
    assert stats["k_fp64"] >= 1 and stats["k_fp64"] < len(KPTS)
    ref = toy.expected(threshold=1e-5)[0]
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=3e-3)
    healthy = np.nonzero(toy.lam > 1e-3)[0]
    assert healthy.size
    for k in healthy[:2]:  # a healthy k point alone gives the same levels as inside the mixed batch
        d = toy.data()
        d[A.KPOINT_KEY] = torch.as_tensor(toy.k[k:k + 1], dtype=toy.dtype)
        alone = run(mod, d, ill_threshold=1e-5)[0]
        w = window(alone, ref[k])
        np.testing.assert_allclose(alone[w], got[k][w], atol=2e-3)


def test_float64_module_needs_no_precision_routing():
    toy = Toy(dtype=torch.float64, spin="diag", r_scale=0.01, target=5e-5)  # below the float32 limit (1e-4), above ill_threshold
    mod = toy.module()
    got = run(mod, toy.data())
    assert mod.last_clip_stats["k_fp64"] == 0
    ref = toy.expected()[0]
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=1e-8)


def test_kchunk_nk_and_tf32_flag():
    toy = Toy(spin="diag", r_scale=0.05, target=2e-3)
    mod = toy.module(kchunk=2)
    got = run(mod, toy.data())
    assert mod.last_clip_stats["kchunk"] == 2
    ref = toy.expected()[0]
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=1e-8)
    mod = toy.module()
    run(mod, toy.data(), nk=3)
    assert mod.last_clip_stats["kchunk"] == 3
    before = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        run(mod, toy.data())
        assert torch.backends.cuda.matmul.allow_tf32 is True  # restored after the fast path
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before[0]
        torch.set_float32_matmul_precision(before[1])


def test_h0_with_another_layout_is_rejected():
    toy = Toy()
    d = toy.data()
    d[A.EDGE_H0_KEY] = d[A.EDGE_H0_KEY][:, :-1]
    with pytest.raises(ValueError, match="layout"):
        run(toy.module(), d)
    d[A.NODE_H0_KEY] = d[A.NODE_H0_KEY][:, : toy.idp.reduced_matrix_element // 8]
    with pytest.raises(ValueError, match="layout"):
        run(toy.module(), d)
    # in grad mode the same data take the legacy path and are not an error (training must not be affected)
    with torch.enable_grad():
        toy.module()(d)


def test_non_finite_input_is_never_returned_silently():
    toy = Toy(dtype=torch.float32)
    d = toy.data()
    d[A.KPOINT_KEY][1, 0] = float("nan")
    mod = toy.module()
    with pytest.raises((FloatingPointError, torch.linalg.LinAlgError)):
        run(mod, d)


def test_out_of_memory_retries_with_a_smaller_chunk(monkeypatch):
    toy = Toy(spin="diag", r_scale=0.05, target=2e-3)
    mod = toy.module()
    ref = run(mod, toy.data())
    original = band_fast.SpectralBandSolver._solve_chunk

    def flaky(self, kp, *args):
        if kp.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(self, kp, *args)

    monkeypatch.setattr(band_fast.SpectralBandSolver, "_solve_chunk", flaky)
    got = run(toy.module(kchunk=6), toy.data())
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_screen_diagnostics_find_every_out_of_bound_k_point():
    toy = Toy(spin="diag", r_scale=0.05, target=2e-3)
    mod = toy.module()
    mod.clip_diagnostics = True
    run(mod, toy.data())
    diag = mod.last_clip_stats["diag"]
    outside = (np.abs(diag["mu_min"]) > 10.0) | (np.abs(diag["mu_max"]) > 10.0)
    assert outside.any()
    assert not (outside & ~diag["flag"]).any()  # no false negative
    assert set(diag["k"].tolist()) == set(range(len(KPTS)))


# ------------------------------------------------------------------ CUDA
def herm_batch(batch, n, seed, dtype, device):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, n, n, generator=g, dtype=torch.float64) + 1j * torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    return (0.5 * (a + a.mH)).to(dtype).to(device)


@requires_cuda
@pytest.mark.parametrize("dtype,tol", [(torch.complex64, 2e-3), (torch.complex128, 1e-9)])
def test_cusolver_batched_matches_torch(dtype, tol):
    from dptb.nn.cusolver_batched import get_batched_eigh

    be = get_batched_eigh("cuda")
    for n in (7, 40, 130):
        A_ = herm_batch(5, n, n, dtype, "cuda")
        w, v = be.eigh(A_.clone())
        assert torch.allclose(w, torch.linalg.eigvalsh(A_), atol=tol)
        assert torch.allclose(A_ @ v, v * w[:, None, :].to(A_.dtype), atol=10 * tol)  # the returned vectors solve A v = w v
        assert torch.allclose(be.eigvalsh(A_.clone()), w, atol=tol)


@requires_cuda
def test_cusolver_batched_reports_a_bad_matrix_instead_of_returning_nan():
    from dptb.nn.cusolver_batched import CusolverBatchError, get_batched_eigh

    be = get_batched_eigh("cuda")
    A_ = herm_batch(4, 48, 1, torch.complex64, "cuda")
    A_[2, 5, 7] = A_[2, 7, 5] = float("nan")
    with pytest.raises(CusolverBatchError) as error:
        be.eigvalsh(A_.clone())
    assert error.value.bad.tolist() == [2]
    assert be.eigvalsh(A_[[0, 1, 3]].contiguous().clone()).isfinite().all()  # the finite matrices are fine


@requires_cuda
def test_cusolver_nan_batch_falls_back_and_is_counted():
    from dptb.nn.cusolver_batched import get_batched_eigh

    toy = Toy(dtype=torch.float32)
    g = toy.graph
    solver = band_fast.SpectralBandSolver(toy.idp, g["types"].cuda(), g["ei"].cuda(), g["esh"].cuda(), toy.S, toy.H0, toy.Rrec, "cuda", "fp32", kind="uu",
                                          batched=get_batched_eigh("cuda"))
    A_ = herm_batch(4, 48, 3, torch.complex64, "cuda")
    A_[2, 5, 7] = A_[2, 7, 5] = float("nan")
    solver.stats = {}
    with pytest.raises(torch.linalg.LinAlgError):  # torch refuses the NaN matrix too: nothing is returned silently
        solver._eigvalsh(A_)
    assert solver.stats["cusolver_fallback"] == 1


@requires_cuda
def test_cuda_fast_path_uses_cusolver_and_matches_the_cpu_result():
    toy = Toy(dtype=torch.float32, spin="diag", r_scale=0.05, target=2e-3)
    mod = toy.module("cuda")
    d = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in toy.data().items()}
    got = run(mod, d)
    assert mod.last_clip_stats["backend"] == "cusolver" and mod.last_clip_stats["cusolver_fallback"] == 0
    ref = toy.expected()[0]
    win = window(got, ref)
    np.testing.assert_allclose(got[win], ref[win], atol=3e-3)
    d = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in toy.data().items()}
    cpu_backend = toy.module("cuda", eig_backend="torch")
    again = run(cpu_backend, d)
    assert cpu_backend.last_clip_stats["backend"] == "torch"
    np.testing.assert_allclose(got[win], again[win], atol=3e-3)


@requires_cuda
def test_cuda_result_does_not_depend_on_the_stream():
    toy = Toy(dtype=torch.float32, spin="diag", r_scale=0.05, target=2e-3)
    mod = toy.module("cuda")
    cuda = lambda: {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in toy.data().items()}
    default = run(mod, cuda())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        side = run(mod, cuda())
    stream.synchronize()
    assert np.array_equal(default, side)
