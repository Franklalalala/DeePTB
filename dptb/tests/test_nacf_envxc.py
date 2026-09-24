"""Grid-free environment-XC tables, plan, McWEDA stabilization and the table build tool (CPU, synthetic species).

Everything runs on CPU with the pure-Python topology; the only spatial quadrature is the independent
three-centre reference used to test the finite-rank density expansion.
"""
import json
import math

import numpy as np
import pytest
import torch
from scipy.special import roots_legendre

from dptb.data.interfaces.p2_table import RadialBlockTable, abacus_m_order, real_sph_abacus
from dptb.nacf.envxc import (STABILIZATIONS, EnvXCBank, EnvXCStore, lda_pz81_v_dv_torch, mcweda_composition,
                             python_edge_topology, reference_edge_envxc, residual_rescale, shell_kappa)
from dptb.nacf.envxc_tables import (AtomicSource, build_density_projectors, build_table_values, distance_grid,
                                    lda_pz81_v_dv, save_species, sha256_file, two_centre_block)
from dptb.tests.nacf_support import (SPECIES, edge_overlap, float64_default, make_structure, shared_envxc_root,
                                     synthetic_source)
from tools import build_nacf_envxc_tables as cli

ARMS = ("d2", "d2_moment", "mcweda")


@pytest.fixture(autouse=True, scope="module")
def _float64():
    with float64_default():
        yield


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return shared_envxc_root(tmp_path_factory)


@pytest.fixture(scope="module")
def bonded(root):
    """Three atoms in a small orthorhombic cell (many periodic images) and the exact edge overlaps."""
    _, sources, _ = root
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    return g, edge_overlap(sources, g)


# --------------------------------------------------------------------------- reference quadratures
def ao_functions(source):
    return [(source.radial_fn(c), l, m) for c, l in enumerate(source.shells) for m in abacus_m_order(l)]


def envelope_functions(source):
    return [(source.envelope_fn(c), 0, 0) for c in range(source.nshells)]


def direct_three_centre(left, r_left, right, r_right, rho_fn, r_k, rmax, order=(160, 40, 80)):
    """Grid centred on the third centre: sum_g w_g f_mu(r-R_A) rho(|r-R_K|) g_nu(r-R_B)."""
    nr, nmu, nphi = order
    x, wx = roots_legendre(nr)
    mu, wmu = roots_legendre(nmu)
    r = (x + 1) * rmax / 2
    wr = wx * rmax / 2 * r * r
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    sin = np.sqrt(1 - mu**2)
    dirs = np.stack(np.broadcast_arrays(sin[:, None] * np.cos(phi), sin[:, None] * np.sin(phi), mu[:, None]), -1).reshape(-1, 3)
    wang = np.repeat(wmu, nphi) * 2 * np.pi / nphi
    xyz = (r[:, None, None] * dirs[None]).reshape(-1, 3) + r_k
    w = (wr[:, None] * wang[None]).ravel()
    rho = rho_fn(np.linalg.norm(xyz - r_k, axis=1))

    def evaluate(functions, centre):
        d = xyz - centre
        dist = np.linalg.norm(d, axis=1)
        return np.stack([f(dist) * real_sph_abacus(l, m, d) for f, l, m in functions], 1)
    return evaluate(left, r_left).T @ ((w * rho)[:, None] * evaluate(right, r_right))


def factor_block(store, kind, k, a, b, r_a, r_b, r_k):
    fa = store.table(kind, k, a).evaluate(r_a - r_k)
    fb = store.table(kind, k, b).evaluate(r_b - r_k)
    return fa.T @ (store.epsilon(k)[:, None] * fb)


# --------------------------------------------------------------------------- tables
def test_pz81_derivative_numpy_and_torch():
    n = np.geomspace(1e-9, 20.0, 400)
    v, dv = lda_pz81_v_dv(n)
    h = 1e-5
    fd = (lda_pz81_v_dv(n * (1 + h))[0] - lda_pz81_v_dv(n * (1 - h))[0]) / (2 * h * n)
    assert np.max(np.abs(fd - dv) / np.maximum(np.abs(fd), 1e-12)) < 1e-6
    tv, tdv = lda_pz81_v_dv_torch(torch.tensor(n))
    assert np.max(np.abs(tv.numpy() - v) / np.abs(v)) < 1e-12 and np.max(np.abs(tdv.numpy() - dv) / np.abs(dv)) < 1e-12
    zero_v, zero_dv = lda_pz81_v_dv_torch(torch.tensor([0.0, 1e-30]))
    assert torch.equal(zero_v, torch.zeros(2)) and torch.equal(zero_dv, torch.zeros(2))


def test_density_projectors_cover_the_density_support(root):
    _, sources, proj = root
    for s, p in proj.items():
        assert p.norb == sum(2 * l + 1 for l in p.q_l)
        assert np.all(p.epsilon_radial > 0)
        assert p.metadata["normalized_rho_metric_offdiag_max"] < 1e-8
        assert p.cutoff_bohr >= sources[s].orbital_cutoff_bohr
        # electrons dropped beyond the density cutoff are bounded by the threshold times the shell volume scale
        assert p.metadata["electrons_beyond_cutoff"] < 4 * math.pi * p.cutoff_bohr**3 * p.metadata["density_threshold_bohr3"]
        assert p.metadata["electrons_beyond_cutoff"] < p.metadata["electrons_beyond_orbital_cutoff"] + 1e-12


def test_two_centre_block_is_symmetric_and_supports_density_moment():
    src = synthetic_source("Xa", **SPECIES["Xa"])
    left = [(int(l), src.radial_fn(c)) for c, l in enumerate(src.shells)]
    d = 2.1
    block = two_centre_block(left, left, d, 64, src.orbital_cutoff_bohr, src.orbital_cutoff_bohr)
    # same species both sides: swapping the centres mirrors z, so M_mn(d) = (-1)^(l_m + l_n) M_nm(d)
    parity = np.repeat([(-1) ** l for l in src.shells], [2 * l + 1 for l in src.shells])
    assert np.allclose(block, parity[:, None] * parity[None, :] * block.T, atol=1e-9)
    dens = two_centre_block(left, left, d, 64, src.orbital_cutoff_bohr, src.orbital_cutoff_bohr, lambda ra, rb: src.density(ra) + src.density(rb))
    assert dens[0, 0] > 0 and abs(dens[0, 0]) > abs(block[0, 0]) * 1e-3


def test_factorized_density_moments_match_direct_quadrature(root):
    path, sources, proj = root
    store = EnvXCStore(path)
    A, B, K = "Xa", "Yb", "Yb"
    r_a, r_b, r_k = np.zeros(3), np.array([0.0, 0.0, 3.0]), np.array([1.6, 1.1, 1.3])   # off-axis third centre
    rmax = proj[K].cutoff_bohr
    ref = direct_three_centre(ao_functions(sources[A]), r_a, ao_functions(sources[B]), r_b, sources[K].density, r_k, rmax)
    ref_hi = direct_three_centre(ao_functions(sources[A]), r_a, ao_functions(sources[B]), r_b, sources[K].density, r_k, rmax, (220, 56, 112))
    assert np.max(np.abs(ref - ref_hi)) < 1e-5 * max(np.max(np.abs(ref)), 1e-12)
    fact = factor_block(store, "rhofac", K, A, B, r_a, r_b, r_k)
    # fixture configuration (rank 3 / l_buffer 2 / tail 1) measured 4.3% on these synthetic species
    assert np.linalg.norm(fact - ref) / np.linalg.norm(ref) < 0.08
    # mixed sigma-pi channel: the s_A - p_x(B) moment is nonzero for an off-axis centre and reproduced
    px = sum(2 * l + 1 for l in sources[B].shells[:1]) + 1          # first p shell of B, m=+1 (x)
    assert abs(ref[0, px]) > 1e-3 * np.max(np.abs(ref))
    assert abs(fact[0, px] - ref[0, px]) < 0.15 * abs(ref[0, px])
    ref_env = direct_three_centre(envelope_functions(sources[A]), r_a, envelope_functions(sources[B]), r_b, sources[K].density, r_k, rmax)
    fact_env = factor_block(store, "envfac", K, A, B, r_a, r_b, r_k)
    assert np.all(ref_env > 0)
    assert np.linalg.norm(fact_env - ref_env) / np.linalg.norm(ref_env) < 0.08
    # rank convergence: a richer auxiliary space must be clearly better than the poorest one
    rich = build_density_projectors(sources[K], radial_rank=4, l_buffer=3, tail_seeds=2, density_threshold=1e-7)
    poor = build_density_projectors(sources[K], radial_rank=1, l_buffer=0, tail_seeds=0, density_threshold=1e-7)

    def error(p):
        tables = []
        for src in (sources[A], sources[B]):
            f = build_table_values("rhofac", p, src, distances=distance_grid(p.cutoff_bohr + src.orbital_cutoff_bohr, 0.3), order=40)
            tables.append(RadialBlockTable(f["distances"], f["values"], f["left_shells"], f["right_shells"], f["support_bohr"]))
        val = tables[0].evaluate(r_a - r_k).T @ (p.epsilon_ao()[:, None] * tables[1].evaluate(r_b - r_k))
        return np.linalg.norm(val - ref) / np.linalg.norm(ref)
    assert error(rich) < 0.5 * error(poor)


def test_envelope_normalization_and_shell_frobenius_bound_on_exact_tables():
    """Stored envelope moments carry Y00^2 = 1/(4 pi): the s-s same-channel moment at d=0 equals the envelope
    moment exactly, and every shell block obeys ||D_ab||_F <= sqrt(n_a n_b) P_ab on the exact two-centre tables."""
    xa, yb = synthetic_source("Xa", **SPECIES["Xa"]), synthetic_source("Yb", **SPECIES["Yb"])
    dist = np.array([0.0, 1.3, 2.7, 4.6, 7.0, 10.5])
    for a, b in ((xa, xa), (xa, yb)):
        dens = build_table_values("pairmom", a, b, distances=dist, order=48)["values"]        # <mu|rho_pair|nu>
        env = build_table_values("envpair", a, b, distances=dist, order=48)["values"]         # <w_a|rho_pair|w_b>, w=|R|Y00
        kappa = shell_kappa(a.shells, b.shells)
        oa, ob = np.cumsum([0] + [2 * l + 1 for l in a.shells]), np.cumsum([0] + [2 * l + 1 for l in b.shells])
        for k in range(len(dist)):
            for i in range(a.nshells):
                for j in range(b.nshells):
                    block = dens[k, oa[i]:oa[i + 1], ob[j]:ob[j + 1]]
                    assert np.linalg.norm(block) <= kappa[i, j] * env[k, i, j] * (1 + 1e-9) + 1e-14, (a.symbol, b.symbol, k, i, j)
        if a is b:
            # d = 0, same s channel: |R|^2 = R^2 so the signed and the envelope moments coincide (the 4 pi convention)
            for c, l in enumerate(a.shells):
                if l == 0:
                    assert abs(dens[0, oa[c], oa[c]] - env[0, c, c]) < 1e-12 * max(1.0, abs(env[0, c, c]))
            assert env[0, 0, 0] > 1e-3      # non-trivial magnitude, so a missing 4 pi would be caught


# --------------------------------------------------------------------------- plan
@pytest.mark.parametrize("stabilization", STABILIZATIONS)
def test_plan_matches_numpy_reference(root, bonded, stabilization):
    """Plan = independent NumPy reference for every arm (v1) and for the stabilization-dependent McWEDA arm (v2);
    reverse edges are exact transposes and the default stabilization is v1."""
    path, _, _ = root
    g, S = bonded
    store = EnvXCStore(path)
    arms = ARMS if stabilization == "v1" else ("mcweda",)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    plan = bank.prepare(**g, arms=arms, topology="python", stabilization=stabilization, profile=True)
    out = plan(edge_overlap_ao=torch.tensor(S))
    ref = reference_edge_envxc(store, g, arms=arms, edge_overlap_ao=S, stabilization=stabilization)
    rev = plan.reverse.numpy()
    for arm in arms:
        got = out[arm].numpy()
        assert np.isfinite(got).all()
        assert np.max(np.abs(got - ref[arm])) < 1e-9 * max(1.0, np.max(np.abs(ref[arm]))), arm
        assert np.max(np.abs(ref[arm])) > 1e-6, f"{arm} correction vanished on a bonded periodic structure"
        assert torch.equal(out[arm], out[arm][plan.reverse].transpose(-1, -2))
        assert np.max(np.abs(ref[arm] - ref[arm][rev].transpose(0, 2, 1))) < 1e-8
    diag = out["diagnostics"]
    assert diag["stabilization"] == stabilization
    assert np.allclose(diag["b_shell"].numpy(), ref["b_shell"], atol=1e-10)
    # finite-rank leakage: non-overlapping pairs are gated (correction zero, negligible pair-XC elements) and
    # negative numerators occur only in the weak-overlap regime
    assert diag["numerator_without_overlap"] >= 0
    assert diag["gated_pair_xc_abs_max_eV"] < 1e-6
    assert diag["negative_numerators"] == 0 or diag["negative_numerator_max_overlap"] < 0.05
    if stabilization == "v2":
        assert diag["mcweda_total_rescaled_shell_pairs"] == 0
    else:
        default = bank.prepare(**g, arms=("mcweda",), topology="python")(edge_overlap_ao=torch.tensor(S))
        assert default["diagnostics"]["stabilization"] == "v1"
        assert torch.equal(default["mcweda"], out["mcweda"])
    with pytest.raises(ValueError):
        bank.prepare(**g, arms=("mcweda",), topology="python", stabilization="v3")
    with pytest.raises(ValueError):
        reference_edge_envxc(store, g, arms=("mcweda",), edge_overlap_ao=S, stabilization="v3")


def test_zero_environment_limit(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.diag([60.0, 60.0, 60.0]), cut)
    assert g["edge_index"].shape[1] == 2
    S = edge_overlap(sources, g)
    out = EnvXCBank(store, device="cpu", backend="torch").prepare(**g, arms=ARMS, topology="python")(edge_overlap_ao=torch.tensor(S))
    for arm in ARMS:
        assert torch.equal(out[arm], torch.zeros_like(out[arm]))
    assert torch.equal(out["diagnostics"]["b_shell"], torch.zeros_like(out["diagnostics"]["b_shell"]))
    assert out["diagnostics"]["terms"] == 0


def test_translation_and_permutation_invariance(root, bonded):
    path, _, _ = root
    base, _ = bonded
    bank = EnvXCBank(EnvXCStore(path), device="cpu", backend="torch")
    out0 = bank.prepare(**base, arms=("d2",), topology="python")()["d2"].numpy()
    perm = [2, 0, 1]
    inverse = np.argsort(perm)
    moved = dict(base)
    moved["symbols"] = [base["symbols"][p] for p in perm]
    moved["positions_bohr"] = base["positions_bohr"][perm] + np.array([1.7, -2.2, 0.9])
    moved["edge_index"] = inverse[base["edge_index"]]
    out1 = bank.prepare(**moved, arms=("d2",), topology="python")()["d2"].numpy()
    assert np.max(np.abs(out0 - out1)) < 1e-9


def test_pure_d2_loads_no_density_moment_tables(root):
    path, _, _ = root
    store = EnvXCStore(path)
    cut = {s: store.orbital_cutoff(s) for s in store.species}
    g = make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.diag([6.5, 6.5, 6.5]), cut)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    bank.prepare(**g, arms=("d2",), topology="python")()
    assert not any(k.startswith("rhofac") or k.startswith("pairmom") for k in bank.tables.keys())
    bank2 = EnvXCBank(store, device="cpu", backend="torch")
    bank2.prepare(**g, arms=("mcweda",), topology="python")(edge_overlap_ao=torch.zeros((g["edge_index"].shape[1], 9, 9)))
    assert not any(k.startswith("xcbg") for k in bank2.tables.keys())


def test_python_topology_reverse_and_queries(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.diag([6.5, 6.5, 6.5]), cut)
    topo = python_edge_topology(g["positions_bohr"], g["cell_bohr"], g["pbc"], [cut[s] for s in g["symbols"]],
                                [store.q_cutoff(s) for s in g["symbols"]], g["edge_index"], g["edge_cell_shift"])
    assert np.array_equal(topo["reverse"][topo["reverse"]], np.arange(g["edge_index"].shape[1]))
    assert (topo["terms"][:, 0] <= topo["reverse"][topo["terms"][:, 0]]).all()
    q = topo["queries"]
    assert not np.any((q[:, 0] == q[:, 1]) & ~np.any(q[:, 2:], axis=1))   # zero-image self centre never queried


def test_moment_arms_are_bounded_and_the_density_floor_falls_back_to_d2(root, bonded):
    path, _, _ = root
    g, S = bonded
    bank = EnvXCBank(EnvXCStore(path), device="cpu", backend="torch")
    diag = bank.prepare(**g, arms=("d2_moment", "mcweda"), topology="python")(edge_overlap_ao=torch.tensor(S))["diagnostics"]
    assert diag["moment_term_abs_max_eV"] < 1.0 and diag["mcweda_term_abs_max_eV"] < 1.0     # bounded by the XC scale
    # a floor above every reference density removes the moment term everywhere: the arm is the D2 base exactly
    gated = bank.prepare(**g, arms=("d2_moment",), topology="python", moment_density_floor=1e12)(edge_overlap_ao=torch.tensor(S))
    assert gated["diagnostics"]["moment_below_density_floor_elements"] > 0
    plain = bank.prepare(**g, arms=("d2",), topology="python")()
    assert torch.allclose(gated["d2_moment"], plain["d2"], atol=1e-12)


def test_bank_epsilon_follows_dtype_migration(root, bonded):
    """Projector weights cached before ``.to()`` migrate with the bank, and later plans compute in the new dtype."""
    path, _, _ = root
    store = EnvXCStore(path)
    bank = EnvXCBank(store, device="cpu", dtype=torch.float32, backend="torch")
    before = bank.epsilon("Xa")
    assert before.dtype == torch.float32 and "epsilons.epsilon_Xa" in dict(bank.named_buffers())
    bank.to(dtype=torch.float64)
    after = bank.epsilon("Xa")
    assert bank.dtype == torch.float64 and after.dtype == torch.float64
    assert np.allclose(after.numpy(), store.epsilon("Xa")) and after.numel() == store.species["Xa"]["q_norb"]
    out = bank.prepare(**bonded[0], arms=("d2",), topology="python")()
    assert out["d2"].dtype == torch.float64 and torch.isfinite(out["d2"]).all()
    assert all(b.dtype == torch.float64 for b in bank.buffers() if b.is_floating_point())


# --------------------------------------------------------------------------- residual projections and McWEDA versions
def _block_rotation(rng, shells):
    out = np.zeros((sum(2 * l + 1 for l in shells),) * 2)
    o = 0
    for l in shells:
        q, _ = np.linalg.qr(rng.normal(size=(2 * l + 1, 2 * l + 1)))
        out[o:o + 2 * l + 1, o:o + 2 * l + 1] = q
        o += 2 * l + 1
    return torch.tensor(out)


def test_residual_rescale_is_covariant_and_bounds_the_low_density_product():
    """Shell-block Frobenius rescaling commutes with orthogonal shell rotations (elementwise clipping does not),
    and it caps the v' product in the low-density regime (rho_tot ~ 1e-15, moment error ~ 1e-6)."""
    rng = np.random.default_rng(7)
    shells_a, shells_b = (0, 1, 2), (0, 1)
    ia = np.repeat(np.arange(len(shells_a)), [2 * l + 1 for l in shells_a])
    ib = np.repeat(np.arange(len(shells_b)), [2 * l + 1 for l in shells_b])
    kappa = shell_kappa(shells_a, shells_b)
    M = torch.tensor(rng.normal(size=(3, len(ia), len(ib))))
    bound = torch.tensor(np.abs(rng.normal(size=(3, len(shells_a), len(shells_b)))) * 0.8)
    Ra, Rb = _block_rotation(rng, shells_a), _block_rotation(rng, shells_b)
    scaled, count, removed = residual_rescale(M, bound, torch.tensor(ia), torch.tensor(ib))
    rotated_then_scaled, count_r, removed_r = residual_rescale(Ra @ M @ Rb.T, bound, torch.tensor(ia), torch.tensor(ib))
    assert count > 0 and count == count_r and abs(removed - removed_r) < 1e-10
    assert torch.allclose(rotated_then_scaled, Ra @ scaled @ Rb.T, atol=1e-12)
    # elementwise clipping of a p block is frame dependent; the Frobenius projection is not
    t = 0.3
    D = torch.zeros((1, 3, 3))
    D[0, 0, 0] = 2 * t
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert not torch.allclose(R @ torch.clamp(D, -t, t) @ R.T, torch.clamp(R @ D @ R.T, -t, t), atol=1e-6)
    ip = torch.zeros(3, dtype=torch.long)
    b1 = torch.tensor([[[t]]])
    assert torch.allclose(R @ residual_rescale(D, b1, ip, ip)[0] @ R.T, residual_rescale(R @ D @ R.T, b1, ip, ip)[0], atol=1e-12)
    s_np, c_np, r_np = residual_rescale(M.numpy(), bound.numpy(), ia, ib)
    assert np.allclose(s_np, scaled.numpy()) and c_np == count and abs(r_np - removed) < 1e-10
    # low density: rho_tot 1e-15, finite-rank residual 1e-6 bohr^-3, weak overlap Sw 1e-4
    sw, rho_pair, b = 1e-4, 1e-15, 1e-15
    N = torch.full((1, len(shells_a), len(shells_b)), b * sw)
    residual = torch.full((1, len(ia), len(ib)), 1e-6)
    _, dv = lda_pz81_v_dv_torch(torch.tensor([rho_pair + b]))
    assert abs(float(dv)) > 1e10
    unbounded = (dv * residual).abs().max().item()
    bounded_res, n_res, _ = residual_rescale(residual, 2.0 * torch.tensor(kappa)[None] * N, torch.tensor(ia), torch.tensor(ib))
    bounded = (dv * bounded_res).abs().max().item()
    analytic = 2.0 * float(kappa.max()) * sw * abs(float((rho_pair + b) * dv))
    assert unbounded > 1e3 and n_res == kappa.size
    assert bounded <= analytic * (1 + 1e-12) and bounded < 1e-6
    # inside the admissible range nothing changes
    small = 0.1 * residual
    kept, n_kept, _ = residual_rescale(small, 2.0 * torch.tensor(kappa)[None] * torch.full_like(N, 1.0), torch.tensor(ia), torch.tensor(ib))
    assert torch.equal(kept, small) and n_kept == 0


def _mcweda_remainder(N, version, *, Sw=1e-2, P=1e-7, rotation=None):
    """McWEDA remainder of one p-p shell pair (kappa = 3) at pair envelope moment P and environment moment N.

    The environment AO moment carries a fixed finite-rank leak D_env that a positive N resolves; N <= 0 is the
    unresolved branch (no environment moment). The pair residual D_pair - rho_p S is zero, so the whole
    remainder is the environment term.
    """
    R = torch.eye(3) if rotation is None else torch.as_tensor(rotation)
    ia = ib = torch.zeros(3, dtype=torch.long)
    kappa = torch.tensor([[3.0]])
    S = (R @ (torch.eye(3) * Sw) @ R.T)[None]
    rho_p, b = P / Sw, max(N, 0.0) / Sw
    leak = (R @ torch.diag(torch.tensor([1.0, -1.0, 0.0])) @ R.T)[None] * 1e-7
    D_env = leak if N > 0 else torch.zeros_like(leak)
    N_shell = torch.full((1, 1, 1), max(N, 0.0))
    M_env, _, _ = residual_rescale(D_env - b * S, 2.0 * kappa * N_shell, ia, ib)
    rp, rt = torch.full((1, 3, 3), rho_p), torch.full((1, 3, 3), rho_p + b)
    v_t, dv_t = lda_pz81_v_dv_torch(rt)
    v_p, dv_p = lda_pz81_v_dv_torch(rp)
    term, _ = mcweda_composition(version, S, D_env, rho_p * S, M_env, rp, rt, N_shell, torch.full((1, 1, 1), P),
                                 kappa, ia, ib, v_t, dv_t, v_p, dv_p)
    return term[0]


def test_v2_is_continuous_in_the_weak_environment_limit_where_v1_is_not():
    """At fixed pair moment P a finite-rank leak of D_env survives the v1 total projection (bound 2 kappa P), so the
    v1 remainder stays at about 2.66 meV as N -> 0+ although the unresolved branch is exactly zero; v2 decays as O(N)."""
    rows = {N: {version: float(_mcweda_remainder(N, version).abs().max()) for version in STABILIZATIONS}
            for N in (1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 0.0, -1e-24)}
    assert rows[1e-20]["v1"] > 1e-3 and rows[1e-24]["v1"] > 1e-3
    for N in (0.0, -1e-24):
        assert rows[N]["v1"] == 0.0 and rows[N]["v2"] == 0.0
    # O(N) decay: a stable ratio of about 1.4e5 eV Bohr^3 (v'_t times the 2 kappa N envelope bound)
    ratios = [rows[N]["v2"] / N for N in (1e-12, 1e-16, 1e-20, 1e-24)]
    assert all(r < 1e6 for r in ratios) and max(ratios) < 3 * min(ratios), ratios
    assert rows[1e-12]["v2"] < 1e-6 and rows[1e-20]["v2"] < 1e-12
    values = [rows[N]["v2"] for N in (1e-8, 1e-12, 1e-16, 1e-20, 1e-24)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_mcweda_composition_is_shell_covariant_and_versions_agree_without_projection():
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = torch.as_tensor(Q)
    for version in STABILIZATIONS:
        for N in (1e-8, 1e-11):
            a = _mcweda_remainder(N, version)
            b = _mcweda_remainder(N, version, rotation=Q)
            assert a.abs().max() > 0
            assert torch.allclose(R @ a @ R.T, b, rtol=1e-10, atol=1e-10 * float(a.abs().max()))
    # random blocks, NumPy and torch, with active projections
    ia = np.repeat(np.arange(2), [1, 3])
    ib = np.repeat(np.arange(2), [3, 1])
    kappa = np.sqrt(np.array([[1, 3], [3, 9]], dtype=float) * np.array([[3, 1], [9, 3]], dtype=float))
    S, De, Dp, M_env = (rng.normal(size=(2, 4, 4)) * 1e-2 for _ in range(4))
    rho_p = np.abs(rng.normal(size=(2, 4, 4))) * 1e-3 + 1e-4
    b = np.abs(rng.normal(size=(2, 4, 4))) * 1e-3 + 1e-4
    Nsh, Psh = np.abs(rng.normal(size=(2, 2, 2))) * 1e-4, np.abs(rng.normal(size=(2, 2, 2))) * 1e-4
    v = [rng.normal(size=(2, 4, 4)) for _ in range(4)]
    for version in STABILIZATIONS:
        term_np, counters = mcweda_composition(version, S, De, Dp, M_env, rho_p, rho_p + b, Nsh, Psh, kappa, ia, ib, *v)
        term_t, _ = mcweda_composition(version, *(torch.as_tensor(x) for x in (S, De, Dp, M_env, rho_p, rho_p + b, Nsh, Psh, kappa)),
                                       torch.as_tensor(ia), torch.as_tensor(ib), *(torch.as_tensor(x) for x in v))
        assert np.allclose(term_np, term_t.numpy(), atol=1e-14)
        assert counters["mcweda_pair_rescaled_shell_pairs"] > 0
        assert (counters["mcweda_total_rescaled_shell_pairs"] > 0) == (version == "v1")
    # without any active projection the two versions are one formula:
    # D_pair + D_env - rho_t S = (D_pair - rho_p S) + (D_env - b S)
    loose = np.full_like(Nsh, 1e6)
    same = [mcweda_composition(version, S, De, Dp, De - b * S, rho_p, rho_p + b, loose, loose, kappa, ia, ib, *v)
            for version in STABILIZATIONS]
    assert all(c["mcweda_pair_rescaled_shell_pairs"] == 0 and c["mcweda_total_rescaled_shell_pairs"] == 0 for _, c in same)
    assert np.allclose(same[0][0], same[1][0], rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        mcweda_composition("v3", S, De, Dp, M_env, rho_p, rho_p + b, Nsh, Psh, kappa, ia, ib, *v)


# --------------------------------------------------------------------------- build tool: fail-closed identity
BASE_ARGS = ["--kinds", "envnorm", "envfac", "--radial-rank", "1", "--l-buffer", "0", "--tail-seeds", "0",
             "--density-threshold", "1e-5", "--distance-step", "2.0", "--order", "8", "--workers", "1"]


def write_sources(directory, decay_scale=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    for symbol, kw in SPECIES.items():
        synthetic_source(symbol, **{**kw, "decay": kw["decay"] * decay_scale}).save(directory / f"{symbol}.npz")
    structures = directory / "structures.json"
    structures.write_text(json.dumps([sorted(SPECIES)]))
    return structures


def snapshot(path):
    return {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}


def build(sources, structures, output, *extra, base=BASE_ARGS):
    return cli.main(["--sources", str(sources), "--structures", str(structures), "--output", str(output), *base, *extra])


@pytest.fixture()
def built(tmp_path):
    sources = tmp_path / "sources"
    structures = write_sources(sources)
    output = tmp_path / "root"
    assert build(sources, structures, output) == 0
    return sources, structures, output


def test_build_writes_identity_and_same_identity_resume_reuses_artifacts(built):
    sources, structures, output = built
    identity = json.loads((output / cli.IDENTITY_FILE).read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["build_identity"] == identity["build_identity"]
    assert all(row["identity"]["build_identity"] == identity["build_identity"] for row in manifest["species"].values())
    before = snapshot(output)
    lines_before = (output / "progress.jsonl").read_text().splitlines()
    assert build(sources, structures, output, "--resume") == 0
    after = snapshot(output)
    assert {k for k in before if before[k] != after.get(k)} <= {"manifest.json"}   # only its timing fields may differ
    assert (output / "progress.jsonl").read_text().splitlines() == lines_before
    store = EnvXCStore(output)
    assert store.table("envnorm", "Xa", "Yb").values.shape[0] > 2
    assert store.epsilon("Xa").size > 0


# each case applies its change and returns the builds that must be refused
def _changed_setting(option, value):
    def change(sources, structures, output, monkeypatch):
        args = list(BASE_ARGS)
        args[args.index(option) + 1] = value
        return [lambda: build(sources, structures, output, "--resume", base=args)]
    return change


def _changed_code_identity(sources, structures, output, monkeypatch):
    original = cli.sha256_file
    monkeypatch.setattr(cli, "sha256_file",
                        lambda p: "0" * 64 if str(p).endswith("build_nacf_envxc_tables.py") else original(p))
    return [lambda: build(sources, structures, output, "--resume")]


def _changed_source(sources, structures, output, monkeypatch):
    write_sources(sources, decay_scale=1.1)                   # same species names, different atomic data
    return [lambda: build(sources, structures, output, "--resume")]


def _corrupted_table(sources, structures, output, monkeypatch):
    table = next(output.glob("envnorm/*.npz"))
    table.write_bytes(table.read_bytes() + b"corrupt")
    return [lambda: build(sources, structures, output, "--resume")]


def _foreign_species(sources, structures, output, monkeypatch):
    proj = build_density_projectors(AtomicSource.load(sources / "Xa.npz"), radial_rank=2, l_buffer=0, tail_seeds=0,
                                    density_threshold=1e-5)
    save_species(output / "species" / "Xa.npz", proj, identity={"build_identity": "0" * 64, "sources": {}})
    return [lambda: build(sources, structures, output, "--resume")]


def _unidentified_root(sources, structures, output, monkeypatch):
    (output / cli.IDENTITY_FILE).unlink()
    return [lambda: build(sources, structures, output, "--resume"), lambda: build(sources, structures, output)]


def _identified_root_without_resume(sources, structures, output, monkeypatch):
    return [lambda: build(sources, structures, output)]


@pytest.mark.parametrize("change", [
    pytest.param(_changed_setting("--radial-rank", "2"), id="radial_rank"),
    pytest.param(_changed_setting("--order", "12"), id="order"),
    pytest.param(_changed_setting("--distance-step", "1.0"), id="distance_step"),
    pytest.param(_changed_setting("--density-threshold", "1e-6"), id="density_threshold"),
    pytest.param(_changed_setting("--l-buffer", "1"), id="l_buffer"),
    pytest.param(_changed_code_identity, id="code_identity"),
    pytest.param(_changed_source, id="source"),
    pytest.param(_corrupted_table, id="corrupted_table"),
    pytest.param(_foreign_species, id="foreign_species"),
    pytest.param(_unidentified_root, id="unidentified_root"),
    pytest.param(_identified_root_without_resume, id="identified_root_without_resume"),
])
def test_mismatched_build_exits_2_without_touching_the_root(built, monkeypatch, change):
    sources, structures, output = built
    refused = change(sources, structures, output, monkeypatch)
    before = snapshot(output)
    for attempt in refused:
        assert attempt() == 2
    assert snapshot(output) == before


@pytest.mark.parametrize("tamper", ["foreign_identity", "missing_identity"])
def test_store_rejects_artifacts_without_the_manifest_build_identity(built, tamper):
    _, _, output = built
    manifest = json.loads((output / "manifest.json").read_text())
    if tamper == "missing_identity":
        manifest.pop("build_identity")
        (output / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            EnvXCStore(output)
        return
    manifest["build_identity"] = "f" * 64
    (output / "manifest.json").write_text(json.dumps(manifest))
    store = EnvXCStore(output)
    with pytest.raises(ValueError):
        store.table("envnorm", "Xa", "Yb")
    with pytest.raises(ValueError):
        store.epsilon("Xa")


def test_coverage_extension_with_same_identity_is_allowed(built, tmp_path):
    sources, _, output = built
    extended = tmp_path / "structures_ext.json"
    extended.write_text(json.dumps([["Xa"], ["Yb"], ["Xa", "Yb"]]))
    before = snapshot(output)
    assert build(sources, extended, output, "--resume") == 0
    after = snapshot(output)
    assert all(after[k] == v for k, v in before.items() if k not in ("manifest.json", "progress.jsonl"))
    assert json.loads((output / "manifest.json").read_text())["complete"]
