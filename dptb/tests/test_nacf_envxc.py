"""CPU tests for the grid-free environment-XC tables and plan (synthetic species).

Written by Claude Fable 5.1, 2026-09-20. Everything runs on CPU with the pure-Python topology;
the only spatial quadrature is the independent three-centre reference used to test the
finite-rank density expansion.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import roots_legendre

from dptb.data.interfaces.p2_table import abacus_m_order, real_sph_abacus
from dptb.nacf.envxc import (EnvXCBank, EnvXCStore, lda_pz81_v_dv_torch, python_edge_topology,
                             reference_edge_envxc)
from dptb.nacf.envxc_tables import (ENVXC_SCHEMA, AtomicSource, build_density_projectors, build_identity,
                                    build_table_values, distance_grid, lda_pz81_v_dv, save_species, save_table,
                                    sha256_file, sha256_json, table_filename, table_key, two_centre_block)

torch.set_default_dtype(torch.float64)
BACKGROUND_NODES = [0.0, 1e-3, 4e-3, 1.6e-2, 6.4e-2, 0.256]


# --------------------------------------------------------------------------- synthetic species
def synthetic_source(symbol, shells, rcut, z_valence, decay, nlcc=True):
    r_orb = np.linspace(0.0, rcut, 401)
    radial = []
    seen = {}
    for l in shells:
        n = seen.get(l, 0)
        seen[l] = n + 1
        alpha = 0.35 * (1.6 ** n) * (1 + 0.3 * l)
        f = r_orb ** l * np.exp(-alpha * r_orb**2) * np.clip(1 - (r_orb / rcut) ** 2, 0, None) ** 2
        if n:
            f = f * (1 - 0.7 * alpha * r_orb**2)
        f /= math.sqrt(np.trapz(f * f * r_orb**2, r_orb))
        radial.append(f)
    r_rho = np.linspace(0.0, 14.0, 1401)
    val = np.exp(-decay * r_rho**2) + 0.15 * np.exp(-0.25 * decay * r_rho**2)
    val *= z_valence / np.trapz(4 * np.pi * r_rho**2 * val, r_rho)
    core = 0.6 * np.exp(-6.0 * r_rho**2) if nlcc else np.zeros_like(r_rho)
    return AtomicSource(symbol=symbol, r_orb=r_orb, radial=np.array(radial), shells=tuple(shells),
                        orbital_cutoff_bohr=rcut, r_rho=r_rho, rho_val=val, rho_nlcc=core, z_valence=z_valence,
                        identity={"synthetic": True, "symbol": symbol})


SPECIES = {
    "Xa": dict(shells=(0, 0, 1), rcut=5.0, z_valence=3.0, decay=0.55, nlcc=True),
    "Yb": dict(shells=(0, 1, 2), rcut=5.5, z_valence=4.0, decay=0.45, nlcc=False),
}


def build_root(root: Path, *, radial_rank=3, l_buffer=2, tail_seeds=1, step=0.3, order=40):
    sources = {s: synthetic_source(s, **kw) for s, kw in SPECIES.items()}
    proj = {s: build_density_projectors(src, radial_rank=radial_rank, l_buffer=l_buffer, tail_seeds=tail_seeds,
                                        density_threshold=1e-7) for s, src in sources.items()}
    # immutable build identity of the fixture (synthetic sources are identified by their generating parameters)
    settings = {"radial_rank": radial_rank, "l_buffer": l_buffer, "tail_seeds": tail_seeds, "density_threshold": 1e-7,
                "distance_step": step, "order": order, "background_nodes": BACKGROUND_NODES}
    code_identity = {"fixture": "test_nacf_envxc.build_root"}
    build_id = build_identity(settings, code_identity)
    source_sha = {s: sha256_json({"symbol": s, **SPECIES[s]}) for s in sources}
    def identity(kind, key, index=None, *symbols):
        return {"schema": ENVXC_SCHEMA, "kind": kind, "key": key, "index": index, "build_identity": build_id,
                "sources": {x: source_sha[x] for x in symbols}}
    species_rows, tables = {}, {k: {} for k in ("rhofac", "envfac", "envnorm", "envpair", "pairmom", "xcbg")}
    for s, p in proj.items():
        p.metadata["symbol"] = s
        path = root / "species" / f"{s}.npz"
        save_species(path, p, identity=identity("species", s, None, s))
        src = sources[s]
        species_rows[s] = {"array_path": str(path.relative_to(root)), "array_sha256": sha256_file(path),
                           "q_shells": [int(l) for l in p.q_l], "q_norb": int(p.norb), "q_cutoff_bohr": float(p.cutoff_bohr),
                           "orbital_shells": list(src.shells), "orbital_norb": int(src.norb),
                           "orbital_cutoff_bohr": float(src.orbital_cutoff_bohr), "z_valence": src.z_valence, "metadata": p.metadata}
    symbols = sorted(sources)
    for k in symbols:
        for a in symbols:
            for kind in ("rhofac", "envfac"):
                key = table_key(kind, k, a)
                support = proj[k].cutoff_bohr + sources[a].orbital_cutoff_bohr
                table = build_table_values(kind, proj[k], sources[a], distances=distance_grid(support, step), order=order)
                path = root / table_filename(kind, key)
                tables[kind][key] = {"path": str(path.relative_to(root)), "sha256": save_table(path, table, identity=identity(kind, key, None, k, a)),
                                     "left_shells": list(table["left_shells"]), "right_shells": list(table["right_shells"]),
                                     "support_bohr": table["support_bohr"]}
    for a in symbols:
        for b in symbols:
            if a > b:
                continue
            support = sources[a].orbital_cutoff_bohr + sources[b].orbital_cutoff_bohr
            distances = distance_grid(support, step)
            for kind in ("envnorm", "envpair", "pairmom"):
                key = table_key(kind, a, b)
                table = build_table_values(kind, sources[a], sources[b], distances=distances, order=order)
                path = root / table_filename(kind, key)
                tables[kind][key] = {"path": str(path.relative_to(root)), "sha256": save_table(path, table, identity=identity(kind, key, None, a, b)),
                                     "left_shells": list(table["left_shells"]), "right_shells": list(table["right_shells"]),
                                     "support_bohr": table["support_bohr"]}
            for index, bg in enumerate(BACKGROUND_NODES):
                key = table_key("xcbg", a, b, index)
                table = build_table_values("xcbg", sources[a], sources[b], distances=distances, order=order, background=bg)
                path = root / table_filename("xcbg", key)
                tables["xcbg"][key] = {"path": str(path.relative_to(root)), "sha256": save_table(path, table, identity=identity("xcbg", key, index, a, b), background_bohr_minus3=bg),
                                       "left_shells": list(table["left_shells"]), "right_shells": list(table["right_shells"]),
                                       "support_bohr": table["support_bohr"]}
    manifest = {"schema": ENVXC_SCHEMA, "complete": True, "length_unit": "bohr", "density_unit": "bohr^-3", "xc_energy_unit": "eV",
                "harmonic_convention": "deeptb_abacus_real", "endpoint_policy": "exclude_i0_and_jR", "interpolation": "cubic",
                "background_nodes": BACKGROUND_NODES, "species": species_rows, "tables": tables,
                "build_identity": build_id, "code_identity": code_identity, "settings": settings,
                "sources": {s: {"source_sha256": source_sha[s], "synthetic": True} for s in sources}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return sources, proj


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("envxc")
    sources, proj = build_root(path)
    return path, sources, proj


# --------------------------------------------------------------------------- reference quadratures
def ao_functions(source):
    out = []
    for c, l in enumerate(source.shells):
        f = source.radial_fn(c)
        for m in abacus_m_order(l):
            out.append((f, l, m))
    return out


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
    A = evaluate(left, r_left)
    B = evaluate(right, r_right)
    return A.T @ ((w * rho)[:, None] * B)


def factor_block(store, kind, k, a, b, r_a, r_b, r_k):
    fa = store.table(kind, k, a).evaluate(r_a - r_k)
    fb = store.table(kind, k, b).evaluate(r_b - r_k)
    return fa.T @ (store.epsilon(k)[:, None] * fb)


# --------------------------------------------------------------------------- tests
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


def test_density_projectors_metadata(root):
    _, sources, proj = root
    for s, p in proj.items():
        assert p.norb == sum(2 * l + 1 for l in p.q_l)
        assert np.all(p.epsilon_radial > 0)
        assert p.metadata["normalized_rho_metric_offdiag_max"] < 1e-8
        assert p.cutoff_bohr >= sources[s].orbital_cutoff_bohr   # auxiliary space covers the density support
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
    rel = np.linalg.norm(fact - ref) / np.linalg.norm(ref)
    # regression guard for the fixture configuration (rank 3 / l_buffer 2 / tail 1 measured 4.3% on these synthetic
    # species; the cheap rank 2 / l_buffer 1 / tail 1 measured 10.5%). Real-species accuracy/cost is judged in the
    # concentrated validation, not by this synthetic number.
    assert rel < 0.08, rel
    # mixed sigma-pi channel: the s_A - p_x(B) moment is nonzero for an off-axis centre and reproduced by the expansion
    px = sum(2 * l + 1 for l in sources[B].shells[:1]) + 1          # first p shell of B, m=+1 (x)
    assert abs(ref[0, px]) > 1e-3 * np.max(np.abs(ref))
    assert abs(fact[0, px] - ref[0, px]) < 0.15 * abs(ref[0, px])
    # envelope numerator (D2)
    ref_env = direct_three_centre(envelope_functions(sources[A]), r_a, envelope_functions(sources[B]), r_b, sources[K].density, r_k, rmax)
    fact_env = factor_block(store, "envfac", K, A, B, r_a, r_b, r_k)
    assert np.all(ref_env > 0)
    assert np.linalg.norm(fact_env - ref_env) / np.linalg.norm(ref_env) < 0.08   # measured 1.9% (rank 3); 2.5% at rank 2 / l_buffer 1
    # rank convergence: a richer auxiliary space must be clearly better than the poorest one (measured 27% -> 1.9%)
    rich = build_density_projectors(sources[K], radial_rank=4, l_buffer=3, tail_seeds=2, density_threshold=1e-7)
    poor = build_density_projectors(sources[K], radial_rank=1, l_buffer=0, tail_seeds=0, density_threshold=1e-7)
    def error(p):
        step = 0.3
        fa = build_table_values("rhofac", p, sources[A], distances=distance_grid(p.cutoff_bohr + sources[A].orbital_cutoff_bohr, step), order=40)
        fb = build_table_values("rhofac", p, sources[B], distances=distance_grid(p.cutoff_bohr + sources[B].orbital_cutoff_bohr, step), order=40)
        from dptb.data.interfaces.p2_table import RadialBlockTable
        ta = RadialBlockTable(fa["distances"], fa["values"], fa["left_shells"], fa["right_shells"], fa["support_bohr"])
        tb = RadialBlockTable(fb["distances"], fb["values"], fb["left_shells"], fb["right_shells"], fb["support_bohr"])
        val = ta.evaluate(r_a - r_k).T @ (p.epsilon_ao()[:, None] * tb.evaluate(r_b - r_k))
        return np.linalg.norm(val - ref) / np.linalg.norm(ref)
    assert error(rich) < 0.5 * error(poor)


def make_structure(symbols, positions, cell, cutoffs):
    """All directed edges with |R_j + t cell - R_i| < cut_i + cut_j, including periodic images (reverse-closed)."""
    pos = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    from dptb.nacf.envxc import _images_within
    images = _images_within(cell, (True, True, True), 2 * max(cutoffs.values()), pos)
    edges, shifts = [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            for t in images:
                if i == j and not np.any(t):
                    continue
                d = np.linalg.norm(pos[j] + t @ cell - pos[i])
                if d < cutoffs[symbols[i]] + cutoffs[symbols[j]] - 1e-9:
                    edges.append((i, j)); shifts.append(t)
    return {"symbols": list(symbols), "positions_bohr": pos, "cell_bohr": cell, "edge_index": np.array(edges).T,
            "edge_cell_shift": np.array(shifts, dtype=np.int64), "pbc": (True, True, True)}


def edge_overlap(sources, g):
    pos, cell, ei, sh = g["positions_bohr"], g["cell_bohr"], g["edge_index"], g["edge_cell_shift"]
    w = max(src.norb for src in sources.values())
    out = np.zeros((ei.shape[1], w, w))
    from dptb.data.interfaces.p2_table import RadialBlockTable
    cache = {}
    for e in range(ei.shape[1]):
        si, sj = g["symbols"][ei[0, e]], g["symbols"][ei[1, e]]
        vec = pos[ei[1, e]] + sh[e] @ cell - pos[ei[0, e]]
        a, b = sorted((si, sj))
        if (a, b) not in cache:
            sa, sb = sources[a], sources[b]
            left = [(int(l), sa.radial_fn(c)) for c, l in enumerate(sa.shells)]
            right = [(int(l), sb.radial_fn(c)) for c, l in enumerate(sb.shells)]
            support = sa.orbital_cutoff_bohr + sb.orbital_cutoff_bohr
            dist = distance_grid(support, 0.3)
            vals = np.stack([two_centre_block(left, right, float(d), 40, sa.orbital_cutoff_bohr, sb.orbital_cutoff_bohr) for d in dist])
            vals[dist >= support - 1e-12] = 0
            cache[(a, b)] = RadialBlockTable(dist, vals, sa.shells, sb.shells, support)
        block = cache[(a, b)].evaluate(vec) if (si, sj) == (a, b) else cache[(a, b)].evaluate(-vec).T
        out[e, :block.shape[0], :block.shape[1]] = block
    return out


def test_plan_matches_numpy_reference_all_arms(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    S = edge_overlap(sources, g)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    plan = bank.prepare(**g, arms=("d2", "d2_moment", "mcweda"), topology="python", profile=True)
    out = plan(edge_overlap_ao=torch.tensor(S))
    ref = reference_edge_envxc(store, g, arms=("d2", "d2_moment", "mcweda"), edge_overlap_ao=S)
    for arm in ("d2", "d2_moment", "mcweda"):
        got = out[arm].numpy()
        assert np.isfinite(got).all()
        assert np.max(np.abs(got - ref[arm])) < 1e-9 * max(1.0, np.max(np.abs(ref[arm]))), arm
        assert np.max(np.abs(ref[arm])) > 1e-6, f"{arm} correction vanished on a bonded periodic structure"
    diag = out["diagnostics"]
    assert np.allclose(diag["b_shell"].numpy(), ref["b_shell"], atol=1e-10)
    # finite-rank leakage: the factorized numerator is not exactly zero (or positive) for weakly / non-overlapping
    # envelopes. Non-overlapping pairs are gated (correction zero) and their pair-XC elements must be negligible;
    # negative numerators may only occur in the weak-overlap regime (normalized envelope overlap < 1e-3).
    assert diag["numerator_without_overlap"] >= 0
    assert diag["gated_pair_xc_abs_max_eV"] < 1e-6
    # regression guard on the synthetic fixture (measured: negatives only up to normalized overlap ~3e-3), not a physical gate
    assert diag["negative_numerators"] == 0 or diag["negative_numerator_max_overlap"] < 0.05
    # reverse edges are exact transposes; the numpy reference agrees to rounding
    rev = plan.reverse.numpy()
    for arm in ("d2", "d2_moment", "mcweda"):
        assert torch.equal(out[arm], out[arm][plan.reverse].transpose(-1, -2))
        assert np.max(np.abs(ref[arm] - ref[arm][rev].transpose(0, 2, 1))) < 1e-8


def test_zero_environment_limit(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.diag([60.0, 60.0, 60.0]), cut)
    assert g["edge_index"].shape[1] == 2
    S = edge_overlap(sources, g)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    out = bank.prepare(**g, arms=("d2", "d2_moment", "mcweda"), topology="python")(edge_overlap_ao=torch.tensor(S))
    for arm in ("d2", "d2_moment", "mcweda"):
        assert torch.equal(out[arm], torch.zeros_like(out[arm]))
    assert torch.equal(out["diagnostics"]["b_shell"], torch.zeros_like(out["diagnostics"]["b_shell"]))
    assert out["diagnostics"]["terms"] == 0


def test_translation_and_permutation_invariance(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    base = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    bank = EnvXCBank(store, device="cpu", backend="torch")
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
    bank = EnvXCBank(store, device="cpu", backend="torch")
    cut = {s: store.orbital_cutoff(s) for s in store.species}
    g = make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.diag([6.5, 6.5, 6.5]), cut)
    bank.prepare(**g, arms=("d2",), topology="python")()
    assert not any(k.startswith("rhofac") or k.startswith("pairmom") for k in bank.tables.keys())
    bank2 = EnvXCBank(store, device="cpu", backend="torch")
    bank2.prepare(**g, arms=("mcweda",), topology="python")(edge_overlap_ao=torch.zeros((g["edge_index"].shape[1], 9, 9)))
    assert not any(k.startswith("xcbg") for k in bank2.tables.keys())


def test_python_topology_matches_brute_force_counts(root):
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


def test_envelope_normalization_and_shell_frobenius_bound_on_exact_tables():
    """Stored envelope moments carry Y00^2 = 1/(4 pi): the s-s same-channel moment at d=0 equals the envelope
    moment exactly, and every shell block obeys ||D_ab||_F <= sqrt(n_a n_b) P_ab on the exact two-centre tables."""
    from dptb.nacf.envxc import shell_kappa
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
            # d = 0, same s channel: |R|^2 = R^2 so the signed and the envelope moments coincide (tests the 4 pi convention)
            for c, l in enumerate(a.shells):
                if l == 0:
                    assert abs(dens[0, oa[c], oa[c]] - env[0, c, c]) < 1e-12 * max(1.0, abs(env[0, c, c]))
            assert env[0, 0, 0] > 1e-3      # non-trivial magnitude, so a missing 4 pi would have been caught


def test_residual_rescale_is_covariant_and_bounds_the_low_density_product():
    """Shell-block Frobenius rescaling commutes with orthogonal shell rotations (elementwise clipping does not),
    and it caps the v' product in the replayed counterexample regime (rho_tot ~ 1e-15, moment error ~ 1e-6)."""
    from dptb.nacf.envxc import residual_rescale, shell_kappa
    rng = np.random.default_rng(7)
    shells_a, shells_b = (0, 1, 2), (0, 1)
    ia = np.repeat(np.arange(len(shells_a)), [2 * l + 1 for l in shells_a])
    ib = np.repeat(np.arange(len(shells_b)), [2 * l + 1 for l in shells_b])
    kappa = shell_kappa(shells_a, shells_b)
    E = 3
    M = torch.tensor(rng.normal(size=(E, len(ia), len(ib))))
    bound = torch.tensor(np.abs(rng.normal(size=(E, len(shells_a), len(shells_b)))) * 0.8)
    def block_rotation(shells):
        blocks = []
        for l in shells:
            q, _ = np.linalg.qr(rng.normal(size=(2 * l + 1, 2 * l + 1)))
            blocks.append(q)
        out = np.zeros((sum(2 * l + 1 for l in shells),) * 2)
        o = 0
        for q in blocks:
            n = q.shape[0]; out[o:o + n, o:o + n] = q; o += n
        return torch.tensor(out)
    Ra, Rb = block_rotation(shells_a), block_rotation(shells_b)
    scaled, count, removed = residual_rescale(M, bound, torch.tensor(ia), torch.tensor(ib))
    rotated_then_scaled, count_r, removed_r = residual_rescale(Ra @ M @ Rb.T, bound, torch.tensor(ia), torch.tensor(ib))
    assert count > 0 and count == count_r and abs(removed - removed_r) < 1e-10
    assert torch.allclose(rotated_then_scaled, Ra @ scaled @ Rb.T, atol=1e-12)
    # the reviewer's counterexample: elementwise clipping of a p block is frame dependent
    t = 0.3
    D = torch.zeros((1, 3, 3)); D[0, 0, 0] = 2 * t
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    clip_then_rotate = R @ torch.clamp(D, -t, t) @ R.T
    rotate_then_clip = torch.clamp(R @ D @ R.T, -t, t)
    assert not torch.allclose(clip_then_rotate, rotate_then_clip, atol=1e-6)
    ip = torch.zeros(3, dtype=torch.long)
    b1 = torch.tensor([[[t]]])
    fro_a = R @ residual_rescale(D, b1, ip, ip)[0] @ R.T
    fro_b = residual_rescale(R @ D @ R.T, b1, ip, ip)[0]
    assert torch.allclose(fro_a, fro_b, atol=1e-12)
    # NumPy twin
    s_np, c_np, r_np = residual_rescale(M.numpy(), bound.numpy(), ia, ib)
    assert np.allclose(s_np, scaled.numpy()) and c_np == count and abs(r_np - removed) < 1e-10
    # low-density replay: rho_tot 1e-15 (measured 1.9e-15), finite-rank residual 1e-6 bohr^-3, weak overlap Sw 1e-4
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


def test_moment_arms_report_stabilization_diagnostics(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    S = edge_overlap(sources, g)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    out = bank.prepare(**g, arms=("d2_moment", "mcweda"), topology="python")(edge_overlap_ao=torch.tensor(S))
    diag = out["diagnostics"]
    for key in ("moment_env_rescaled_shell_pairs", "moment_env_removed_frobenius_bohr3", "mcweda_total_rescaled_shell_pairs",
                "mcweda_pair_rescaled_shell_pairs", "moment_unresolved_env_shell_pairs", "environment_dominated_shell_pairs",
                "moment_term_abs_max_eV", "mcweda_term_abs_max_eV", "moment_density_floor"):
        assert key in diag
    assert diag["moment_term_abs_max_eV"] < 1.0 and diag["mcweda_term_abs_max_eV"] < 1.0     # bounded by the XC scale
    # the optional explicit floor removes the term where rho_tot is below it and counts what it removed; with a floor
    # above every reference density (including leakage-inflated backgrounds) the arm falls back to the D2 base exactly
    gated = bank.prepare(**g, arms=("d2_moment",), topology="python", moment_density_floor=1e12)(edge_overlap_ao=torch.tensor(S))
    assert gated["diagnostics"]["moment_below_density_floor_elements"] > 0
    plain = bank.prepare(**g, arms=("d2",), topology="python")()
    assert torch.allclose(gated["d2_moment"], plain["d2"], atol=1e-12)


def test_bank_epsilon_follows_dtype_migration(root):
    """Reviewer reproduction (F2): projector weights cached before ``.to()`` must migrate with the bank."""
    path, _, _ = root
    store = EnvXCStore(path)
    bank = EnvXCBank(store, device="cpu", dtype=torch.float32, backend="torch")
    before = bank.epsilon("Xa")
    assert before.dtype == torch.float32 and "epsilons.epsilon_Xa" in dict(bank.named_buffers())
    bank.to(dtype=torch.float64)
    after = bank.epsilon("Xa")
    assert bank.dtype == torch.float64 and after.dtype == torch.float64
    assert np.allclose(after.numpy(), store.epsilon("Xa")) and after.numel() == store.species["Xa"]["q_norb"]
    # the compiled tables moved as well, and a plan built afterwards computes in the migrated dtype
    cut = {s: store.orbital_cutoff(s) for s in store.species}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    out = bank.prepare(**g, arms=("d2",), topology="python")()
    assert out["d2"].dtype == torch.float64 and torch.isfinite(out["d2"]).all()
    assert all(b.dtype == torch.float64 for b in bank.buffers() if b.is_floating_point())
