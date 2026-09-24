"""Synthetic NACF inputs shared by the NACF test modules (not collected; import from dptb.tests.nacf_support)."""
import itertools
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable
from dptb.nacf.envxc_tables import (ENVXC_SCHEMA, AtomicSource, build_density_projectors, build_identity,
                                    build_table_values, distance_grid, save_species, save_table, sha256_file,
                                    sha256_json, table_filename, table_key, two_centre_block)

RY_TO_EV = 13.605698


@contextmanager
def float64_default():
    """Compute in float64 inside the block; the previous default dtype is restored afterwards."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# --------------------------------------------------------------------------- P2/P23 stores
def stores():
    """One-AO synthetic P2 and P23 stores for species X and Y (s orbitals, s projectors, quadratic radial decay)."""
    species = {s: {'orbital_norb': 1, 'orbital_cutoff_bohr': 2., 'orbital_shells': [0],
                   'projector_norb': 1, 'projector_max_cutoff_bohr': 1.,
                   'projector_shells': [0], 'projector_cutoffs_bohr': [1.],
                   'vna_cutoff_bohr': 1., 'vna_projector_norb': 1}
               for s in ('X', 'Y')}

    def radial(value, support):
        r = np.linspace(0., support, 31)
        return RadialBlockTable(r, (value * (1 - r / support) ** 2)[:, None, None], (0,), (0,), support)

    p2 = SimpleNamespace(species=species)
    p2.onsite_component = lambda s, k: np.array([[2. if k == 'p2_base' else 1.]])
    p2.base_component = lambda a, b, k: radial(1. if k == 'p2_base' else .4, 4.)
    p2.projector = lambda a, b: radial(.2 if a == 'X' else .3, 3.)
    p2.d_eff = lambda s: np.array([[2. if s == 'X' else 3.]])
    p23 = SimpleNamespace(species=species, manifest_sha256='synthetic')
    p23.has_factor = lambda a, b: True
    p23.factor = lambda a, b: radial(.3 if a == 'X' else .4, 3.)
    p23.epsilon = lambda s: np.array([1.5 if s == 'X' else 2.])
    return p2, p23


def explicit_soc_blocks(p2, symbols, positions, keys, d_spinor):
    """Independent complex contraction for the one-AO stores: scalar P2 base (spin diagonal) plus
    sum_K q_i(K) D_K q_j(K) over every projector centre K, for non-periodic ``keys`` = [(i, j), ...]."""
    blocks = []
    for i, j in keys:
        scalar = (p2.onsite_component(symbols[i], 'p2_base') if i == j else
                  p2.base_component(symbols[i], symbols[j], 'p2_base').evaluate(positions[j] - positions[i]))[0, 0]
        block = np.eye(2, dtype=complex) * scalar
        for k, s in enumerate(symbols):
            qi = p2.projector(s, symbols[i]).evaluate(positions[i] - positions[k])[0, 0]
            qj = p2.projector(s, symbols[j]).evaluate(positions[j] - positions[k])[0, 0]
            block += qi * d_spinor(s) * qj
        blocks.append(block)
    return np.stack(blocks)


# --------------------------------------------------------------------------- onsite densities
def synthetic_density(device, nlcc=True, seed=0):
    """Clamped-cubic valence (and optional NLCC, crossing zero) density on strictly increasing, nonuniform knots."""
    from scipy.interpolate import CubicSpline
    from dptb.nacf.onsite import SplineDensity
    rng = np.random.default_rng(seed)
    knots = np.concatenate(([0.0], np.sort(rng.uniform(0.05, 4.0, 30)), [4.5]))
    channels = [CubicSpline(knots, np.exp(-knots) * (1 + 0.3 * np.sin(3 * knots))).c]
    if nlcc:
        channels.append(CubicSpline(knots, 0.2 * np.exp(-2 * knots) - 0.05).c)
    return SplineDensity(torch.tensor(knots, device=device, dtype=torch.float64),
                         [torch.tensor(c, device=device, dtype=torch.float64) for c in channels])


# --------------------------------------------------------------------------- environment-XC species
BACKGROUND_NODES = [0.0, 1e-3, 4e-3, 1.6e-2, 6.4e-2, 0.256]

SPECIES = {
    "Xa": dict(shells=(0, 0, 1), rcut=5.0, z_valence=3.0, decay=0.55, nlcc=True),
    "Yb": dict(shells=(0, 1, 2), rcut=5.5, z_valence=4.0, decay=0.45, nlcc=False),
}


def synthetic_source(symbol, shells, rcut, z_valence, decay, nlcc=True):
    """Normalized smooth radial orbitals and a two-Gaussian valence density with the requested charge."""
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


def build_root(root: Path, *, radial_rank=3, l_buffer=2, tail_seeds=1, step=0.3, order=40):
    """Write a complete, identified environment-XC table root for SPECIES; returns (sources, projectors)."""
    sources = {s: synthetic_source(s, **kw) for s, kw in SPECIES.items()}
    proj = {s: build_density_projectors(src, radial_rank=radial_rank, l_buffer=l_buffer, tail_seeds=tail_seeds,
                                        density_threshold=1e-7) for s, src in sources.items()}
    # synthetic sources are identified by their generating parameters
    settings = {"radial_rank": radial_rank, "l_buffer": l_buffer, "tail_seeds": tail_seeds, "density_threshold": 1e-7,
                "distance_step": step, "order": order, "background_nodes": BACKGROUND_NODES}
    code_identity = {"fixture": "nacf_support.build_root"}
    build_id = build_identity(settings, code_identity)
    source_sha = {s: sha256_json({"symbol": s, **SPECIES[s]}) for s in sources}

    def identity(kind, key, index=None, *symbols):
        return {"schema": ENVXC_SCHEMA, "kind": kind, "key": key, "index": index, "build_identity": build_id,
                "sources": {x: source_sha[x] for x in symbols}}

    def row(path, sha, table):
        return {"path": str(path.relative_to(root)), "sha256": sha, "left_shells": list(table["left_shells"]),
                "right_shells": list(table["right_shells"]), "support_bohr": table["support_bohr"]}

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
                tables[kind][key] = row(path, save_table(path, table, identity=identity(kind, key, None, k, a)), table)
    for a in symbols:
        for b in symbols:
            if a > b:
                continue
            distances = distance_grid(sources[a].orbital_cutoff_bohr + sources[b].orbital_cutoff_bohr, step)
            for kind in ("envnorm", "envpair", "pairmom"):
                key = table_key(kind, a, b)
                table = build_table_values(kind, sources[a], sources[b], distances=distances, order=order)
                path = root / table_filename(kind, key)
                tables[kind][key] = row(path, save_table(path, table, identity=identity(kind, key, None, a, b)), table)
            for index, bg in enumerate(BACKGROUND_NODES):
                key = table_key("xcbg", a, b, index)
                table = build_table_values("xcbg", sources[a], sources[b], distances=distances, order=order, background=bg)
                path = root / table_filename("xcbg", key)
                sha = save_table(path, table, identity=identity("xcbg", key, index, a, b), background_bohr_minus3=bg)
                tables["xcbg"][key] = row(path, sha, table)
    manifest = {"schema": ENVXC_SCHEMA, "complete": True, "length_unit": "bohr", "density_unit": "bohr^-3", "xc_energy_unit": "eV",
                "harmonic_convention": "deeptb_abacus_real", "endpoint_policy": "exclude_i0_and_jR", "interpolation": "cubic",
                "background_nodes": BACKGROUND_NODES, "species": species_rows, "tables": tables,
                "build_identity": build_id, "code_identity": code_identity, "settings": settings,
                "sources": {s: {"source_sha256": source_sha[s], "synthetic": True} for s in sources}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return sources, proj


_SHARED_ROOTS = {}


def shared_envxc_root(tmp_path_factory):
    """One read-only synthetic table root per test session: (path, sources, projectors). Callers must not write to it."""
    session = tmp_path_factory.getbasetemp()
    if session not in _SHARED_ROOTS:
        path = tmp_path_factory.mktemp("envxc_root")
        _SHARED_ROOTS[session] = (path, *build_root(path))
    return _SHARED_ROOTS[session]


def make_structure(symbols, positions, cell, cutoffs):
    """All directed edges with |R_j + t cell - R_i| < cut_i + cut_j over the lattice images t (reverse-closed)."""
    pos = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    radius = 2 * max(cutoffs.values())
    heights = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    reach = [int(np.ceil(radius / h)) + 1 for h in heights]
    images = [np.array(t) for t in itertools.product(*(range(-n, n + 1) for n in reach))]
    edges, shifts = [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            for t in images:
                if i == j and not np.any(t):
                    continue
                if np.linalg.norm(pos[j] + t @ cell - pos[i]) < cutoffs[symbols[i]] + cutoffs[symbols[j]] - 1e-9:
                    edges.append((i, j))
                    shifts.append(t)
    return {"symbols": list(symbols), "positions_bohr": pos, "cell_bohr": cell, "edge_index": np.array(edges).T,
            "edge_cell_shift": np.array(shifts, dtype=np.int64), "pbc": (True, True, True)}


def edge_overlap(sources, g):
    """Padded AO overlap blocks of every edge of ``g`` from exact two-centre quadrature tables."""
    pos, cell, ei, sh = g["positions_bohr"], g["cell_bohr"], g["edge_index"], g["edge_cell_shift"]
    w = max(src.norb for src in sources.values())
    out = np.zeros((ei.shape[1], w, w))
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
