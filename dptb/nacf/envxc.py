"""Grid-free hopping environment XC: D2 background tables, GSN moment term and McWEDA remainder.

Written by Claude Fable 5.1, 2026-09-20. Online work is topology, table lookup/rotation,
finite-rank contraction and scalar v_xc/v'_xc evaluation. No spatial quadrature, no runtime
table construction. Tables come from :mod:`dptb.nacf.envxc_tables` (schema nacf-envxc/v1).

Per directed edge e=(i,j,R), shell pair (a in i, b in j) and AO pair (mu in a, nu in b):

    N_ab   = sum_k <w_a|rho_k|w_b>          ~= sum_k G(k,i)^T eps_k G(k,j)     (envfac contraction)
    Sw_ab  = <w_a|w_b>(d)                                                       (envnorm table)
    b_ab   = N_ab / Sw_ab                    (0 where Sw_ab == 0: disjoint envelopes => block is 0)
    D2     : delta_mn = X_mn(d, b_ab) - X_mn(d, 0)        (xcbg layers, shifted-log interpolation)
    moment : delta_mn = v'(rho_pair_ab + b_ab) (Denv_mn - b_ab S_mn),  Denv = sum_k F(k,i)^T eps_k F(k,j)
    McWEDA : delta_mn = [v(rt)-v(rp)] S + v'(rt)(Dp+Denv - rt S) - v'(rp)(Dp - rp S),
             rp = <w_a|rho_pair|w_b>/Sw_ab, rt = rp + b_ab, Dp = <mu|rho_pair|nu>

All returned blocks are environment corrections to be added to the pair-only XC block
X(d,0); they are in eV and in the ABACUS AO gauge of the bank tables. Reverse edges are the
exact transposes of the representative edge (computed once). Endpoints (i,0) and (j,R) are
excluded by the topology; their other periodic images are retained.

Stabilization of the moment arms (2026-09-20, after two real fixed100 counterexamples):
the finite-rank moment Denv carries an absolute error that does not vanish with the density,
while v'(rho_tot) ~ rho^(-2/3) diverges; on long edges (rho_tot ~ 1e-12 .. 1e-15) the product
reached 1e4 .. 1e6 meV. The stored envelope moments are <w_a|rho|w_b> with w = |R| Y00, i.e. they
carry the 1/(4 pi) of Y00^2. For a shell block (n_a = 2l_a+1, n_b = 2l_b+1) Cauchy-Schwarz with
the addition theorem sum_m Y_lm^2 = n/(4 pi) gives the exact, rotation-invariant inequalities

    ||<a|rho|b>||_F <= kappa_ab <w_a|rho|w_b>,  ||S_ab||_F <= kappa_ab <w_a|w_b>,  kappa_ab = sqrt(n_a n_b),

hence for the residuals that multiply v':  ||Denv - b S|| <= 2 kappa N,  ||Dp + Denv - rho_t S||
<= 2 kappa (P + N),  ||Dp - rho_p S|| <= 2 kappa P  (N, P = stored env / pair envelope moments,
b = N/Sw, rho_p = P/Sw, rho_t = rho_p + b). Each residual shell block is rescaled by
min(1, bound/||M||_F): a covariant projection (Frobenius norm is invariant under orthogonal shell
rotations; signs and the block direction are kept), applied consistently to the environment,
total and pair residuals so the zero-environment cancellation of McWEDA stays exact. It bounds
the moment term by 2 kappa Sw |rho_t v'(rho_t)| ~ (2/3) kappa Sw |v_xc(rho_t)|, the scale of the
D2 correction itself, vanishing with the density. Where the factorized envelope moment is
unresolved (N_ab <= 0) the environment moment of that shell pair is omitted (b = 0, Denv = 0):
d2_moment falls back to D2, mcweda to the direct pair term. This is a consistency projection
between two approximate estimates (finite-rank signed moments vs factorized envelope moments),
not an exact recovery; the rescaled shell pairs, removed Frobenius mass and omitted pairs are
reported. Pure d2 is unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from dptb.data.interfaces.p2_table import RadialBlockTable
from .envxc_tables import ENVXC_SCHEMA, PAIR_KINDS, RY_TO_EV_XC, TABLE_KINDS, sha256_file, table_key
from .radial import TorchRadialBlockTable
from .topology import build_edge_topology

ARMS = ("d2", "d2_moment", "mcweda")


def shell_kappa(shells_a, shells_b) -> np.ndarray:
    """kappa_ab = sqrt((2l_a+1)(2l_b+1)): Frobenius bound factor of a shell block relative to the stored envelope moment."""
    na = np.array([2 * int(l) + 1 for l in shells_a], dtype=np.float64)
    nb = np.array([2 * int(l) + 1 for l in shells_b], dtype=np.float64)
    return np.sqrt(na[:, None] * nb[None, :])


def residual_rescale(M, bound, ia, ib):
    """Shell-block Frobenius projection: scale every shell block M[..., a, b] by min(1, bound_ab / ||M_ab||_F).

    ``M`` is [E, ni, nj] (torch or NumPy), ``bound`` [E, nsa, nsb] >= 0, ``ia``/``ib`` the shell index of every
    AO row/column. Covariant under orthogonal rotations inside each shell (the Frobenius norm is invariant),
    keeps signs and the block direction. Returns (scaled M, number of rescaled shell blocks, removed Frobenius
    mass sum(||M|| - bound)+).
    """
    if isinstance(M, torch.Tensor):
        E, ni, nj = M.shape
        nsa, nsb = bound.shape[1:]
        rows = M.new_zeros((E, nsa, nj)).index_add_(1, ia, M.square())
        norm = M.new_zeros((E, nsa, nsb)).index_add_(2, ib, rows).sqrt()
        over = norm > bound
        factor = torch.where(over, bound / torch.where(over, norm, torch.ones_like(norm)), torch.ones_like(norm))
        return M * factor[:, ia][:, :, ib], int(over.sum().item()), float((norm - bound).clamp_min(0.0).sum().item())
    M = np.asarray(M, dtype=np.float64)
    E, ni, nj = M.shape
    nsa, nsb = bound.shape[1:]
    rows = np.zeros((E, nsa, nj))
    np.add.at(rows, (slice(None), ia), M * M)
    blocks = np.zeros((E, nsa, nsb))
    np.add.at(blocks, (slice(None), slice(None), ib), rows)
    norm = np.sqrt(blocks)
    over = norm > bound
    factor = np.where(over, bound / np.where(over, norm, 1.0), 1.0)
    return M * factor[:, ia][:, :, ib], int(np.count_nonzero(over)), float(np.maximum(norm - bound, 0.0).sum())


# --------------------------------------------------------------------------- LDA-PZ81 (torch)
def lda_pz81_v_dv_torch(rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v_xc (eV) and dv_xc/drho (eV bohr^3); zero for rho <= 1e-20. Same formula as the accepted CUDA parity code."""
    mask = rho > 1e-20
    n = torch.where(mask, rho, torch.ones_like(rho))
    r = (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0)
    hi = r < 1.0
    logr = torch.log(r)
    e_hi = 0.0311 * logr - 0.048 + 0.002 * r * logr - 0.0116 * r
    ep_hi = 0.0311 / r + 0.002 * (logr + 1.0) - 0.0116
    epp_hi = -0.0311 / r.square() + 0.002 / r
    den = 1.0 + 1.0529 * torch.sqrt(r) + 0.3334 * r
    dp = 1.0529 / (2.0 * torch.sqrt(r)) + 0.3334
    dpp = -1.0529 / (4.0 * r.pow(1.5))
    e = torch.where(hi, e_hi, -0.1423 / den)
    ep = torch.where(hi, ep_hi, 0.1423 * dp / den.square())
    epp = torch.where(hi, epp_hi, 0.1423 * dpp / den.square() - 0.2846 * dp.square() / den.pow(3))
    vx = -(3.0 / math.pi * n) ** (1.0 / 3.0)
    v = 2.0 * RY_TO_EV_XC * (vx + e - r * ep / 3.0)
    dv = 2.0 * RY_TO_EV_XC * (vx / (3.0 * n) - (2.0 * ep - r * epp) * r / (9.0 * n))
    zero = torch.zeros_like(rho)
    return torch.where(mask, v, zero), torch.where(mask, dv, zero)


# --------------------------------------------------------------------------- store
class EnvXCStore:
    """Checksum-aware reader of one nacf-envxc/v1 table root."""

    def __init__(self, root: str | Path, *, max_cached_tables: int = 256, verify_checksums: bool = True):
        self.root = Path(root).resolve()
        path = self.root / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.manifest_path = path
        self.manifest_sha256 = sha256_file(path)
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("schema") != ENVXC_SCHEMA:
            raise ValueError(f"envxc schema {m.get('schema')!r} != {ENVXC_SCHEMA!r}")
        if m.get("complete") is not True:
            raise ValueError("envxc manifest is not complete")
        if m.get("length_unit") != "bohr" or m.get("density_unit") != "bohr^-3" or m.get("xc_energy_unit") != "eV":
            raise ValueError("envxc manifest units are not the expected bohr / bohr^-3 / eV")
        if m.get("harmonic_convention") != "deeptb_abacus_real" or m.get("endpoint_policy") != "exclude_i0_and_jR":
            raise ValueError("envxc manifest gauge or endpoint policy mismatch")
        self.build_identity = m.get("build_identity")
        if not isinstance(self.build_identity, str) or len(self.build_identity) != 64:
            raise ValueError("envxc manifest has no build identity: unidentified table root, rebuild it into a new root")
        self.manifest = m
        self.species: Mapping[str, Mapping[str, Any]] = m["species"]
        self.tables: Mapping[str, Mapping[str, Mapping[str, Any]]] = m["tables"]
        self.background_nodes = None if m.get("background_nodes") is None else np.asarray(m["background_nodes"], dtype=np.float64)
        if self.background_nodes is not None:
            if self.background_nodes.ndim != 1 or self.background_nodes[0] != 0.0 or np.any(np.diff(self.background_nodes) <= 0):
                raise ValueError("background nodes must start at 0 and increase strictly")
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.verify_checksums = bool(verify_checksums)
        self._cache: OrderedDict[str, RadialBlockTable] = OrderedDict()
        self._species_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._verified: set[Path] = set()
        for symbol, row in self.species.items():
            if sum(2 * int(l) + 1 for l in row["orbital_shells"]) != int(row["orbital_norb"]):
                raise ValueError(f"{symbol}: orbital shell/dimension mismatch")
            if sum(2 * int(l) + 1 for l in row["q_shells"]) != int(row["q_norb"]):
                raise ValueError(f"{symbol}: projector shell/dimension mismatch")
            if float(row["q_cutoff_bohr"]) <= 0 or float(row["orbital_cutoff_bohr"]) <= 0:
                raise ValueError(f"{symbol}: invalid cutoffs")
        for kind, rows in self.tables.items():
            if kind not in TABLE_KINDS:
                raise ValueError(f"unknown table kind {kind}")
            for key, row in rows.items():
                if kind == "xcbg" and self.background_nodes is None:
                    raise ValueError("xcbg tables require background_nodes")
                if not row.get("imported") and str(row.get("path", "")).startswith(("/", "\\")):
                    raise ValueError(f"{kind}:{key} path must be relative unless imported")

    # ---- metadata ---------------------------------------------------------------------
    def has(self, kind: str, left: str, right: str, index: int | None = None) -> bool:
        return table_key(kind, left, right, index) in self.tables.get(kind, {})

    def orbital_shells(self, symbol: str) -> tuple[int, ...]:
        return tuple(int(l) for l in self.species[symbol]["orbital_shells"])

    def orbital_cutoff(self, symbol: str) -> float:
        return float(self.species[symbol]["orbital_cutoff_bohr"])

    def q_shells(self, symbol: str) -> tuple[int, ...]:
        return tuple(int(l) for l in self.species[symbol]["q_shells"])

    def q_cutoff(self, symbol: str) -> float:
        return float(self.species[symbol]["q_cutoff_bohr"])

    def nshells(self, symbol: str) -> int:
        return len(self.species[symbol]["orbital_shells"])

    def _verify(self, path: Path, expected: Any, label: str) -> None:
        if not self.verify_checksums or path in self._verified:
            return
        expected = str(expected or "").lower()
        if len(expected) != 64:
            raise ValueError(f"{label} has no valid SHA256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} checksum mismatch: manifest={expected} actual={actual}")
        self._verified.add(path)

    def _resolve(self, row: Mapping[str, Any], label: str) -> Path:
        raw = Path(str(row["path"]))
        path = raw if raw.is_absolute() else (self.root / raw)
        path = path.resolve()
        if not raw.is_absolute() and self.root not in path.parents:
            raise ValueError(f"{label} escapes the table root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def epsilon(self, symbol: str) -> np.ndarray:
        return self._arrays(symbol)["epsilon_ao"]

    def _arrays(self, symbol: str) -> dict[str, np.ndarray]:
        cached = self._species_arrays.get(symbol)
        if cached is not None:
            return cached
        row = self.species[symbol]
        path = self._resolve(row, f"species:{symbol}") if "path" in row else self._resolve({"path": row["array_path"]}, f"species:{symbol}")
        self._verify(path, row.get("array_sha256"), f"species:{symbol}")
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]) for name in z.files if name not in ("metadata", "identity")}
            identity = json.loads(str(z["identity"])) if "identity" in z.files else {}
        self._check_artifact_identity(identity, {symbol}, f"species:{symbol}")
        eps = np.asarray(arrays["epsilon_ao"], dtype=np.float64)
        if eps.shape != (int(row["q_norb"]),) or not np.isfinite(eps).all() or np.any(eps <= 0):
            raise ValueError(f"{symbol}: invalid epsilon_ao")
        arrays["epsilon_ao"] = eps
        self._species_arrays[symbol] = arrays
        return arrays

    def _check_artifact_identity(self, identity: Mapping[str, Any], symbols, label: str) -> None:
        """Every artifact built by this schema must carry the manifest's build identity and the manifest's source SHA256s."""
        if not identity or identity.get("build_identity") != self.build_identity:
            raise ValueError(f"{label}: artifact build identity {identity.get('build_identity') if identity else None!r} does not match "
                             f"the manifest {self.build_identity!r}; the table root is inconsistent, rebuild it into a new root")
        for s in symbols:
            expected = self.manifest.get("sources", {}).get(s, {}).get("source_sha256")
            if expected is not None and identity.get("sources", {}).get(s) != expected:
                raise ValueError(f"{label}: source {s} SHA256 in the artifact differs from the manifest; the table root is inconsistent")

    def table(self, kind: str, left: str, right: str, index: int | None = None) -> RadialBlockTable:
        key = table_key(kind, left, right, index)
        cache_key = f"{kind}:{key}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        rows = self.tables.get(kind, {})
        if key not in rows:
            raise KeyError(f"envxc table missing: {kind} {key}")
        row = rows[key]
        path = self._resolve(row, f"{kind}:{key}")
        self._verify(path, row.get("sha256"), f"{kind}:{key}")
        with np.load(path, allow_pickle=False) as z:
            if not row.get("imported"):
                identity = json.loads(str(z["identity"])) if "identity" in z.files else {}
                self._check_artifact_identity(identity, set(key.split("|")[:2]), f"{kind}:{key}")
            values = np.asarray(z["values_eV"] if "values_eV" in z.files else z["values"])
            table = RadialBlockTable(distances=np.asarray(z["distances"], dtype=np.float64), values=values,
                                     left_shells=tuple(int(x) for x in z["left_shells"]),
                                     right_shells=tuple(int(x) for x in z["right_shells"]),
                                     support_bohr=float(np.asarray(z["support_bohr"])),
                                     interpolation=str(self.manifest.get("interpolation", "cubic")))
        a, b = key.split("|")[:2]
        if kind in ("rhofac", "envfac"):
            expected_left = self.q_shells(a)
            expected_right = self.orbital_shells(b) if kind == "rhofac" else (0,) * self.nshells(b)
        elif kind in ("envnorm", "envpair"):
            expected_left, expected_right = (0,) * self.nshells(a), (0,) * self.nshells(b)
        else:
            expected_left, expected_right = self.orbital_shells(a), self.orbital_shells(b)
        if table.left_shells != expected_left or table.right_shells != expected_right:
            raise ValueError(f"{kind}:{key} shell contract mismatch {table.left_shells}/{table.right_shells}")
        self._cache[cache_key] = table
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_cached_tables:
            self._cache.popitem(last=False)
        return table


# --------------------------------------------------------------------------- bank
class EnvXCBank(nn.Module):
    """Device-resident compiled envxc tables shared by many structure plans.

    ``overlap_bank`` (an :class:`dptb.nacf.assembly.NACFTableBank`) supplies AO overlap tables
    for the moment/McWEDA arms when the caller does not pass the edge overlap blocks it already
    computed. Tables are compiled once here, never inside forward.
    """

    def __init__(self, store: EnvXCStore, *, device="cuda", dtype=torch.float64, backend="auto",
                 prepared_cache_dir=None, overlap_bank=None):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("envxc bank requires float32 or float64")
        self.store = store
        self.backend = backend
        self.prepared_cache_dir = prepared_cache_dir
        self.overlap_bank = overlap_bank
        self.tables = nn.ModuleDict()
        # projector weights are registered buffers (``epsilons.epsilon_<symbol>``) so that ``.to()`` / ``_apply``
        # migrate them together with the anchor and the compiled tables
        self.epsilons = nn.Module()
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype))

    @property
    def device(self):
        return self._anchor.device

    @property
    def dtype(self):
        return self._anchor.dtype

    def table(self, kind: str, left: str, right: str, index: int | None = None) -> str:
        key = f"{kind}_{table_key(kind, left, right, index).replace('|', '_')}"
        if key not in self.tables:
            source = self.store.table(kind, left, right, index)
            if self.prepared_cache_dir is None:
                self.tables[key] = TorchRadialBlockTable(source, device=self.device, dtype=self.dtype, backend=self.backend)
            else:
                from .prepared import cached_table
                self.tables[key] = cached_table(source, self.prepared_cache_dir, device=self.device, dtype=self.dtype, backend=self.backend)
        return key

    def overlap_table(self, left: str, right: str) -> TorchRadialBlockTable:
        if self.overlap_bank is None:
            raise RuntimeError("moment/McWEDA arms need edge overlap blocks or an overlap bank")
        return self.overlap_bank.tables[self.overlap_bank.table("overlap", left, right)]

    def epsilon(self, symbol: str) -> torch.Tensor:
        name = f"epsilon_{symbol}"
        value = getattr(self.epsilons, name, None)
        if value is None:
            self.epsilons.register_buffer(name, torch.as_tensor(self.store.epsilon(symbol), device=self.device, dtype=self.dtype))
            value = getattr(self.epsilons, name)
        return value

    def prepare(self, symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift, *, pbc=(True, True, True), **options):
        return self.prepare_batch([dict(symbols=symbols, positions_bohr=positions_bohr, cell_bohr=cell_bohr,
                                        edge_index=edge_index, edge_cell_shift=edge_cell_shift, pbc=pbc)], **options)

    def prepare_batch(self, geometries, **options):
        return NACFEnvXCPlan(self, geometries, **options)


# --------------------------------------------------------------------------- plan
class NACFEnvXCPlan(nn.Module):
    """Geometry-bound, optionally batched, environment-XC hopping corrections.

    ``arms`` selects which corrections are assembled; only the tables and contractions an arm
    needs are loaded and executed (pure ``d2`` never evaluates AO density moments, ``mcweda``
    never loads background layers). ``forward`` returns a dict with one [E, width, width]
    block per arm (eV, original edge row order) and a ``diagnostics`` dict.
    """

    def __init__(self, bank: EnvXCBank, geometries: Sequence[Mapping[str, Any]], *, arms=("d2",),
                 library=None, max_terms=10_000_000, chunk_bytes=32 * 1024 * 1024,
                 background_method="cubic", layer_chunk_bytes=256 * 1024 * 1024, profile=False,
                 topology="native", overlap_floor=1e-8, moment_density_floor=0.0):
        """``overlap_floor``: shell pairs whose positive-envelope overlap <w_a|w_b> (dimensionless, <= 1 for
        normalized radials) is not above this value are treated as non-overlapping: their background is
        undefined and their correction is zero. Near the support edge both <w_a|rho_env|w_b> and <w_a|w_b>
        vanish while the finite-rank numerator keeps an absolute error, so the ratio is meaningless there;
        the gated pair-XC elements are themselves bounded by max|v_xc| times that overlap. The number of
        gated shell pairs and the largest gated pair-XC element are reported in the diagnostics.

        ``moment_density_floor`` (bohr^-3, default 0 = off): optional explicit validity gate for the moment
        arms; elements whose reference density rho_tot is below it get no moment term (counted). The
        default stabilization is the exact envelope bound (module docstring), which needs no threshold."""
        super().__init__()
        arms = tuple(arms)
        if not arms or any(a not in ARMS for a in arms):
            raise ValueError(f"arms must be a non-empty subset of {ARMS}")
        if background_method not in ("cubic", "linear"):
            raise ValueError("background_method must be cubic or linear")
        if topology not in ("native", "python"):
            raise ValueError("topology must be native or python")
        if not (0.0 <= float(overlap_floor) < 1.0):
            raise ValueError("overlap_floor must lie in [0, 1)")
        self.overlap_floor = float(overlap_floor)
        if float(moment_density_floor) < 0.0:
            raise ValueError("moment_density_floor must be non-negative")
        self.moment_density_floor = float(moment_density_floor)
        if not geometries:
            raise ValueError("at least one geometry is required")
        self.bank = bank
        self.arms = arms
        self.need_layers = any(a in ("d2", "d2_moment") for a in arms)
        self.need_moment = any(a in ("d2_moment", "mcweda") for a in arms)
        self.need_pairmom = "mcweda" in arms
        self.background_method = background_method
        self.chunk_bytes = int(chunk_bytes)
        self.layer_chunk_bytes = int(layer_chunk_bytes)
        self.profile = bool(profile)
        self.timing: dict[str, float] = {}
        store = bank.store
        device, dtype = bank.device, bank.dtype
        itemsize = torch.empty((), dtype=dtype).element_size()
        if self.need_layers and store.background_nodes is None:
            raise ValueError("d2 arms need background (xcbg) layers in the table root")

        def reg(name, array, integer=False):
            self.register_buffer(name, torch.as_tensor(np.array(array, copy=True), dtype=torch.long if integer else dtype, device=device))

        positions, cells, queries, terms, edges, shifts, reverse, all_symbols = [], [], [], [], [], [], [], []
        edge_ptr = [0]
        n_atoms = n_queries = n_edges = n_terms = 0
        self.topology_stats = []
        for graph, g in enumerate(geometries):
            symbols = tuple(str(s) for s in g["symbols"])
            for s in symbols:
                if s not in store.species:
                    raise KeyError(f"envxc tables do not cover species {s}")
            pos = np.asarray(g["positions_bohr"], dtype=np.float64)
            cell = np.asarray(g["cell_bohr"], dtype=np.float64)
            if pos.shape != (len(symbols), 3) or cell.shape != (3, 3):
                raise ValueError("invalid geometry shapes")
            ei = np.asarray(g["edge_index"], dtype=np.int64)
            sh = np.asarray(g["edge_cell_shift"], dtype=np.int64)
            ao_cut = [store.orbital_cutoff(s) for s in symbols]
            centre_cut = [store.q_cutoff(s) for s in symbols]
            if topology == "native":
                topo = build_edge_topology(pos, cell, g.get("pbc", (True, True, True)), ao_cut, centre_cut,
                                           ei, sh, library=library, max_terms=max(1, max_terms - n_terms), mode="edge_vna")
            else:
                topo = python_edge_topology(pos, cell, g.get("pbc", (True, True, True)), ao_cut, centre_cut, ei, sh)
            q, t = topo["queries"], topo["terms"]
            if n_terms + len(t) > max_terms:
                raise ValueError("third-centre term budget exceeded")
            raw = len(q)
            needed = np.zeros(len(q), dtype=bool)
            if len(t):
                needed[t[:, 1:3]] = True
            active = np.flatnonzero(needed)
            remap = np.full(len(q), -1, dtype=np.int64)
            remap[active] = np.arange(len(active))
            if len(t):
                t[:, 1:3] = remap[t[:, 1:3]]
            q = q[active].copy()
            self.topology_stats.append({k: topo[k] for k in ("broad_pairs", "search_s", "join_s")}
                                       | {"queries_before_pruning": raw, "queries": len(q), "terms": len(t), "edges": ei.shape[1]})
            q[:, :2] += n_atoms
            q = np.column_stack((q, np.full(len(q), graph, dtype=np.int64)))
            t[:, 0] += n_edges
            t[:, 1:3] += n_queries
            positions.append(pos); cells.append(cell); queries.append(q); terms.append(t)
            edges.append(ei.T + n_atoms); shifts.append(sh); reverse.append(topo["reverse"] + n_edges)
            n_atoms += len(symbols); n_queries += len(q); n_edges += ei.shape[1]; n_terms += len(t)
            edge_ptr.append(n_edges); all_symbols.extend(symbols)
        all_symbols = np.asarray(all_symbols)
        self.symbols = tuple(all_symbols.tolist())
        species, codes = np.unique(all_symbols, return_inverse=True)
        species = [str(s) for s in species]
        ns = len(species)
        self.nedges, self.nqueries, self.nterms, self.natoms = n_edges, n_queries, n_terms, n_atoms
        self.width = max(sum(2 * l + 1 for l in store.orbital_shells(s)) for s in species)
        self.nsmax = max(store.nshells(s) for s in species)
        reg("positions", np.concatenate(positions)); reg("cells", np.stack(cells))
        edge_array = np.concatenate(edges)
        q_all = np.concatenate(queries) if queries else np.empty((0, 6), dtype=np.int64)
        t_all = np.concatenate(terms) if terms else np.empty((0, 3), dtype=np.int64)
        rev = np.concatenate(reverse)
        reg("edge_index", edge_array.T, True); reg("edge_cell_shift", np.concatenate(shifts), True)
        reg("reverse", rev, True); reg("edge_ptr", edge_ptr, True)
        representative = np.arange(n_edges) <= rev
        reg("representative", representative, True)
        self.edge_slices = tuple(zip(edge_ptr[:-1], edge_ptr[1:]))
        # shell->AO index maps per species (for broadcasting shell-pair scalars to matrix elements)
        self.species_meta = {}
        for s in species:
            shells = store.orbital_shells(s)
            self.species_meta[s] = {"shells": shells, "norb": sum(2 * l + 1 for l in shells), "nshells": len(shells)}
            reg(f"shell_of_ao_{s}", np.repeat(np.arange(len(shells)), [2 * l + 1 for l in shells]), True)
        # ---- factor queries grouped by (centre species, AO species) ----------------------------
        self.factor_specs = []
        pair_ids = {}
        if len(q_all):
            q_pairs = codes[q_all[:, 1]] * ns + codes[q_all[:, 0]]
            unique_pairs, group = np.unique(q_pairs, return_inverse=True)
            local = np.empty(len(q_all), dtype=np.int64)
            for number, pair in enumerate(unique_pairs):
                centre, ao = divmod(int(pair), ns)
                rows = np.flatnonzero(group == number)
                local[rows] = np.arange(len(rows))
                sk, sa = species[centre], species[ao]
                reg(f"fq_{number}", q_all[rows], True)
                env_key = bank.table("envfac", sk, sa)
                rho_key = bank.table("rhofac", sk, sa) if self.need_moment else None
                self.factor_specs.append((number, sk, sa, env_key, rho_key))
                pair_ids[(centre, ao)] = number
            triples = ((codes[edge_array[t_all[:, 0], 0]] * ns + codes[edge_array[t_all[:, 0], 1]]) * ns + codes[q_all[t_all[:, 1], 1]])
            unique_triples, group = np.unique(triples, return_inverse=True)
            self.contraction_specs = []
            for number, triple in enumerate(unique_triples):
                pair, sk_code = divmod(int(triple), ns)
                si_code, sj_code = divmod(pair, ns)
                rows = t_all[group == number].copy()
                rows[:, 1:3] = local[rows[:, 1:3]]
                si, sj, sk = species[si_code], species[sj_code], species[sk_code]
                eps = bank.epsilon(sk)
                ni, nj = self.species_meta[si]["norb"], self.species_meta[sj]["norb"]
                nsi, nsj = self.species_meta[si]["nshells"], self.species_meta[sj]["nshells"]
                per_term = itemsize * (len(eps) * (2 * max(ni, nsi) + max(nj, nsj)) + ni * nj)
                chunk = max(1, min(2048, self.chunk_bytes // max(1, per_term)))
                reg(f"terms_{number}", rows, True)
                self.contraction_specs.append((number, pair_ids[(sk_code, si_code)], pair_ids[(sk_code, sj_code)], sk, ni, nj, nsi, nsj, chunk))
        else:
            self.contraction_specs = []
        # ---- representative edges grouped by (sorted species pair, reversed) --------------------
        pos_all = np.concatenate(positions)
        shift_all = np.concatenate(shifts)
        graph_of_edge = np.repeat(np.arange(len(geometries)), [b - a for a, b in self.edge_slices])
        vec = pos_all[edge_array[:, 1]] - pos_all[edge_array[:, 0]] + np.einsum("ei,eij->ej", shift_all.astype(np.float64), np.stack(cells)[graph_of_edge])
        self.pair_specs = []
        rep_rows = np.flatnonzero(representative)
        groups: dict[tuple[str, str, bool], list[int]] = {}
        for e in rep_rows:
            si, sj = self.symbols[edge_array[e, 0]], self.symbols[edge_array[e, 1]]
            groups.setdefault((min(si, sj), max(si, sj), si > sj), []).append(int(e))
        self.n_layers = 0 if store.background_nodes is None else len(store.background_nodes)
        for number, ((sa, sb, rev_flag), rows) in enumerate(sorted(groups.items())):
            rows = np.asarray(rows, dtype=np.int64)
            reg(f"pe_{number}", rows, True)
            reg(f"pv_{number}", vec[rows] * (-1.0 if rev_flag else 1.0))
            keys = {"envnorm": bank.table("envnorm", sa, sb)}
            if self.need_moment:
                keys["envpair"] = bank.table("envpair", sa, sb)
            if self.need_pairmom:
                keys["pairmom"] = bank.table("pairmom", sa, sb)
            if self.need_layers:
                keys["layers"] = tuple(bank.table("xcbg", sa, sb, k) for k in range(self.n_layers))
            self.pair_specs.append((number, sa, sb, rev_flag, keys))
            if self.need_moment:
                si_, sj_ = (sb, sa) if rev_flag else (sa, sb)
                if not hasattr(self, f"kappa_{si_}_{sj_}"):
                    reg(f"kappa_{si_}_{sj_}", shell_kappa(self.species_meta[si_]["shells"], self.species_meta[sj_]["shells"]))
        if self.need_layers:
            nodes = store.background_nodes
            reg("bg_nodes", nodes)
            reg("bg_log_nodes", np.log(nodes + nodes[1]))
        self.fused = None

    # ---- fusion (optional, CUDA) -------------------------------------------------------------
    def enable_fusion(self, *, include_layers=True, contraction=True):
        """One fused radial launch for every factor/pair table; fused atomic contraction on CUDA."""
        if self.positions.device.type != "cuda":
            raise ValueError("fusion requires a CUDA plan")
        from .fusion import RadialMultiPlan
        groups, order = [], []
        for number, sk, sa, env_key, rho_key in self.factor_specs:
            count = len(getattr(self, f"fq_{number}"))
            groups.append(([self.bank.tables[env_key]], count)); order.append(("factor", number, "envfac"))
            if rho_key is not None:
                groups.append(([self.bank.tables[rho_key]], count)); order.append(("factor", number, "rhofac"))
        layer_bytes = 0
        for number, sa, sb, rev_flag, keys in self.pair_specs:
            count = len(getattr(self, f"pe_{number}"))
            s_tables = [self.bank.tables[keys["envnorm"]]]
            names = ["envnorm"]
            if "envpair" in keys:
                s_tables.append(self.bank.tables[keys["envpair"]]); names.append("envpair")
            groups.append((s_tables, count)); order.append(("pair", number, tuple(names)))
            ao_tables, ao_names = [], []
            if "pairmom" in keys:
                ao_tables.append(self.bank.tables[keys["pairmom"]]); ao_names.append("pairmom")
            if include_layers and "layers" in keys:
                na, nb = self.species_meta[sa]["norb"], self.species_meta[sb]["norb"]
                layer_bytes += count * na * nb * self.n_layers * torch.empty((), dtype=self.bank.dtype).element_size()
                if layer_bytes <= self.layer_chunk_bytes:
                    ao_tables.extend(self.bank.tables[k] for k in keys["layers"]); ao_names.append("layers")
            if ao_tables:
                groups.append((ao_tables, count)); order.append(("pair", number, tuple(ao_names)))
        self.fused = (RadialMultiPlan(groups), order) if groups else None
        self.fused_contraction = bool(contraction)
        return self

    # ---- helpers -----------------------------------------------------------------------------
    def _sync(self):
        if self.profile and self.positions.device.type == "cuda":
            torch.cuda.synchronize(self.positions.device)

    def _factor_delta(self, number):
        q = getattr(self, f"fq_{number}")
        translation = torch.einsum("qi,qij->qj", q[:, 2:5].to(self.positions.dtype), self.cells[q[:, 5]])
        return self.positions[q[:, 0]] - self.positions[q[:, 1]] + translation

    def _contract(self, a, eps, b, rows, out, ni, nj, chunk):
        if getattr(self, "fused_contraction", False) and out.is_cuda:
            from .fusion import contract_add
            contract_add(a.contiguous(), eps.contiguous(), b.contiguous(), rows.contiguous(), out)
            return
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            left, right = a[part[:, 1]], b[part[:, 2]]
            out[:, :ni, :nj].index_add_(0, part[:, 0], (left * eps[None, :, None]).transpose(-1, -2) @ right)

    def _background_weights(self, b):
        """Shifted-log Lagrange weights on the background axis; above the last node the last node is used (no extrapolation)."""
        nodes, log_nodes = self.bg_nodes, self.bg_log_nodes
        n = nodes.numel()
        lb = torch.log(b.clamp_min(0.0) + nodes[1])
        idx = (torch.searchsorted(log_nodes, lb.reshape(-1).contiguous(), right=True) - 1).clamp(0, n - 2).reshape(b.shape)
        if self.background_method == "linear":
            x0, x1 = log_nodes[idx], log_nodes[idx + 1]
            t = ((lb - x0) / (x1 - x0)).clamp(0.0, 1.0)
            cols = torch.stack((idx, idx + 1), -1)
            wts = torch.stack((1 - t, t), -1)
        else:
            base = (idx - 1).clamp(0, n - 4)
            cols = base[..., None] + torch.arange(4, device=b.device)
            xs = log_nodes[cols]
            wts = torch.ones_like(xs)
            for a in range(4):
                for c in range(4):
                    if a != c:
                        wts[..., a] = wts[..., a] * (lb - xs[..., c]) / (xs[..., a] - xs[..., c])
        out_hi = lb > log_nodes[-1]
        if bool(out_hi.any()):
            last = torch.zeros_like(wts)
            last[..., -1] = 1.0
            wts = torch.where(out_hi[..., None], last, wts)
        return cols, wts, out_hi

    def _blend_layers(self, layers, b_elem):
        """layers [L, E, na, nb] -> sum_k w_k layer_{col_k} with per-element columns."""
        cols, wts, out_hi = self._background_weights(b_elem)
        out = torch.zeros_like(b_elem)
        for k in range(cols.shape[-1]):
            out = out + wts[..., k] * torch.gather(layers, 0, cols[..., k][None])[0]
        return out, out_hi

    def _evaluate_layers(self, keys, vec, fused_values):
        """Return [L, Ec, na, nb] for a pair group, from fused buffers or per-layer evaluation."""
        if fused_values is not None and "layers" in fused_values:
            return fused_values["layers"]
        return torch.stack([self.bank.tables[k](vec) for k in keys])

    # ---- forward -----------------------------------------------------------------------------
    def forward(self, edge_overlap_ao=None):
        dev, dtype = self.positions.device, self.positions.dtype
        E, w, nsm = self.nedges, self.width, self.nsmax
        timing = {}
        self._sync(); t0 = time.perf_counter()
        # 1. factor tables --------------------------------------------------------------------
        env_values, rho_values, fused_pair = {}, {}, {}
        if self.fused is not None:
            plan, order = self.fused
            vectors = []
            for kind, number, names in order:
                vectors.append(self._factor_delta(number) if kind == "factor" else getattr(self, f"pv_{number}"))
            outputs = plan(torch.cat(vectors)) if vectors else []
            index = 0
            for kind, number, names in order:
                if kind == "factor":
                    (env_values if names == "envfac" else rho_values)[number] = outputs[index]; index += 1
                else:
                    slot = fused_pair.setdefault(number, {})
                    for name in names:
                        if name == "layers":
                            slot["layers"] = torch.stack(outputs[index:index + self.n_layers]); index += self.n_layers
                        else:
                            slot[name] = outputs[index]; index += 1
        else:
            for number, sk, sa, env_key, rho_key in self.factor_specs:
                delta = self._factor_delta(number)
                env_values[number] = self.bank.tables[env_key](delta)
                if rho_key is not None:
                    rho_values[number] = self.bank.tables[rho_key](delta)
        self._sync(); timing["factor_queries"] = time.perf_counter() - t0; t0 = time.perf_counter()
        # 2. third-centre contractions ------------------------------------------------------------
        N = self.positions.new_zeros((E, nsm, nsm))
        Denv = self.positions.new_zeros((E, w, w)) if self.need_moment else None
        for number, left, right, sk, ni, nj, nsi, nsj, chunk in self.contraction_specs:
            rows = getattr(self, f"terms_{number}")
            eps = self.bank.epsilon(sk)
            self._contract(env_values[left], eps, env_values[right], rows, N, nsi, nsj, chunk)
            if Denv is not None:
                self._contract(rho_values[left], eps, rho_values[right], rows, Denv, ni, nj, chunk)
        self._sync(); timing["contraction"] = time.perf_counter() - t0; t0 = time.perf_counter()
        # 3. pair tables on representative edges ------------------------------------------------
        Sw = self.positions.new_zeros((E, nsm, nsm))
        Pw = self.positions.new_zeros((E, nsm, nsm)) if self.need_moment else None
        Dp = self.positions.new_zeros((E, w, w)) if self.need_pairmom else None
        S = None
        if self.need_moment:
            if edge_overlap_ao is not None:
                if edge_overlap_ao.shape != (E, w, w):
                    raise ValueError("edge_overlap_ao must be [edges, width, width] in plan order")
                S = edge_overlap_ao.to(dtype)
            else:
                S = self.positions.new_zeros((E, w, w))
        group_cache = []
        for number, sa, sb, rev_flag, keys in self.pair_specs:
            rows, vec = getattr(self, f"pe_{number}"), getattr(self, f"pv_{number}")
            fv = fused_pair.get(number)
            def get(name):
                if fv is not None and name in fv:
                    return fv[name]
                return self.bank.tables[keys[name]](vec)
            nsa, nsb = self.species_meta[sa]["nshells"], self.species_meta[sb]["nshells"]
            na, nb = self.species_meta[sa]["norb"], self.species_meta[sb]["norb"]
            def place(target, block):
                if rev_flag:
                    block = block.transpose(1, 2)
                target[rows, :block.shape[1], :block.shape[2]] = block
            place(Sw, get("envnorm"))
            if Pw is not None:
                place(Pw, get("envpair"))
            if Dp is not None:
                place(Dp, get("pairmom"))
            if S is not None and edge_overlap_ao is None:
                place(S, self.bank.overlap_table(sa, sb)(vec))
            group_cache.append((number, sa, sb, rev_flag, keys, rows, vec, fv, na, nb))
        self._sync(); timing["pair_tables"] = time.perf_counter() - t0; t0 = time.perf_counter()
        # 4. shell-pair scalars ------------------------------------------------------------------
        mask_shell = Sw > self.overlap_floor
        b_shell = torch.where(mask_shell, N / torch.where(mask_shell, Sw, torch.ones_like(Sw)), torch.zeros_like(N))
        neg_mask = (b_shell < 0) & mask_shell
        negative = int(neg_mask.sum().item())
        # finite-rank leakage makes the (exactly positive) numerator slightly negative only where the envelope
        # overlap is weak; record how large an overlap it reaches so the regime is visible, then clamp
        negative_max_overlap = float(Sw[neg_mask].max().item()) if negative else 0.0
        negative_min_value = float(b_shell[neg_mask].min().item()) if negative else 0.0
        b_shell = b_shell.clamp_min(0.0)
        leaked = int(((~mask_shell) & (N.abs() > 0)).sum().item())
        gated = int(((~mask_shell) & (Sw > 0)).sum().item())
        gated_block_abs_max = 0.0
        rho_pair_shell = None
        if self.need_moment:
            rho_pair_shell = torch.where(mask_shell, Pw / torch.where(mask_shell, Sw, torch.ones_like(Sw)), torch.zeros_like(N)).clamp_min(0.0)
        out = {a: self.positions.new_zeros((E, w, w)) for a in self.arms}
        above = 0
        moment_stats = {"rho_tot_min": None, "dv_abs_max": 0.0,
                        "moment_env_rescaled_shell_pairs": 0, "moment_env_removed_frobenius_bohr3": 0.0,
                        "mcweda_total_rescaled_shell_pairs": 0, "mcweda_total_removed_frobenius_bohr3": 0.0,
                        "mcweda_pair_rescaled_shell_pairs": 0, "mcweda_pair_removed_frobenius_bohr3": 0.0,
                        "moment_below_density_floor_elements": 0, "moment_density_floor": self.moment_density_floor,
                        "moment_term_abs_max_eV": 0.0, "mcweda_term_abs_max_eV": 0.0}
        if self.need_moment:
            moment_stats["moment_unresolved_env_shell_pairs"] = negative
            moment_stats["environment_dominated_shell_pairs"] = int(((b_shell > rho_pair_shell) & mask_shell).sum().item())
        for number, sa, sb, rev_flag, keys, rows, vec, fv, na, nb in group_cache:
            si, sj = (sb, sa) if rev_flag else (sa, sb)
            ia, ib = getattr(self, f"shell_of_ao_{si}"), getattr(self, f"shell_of_ao_{sj}")
            ni, nj = len(ia), len(ib)
            bs = b_shell[rows][:, ia][:, :, ib]                          # [Ec, ni, nj]
            ms = mask_shell[rows][:, ia][:, :, ib]
            if self.need_layers:
                # layers are stored A|B (sorted); the edge block is transposed when the edge is reversed
                for start in range(0, len(rows), self._layer_chunk(na, nb)):
                    sl = slice(start, start + self._layer_chunk(na, nb))
                    layers = self._evaluate_layers(keys["layers"], vec[sl], None if fv is None or "layers" not in fv else {"layers": fv["layers"][:, sl]})
                    if rev_flag:
                        layers = layers.transpose(-1, -2)
                    blended, out_hi = self._blend_layers(layers, bs[sl])
                    above += int((out_hi & ms[sl]).sum().item())
                    gated_elems = (~ms[sl]) & (layers[0] != 0)
                    if bool(gated_elems.any()):
                        gated_block_abs_max = max(gated_block_abs_max, float(layers[0][gated_elems].abs().max().item()))
                    delta2 = torch.where(ms[sl], blended - layers[0], torch.zeros_like(blended))
                    for arm in ("d2", "d2_moment"):
                        if arm in out:
                            out[arm][rows[sl], :ni, :nj] = delta2
            if self.need_moment:
                rp = rho_pair_shell[rows][:, ia][:, :, ib]
                rt = rp + bs
                valid = ms & (rt > 0)
                if self.moment_density_floor > 0.0:
                    below = valid & (rt < self.moment_density_floor)
                    moment_stats["moment_below_density_floor_elements"] += int(below.sum().item())
                    valid = valid & ~below
                Se = S[rows, :ni, :nj]
                # covariant consistency projection (module docstring): every residual that multiplies v' is
                # rescaled per shell block to its envelope bound; unresolved environment moments are omitted
                nsi_, nsj_ = self.species_meta[si]["nshells"], self.species_meta[sj]["nshells"]
                kappa = getattr(self, f"kappa_{si}_{sj}")
                Nsh = N[rows, :nsi_, :nsj_]
                resolved_sh = Nsh > 0
                Nsh = Nsh.clamp_min(0.0)
                De = Denv[rows, :ni, :nj] * resolved_sh[:, ia][:, :, ib]
                M_env, n_res, removed = residual_rescale(De - bs * Se, 2.0 * kappa * Nsh, ia, ib)
                moment_stats["moment_env_rescaled_shell_pairs"] += n_res
                moment_stats["moment_env_removed_frobenius_bohr3"] += removed
                v_t, dv_t = lda_pz81_v_dv_torch(torch.where(valid, rt, torch.ones_like(rt)))
                if valid.any():
                    moment_stats["rho_tot_min"] = float(rt[valid].min().item()) if moment_stats["rho_tot_min"] is None else min(moment_stats["rho_tot_min"], float(rt[valid].min().item()))
                    moment_stats["dv_abs_max"] = max(moment_stats["dv_abs_max"], float(dv_t[valid].abs().max().item()))
                if "d2_moment" in out:
                    term = torch.where(valid, dv_t * M_env, torch.zeros_like(M_env))
                    moment_stats["moment_term_abs_max_eV"] = max(moment_stats["moment_term_abs_max_eV"], float(term.abs().max().item()) if term.numel() else 0.0)
                    out["d2_moment"][rows, :ni, :nj] = out["d2_moment"][rows, :ni, :nj] + term
                if "mcweda" in out:
                    valid_p = ms & (rp > 0)
                    v_p, dv_p = lda_pz81_v_dv_torch(torch.where(valid_p, rp, torch.ones_like(rp)))
                    Psh = Pw[rows, :nsi_, :nsj_].clamp_min(0.0)
                    Dpe = Dp[rows, :ni, :nj]
                    M_tot, n_t, rem_t = residual_rescale(Dpe + De - rt * Se, 2.0 * kappa * (Psh + Nsh), ia, ib)
                    M_pair, n_p, rem_p = residual_rescale(Dpe - rp * Se, 2.0 * kappa * Psh, ia, ib)
                    moment_stats["mcweda_total_rescaled_shell_pairs"] += n_t
                    moment_stats["mcweda_total_removed_frobenius_bohr3"] += rem_t
                    moment_stats["mcweda_pair_rescaled_shell_pairs"] += n_p
                    moment_stats["mcweda_pair_removed_frobenius_bohr3"] += rem_p
                    term = (v_t - v_p) * Se + dv_t * M_tot - dv_p * M_pair
                    term = torch.where(valid & valid_p, term, torch.zeros_like(term))
                    moment_stats["mcweda_term_abs_max_eV"] = max(moment_stats["mcweda_term_abs_max_eV"], float(term.abs().max().item()) if term.numel() else 0.0)
                    out["mcweda"][rows, :ni, :nj] = term
        self._sync(); timing["xc_assembly"] = time.perf_counter() - t0
        # 5. reverse edges are exact transposes of the representative rows -----------------------
        rep = self.representative.bool()[:, None, None]
        for arm in out:
            out[arm] = torch.where(rep, out[arm], out[arm][self.reverse].transpose(-1, -2))
        b_full = torch.where(rep, b_shell, b_shell[self.reverse].transpose(-1, -2))
        diagnostics = {"b_shell": b_full, "mask_shell": torch.where(rep, mask_shell, mask_shell[self.reverse].transpose(-1, -2)),
                       "negative_numerators": negative, "negative_numerator_max_overlap": negative_max_overlap,
                       "negative_numerator_min_b": negative_min_value, "numerator_without_overlap": leaked,
                       "shell_pairs_below_overlap_floor": gated, "gated_pair_xc_abs_max_eV": gated_block_abs_max,
                       "overlap_floor": self.overlap_floor,
                       "elements_above_last_node": above, "terms": self.nterms, "queries": self.nqueries,
                       "topology": self.topology_stats, **moment_stats}
        diagnostics["N_shell"] = torch.where(rep, N, N[self.reverse].transpose(-1, -2))
        diagnostics["Sw_shell"] = torch.where(rep, Sw, Sw[self.reverse].transpose(-1, -2))
        if rho_pair_shell is not None:
            diagnostics["rho_pair_shell"] = torch.where(rep, rho_pair_shell, rho_pair_shell[self.reverse].transpose(-1, -2))
        if Denv is not None:
            diagnostics["D_env"] = torch.where(rep, Denv, Denv[self.reverse].transpose(-1, -2))
        if self.profile:
            self.timing = timing
            diagnostics["timing_s"] = timing
        out["diagnostics"] = diagnostics
        out["edge_ptr"] = self.edge_ptr
        return out

    def _layer_chunk(self, na, nb):
        itemsize = torch.empty((), dtype=self.positions.dtype).element_size()
        per_edge = max(1, self.n_layers * na * nb * itemsize)
        return max(1, self.layer_chunk_bytes // per_edge)


# --------------------------------------------------------------------------- numpy reference
def _pair_block_cpu(store: EnvXCStore, kind, sa, sb, vec, index=None):
    """Evaluate a sorted-pair table for an edge (sa -> sb) with displacement vec, handling the reversed orientation."""
    a, b = sorted((sa, sb))
    table = store.table(kind, a, b, index)
    if (sa, sb) == (a, b):
        return table.evaluate(vec)
    return table.evaluate(-vec).T


def _lagrange_weights(nodes, b, method):
    log_nodes = np.log(nodes + nodes[1])
    lb = math.log(max(b, 0.0) + nodes[1])
    n = len(nodes)
    idx = min(max(int(np.searchsorted(log_nodes, lb, side="right")) - 1, 0), n - 2)
    if lb > log_nodes[-1]:
        return [n - 1], [1.0], True
    if method == "linear":
        x0, x1 = log_nodes[idx], log_nodes[idx + 1]
        t = min(max((lb - x0) / (x1 - x0), 0.0), 1.0)
        return [idx, idx + 1], [1 - t, t], False
    base = min(max(idx - 1, 0), n - 4)
    cols = [base + k for k in range(4)]
    xs = log_nodes[cols]
    wts = []
    for a in range(4):
        wa = 1.0
        for c in range(4):
            if a != c:
                wa *= (lb - xs[c]) / (xs[a] - xs[c])
        wts.append(wa)
    return cols, wts, False


def _images_within(cell, pbc, radius, positions=None):
    """Integer translations whose images can lie within ``radius`` of any atom (arbitrary home-cell representatives)."""
    inv = np.linalg.inv(cell)
    bounds = radius * np.linalg.norm(inv, axis=0)
    if positions is not None and len(positions):
        frac = np.asarray(positions, dtype=np.float64) @ inv
        bounds = bounds + (frac.max(axis=0) - frac.min(axis=0))
    bounds = np.ceil(bounds).astype(int) + 1
    ranges = [range(-int(b), int(b) + 1) if p else range(0, 1) for b, p in zip(bounds, pbc)]
    return np.array([[x, y, z] for x in ranges[0] for y in ranges[1] for z in ranges[2]], dtype=np.int64)


def python_edge_topology(positions, cell, pbc, ao_cutoffs, centre_cutoffs, edge_index, edge_cell_shift):
    """Pure-Python equivalent of ``build_edge_topology(mode='edge_vna')`` for tests and machines without the native library.

    Queries are (AO, centre, sx, sy, sz) with displacement pos[AO] - pos[centre] + shift @ cell; terms are
    (edge row, left query, right query) for representative edges (row <= reverse row) only. The zero-image
    self centre never appears; the (j,R) endpoint of each edge is excluded from that edge's terms.
    """
    pos = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)
    ao = np.asarray(ao_cutoffs, dtype=np.float64)
    cc = np.asarray(centre_cutoffs, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    shifts = np.asarray(edge_cell_shift, dtype=np.int64)
    n, E = len(pos), edges.shape[1]
    rows = {(int(i), int(j), int(s[0]), int(s[1]), int(s[2])): r for r, ((i, j), s) in enumerate(zip(edges.T, shifts))}
    if len(rows) != E:
        raise ValueError("duplicate directed edges")
    reverse = np.empty(E, dtype=np.int64)
    for (i, j, x, y, z), r in rows.items():
        key = (j, i, -x, -y, -z)
        if key not in rows:
            raise ValueError("every edge must have its reverse")
        reverse[r] = rows[key]
    images = _images_within(cell, pbc, float(ao.max() + cc.max()) + 1e-9, pos)
    query_index: dict[tuple[int, int, int, int, int], int] = {}
    queries: list[tuple[int, int, int, int, int]] = []
    by_atom: list[list[tuple[int, tuple[int, int, int]]]] = [[] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            deltas = pos[i] - pos[k] - images @ cell
            distances = np.linalg.norm(deltas, axis=1)
            for t, d in zip(images, distances):
                if i == k and not np.any(t):
                    continue
                if d < ao[i] + cc[k] - 1e-12:
                    key = (i, k, -int(t[0]), -int(t[1]), -int(t[2]))
                    query_index[key] = len(queries)
                    queries.append(key)
                    by_atom[i].append((k, (int(t[0]), int(t[1]), int(t[2]))))
    terms = []
    for r in range(E):
        if r > reverse[r]:
            continue
        i, j = int(edges[0, r]), int(edges[1, r])
        R = tuple(int(x) for x in shifts[r])
        for k, t in by_atom[i]:
            if k == j and t == R:
                continue
            right = (j, k, R[0] - t[0], R[1] - t[1], R[2] - t[2])
            if right in query_index:
                terms.append((r, query_index[(i, k, -t[0], -t[1], -t[2])], query_index[right]))
    return {"queries": np.array(queries, dtype=np.int64).reshape(-1, 5), "terms": np.array(terms, dtype=np.int64).reshape(-1, 3),
            "reverse": reverse, "broad_pairs": len(queries), "search_s": 0.0, "join_s": 0.0}


def reference_edge_envxc(store: EnvXCStore, geometry: Mapping[str, Any], *, arms=("d2",), background_method="cubic",
                         edge_overlap_ao=None, overlap_floor=1e-8, moment_density_floor=0.0) -> dict[str, Any]:
    """Independent NumPy implementation of the plan formulas (brute-force periodic images,
    CPU RadialBlockTable evaluation). For tests and small structures; returns per-arm [E,w,w]."""
    arms = tuple(arms)
    symbols = [str(s) for s in geometry["symbols"]]
    pos = np.asarray(geometry["positions_bohr"], dtype=np.float64)
    cell = np.asarray(geometry["cell_bohr"], dtype=np.float64)
    pbc = np.asarray(geometry.get("pbc", (True, True, True)), dtype=bool)
    ei = np.asarray(geometry["edge_index"], dtype=np.int64)
    sh = np.asarray(geometry["edge_cell_shift"], dtype=np.int64)
    E = ei.shape[1]
    need_layers = any(a in ("d2", "d2_moment") for a in arms)
    need_moment = any(a in ("d2_moment", "mcweda") for a in arms)
    norb = {s: sum(2 * l + 1 for l in store.orbital_shells(s)) for s in set(symbols)}
    nsh = {s: store.nshells(s) for s in set(symbols)}
    shell_of = {s: np.repeat(np.arange(nsh[s]), [2 * l + 1 for l in store.orbital_shells(s)]) for s in set(symbols)}
    w = max(norb.values())
    radius = max(store.orbital_cutoff(s) for s in set(symbols)) + max(store.q_cutoff(s) for s in set(symbols))
    out = {a: np.zeros((E, w, w)) for a in arms}
    b_all = np.zeros((E, max(nsh.values()), max(nsh.values())))
    nodes = store.background_nodes
    for e in range(E):
        i, j = int(ei[0, e]), int(ei[1, e])
        si, sj = symbols[i], symbols[j]
        ri, rj = pos[i], pos[j] + sh[e] @ cell
        vec = rj - ri
        ni, nj, nsi, nsj = norb[si], norb[sj], nsh[si], nsh[sj]
        N = np.zeros((nsi, nsj)); Denv = np.zeros((ni, nj))
        # image range must cover centres near BOTH endpoints (rj may sit in a neighbouring cell)
        images = _images_within(cell, pbc, radius, np.vstack([pos, rj[None]]))
        for k, sk in enumerate(symbols):
            qc = store.q_cutoff(sk)
            for t in images:
                rk = pos[k] + t @ cell
                if k == i and not np.any(t):
                    continue
                if k == j and np.array_equal(t, sh[e]):
                    continue
                if np.linalg.norm(ri - rk) >= store.orbital_cutoff(si) + qc - 1e-12:
                    continue
                if np.linalg.norm(rj - rk) >= store.orbital_cutoff(sj) + qc - 1e-12:
                    continue
                eps = store.epsilon(sk)
                gi = store.table("envfac", sk, si).evaluate(ri - rk)
                gj = store.table("envfac", sk, sj).evaluate(rj - rk)
                N += gi.T @ (eps[:, None] * gj)
                if need_moment:
                    fi = store.table("rhofac", sk, si).evaluate(ri - rk)
                    fj = store.table("rhofac", sk, sj).evaluate(rj - rk)
                    Denv += fi.T @ (eps[:, None] * fj)
        Sw = _pair_block_cpu(store, "envnorm", si, sj, vec)
        mask = Sw > overlap_floor
        b = np.where(mask, N / np.where(mask, Sw, 1.0), 0.0)
        b = np.maximum(b, 0.0)
        b_all[e, :nsi, :nsj] = b
        be = b[shell_of[si]][:, shell_of[sj]]
        me = mask[shell_of[si]][:, shell_of[sj]]
        if need_layers:
            layers = np.stack([_pair_block_cpu(store, "xcbg", si, sj, vec, k) for k in range(len(nodes))])
            blended = np.zeros((ni, nj))
            for mu in range(ni):
                for nu in range(nj):
                    cols, wts, _ = _lagrange_weights(nodes, float(be[mu, nu]), background_method)
                    blended[mu, nu] = sum(wt * layers[c, mu, nu] for c, wt in zip(cols, wts))
            delta2 = np.where(me, blended - layers[0], 0.0)
            for arm in ("d2", "d2_moment"):
                if arm in out:
                    out[arm][e, :ni, :nj] += delta2
        if need_moment:
            Pw = _pair_block_cpu(store, "envpair", si, sj, vec)
            rp = np.maximum(np.where(mask, Pw / np.where(mask, Sw, 1.0), 0.0), 0.0)
            rpe = rp[shell_of[si]][:, shell_of[sj]]
            rte = rpe + be
            if edge_overlap_ao is not None:
                Se = np.asarray(edge_overlap_ao)[e, :ni, :nj]
            else:
                raise ValueError("reference needs edge_overlap_ao for moment arms")
            from .envxc_tables import lda_pz81_v_dv
            valid = me & (rte > 0)
            if moment_density_floor > 0.0:
                valid &= rte >= moment_density_floor
            kappa = shell_kappa(store.orbital_shells(si), store.orbital_shells(sj))
            Nsh = np.maximum(N, 0.0)
            De = Denv * (N > 0)[shell_of[si]][:, shell_of[sj]]
            M_env, _, _ = residual_rescale((De - be * Se)[None], (2.0 * kappa * Nsh)[None], shell_of[si], shell_of[sj])
            v_t, dv_t = lda_pz81_v_dv(np.where(valid, rte, 1.0))
            if "d2_moment" in out:
                out["d2_moment"][e, :ni, :nj] += np.where(valid, dv_t * M_env[0], 0.0)
            if "mcweda" in out:
                Dp = _pair_block_cpu(store, "pairmom", si, sj, vec)
                Psh = np.maximum(Pw, 0.0)
                M_tot, _, _ = residual_rescale((Dp + De - rte * Se)[None], (2.0 * kappa * (Psh + Nsh))[None], shell_of[si], shell_of[sj])
                M_pair, _, _ = residual_rescale((Dp - rpe * Se)[None], (2.0 * kappa * Psh)[None], shell_of[si], shell_of[sj])
                valid_p = me & (rpe > 0)
                v_p, dv_p = lda_pz81_v_dv(np.where(valid_p, rpe, 1.0))
                term = (v_t - v_p) * Se + dv_t * M_tot[0] - dv_p * M_pair[0]
                out["mcweda"][e, :ni, :nj] = np.where(valid & valid_p, term, 0.0)
    out["b_shell"] = b_all
    return out


__all__ = ["ARMS", "EnvXCStore", "EnvXCBank", "NACFEnvXCPlan", "reference_edge_envxc", "lda_pz81_v_dv_torch",
           "python_edge_topology", "residual_rescale", "shell_kappa"]
