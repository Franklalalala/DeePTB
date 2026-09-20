"""Explicit, versioned full NACF candidate prior (independent audit 2026-09-20, item F3).

    node = P23 + onsite XC (accepted atom-centred local quadrature, fixed neighbourhood) + c S
    edge = P2 + edge VNA3c + direct pair XC + c S + environment XC (one grid-free arm)
    c    = 4 pi (sum_a M2_a) Ry_to_eV / (3 Omega)       (ABACUS potential zero point; fully periodic cells only)

This is the composition the fixed100 evaluation harness assembled by hand (accepted ``prepare`` / ``site_eval`` /
``xc_edges`` / ``envxc_plan``), as one maintained entry with an explicit recipe identity. Everything is injected:
the P2/P23/overlap table bank, the environment-XC bank, the onsite evaluator (species quadratures, density bank,
potential), the direct pair-XC tables, the atomic second moments, and the quadrature order policy. Nothing is
discovered from directories, mpids, H, H0 or labels. The existing :class:`NACFGeometryPredictor` (P23/P2 prior of
the trained checkpoints) is untouched; this entry does not change what any checkpoint was trained on.

Identity. At construction every table family must be bound to the same P2 manifest (already enforced by the bank
for P23/overlap), the recipe options must be supported by the injected tables (arm, stabilization, background
layers), and every species covered by all families must agree on AO shells, orbital cutoff and the UPF/ORB source
hashes each family declares. At ``prepare`` the species of the structure must be covered by every family. A missing
or contradictory identity raises :class:`CandidateIdentityError`; nothing falls back silently.

Boundaries stated honestly: the onsite evaluator objects (quadrature, density splines, potential) carry no
provenance of their own, so the caller passes ``onsite_identity`` (species sources, potential name, density
definition) and this module can only check it against the other families, not derive it. The order policy is
geometry-only: ``FixedOrderPolicy`` or the accepted convergence rule re-run on the new structure
(``ConvergenceOrderPolicy``), never a fixed100 lookup. Scalar (non-SOC) blocks only; SOC banks are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .envxc import ARMS, STABILIZATIONS, EnvXCBank
from .onsite import OnsiteXCEvaluator, onsite_neighbor_lists

RECIPE_SCHEMA = "nacf-candidate-prior/v1"
SOURCE_KEYS = ("upf_sha256", "orbital_sha256", "source_sha256")
XC_FUNCTIONAL = "LDA exchange + PZ81 correlation, unpolarized"
DENSITY_DEFINITION = "normalized neutral valence (r=0 repair) + unscaled NLCC"
ZERO_POINT = "cS: c = 4 pi sum(M2_valence) Ry_to_eV / (3 Omega), fully periodic"


class CandidateIdentityError(ValueError):
    """Missing, unsupported or contradictory recipe/table identity."""


# --------------------------------------------------------------------------- order policies
class OrderPolicy:
    """Geometry-only per-atom quadrature order selection. ``select`` returns (orders, checks)."""

    def identity(self) -> dict[str, Any]:
        raise NotImplementedError

    def select(self, evaluator: OnsiteXCEvaluator, g: Mapping[str, Any], width: int):
        raise NotImplementedError


class FixedOrderPolicy(OrderPolicy):
    def __init__(self, order=(128, 24, 48)):
        self.order = tuple(int(x) for x in order)
        if len(self.order) != 3 or min(self.order) <= 0:
            raise ValueError("order must be three positive integers (radial, mu, phi)")

    def identity(self):
        return {"policy": "fixed", "order": list(self.order)}

    def select(self, evaluator, g, width):
        n = len(g["symbols"])
        return [self.order] * n, [{"atom": i, "symbol": s, "selected_order": self.order, "passed": True} for i, s in enumerate(g["symbols"])]


class ConvergenceOrderPolicy(OrderPolicy):
    """The accepted fixed100 rule re-run on the structure at hand (no history lookup).

    Every atom is evaluated at ``medium`` and ``fine``; where max|medium - fine| >= ``tolerance_eV`` the atom uses
    ``fine`` and is re-checked against ``finer``. Optionally the medium block is compared with the block of the
    ``tail_radius_bohr`` neighbourhood (accepted 23 vs 27 Bohr check). ``strict=True`` raises when a check fails;
    otherwise the failure is recorded in the checks. Costs two (or three) onsite evaluations per structure.
    """

    def __init__(self, medium=(128, 24, 48), fine=(256, 40, 80), finer=(384, 56, 112), *, tolerance_eV=1e-3,
                 tail_radius_bohr=None, strict=False):
        self.medium, self.fine, self.finer = (tuple(int(x) for x in o) for o in (medium, fine, finer))
        self.tolerance_eV = float(tolerance_eV)
        self.tail_radius_bohr = None if tail_radius_bohr is None else float(tail_radius_bohr)
        self.strict = bool(strict)
        if self.tolerance_eV <= 0:
            raise ValueError("tolerance_eV must be positive")

    def identity(self):
        return {"policy": "convergence", "medium": list(self.medium), "fine": list(self.fine), "finer": list(self.finer),
                "tolerance_eV": self.tolerance_eV, "tail_radius_bohr": self.tail_radius_bohr, "strict": self.strict}

    def select(self, evaluator, g, width):
        symbols = list(g["symbols"])
        n = len(symbols)
        neighbors = evaluator.neighbors(g)
        medium = evaluator(g, width, [self.medium] * n, neighbors=neighbors)
        fine = evaluator(g, width, [self.fine] * n, neighbors=neighbors)
        error = (medium - fine).abs().flatten(1).amax(1).tolist()
        tail = None
        if self.tail_radius_bohr is not None:
            inner = onsite_neighbor_lists(g, self.tail_radius_bohr)
            tail = (medium - evaluator(g, width, [self.medium] * n, neighbors=inner)).abs().flatten(1).amax(1).tolist()
        orders, checks = [], []
        for i, s in enumerate(symbols):
            row = {"atom": i, "symbol": s, "medium_fine_eV": error[i], "fine_finer_eV": None}
            if error[i] < self.tolerance_eV:
                order, converged = self.medium, True
            else:
                finer = evaluator.blocks(evaluator.qgrid(s, self.finer), [neighbors[i]])[0]
                m = finer.shape[0]
                row["fine_finer_eV"] = float((fine[i, :m, :m] - finer).abs().max())
                order, converged = self.fine, row["fine_finer_eV"] < self.tolerance_eV
            if tail is not None:
                row["environment_tail_eV"] = tail[i]
                converged = converged and tail[i] < self.tolerance_eV
            row.update(selected_order=order, passed=bool(converged))
            orders.append(order); checks.append(row)
        if self.strict and not all(c["passed"] for c in checks):
            raise RuntimeError("onsite quadrature did not converge for atoms " + str([c["atom"] for c in checks if not c["passed"]]))
        return orders, checks


# --------------------------------------------------------------------------- injected providers
def normalize_sources(row: Mapping[str, Any] | None) -> dict[str, str]:
    """Species source hashes in one vocabulary: ``upf_sha256``, ``orbital_sha256``, ``source_sha256``."""
    if not row:
        return {}
    out = {}
    for key in ("upf_sha256", "orbital_sha256", "source_sha256"):
        if row.get(key):
            out[key] = str(row[key])
    nested = row.get("sha256")
    if isinstance(nested, Mapping):
        for name, key in (("upf", "upf_sha256"), ("orbital", "orbital_sha256")):
            if nested.get(name):
                out[key] = str(nested[name])
    for name, key in (("upf", "upf_sha256"), ("orbital", "orbital_sha256")):
        if isinstance(row.get(name), str) and len(row[name]) == 64:
            out[key] = row[name]
    return out


class PairXCTables:
    """Direct two-centre XC tables <mu|v_xc[rho_i + rho_j]|nu> per sorted species pair (eV, ABACUS gauge).

    ``tables`` maps ``(a, b)`` with ``a <= b`` to a ``TorchRadialBlockTable`` (or a CPU ``RadialBlockTable`` compiled
    here). ``sources`` maps species to their declared source hashes. ``from_manifest`` reads the documented layout
    ``{"pairs": {"A|B": {"file", "sha256", "passed", "atomic_sources": {"A": {"upf", "orbital"}}}}}``.
    """

    def __init__(self, tables: Mapping[tuple[str, str], Any], *, sources: Mapping[str, Mapping[str, str]], manifest_sha256=None,
                 device="cuda", dtype=torch.float64, backend="auto"):
        from .radial import TorchRadialBlockTable
        self.tables = {}
        for (a, b), table in tables.items():
            if a > b:
                raise ValueError(f"pair XC tables are keyed by sorted species pairs, got {(a, b)}")
            self.tables[(a, b)] = table if isinstance(table, TorchRadialBlockTable) else TorchRadialBlockTable(table, device=device, dtype=dtype, backend=backend)
        self.sources = {s: normalize_sources(v) for s, v in sources.items()}
        self.manifest_sha256 = manifest_sha256
        # Providers are immutable for a plan's lifetime. Direct injection without
        # a manifest binds the actual compiled buffers once, outside forward.
        self.content_sha256 = {}
        if manifest_sha256 is None:
            for pair, table in self.tables.items():
                digest = hashlib.sha256(json.dumps({'left_shells': table.left_shells,
                    'right_shells': table.right_shells, 'support_bohr': table.support_bohr}, sort_keys=True).encode())
                for name, tensor in sorted(table.named_buffers()):
                    array = tensor.detach().cpu().contiguous().numpy()
                    digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
                    digest.update(array.tobytes())
                self.content_sha256['|'.join(pair)] = digest.hexdigest()

    @classmethod
    def from_manifest(cls, path, *, device="cuda", dtype=torch.float64, backend="auto", species=None, verify=True):
        from dptb.data.interfaces.p2_table import RadialBlockTable
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        tables, sources = {}, {}
        for key, row in manifest["pairs"].items():
            a, b = key.split("|")
            if species is not None and (a not in species or b not in species):
                continue
            if row.get("passed") is False:
                raise CandidateIdentityError(f"pair XC table {key} did not pass its build check")
            file = Path(row["file"])
            if not file.is_absolute():
                file = path.parent / file
            if verify and hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
                raise CandidateIdentityError(f"pair XC table {key} checksum mismatch")
            with np.load(file, allow_pickle=False) as z:
                tables[(a, b)] = RadialBlockTable(z["distances"], z["values_eV"], tuple(int(x) for x in z["left_shells"]),
                                                  tuple(int(x) for x in z["right_shells"]), float(z["support_bohr"]))
            for s, src in row.get("atomic_sources", {}).items():
                found = normalize_sources(src)
                if s in sources and sources[s] != found:
                    raise CandidateIdentityError(f"pair XC manifest declares two different sources for {s}")
                sources[s] = found
        return cls(tables, sources=sources, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), device=device, dtype=dtype, backend=backend)

    def has(self, a, b):
        return (min(a, b), max(a, b)) in self.tables

    def table(self, a, b):
        return self.tables[(min(a, b), max(a, b))]

    def identity(self):
        return {"family": "pair_xc", "manifest_sha256": self.manifest_sha256, "pairs": sorted("|".join(k) for k in self.tables),
                "content_sha256": self.content_sha256, "species_sources": self.sources}


class AtomicMoments:
    """Valence-density second moments M2 = int rho_val r^2 d^3r (Bohr^2) per species, for the cS zero point.

    ``from_json`` reads ``{symbol: {"M2_bohr2", "upf_sha256", "orbital_sha256", ...}}`` (the accepted atomic table).
    """

    def __init__(self, m2_bohr2: Mapping[str, float], *, sources: Mapping[str, Mapping[str, str]], manifest_sha256=None, approximation=None):
        self.m2 = {s: float(v) for s, v in m2_bohr2.items()}
        if any(not math.isfinite(v) or v <= 0 for v in self.m2.values()):
            raise ValueError("M2 moments must be finite and positive")
        self.sources = {s: normalize_sources(v) for s, v in sources.items()}
        self.manifest_sha256 = manifest_sha256
        self.approximation = approximation

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        return cls({s: r["M2_bohr2"] for s, r in rows.items()}, sources={s: r for s, r in rows.items()},
                   manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   approximation=next((r.get("approximation") for r in rows.values()), None))

    def has(self, s):
        return s in self.m2

    def m2_bohr2(self, s):
        return self.m2[s]

    def identity(self):
        return {"family": "atomic_moments", "manifest_sha256": self.manifest_sha256, "approximation": self.approximation,
                "species": sorted(self.m2), "m2_bohr2": dict(self.m2), "species_sources": self.sources}


# --------------------------------------------------------------------------- recipe
@dataclass(frozen=True)
class CandidateRecipe:
    """Versioned recipe identity; ``envxc_arm`` and ``stabilization`` must be stated explicitly."""
    envxc_arm: str
    stabilization: str
    order_policy: OrderPolicy
    onsite_radius_bohr: float = 27.0
    xc_functional: str = "LDA exchange + PZ81 correlation, unpolarized"
    density_definition: str = "normalized neutral valence (r=0 repair) + unscaled NLCC"
    zero_point: str = "cS: c = 4 pi sum(M2_valence) Ry_to_eV / (3 Omega), fully periodic"
    overlap_floor: float = 1e-8
    moment_density_floor: float = 0.0
    fusion: Mapping[str, Any] = field(default_factory=lambda: {"radial": False, "contraction": False})

    def __post_init__(self):
        for name, expected in (("xc_functional", XC_FUNCTIONAL), ("density_definition", DENSITY_DEFINITION), ("zero_point", ZERO_POINT)):
            if getattr(self, name) != expected:
                raise CandidateIdentityError(f"unsupported {name}: {getattr(self, name)!r}; implemented convention is {expected!r}")
        if self.envxc_arm not in ARMS:
            raise CandidateIdentityError(f"unsupported environment arm {self.envxc_arm!r}; expected one of {ARMS}")
        if self.stabilization not in STABILIZATIONS:
            raise CandidateIdentityError(f"unsupported stabilization {self.stabilization!r}; expected one of {STABILIZATIONS}")
        if not isinstance(self.order_policy, OrderPolicy):
            raise CandidateIdentityError("order_policy must be an OrderPolicy instance")
        if not (math.isfinite(self.onsite_radius_bohr) and self.onsite_radius_bohr > 0):
            raise CandidateIdentityError("onsite_radius_bohr must be positive")
        if not (math.isfinite(self.overlap_floor) and self.overlap_floor > 0
                and math.isfinite(self.moment_density_floor) and self.moment_density_floor >= 0):
            raise CandidateIdentityError("overlap floor must be finite positive and moment density floor finite nonnegative")
        if self.fusion.get("contraction"):
            # rejected under the strict FP32 packing gate (hopping fusion report); never a silent default
            raise CandidateIdentityError("fused contraction is not accepted for the candidate prior; evaluate it explicitly outside the recipe")

    def identity(self):
        return {"schema": RECIPE_SCHEMA, "envxc_arm": self.envxc_arm, "stabilization": self.stabilization,
                "order_policy": self.order_policy.identity(), "onsite_radius_bohr": self.onsite_radius_bohr,
                "xc_functional": self.xc_functional, "density_definition": self.density_definition, "zero_point": self.zero_point,
                "overlap_floor": self.overlap_floor, "moment_density_floor": self.moment_density_floor, "fusion": dict(self.fusion)}


# --------------------------------------------------------------------------- plan
class CandidatePriorPlan:
    def __init__(self, recipe: CandidateRecipe, *, table_bank, envxc_bank: EnvXCBank, onsite: OnsiteXCEvaluator,
                 onsite_identity: Mapping[str, Any], pair_xc: PairXCTables, atomic_moments: AtomicMoments,
                 topology_library=None, max_terms=10_000_000):
        self.recipe = recipe
        self.bank, self.envxc, self.onsite, self.pair_xc, self.moments = table_bank, envxc_bank, onsite, pair_xc, atomic_moments
        self.onsite_identity = dict(onsite_identity or {})
        self.library, self.max_terms = topology_library, int(max_terms)
        self.device, self.dtype = table_bank._anchor.device, table_bank._anchor.dtype
        self.identity = json.loads(json.dumps(self._validate(), sort_keys=True, default=str))
        self.identity_sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True, default=str).encode()).hexdigest()

    # ---- identity ---------------------------------------------------------------------------
    def _validate(self):
        r = self.recipe
        bank, store = self.bank, self.envxc.store
        if getattr(bank, "soc", None) is not None:
            raise CandidateIdentityError("the candidate prior is scalar; SOC projector banks are not supported")
        if self.envxc.device != self.device or self.envxc.dtype != self.dtype:
            raise CandidateIdentityError("environment-XC bank and table bank must share device and dtype")
        onsite_device = torch.device(self.onsite.device)
        if onsite_device.type == 'cuda' and onsite_device.index is None:
            onsite_device = torch.device('cuda', torch.cuda.current_device())
        if onsite_device != self.device:
            raise CandidateIdentityError("onsite evaluator and table bank must share a device")
        if abs(self.onsite.radius - r.onsite_radius_bohr) > 1e-12:
            raise CandidateIdentityError(f"onsite evaluator radius {self.onsite.radius} differs from the recipe {r.onsite_radius_bohr}")
        if r.envxc_arm in ("d2", "d2_moment") and store.background_nodes is None:
            raise CandidateIdentityError("the injected environment-XC tables have no background layers for a D2 arm")
        for key in ("species_sources", "potential", "density_definition"):
            if key not in self.onsite_identity:
                raise CandidateIdentityError(f"onsite_identity must declare {key!r} (the evaluator objects carry no provenance)")
        if self.onsite_identity["density_definition"] != r.density_definition:
            raise CandidateIdentityError("onsite density definition differs from the recipe")
        envxc_def = {row.get("density_definition") for row in store.manifest.get("sources", {}).values() if isinstance(row, Mapping)}
        envxc_def.discard(None)
        if envxc_def and any(not d.startswith(r.density_definition.split(";")[0]) for d in envxc_def):
            raise CandidateIdentityError(f"environment-XC tables declare a different density definition: {sorted(envxc_def)}")
        p2 = bank.p2.species
        families = {
            "p2": {"manifest_sha256": bank.p2_manifest_sha256, "species": {s: normalize_sources(row) for s, row in p2.items()}},
            "p23": {"manifest_sha256": getattr(bank.p23, "manifest_sha256", None),
                    "species": {s: normalize_sources(row) for s, row in getattr(bank.p23, "species", {}).items()}},
            "envxc": {"manifest_sha256": store.manifest_sha256, "build_identity": store.build_identity,
                      "species": {s: normalize_sources(store.manifest.get("sources", {}).get(s)) for s in store.species}},
            "pair_xc": {**self.pair_xc.identity(), "species": dict(self.pair_xc.sources)},
            "atomic_moments": {**self.moments.identity(), "species": dict(self.moments.sources)},
            "onsite": {"species": {s: normalize_sources(v) for s, v in self.onsite_identity["species_sources"].items()},
                       "potential": self.onsite_identity["potential"], "engine": self.onsite.engine},
        }
        if bank.overlap is not bank.p2:
            families["overlap"] = {"manifest_sha256": getattr(bank.overlap, "manifest_sha256", None),
                                   "source_p2_manifest_sha256": bank.overlap.manifest.get("source_p2_manifest_sha256")}
        # cross-family provenance for every species known to every family: AO shells, cutoff and declared sources
        common = set(p2) & set(store.species) & set(self.moments.m2) & set(families["onsite"]["species"])
        problems = []
        for s in sorted(common):
            problems.extend(self._species_problems(s, families))
        if problems:
            raise CandidateIdentityError("cross-family identity mismatch:\n  " + "\n  ".join(problems))
        return {"schema": RECIPE_SCHEMA, "recipe": r.identity(), "families": families, "ry_to_ev": bank.ry_to_ev,
                "validated_species": sorted(common), "device": str(self.device), "dtype": str(self.dtype)}

    def _species_problems(self, s, families):
        problems = []
        p2 = self.bank.p2.species[s]; store = self.envxc.store
        shells = tuple(int(l) for l in p2["orbital_shells"])
        if store.orbital_shells(s) != shells:
            problems.append(f"{s}: AO shells P2 {shells} vs envxc {store.orbital_shells(s)}")
        if abs(store.orbital_cutoff(s) - float(p2["orbital_cutoff_bohr"])) > 1e-9:
            problems.append(f"{s}: orbital cutoff P2 {p2['orbital_cutoff_bohr']} vs envxc {store.orbital_cutoff(s)}")
        declared = {name: fam["species"].get(s, {}) for name, fam in families.items() if "species" in fam}
        reference = declared['p2']
        required = set(reference) & {'upf_sha256', 'orbital_sha256'}
        if not required and 'source_sha256' in reference:
            required = {'source_sha256'}
        for name, src in declared.items():
            if not src:
                problems.append(f"{s}: family {name} declares no source hash")
            elif not required or not required.issubset(src):
                problems.append(f"{s}: family {name} has no comparable complete P2 source identity; required {sorted(required)}")
        for key in SOURCE_KEYS:
            values = {name: src[key] for name, src in declared.items() if key in src}
            if len(set(values.values())) > 1:
                problems.append(f"{s}: {key} differs across families {values}")
        return problems

    def check_species(self, symbols, edge_index=None):
        missing = []
        for s in sorted(set(symbols)):
            if s not in self.bank.p2.species:
                missing.append(f"P2 tables: {s}")
            if s not in self.envxc.store.species:
                missing.append(f"environment-XC tables: {s}")
            if not self.moments.has(s):
                missing.append(f"atomic moments: {s}")
            if s not in self.onsite_identity["species_sources"]:
                missing.append(f"onsite identity: {s}")
        pairs = ({tuple(sorted((symbols[i], symbols[j]))) for i, j in np.asarray(edge_index).T}
                 if edge_index is not None else {(a, b) for a in symbols for b in symbols if a <= b})
        for a, b in sorted(pairs):
            if not self.pair_xc.has(a, b):
                missing.append(f"pair XC table: {a}|{b}")
        if missing:
            raise CandidateIdentityError("structure species not covered: " + ", ".join(missing))
        problems = []
        for s in sorted(set(symbols)):
            if s not in self.identity["validated_species"]:
                problems.extend(self._species_problems(s, self.identity["families"]))
        if problems:
            raise CandidateIdentityError("cross-family identity mismatch:\n  " + "\n  ".join(problems))

    # ---- structures -------------------------------------------------------------------------
    def prepare(self, g: Mapping[str, Any]):
        return PreparedCandidate(self, g)


class PreparedCandidate:
    """Geometry-bound plans of one structure; ``forward`` returns eV AO blocks in the graph row order."""

    def __init__(self, plan: CandidatePriorPlan, g):
        t0 = time.perf_counter()
        self.plan = plan
        symbols = [str(s) for s in g["symbols"]]
        pbc = tuple(bool(x) for x in g.get("pbc", (True, True, True)))
        if not all(pbc):
            raise CandidateIdentityError("the cS zero point of this recipe is defined for fully periodic cells only")
        plan.check_species(symbols, g['edge_index'])
        pos = np.array(g["positions_bohr"], dtype=np.float64, copy=True); cell = np.array(g["cell_bohr"], dtype=np.float64, copy=True)
        ei = np.array(g["edge_index"], dtype=np.int64, copy=True); sh = np.array(g["edge_cell_shift"], dtype=np.int64, copy=True)
        self.geometry = dict(symbols=symbols, positions_bohr=pos, cell_bohr=cell, edge_index=ei, edge_cell_shift=sh, pbc=pbc)
        r = plan.recipe
        bank = plan.bank
        self.assembly = bank.prepare(symbols, pos, cell, ei, sh, pbc=pbc, topology="native", library=plan.library, max_terms=plan.max_terms)
        self.edge_vna = bank.prepare_edge_vna(symbols, pos, cell, ei, sh, pbc=pbc, library=plan.library, max_terms=plan.max_terms)
        self.envxc = plan.envxc.prepare(symbols, pos, cell, ei, sh, pbc=pbc, arms=(r.envxc_arm,), library=plan.library, max_terms=plan.max_terms,
                                        overlap_floor=r.overlap_floor, moment_density_floor=r.moment_density_floor, stabilization=r.stabilization)
        if r.fusion.get("radial") and plan.device.type == "cuda":
            from .fusion import enable_fusion
            enable_fusion(self.assembly, radial=True, contraction=False)
            enable_fusion(self.edge_vna, radial=True, contraction=False)
            self.envxc.enable_fusion(include_layers=True, contraction=False)
        self.width = self.assembly.width
        # direct pair XC: sorted-pair tables, reversed edges use the transposed block of the reversed vector
        vec = pos[ei[1]] - pos[ei[0]] + sh @ cell
        groups: dict[tuple[str, str, bool], list[int]] = {}
        for k, (i, j) in enumerate(ei.T):
            a, b = symbols[i], symbols[j]
            groups.setdefault((min(a, b), max(a, b), a > b), []).append(k)
        self.pair_groups = [(plan.pair_xc.table(a, b), torch.as_tensor(ids, device=plan.device),
                             torch.as_tensor(vec[ids] * (-1.0 if rev else 1.0), device=plan.device, dtype=plan.dtype), rev)
                            for (a, b, rev), ids in groups.items()]
        volume = abs(float(np.linalg.det(cell)))
        if not volume > 0:
            raise ValueError("degenerate cell")
        self.c_ev = 4.0 * math.pi * sum(plan.moments.m2_bohr2(s) for s in symbols) * bank.ry_to_ev / (3.0 * volume)
        self.orders, self.order_checks = r.order_policy.select(plan.onsite, self.geometry, self.width)
        self.prepare_seconds = time.perf_counter() - t0

    def pair_xc_blocks(self):
        E, w = self.assembly.nedges, self.width
        out = torch.zeros((E, w, w), device=self.plan.device, dtype=self.plan.dtype)
        for table, ids, vec, rev in self.pair_groups:
            block = table(vec)
            if rev:
                block = block.transpose(1, 2)
            out[ids, :block.shape[1], :block.shape[2]] = block
        return out

    def forward(self):
        arm = self.plan.recipe.envxc_arm
        b = self.assembly()
        vna = self.edge_vna()["edge_vna_ao_ev"]
        pair = self.pair_xc_blocks()
        site = self.plan.onsite(self.geometry, self.width, self.orders)
        env = self.envxc(edge_overlap_ao=b["edge_overlap_ao"])
        node = b["node_p23_ao_ev"] + site + self.c_ev * b["node_overlap_ao"]
        edge = b["edge_p2_ao_ev"] + vna + pair + self.c_ev * b["edge_overlap_ao"] + env[arm]
        diagnostics = {"orders": [list(o) for o in self.orders], "order_checks": self.order_checks, "c_ev": self.c_ev,
                       "onsite_stats": self.plan.onsite.last_stats, "onsite_bank_rebuilds": self.plan.onsite.bank_rebuilds,
                       "envxc": {k: v for k, v in env["diagnostics"].items() if not isinstance(v, (torch.Tensor, np.ndarray))},
                       "recipe_identity_sha256": self.plan.identity_sha256, "prepare_seconds": self.prepare_seconds}
        return {"node_ao_ev": node, "edge_ao_ev": edge, "node_overlap_ao": b["node_overlap_ao"], "edge_overlap_ao": b["edge_overlap_ao"],
                "components": {"node_p23_ao_ev": b["node_p23_ao_ev"], "onsite_xc_ao_ev": site, "edge_p2_ao_ev": b["edge_p2_ao_ev"],
                               "edge_vna_ao_ev": vna, "pair_xc_ao_ev": pair, "envxc_ao_ev": env[arm]},
                "edge_index": self.assembly.edge_index, "edge_cell_shift": self.assembly.edge_cell_shift, "diagnostics": diagnostics}

    __call__ = forward

    def feature_plan(self, idp, *, output_dtype=torch.float32, mapping="compact", packing_backend="cuda"):
        """Pack AO blocks into checkpoint RME features with the same mapper as the trained models."""
        from .assembly import NACFFeaturePlan
        return NACFFeaturePlan(self.assembly, idp, output_dtype=output_dtype, mapping=mapping, packing_backend=packing_backend)


__all__ = ["RECIPE_SCHEMA", "CandidateIdentityError", "OrderPolicy", "FixedOrderPolicy", "ConvergenceOrderPolicy", "PairXCTables",
           "AtomicMoments", "CandidateRecipe", "CandidatePriorPlan", "PreparedCandidate", "normalize_sources"]
