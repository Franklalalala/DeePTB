"""Explicit, versioned full NACF candidate prior (independent audit 2026-09-20, item F3; reviewer fixes 2026-09-21).

    node = P23 + onsite XC (accepted atom-centred local quadrature, fixed neighbourhood) + c S
    edge = P2 + edge VNA3c + direct pair XC + c S + environment XC (one grid-free arm)
    c    = 4 pi (sum_a M2_a) Ry_to_eV / (3 Omega)       (ABACUS potential zero point; fully periodic cells only)

This is the composition the fixed100 evaluation harness assembled by hand (accepted ``prepare`` / ``site_eval`` /
``xc_edges`` / ``envxc_plan``), as one maintained entry with an explicit recipe identity. Everything is injected:
the P2/P23/overlap table bank, the environment-XC bank, the onsite evaluator (species quadratures, density bank,
potential), the direct pair-XC tables, the atomic second moments, and the quadrature order policy. Nothing is
discovered from directories, mpids, H, H0 or labels. The existing :class:`NACFGeometryPredictor` (P23/P2 prior of
the trained checkpoints) is untouched; this entry does not change what any checkpoint was trained on.

Module map. :mod:`dptb.nacf.candidate_policy` declares the recipe (canonical XC vocabulary, frozen order
policies, immutable fusion settings); :mod:`dptb.nacf.candidate_checks` validates raw structure arrays and
provider declarations (pair-XC shell headers against P2, source hashes, XC declaration); this module holds the
injected numerical providers (:class:`PairXCTables`, :class:`AtomicMoments`), binds a recipe to them
(:class:`CandidatePriorPlan`) and evaluates structures (:class:`PreparedCandidate`). Every public name of the
0920 entry is still importable from here.

Identity. At construction every table family must be bound to the same P2 manifest (already enforced by the bank
for P23/overlap), the recipe options must be supported by the injected tables (arm, stabilization, background
layers), every species covered by all families must agree on AO shells, orbital cutoff and the UPF/ORB source
hashes each family declares, every pair-XC table header must carry the P2 shell sequences of its two species
(source hashes and matrix shapes do not establish the shell gauge), and the onsite provider must declare the
recipe's functional. At ``prepare`` the raw structure arrays are validated before any cast, the species of the
structure must be covered by every family and the bound recipe must still have its construction identity. A
missing or contradictory identity raises :class:`CandidateIdentityError`, a malformed structure
:class:`CandidateInputError`; nothing falls back silently.

Boundaries stated honestly: the onsite evaluator objects (quadrature, density splines, potential) carry no
provenance of their own, so the caller passes ``onsite_identity`` (species sources, canonical ``xc_functional`` or
a recognizable ``potential`` label, density definition) and this module can only check it against the recipe and
the other families, not derive what the callable computes. The order policy is geometry-only: ``FixedOrderPolicy``
(a choice, reported without any convergence claim) or the accepted convergence rule re-run on the new structure
(``ConvergenceOrderPolicy``), never a fixed100 lookup. Scalar (non-SOC) blocks only; SOC banks are rejected.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .candidate_checks import (SOURCE_KEYS, CandidateInputError, declared_xc, normalize_sources, pair_shell_problems,
                               species_source_problems, validated_geometry)
from .candidate_policy import (DENSITY_DEFINITION, RECIPE_SCHEMA, XC_FUNCTIONAL, XC_KEY, ZERO_POINT, CandidateIdentityError,
                               CandidateRecipe, ConvergenceOrderPolicy, FixedOrderPolicy, FusionSettings, OrderPolicy)
from .envxc import EnvXCBank
from .onsite import OnsiteXCEvaluator


# --------------------------------------------------------------------------- injected providers
class PairXCTables:
    """Direct two-centre XC tables <mu|v_xc[rho_i + rho_j]|nu> per sorted species pair (eV, ABACUS gauge).

    ``tables`` maps ``(a, b)`` with ``a <= b`` to a ``TorchRadialBlockTable`` (or a CPU ``RadialBlockTable`` compiled
    here). ``sources`` maps species to their declared source hashes. ``from_manifest`` reads the documented layout
    ``{"pairs": {"A|B": {"file", "sha256", "passed", "atomic_sources": {"A": {"upf", "orbital"}}}}}``. The plan
    compares each table's ordered ``left_shells``/``right_shells`` header with the P2 species shells at binding.
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


# --------------------------------------------------------------------------- plan
def _snapshot(value):
    """JSON round trip: an owned, plain-data copy of a caller mapping (tuples become lists, unknown objects strings)."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


class CandidatePriorPlan:
    """A recipe bound to injected providers, validated once; ``prepare`` evaluates structures.

    Binding copies ``onsite_identity`` and snapshots ``recipe.identity()``; the snapshot is hashed in
    ``identity_sha256`` and compared again before every ``prepare``, so a mutated custom policy is refused rather
    than evaluated under a stale hash. Providers and their tensors must stay immutable for the plan lifetime.
    """

    def __init__(self, recipe: CandidateRecipe, *, table_bank, envxc_bank: EnvXCBank, onsite: OnsiteXCEvaluator,
                 onsite_identity: Mapping[str, Any], pair_xc: PairXCTables, atomic_moments: AtomicMoments,
                 topology_library=None, max_terms=10_000_000):
        if not isinstance(recipe, CandidateRecipe):
            raise CandidateIdentityError("recipe must be a CandidateRecipe")
        self.recipe = recipe
        self.bank, self.envxc, self.onsite, self.pair_xc, self.moments = table_bank, envxc_bank, onsite, pair_xc, atomic_moments
        self.onsite_identity = _snapshot(dict(onsite_identity or {}))
        self.library, self.max_terms = topology_library, int(max_terms)
        self.device, self.dtype = table_bank._anchor.device, table_bank._anchor.dtype
        self.identity = _snapshot(self._validate())
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
        onsite_xc = declared_xc(self.onsite_identity, r)
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
                       "potential": self.onsite_identity["potential"], "xc_functional": onsite_xc, "engine": self.onsite.engine},
        }
        if bank.overlap is not bank.p2:
            families["overlap"] = {"manifest_sha256": getattr(bank.overlap, "manifest_sha256", None),
                                   "source_p2_manifest_sha256": bank.overlap.manifest.get("source_p2_manifest_sha256")}
        # cross-family provenance for every species known to every family: AO shells, cutoff and declared sources
        common = set(p2) & set(store.species) & set(self.moments.m2) & set(families["onsite"]["species"])
        problems = []
        for s in sorted(common):
            problems.extend(self._species_problems(s, families))
        # every pair-XC table header must carry the P2 shell sequences of its species: equal AO counts are not enough
        problems.extend(pair_shell_problems(self.pair_xc.tables, p2))
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
        problems.extend(species_source_problems(s, declared))
        return problems

    def check_bound_identity(self):
        """Refuse to continue when the bound recipe (or its order policy) no longer has its construction identity."""
        if _snapshot(self.recipe.identity()) != self.identity["recipe"]:
            raise CandidateIdentityError("the bound recipe or order policy changed after the plan was constructed; "
                                         "the reported identity_sha256 would be stale. Construct a new plan.")

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
        """Bind one structure: raw array validation, coverage, geometry-only plans and quadrature orders."""
        self.check_bound_identity()
        return PreparedCandidate(self, g)


class PreparedCandidate:
    """Geometry-bound plans of one structure; ``forward`` returns eV AO blocks in the graph row order."""

    def __init__(self, plan: CandidatePriorPlan, g):
        t0 = time.perf_counter()
        self.plan = plan
        # raw integer/finite/shape/range checks and owned copies come first: no cast, provider or plan sees the caller's
        # arrays before they are known to be a legal graph
        self.geometry = validated_geometry(g)
        symbols, pos, cell, ei, sh, pbc = (self.geometry[k] for k in ("symbols", "positions_bohr", "cell_bohr", "edge_index", "edge_cell_shift", "pbc"))
        if not all(pbc):
            raise CandidateIdentityError("the cS zero point of this recipe is defined for fully periodic cells only")
        plan.check_species(symbols, ei)
        r = plan.recipe
        bank = plan.bank
        self.assembly = bank.prepare(symbols, pos, cell, ei, sh, pbc=pbc, topology="native", library=plan.library, max_terms=plan.max_terms)
        self.edge_vna = bank.prepare_edge_vna(symbols, pos, cell, ei, sh, pbc=pbc, library=plan.library, max_terms=plan.max_terms)
        self.envxc = plan.envxc.prepare(symbols, pos, cell, ei, sh, pbc=pbc, arms=(r.envxc_arm,), library=plan.library, max_terms=plan.max_terms,
                                        overlap_floor=r.overlap_floor, moment_density_floor=r.moment_density_floor, stabilization=r.stabilization)
        if r.fusion.radial and plan.device.type == "cuda":
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

    def convergence_summary(self):
        """``(checked, converged)`` over the order checks: ``converged`` is ``None`` unless every atom was actually compared."""
        checks = self.order_checks
        checked = bool(checks) and all(c.get("convergence_checked") for c in checks)
        return checked, (all(bool(c.get("converged")) for c in checks) if checked else None)

    def forward(self):
        arm = self.plan.recipe.envxc_arm
        b = self.assembly()
        vna = self.edge_vna()["edge_vna_ao_ev"]
        pair = self.pair_xc_blocks()
        site = self.plan.onsite(self.geometry, self.width, self.orders)
        env = self.envxc(edge_overlap_ao=b["edge_overlap_ao"])
        node = b["node_p23_ao_ev"] + site + self.c_ev * b["node_overlap_ao"]
        edge = b["edge_p2_ao_ev"] + vna + pair + self.c_ev * b["edge_overlap_ao"] + env[arm]
        checked, converged = self.convergence_summary()
        diagnostics = {"orders": [list(o) for o in self.orders], "order_checks": self.order_checks,
                       "convergence_checked": checked, "converged": converged, "c_ev": self.c_ev,
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


__all__ = ["RECIPE_SCHEMA", "XC_KEY", "XC_FUNCTIONAL", "DENSITY_DEFINITION", "ZERO_POINT", "SOURCE_KEYS",
           "CandidateIdentityError", "CandidateInputError", "OrderPolicy", "FixedOrderPolicy", "ConvergenceOrderPolicy",
           "FusionSettings", "PairXCTables", "AtomicMoments", "CandidateRecipe", "CandidatePriorPlan", "PreparedCandidate",
           "normalize_sources", "validated_geometry"]
