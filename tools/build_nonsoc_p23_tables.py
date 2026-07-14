#!/usr/bin/env python3
"""Build reusable factorized local-VNA three-centre tables for non-SOC P23.

The table stores ``<q_K|phi_A(R)>`` where ``q_K = VNA_K p_K`` and the
OpenMX-inspired radial projectors ``p_K`` are diagonal in a V-weighted metric.
At structure assembly time one local three-centre block is recovered as

    B(K,i).T @ diag(epsilon_K) @ B(K,j).

Only AO/VNA species pairs proven active by the raw200 factor census are built.
The P2 table manifest is required so ORB/UPF identity cannot silently drift.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import eval_legendre, spherical_jn

from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    canonical_species_source_identity,
    real_sph_abacus,
    species_source_identity_sha256,
)
from dptb.data.interfaces.p23_table import (
    P23_FACTOR_SUPPORT_SEMANTICS,
    P23_VNA_TABLE_SCHEMA,
)

# Executing a tool puts this directory on sys.path.  Reusing the audited P2
# parser/SBT data objects keeps shell order and radial normalization identical.
from build_nonsoc_p2_tables import (  # type: ignore
    Channel,
    OrbitalLike,
    _angular_grid,
    _channel_transforms,
    _distance_grid,
    _load_gate1,
    _load_species,
    _selected_cases,
)


RY_TO_EV = 13.605698
HARTREE_TO_EV = 27.211386245988
HEARTBEAT_SCHEMA = "deeptb.p23_factorized_vna_table_heartbeat/v1"
PARENT_P2_OPERATOR = "T + unique(VNA_i,VNA_j) + factorized all-K Vnl"
PARENT_P2_SOC_MODE = "non-SOC scalar spin trace"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _heartbeat(path: Path, *, stage: str, **value: Any) -> None:
    _atomic_json(
        path,
        {
            "schema": HEARTBEAT_SCHEMA,
            "stage": stage,
            "pid": os.getpid(),
            "updated_unix": time.time(),
            **value,
        },
    )


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _radial_inner(
    r: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    potential_ev: np.ndarray | None = None,
) -> float:
    weight = 1.0 if potential_ev is None else potential_ev
    return _integrate(r * r * left * right * weight, r)


def _vna_cutoff(
    radial: np.ndarray, values_ry: np.ndarray, relative_tolerance: float
) -> float:
    radial = np.asarray(radial, dtype=np.float64)
    values = np.asarray(values_ry, dtype=np.float64)
    peak = float(np.max(np.abs(values), initial=0.0))
    threshold = max(1.0e-14, float(relative_tolerance) * peak)
    active = np.flatnonzero(np.abs(values) > threshold)
    if active.size == 0:
        raise ValueError("neutral-atom potential has no active radial samples")
    return float(radial[min(int(active[-1]) + 1, radial.size - 1)])


@dataclass(frozen=True)
class VNAProjectors:
    orbital: OrbitalLike
    epsilon_ao: np.ndarray
    radial_epsilon: np.ndarray
    radial_l: np.ndarray
    radial_n: np.ndarray
    metadata: dict[str, Any]


def _initial_candidate(
    orbital: OrbitalLike,
    vna_hartree: np.ndarray,
    l_value: int,
    radial_index: int,
) -> tuple[np.ndarray, str]:
    by_l: dict[int, list[Channel]] = {}
    for channel in orbital.channels:
        by_l.setdefault(int(channel.l), []).append(channel)
    lmax = max(by_l)
    seed = 0.1 * vna_hartree + 1.0e-13
    if l_value <= lmax:
        channels = by_l.get(l_value, [])
        if not channels:
            raise ValueError(f"orbital has no radial channel for l={l_value}")
        if radial_index < len(channels):
            return np.asarray(channels[radial_index].radial).copy(), "pao"
        return (
            np.asarray(channels[0].radial) * np.power(seed, radial_index),
            "pao0_times_vna_power",
        )
    highest = by_l[lmax]
    if radial_index < len(highest):
        return (
            np.asarray(highest[radial_index].radial)
            * np.power(orbital.r, l_value - lmax),
            "highest_l_times_r_power",
        )
    return (
        np.asarray(highest[-1].radial)
        * np.power(seed, radial_index - len(highest) + 1),
        "highest_l_last_times_vna_power",
    )


def build_vna_projectors(
    orbital: OrbitalLike,
    vna_r_bohr: np.ndarray,
    vna_values_ry: np.ndarray,
    *,
    radial_rank: int,
    l_buffer: int,
    vna_cutoff_bohr: float,
    standard_norm_tol: float = 1.0e-18,
    metric_tol_ev: float = 1.0e-12,
    reorthogonalization_passes: int = 2,
) -> VNAProjectors:
    """Build the same diagonal V-metric projector family as the Si pilot."""

    if radial_rank <= 0 or l_buffer < 0:
        raise ValueError("radial_rank must be positive and l_buffer non-negative")
    r = np.asarray(orbital.r, dtype=np.float64)
    source_r = np.asarray(vna_r_bohr, dtype=np.float64)
    source_v = np.asarray(vna_values_ry, dtype=np.float64) * RY_TO_EV
    potential_ev = np.interp(r, source_r, source_v, left=source_v[0], right=0.0)
    potential_ev[r > float(vna_cutoff_bohr) + 1.0e-12] = 0.0
    potential_hartree = potential_ev / HARTREE_TO_EV
    orbital_lmax = max(int(channel.l) for channel in orbital.channels)
    # q = V p cannot extend beyond either V or the finite-support projector p.
    # The OpenMX-inspired p seeds are built on the PAO grid, so advertising the
    # longer physical VNA tail as q support would retain finite-k ringing in a
    # region where the exact factor overlap is mathematically zero.
    projector_cutoff_bohr = min(
        float(vna_cutoff_bohr), float(orbital.rcut), float(r[-1])
    )

    channels: list[Channel] = []
    radial_epsilon: list[float] = []
    radial_l: list[int] = []
    radial_n: list[int] = []
    seed_kinds: list[str] = []
    normalized_offdiag_max = 0.0
    for l_value in range(orbital_lmax + int(l_buffer) + 1):
        projectors_l: list[np.ndarray] = []
        vnorm_l: list[float] = []
        for radial_index in range(int(radial_rank)):
            candidate, seed_kind = _initial_candidate(
                orbital, potential_hartree, l_value, radial_index
            )
            norm2 = _radial_inner(r, candidate, candidate)
            if not math.isfinite(norm2) or norm2 <= standard_norm_tol:
                raise ValueError(
                    f"near-null standard projector l={l_value} n={radial_index}: {norm2}"
                )
            projector = candidate / math.sqrt(norm2)
            for _ in range(int(reorthogonalization_passes)):
                for previous, previous_vnorm in zip(projectors_l, vnorm_l):
                    projector -= (
                        _radial_inner(r, previous, projector, potential_ev)
                        / previous_vnorm
                    ) * previous
            vnorm = _radial_inner(r, projector, projector, potential_ev)
            if not math.isfinite(vnorm) or abs(vnorm) <= metric_tol_ev:
                raise ValueError(
                    f"near-null V metric l={l_value} n={radial_index}: {vnorm:.6e} eV"
                )
            weighted = potential_ev * projector
            channels.append(
                Channel(
                    l=l_value,
                    radial=np.asarray(weighted, dtype=np.float64),
                    cutoff_bohr=projector_cutoff_bohr,
                    source_index=len(channels),
                )
            )
            projectors_l.append(projector)
            vnorm_l.append(vnorm)
            radial_epsilon.append(1.0 / vnorm)
            radial_l.append(l_value)
            radial_n.append(radial_index)
            seed_kinds.append(seed_kind)

        metric = np.asarray(
            [
                [
                    _radial_inner(r, left, right, potential_ev)
                    for right in projectors_l
                ]
                for left in projectors_l
            ],
            dtype=np.float64,
        )
        scale = np.sqrt(np.abs(np.diag(metric)))
        denom = np.maximum(scale[:, None] * scale[None, :], metric_tol_ev)
        normalized = np.abs(metric) / denom
        normalized -= np.diag(np.diag(normalized))
        normalized_offdiag_max = max(
            normalized_offdiag_max, float(np.max(normalized, initial=0.0))
        )

    q_orbital = OrbitalLike(
        symbol=orbital.symbol,
        r=r,
        channels=tuple(channels),
        source=orbital.source,
    )
    epsilon_ao = np.concatenate(
        [
            np.full(2 * l_value + 1, epsilon, dtype=np.float64)
            for l_value, epsilon in zip(radial_l, radial_epsilon)
        ]
    )
    epsilon_abs = np.abs(np.asarray(radial_epsilon, dtype=np.float64))
    metadata = {
        "radial_rank": int(radial_rank),
        "l_buffer": int(l_buffer),
        "lmax": int(orbital_lmax + l_buffer),
        "physical_vna_cutoff_bohr": float(vna_cutoff_bohr),
        "projector_cutoff_bohr": projector_cutoff_bohr,
        "radial_projectors": len(radial_epsilon),
        "angular_projectors": int(q_orbital.norb),
        "normalized_v_metric_offdiag_max": normalized_offdiag_max,
        "epsilon_abs_min_1_per_ev": float(np.min(epsilon_abs)),
        "epsilon_abs_max_1_per_ev": float(np.max(epsilon_abs)),
        "epsilon_abs_dynamic_range": float(
            np.max(epsilon_abs)
            / max(float(np.min(epsilon_abs)), np.finfo(np.float64).tiny)
        ),
        "seed_kinds": seed_kinds,
    }
    return VNAProjectors(
        orbital=q_orbital,
        epsilon_ao=epsilon_ao,
        radial_epsilon=np.asarray(radial_epsilon, dtype=np.float64),
        radial_l=np.asarray(radial_l, dtype=np.int16),
        radial_n=np.asarray(radial_n, dtype=np.int16),
        metadata=metadata,
    )


def _expanded_transform(
    orbital: OrbitalLike, k: np.ndarray, *, left: bool
) -> np.ndarray:
    channel_values = _channel_transforms(orbital, k)
    output = np.empty((len(k), orbital.norb), dtype=np.complex128)
    for column, descriptor in enumerate(orbital.descriptors()):
        phase = (1j) ** int(descriptor.l) if left else (-1j) ** int(descriptor.l)
        output[:, column] = phase * channel_values[int(descriptor.channel_index)]
    return output


def _angular_coefficients(
    left: OrbitalLike, right: OrbitalLike, *, n_mu: int, n_phi: int
) -> tuple[np.ndarray, ...]:
    left_items = left.descriptors()
    right_items = right.descriptors()
    directions, weights = _angular_grid(n_mu, n_phi)
    y_left = np.column_stack(
        [real_sph_abacus(item.l, item.m, directions) for item in left_items]
    )
    y_right = np.column_stack(
        [real_sph_abacus(item.l, item.m, directions) for item in right_items]
    )
    lmax = max((item.l for item in left_items), default=0) + max(
        (item.l for item in right_items), default=0
    )
    mu = directions[:, 2]
    return tuple(
        np.einsum(
            "pa,p,pb->ab",
            y_left,
            eval_legendre(l_value, mu) * weights,
            y_right,
            optimize=True,
        )
        for l_value in range(lmax + 1)
    )


def _build_values_fast(
    *,
    left: OrbitalLike,
    right: OrbitalLike,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    coefficients: Sequence[np.ndarray],
    k: np.ndarray,
    factor: np.ndarray,
    distances: np.ndarray,
    support_bohr: float,
) -> np.ndarray:
    """Build only the exact axial ``m_left == m_right`` matrix entries."""

    left_items = left.descriptors()
    right_items = right.descriptors()
    pair_left: list[int] = []
    pair_right: list[int] = []
    for row, left_item in enumerate(left_items):
        for col, right_item in enumerate(right_items):
            if int(left_item.m) == int(right_item.m):
                pair_left.append(row)
                pair_right.append(col)
    rows = np.asarray(pair_left, dtype=np.int64)
    cols = np.asarray(pair_right, dtype=np.int64)
    radial = left_transform[:, rows] * right_transform[:, cols]
    pair_values = np.zeros((len(distances), len(rows)), dtype=np.float64)
    kd = np.outer(np.asarray(distances, dtype=np.float64), k)
    for l_value, coefficient_matrix in enumerate(coefficients):
        coefficient = np.asarray(coefficient_matrix)[rows, cols]
        active = np.flatnonzero(np.abs(coefficient) > 1.0e-13)
        if active.size == 0:
            continue
        kernel = spherical_jn(l_value, kd) * factor[None, :]
        phase = (2 * l_value + 1) * ((-1j) ** l_value)
        weighted_radial = radial[:, active] * coefficient[active][None, :]
        pair_values[:, active] += (phase * (kernel @ weighted_radial)).real
    values = np.zeros(
        (len(distances), left.norb, right.norb), dtype=np.float64
    )
    values[:, rows, cols] = pair_values
    values[np.asarray(distances) >= float(support_bohr) - 1.0e-12] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("factorized VNA radial table produced non-finite values")
    return values


def _pair_filename(vna: str, ao: str) -> str:
    return f"{vna}__{ao}.npz"


def _pair_key(vna: str, ao: str) -> str:
    return f"{vna}|{ao}"


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = args.output.resolve()
    if root.exists() and args.overwrite:
        if root == Path(root.anchor) or root == Path.home():
            raise ValueError(f"refusing unsafe overwrite target {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"

    factor_report = json.loads(args.factor_census.read_text(encoding="utf-8"))
    if factor_report.get("schema") != "deeptb.vna3c_factor_reuse_census/v1":
        raise ValueError("factor census schema mismatch")
    if factor_report.get("failures"):
        raise ValueError("factor census contains failures")
    if int(factor_report.get("summary", {}).get("case_count", 0)) <= 0:
        raise ValueError("factor census contains no cases")
    contract = factor_report.get("contract", {})
    base_contract = contract.get("base_geometry_contract", {})
    if int(contract.get("l_buffer", -1)) != int(args.l_buffer):
        raise ValueError("factor census and builder l_buffer differ")
    if base_contract.get("endpoint_policy") != (
        "exclude exact (i,0) and (j,Rj) identities already in P2 base"
    ):
        raise ValueError("factor census endpoint policy is incompatible")
    vna_tolerance = float(base_contract.get("vna_relative_tail_tolerance", -1.0))
    if not math.isclose(vna_tolerance, float(args.vna_relative_tolerance)):
        raise ValueError("factor census and builder VNA cutoff tolerance differ")

    active_pairs = sorted(
        {
            (str(row["vna_species"]), str(row["ao_species"]))
            for row in factor_report["factor_buckets"]
            if int(row.get("factor_uses", 0)) > 0
        }
    )
    if not active_pairs:
        raise ValueError("factor census has no active species pairs")

    p2_manifest_path = args.p2_table_root.resolve() / "manifest.json"
    p2_manifest = json.loads(p2_manifest_path.read_text(encoding="utf-8"))
    if p2_manifest.get("schema") != P2_TABLE_SCHEMA:
        raise ValueError("base P2 table schema is incompatible")
    if p2_manifest.get("complete") is not True:
        raise ValueError("base P2 table is not complete")
    p2_settings = p2_manifest.get("build_settings", {})
    if p2_manifest.get("energy_unit") != "Ry":
        raise ValueError("base P2 table energy unit must be Ry")
    if p2_settings.get("operator") != PARENT_P2_OPERATOR:
        raise ValueError(
            "base P2 operator does not prove endpoint-VNA/nonlocal coverage: "
            f"{p2_settings.get('operator')!r}"
        )
    if p2_settings.get("soc_mode") != PARENT_P2_SOC_MODE:
        raise ValueError("base P2 table is not the required non-SOC scalar contract")
    # Hardened P2 tables declare this identity explicitly.  The qualified
    # raw200 production table predates that additive manifest field, but its
    # per-species ORB/UPF hashes are complete.  Derive the exact same canonical
    # identity and reject any inconsistent declaration instead of forcing an
    # expensive rebuild of immutable P2 values.
    p2_species_identity = canonical_species_source_identity(p2_manifest["species"])
    p2_identity = species_source_identity_sha256(p2_species_identity)
    declared_p2_identity = p2_manifest.get("species_source_identity_sha256")
    if declared_p2_identity not in (None, "") and (
        str(declared_p2_identity).lower() != p2_identity
    ):
        raise ValueError("base P2 table species source identity is inconsistent")

    gate1 = _load_gate1(args.gate1_script.resolve())
    cases = _selected_cases(args.dataset_root.resolve(), [])
    _, loaded = _load_species(gate1, cases, args.pp_orb.resolve(), None)
    symbols = sorted(loaded)
    if set(symbols) != set(p2_manifest.get("species", {})):
        raise ValueError("P23 and base P2 species sets differ")
    for vna, ao in active_pairs:
        if vna not in loaded or ao not in loaded:
            raise KeyError(f"active pair {vna}|{ao} is absent from PP/ORB inputs")

    code_identity = {
        "builder_script_sha256": _sha256(Path(__file__).resolve()),
        "p23_interface_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "dptb/data/interfaces/p23_table.py"
        ),
        "gate1_script_sha256": _sha256(args.gate1_script.resolve()),
    }
    code_identity_sha256 = _json_sha256(code_identity)
    settings = {
        "radial_rank": int(args.radial_rank),
        "l_buffer": int(args.l_buffer),
        "distance_step_bohr": float(args.distance_step),
        "kmax_bohr_inv": float(args.kmax),
        "n_k": int(args.n_k),
        "n_mu": int(args.n_mu),
        "n_phi": int(args.n_phi),
        "interpolation": str(args.interpolation),
        "vna_relative_tolerance": float(args.vna_relative_tolerance),
    }
    partial_path = root / "manifest.partial.json"
    old_partial: dict[str, Any] | None = None
    if partial_path.is_file() and not args.overwrite:
        old_partial = json.loads(partial_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("schema", P23_VNA_TABLE_SCHEMA),
            ("build_settings", settings),
            ("code_identity_sha256", code_identity_sha256),
            ("base_p2_table_manifest_sha256", _sha256(p2_manifest_path)),
            ("factor_census_sha256", _sha256(args.factor_census.resolve())),
        ):
            if old_partial.get(key) != expected:
                raise ValueError(
                    f"partial P23 table {key} changed; use a new root or --overwrite"
                )

    _heartbeat(
        heartbeat,
        stage="projectors",
        completed=0,
        total=len(symbols),
        factor_tables_complete=len((old_partial or {}).get("factor_tables", {})),
        factor_tables_total=len(active_pairs),
    )
    projectors: dict[str, VNAProjectors] = {}
    species_manifest: dict[str, Any] = {}
    for index, symbol in enumerate(symbols, start=1):
        item = loaded[symbol]
        orbital: OrbitalLike = item["orbital"]
        r_vna, vna_ry = gate1.neutral_atom_potential_radial(item["upf"])
        cutoff = _vna_cutoff(r_vna, vna_ry, args.vna_relative_tolerance)
        projector = build_vna_projectors(
            orbital,
            r_vna,
            vna_ry,
            radial_rank=args.radial_rank,
            l_buffer=args.l_buffer,
            vna_cutoff_bohr=cutoff,
        )
        projectors[symbol] = projector
        relative = Path("species") / f"{symbol}.npz"
        path = root / relative
        _atomic_npz(
            path,
            epsilon_ao=projector.epsilon_ao.astype(np.float64),
            radial_epsilon=projector.radial_epsilon,
            radial_l=projector.radial_l,
            radial_n=projector.radial_n,
        )
        p2_row = p2_manifest["species"][symbol]
        if _sha256(item["orb_path"]) != str(p2_row["orbital_sha256"]):
            raise ValueError(f"{symbol}: ORB differs from base P2 table")
        if _sha256(item["upf_path"]) != str(p2_row["upf_sha256"]):
            raise ValueError(f"{symbol}: UPF differs from base P2 table")
        species_manifest[symbol] = {
            "orbital_shells": list(orbital.shells),
            "orbital_norb": int(orbital.norb),
            "orbital_cutoff_bohr": float(orbital.rcut),
            "orbital_sha256": str(p2_row["orbital_sha256"]),
            "upf_sha256": str(p2_row["upf_sha256"]),
            "physical_vna_cutoff_bohr": float(cutoff),
            "vna_cutoff_bohr": float(projector.orbital.rcut),
            "vna_projector_shells": list(projector.orbital.shells),
            "vna_projector_norb": int(projector.orbital.norb),
            "projector_metadata": projector.metadata,
            "array_path": relative.as_posix(),
            "array_sha256": _sha256(path),
        }
        _heartbeat(
            heartbeat,
            stage="projectors",
            completed=index,
            total=len(symbols),
            symbol=symbol,
            factor_tables_complete=len((old_partial or {}).get("factor_tables", {})),
            factor_tables_total=len(active_pairs),
        )

    k = np.linspace(0.0, float(args.kmax), int(args.n_k), dtype=np.float64)
    wk = np.ones(len(k), dtype=np.float64)
    wk[[0, -1]] = 0.5
    wk *= float(args.kmax) / (len(k) - 1)
    factor = (2.0 / np.pi) * wk * k * k
    left_transforms = {
        symbol: _expanded_transform(projectors[symbol].orbital, k, left=True)
        for symbol in symbols
    }
    right_transforms = {
        symbol: _expanded_transform(loaded[symbol]["orbital"], k, left=False)
        for symbol in symbols
    }
    coefficient_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[np.ndarray, ...]
    ] = {}
    for vna, ao in active_pairs:
        cache_key = (
            projectors[vna].orbital.shells,
            loaded[ao]["orbital"].shells,
        )
        if cache_key not in coefficient_cache:
            coefficient_cache[cache_key] = _angular_coefficients(
                projectors[vna].orbital,
                loaded[ao]["orbital"],
                n_mu=args.n_mu,
                n_phi=args.n_phi,
            )

    factor_tables = dict((old_partial or {}).get("factor_tables", {}))
    manifest = {
        "schema": P23_VNA_TABLE_SCHEMA,
        "complete": False,
        "length_unit": "bohr",
        "factor_energy_unit": "eV",
        "epsilon_unit": "1/eV",
        "harmonic_convention": "deeptb_abacus_real",
        "endpoint_policy": "exclude_i0_and_jR",
        "factor_support_semantics": P23_FACTOR_SUPPORT_SEMANTICS,
        "interpolation": str(args.interpolation),
        "build_settings": settings,
        "code_identity": code_identity,
        "code_identity_sha256": code_identity_sha256,
        "base_p2_table_manifest": str(p2_manifest_path),
        "base_p2_table_manifest_sha256": _sha256(p2_manifest_path),
        "base_p2_operator_contract": {
            "operator": PARENT_P2_OPERATOR,
            "soc_mode": PARENT_P2_SOC_MODE,
            "energy_unit": "Ry",
            "p23_endpoint_exclusion_semantics": (
                "exclude (i,0) and (j,R) VNA centres because base P2 already "
                "contains unique endpoint VNA_i/VNA_j exactly once"
            ),
        },
        "species_source_identity_sha256": p2_identity,
        "species_source_identity": p2_species_identity,
        "factor_census": str(args.factor_census.resolve()),
        "factor_census_sha256": _sha256(args.factor_census.resolve()),
        "factor_census_schema": factor_report["schema"],
        "factor_census_case_count": int(factor_report["summary"]["case_count"]),
        "factor_census_directed_terms": int(
            factor_report["summary"]["true_third_centres"]
        ),
        "factor_census_term_semantics": (
            "exact physical-VNA support census used to select active species pairs; "
            "factorized q-support contraction counts can be smaller"
        ),
        "source": {
            "dataset_root": str(args.dataset_root.resolve()),
            "pp_orb": str(args.pp_orb.resolve()),
            "gate1_script": str(args.gate1_script.resolve()),
        },
        "species": species_manifest,
        "active_factor_pairs": [f"{vna}|{ao}" for vna, ao in active_pairs],
        "factor_tables": factor_tables,
        "updated_unix": time.time(),
    }
    _atomic_json(partial_path, manifest)

    reusable: set[tuple[str, str]] = set()
    for vna, ao in active_pairs:
        key = _pair_key(vna, ao)
        row = factor_tables.get(key)
        if not isinstance(row, Mapping):
            continue
        path = root / str(row.get("path", ""))
        if path.is_file() and _sha256(path) == str(row.get("sha256", "")):
            reusable.add((vna, ao))
    pending = [pair for pair in active_pairs if pair not in reusable]

    def build_pair(pair: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        vna, ao = pair
        q = projectors[vna].orbital
        orbital = loaded[ao]["orbital"]
        support = float(species_manifest[vna]["vna_cutoff_bohr"]) + float(
            species_manifest[ao]["orbital_cutoff_bohr"]
        )
        distances = _distance_grid(support, args.distance_step)
        coefficients = coefficient_cache[(q.shells, orbital.shells)]
        values = _build_values_fast(
            left=q,
            right=orbital,
            left_transform=left_transforms[vna],
            right_transform=right_transforms[ao],
            coefficients=coefficients,
            k=k,
            factor=factor,
            distances=distances,
            support_bohr=support,
        )
        relative = Path("factors") / _pair_filename(vna, ao)
        path = root / relative
        _atomic_npz(
            path,
            distances=distances,
            values=values.astype(np.float32),
            left_shells=np.asarray(q.shells, dtype=np.int16),
            right_shells=np.asarray(orbital.shells, dtype=np.int16),
            support_bohr=np.asarray(support, dtype=np.float64),
        )
        return _pair_key(vna, ao), {
            "path": relative.as_posix(),
            "sha256": _sha256(path),
            "support_bohr": support,
            "distance_knots": len(distances),
            "shape": [len(distances), q.norb, orbital.norb],
            "value_dtype": "float32",
            "code_identity_sha256": code_identity_sha256,
        }

    complete = len(reusable)
    failed: list[dict[str, str]] = []
    _heartbeat(
        heartbeat,
        stage="factors",
        completed=complete,
        total=len(active_pairs),
        failed=0,
        workers=int(args.workers),
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(build_pair, pair): pair for pair in pending}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                key, row = future.result()
                factor_tables[key] = row
                complete += 1
                manifest["factor_tables"] = factor_tables
                manifest["updated_unix"] = time.time()
                _atomic_json(partial_path, manifest)
            except Exception as exc:
                failed.append(
                    {
                        "pair": f"{pair[0]}|{pair[1]}",
                        "type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            _heartbeat(
                heartbeat,
                stage="factors",
                completed=complete,
                total=len(active_pairs),
                failed=len(failed),
                pair=f"{pair[0]}|{pair[1]}",
                workers=int(args.workers),
            )
    if failed:
        manifest["failures"] = failed
        _atomic_json(partial_path, manifest)
        raise RuntimeError(f"failed to build {len(failed)} P23 factor tables")
    if set(factor_tables) != {_pair_key(*pair) for pair in active_pairs}:
        raise RuntimeError("completed P23 factor table set is not the active census set")

    manifest["complete"] = True
    manifest["factor_tables"] = {
        key: factor_tables[key] for key in sorted(factor_tables)
    }
    manifest["build_seconds"] = time.time() - started
    manifest["completed_unix"] = time.time()
    manifest["updated_unix"] = time.time()
    manifest.pop("failures", None)
    final_path = root / "manifest.json"
    _atomic_json(final_path, manifest)
    _heartbeat(
        heartbeat,
        stage="complete",
        completed=len(active_pairs),
        total=len(active_pairs),
        failed=0,
        manifest=str(final_path),
        manifest_sha256=_sha256(final_path),
        build_seconds=manifest["build_seconds"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pp-orb", type=Path, required=True)
    parser.add_argument("--gate1-script", type=Path, required=True)
    parser.add_argument("--p2-table-root", type=Path, required=True)
    parser.add_argument("--factor-census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radial-rank", type=int, default=4)
    parser.add_argument("--l-buffer", type=int, default=2)
    parser.add_argument("--distance-step", type=float, default=0.02)
    parser.add_argument("--kmax", type=float, default=80.0)
    parser.add_argument("--n-k", type=int, default=400)
    parser.add_argument("--n-mu", type=int, default=18)
    parser.add_argument("--n-phi", type=int, default=36)
    parser.add_argument("--interpolation", choices=("linear", "cubic"), default="cubic")
    parser.add_argument("--vna-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "species": len(result["species"]),
                "factor_tables": len(result["factor_tables"]),
                "build_seconds": result["build_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
