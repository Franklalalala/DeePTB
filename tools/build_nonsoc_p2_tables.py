#!/usr/bin/env python3
"""Build immutable non-SOC P2 radial-table shards from ABACUS ORB/UPF files.

This is an *offline* program.  It may spend substantial CPU time producing a
high-quality table, but the resulting table is consumed without quadrature by
``dptb.data.interfaces.p2_table``.  Optional S/T/VNA/Vnl component arrays are
table-level diagnostic/oracle payloads in this builder.  They are not
materialized as the current ``node_p2``/``edge_p2`` training side channels.
The implementation uses spherical-Bessel two-centre transforms for

* kinetic blocks;
* ``VNA_i`` and ``VNA_j`` by multiplying the corresponding radial orbital by
  its central neutral-atom potential before the same transform; and
* AO--projector overlaps used by the factorised all-centre nonlocal term.

The table is sharded by ordered element pair so a broad periodic-table build
can be resumed and inference only loads the species present in one structure.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.special import eval_legendre, spherical_jn

from dptb.data.interfaces.p2_table import (
    P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
    P2_TABLE_SCHEMA,
    P2TableStore,
    abacus_m_order,
    canonical_species_source_identity,
    normalized_p2_base_component_contract,
    p2_base_component_contract,
    real_sph_abacus,
    species_source_identity_sha256,
    validate_p2_component_numerical_contract,
)


P2_TABLE_CODE_IDENTITY_SCHEMA = "deeptb.p2_table_code_identity/v1"
P2_COMPONENT_TABLE_SEMANTICS_SCHEMA = "deeptb.p2_component_table_semantics/v1"


@dataclass(frozen=True)
class Channel:
    l: int
    radial: np.ndarray
    cutoff_bohr: float
    source_index: int


@dataclass(frozen=True)
class Descriptor:
    channel_index: int
    l: int
    m: int


@dataclass
class OrbitalLike:
    symbol: str
    r: np.ndarray
    channels: tuple[Channel, ...]
    source: Path

    @property
    def rcut(self) -> float:
        return max((float(c.cutoff_bohr) for c in self.channels), default=0.0)

    @property
    def shells(self) -> tuple[int, ...]:
        return tuple(int(c.l) for c in self.channels)

    @property
    def norb(self) -> int:
        return sum(2 * int(c.l) + 1 for c in self.channels)

    def descriptors(self) -> list[Descriptor]:
        output: list[Descriptor] = []
        for channel_index, channel in enumerate(self.channels):
            for m in abacus_m_order(channel.l):
                output.append(Descriptor(channel_index, int(channel.l), int(m)))
        return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray, *, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _load_gate1(path: Path):
    spec = importlib.util.spec_from_file_location("deeptb_gate1_p2_oracle", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Gate-1 script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _angular_grid(n_mu: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    mu, w_mu = np.polynomial.legendre.leggauss(int(n_mu))
    phi = (np.arange(int(n_phi), dtype=np.float64) + 0.5) * (
        2.0 * np.pi / int(n_phi)
    )
    directions = []
    weights = []
    w_phi = 2.0 * np.pi / int(n_phi)
    for value, weight in zip(mu, w_mu):
        sine = np.sqrt(max(0.0, 1.0 - value * value))
        directions.append(
            np.stack(
                [
                    sine * np.cos(phi),
                    sine * np.sin(phi),
                    np.full_like(phi, value),
                ],
                axis=1,
            )
        )
        weights.append(np.full(int(n_phi), weight * w_phi, dtype=np.float64))
    return np.vstack(directions), np.concatenate(weights)


def _as_orbital_like(orb: Any, source: Path) -> OrbitalLike:
    channels = tuple(
        Channel(
            l=int(channel.l),
            radial=np.asarray(channel.radial, dtype=np.float64),
            cutoff_bohr=float(orb.rcut),
            source_index=int(getattr(channel, "source_index", index)),
        )
        for index, channel in enumerate(orb.channels)
    )
    output = OrbitalLike(
        symbol=str(orb.element),
        r=np.asarray(orb.r, dtype=np.float64),
        channels=channels,
        source=source,
    )
    if output.r.ndim != 1 or output.r.size < 3:
        raise ValueError(f"bad orbital radial grid for {output.symbol}")
    return output


def _vna_weighted(orbital: OrbitalLike, r_vna: np.ndarray, vna_ry: np.ndarray) -> OrbitalLike:
    potential = np.interp(
        orbital.r,
        np.asarray(r_vna, dtype=np.float64),
        np.asarray(vna_ry, dtype=np.float64),
        left=float(np.asarray(vna_ry)[0]),
        right=0.0,
    )
    channels = tuple(
        Channel(
            l=channel.l,
            radial=np.asarray(channel.radial * potential, dtype=np.float64),
            cutoff_bohr=channel.cutoff_bohr,
            source_index=channel.source_index,
        )
        for channel in orbital.channels
    )
    return OrbitalLike(
        symbol=orbital.symbol,
        r=orbital.r,
        channels=channels,
        source=orbital.source,
    )


def _projector_orbital(symbol: str, upf: Any, source: Path) -> OrbitalLike:
    from h0rebuild.radial import projector_beta_values

    r = np.asarray(upf.r, dtype=np.float64)
    channels = []
    for index, projector in enumerate(upf.projectors):
        radial = projector_beta_values(
            r,
            np.asarray(projector.radial_u, dtype=np.float64),
            float(projector.cutoff_radius),
            r,
        )
        radial = np.where(r <= float(projector.cutoff_radius) + 1.0e-12, radial, 0.0)
        channels.append(
            Channel(
                l=int(projector.l),
                radial=np.asarray(radial, dtype=np.float64),
                cutoff_bohr=float(projector.cutoff_radius),
                source_index=index,
            )
        )
    return OrbitalLike(symbol=symbol, r=r, channels=tuple(channels), source=source)


def _spin_trace_weight(l: int, j: float | None, has_so: bool) -> float:
    if not has_so or j is None or l == 0:
        return 1.0
    if abs(float(j) - (float(l) + 0.5)) < 1.0e-8:
        return (float(l) + 1.0) / (2.0 * float(l) + 1.0)
    if l > 0 and abs(float(j) - (float(l) - 0.5)) < 1.0e-8:
        return float(l) / (2.0 * float(l) + 1.0)
    raise ValueError(f"incompatible fully-relativistic projector l={l}, j={j}")


def _same_projector_group(a: Any, b: Any, has_so: bool) -> bool:
    if int(a.l) != int(b.l):
        return False
    if not has_so:
        return True
    if a.j is None or b.j is None:
        return a.j is None and b.j is None
    return abs(float(a.j) - float(b.j)) < 1.0e-8


def _effective_d(upf: Any) -> np.ndarray:
    projectors = list(upf.projectors)
    offsets = np.concatenate(
        ([0], np.cumsum([2 * int(p.l) + 1 for p in projectors]))
    ).astype(int)
    output = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.float64)
    for a, projector_a in enumerate(projectors):
        for b, projector_b in enumerate(projectors):
            if not _same_projector_group(projector_a, projector_b, bool(upf.has_so)):
                continue
            weight_a = _spin_trace_weight(
                int(projector_a.l), projector_a.j, bool(upf.has_so)
            )
            weight_b = _spin_trace_weight(
                int(projector_b.l), projector_b.j, bool(upf.has_so)
            )
            value = float(upf.dij_ry[a, b]) * math.sqrt(weight_a * weight_b)
            width = 2 * int(projector_a.l) + 1
            rows = slice(int(offsets[a]), int(offsets[a + 1]))
            cols = slice(int(offsets[b]), int(offsets[b + 1]))
            output[rows, cols] = np.eye(width) * value
    return output


@dataclass
class SBTContext:
    left: OrbitalLike
    right: OrbitalLike
    k: np.ndarray
    factor: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    radial: np.ndarray


def _channel_transforms(orbital: OrbitalLike, k: np.ndarray) -> np.ndarray:
    output = []
    r = np.asarray(orbital.r, dtype=np.float64)
    for channel in orbital.channels:
        jl = spherical_jn(int(channel.l), np.outer(k, r))
        output.append(
            _integrate_trapezoid(
                (r * r * channel.radial)[None, :] * jl,
                r,
                axis=1,
            )
        )
    return np.asarray(output, dtype=np.float64)


def _sbt_context(
    left: OrbitalLike,
    right: OrbitalLike,
    *,
    kmax: float,
    n_k: int,
    n_mu: int,
    n_phi: int,
) -> SBTContext:
    left_items = left.descriptors()
    right_items = right.descriptors()
    k = np.linspace(0.0, float(kmax), int(n_k), dtype=np.float64)
    wk = np.ones(int(n_k), dtype=np.float64)
    wk[[0, -1]] = 0.5
    wk *= float(kmax) / (int(n_k) - 1)
    left_transform = _channel_transforms(left, k)
    right_transform = _channel_transforms(right, k)
    radial_left = np.empty((len(k), len(left_items)), dtype=np.complex128)
    radial_right = np.empty((len(k), len(right_items)), dtype=np.complex128)
    for column, descriptor in enumerate(left_items):
        radial_left[:, column] = (
            (1j) ** int(descriptor.l)
        ) * left_transform[int(descriptor.channel_index)]
    for column, descriptor in enumerate(right_items):
        radial_right[:, column] = (
            (-1j) ** int(descriptor.l)
        ) * right_transform[int(descriptor.channel_index)]

    directions, angular_weights = _angular_grid(n_mu, n_phi)
    y_left = np.column_stack(
        [real_sph_abacus(item.l, item.m, directions) for item in left_items]
    )
    y_right = np.column_stack(
        [real_sph_abacus(item.l, item.m, directions) for item in right_items]
    )
    lmax = max((item.l for item in left_items), default=0) + max(
        (item.l for item in right_items), default=0
    )
    coefficients = []
    mu = directions[:, 2]
    for l in range(int(lmax) + 1):
        coefficients.append(
            np.einsum(
                "pa,p,pb->ab",
                y_left,
                eval_legendre(l, mu) * angular_weights,
                y_right,
                optimize=True,
            )
        )
    return SBTContext(
        left=left,
        right=right,
        k=k,
        factor=(2.0 / np.pi) * wk * k * k,
        coefficients=tuple(coefficients),
        radial=radial_left[:, :, None] * radial_right[:, None, :],
    )


def _build_values(
    context: SBTContext,
    distances: np.ndarray,
    *,
    kinetic: bool = False,
    distance_chunk: int = 128,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    left_items = context.left.descriptors()
    right_items = context.right.descriptors()
    output = np.zeros(
        (len(distances), len(left_items), len(right_items)), dtype=np.float64
    )
    factor = context.factor * (context.k * context.k if kinetic else 1.0)
    for start in range(0, len(distances), max(1, int(distance_chunk))):
        stop = min(start + max(1, int(distance_chunk)), len(distances))
        kd = np.outer(context.k, distances[start:stop])
        target = output[start:stop]
        for l, coefficient in enumerate(context.coefficients):
            phase = (2 * l + 1) * ((-1j) ** l)
            jl = spherical_jn(l, kd)
            integral = np.einsum(
                "k,kab,kd->dab",
                factor,
                context.radial,
                jl,
                optimize=True,
            )
            target += (phase * integral * coefficient[None, :, :]).real

    # Exact axial selection rule and exact finite support.  The latter removes
    # harmless finite-k ringing that would otherwise leak through D_eff.
    for row, left_item in enumerate(left_items):
        left_channel = context.left.channels[left_item.channel_index]
        for col, right_item in enumerate(right_items):
            right_channel = context.right.channels[right_item.channel_index]
            if int(left_item.m) != int(right_item.m):
                output[:, row, col] = 0.0
                continue
            support = float(left_channel.cutoff_bohr + right_channel.cutoff_bohr)
            output[distances >= support - 1.0e-12, row, col] = 0.0
    return output


def _distance_grid(support: float, step: float) -> np.ndarray:
    if support <= 0.0 or step <= 0.0:
        raise ValueError("support and distance step must be positive")
    end = math.ceil(float(support) / float(step)) * float(step)
    if end < support + 0.25 * step:
        end += float(step)
    count = int(round(end / float(step))) + 1
    return np.linspace(0.0, end, count, dtype=np.float64)


def _pair_filename(left: str, right: str) -> str:
    return f"{left}__{right}.npz"


def _pair_key(left: str, right: str) -> str:
    return f"{left}|{right}"


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "distance_step_bohr": float(args.distance_step),
        "kmax_bohr_inv": float(args.kmax),
        "n_k": int(args.n_k),
        "n_mu": int(args.n_mu),
        "n_phi": int(args.n_phi),
        "interpolation": str(args.interpolation),
        "dtype": "float32",
        "operator": "T + unique(VNA_i,VNA_j) + factorized all-K Vnl",
        "soc_mode": "non-SOC scalar spin trace",
        "base_components": str(getattr(args, "base_components", "none")),
    }


def _require_sha256_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} is missing a valid SHA256 checksum")
    return text


def _code_identity_sha256(identity: dict[str, Any]) -> str:
    if identity.get("schema") != P2_TABLE_CODE_IDENTITY_SCHEMA:
        raise ValueError("P2 table code identity has the wrong schema")
    canonical = {
        "schema": P2_TABLE_CODE_IDENTITY_SCHEMA,
        "gate1_script_sha256": _require_sha256_text(
            identity.get("gate1_script_sha256"),
            label="gate1_script_sha256",
        ),
        "builder_script_sha256": _require_sha256_text(
            identity.get("builder_script_sha256"),
            label="builder_script_sha256",
        ),
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _current_code_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": P2_TABLE_CODE_IDENTITY_SCHEMA,
        "gate1_script_sha256": _sha256(args.gate1_script.resolve()),
        "builder_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _component_table_semantics(mode: str) -> dict[str, Any]:
    return {
        "schema": P2_COMPONENT_TABLE_SEMANTICS_SCHEMA,
        "base_components": str(mode),
        "scope": "component_table_diagnostic_oracle_only",
        "materialized_lmdb_side_channels": ["node_p2", "edge_p2"],
        "not_materialized_lmdb_side_channels": [
            "node_overlap",
            "edge_overlap",
            "node_kinetic",
            "edge_kinetic",
            "node_vna",
            "edge_vna",
            "node_vnl",
            "edge_vnl",
        ],
    }


def _normalized_build_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("P2 build_settings must be a mapping")
    normalized = dict(settings)
    # Tables created before optional components existed are precisely mode none.
    normalized.setdefault("base_components", "none")
    return normalized


def _selected_cases(dataset_root: Path, case_names: Sequence[str]) -> list[Path]:
    cases_root = dataset_root / "cases"
    cases = sorted(path for path in cases_root.iterdir() if path.is_dir())
    if case_names:
        wanted = set(case_names)
        cases = [path for path in cases if path.name in wanted]
        missing = wanted - {path.name for path in cases}
        if missing:
            raise FileNotFoundError(f"requested cases missing: {sorted(missing)}")
    if not cases:
        raise ValueError("no cases selected for P2 table build")
    return cases


def _load_species(
    gate1: Any,
    cases: Sequence[Path],
    pp_orb: Path,
    requested_symbols: set[str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    specs: dict[str, Any] = {}
    parsed_cases = []
    for case in cases:
        parsed = gate1.parse_stru(case / "STRU")
        parsed_cases.append(parsed)
        for symbol, spec in parsed.specs.items():
            if requested_symbols is not None and symbol not in requested_symbols:
                continue
            old = specs.get(symbol)
            signature = (str(spec.pseudo), str(spec.orbital))
            if old is not None and (str(old.pseudo), str(old.orbital)) != signature:
                raise ValueError(f"inconsistent ORB/UPF mapping for {symbol}")
            specs[symbol] = spec
    if requested_symbols is not None:
        missing = requested_symbols - set(specs)
        if missing:
            raise ValueError(f"requested species absent from selected cases: {sorted(missing)}")

    loaded: dict[str, dict[str, Any]] = {}
    for symbol, spec in sorted(specs.items()):
        orb_path = pp_orb / spec.orbital
        upf_path = pp_orb / spec.pseudo
        if not orb_path.is_file() or not upf_path.is_file():
            raise FileNotFoundError(f"missing ORB/UPF for {symbol}: {orb_path}, {upf_path}")
        orb = gate1.read_abacus_orb(orb_path)
        upf, repair = gate1.read_upf_compat(upf_path)
        orbital = _as_orbital_like(orb, orb_path)
        projector = _projector_orbital(symbol, upf, upf_path)
        r_vna, vna = gate1.neutral_atom_potential_radial(upf)
        loaded[symbol] = {
            "orbital": orbital,
            "weighted": _vna_weighted(orbital, r_vna, vna),
            "projector": projector,
            "upf": upf,
            "orb_path": orb_path,
            "upf_path": upf_path,
            "repair": repair,
        }
    return specs, loaded


def _assert_resume_species_identity(
    old_manifest: dict[str, Any],
    current_identity: dict[str, Any],
    current_identity_sha256: str,
) -> None:
    """Refuse to inherit shards built from any other ORB/UPF contract."""

    try:
        old_identity = canonical_species_source_identity(old_manifest["species"])
        old_identity_sha256 = species_source_identity_sha256(old_identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "existing P2 manifest lacks a complete ORB/UPF species identity; "
            "use a new root or --overwrite"
        ) from exc

    declared_identity = old_manifest.get("species_source_identity")
    declared_sha256 = old_manifest.get("species_source_identity_sha256")
    if (declared_identity is None) != (declared_sha256 is None):
        raise ValueError(
            "existing P2 manifest has an incomplete species identity declaration"
        )
    if declared_identity is not None:
        if declared_identity != old_identity:
            raise ValueError(
                "existing P2 species identity disagrees with its species metadata"
            )
        if str(declared_sha256).lower() != old_identity_sha256:
            raise ValueError("existing P2 species identity SHA256 is invalid")

    if old_identity_sha256 != current_identity_sha256 or old_identity != current_identity:
        old_symbols = sorted(old_identity.get("species", {}))
        current_symbols = sorted(current_identity.get("species", {}))
        raise ValueError(
            "ORB/UPF/species source identity changed; refusing stale shard reuse "
            f"(old_sha256={old_identity_sha256}, "
            f"current_sha256={current_identity_sha256}, "
            f"old_species={old_symbols}, current_species={current_symbols}). "
            "Use a new root or --overwrite."
        )


def _assert_resume_code_identity(
    old_manifest: dict[str, Any],
    current_code_identity: dict[str, Any],
    current_code_identity_sha256: str,
) -> None:
    """Refuse to inherit shards built by another Gate1 or builder script."""

    source = old_manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError(
            "existing P2 manifest lacks Gate1/builder code identity; "
            "use a new root or --overwrite"
        )
    try:
        old_code_identity = {
            "schema": P2_TABLE_CODE_IDENTITY_SCHEMA,
            "gate1_script_sha256": _require_sha256_text(
                source.get("gate1_script_sha256"),
                label="existing gate1_script_sha256",
            ),
            "builder_script_sha256": _require_sha256_text(
                source.get("builder_script_sha256"),
                label="existing builder_script_sha256",
            ),
        }
        declared_code_identity_sha256 = _require_sha256_text(
            source.get("code_identity_sha256"),
            label="existing code_identity_sha256",
        )
    except ValueError as exc:
        raise ValueError(
            "existing P2 manifest lacks Gate1/builder code identity; "
            "use a new root or --overwrite"
        ) from exc
    old_code_identity_sha256 = _code_identity_sha256(old_code_identity)
    if declared_code_identity_sha256 != old_code_identity_sha256:
        raise ValueError(
            "existing P2 manifest code_identity_sha256 is inconsistent with "
            "its recorded Gate1/builder hashes; use a new root or --overwrite"
        )
    if (
        old_code_identity != current_code_identity
        or old_code_identity_sha256 != current_code_identity_sha256
    ):
        raise ValueError(
            "Gate1/builder code identity changed; refusing stale P2 shard reuse "
            f"(old_sha256={old_code_identity_sha256}, "
            f"current_sha256={current_code_identity_sha256}). "
            "Use a new root or --overwrite."
        )


def _verify_reusable_shard(
    *,
    root: Path,
    path: Path,
    relative: Path,
    metadata: dict[str, Any],
    label: str,
    expected_code_identity_sha256: str,
) -> None:
    expected_relative = relative.as_posix()
    if str(metadata.get("path")) != expected_relative:
        raise ValueError(
            f"{label} manifest path {metadata.get('path')!r} != {expected_relative!r}"
        )
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"{label} shard escapes table root: {resolved}")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha256 = str(metadata.get("sha256", "")).strip().lower()
    if len(expected_sha256) != 64:
        raise ValueError(f"{label} is missing a valid shard SHA256")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} checksum mismatch while resuming: "
            f"manifest={expected_sha256}, actual={actual_sha256}"
        )
    shard_code_identity_sha256 = _require_sha256_text(
        metadata.get("code_identity_sha256"),
        label=f"{label} code_identity_sha256",
    )
    if shard_code_identity_sha256 != expected_code_identity_sha256:
        raise ValueError(
            f"{label} code identity mismatch: "
            f"manifest={shard_code_identity_sha256}, "
            f"current={expected_code_identity_sha256}. "
            "Use a new root or --overwrite."
        )


def _pending_component_validation_store(
    root: Path,
    manifest: dict[str, Any],
) -> P2TableStore:
    """Create the narrow store view needed to qualify an incomplete table.

    ``P2TableStore`` intentionally rejects incomplete public manifests.  The
    builder must nevertheless run the component gate before publishing
    ``complete=true``.  This private view initializes only the reader state
    used by ``validate_p2_component_numerical_contract`` and never relaxes the
    public store contract.
    """

    store = P2TableStore.__new__(P2TableStore)
    store.root = root.resolve()
    store.manifest = manifest
    store.species = manifest["species"]
    store.base_tables = manifest["base_tables"]
    store.projector_tables = manifest["projector_tables"]
    contract = normalized_p2_base_component_contract(manifest)
    store.base_component_contract = contract
    store.base_component_mode = str(contract["mode"])
    store.base_component_arrays = {
        str(name): str(spec["array"])
        for name, spec in contract["base_shard"].items()
        if bool(spec["available"])
    }
    store.onsite_component_arrays = {
        str(name): str(spec["array"])
        for name, spec in contract["species_shard"].items()
        if bool(spec["available"])
    }
    store.max_cached_tables = 64
    store.verify_checksums = True
    store._cache = OrderedDict()
    store._species_arrays = {}
    store._verified_paths = set()
    store._source_hash_cache = {}
    source = manifest.get("source")
    store.table_code_identity_sha256 = (
        source.get("code_identity_sha256")
        if isinstance(source, dict)
        else None
    )
    return store


def _qualify_and_publish_manifest(
    *,
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    component_reciprocity_tolerance: float,
    component_reconstruction_tolerance: float,
    component_gate_distance_samples: int,
) -> dict[str, Any]:
    """Qualify all shards before atomically publishing a completed manifest."""

    manifest["complete"] = False
    manifest["component_numerical_gate"] = {
        "schema": P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
        "status": "pending",
    }
    manifest["updated_unix"] = time.time()
    _atomic_json(manifest_path, manifest)
    try:
        component_gate = validate_p2_component_numerical_contract(
            _pending_component_validation_store(root, manifest),
            reciprocity_tolerance=float(component_reciprocity_tolerance),
            reconstruction_tolerance=float(component_reconstruction_tolerance),
            distance_samples=int(component_gate_distance_samples),
        )
    except Exception as exc:
        manifest["complete"] = False
        manifest["component_numerical_gate"] = {
            "schema": P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
            "status": "fail",
            "error": str(exc),
        }
        manifest["updated_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        raise
    if (
        component_gate.get("schema") != P2_COMPONENT_NUMERICAL_GATE_SCHEMA
        or component_gate.get("status") != "pass"
    ):
        manifest["complete"] = False
        manifest["component_numerical_gate"] = component_gate
        manifest["updated_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        raise ValueError(
            "P2 component numerical gate failed: "
            + "; ".join(component_gate.get("failures", []))
        )

    # One atomic replacement publishes both qualification evidence and the
    # completed state.  Before this write, every observable manifest remains
    # incomplete even if validation is interrupted.
    manifest["component_numerical_gate"] = component_gate
    manifest["complete"] = True
    manifest["updated_unix"] = time.time()
    _atomic_json(manifest_path, manifest)
    return component_gate


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gate1 = _load_gate1(args.gate1_script.resolve())
    cases = _selected_cases(args.dataset_root.resolve(), args.case)
    requested = None
    if args.species:
        requested = {item.strip() for item in args.species.split(",") if item.strip()}
    _, species_data = _load_species(
        gate1, cases, args.pp_orb.resolve(), requested
    )
    symbols = sorted(species_data)
    settings = _settings(args)
    code_identity = _current_code_identity(args)
    code_identity_hash = _code_identity_sha256(code_identity)
    component_contract = p2_base_component_contract(settings["base_components"])
    component_semantics = _component_table_semantics(settings["base_components"])
    base_component_arrays = {
        name: spec["array"]
        for name, spec in component_contract["base_shard"].items()
        if spec["available"]
    }
    onsite_component_arrays = {
        name: spec["array"]
        for name, spec in component_contract["species_shard"].items()
        if spec["available"]
    }
    manifest_path = root / "manifest.json"
    old_manifest = None
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("schema") != P2_TABLE_SCHEMA:
            raise ValueError("refusing to reuse a table root with another schema")
        if not args.overwrite:
            old_settings = _normalized_build_settings(
                old_manifest.get("build_settings")
            )
            if old_settings != settings:
                raise ValueError(
                    "table build settings changed; use a new root or --overwrite"
                )
            old_component_contract = normalized_p2_base_component_contract(
                old_manifest
            )
            if old_component_contract != component_contract:
                raise ValueError(
                    "table base component contract changed; use a new root or "
                    "--overwrite"
                )

    species_manifest: dict[str, Any] = {}
    species_arrays: dict[str, dict[str, np.ndarray]] = {}
    for symbol in symbols:
        item = species_data[symbol]
        orbital: OrbitalLike = item["orbital"]
        projector: OrbitalLike = item["projector"]
        upf = item["upf"]
        species_path = Path("species") / f"{symbol}.npz"
        species_manifest[symbol] = {
            "orbital_shells": list(orbital.shells),
            "orbital_norb": int(orbital.norb),
            "orbital_cutoff_bohr": float(orbital.rcut),
            "projector_shells": list(projector.shells),
            "projector_norb": int(projector.norb),
            "projector_cutoffs_bohr": [
                float(channel.cutoff_bohr) for channel in projector.channels
            ],
            "projector_max_cutoff_bohr": float(projector.rcut),
            "orbital_file": str(item["orb_path"]),
            "orbital_sha256": _sha256(item["orb_path"]),
            "upf_file": str(item["upf_path"]),
            "upf_sha256": _sha256(item["upf_path"]),
            "upf_has_so": bool(upf.has_so),
            "upf_metadata_repair": item["repair"],
            "array_path": species_path.as_posix(),
            "onsite_component_arrays": dict(onsite_component_arrays),
            "code_identity_sha256": code_identity_hash,
        }
        species_arrays[symbol] = {
            "d_eff": _effective_d(upf).astype(np.float64),
            "onsite_base": np.zeros((orbital.norb, orbital.norb), dtype=np.float64),
        }
        if "overlap" in onsite_component_arrays:
            species_arrays[symbol]["onsite_overlap"] = np.zeros(
                (orbital.norb, orbital.norb), dtype=np.float64
            )
        if "kinetic" in onsite_component_arrays:
            species_arrays[symbol]["onsite_kinetic"] = np.zeros(
                (orbital.norb, orbital.norb), dtype=np.float64
            )
            species_arrays[symbol]["onsite_vna_endpoint"] = np.zeros(
                (orbital.norb, orbital.norb), dtype=np.float64
            )

    species_source_identity = canonical_species_source_identity(species_manifest)
    species_source_identity_hash = species_source_identity_sha256(
        species_source_identity
    )
    if old_manifest is not None and not args.overwrite:
        _assert_resume_species_identity(
            old_manifest,
            species_source_identity,
            species_source_identity_hash,
        )
        _assert_resume_code_identity(
            old_manifest,
            code_identity,
            code_identity_hash,
        )

    manifest: dict[str, Any] = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "build_settings": settings,
        "base_component_contract": component_contract,
        "component_table_semantics": component_semantics,
        "source": {
            "dataset_root": str(args.dataset_root.resolve()),
            "pp_orb": str(args.pp_orb.resolve()),
            "gate1_script": str(args.gate1_script.resolve()),
            "gate1_script_sha256": code_identity["gate1_script_sha256"],
            "builder_script": str(Path(__file__).resolve()),
            "builder_script_sha256": code_identity["builder_script_sha256"],
            "code_identity_schema": P2_TABLE_CODE_IDENTITY_SCHEMA,
            "code_identity_sha256": code_identity_hash,
            "selected_cases": [case.name for case in cases],
        },
        "species": species_manifest,
        "species_source_identity": species_source_identity,
        "species_source_identity_sha256": species_source_identity_hash,
        "base_tables": {},
        "projector_tables": {},
        "complete": False,
        "updated_unix": time.time(),
    }
    if old_manifest is not None and not args.overwrite:
        for key, raw_metadata in old_manifest.get("base_tables", {}).items():
            metadata = dict(raw_metadata)
            declared_arrays = metadata.get("component_arrays")
            if declared_arrays is None and settings["base_components"] == "none":
                declared_arrays = {"p2_base": "values"}
            if declared_arrays != base_component_arrays:
                raise ValueError(
                    f"base:{key} component arrays do not match resume contract"
                )
            metadata["component_arrays"] = dict(base_component_arrays)
            manifest["base_tables"][key] = metadata
        manifest["projector_tables"].update(old_manifest.get("projector_tables", {}))
    _atomic_json(manifest_path, manifest)

    sbt_args = {
        "kmax": float(args.kmax),
        "n_k": int(args.n_k),
        "n_mu": int(args.n_mu),
        "n_phi": int(args.n_phi),
    }
    built = 0
    skipped = 0
    base_pairs = [(left, right) for left in symbols for right in symbols]
    for pair_index, (left_symbol, right_symbol) in enumerate(base_pairs, start=1):
        key = _pair_key(left_symbol, right_symbol)
        relative = Path("base") / _pair_filename(left_symbol, right_symbol)
        path = root / relative
        if path.is_file() and key in manifest["base_tables"] and not args.overwrite:
            _verify_reusable_shard(
                root=root,
                path=path,
                relative=relative,
                metadata=manifest["base_tables"][key],
                label=f"base:{key}",
                expected_code_identity_sha256=code_identity_hash,
            )
            skipped += 1
            with np.load(path, allow_pickle=False) as payload:
                missing_components = sorted(
                    set(base_component_arrays.values()) - set(payload.files)
                )
                if missing_components:
                    raise ValueError(
                        f"base:{key} is missing declared component arrays "
                        f"{missing_components}"
                    )
                if left_symbol == right_symbol:
                    missing_onsite = sorted(
                        set(onsite_component_arrays.values()) - set(payload.files)
                    )
                    if missing_onsite:
                        raise ValueError(
                            f"base:{key} is missing declared onsite arrays "
                            f"{missing_onsite}"
                        )
                    for array_name in onsite_component_arrays.values():
                        species_arrays[left_symbol][array_name] = np.asarray(
                            payload[array_name], dtype=np.float64
                        )
            continue
        left = species_data[left_symbol]
        right = species_data[right_symbol]
        orbital_left: OrbitalLike = left["orbital"]
        orbital_right: OrbitalLike = right["orbital"]
        support = orbital_left.rcut + orbital_right.rcut
        distances = _distance_grid(support, float(args.distance_step))
        orbital_context = _sbt_context(
            orbital_left, orbital_right, **sbt_args
        )
        kinetic = _build_values(
            orbital_context,
            distances,
            kinetic=True,
            distance_chunk=args.distance_chunk,
        )
        overlap = None
        if settings["base_components"] in {"overlap", "all"}:
            overlap = _build_values(
                orbital_context,
                distances,
                distance_chunk=args.distance_chunk,
            )
        v_left = _build_values(
            _sbt_context(left["weighted"], orbital_right, **sbt_args),
            distances,
            distance_chunk=args.distance_chunk,
        )
        v_right = _build_values(
            _sbt_context(orbital_left, right["weighted"], **sbt_args),
            distances,
            distance_chunk=args.distance_chunk,
        )
        values = kinetic + v_left + v_right
        values[distances >= support - 1.0e-12] = 0.0
        onsite = np.asarray(kinetic[0] + v_left[0], dtype=np.float64)
        arrays = {
            "distances": distances,
            "values": values.astype(np.float32),
            "left_shells": np.asarray(orbital_left.shells, dtype=np.int16),
            "right_shells": np.asarray(orbital_right.shells, dtype=np.int16),
            "support_bohr": np.asarray(support, dtype=np.float64),
        }
        if overlap is not None:
            arrays["overlap_values"] = overlap.astype(np.float32)
        if settings["base_components"] == "all":
            arrays["kinetic_values"] = kinetic.astype(np.float32)
            arrays["vna_left_values"] = v_left.astype(np.float32)
            arrays["vna_right_values"] = v_right.astype(np.float32)
        if left_symbol == right_symbol:
            arrays["onsite_base"] = onsite.astype(np.float64)
            species_arrays[left_symbol]["onsite_base"] = onsite
            if overlap is not None:
                onsite_overlap = np.asarray(overlap[0], dtype=np.float64)
                arrays["onsite_overlap"] = onsite_overlap
                species_arrays[left_symbol]["onsite_overlap"] = onsite_overlap
            if settings["base_components"] == "all":
                onsite_kinetic = np.asarray(kinetic[0], dtype=np.float64)
                # Coincident endpoints represent one unique VNA centre.  Do
                # not add v_right[0] a second time.
                onsite_vna_endpoint = np.asarray(v_left[0], dtype=np.float64)
                arrays["onsite_kinetic"] = onsite_kinetic
                arrays["onsite_vna_endpoint"] = onsite_vna_endpoint
                species_arrays[left_symbol]["onsite_kinetic"] = onsite_kinetic
                species_arrays[left_symbol][
                    "onsite_vna_endpoint"
                ] = onsite_vna_endpoint
        _atomic_npz(path, **arrays)
        manifest["base_tables"][key] = {
            "path": relative.as_posix(),
            "support_bohr": float(support),
            "distance_count": int(len(distances)),
            "interpolation": str(args.interpolation),
            "component_arrays": dict(base_component_arrays),
            "code_identity_sha256": code_identity_hash,
            "sha256": _sha256(path),
        }
        manifest["updated_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        _atomic_json(
            root / "heartbeat.json",
            {
                "schema": P2_TABLE_SCHEMA,
                "stage": "base_tables",
                "pair": key,
                "complete": pair_index,
                "total": len(base_pairs),
                "updated_unix": time.time(),
            },
        )
        built += 1

    projector_pairs = [
        (projector_symbol, orbital_symbol)
        for projector_symbol in symbols
        if species_data[projector_symbol]["projector"].norb > 0
        for orbital_symbol in symbols
    ]
    for pair_index, (projector_symbol, orbital_symbol) in enumerate(
        projector_pairs, start=1
    ):
        key = _pair_key(projector_symbol, orbital_symbol)
        relative = Path("projector") / _pair_filename(projector_symbol, orbital_symbol)
        path = root / relative
        if path.is_file() and key in manifest["projector_tables"] and not args.overwrite:
            _verify_reusable_shard(
                root=root,
                path=path,
                relative=relative,
                metadata=manifest["projector_tables"][key],
                label=f"projector:{key}",
                expected_code_identity_sha256=code_identity_hash,
            )
            skipped += 1
            continue
        projector: OrbitalLike = species_data[projector_symbol]["projector"]
        orbital: OrbitalLike = species_data[orbital_symbol]["orbital"]
        support = projector.rcut + orbital.rcut
        distances = _distance_grid(support, float(args.distance_step))
        values = _build_values(
            _sbt_context(projector, orbital, **sbt_args),
            distances,
            distance_chunk=args.distance_chunk,
        )
        values[distances >= support - 1.0e-12] = 0.0
        _atomic_npz(
            path,
            distances=distances,
            values=values.astype(np.float32),
            left_shells=np.asarray(projector.shells, dtype=np.int16),
            right_shells=np.asarray(orbital.shells, dtype=np.int16),
            support_bohr=np.asarray(support, dtype=np.float64),
        )
        manifest["projector_tables"][key] = {
            "path": relative.as_posix(),
            "support_bohr": float(support),
            "distance_count": int(len(distances)),
            "interpolation": str(args.interpolation),
            "code_identity_sha256": code_identity_hash,
            "sha256": _sha256(path),
        }
        manifest["updated_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        _atomic_json(
            root / "heartbeat.json",
            {
                "schema": P2_TABLE_SCHEMA,
                "stage": "projector_tables",
                "pair": key,
                "complete": pair_index,
                "total": len(projector_pairs),
                "updated_unix": time.time(),
            },
        )
        built += 1

    for symbol in symbols:
        relative = Path(species_manifest[symbol]["array_path"])
        _atomic_npz(root / relative, **species_arrays[symbol])
        species_manifest[symbol]["array_sha256"] = _sha256(root / relative)
    manifest["species"] = species_manifest
    manifest["complete"] = False
    manifest["updated_unix"] = time.time()
    manifest["build_seconds"] = time.time() - started
    manifest["built_shards_this_run"] = built
    manifest["skipped_shards_this_run"] = skipped
    _qualify_and_publish_manifest(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        component_reciprocity_tolerance=float(
            getattr(args, "component_reciprocity_tolerance", 1.0e-6)
        ),
        component_reconstruction_tolerance=float(
            getattr(args, "component_reconstruction_tolerance", 1.0e-6)
        ),
        component_gate_distance_samples=int(
            getattr(args, "component_gate_distance_samples", 33)
        ),
    )
    _atomic_json(
        root / "heartbeat.json",
        {
            "schema": P2_TABLE_SCHEMA,
            "stage": "complete",
            "species": len(symbols),
            "base_tables": len(manifest["base_tables"]),
            "projector_tables": len(manifest["projector_tables"]),
            "build_seconds": manifest["build_seconds"],
            "updated_unix": time.time(),
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pp-orb", type=Path, required=True)
    parser.add_argument("--gate1-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--species", default="")
    parser.add_argument("--distance-step", type=float, default=0.02)
    parser.add_argument("--kmax", type=float, default=70.0)
    parser.add_argument("--n-k", type=int, default=600)
    parser.add_argument("--n-mu", type=int, default=18)
    parser.add_argument("--n-phi", type=int, default=36)
    parser.add_argument("--distance-chunk", type=int, default=128)
    parser.add_argument("--interpolation", choices=["linear", "cubic"], default="cubic")
    parser.add_argument(
        "--base-components",
        choices=["none", "overlap", "all"],
        default="none",
        help=(
            "optional reusable base arrays: none keeps only P2 base; overlap "
            "also stores S; all additionally stores kinetic and endpoint VNA "
            "left/right arrays. These component arrays are table-level "
            "diagnostic/oracle payloads and are not materialized as current "
            "node_p2/edge_p2 training side channels."
        ),
    )
    parser.add_argument("--component-reciprocity-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--component-reconstruction-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--component-gate-distance-samples", type=int, default=33)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args)
    print(
        json.dumps(
            {
                "table_root": str(args.output.resolve()),
                "species": sorted(result["species"]),
                "base_components": result["base_component_contract"]["mode"],
                "component_table_scope": result["component_table_semantics"][
                    "scope"
                ],
                "base_tables": len(result["base_tables"]),
                "projector_tables": len(result["projector_tables"]),
                "component_numerical_gate": result["component_numerical_gate"][
                    "status"
                ],
                "build_seconds": result["build_seconds"],
                "complete": result["complete"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
