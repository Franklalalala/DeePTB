"""Fast non-SOC P2 assembly from versioned two-centre radial tables.

The expensive ORB/UPF quadrature belongs to the offline table builder.  This
module is deliberately independent of that builder (and of ``h0rebuild``): at
training/inference time it only loads immutable table shards, interpolates
canonical Slater--Koster blocks, rotates them, and contracts factorised
projector overlaps.

The assembled operator is the Gate-1 P2 contract in Rydberg::

    T + VNA(i) + VNA(j) + sum_K <phi_i|beta_K> D_K <beta_K|phi_j>

For an onsite block ``i == j, R == 0`` the coincident VNA centre is counted
once.  The table builder stores that special block explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
try:  # SciPy >= 1.15
    from scipy.special import sph_harm_y as _scipy_sph_harm

    def _complex_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray):
        return _scipy_sph_harm(l, m, theta, phi)

except ImportError:  # pragma: no cover - exercised on older supported SciPy
    from scipy.special import sph_harm as _legacy_sph_harm

    def _complex_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray):
        # Legacy order is (m, l, azimuth=phi, polar=theta).
        return _legacy_sph_harm(m, l, phi, theta)

try:
    from .p2_batch import (
        ExactDirectionRotationCache,
        StructureLocalP2BatchContext,
        canonical_float64_bytes,
        evaluate_many as _evaluate_many_radial_table,
    )
except ImportError:  # pragma: no cover - supports path-loaded smoke scripts
    try:
        import importlib.util as _importlib_util
        import sys as _sys

        _p2_batch_path = Path(__file__).with_name("p2_batch.py")
        _p2_batch_spec = _importlib_util.spec_from_file_location(
            "_deeptb_p2_batch_path_loaded",
            _p2_batch_path,
        )
        if _p2_batch_spec is None or _p2_batch_spec.loader is None:
            raise
        _p2_batch_module = _importlib_util.module_from_spec(_p2_batch_spec)
        _sys.modules[_p2_batch_spec.name] = _p2_batch_module
        _p2_batch_spec.loader.exec_module(_p2_batch_module)
        ExactDirectionRotationCache = _p2_batch_module.ExactDirectionRotationCache
        StructureLocalP2BatchContext = (
            _p2_batch_module.StructureLocalP2BatchContext
        )
        canonical_float64_bytes = _p2_batch_module.canonical_float64_bytes
        _evaluate_many_radial_table = _p2_batch_module.evaluate_many
    except (ImportError, OSError):
        from dptb.data.interfaces.p2_batch import (
            ExactDirectionRotationCache,
            StructureLocalP2BatchContext,
            canonical_float64_bytes,
            evaluate_many as _evaluate_many_radial_table,
        )


P2_TABLE_SCHEMA = "deeptb.p2_radial_table/v1"
P2_SPECIES_SOURCE_IDENTITY_SCHEMA = "deeptb.p2_species_source_identity/v1"
P2_BASE_COMPONENT_SCHEMA = "deeptb.p2_base_components/v1"
P2_COMPONENT_NUMERICAL_GATE_SCHEMA = "deeptb.p2_component_numerical_gate/v1"
P2_BASE_COMPONENT_MODES = ("none", "overlap", "all")


def p2_base_component_contract(mode: str) -> dict[str, Any]:
    """Canonical manifest contract for optional reusable base components."""

    normalized = str(mode).lower()
    if normalized not in P2_BASE_COMPONENT_MODES:
        raise ValueError(
            f"base component mode {mode!r} is not one of {P2_BASE_COMPONENT_MODES}"
        )
    has_overlap = normalized in {"overlap", "all"}
    has_all = normalized == "all"
    return {
        "schema": P2_BASE_COMPONENT_SCHEMA,
        "mode": normalized,
        "base_shard": {
            "p2_base": {"array": "values", "unit": "Ry", "available": True},
            "overlap": {
                "array": "overlap_values",
                "unit": "dimensionless",
                "available": has_overlap,
            },
            "kinetic": {
                "array": "kinetic_values",
                "unit": "Ry",
                "available": has_all,
            },
            "vna_left": {
                "array": "vna_left_values",
                "unit": "Ry",
                "available": has_all,
            },
            "vna_right": {
                "array": "vna_right_values",
                "unit": "Ry",
                "available": has_all,
            },
        },
        "species_shard": {
            "p2_base": {
                "array": "onsite_base",
                "unit": "Ry",
                "available": True,
            },
            "overlap": {
                "array": "onsite_overlap",
                "unit": "dimensionless",
                "available": has_overlap,
            },
            "kinetic": {
                "array": "onsite_kinetic",
                "unit": "Ry",
                "available": has_all,
            },
            "vna_endpoint": {
                "array": "onsite_vna_endpoint",
                "unit": "Ry",
                "available": has_all,
            },
        },
        "onsite_vna_semantics": "unique_endpoint_counted_once",
    }


def normalized_p2_base_component_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the declared component contract; legacy omission means none."""

    declared = manifest.get("base_component_contract")
    if declared is None:
        return p2_base_component_contract("none")
    if not isinstance(declared, Mapping):
        raise ValueError("base_component_contract must be a mapping")
    mode = str(declared.get("mode", "")).lower()
    expected = p2_base_component_contract(mode)
    if dict(declared) != expected:
        raise ValueError(
            "P2 base component contract is not the canonical declaration for "
            f"mode {mode!r}"
        )
    return expected


def _available_component_arrays(
    contract: Mapping[str, Any], section: str
) -> dict[str, str]:
    return {
        str(name): str(spec["array"])
        for name, spec in contract[section].items()
        if bool(spec["available"])
    }


def _normalized_sha256(value: Any, *, label: str) -> str:
    checksum = str(value or "").strip().lower()
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError(f"{label} is missing a valid SHA256 checksum")
    return checksum


def canonical_species_source_identity(
    species: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the content-addressed species/basis identity of a P2 table.

    Paths are deliberately excluded: relocating an identical ORB/UPF bundle is
    safe, whereas changing either file, its shell sequence, or its projector
    interpretation is not.  The symbol is part of the mapping key, so swapping
    two otherwise valid source files also changes the identity.
    """

    canonical: dict[str, Any] = {}
    for symbol in sorted(species):
        metadata = species[symbol]
        orbital_shells = [int(x) for x in metadata.get("orbital_shells", [])]
        projector_shells = [int(x) for x in metadata.get("projector_shells", [])]
        projector_cutoffs = [
            float(x) for x in metadata.get("projector_cutoffs_bohr", [])
        ]
        if len(projector_shells) != len(projector_cutoffs):
            raise ValueError(
                f"species {symbol!r} has inconsistent projector shells/cutoffs"
            )
        orbital_norb = int(metadata.get("orbital_norb", -1))
        projector_norb = int(metadata.get("projector_norb", -1))
        expected_orbital_norb = sum(2 * l + 1 for l in orbital_shells)
        expected_projector_norb = sum(2 * l + 1 for l in projector_shells)
        if orbital_norb != expected_orbital_norb:
            raise ValueError(
                f"species {symbol!r} orbital_norb {orbital_norb} does not match "
                f"shell sequence {orbital_shells} ({expected_orbital_norb})"
            )
        if projector_norb != expected_projector_norb:
            raise ValueError(
                f"species {symbol!r} projector_norb {projector_norb} does not "
                f"match shell sequence {projector_shells} "
                f"({expected_projector_norb})"
            )
        canonical[str(symbol)] = {
            "orbital_sha256": _normalized_sha256(
                metadata.get("orbital_sha256"),
                label=f"species:{symbol}:orbital_sha256",
            ),
            "upf_sha256": _normalized_sha256(
                metadata.get("upf_sha256"),
                label=f"species:{symbol}:upf_sha256",
            ),
            "orbital_shells": orbital_shells,
            "orbital_norb": orbital_norb,
            "orbital_cutoff_bohr": float(metadata["orbital_cutoff_bohr"]),
            "projector_shells": projector_shells,
            "projector_norb": projector_norb,
            "projector_cutoffs_bohr": projector_cutoffs,
            "projector_max_cutoff_bohr": float(
                metadata["projector_max_cutoff_bohr"]
            ),
            "upf_has_so": bool(metadata.get("upf_has_so", False)),
        }
    if not canonical:
        raise ValueError("P2 table species identity cannot be empty")
    return {
        "schema": P2_SPECIES_SOURCE_IDENTITY_SCHEMA,
        "species": canonical,
    }


def species_source_identity_sha256(
    species_or_identity: Mapping[str, Any],
) -> str:
    """Hash a canonical species identity or a manifest ``species`` mapping."""

    if species_or_identity.get("schema") == P2_SPECIES_SOURCE_IDENTITY_SCHEMA:
        identity = dict(species_or_identity)
    else:
        identity = canonical_species_source_identity(species_or_identity)
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _abs_distribution(chunks: Sequence[np.ndarray]) -> dict[str, Any]:
    if not chunks:
        return {
            "count": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p50_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
        }
    values = np.concatenate(
        [np.ravel(np.asarray(chunk, dtype=np.float64)) for chunk in chunks]
    )
    if values.size == 0:
        return {
            "count": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p50_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
        }
    abs_values = np.abs(values)
    return {
        "count": int(abs_values.size),
        "max_abs": float(np.max(abs_values, initial=0.0)),
        "mean_abs": float(np.mean(abs_values)),
        "p50_abs": float(np.quantile(abs_values, 0.50)),
        "p95_abs": float(np.quantile(abs_values, 0.95)),
        "p99_abs": float(np.quantile(abs_values, 0.99)),
    }


def _component_gate_displacements(
    support_bohr: float,
    *,
    distance_samples: int,
) -> list[np.ndarray]:
    support = float(support_bohr)
    if support <= 0.0:
        return [np.zeros(3, dtype=np.float64)]
    count = max(2, int(distance_samples))
    distances = np.linspace(0.0, 0.97 * support, count, dtype=np.float64)
    directions = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.31, -0.47, 0.826], dtype=np.float64),
    ]
    directions[1] /= np.linalg.norm(directions[1])
    displacements = [np.zeros(3, dtype=np.float64)]
    for distance in distances[1:]:
        for direction in directions:
            displacements.append(direction * float(distance))
    return displacements


def validate_p2_component_numerical_contract(
    store: "P2TableStore",
    *,
    reciprocity_tolerance: float = 1.0e-6,
    reconstruction_tolerance: float = 1.0e-6,
    distance_samples: int = 33,
) -> dict[str, Any]:
    """Fail-closed numerical validation for declared optional P2 components."""

    reciprocal_components = {
        "overlap": "overlap",
        "kinetic": "kinetic",
        "vna_left": "vna_right",
        "vna_right": "vna_left",
    }
    reciprocity_by_component: dict[str, Any] = {}
    failures: list[str] = []
    for component, reverse_component in reciprocal_components.items():
        if (
            component not in store.base_component_arrays
            or reverse_component not in store.base_component_arrays
        ):
            continue
        chunks: list[np.ndarray] = []
        sample_count = 0
        for key in sorted(store.base_tables):
            left, right = key.split("|", 1)
            table = store.base_component(left, right, component)
            reverse = store.base_component(right, left, reverse_component)
            support = min(float(table.support_bohr), float(reverse.support_bohr))
            for displacement in _component_gate_displacements(
                support, distance_samples=distance_samples
            ):
                direct = table.evaluate(displacement)
                opposite = reverse.evaluate(-displacement).T
                chunks.append(direct - opposite)
                sample_count += 1
        summary = _abs_distribution(chunks)
        passed = summary["max_abs"] <= float(reciprocity_tolerance)
        if not passed:
            failures.append(
                f"{component}->{reverse_component} reciprocity max "
                f"{summary['max_abs']:.6e} exceeds "
                f"{float(reciprocity_tolerance):.6e}"
            )
        reciprocity_by_component[component] = {
            "reverse_component": reverse_component,
            "pair_count": int(len(store.base_tables)),
            "sample_points": int(sample_count),
            "summary": summary,
            "pass": passed,
        }

    onsite_transpose_by_component: dict[str, Any] = {}
    for component in ("overlap", "kinetic", "vna_endpoint"):
        if component not in store.onsite_component_arrays:
            continue
        chunks = []
        for symbol in sorted(store.species):
            value = store.onsite_component(symbol, component)
            chunks.append(value - value.T)
        summary = _abs_distribution(chunks)
        passed = summary["max_abs"] <= float(reciprocity_tolerance)
        if not passed:
            failures.append(
                f"onsite {component} transpose max {summary['max_abs']:.6e} "
                f"exceeds {float(reciprocity_tolerance):.6e}"
            )
        onsite_transpose_by_component[component] = {
            "species_count": int(len(store.species)),
            "summary": summary,
            "pass": passed,
        }

    base_reconstruction_chunks: list[np.ndarray] = []
    onsite_reconstruction_chunks: list[np.ndarray] = []
    if {"p2_base", "kinetic", "vna_left", "vna_right"}.issubset(
        store.base_component_arrays
    ):
        for key in sorted(store.base_tables):
            left, right = key.split("|", 1)
            p2_base = np.asarray(
                store.base_component(left, right, "p2_base").values,
                dtype=np.float64,
            )
            kinetic = np.asarray(
                store.base_component(left, right, "kinetic").values,
                dtype=np.float64,
            )
            vna_left = np.asarray(
                store.base_component(left, right, "vna_left").values,
                dtype=np.float64,
            )
            vna_right = np.asarray(
                store.base_component(left, right, "vna_right").values,
                dtype=np.float64,
            )
            base_reconstruction_chunks.append(
                p2_base - (kinetic + vna_left + vna_right)
            )
    if {"p2_base", "kinetic", "vna_endpoint"}.issubset(
        store.onsite_component_arrays
    ):
        for symbol in sorted(store.species):
            p2_base = store.onsite_component(symbol, "p2_base")
            kinetic = store.onsite_component(symbol, "kinetic")
            vna_endpoint = store.onsite_component(symbol, "vna_endpoint")
            onsite_reconstruction_chunks.append(p2_base - (kinetic + vna_endpoint))

    base_reconstruction = _abs_distribution(base_reconstruction_chunks)
    onsite_reconstruction = _abs_distribution(onsite_reconstruction_chunks)
    combined_reconstruction = _abs_distribution(
        base_reconstruction_chunks + onsite_reconstruction_chunks
    )
    reconstruction_pass = combined_reconstruction["max_abs"] <= float(
        reconstruction_tolerance
    )
    if not reconstruction_pass:
        failures.append(
            "component reconstruction max "
            f"{combined_reconstruction['max_abs']:.6e} exceeds "
            f"{float(reconstruction_tolerance):.6e}"
        )

    return {
        "schema": P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
        "mode": store.base_component_mode,
        "reciprocity_tolerance": float(reciprocity_tolerance),
        "reconstruction_tolerance": float(reconstruction_tolerance),
        "distance_samples_per_pair_axis": int(max(2, int(distance_samples))),
        "reciprocity_by_component": reciprocity_by_component,
        "onsite_transpose_by_component": onsite_transpose_by_component,
        "reconstruction": {
            "base": base_reconstruction,
            "onsite": onsite_reconstruction,
            "combined": combined_reconstruction,
            "pass": reconstruction_pass,
        },
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    radius = np.linalg.norm(vectors, axis=-1)
    safe = np.where(radius > 0.0, radius, 1.0)
    theta = np.arccos(np.clip(vectors[..., 2] / safe, -1.0, 1.0))
    phi = np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), 2.0 * np.pi)
    theta = np.where(radius > 0.0, theta, 0.0)
    phi = np.where(radius > 0.0, phi, 0.0)
    return radius, theta, phi


def real_sph_abacus(l: int, m: int, vectors: np.ndarray) -> np.ndarray:
    """ABACUS real spherical harmonic used by the direct P2 oracle.

    The order of a shell is handled separately as ``0,+1,-1,+2,-2,...``.
    This function intentionally mirrors ``h0rebuild.harmonics`` without
    importing the offline package in the inference path.
    """

    _, theta, phi = _angles(vectors)
    if m == 0:
        return _complex_sph_harm(int(l), 0, theta, phi).real
    ylm = _complex_sph_harm(int(l), abs(int(m)), theta, phi)
    return np.sqrt(2.0) * (ylm.real if m > 0 else ylm.imag)


def abacus_m_order(l: int) -> list[int]:
    order = [0]
    for m in range(1, int(l) + 1):
        order.extend((m, -m))
    return order


def rotation_z_to(direction: np.ndarray) -> np.ndarray:
    """Return a proper Cartesian rotation mapping +z to ``direction``."""

    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("rotation direction must be non-zero")
    n = direction / norm
    z = np.asarray([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(z, n), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-14:
        return np.eye(3)
    if cosine < -1.0 + 1.0e-14:
        # pi around x: proper rotation with determinant +1.
        return np.diag([1.0, -1.0, -1.0])
    cross = np.cross(z, n)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


class RealHarmonicRotator:
    """Small, convention-safe real-harmonic rotation matrices.

    The basis transformation is recovered once from an overdetermined angular
    collocation grid.  A final polar projection removes numerical loss of
    orthogonality.  This is inexpensive for the NAO shells used here (l <= 4)
    and avoids assuming that another library uses ABACUS's real-harmonic signs.
    """

    def __init__(self, lmax: int):
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative")
        n_mu = max(8, 2 * self.lmax + 5)
        n_phi = max(16, 4 * self.lmax + 8)
        mu, _ = np.polynomial.legendre.leggauss(n_mu)
        phi = (np.arange(n_phi, dtype=np.float64) + 0.5) * (2.0 * np.pi / n_phi)
        directions = []
        for value in mu:
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
        self.directions = np.vstack(directions)
        self._base: dict[int, np.ndarray] = {}
        self._pinv: dict[int, np.ndarray] = {}
        for l in range(self.lmax + 1):
            matrix = np.column_stack(
                [real_sph_abacus(l, m, self.directions) for m in abacus_m_order(l)]
            )
            self._base[l] = matrix
            self._pinv[l] = np.linalg.pinv(matrix)

    def matrix(self, l: int, cartesian_rotation: np.ndarray) -> np.ndarray:
        l = int(l)
        if l < 0 or l > self.lmax:
            raise ValueError(f"l={l} is outside configured lmax={self.lmax}")
        q = np.asarray(cartesian_rotation, dtype=np.float64)
        if q.shape != (3, 3):
            raise ValueError(f"Cartesian rotation must be [3,3], got {q.shape}")
        rotated = self.directions @ q.T
        values = np.column_stack(
            [real_sph_abacus(l, m, rotated) for m in abacus_m_order(l)]
        )
        candidate = (self._pinv[l] @ values).T
        u, _, vt = np.linalg.svd(candidate)
        return u @ vt


def _shell_groups(shells: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    groups: list[tuple[int, np.ndarray]] = []
    offset = 0
    for l_raw in shells:
        l = int(l_raw)
        width = 2 * l + 1
        groups.append((l, np.arange(offset, offset + width, dtype=np.int64)))
        offset += width
    return groups


@dataclass
class RadialBlockTable:
    distances: np.ndarray
    values: np.ndarray
    left_shells: tuple[int, ...]
    right_shells: tuple[int, ...]
    support_bohr: float
    interpolation: str = "cubic"

    def __post_init__(self) -> None:
        self.distances = np.asarray(self.distances, dtype=np.float64)
        self.values = np.asarray(self.values)
        self.left_shells = tuple(int(x) for x in self.left_shells)
        self.right_shells = tuple(int(x) for x in self.right_shells)
        if self.distances.ndim != 1 or self.distances.size < 2:
            raise ValueError("radial table needs at least two distance knots")
        if not np.all(np.diff(self.distances) > 0.0):
            raise ValueError("radial table distances must be strictly increasing")
        expected = (
            self.distances.size,
            sum(2 * l + 1 for l in self.left_shells),
            sum(2 * l + 1 for l in self.right_shells),
        )
        if self.values.shape != expected:
            raise ValueError(f"radial table values {self.values.shape} != {expected}")
        if not np.isfinite(self.values).all():
            raise ValueError("radial table contains NaN or infinity")
        if self.support_bohr <= 0.0:
            raise ValueError("radial table support must be positive")
        if self.distances[0] > 1.0e-12:
            raise ValueError("radial table must include d=0")
        if self.distances[-1] + 1.0e-12 < self.support_bohr:
            raise ValueError("radial table does not cover its declared support")
        self.interpolation = str(self.interpolation).lower()
        if self.interpolation not in {"linear", "cubic"}:
            raise ValueError("interpolation must be 'linear' or 'cubic'")
        lmax = max((*self.left_shells, *self.right_shells), default=0)
        self._rotator = RealHarmonicRotator(lmax)
        self._spline = None
        if self.interpolation == "cubic" and self.distances.size >= 4:
            self._spline = CubicSpline(
                self.distances,
                np.asarray(self.values, dtype=np.float64),
                axis=0,
                extrapolate=False,
            )
        self._rotation_cache = ExactDirectionRotationCache(
            self._rotator, rotation_z_to
        )

    def _interpolate(self, distance: float) -> np.ndarray:
        if distance >= self.support_bohr - 1.0e-12:
            return np.zeros(self.values.shape[1:], dtype=np.float64)
        if distance < -1.0e-12:
            raise ValueError("distance cannot be negative")
        d = max(0.0, float(distance))
        if self._spline is not None:
            return np.asarray(self._spline(d), dtype=np.float64)
        upper = int(np.searchsorted(self.distances, d, side="right"))
        upper = min(max(upper, 1), self.distances.size - 1)
        lower = upper - 1
        d0 = float(self.distances[lower])
        d1 = float(self.distances[upper])
        weight = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
        return (
            (1.0 - weight) * np.asarray(self.values[lower], dtype=np.float64)
            + weight * np.asarray(self.values[upper], dtype=np.float64)
        )

    def evaluate(self, displacement_bohr: np.ndarray) -> np.ndarray:
        displacement = np.asarray(displacement_bohr, dtype=np.float64)
        if displacement.shape != (3,):
            raise ValueError(f"displacement must be length 3, got {displacement.shape}")
        distance = float(np.linalg.norm(displacement))
        canonical = self._interpolate(distance)
        if distance <= 1.0e-14 or not np.any(canonical):
            return canonical
        rotation = rotation_z_to(displacement / distance)
        left_matrices = {
            l: self._rotator.matrix(l, rotation) for l in set(self.left_shells)
        }
        right_matrices = {
            l: self._rotator.matrix(l, rotation) for l in set(self.right_shells)
        }
        output = np.empty_like(canonical)
        for l_left, left in _shell_groups(self.left_shells):
            ml = left_matrices[l_left]
            for l_right, right in _shell_groups(self.right_shells):
                mr = right_matrices[l_right]
                output[np.ix_(left, right)] = (
                    ml @ canonical[np.ix_(left, right)] @ mr.T
                )
        return output

    def evaluate_many(
        self,
        displacements_bohr: np.ndarray,
        *,
        rotation_cache: ExactDirectionRotationCache | None = None,
    ) -> np.ndarray:
        """Vectorized equivalent of repeated scalar ``evaluate`` calls."""

        cache = self._rotation_cache if rotation_cache is None else rotation_cache
        return _evaluate_many_radial_table(self, displacements_bohr, rotation_cache=cache)

    def clear_rotation_cache(self, *, reset_stats: bool = True) -> None:
        self._rotation_cache.clear(reset_stats=reset_stats)

    def rotation_cache_snapshot(self) -> dict[str, Any]:
        return self._rotation_cache.snapshot()


def _translation_bounds(cell_bohr: np.ndarray, radius: float) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(cell_bohr, dtype=np.float64))
    return np.ceil(float(radius) * np.linalg.norm(inverse, axis=0)).astype(int) + 1


def nearby_atom_images(
    cell_bohr: np.ndarray,
    base_center: np.ndarray,
    target: np.ndarray,
    radius: float,
) -> Iterable[tuple[tuple[int, int, int], np.ndarray]]:
    """Yield periodic images of ``base_center`` within ``radius`` of target."""

    cell = np.asarray(cell_bohr, dtype=np.float64)
    base = np.asarray(base_center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    guess = np.rint((target - base) @ inverse).astype(int)
    bounds = _translation_bounds(cell, radius)
    for offset in itertools.product(*(range(-b, b + 1) for b in bounds)):
        translation = guess + np.asarray(offset, dtype=int)
        center = base + translation @ cell
        if np.linalg.norm(center - target) <= float(radius) + 1.0e-12:
            yield tuple(int(x) for x in translation), center


def _table_key(left: str, right: str) -> str:
    return f"{left}|{right}"


class P2TableStore:
    """Lazy, checksum-aware reader for a sharded P2 radial-table directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_cached_tables: int = 64,
        verify_checksums: bool = True,
        require_explicit_source_identity: bool = False,
    ):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"P2 table manifest missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != P2_TABLE_SCHEMA:
            raise ValueError(
                f"P2 table schema {self.manifest.get('schema')!r} != {P2_TABLE_SCHEMA!r}"
            )
        if self.manifest.get("length_unit") != "bohr":
            raise ValueError("P2 table length unit must be bohr")
        if self.manifest.get("energy_unit") != "Ry":
            raise ValueError("P2 table energy unit must be Ry")
        if self.manifest.get("complete") is not True:
            raise ValueError("P2 table manifest is not marked complete")
        self.species: Mapping[str, Mapping[str, Any]] = self.manifest["species"]
        self.base_tables: Mapping[str, Mapping[str, Any]] = self.manifest["base_tables"]
        self.projector_tables: Mapping[str, Mapping[str, Any]] = self.manifest[
            "projector_tables"
        ]
        self.base_component_contract = normalized_p2_base_component_contract(
            self.manifest
        )
        self.base_component_mode = str(self.base_component_contract["mode"])
        self.base_component_arrays = _available_component_arrays(
            self.base_component_contract, "base_shard"
        )
        self.onsite_component_arrays = _available_component_arrays(
            self.base_component_contract, "species_shard"
        )
        for key, metadata in self.base_tables.items():
            declared_arrays = metadata.get("component_arrays")
            if declared_arrays is None and self.base_component_mode == "none":
                declared_arrays = {"p2_base": "values"}
            if declared_arrays != self.base_component_arrays:
                raise ValueError(
                    f"base:{key} component arrays {declared_arrays!r} do not match "
                    f"manifest contract {self.base_component_arrays!r}"
                )
        for symbol, metadata in self.species.items():
            declared_arrays = metadata.get("onsite_component_arrays")
            if declared_arrays is None and self.base_component_mode == "none":
                declared_arrays = {"p2_base": "onsite_base"}
            if declared_arrays != self.onsite_component_arrays:
                raise ValueError(
                    f"species:{symbol} onsite component arrays "
                    f"{declared_arrays!r} do not match manifest contract "
                    f"{self.onsite_component_arrays!r}"
                )
        declared_identity = self.manifest.get("species_source_identity")
        declared_identity_sha256 = self.manifest.get(
            "species_source_identity_sha256"
        )
        self.has_explicit_source_identity = (
            declared_identity is not None and declared_identity_sha256 is not None
        )
        if (declared_identity is None) != (declared_identity_sha256 is None):
            raise ValueError(
                "P2 manifest must declare both species_source_identity and its SHA256"
            )
        try:
            derived_identity = canonical_species_source_identity(self.species)
            derived_identity_sha256 = species_source_identity_sha256(derived_identity)
        except (KeyError, TypeError, ValueError) as exc:
            if require_explicit_source_identity or declared_identity is not None:
                raise ValueError(
                    "P2 manifest cannot establish a complete ORB/UPF species identity"
                ) from exc
            derived_identity = None
            derived_identity_sha256 = None
        if declared_identity is not None:
            if declared_identity != derived_identity:
                raise ValueError(
                    "P2 manifest species_source_identity disagrees with species metadata"
                )
            expected_identity_sha256 = _normalized_sha256(
                declared_identity_sha256,
                label="species_source_identity_sha256",
            )
            if expected_identity_sha256 != derived_identity_sha256:
                raise ValueError(
                    "P2 manifest species source identity checksum mismatch: "
                    f"manifest={expected_identity_sha256}, "
                    f"derived={derived_identity_sha256}"
                )
        if require_explicit_source_identity and not self.has_explicit_source_identity:
            raise ValueError(
                "P2 manifest lacks an explicit species source identity; rebuild or "
                "resume once with the hardened builder before production use"
            )
        self.species_source_identity = derived_identity
        self.species_source_identity_sha256 = derived_identity_sha256
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.verify_checksums = bool(verify_checksums)
        self._cache: OrderedDict[str, RadialBlockTable] = OrderedDict()
        self._species_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._verified_paths: set[Path] = set()
        self._source_hash_cache: dict[Path, tuple[int, int, str]] = {}
        symbols = sorted(self.species)
        expected_base = {_table_key(left, right) for left in symbols for right in symbols}
        missing_base = sorted(expected_base - set(self.base_tables))
        if missing_base:
            raise ValueError(
                f"P2 table is missing {len(missing_base)} ordered base pairs; "
                f"first={missing_base[:5]}."
            )
        expected_projector = {
            _table_key(projector, orbital)
            for projector in symbols
            if int(self.species[projector].get("projector_norb", 0)) > 0
            for orbital in symbols
        }
        missing_projector = sorted(expected_projector - set(self.projector_tables))
        if missing_projector:
            raise ValueError(
                f"P2 table is missing {len(missing_projector)} ordered projector pairs; "
                f"first={missing_projector[:5]}."
            )

    @property
    def verified_path_count(self) -> int:
        return len(self._verified_paths)

    def _source_file_sha256(self, path: Path) -> str:
        """Hash immutable source files once, invalidating on stat changes."""

        before = path.stat()
        signature = (int(before.st_size), int(before.st_mtime_ns))
        cached = self._source_hash_cache.get(path)
        if cached is not None and cached[:2] == signature:
            return cached[2]
        checksum = _sha256(path)
        after = path.stat()
        if (int(after.st_size), int(after.st_mtime_ns)) != signature:
            raise ValueError(f"source file changed while hashing: {path}")
        self._source_hash_cache[path] = (*signature, checksum)
        return checksum

    def validate_sample_contract(
        self,
        *,
        symbols: Sequence[str],
        atom_shells: Sequence[Sequence[int]],
        source_files: Mapping[str, Mapping[str, str | Path]] | None = None,
        require_source_files: bool = False,
    ) -> dict[str, Any]:
        """Bind one structure's exact basis and ORB/UPF files to this table.

        ``orbital_norb`` alone is insufficient: ``1s1p`` and ``4s`` both have
        four AO functions but require different rotations and table channels.
        Production consumers should also provide the ORB/UPF paths selected by
        the structure so their bytes can be checked against the manifest.
        """

        sample_symbols = [str(symbol) for symbol in symbols]
        sample_shells = [tuple(int(l) for l in shells) for shells in atom_shells]
        if len(sample_symbols) != len(sample_shells):
            raise ValueError(
                f"sample symbols ({len(sample_symbols)}) and shell rows "
                f"({len(sample_shells)}) differ"
            )
        for atom_index, (symbol, shells) in enumerate(
            zip(sample_symbols, sample_shells)
        ):
            if symbol not in self.species:
                raise KeyError(f"sample species {symbol!r} is absent from P2 table")
            expected_shells = tuple(
                int(l) for l in self.species[symbol].get("orbital_shells", [])
            )
            if shells != expected_shells:
                raise ValueError(
                    f"atom {atom_index} ({symbol}) shell sequence {list(shells)} "
                    f"does not match P2 table {list(expected_shells)}"
                )

        used_symbols = sorted(set(sample_symbols))
        if require_source_files and source_files is None:
            raise ValueError(
                "production sample validation requires structure-selected ORB/UPF files"
            )
        verified_sources: list[dict[str, str]] = []
        if source_files is not None:
            if self.species_source_identity is None:
                raise ValueError(
                    "P2 table lacks ORB/UPF hashes required for sample source validation"
                )
            for symbol in used_symbols:
                if symbol not in source_files:
                    raise KeyError(f"sample source files are missing species {symbol!r}")
                files = source_files[symbol]
                orbital_path = Path(files["orbital"]).resolve()
                upf_path = Path(files["upf"]).resolve()
                if not orbital_path.is_file() or not upf_path.is_file():
                    raise FileNotFoundError(
                        f"sample ORB/UPF missing for {symbol}: "
                        f"{orbital_path}, {upf_path}"
                    )
                expected = self.species[symbol]
                orbital_sha256 = self._source_file_sha256(orbital_path)
                upf_sha256 = self._source_file_sha256(upf_path)
                expected_orbital = _normalized_sha256(
                    expected.get("orbital_sha256"),
                    label=f"species:{symbol}:orbital_sha256",
                )
                expected_upf = _normalized_sha256(
                    expected.get("upf_sha256"),
                    label=f"species:{symbol}:upf_sha256",
                )
                if orbital_sha256 != expected_orbital:
                    raise ValueError(
                        f"sample species {symbol} ORB checksum mismatch: "
                        f"table={expected_orbital}, sample={orbital_sha256}"
                    )
                if upf_sha256 != expected_upf:
                    raise ValueError(
                        f"sample species {symbol} UPF checksum mismatch: "
                        f"table={expected_upf}, sample={upf_sha256}"
                    )
                verified_sources.append(
                    {
                        "symbol": symbol,
                        "orbital_sha256": orbital_sha256,
                        "upf_sha256": upf_sha256,
                    }
                )
        return {
            "atom_count": len(sample_symbols),
            "species": used_symbols,
            "shell_sequence_verified": True,
            "source_files_verified": source_files is not None,
            "verified_sources": verified_sources,
            "species_source_identity_sha256": self.species_source_identity_sha256,
        }

    def _verify_path(self, path: Path, expected: Any, *, label: str) -> None:
        if not self.verify_checksums or path in self._verified_paths:
            return
        expected_text = str(expected or "").strip().lower()
        if len(expected_text) != 64:
            raise ValueError(f"{label} is missing a valid SHA256 checksum")
        actual = _sha256(path)
        if actual != expected_text:
            raise ValueError(
                f"{label} checksum mismatch: manifest={expected_text}, actual={actual}"
            )
        self._verified_paths.add(path)

    def _load_table(
        self,
        category: str,
        key: str,
        metadata: Mapping[str, Any],
        *,
        value_array: str = "values",
    ) -> RadialBlockTable:
        value_array = str(value_array)
        cache_key = f"{category}:{key}:{value_array}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        path = (self.root / str(metadata["path"])).resolve()
        if self.root not in path.parents:
            raise ValueError(f"table shard escapes table root: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        self._verify_path(path, metadata.get("sha256"), label=f"{category}:{key}")
        with np.load(path, allow_pickle=False) as payload:
            if value_array not in payload.files:
                raise KeyError(
                    f"{category}:{key} shard does not contain declared array "
                    f"{value_array!r}"
                )
            table = RadialBlockTable(
                distances=np.asarray(payload["distances"]),
                values=np.asarray(payload[value_array]),
                left_shells=tuple(int(x) for x in payload["left_shells"]),
                right_shells=tuple(int(x) for x in payload["right_shells"]),
                support_bohr=float(np.asarray(payload["support_bohr"]).item()),
                interpolation=str(metadata.get("interpolation", "cubic")),
            )
        self._cache[cache_key] = table
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_cached_tables:
            self._cache.popitem(last=False)
        return table

    def base(
        self, left: str, right: str, *, value_array: str = "values"
    ) -> RadialBlockTable:
        key = _table_key(left, right)
        if key not in self.base_tables:
            raise KeyError(f"P2 base table missing for ordered pair {key}")
        declared_arrays = set(self.base_component_arrays.values())
        if str(value_array) not in declared_arrays:
            raise KeyError(
                f"P2 base array {value_array!r} is unavailable in component mode "
                f"{self.base_component_mode!r}; declared={sorted(declared_arrays)}"
            )
        return self._load_table(
            "base", key, self.base_tables[key], value_array=str(value_array)
        )

    def base_component(
        self, left: str, right: str, component: str
    ) -> RadialBlockTable:
        name = str(component)
        if name not in self.base_component_arrays:
            raise KeyError(
                f"P2 base component {name!r} is unavailable in mode "
                f"{self.base_component_mode!r}; available="
                f"{sorted(self.base_component_arrays)}"
            )
        return self.base(
            left,
            right,
            value_array=self.base_component_arrays[name],
        )

    def projector(self, projector_species: str, orbital_species: str) -> RadialBlockTable:
        key = _table_key(projector_species, orbital_species)
        if key not in self.projector_tables:
            raise KeyError(f"P2 projector table missing for ordered pair {key}")
        return self._load_table(
            "projector", key, self.projector_tables[key], value_array="values"
        )

    def species_arrays(self, symbol: str) -> Mapping[str, np.ndarray]:
        cached = self._species_arrays.get(symbol)
        if cached is not None:
            return cached
        if symbol not in self.species:
            raise KeyError(f"species {symbol!r} is absent from P2 table")
        path = (self.root / str(self.species[symbol]["array_path"])).resolve()
        if self.root not in path.parents:
            raise ValueError(f"species shard escapes table root: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        self._verify_path(
            path,
            self.species[symbol].get("array_sha256"),
            label=f"species:{symbol}",
        )
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        self._species_arrays[symbol] = arrays
        return arrays

    def onsite(self, symbol: str) -> np.ndarray:
        return self.onsite_component(symbol, "p2_base")

    def onsite_component(self, symbol: str, component: str) -> np.ndarray:
        if symbol not in self.species:
            raise KeyError(f"species {symbol!r} is absent from P2 table")
        name = str(component)
        if name not in self.onsite_component_arrays:
            raise KeyError(
                f"P2 onsite component {name!r} is unavailable in mode "
                f"{self.base_component_mode!r}; available="
                f"{sorted(self.onsite_component_arrays)}"
            )
        array_name = self.onsite_component_arrays[name]
        arrays = self.species_arrays(symbol)
        if array_name not in arrays:
            raise KeyError(
                f"species:{symbol} shard is missing declared onsite component "
                f"array {array_name!r}"
            )
        onsite = np.asarray(arrays[array_name], dtype=np.float64)
        expected = int(self.species[symbol]["orbital_norb"])
        if onsite.shape != (expected, expected):
            raise ValueError(
                f"onsite {symbol}:{name} shape {onsite.shape} != "
                f"{(expected, expected)}"
            )
        if not np.isfinite(onsite).all():
            raise ValueError(f"onsite {symbol}:{name} contains NaN or infinity")
        return onsite

    def d_eff(self, symbol: str) -> np.ndarray:
        arrays = self.species_arrays(symbol)
        value = np.asarray(arrays["d_eff"], dtype=np.float64)
        expected = int(self.species[symbol]["projector_norb"])
        if value.shape != (expected, expected):
            raise ValueError(f"D_eff {symbol} shape {value.shape} != {(expected, expected)}")
        return value


@dataclass(frozen=True)
class _BlockGeometry:
    symbols: tuple[str, ...]
    positions: np.ndarray
    cell: np.ndarray
    shift: np.ndarray
    i: int
    j: int
    si: str
    sj: str
    center_i: np.ndarray
    center_j: np.ndarray
    displacement: np.ndarray
    distance: float
    cutoff_i: float
    cutoff_j: float
    ni: int
    nj: int
    is_onsite: bool

    @property
    def outside_support(self) -> bool:
        return self.distance > self.cutoff_i + self.cutoff_j + 1.0e-10


class P2TableAssembler:
    """Assemble non-SOC P2 AO blocks without numerical integration.

    Mutable table, rotation, and assembly statistics are designed for one
    data-loader worker or one caller-owned assembler.  Do not share an assembler
    instance across threads without external synchronization.
    """

    def __init__(
        self,
        store: P2TableStore,
        *,
        projector_overlap_cache_max_bytes: int = 64 * 1024 * 1024,
    ):
        self.store = store
        self.projector_overlap_cache_max_bytes = max(
            0, int(projector_overlap_cache_max_bytes)
        )
        self._last_batch_stats: dict[str, Any] | None = None

    def _block_geometry(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        i: int,
        j: int,
        translation: Sequence[int],
    ) -> _BlockGeometry:
        normalized_symbols = tuple(str(x) for x in symbols)
        positions = np.asarray(positions_bohr, dtype=np.float64)
        cell = np.asarray(cell_bohr, dtype=np.float64)
        if positions.shape != (len(normalized_symbols), 3):
            raise ValueError("positions must have shape [n_atoms,3]")
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape [3,3]")
        shift_raw = np.asarray(translation)
        if shift_raw.shape != (3,) or not np.isfinite(shift_raw).all():
            raise ValueError("translation must have three finite integers")
        shift = shift_raw.astype(np.int64)
        if not np.array_equal(shift_raw, shift):
            raise ValueError("translation must contain exact integers")
        atom_i, atom_j = int(i), int(j)
        if atom_i < 0 or atom_i >= len(normalized_symbols):
            raise IndexError(f"atom i={atom_i} is outside the structure")
        if atom_j < 0 or atom_j >= len(normalized_symbols):
            raise IndexError(f"atom j={atom_j} is outside the structure")
        si, sj = normalized_symbols[atom_i], normalized_symbols[atom_j]
        if si not in self.store.species or sj not in self.store.species:
            raise KeyError(f"P2 table does not cover block species {si}|{sj}")
        center_i = positions[atom_i]
        center_j = positions[atom_j] + shift @ cell
        displacement = center_j - center_i
        distance = float(np.linalg.norm(displacement))
        return _BlockGeometry(
            symbols=normalized_symbols,
            positions=positions,
            cell=cell,
            shift=shift,
            i=atom_i,
            j=atom_j,
            si=si,
            sj=sj,
            center_i=center_i,
            center_j=center_j,
            displacement=displacement,
            distance=distance,
            cutoff_i=float(self.store.species[si]["orbital_cutoff_bohr"]),
            cutoff_j=float(self.store.species[sj]["orbital_cutoff_bohr"]),
            ni=int(self.store.species[si]["orbital_norb"]),
            nj=int(self.store.species[sj]["orbital_norb"]),
            is_onsite=atom_i == atom_j and bool(np.all(shift == 0)),
        )

    @staticmethod
    def _zero_block(geometry: _BlockGeometry) -> np.ndarray:
        return np.zeros((geometry.ni, geometry.nj), dtype=np.float64)

    def _base_component_block(
        self, geometry: _BlockGeometry, component: str
    ) -> np.ndarray:
        if geometry.is_onsite:
            return self.store.onsite_component(geometry.si, component).copy()
        return self.store.base_component(
            geometry.si, geometry.sj, component
        ).evaluate(geometry.displacement)

    def _projector_overlap(
        self,
        projector_species: str,
        orbital_species: str,
        displacement_bohr: np.ndarray,
    ) -> np.ndarray:
        displacement = np.asarray(displacement_bohr, dtype=np.float64)
        distance = float(np.linalg.norm(displacement))
        block = self.store.projector(projector_species, orbital_species).evaluate(
            displacement
        )
        meta = self.store.species[projector_species]
        orbital_cutoff = float(self.store.species[orbital_species]["orbital_cutoff_bohr"])
        cutoffs = [float(x) for x in meta["projector_cutoffs_bohr"]]
        shells = [int(x) for x in meta["projector_shells"]]
        if len(cutoffs) != len(shells):
            raise ValueError(f"projector cutoff metadata mismatch for {projector_species}")
        offset = 0
        for l, cutoff in zip(shells, cutoffs):
            width = 2 * l + 1
            if distance > cutoff + orbital_cutoff + 1.0e-12:
                block[offset : offset + width] = 0.0
            offset += width
        return block

    def _projector_overlaps_many(
        self,
        projector_species: str,
        orbital_species: str,
        displacements_bohr: np.ndarray,
    ) -> np.ndarray:
        vectors = np.asarray(displacements_bohr, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError(f"displacements must be [n,3], got {vectors.shape}")
        projector_norb = int(self.store.species[projector_species]["projector_norb"])
        orbital_norb = int(self.store.species[orbital_species]["orbital_norb"])
        if vectors.shape[0] == 0:
            return np.empty((0, projector_norb, orbital_norb), dtype=np.float64)
        blocks = np.asarray(
            self.store.projector(projector_species, orbital_species).evaluate_many(
                vectors
            ),
            dtype=np.float64,
        ).copy()
        if blocks.shape != (vectors.shape[0], projector_norb, orbital_norb):
            raise ValueError(
                f"projector overlap batch {projector_species}|{orbital_species} "
                f"shape {blocks.shape} != "
                f"{(vectors.shape[0], projector_norb, orbital_norb)}"
            )
        meta = self.store.species[projector_species]
        orbital_cutoff = float(self.store.species[orbital_species]["orbital_cutoff_bohr"])
        cutoffs = [float(x) for x in meta["projector_cutoffs_bohr"]]
        shells = [int(x) for x in meta["projector_shells"]]
        if len(cutoffs) != len(shells):
            raise ValueError(f"projector cutoff metadata mismatch for {projector_species}")
        distances = np.linalg.norm(vectors, axis=1)
        offset = 0
        for l, cutoff in zip(shells, cutoffs):
            width = 2 * l + 1
            inactive = distances > cutoff + orbital_cutoff + 1.0e-12
            if np.any(inactive):
                blocks[inactive, offset : offset + width] = 0.0
            offset += width
        if not np.isfinite(blocks).all():
            raise ValueError(
                f"projector overlap batch {projector_species}|{orbital_species} "
                "contains NaN or infinity"
            )
        return blocks

    @staticmethod
    def _canonical_block_key(raw_key: Sequence[int]) -> tuple[int, int, int, int, int]:
        raw = np.asarray(raw_key)
        if raw.shape != (5,) or not np.isfinite(raw).all():
            raise ValueError(f"block key must be (i,j,rx,ry,rz), got {raw_key!r}")
        integer = raw.astype(np.int64)
        if not np.array_equal(raw, integer):
            raise ValueError(f"block key must contain exact integers, got {raw_key!r}")
        return tuple(int(x) for x in integer)

    @staticmethod
    def _canonical_displacement_key(
        displacement_bohr: np.ndarray,
    ) -> tuple[bytes, np.ndarray]:
        value = np.ascontiguousarray(displacement_bohr, dtype=np.float64).copy()
        if value.shape != (3,):
            raise ValueError(f"displacement must have shape (3,), got {value.shape}")
        value[value == 0.0] = 0.0
        return canonical_float64_bytes(value), value

    @staticmethod
    def _merge_batch_stats(
        context: StructureLocalP2BatchContext,
        assembly: Mapping[str, Any],
    ) -> dict[str, Any]:
        stats = context.snapshot()
        stats["assembly"] = dict(assembly)
        return stats

    def _assemble_sparse_blocks(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        block_keys: Iterable[Sequence[int]],
        stats_mode: str,
    ) -> dict[tuple[int, int, int, int, int], np.ndarray]:
        self._last_batch_stats = None
        symbols = tuple(str(x) for x in symbols)
        context = StructureLocalP2BatchContext(
            cell_bohr,
            projector_overlap_cache_max_bytes=self.projector_overlap_cache_max_bytes,
        )
        assembly_stats: dict[str, Any] = {
            "mode": stats_mode,
            "status": "running",
            "requested_blocks": 0,
            "outside_support_blocks": 0,
            "base_evaluate_many_calls": 0,
            "base_evaluate_many_queries": 0,
            "projector_evaluate_many_calls": 0,
            "projector_evaluate_many_queries": 0,
            "projector_overlap_unique": 0,
            "vnl_contractions": 0,
            "uses_radial_evaluate_many": False,
        }
        try:
            keys = [self._canonical_block_key(key) for key in block_keys]
            assembly_stats["requested_blocks"] = len(keys)
            if len(set(keys)) != len(keys):
                raise ValueError("P2 sparse block keys contain duplicates")
            output: dict[tuple[int, int, int, int, int], np.ndarray] = {}
            base_buckets: dict[
                tuple[str, str],
                list[tuple[tuple[int, int, int, int, int], np.ndarray]],
            ] = {}
            overlap_displacements: dict[tuple[str, str, bytes], np.ndarray] = {}
            overlap_buckets: dict[
                tuple[str, str], list[tuple[tuple[str, str, bytes], np.ndarray]]
            ] = {}
            contractions: list[
                tuple[
                    tuple[int, int, int, int, int],
                    str,
                    tuple[str, str, bytes],
                    tuple[str, str, bytes],
                ]
            ] = []

            def register_overlap(
                projector_species: str,
                orbital_species: str,
                displacement: np.ndarray,
            ) -> tuple[str, str, bytes]:
                key_bytes, canonical = self._canonical_displacement_key(displacement)
                key = (str(projector_species), str(orbital_species), key_bytes)
                if key not in overlap_displacements:
                    overlap_displacements[key] = canonical
                    overlap_buckets.setdefault(
                        (str(projector_species), str(orbital_species)), []
                    ).append((key, canonical))
                return key

            for key in keys:
                i, j, rx, ry, rz = key
                geometry = self._block_geometry(
                    symbols=symbols,
                    positions_bohr=positions_bohr,
                    cell_bohr=cell_bohr,
                    i=i,
                    j=j,
                    translation=(rx, ry, rz),
                )
                output[key] = self._zero_block(geometry)
                if geometry.outside_support:
                    assembly_stats["outside_support_blocks"] += 1
                elif geometry.is_onsite:
                    output[key] += self.store.onsite_component(
                        geometry.si, "p2_base"
                    ).copy()
                else:
                    base_buckets.setdefault((geometry.si, geometry.sj), []).append(
                        (key, geometry.displacement.copy())
                    )
                midpoint = 0.5 * (geometry.center_i + geometry.center_j)
                for atom_index, projector_species in enumerate(geometry.symbols):
                    projector_meta = self.store.species[projector_species]
                    max_projector_cutoff = float(
                        projector_meta["projector_max_cutoff_bohr"]
                    )
                    if int(projector_meta["projector_norb"]) == 0:
                        continue
                    radius = (
                        max_projector_cutoff
                        + max(geometry.cutoff_i, geometry.cutoff_j)
                        + 0.5 * geometry.distance
                    )
                    for _, projector_center in context.iter_images(
                        geometry.cell,
                        geometry.positions[atom_index],
                        midpoint,
                        radius,
                    ):
                        di = geometry.center_i - projector_center
                        dj = geometry.center_j - projector_center
                        if (
                            np.linalg.norm(di)
                            > max_projector_cutoff + geometry.cutoff_i + 1.0e-12
                        ):
                            continue
                        if (
                            np.linalg.norm(dj)
                            > max_projector_cutoff + geometry.cutoff_j + 1.0e-12
                        ):
                            continue
                        qi_key = register_overlap(projector_species, geometry.si, di)
                        qj_key = register_overlap(projector_species, geometry.sj, dj)
                        contractions.append((key, projector_species, qi_key, qj_key))

            for (si, sj), entries in base_buckets.items():
                displacements = np.stack([displacement for _, displacement in entries])
                blocks = self.store.base_component(si, sj, "p2_base").evaluate_many(
                    displacements
                )
                assembly_stats["base_evaluate_many_calls"] += 1
                assembly_stats["base_evaluate_many_queries"] += int(len(entries))
                assembly_stats["uses_radial_evaluate_many"] = True
                for (key, _), block in zip(entries, blocks):
                    output[key] += np.asarray(block, dtype=np.float64)

            overlap_values: dict[tuple[str, str, bytes], np.ndarray] = {}
            for (projector_species, orbital_species), entries in overlap_buckets.items():
                displacements = np.stack([displacement for _, displacement in entries])
                blocks = self._projector_overlaps_many(
                    projector_species, orbital_species, displacements
                )
                assembly_stats["projector_evaluate_many_calls"] += 1
                assembly_stats["projector_evaluate_many_queries"] += int(len(entries))
                assembly_stats["uses_radial_evaluate_many"] = True
                for (key, _), block in zip(entries, blocks):
                    overlap_values[key] = np.asarray(block, dtype=np.float64)
            assembly_stats["projector_overlap_unique"] = len(overlap_values)
            assembly_stats["vnl_contractions"] = len(contractions)

            d_eff_cache: dict[str, np.ndarray] = {}
            for key, projector_species, qi_key, qj_key in contractions:
                d_eff = d_eff_cache.get(projector_species)
                if d_eff is None:
                    d_eff = self.store.d_eff(projector_species)
                    d_eff_cache[projector_species] = d_eff
                output[key] += overlap_values[qi_key].T @ d_eff @ overlap_values[qj_key]

            for key, block in output.items():
                if not np.isfinite(block).all():
                    raise ValueError(f"assembled P2 block {key} contains NaN or infinity")
            assembly_stats["status"] = "success"
            self._last_batch_stats = self._merge_batch_stats(context, assembly_stats)
            return output
        except Exception as exc:
            assembly_stats["status"] = "failed"
            assembly_stats["error"] = f"{type(exc).__name__}: {exc}"
            self._last_batch_stats = self._merge_batch_stats(context, assembly_stats)
            raise

    def _assemble_vnl(
        self,
        geometry: _BlockGeometry,
        *,
        context: StructureLocalP2BatchContext | None = None,
    ) -> np.ndarray:
        vnl = self._zero_block(geometry)
        midpoint = 0.5 * (geometry.center_i + geometry.center_j)
        for atom_index, projector_species in enumerate(geometry.symbols):
            projector_meta = self.store.species[projector_species]
            max_projector_cutoff = float(
                projector_meta["projector_max_cutoff_bohr"]
            )
            if int(projector_meta["projector_norb"]) == 0:
                continue
            radius = (
                max_projector_cutoff
                + max(geometry.cutoff_i, geometry.cutoff_j)
                + 0.5 * geometry.distance
            )
            image_iter = (
                nearby_atom_images(
                    geometry.cell,
                    geometry.positions[atom_index],
                    midpoint,
                    radius,
                )
                if context is None
                else context.iter_images(
                    geometry.cell,
                    geometry.positions[atom_index],
                    midpoint,
                    radius,
                )
            )
            for _, projector_center in image_iter:
                di = geometry.center_i - projector_center
                dj = geometry.center_j - projector_center
                if (
                    np.linalg.norm(di)
                    > max_projector_cutoff + geometry.cutoff_i + 1.0e-12
                ):
                    continue
                if (
                    np.linalg.norm(dj)
                    > max_projector_cutoff + geometry.cutoff_j + 1.0e-12
                ):
                    continue
                if context is None:
                    qi = self._projector_overlap(
                        projector_species, geometry.si, di
                    )
                    qj = self._projector_overlap(
                        projector_species, geometry.sj, dj
                    )
                else:
                    qi = context.projector_overlap(
                        projector_species,
                        geometry.si,
                        di,
                        lambda species=projector_species, orbital=geometry.si, disp=di: (
                            self._projector_overlap(species, orbital, disp)
                        ),
                    )
                    qj = context.projector_overlap(
                        projector_species,
                        geometry.sj,
                        dj,
                        lambda species=projector_species, orbital=geometry.sj, disp=dj: (
                            self._projector_overlap(species, orbital, disp)
                        ),
                    )
                vnl += qi.T @ self.store.d_eff(projector_species) @ qj
        if not np.isfinite(vnl).all():
            raise ValueError("assembled P2 Vnl block contains NaN or infinity")
        return vnl

    def _assemble_block_from_geometry(
        self,
        geometry: _BlockGeometry,
        *,
        context: StructureLocalP2BatchContext | None = None,
    ) -> np.ndarray:
        if geometry.outside_support:
            base = self._zero_block(geometry)
        else:
            base = self._base_component_block(geometry, "p2_base")
        vnl = self._assemble_vnl(geometry, context=context)
        output = np.asarray(base + vnl, dtype=np.float64)
        if not np.isfinite(output).all():
            raise ValueError("assembled P2 block contains NaN or infinity")
        return output

    def assemble_block(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        i: int,
        j: int,
        translation: Sequence[int] = (0, 0, 0),
    ) -> np.ndarray:
        geometry = self._block_geometry(
            symbols=symbols,
            positions_bohr=positions_bohr,
            cell_bohr=cell_bohr,
            i=i,
            j=j,
            translation=translation,
        )
        return self._assemble_block_from_geometry(geometry)

    def assemble_sparse_blocks(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        block_keys: Iterable[Sequence[int]],
    ) -> dict[tuple[int, int, int, int, int], np.ndarray]:
        """Assemble only requested ``(i,j,rx,ry,rz)`` blocks using batch lookup.

        The scalar ``assemble_block`` method remains the oracle.  This path
        gathers required blocks, buckets radial-table queries by shard, calls
        ``RadialBlockTable.evaluate_many`` for base and projector overlaps, then
        scatters the exact AO blocks back to their requested keys.
        """

        return self._assemble_sparse_blocks(
            symbols=symbols,
            positions_bohr=positions_bohr,
            cell_bohr=cell_bohr,
            block_keys=block_keys,
            stats_mode="sparse_required_blocks",
        )

    def assemble_overlap_block(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        i: int,
        j: int,
        translation: Sequence[int] = (0, 0, 0),
    ) -> np.ndarray:
        """Assemble the optional AO overlap table without assuming onsite I."""

        geometry = self._block_geometry(
            symbols=symbols,
            positions_bohr=positions_bohr,
            cell_bohr=cell_bohr,
            i=i,
            j=j,
            translation=translation,
        )
        # Resolve the declared component even outside support, so a table that
        # does not contain overlap cannot silently masquerade as a zero block.
        overlap = self._base_component_block(geometry, "overlap")
        if not np.isfinite(overlap).all():
            raise ValueError("assembled overlap block contains NaN or infinity")
        return np.asarray(overlap, dtype=np.float64)

    def assemble_components_block(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        i: int,
        j: int,
        translation: Sequence[int] = (0, 0, 0),
    ) -> dict[str, Any]:
        """Assemble and audit the fully decomposed optional P2 base contract."""

        required_base = {"p2_base", "overlap", "kinetic", "vna_left", "vna_right"}
        required_onsite = {"p2_base", "overlap", "kinetic", "vna_endpoint"}
        if not required_base.issubset(self.store.base_component_arrays) or not (
            required_onsite.issubset(self.store.onsite_component_arrays)
        ):
            raise KeyError(
                "assemble_components_block requires base component mode 'all'"
            )
        geometry = self._block_geometry(
            symbols=symbols,
            positions_bohr=positions_bohr,
            cell_bohr=cell_bohr,
            i=i,
            j=j,
            translation=translation,
        )
        overlap = self._base_component_block(geometry, "overlap")
        kinetic = self._base_component_block(geometry, "kinetic")
        p2_base = self._base_component_block(geometry, "p2_base")
        output: dict[str, Any] = {
            "overlap": np.asarray(overlap, dtype=np.float64),
            "kinetic": np.asarray(kinetic, dtype=np.float64),
            "p2_base": np.asarray(p2_base, dtype=np.float64),
        }
        if geometry.is_onsite:
            vna_pair = self._base_component_block(geometry, "vna_endpoint")
        else:
            vna_left = self._base_component_block(geometry, "vna_left")
            vna_right = self._base_component_block(geometry, "vna_right")
            output["vna_left"] = np.asarray(vna_left, dtype=np.float64)
            output["vna_right"] = np.asarray(vna_right, dtype=np.float64)
            vna_pair = np.asarray(vna_left + vna_right, dtype=np.float64)
        output["vna_endpoint_pair"] = np.asarray(vna_pair, dtype=np.float64)
        vnl = self._assemble_vnl(geometry)
        output["vnl"] = vnl
        reconstructed_base = np.asarray(kinetic + vna_pair, dtype=np.float64)
        reconstruction_error = float(
            np.max(np.abs(reconstructed_base - p2_base), initial=0.0)
        )
        scale = max(1.0, float(np.max(np.abs(p2_base), initial=0.0)))
        tolerance = 5.0e-6 * scale
        if reconstruction_error > tolerance:
            raise ValueError(
                "P2 component reconstruction disagrees with authoritative base: "
                f"error={reconstruction_error:.6e} Ry, "
                f"tolerance={tolerance:.6e} Ry"
            )
        output["p2"] = np.asarray(p2_base + vnl, dtype=np.float64)
        output["component_reconstruction_max_abs_ry"] = reconstruction_error
        output["component_reconstruction_tolerance_ry"] = tolerance
        if not all(
            np.isfinite(value).all()
            for value in output.values()
            if isinstance(value, np.ndarray)
        ):
            raise ValueError("assembled P2 components contain NaN or infinity")
        return output

    def assemble_dense_rkeys(
        self,
        *,
        symbols: Sequence[str],
        positions_bohr: np.ndarray,
        cell_bohr: np.ndarray,
        r_keys: np.ndarray,
    ) -> np.ndarray:
        symbols = tuple(str(x) for x in symbols)
        r_keys = np.asarray(r_keys)
        if r_keys.ndim != 2 or r_keys.shape[1] != 3:
            raise ValueError(f"r_keys must be [nR,3], got {r_keys.shape}")
        integer_keys = r_keys.astype(np.int64)
        if not np.array_equal(r_keys, integer_keys):
            raise ValueError("r_keys must contain exact integers")
        norb = [int(self.store.species[s]["orbital_norb"]) for s in symbols]
        offsets = np.concatenate(([0], np.cumsum(norb))).astype(np.int64)
        output = np.zeros(
            (len(integer_keys), int(offsets[-1]), int(offsets[-1])), dtype=np.float64
        )
        block_keys: list[tuple[int, int, int, int, int]] = []
        for r_index, r_key in enumerate(integer_keys):
            rx, ry, rz = (int(value) for value in r_key)
            for i in range(len(symbols)):
                for j in range(len(symbols)):
                    block_keys.append((i, j, rx, ry, rz))
        unique_block_keys = list(dict.fromkeys(block_keys))
        blocks = self._assemble_sparse_blocks(
            symbols=symbols,
            positions_bohr=positions_bohr,
            cell_bohr=cell_bohr,
            block_keys=unique_block_keys,
            stats_mode="dense_rkeys",
        )
        for r_index, r_key in enumerate(integer_keys):
            rx, ry, rz = (int(value) for value in r_key)
            for i in range(len(symbols)):
                rows = slice(int(offsets[i]), int(offsets[i + 1]))
                for j in range(len(symbols)):
                    cols = slice(int(offsets[j]), int(offsets[j + 1]))
                    output[r_index, rows, cols] = blocks[(i, j, rx, ry, rz)]
        return output

    def batch_stats_snapshot(self) -> dict[str, Any] | None:
        """Return stats from the latest dense assembly without cached geometry."""

        if self._last_batch_stats is None:
            return None
        return json.loads(json.dumps(self._last_batch_stats))


__all__ = [
    "P2_BASE_COMPONENT_MODES",
    "P2_BASE_COMPONENT_SCHEMA",
    "P2_COMPONENT_NUMERICAL_GATE_SCHEMA",
    "P2_SPECIES_SOURCE_IDENTITY_SCHEMA",
    "P2_TABLE_SCHEMA",
    "P2TableAssembler",
    "P2TableStore",
    "RadialBlockTable",
    "RealHarmonicRotator",
    "abacus_m_order",
    "canonical_species_source_identity",
    "nearby_atom_images",
    "normalized_p2_base_component_contract",
    "p2_base_component_contract",
    "real_sph_abacus",
    "rotation_z_to",
    "species_source_identity_sha256",
    "validate_p2_component_numerical_contract",
]
