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


P2_TABLE_SCHEMA = "deeptb.p2_radial_table/v1"


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
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.verify_checksums = bool(verify_checksums)
        self._cache: OrderedDict[str, RadialBlockTable] = OrderedDict()
        self._species_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._verified_paths: set[Path] = set()
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
        self, category: str, key: str, metadata: Mapping[str, Any]
    ) -> RadialBlockTable:
        cache_key = f"{category}:{key}"
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
            table = RadialBlockTable(
                distances=np.asarray(payload["distances"]),
                values=np.asarray(payload["values"]),
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

    def base(self, left: str, right: str) -> RadialBlockTable:
        key = _table_key(left, right)
        if key not in self.base_tables:
            raise KeyError(f"P2 base table missing for ordered pair {key}")
        return self._load_table("base", key, self.base_tables[key])

    def projector(self, projector_species: str, orbital_species: str) -> RadialBlockTable:
        key = _table_key(projector_species, orbital_species)
        if key not in self.projector_tables:
            raise KeyError(f"P2 projector table missing for ordered pair {key}")
        return self._load_table("projector", key, self.projector_tables[key])

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
        arrays = self.species_arrays(symbol)
        onsite = np.asarray(arrays["onsite_base"], dtype=np.float64)
        expected = int(self.species[symbol]["orbital_norb"])
        if onsite.shape != (expected, expected):
            raise ValueError(f"onsite {symbol} shape {onsite.shape} != {(expected, expected)}")
        return onsite

    def d_eff(self, symbol: str) -> np.ndarray:
        arrays = self.species_arrays(symbol)
        value = np.asarray(arrays["d_eff"], dtype=np.float64)
        expected = int(self.species[symbol]["projector_norb"])
        if value.shape != (expected, expected):
            raise ValueError(f"D_eff {symbol} shape {value.shape} != {(expected, expected)}")
        return value


class P2TableAssembler:
    """Assemble non-SOC P2 AO blocks without numerical integration."""

    def __init__(self, store: P2TableStore):
        self.store = store

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
        symbols = tuple(str(x) for x in symbols)
        positions = np.asarray(positions_bohr, dtype=np.float64)
        cell = np.asarray(cell_bohr, dtype=np.float64)
        if positions.shape != (len(symbols), 3):
            raise ValueError("positions must have shape [n_atoms,3]")
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape [3,3]")
        shift = np.asarray(translation, dtype=np.int64)
        if shift.shape != (3,):
            raise ValueError("translation must have three integers")
        si, sj = symbols[int(i)], symbols[int(j)]
        center_i = positions[int(i)]
        center_j = positions[int(j)] + shift @ cell
        displacement = center_j - center_i
        distance = float(np.linalg.norm(displacement))
        cutoff_i = float(self.store.species[si]["orbital_cutoff_bohr"])
        cutoff_j = float(self.store.species[sj]["orbital_cutoff_bohr"])
        ni = int(self.store.species[si]["orbital_norb"])
        nj = int(self.store.species[sj]["orbital_norb"])
        if distance > cutoff_i + cutoff_j + 1.0e-10:
            return np.zeros((ni, nj), dtype=np.float64)

        is_onsite = int(i) == int(j) and bool(np.all(shift == 0))
        base = (
            self.store.onsite(si).copy()
            if is_onsite
            else self.store.base(si, sj).evaluate(displacement)
        )
        vnl = np.zeros((ni, nj), dtype=np.float64)
        midpoint = 0.5 * (center_i + center_j)
        for atom_index, projector_species in enumerate(symbols):
            projector_meta = self.store.species[projector_species]
            max_projector_cutoff = float(projector_meta["projector_max_cutoff_bohr"])
            if int(projector_meta["projector_norb"]) == 0:
                continue
            radius = (
                max_projector_cutoff
                + max(cutoff_i, cutoff_j)
                + 0.5 * distance
            )
            for _, projector_center in nearby_atom_images(
                cell, positions[atom_index], midpoint, radius
            ):
                di = center_i - projector_center
                dj = center_j - projector_center
                if np.linalg.norm(di) > max_projector_cutoff + cutoff_i + 1.0e-12:
                    continue
                if np.linalg.norm(dj) > max_projector_cutoff + cutoff_j + 1.0e-12:
                    continue
                qi = self._projector_overlap(projector_species, si, di)
                qj = self._projector_overlap(projector_species, sj, dj)
                vnl += qi.T @ self.store.d_eff(projector_species) @ qj
        output = np.asarray(base + vnl, dtype=np.float64)
        if not np.isfinite(output).all():
            raise ValueError("assembled P2 block contains NaN or infinity")
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
        for r_index, r_key in enumerate(integer_keys):
            for i in range(len(symbols)):
                rows = slice(int(offsets[i]), int(offsets[i + 1]))
                for j in range(len(symbols)):
                    cols = slice(int(offsets[j]), int(offsets[j + 1]))
                    output[r_index, rows, cols] = self.assemble_block(
                        symbols=symbols,
                        positions_bohr=positions_bohr,
                        cell_bohr=cell_bohr,
                        i=i,
                        j=j,
                        translation=r_key,
                    )
        return output


__all__ = [
    "P2_TABLE_SCHEMA",
    "P2TableAssembler",
    "P2TableStore",
    "RadialBlockTable",
    "RealHarmonicRotator",
    "abacus_m_order",
    "nearby_atom_images",
    "real_sph_abacus",
    "rotation_z_to",
]
