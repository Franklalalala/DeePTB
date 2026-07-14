"""Exact local acceleration helpers for non-SOC P2 table lookup.

The helpers in this module are deliberately small and NumPy-only.  They do not
change the P2 data contract: distances, directions, image translations, and
projector-overlap keys are exact with respect to the scalar implementation.

The mutable LRU/statistics objects are intended to be owned by one data-loader
worker or one assembly call.  They are not thread-safe shared process state.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import itertools
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    skipped: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.hits / self.requests

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requests"] = self.requests
        result["hit_rate"] = self.hit_rate
        return result


def canonical_float64_bytes(vector: np.ndarray) -> bytes:
    """Return an exact key, folding only the sign bit of zero."""

    value = np.ascontiguousarray(vector, dtype=np.float64).copy()
    value[value == 0.0] = 0.0
    return value.tobytes()


def shell_groups(shells: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    groups: list[tuple[int, np.ndarray]] = []
    offset = 0
    for raw_l in shells:
        l_value = int(raw_l)
        width = 2 * l_value + 1
        groups.append((l_value, np.arange(offset, offset + width, dtype=np.int64)))
        offset += width
    return groups


class ExactDirectionRotationCache:
    """Bounded LRU of real-harmonic rotation matrices by exact direction."""

    def __init__(
        self,
        rotator: Any,
        rotation_z_to: Callable[[np.ndarray], np.ndarray],
        *,
        max_entries: int = 4096,
    ) -> None:
        self.rotator = rotator
        self.rotation_z_to = rotation_z_to
        self.max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[tuple[int, bytes], np.ndarray] = OrderedDict()
        self.stats = CacheStats()

    def clear(self, *, reset_stats: bool = True) -> None:
        self._cache.clear()
        if reset_stats:
            self.stats = CacheStats()

    def matrix(
        self,
        l_value: int,
        displacement: np.ndarray,
        *,
        distance: float | None = None,
    ) -> np.ndarray:
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(f"displacement must have shape (3,), got {vector.shape}")
        radius = float(np.linalg.norm(vector)) if distance is None else float(distance)
        if radius <= 0.0:
            raise ValueError("rotation direction must be non-zero")
        direction = vector / radius
        key = (int(l_value), canonical_float64_bytes(direction))
        cached = self._cache.get(key)
        if cached is not None:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return cached

        self.stats.misses += 1
        rotation = self.rotation_z_to(direction)
        matrix = np.asarray(
            self.rotator.matrix(int(l_value), rotation), dtype=np.float64
        )
        matrix.setflags(write=False)
        self._cache[key] = matrix
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self.stats.evictions += 1
        return matrix

    def snapshot(self) -> dict[str, Any]:
        result = self.stats.to_dict()
        result.update(
            {
                "entries": len(self._cache),
                "capacity": self.max_entries,
                "key_semantics": "l + exact normalized float64 direction bytes",
            }
        )
        return result


def interpolate_many(table: Any, distances: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``RadialBlockTable._interpolate``."""

    query = np.asarray(distances, dtype=np.float64)
    if query.ndim != 1:
        raise ValueError(f"distances must be one-dimensional, got {query.shape}")
    if not np.isfinite(query).all():
        raise ValueError("distances must contain only finite values")
    if np.any(query < -1.0e-12):
        raise ValueError("distance cannot be negative")

    shape = (query.size,) + tuple(table.values.shape[1:])
    output = np.zeros(shape, dtype=np.float64)
    active = query < float(table.support_bohr) - 1.0e-12
    if not np.any(active):
        return output

    active_query = np.maximum(query[active], 0.0)
    if table._spline is not None:
        output[active] = np.asarray(table._spline(active_query), dtype=np.float64)
        return output

    knots = np.asarray(table.distances, dtype=np.float64)
    upper = np.searchsorted(knots, active_query, side="right")
    upper = np.clip(upper, 1, knots.size - 1)
    lower = upper - 1
    d0 = knots[lower]
    d1 = knots[upper]
    weights = np.divide(
        active_query - d0,
        d1 - d0,
        out=np.zeros_like(active_query),
        where=d1 != d0,
    )
    values = np.asarray(table.values, dtype=np.float64)
    broadcast = (slice(None),) + (None,) * (values.ndim - 1)
    w = weights[broadcast]
    output[active] = (1.0 - w) * values[lower] + w * values[upper]
    return output


def evaluate_many(
    table: Any,
    displacements: np.ndarray,
    *,
    rotation_cache: ExactDirectionRotationCache,
) -> np.ndarray:
    """Batch-evaluate one radial table with scalar-equivalent semantics."""

    vectors = np.asarray(displacements, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"displacements must be [n,3], got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("displacements must contain only finite values")
    distances = np.linalg.norm(vectors, axis=1)
    canonical = interpolate_many(table, distances)
    if vectors.shape[0] == 0:
        return canonical
    output = canonical.copy()
    has_content = np.any(canonical.reshape(canonical.shape[0], -1), axis=1)
    active_indices = np.flatnonzero((distances > 1.0e-14) & has_content)
    if active_indices.size == 0:
        return output

    active_vectors = vectors[active_indices]
    active_distances = distances[active_indices]
    unique_l = sorted(set((*table.left_shells, *table.right_shells)))
    rotations: dict[int, np.ndarray] = {}
    for l_value in unique_l:
        rotations[l_value] = np.stack(
            [
                rotation_cache.matrix(l_value, vector, distance=float(radius))
                for vector, radius in zip(active_vectors, active_distances)
            ],
            axis=0,
        )

    active_canonical = canonical[active_indices]
    active_output = np.empty_like(active_canonical)
    for l_left, left in shell_groups(table.left_shells):
        left_rotation = rotations[l_left]
        for l_right, right in shell_groups(table.right_shells):
            right_rotation_t = np.swapaxes(rotations[l_right], 1, 2)
            block = active_canonical[:, left][:, :, right]
            active_output[:, left[:, None], right] = np.matmul(
                np.matmul(left_rotation, block), right_rotation_t
            )
    output[active_indices] = active_output
    return output


class VectorizedNearbyImageEnumerator:
    """Structure-local equivalent of ``nearby_atom_images`` with one inverse."""

    def __init__(self, cell_bohr: np.ndarray):
        cell = np.asarray(cell_bohr, dtype=np.float64)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must have shape (3,3), got {cell.shape}")
        self.cell = cell
        self.inverse = np.linalg.inv(cell)
        self._inverse_axis_norm = np.linalg.norm(self.inverse, axis=0)
        self._offsets: dict[tuple[int, int, int], np.ndarray] = {}
        self.offset_cache_stats = CacheStats()
        self.max_offset_rows = 250_000
        self.max_offset_bytes = 6 * 1024 * 1024

    def clear_offset_cache(self, *, reset_stats: bool = True) -> None:
        self._offsets.clear()
        if reset_stats:
            self.offset_cache_stats = CacheStats()

    def _bounds(self, radius: float) -> tuple[int, int, int]:
        values = np.ceil(float(radius) * self._inverse_axis_norm).astype(int) + 1
        return tuple(int(x) for x in values)

    def _offset_grid(self, bounds: tuple[int, int, int]) -> np.ndarray | None:
        cached = self._offsets.get(bounds)
        if cached is not None:
            self.offset_cache_stats.hits += 1
            return cached
        row_count = 1
        for bound in bounds:
            row_count *= 2 * int(bound) + 1
        byte_count = row_count * 3 * np.dtype(np.int64).itemsize
        if row_count > self.max_offset_rows or byte_count > self.max_offset_bytes:
            self.offset_cache_stats.skipped += 1
            return None
        self.offset_cache_stats.misses += 1
        grid = np.asarray(
            list(
                itertools.product(
                    *(range(-bound, bound + 1) for bound in bounds)
                )
            ),
            dtype=np.int64,
        )
        grid.setflags(write=False)
        self._offsets[bounds] = grid
        return grid

    def _query_arrays_chunked(
        self,
        bounds: tuple[int, int, int],
        guess: np.ndarray,
        base: np.ndarray,
        target_value: np.ndarray,
        radius_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        chunk_rows = max(
            1,
            min(
                self.max_offset_rows,
                self.max_offset_bytes // (3 * np.dtype(np.int64).itemsize),
            ),
        )
        translation_chunks: list[np.ndarray] = []
        center_chunks: list[np.ndarray] = []
        pending: list[tuple[int, int, int]] = []
        ranges = tuple(range(-bound, bound + 1) for bound in bounds)

        def flush() -> None:
            if not pending:
                return
            offsets = np.asarray(pending, dtype=np.int64)
            pending.clear()
            translations = offsets + guess[None, :]
            centers = base[None, :] + translations @ self.cell
            distances = np.linalg.norm(centers - target_value[None, :], axis=1)
            active = distances <= radius_value + 1.0e-12
            if np.any(active):
                translation_chunks.append(translations[active])
                center_chunks.append(centers[active])

        for offset in itertools.product(*ranges):
            pending.append(offset)
            if len(pending) >= chunk_rows:
                flush()
        flush()
        if not translation_chunks:
            return (
                np.empty((0, 3), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
            )
        return np.concatenate(translation_chunks, axis=0), np.concatenate(
            center_chunks, axis=0
        )

    def query_arrays(
        self,
        base_center: np.ndarray,
        target: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        base = np.asarray(base_center, dtype=np.float64)
        target_value = np.asarray(target, dtype=np.float64)
        if base.shape != (3,) or target_value.shape != (3,):
            raise ValueError("base_center and target must each have shape (3,)")
        radius_value = float(radius)
        if radius_value < 0.0:
            raise ValueError("radius cannot be negative")
        guess = np.rint((target_value - base) @ self.inverse).astype(np.int64)
        bounds = self._bounds(radius_value)
        offsets = self._offset_grid(bounds)
        if offsets is None:
            return self._query_arrays_chunked(
                bounds,
                guess,
                base,
                target_value,
                radius_value,
            )
        translations = offsets + guess[None, :]
        centers = base[None, :] + translations @ self.cell
        distances = np.linalg.norm(centers - target_value[None, :], axis=1)
        active = distances <= radius_value + 1.0e-12
        return translations[active], centers[active]

    def iter_images(
        self,
        cell_bohr: np.ndarray,
        base_center: np.ndarray,
        target: np.ndarray,
        radius: float,
    ) -> Iterable[tuple[tuple[int, int, int], np.ndarray]]:
        supplied_cell = np.asarray(cell_bohr, dtype=np.float64)
        if supplied_cell.shape != (3, 3) or not np.array_equal(
            supplied_cell, self.cell
        ):
            raise ValueError("enumerator is bound to a different cell")
        translations, centers = self.query_arrays(base_center, target, radius)
        for translation, center in zip(translations, centers):
            yield tuple(int(x) for x in translation), center

    def snapshot(self) -> dict[str, Any]:
        result = self.offset_cache_stats.to_dict()
        result.update(
            {
                "cached_bound_grids": len(self._offsets),
                "cached_offset_rows": int(
                    sum(grid.shape[0] for grid in self._offsets.values())
                ),
                "max_offset_rows": int(self.max_offset_rows),
                "max_offset_bytes": int(self.max_offset_bytes),
                "cell_inverse_computations": 1,
            }
        )
        return result


class ExactProjectorOverlapCache:
    """Byte-budgeted structure-local LRU for projector overlap blocks."""

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._cache: OrderedDict[tuple[str, str, bytes], tuple[np.ndarray, int]] = (
            OrderedDict()
        )
        self.stats = CacheStats()

    def clear(self, *, reset_stats: bool = True) -> None:
        self._cache.clear()
        self.current_bytes = 0
        if reset_stats:
            self.stats = CacheStats()

    def get_or_compute(
        self,
        projector_species: str,
        orbital_species: str,
        displacement_bohr: np.ndarray,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        displacement = np.asarray(displacement_bohr, dtype=np.float64)
        if displacement.shape != (3,):
            raise ValueError(f"displacement must have shape (3,), got {displacement.shape}")
        key = (
            str(projector_species),
            str(orbital_species),
            canonical_float64_bytes(displacement),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return cached[0]

        self.stats.misses += 1
        value = np.asarray(compute(), dtype=np.float64)
        value.setflags(write=False)
        size = int(value.nbytes)
        if size > self.max_bytes:
            self.stats.skipped += 1
            return value
        self._cache[key] = (value, size)
        self.current_bytes += size
        self._cache.move_to_end(key)
        while self.current_bytes > self.max_bytes and self._cache:
            _, (_, old_size) = self._cache.popitem(last=False)
            self.current_bytes -= old_size
            self.stats.evictions += 1
        return value

    def snapshot(self) -> dict[str, Any]:
        result = self.stats.to_dict()
        result.update(
            {
                "entries": len(self._cache),
                "current_bytes": int(self.current_bytes),
                "capacity_bytes": int(self.max_bytes),
                "key_semantics": (
                    "projector species + orbital species + exact float64 "
                    "displacement bytes"
                ),
            }
        )
        return result


class StructureLocalP2BatchContext:
    """Per-``assemble_dense_rkeys`` context; stores no cross-structure geometry."""

    def __init__(
        self,
        cell_bohr: np.ndarray,
        *,
        projector_overlap_cache_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.images = VectorizedNearbyImageEnumerator(cell_bohr)
        self.projector_overlaps = ExactProjectorOverlapCache(
            max_bytes=projector_overlap_cache_max_bytes
        )

    def iter_images(
        self,
        cell_bohr: np.ndarray,
        base_center: np.ndarray,
        target: np.ndarray,
        radius: float,
    ) -> Iterable[tuple[tuple[int, int, int], np.ndarray]]:
        return self.images.iter_images(cell_bohr, base_center, target, radius)

    def projector_overlap(
        self,
        projector_species: str,
        orbital_species: str,
        displacement_bohr: np.ndarray,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        return self.projector_overlaps.get_or_compute(
            projector_species,
            orbital_species,
            displacement_bohr,
            compute,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "deeptb.p2_batch_context_stats/v1",
            "projector_overlap_cache": self.projector_overlaps.snapshot(),
            "nearby_image_enumerator": self.images.snapshot(),
            "lifecycle": "one assemble_dense_rkeys call",
        }


def active_table_working_set(
    active_species: Iterable[str],
    species_metadata: Mapping[str, Mapping[str, Any]] | Any,
    *,
    configured_capacity: int = 64,
    base_component_arrays: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Count table shards required by one structure's active species set."""

    species = tuple(sorted(set(str(x) for x in active_species)))
    if base_component_arrays is None:
        base_component_array_count = 1
    elif isinstance(base_component_arrays, Mapping):
        base_component_array_count = len(set(str(x) for x in base_component_arrays.values()))
    else:
        base_component_array_count = len(set(str(x) for x in base_component_arrays))
    base_component_array_count = max(1, int(base_component_array_count))
    projector_species = tuple(
        symbol
        for symbol in species
        if int(species_metadata[symbol].get("projector_norb", 0)) > 0
    )
    base = len(species) ** 2 * base_component_array_count
    projector = len(projector_species) * len(species)
    required = base + projector
    capacity = int(configured_capacity)
    return {
        "active_species": list(species),
        "projector_species": list(projector_species),
        "base_component_array_count": base_component_array_count,
        "base_table_count": base,
        "projector_table_count": projector,
        "total_table_count": required,
        "configured_capacity": capacity,
        "fits_without_eviction": required <= capacity,
        "minimum_capacity": required,
    }


def simulate_lru(requests: Iterable[Any], capacity: int) -> dict[str, Any]:
    """Simulate an LRU cache for a supplied request trace."""

    limit = max(1, int(capacity))
    cache: OrderedDict[Any, None] = OrderedDict()
    stats = CacheStats()
    unique: set[Any] = set()
    for key in requests:
        unique.add(key)
        if key in cache:
            stats.hits += 1
            cache.move_to_end(key)
            continue
        stats.misses += 1
        cache[key] = None
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)
            stats.evictions += 1
    result = stats.to_dict()
    result.update(
        {
            "capacity": limit,
            "unique_keys": len(unique),
            "resident_keys": len(cache),
        }
    )
    return result


def dense_pair_request_trace(
    symbols: Sequence[str],
    species_metadata: Mapping[str, Mapping[str, Any]] | Any,
    *,
    base_component_arrays: Mapping[str, str] | Sequence[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Build a conservative table-request trace for one dense-R assembly."""

    atoms = tuple(str(x) for x in symbols)
    if base_component_arrays is None:
        base_arrays = ("values",)
    elif isinstance(base_component_arrays, Mapping):
        base_arrays = tuple(sorted(set(str(x) for x in base_component_arrays.values())))
    else:
        base_arrays = tuple(sorted(set(str(x) for x in base_component_arrays)))
    requests: list[tuple[str, str, str]] = []
    for left in atoms:
        for right in atoms:
            for array_name in base_arrays:
                requests.append((f"base:{array_name}", left, right))
            for projector in atoms:
                if int(species_metadata[projector].get("projector_norb", 0)) <= 0:
                    continue
                requests.append(("projector", projector, left))
                requests.append(("projector", projector, right))
    return requests


__all__ = [
    "CacheStats",
    "ExactDirectionRotationCache",
    "ExactProjectorOverlapCache",
    "StructureLocalP2BatchContext",
    "VectorizedNearbyImageEnumerator",
    "active_table_working_set",
    "canonical_float64_bytes",
    "dense_pair_request_trace",
    "evaluate_many",
    "interpolate_many",
    "shell_groups",
    "simulate_lru",
]
