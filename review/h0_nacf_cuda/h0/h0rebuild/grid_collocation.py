from __future__ import annotations

"""Exact sparse collocation of finite-range numerical orbitals on an FFT grid.

The local part of an LCAO Hamiltonian has the form

    H_ij^loc = sum_g phi_i(r_g) V(r_g) phi_j(r_g) * Omega / N_grid.

For a fixed geometry and FFT grid, the expensive orbital values depend only on
one orbital centre at a time, not on the pair or on the potential.  This module
therefore caches the non-zero support of every centred AO set and intersects two
supports by their *unwrapped* integer FFT indices.  The potential is periodic
and is indexed only after reduction modulo the FFT shape.

The construction is algebraically identical to direct enumeration of the pair
support intersection.  It changes data reuse, not the physical approximation or
the grid.  This is the sparse-grid/collocation pattern used by localized-orbital
real-space implementations; no external project source is copied here.
"""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .radial import OrbitalEvaluator
from .reciprocal import PeriodicField


def _cartesian_box_index_bounds(
    field: PeriodicField,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an integer FFT-index box enclosing a Cartesian box.

    The eight Cartesian corners are transformed because simply transforming the
    lower/upper vectors is incorrect for an oblique cell.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    cell = np.asarray(field.cell_bohr, dtype=float)
    inv_cell = np.linalg.inv(cell)
    shape = np.asarray(field.shape, dtype=np.int64)
    corners = np.asarray(
        [[x, y, z] for x, y, z in product(*zip(lower, upper))],
        dtype=float,
    )
    scaled = (corners @ inv_cell) * shape
    nmin = np.floor(np.min(scaled, axis=0)).astype(np.int64) - 1
    nmax = np.ceil(np.max(scaled, axis=0)).astype(np.int64) + 1
    return nmin, nmax


def iter_fft_sphere_nodes(
    field: PeriodicField,
    center_bohr: np.ndarray,
    radius_bohr: float,
    *,
    chunk_size: int = 100_000,
):
    """Yield ``(points, integer_indices)`` inside one unwrapped support sphere."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    center = np.asarray(center_bohr, dtype=float)
    radius = float(radius_bohr)
    if center.shape != (3,):
        raise ValueError(f"center_bohr must have shape (3,), got {center.shape}")
    if radius < 0.0:
        raise ValueError("radius_bohr must be non-negative")

    nmin, nmax = _cartesian_box_index_bounds(field, center - radius, center + radius)
    counts = nmax - nmin + 1
    if np.any(counts <= 0):
        return
    total = int(np.prod(counts, dtype=np.int64))
    yz = int(counts[1] * counts[2])
    shape = np.asarray(field.shape, dtype=np.int64)
    cell = np.asarray(field.cell_bohr, dtype=float)

    for start in range(0, total, chunk_size):
        linear = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        ix = linear // yz
        rem = linear % yz
        iy = rem // int(counts[2])
        iz = rem % int(counts[2])
        integer_index = np.column_stack((ix, iy, iz)) + nmin
        points = (integer_index / shape) @ cell
        delta = points - center
        mask = np.einsum("ij,ij->i", delta, delta) <= radius**2 + 1.0e-12
        if np.any(mask):
            yield points[mask], integer_index[mask]


def _row_keys(indices: np.ndarray) -> np.ndarray:
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must have shape (n,3), got {indices.shape}")
    return indices.view(np.dtype((np.void, indices.dtype.itemsize * 3))).reshape(-1)


@dataclass(frozen=True)
class AOSupport:
    """Non-zero values of one atom-centred AO set on unwrapped FFT nodes."""

    integer_indices: np.ndarray
    values: np.ndarray
    center_bohr: np.ndarray
    radius_bohr: float

    @property
    def npoints(self) -> int:
        return int(self.integer_indices.shape[0])

    @property
    def norb(self) -> int:
        return int(self.values.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.integer_indices.nbytes + self.values.nbytes + self.center_bohr.nbytes)


@dataclass(frozen=True)
class PairCollocation:
    """AO values on the exact common FFT nodes of a localized orbital pair."""

    integer_indices: np.ndarray
    wrapped_indices: np.ndarray
    left_values: np.ndarray
    right_values: np.ndarray
    grid_weight: float
    field_shape: tuple[int, int, int]

    @property
    def npoints(self) -> int:
        return int(self.integer_indices.shape[0])

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return int(self.left_values.shape[1]), int(self.right_values.shape[1])

    def sample(self, potential_values: np.ndarray) -> np.ndarray:
        values = np.asarray(potential_values)
        if values.shape == self.field_shape:
            w = self.wrapped_indices
            return values[w[:, 0], w[:, 1], w[:, 2]]
        if values.ndim == 1 and values.shape[0] == self.npoints:
            return values
        raise ValueError(
            "potential_values must be a full field with shape "
            f"{self.field_shape} or sampled values with shape ({self.npoints},), "
            f"got {values.shape}"
        )

    def contract(self, potential_values: np.ndarray) -> np.ndarray:
        """Return ``<left|V|right>`` using the exact collocation nodes."""
        ni, nj = self.matrix_shape
        if self.npoints == 0:
            dtype = np.result_type(self.left_values, self.right_values, potential_values, float)
            return np.zeros((ni, nj), dtype=dtype)
        sampled = self.sample(potential_values)
        return self.left_values.T @ (sampled[:, None] * self.right_values) * self.grid_weight

    def contract_many(self, potential_values: np.ndarray) -> np.ndarray:
        """Contract many potentials, returning ``[nfield, ni, nj]``.

        ``potential_values`` may be ``[nfield, *field_shape]`` or already sampled
        as ``[nfield, npoints]``.
        """
        fields = np.asarray(potential_values)
        if fields.ndim == 4 and tuple(fields.shape[1:]) == self.field_shape:
            w = self.wrapped_indices
            sampled = fields[:, w[:, 0], w[:, 1], w[:, 2]]
        elif fields.ndim == 2 and fields.shape[1] == self.npoints:
            sampled = fields
        else:
            raise ValueError(
                "potential_values must have shape [nfield,*field_shape] or "
                f"[nfield,{self.npoints}], got {fields.shape}"
            )
        ni, nj = self.matrix_shape
        if self.npoints == 0:
            dtype = np.result_type(self.left_values, self.right_values, fields, float)
            return np.zeros((fields.shape[0], ni, nj), dtype=dtype)
        return (
            np.einsum(
                "gi,fg,gj->fij",
                self.left_values,
                sampled,
                self.right_values,
                optimize=True,
            )
            * self.grid_weight
        )


class FFTGridAOCache:
    """LRU cache for exact AO collocation supports on one periodic FFT grid.

    Parameters
    ----------
    field
        The periodic field defining the cell, grid shape, and quadrature weight.
    max_bytes
        Maximum resident cache size. ``None`` means unbounded; ``0`` disables
        residency while retaining the same exact algorithm.
    key_decimals
        Optional decimal rounding used to identify centres supplied through
        numerically noisy external paths.  ``None`` (the default) keys centres by
        their exact float64 bytes and is required for strict reproduction.  A
        numeric value is an explicit cache-identity approximation: it can merge
        centres that are closer than the chosen decimal resolution.
    """

    def __init__(
        self,
        field: PeriodicField,
        *,
        chunk_size: int = 100_000,
        max_bytes: int | None = 512 * 1024**2,
        key_decimals: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative or None")
        self.field = field
        self.chunk_size = int(chunk_size)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.key_decimals = None if key_decimals is None else int(key_decimals)
        self._supports: OrderedDict[tuple[Any, ...], AOSupport] = OrderedDict()
        self._cached_bytes = 0
        self._stats: dict[str, int] = {
            "support_requests": 0,
            "support_hits": 0,
            "support_misses": 0,
            "support_builds": 0,
            "evaluated_grid_points": 0,
            "pair_collocations": 0,
            "empty_pair_collocations": 0,
            "intersection_grid_points": 0,
            "evictions": 0,
            "oversized_support_skips": 0,
            "peak_cached_bytes": 0,
        }

    def _key(self, evaluator: OrbitalEvaluator, center_bohr: np.ndarray) -> tuple[Any, ...]:
        center = np.asarray(center_bohr, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError(f"center_bohr must have shape (3,), got {center.shape}")
        if self.key_decimals is None:
            center_key: Any = center.tobytes()
        else:
            center_key = tuple(
                float(x) for x in np.round(center, self.key_decimals)
            )
        return (id(evaluator), evaluator.norb, float(evaluator.basis.rcut), center_key)

    def _build_support(
        self,
        evaluator: OrbitalEvaluator,
        center_bohr: np.ndarray,
    ) -> AOSupport:
        index_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        center = np.asarray(center_bohr, dtype=float)
        for points, indices in iter_fft_sphere_nodes(
            self.field,
            center,
            evaluator.basis.rcut,
            chunk_size=self.chunk_size,
        ):
            index_chunks.append(indices)
            value_chunks.append(evaluator.values(points, center))
        if index_chunks:
            integer_indices = np.concatenate(index_chunks, axis=0)
            values = np.concatenate(value_chunks, axis=0)
        else:
            integer_indices = np.empty((0, 3), dtype=np.int64)
            values = np.empty((0, evaluator.norb), dtype=float)
        integer_indices.setflags(write=False)
        values.setflags(write=False)
        center_copy = center.copy()
        center_copy.setflags(write=False)
        self._stats["support_builds"] += 1
        self._stats["evaluated_grid_points"] += int(integer_indices.shape[0])
        return AOSupport(
            integer_indices=integer_indices,
            values=values,
            center_bohr=center_copy,
            radius_bohr=float(evaluator.basis.rcut),
        )

    def support(self, evaluator: OrbitalEvaluator, center_bohr: np.ndarray) -> AOSupport:
        self._stats["support_requests"] += 1
        key = self._key(evaluator, center_bohr)
        cached = self._supports.get(key)
        if cached is not None:
            self._supports.move_to_end(key)
            self._stats["support_hits"] += 1
            return cached

        self._stats["support_misses"] += 1
        support = self._build_support(evaluator, center_bohr)
        if self.max_bytes == 0:
            self._stats["oversized_support_skips"] += 1
            return support
        if self.max_bytes is not None and support.nbytes > self.max_bytes:
            self._stats["oversized_support_skips"] += 1
            return support
        while (
            self.max_bytes is not None
            and self._supports
            and self._cached_bytes + support.nbytes > self.max_bytes
        ):
            _, old = self._supports.popitem(last=False)
            self._cached_bytes -= old.nbytes
            self._stats["evictions"] += 1
        self._supports[key] = support
        self._cached_bytes += support.nbytes
        self._stats["peak_cached_bytes"] = max(
            self._stats["peak_cached_bytes"], self._cached_bytes
        )
        return support

    def collocate_pair(
        self,
        evaluator_i: OrbitalEvaluator,
        evaluator_j: OrbitalEvaluator,
        center_i: np.ndarray,
        center_j: np.ndarray,
    ) -> PairCollocation:
        left = self.support(evaluator_i, center_i)
        right = self.support(evaluator_j, center_j)
        self._stats["pair_collocations"] += 1

        if left.npoints == 0 or right.npoints == 0:
            ia = np.empty(0, dtype=np.int64)
            ib = np.empty(0, dtype=np.int64)
        else:
            _, ia, ib = np.intersect1d(
                _row_keys(left.integer_indices),
                _row_keys(right.integer_indices),
                assume_unique=True,
                return_indices=True,
            )
            if ia.size:
                common = left.integer_indices[ia]
                order = np.lexsort((common[:, 2], common[:, 1], common[:, 0]))
                ia = ia[order]
                ib = ib[order]

        if ia.size == 0:
            self._stats["empty_pair_collocations"] += 1
            integer_indices = np.empty((0, 3), dtype=np.int64)
            wrapped = np.empty((0, 3), dtype=np.int64)
            left_values = np.empty((0, left.norb), dtype=left.values.dtype)
            right_values = np.empty((0, right.norb), dtype=right.values.dtype)
        else:
            integer_indices = np.asarray(left.integer_indices[ia])
            wrapped = np.mod(integer_indices, np.asarray(self.field.shape, dtype=np.int64))
            left_values = np.asarray(left.values[ia])
            right_values = np.asarray(right.values[ib])
        self._stats["intersection_grid_points"] += int(ia.size)
        return PairCollocation(
            integer_indices=integer_indices,
            wrapped_indices=wrapped,
            left_values=left_values,
            right_values=right_values,
            grid_weight=self.field.grid_weight,
            field_shape=self.field.shape,
        )

    def contract_pair(
        self,
        evaluator_i: OrbitalEvaluator,
        evaluator_j: OrbitalEvaluator,
        center_i: np.ndarray,
        center_j: np.ndarray,
        potential_values: np.ndarray | None = None,
    ) -> np.ndarray:
        collocation = self.collocate_pair(evaluator_i, evaluator_j, center_i, center_j)
        if potential_values is None:
            potential_values = self.field.values_ry
        return collocation.contract(potential_values)

    def clear(self) -> None:
        self._supports.clear()
        self._cached_bytes = 0

    def stats(self) -> dict[str, int | float | None]:
        out: dict[str, int | float | None] = dict(self._stats)
        out.update(
            {
                "resident_supports": len(self._supports),
                "cached_bytes": int(self._cached_bytes),
                "max_bytes": self.max_bytes,
                "chunk_size": self.chunk_size,
                "key_decimals": self.key_decimals,
            }
        )
        requests = int(self._stats["support_requests"])
        out["support_hit_rate"] = (
            float(self._stats["support_hits"] / requests) if requests else 0.0
        )
        return out
