from __future__ import annotations

"""Error-controlled low-rank representations of AO pair products.

This module is inspired by resolution-of-identity (RI), tensor
hypercontraction, and interpolative separable density fitting (ISDF): instead of
storing every AO-product column on every grid point, it compresses the sampled
pair-product matrix with a truncated SVD.  It is intentionally a small,
auditable reference primitive rather than a claim of a production ISDF
implementation.

For one potential the exact sparse collocation backend is usually preferable.
The low-rank object becomes useful when the same geometry is contracted against
many local potentials (SCF steps, response calculations, or component-wise
diagnostics), because the AO-product factorization is reused.
"""

from dataclasses import dataclass

import numpy as np

from .grid_collocation import PairCollocation


def _select_rank(
    singular_values: np.ndarray,
    rel_tol: float,
    max_rank: int | None,
    min_rank: int,
) -> tuple[int, float, bool]:
    if not 0.0 <= rel_tol < 1.0:
        raise ValueError("rel_tol must satisfy 0 <= rel_tol < 1")
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative")
    if max_rank is not None and max_rank < 0:
        raise ValueError("max_rank must be non-negative or None")
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0 or not np.any(s):
        return 0, 0.0, True
    sq = s * s
    total = float(np.sum(sq))
    tail = np.empty(s.size + 1, dtype=float)
    tail[-1] = 0.0
    tail[:-1] = np.cumsum(sq[::-1])[::-1]
    errors = np.sqrt(np.maximum(tail, 0.0) / total)
    candidates = np.flatnonzero(errors <= rel_tol)
    rank = int(candidates[0]) if candidates.size else int(s.size)
    rank = max(rank, min(int(min_rank), int(s.size)))
    if max_rank is not None:
        rank = min(rank, int(max_rank), int(s.size))
    achieved = float(errors[rank])
    return rank, achieved, bool(achieved <= rel_tol + 32.0 * np.finfo(float).eps)


@dataclass(frozen=True)
class PairProductLowRank:
    """Truncated-SVD representation of an exact pair-product collocation."""

    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    wrapped_indices: np.ndarray
    field_shape: tuple[int, int, int]
    matrix_shape: tuple[int, int]
    grid_weight: float
    full_rank: int
    requested_rel_tol: float
    achieved_rel_frobenius_error: float
    tolerance_met: bool

    @classmethod
    def from_collocation(
        cls,
        collocation: PairCollocation,
        *,
        rel_tol: float = 1.0e-8,
        max_rank: int | None = None,
        min_rank: int = 0,
    ) -> "PairProductLowRank":
        ni, nj = collocation.matrix_shape
        if collocation.npoints == 0:
            products = np.empty((0, ni * nj), dtype=float)
        else:
            products = np.einsum(
                "gi,gj->gij",
                collocation.left_values,
                collocation.right_values,
                optimize=True,
            ).reshape(collocation.npoints, ni * nj)
        u, s, vh = np.linalg.svd(products, full_matrices=False)
        rank, achieved, met = _select_rank(s, rel_tol, max_rank, min_rank)
        return cls(
            left_vectors=np.asarray(u[:, :rank]),
            singular_values=np.asarray(s[:rank]),
            right_vectors=np.asarray(vh[:rank, :]),
            wrapped_indices=np.asarray(collocation.wrapped_indices),
            field_shape=collocation.field_shape,
            matrix_shape=collocation.matrix_shape,
            grid_weight=float(collocation.grid_weight),
            full_rank=int(s.size),
            requested_rel_tol=float(rel_tol),
            achieved_rel_frobenius_error=float(achieved),
            tolerance_met=met,
        )

    @property
    def rank(self) -> int:
        return int(self.singular_values.size)

    @property
    def npoints(self) -> int:
        return int(self.left_vectors.shape[0])

    @property
    def compression_ratio(self) -> float:
        ni, nj = self.matrix_shape
        dense = self.npoints * ni * nj
        stored = self.left_vectors.size + self.singular_values.size + self.right_vectors.size
        return float(dense / stored) if stored else float("inf")

    @property
    def nbytes(self) -> int:
        return int(
            self.left_vectors.nbytes
            + self.singular_values.nbytes
            + self.right_vectors.nbytes
            + self.wrapped_indices.nbytes
        )

    def _sample(self, potential_values: np.ndarray) -> np.ndarray:
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
        ni, nj = self.matrix_shape
        sampled = self._sample(potential_values)
        if self.rank == 0:
            dtype = np.result_type(sampled, self.left_vectors, float)
            return np.zeros((ni, nj), dtype=dtype)
        projected = self.left_vectors.T @ sampled
        flat = self.right_vectors.T @ (self.singular_values * projected)
        return (flat * self.grid_weight).reshape(ni, nj)

    def contract_many(self, potential_values: np.ndarray) -> np.ndarray:
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
        if self.rank == 0:
            dtype = np.result_type(sampled, self.left_vectors, float)
            return np.zeros((fields.shape[0], ni, nj), dtype=dtype)
        projected = sampled @ self.left_vectors
        scaled = projected * self.singular_values[None, :]
        flat = scaled @ self.right_vectors
        return (flat * self.grid_weight).reshape(fields.shape[0], ni, nj)

    def diagnostics(self) -> dict[str, int | float | bool]:
        return {
            "npoints": self.npoints,
            "matrix_rows": int(self.matrix_shape[0]),
            "matrix_cols": int(self.matrix_shape[1]),
            "full_rank": self.full_rank,
            "rank": self.rank,
            "requested_rel_tol": self.requested_rel_tol,
            "achieved_rel_frobenius_error": self.achieved_rel_frobenius_error,
            "tolerance_met": self.tolerance_met,
            "compression_ratio": self.compression_ratio,
            "storage_bytes": self.nbytes,
        }
