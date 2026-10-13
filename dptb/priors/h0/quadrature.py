from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .grid_collocation import FFTGridAOCache
from .radial import OrbitalEvaluator
from .reciprocal import PeriodicField


@dataclass(frozen=True)
class CartesianGrid:
    points: np.ndarray
    weight: float
    effective_step: tuple[float, float, float]


def _intersection_bounds(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    ca = np.asarray(center_a, dtype=float)
    cb = np.asarray(center_b, dtype=float)
    if np.linalg.norm(ca - cb) > radius_a + radius_b:
        return None
    lower = np.maximum(ca - radius_a, cb - radius_b)
    upper = np.minimum(ca + radius_a, cb + radius_b)
    if np.any(upper <= lower):
        return None
    return lower, upper


def intersection_grid(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
    target_step: float,
) -> CartesianGrid:
    """Midpoint Cartesian grid over the intersection of two support spheres."""
    if target_step <= 0.0:
        raise ValueError("target_step must be positive")
    ca = np.asarray(center_a, dtype=float)
    cb = np.asarray(center_b, dtype=float)
    bounds = _intersection_bounds(ca, radius_a, cb, radius_b)
    if bounds is None:
        return CartesianGrid(np.empty((0, 3)), 0.0, (0.0, 0.0, 0.0))
    lower, upper = bounds
    lengths = upper - lower
    counts = np.maximum(1, np.ceil(lengths / target_step).astype(int))
    steps = lengths / counts
    axes = [lower[d] + (np.arange(counts[d]) + 0.5) * steps[d] for d in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    mask = (
        np.einsum("ij,ij->i", points - ca, points - ca) <= radius_a**2
    ) & (
        np.einsum("ij,ij->i", points - cb, points - cb) <= radius_b**2
    )
    return CartesianGrid(points[mask], float(np.prod(steps)), tuple(float(x) for x in steps))


def _fft_intersection_chunks(
    field: PeriodicField,
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
    *,
    chunk_size: int,
):
    """Yield unwrapped FFT nodes and wrapped field indices in a sphere intersection.

    Real-space ``H(R)`` matrix elements use localized orbitals centered at an
    unwrapped atom image, but the local potential is periodic.  Integer FFT
    indices are therefore enumerated in unwrapped space and reduced modulo the
    FFT shape only when indexing the field.
    """
    ca = np.asarray(center_a, dtype=float)
    cb = np.asarray(center_b, dtype=float)
    bounds = _intersection_bounds(ca, radius_a, cb, radius_b)
    if bounds is None:
        return
    lower, upper = bounds
    cell = np.asarray(field.cell_bohr, dtype=float)
    inv_cell = np.linalg.inv(cell)
    shape = np.asarray(field.shape, dtype=np.int64)

    # Transform all eight Cartesian bounding-box corners.  Their fractional
    # extrema enclose the complete box even for an oblique cell.
    corners = np.asarray(
        [[x, y, z] for x, y, z in product(*zip(lower, upper))],
        dtype=float,
    )
    scaled = (corners @ inv_cell) * shape
    nmin = np.floor(np.min(scaled, axis=0)).astype(np.int64) - 1
    nmax = np.ceil(np.max(scaled, axis=0)).astype(np.int64) + 1
    counts = nmax - nmin + 1
    if np.any(counts <= 0):
        return
    total = int(np.prod(counts, dtype=np.int64))
    yz = int(counts[1] * counts[2])
    for start in range(0, total, chunk_size):
        linear = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        ix = linear // yz
        rem = linear % yz
        iy = rem // int(counts[2])
        iz = rem % int(counts[2])
        integer_index = np.column_stack((ix, iy, iz)) + nmin
        points = (integer_index / shape) @ cell
        da = points - ca
        db = points - cb
        mask = (
            np.einsum("ij,ij->i", da, da) <= radius_a**2 + 1.0e-12
        ) & (
            np.einsum("ij,ij->i", db, db) <= radius_b**2 + 1.0e-12
        )
        if not np.any(mask):
            continue
        integer_index = integer_index[mask]
        points = points[mask]
        wrapped = np.mod(integer_index, shape)
        yield points, wrapped


def local_pair_integral_fft_grid(
    evaluator_i: OrbitalEvaluator,
    evaluator_j: OrbitalEvaluator,
    center_i: np.ndarray,
    center_j: np.ndarray,
    field: PeriodicField,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Contract ``<phi_i|Veff|phi_j>`` on the exact periodic FFT nodes."""
    out = np.zeros((evaluator_i.norb, evaluator_j.norb), dtype=float)
    for points, wrapped in _fft_intersection_chunks(
        field,
        center_i,
        evaluator_i.basis.rcut,
        center_j,
        evaluator_j.basis.rcut,
        chunk_size=chunk_size,
    ):
        pi = evaluator_i.values(points, center_i)
        pj = evaluator_j.values(points, center_j)
        values = field.values_ry[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]
        out += pi.T @ (values[:, None] * pj) * field.grid_weight
    return out


def local_pair_integral_fft_grid_cached(
    evaluator_i: OrbitalEvaluator,
    evaluator_j: OrbitalEvaluator,
    center_i: np.ndarray,
    center_j: np.ndarray,
    field: PeriodicField,
    *,
    cache: FFTGridAOCache | None = None,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Exact FFT-grid contraction with reusable sparse AO collocation.

    The result uses the same grid nodes and quadrature weight as
    :func:`local_pair_integral_fft_grid`.  Only the orbital-evaluation dataflow
    changes: each atom/image support is evaluated once and reused across pairs.
    """
    if cache is None:
        cache = FFTGridAOCache(field, chunk_size=chunk_size)
    if cache.field is not field:
        raise ValueError("FFTGridAOCache is tied to a different PeriodicField")
    return cache.contract_pair(
        evaluator_i,
        evaluator_j,
        center_i,
        center_j,
        field.values_ry,
    )


def scalar_pair_integrals(
    evaluator_i: OrbitalEvaluator,
    evaluator_j: OrbitalEvaluator,
    center_i: np.ndarray,
    center_j: np.ndarray,
    field: PeriodicField | None,
    step_bohr: float,
    chunk_size: int = 100_000,
    *,
    local_integration: str = "fft_grid",
    local_cache: FFTGridAOCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``S``, ``T``, and the complete local ``Veff`` block in Rydberg.

    ``S`` and ``T`` retain the transparent midpoint reference quadrature.  The
    production local path evaluates AO products on the exact periodic FFT nodes
    used for ``Vloc+VH+Vxc``.  The old interpolated midpoint path remains an
    explicit diagnostic mode and is never selected silently.
    """
    mode = str(local_integration).lower().replace("-", "_")
    aliases = {
        "fft": "fft_grid",
        "exact_fft": "fft_grid",
        "cached_fft": "fft_grid_cached",
        "exact_fft_cached": "fft_grid_cached",
        "collocation": "fft_grid_cached",
        "midpoint": "midpoint_interpolated",
        "interpolated": "midpoint_interpolated",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"fft_grid", "fft_grid_cached", "midpoint_interpolated"}:
        raise ValueError(
            "local_integration must be 'fft_grid', 'fft_grid_cached', "
            "or 'midpoint_interpolated'"
        )

    ni, nj = evaluator_i.norb, evaluator_j.norb
    s = np.zeros((ni, nj), dtype=float)
    t = np.zeros((ni, nj), dtype=float)
    v = np.zeros((ni, nj), dtype=float)
    grid = intersection_grid(
        center_i, evaluator_i.basis.rcut, center_j, evaluator_j.basis.rcut, step_bohr
    )
    if len(grid.points):
        dv = grid.weight
        for start in range(0, len(grid.points), chunk_size):
            pts = grid.points[start : start + chunk_size]
            pi = evaluator_i.values(pts, center_i)
            pj = evaluator_j.values(pts, center_j)
            li = evaluator_i.values(pts, center_i, laplacian=True)
            lj = evaluator_j.values(pts, center_j, laplacian=True)
            s += pi.T @ pj * dv
            # In Rydberg atomic units, kinetic operator is -nabla^2.
            t += -0.5 * (pi.T @ lj + li.T @ pj) * dv
            if field is not None and mode == "midpoint_interpolated":
                vp = field.interpolate(pts)
                v += pi.T @ (vp[:, None] * pj) * dv

    if field is not None and mode == "fft_grid":
        v = local_pair_integral_fft_grid(
            evaluator_i,
            evaluator_j,
            center_i,
            center_j,
            field,
            chunk_size=chunk_size,
        )
    elif field is not None and mode == "fft_grid_cached":
        v = local_pair_integral_fft_grid_cached(
            evaluator_i,
            evaluator_j,
            center_i,
            center_j,
            field,
            cache=local_cache,
            chunk_size=chunk_size,
        )
    return s, t, v
