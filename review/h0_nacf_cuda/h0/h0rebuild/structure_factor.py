"""Portable type-1 Gaussian NUFFT for atom structure factors (CPU float64).

Computes sum_j exp(-2*pi*i*k.frac_j) in NumPy FFT mode order. This is an
explicit approximation, never a replacement silently selected by a heuristic.
At fixed requested eps and oversampling: O(N*w^3 + K*log(K)) work, O(K) scratch;
K is the oversampled grid size and w the fixed one-dimensional stencil width.
No G-by-N array is constructed, and no dense fallback is used on failure.
"""
from __future__ import annotations

from math import prod

import numpy as np
from scipy.fft import fftn, next_fast_len


OVERSAMPLING = 3
MIN_EPS = 1.0e-12
MAX_EPS = 1.0e-3


def gaussian_nufft_parameters(
    shape: tuple[int, int, int],
    eps: float = 1.0e-12,
    max_work_mb: float | None = 512.0,
) -> dict[str, object]:
    """Window from Gaussian alias/tail estimates, not a bound on final H0.

    With K_d >= 3*n_d, |k_d/K_d| <= 1/6. The nearest Poisson alias divided
    by the main Gaussian Fourier coefficient is <= exp(-8*pi^2*tau/3).
    Per-axis inverse Gaussian amplification is <= exp(pi^2*tau/9).
    Set these alias/tail scales below eps/64 with a one-node tail margin.
    Floating-point roundoff and subsequent nonlinear XC are separate errors.
    """
    if len(shape) != 3 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
        for n in shape
    ):
        raise ValueError("shape must contain three positive integers")
    eps = float(eps)
    if not np.isfinite(eps) or not MIN_EPS <= eps <= MAX_EPS:
        raise ValueError(f"structure_factor_eps must lie in [{MIN_EPS}, {MAX_EPS}]")
    if max_work_mb is not None and (not np.isfinite(max_work_mb) or max_work_mb <= 0):
        raise ValueError("structure_factor_max_work_mb must be positive or None")
    work_shape = tuple(int(next_fast_len(OVERSAMPLING * int(n))) for n in shape)
    log_scale = float(np.log(64.0 / eps))
    tau = log_scale / (4 * np.pi**2 * (1 - 1 / OVERSAMPLING))
    half_width = int(np.ceil(np.sqrt(4 * tau * (log_scale + np.pi**2 * tau / 9)))) + 1
    width = 2 * half_width + 1
    # Conservative estimate for our grid/FFT/gather/window arrays. This is NOT
    # a hard bound on the whole H0 process or on SciPy's internal allocator.
    estimate = 48 * prod(work_shape) + 32 * prod(shape) + 32 * width**3
    if max_work_mb is not None and estimate > float(max_work_mb) * 1024**2:
        raise MemoryError(
            f"Gaussian NUFFT estimated scratch {estimate / 1024**2:.1f} MiB exceeds "
            f"structure_factor_max_work_mb={max_work_mb}. Raise the explicit budget "
            "or use a smaller grid; no quadratic fallback is performed."
        )
    return dict(
        algorithm="gaussian_type1_cpu_float64",
        requested_eps=eps,
        error_normalization="absolute SF error / number of atoms; not per-mode relative or H0 error",
        oversampling=OVERSAMPLING,
        work_shape=list(work_shape),
        tau_grid_units=float(tau),
        stencil_half_width=half_width,
        stencil_width=width,
        estimated_scratch_bytes=int(estimate),
        max_work_mb=max_work_mb,
        fft_workers=1,
    )


def gaussian_structure_factor(
    fractional_positions: np.ndarray,
    shape: tuple[int, int, int],
    *,
    eps: float = 1.0e-12,
    max_work_mb: float | None = 512.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Unweighted periodic structure factor in fftfreq order, with exact DC.

    Spread a truncated Gaussian on an oversampled *periodic* grid, do the
    negative-sign FFT, then divide by the Gaussian Fourier transform.
    Coordinates are fractional in the original row-lattice basis. Thus this
    formula works unchanged for triclinic cells. It handles odd/even shapes,
    negative Nyquist modes and arbitrarily many images within a small grid.
    """
    frac = np.asarray(fractional_positions, dtype=np.float64)
    if frac.ndim != 2 or frac.shape[1] != 3 or not np.isfinite(frac).all():
        raise ValueError("fractional_positions must have finite shape (N,3)")
    meta = gaussian_nufft_parameters(shape, eps, max_work_mb)
    shape = tuple(int(n) for n in shape)
    meta.update(atoms=len(frac), dense_phase_elements=0, dc_enforced_exact=True)
    half_width = int(meta["stencil_half_width"])
    width = int(meta["stencil_width"])
    meta["nominal_stencil_updates"] = int(len(frac) * width**3)
    meta["spreading_updates"] = int(len(frac) * prod(min(int(k), width) for k in meta["work_shape"]))
    if len(frac) == 0:
        return np.zeros(shape, dtype=np.complex128), meta
    K = np.asarray(meta["work_shape"], dtype=np.int64)
    tau = float(meta["tau_grid_units"])
    grid = np.zeros(tuple(K), dtype=np.float64)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int64)
    for point in frac:
        u = np.remainder(point, 1.0) * K
        indices = [int(np.floor(u[d])) + offsets for d in range(3)]
        weights = [np.exp(-(indices[d] - u[d])**2 / (4 * tau)) for d in range(3)]
        wrapped = [indices[d] % K[d] for d in range(3)]
        # Fold repeated periodic stencil indices in each 1D weight first.
        # A 3D fancy-index += is then safe (all triples are unique), avoiding
        # the much slower general-purpose np.add.at on every 3D stencil.
        # The folded arrays have length <= width, never scale with system size.
        for d in range(3):
            if K[d] < width:
                weights[d] = np.bincount(wrapped[d], weights=weights[d], minlength=int(K[d]))
                wrapped[d] = np.arange(K[d], dtype=np.int64)
        window = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
        grid[np.ix_(*wrapped)] += window
    transformed = fftn(grid, workers=1, overwrite_x=True)
    modes = [np.rint(np.fft.fftfreq(n, d=1.0/n)).astype(np.int64) for n in shape]
    out = transformed[np.ix_(*(modes[d] % K[d] for d in range(3)))].copy()
    for d in range(3):
        broadcast_shape = [1, 1, 1]
        broadcast_shape[d] = shape[d]
        inverse_gaussian = np.exp(4 * np.pi**2 * tau * (modes[d] / K[d])**2) / np.sqrt(4 * np.pi * tau)
        out *= inverse_gaussian.reshape(broadcast_shape)
    # The zero mode is analytically N: preserve electron count and G=0 gauge.
    out[0, 0, 0] = len(frac)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite Gaussian NUFFT output")
    return out, meta
