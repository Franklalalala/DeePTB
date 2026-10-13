from __future__ import annotations

import numpy as np


def simpson_rab_weights(rab: np.ndarray) -> np.ndarray:
    """Return ABACUS/QE composite-Simpson weights for a UPF radial mesh.

    ``PP_RAB(i)`` is the Jacobian ``dr/dx`` (including the uniform mesh step in
    the UPF convention).  ABACUS ``Integral::Simpson_Integral`` evaluates

    ``integral f(r) dr = sum_i c_i f_i PP_RAB_i``

    with coefficients ``1,4,2,...,4,1`` divided by three.

    Odd meshes (an even number of intervals) use the exact ABACUS composite
    Simpson rule.  Production norm-conserving UPFs (e.g. ONCV Si/Ca) instead
    ship an *even* number of mesh points, for which a single composite-Simpson
    rule does not exist.  Rather than fail closed (the v0.4.1 behaviour, which
    rejected every real UPF), an even mesh is integrated by applying composite
    Simpson to the first ``n-1`` samples (an even number of intervals) and
    closing the final interval ``[n-2, n-1]`` with the trapezoid rule.  This
    keeps every tabulated ``PP_R``/``PP_RAB`` sample untouched -- no resampling,
    padding, or truncation -- and reduces to pure composite Simpson for odd
    meshes.  The trailing trapezoid is second order on a single interval; on the
    thousand-point production meshes where the integrand tail is already
    negligible its contribution is far below the acceptance tolerance.
    """
    rab = np.asarray(rab, dtype=float)
    if rab.ndim != 1:
        raise ValueError(f"PP_RAB must be one-dimensional, got shape {rab.shape}")
    if rab.size < 3:
        raise ValueError(
            "ABACUS-compatible radial integration requires at least three "
            f"mesh points, got {rab.size}."
        )
    if not np.isfinite(rab).all() or np.any(rab <= 0.0):
        raise ValueError("PP_RAB must contain finite positive radial Jacobians")
    n = rab.size
    if n % 2 == 1:
        # Odd mesh (even number of intervals): exact ABACUS composite Simpson.
        coeff = np.ones(n, dtype=float)
        coeff[1:-1:2] = 4.0
        coeff[2:-1:2] = 2.0
        return coeff * rab / 3.0
    # Even mesh (odd number of intervals): composite Simpson on samples
    # 0..n-2 (weights 1,4,2,...,4,1 / 3) plus a trapezoid on [n-2, n-1].
    coeff = np.zeros(n, dtype=float)
    coeff[0] += 1.0 / 3.0
    coeff[1:n - 2:2] += 4.0 / 3.0
    coeff[2:n - 2:2] += 2.0 / 3.0
    coeff[n - 2] += 1.0 / 3.0
    # Closing interval via the trapezoid rule (unit x-step, PP_RAB Jacobian).
    coeff[n - 2] += 0.5
    coeff[n - 1] += 0.5
    return coeff * rab


def simpson_rab(values: np.ndarray, rab: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Integrate tabulated values with ABACUS's exact ``PP_RAB`` convention."""
    array = np.asarray(values)
    moved = np.moveaxis(array, axis, -1)
    weights = simpson_rab_weights(rab)
    if moved.shape[-1] != weights.size:
        raise ValueError(
            f"Radial data length {moved.shape[-1]} does not match PP_RAB length {weights.size}"
        )
    return np.sum(moved * weights, axis=-1)


def cumulative_simpson_rab(values: np.ndarray, rab: np.ndarray) -> np.ndarray:
    """ABACUS ``Simpson_Integral_0toall`` cumulative radial integral.

    Even-index endpoints use composite Simpson; the intervening odd point uses
    the local trapezoid exactly as in the ABACUS implementation.  For an even
    mesh (odd number of intervals) the composite-Simpson pairs cover samples
    ``0..n-2`` and the final interval ``[n-2, n-1]`` is closed with the
    trapezoid rule, so the cumulative endpoint matches :func:`simpson_rab`
    exactly and no sample is dropped.
    """
    values = np.asarray(values)
    weights = np.asarray(rab, dtype=float)
    # Reuse all mesh validation from the full-integral weight builder.
    simpson_rab_weights(weights)
    if values.ndim != 1 or values.shape != weights.shape:
        raise ValueError("cumulative radial data and PP_RAB must be matching 1-D arrays")
    n = values.size
    out = np.zeros(n, dtype=np.result_type(values, float))
    # Largest even index reached by complete Simpson pairs: n-1 (odd mesh) or
    # n-2 (even mesh).  range(1, last_pair_end, 2) is identical to the original
    # range(1, n, 2) for odd meshes, so odd-mesh output is unchanged.
    last_pair_end = n - 1 if (n % 2 == 1) else n - 2
    f3 = values[0] * weights[0]
    for i in range(1, last_pair_end, 2):
        f1 = f3
        f2 = values[i] * weights[i]
        f3 = values[i + 1] * weights[i + 1]
        out[i] = out[i - 1] + 0.5 * (f1 + f2)
        out[i + 1] = out[i - 1] + (f1 + 4.0 * f2 + f3) / 3.0
    if n % 2 == 0:
        # Close the final odd interval with the trapezoid rule.
        fa = values[n - 2] * weights[n - 2]
        fb = values[n - 1] * weights[n - 1]
        out[n - 1] = out[n - 2] + 0.5 * (fa + fb)
    return out
