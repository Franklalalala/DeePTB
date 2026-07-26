"""Fixed-chemical-potential dense H/S operator utilities.

The routines here are intentionally small and dependency-light.  They take
predicted dense DPTB Hamiltonian/overlap matrices, solve the generalized
eigenproblem, evaluate fixed-mu Fermi occupations, and return density-matrix
and energy ledgers suitable for downstream conservation checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:  # SciPy gives the most stable Hermitian generalized eigensolver.
    from scipy import linalg as _scipy_linalg
except Exception:  # pragma: no cover - exercised only in minimal runtimes.
    _scipy_linalg = None

ArrayLike = Any


class FixedMuOperatorError(ValueError):
    """Base class for fail-closed fixed-mu operator validation errors."""


class OverlapConditionError(FixedMuOperatorError):
    """Raised when the overlap matrix is not safely positive definite."""


@dataclass(frozen=True)
class GeneralizedBands:
    """Generalized spectrum for a dense non-orthogonal H/S stack.

    ``eigvecs`` are S-orthonormal column eigenvectors satisfying
    ``C.conj().T @ S @ C = I`` for every leading item.
    """

    eigvals: np.ndarray
    eigvecs: np.ndarray
    min_overlap_eig: np.ndarray
    overlap_condition: np.ndarray


@dataclass(frozen=True)
class EnergyLedger:
    """Band/free/grand-energy bookkeeping at fixed chemical potential.

    ``entropy_term`` is the finite-temperature ``-T*S`` contribution:
    ``g * sum_k w_k * kT * [f log f + (1-f) log(1-f)]``.
    """

    band_energy: np.ndarray
    entropy_term: np.ndarray
    free_energy: np.ndarray
    grand_energy: np.ndarray


@dataclass(frozen=True)
class ConservationLedger:
    """Hamiltonian/DM downstream conservation diagnostics."""

    electron_count_from_density: np.ndarray
    band_energy_from_density: np.ndarray
    electron_count_residual: np.ndarray
    band_energy_residual: np.ndarray
    density_hermiticity_error: np.ndarray


@dataclass(frozen=True)
class FixedMuResult:
    """Observable bundle for ``H, S`` evaluated at a fixed ``mu``."""

    mu: float
    kT: float
    spin_degeneracy: float
    eigvals: np.ndarray
    eigvecs: np.ndarray
    occupations: np.ndarray
    occupation_response: np.ndarray
    density_k: np.ndarray
    density_response_k: np.ndarray
    electron_count: np.ndarray
    dos_like_response: np.ndarray
    density: np.ndarray
    density_response: np.ndarray
    energies: EnergyLedger
    conservation: ConservationLedger
    k_weights: np.ndarray
    k_axis: Optional[int]
    min_overlap_eig: np.ndarray
    overlap_condition: np.ndarray


def hermitian_part(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    return 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))


def _as_float_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise FixedMuOperatorError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise FixedMuOperatorError(f"{name} must be positive, got {value!r}")
    if nonnegative and out < 0.0:
        raise FixedMuOperatorError(f"{name} must be nonnegative, got {value!r}")
    return out


def _validate_matrix_stack(
    h: ArrayLike,
    s: ArrayLike,
    *,
    hermitian_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    h_arr = np.asarray(h)
    s_arr = np.asarray(s)
    if h_arr.shape != s_arr.shape:
        raise FixedMuOperatorError(f"H and S shapes must match, got {h_arr.shape} and {s_arr.shape}")
    if h_arr.ndim < 2 or h_arr.shape[-1] != h_arr.shape[-2]:
        raise FixedMuOperatorError(f"H/S must be square matrices or stacks, got shape {h_arr.shape}")
    if not np.issubdtype(h_arr.dtype, np.number) or not np.issubdtype(s_arr.dtype, np.number):
        raise FixedMuOperatorError("H and S must be numeric arrays")
    dtype = np.result_type(h_arr, s_arr, np.float64)
    h_arr = np.asarray(h_arr, dtype=dtype)
    s_arr = np.asarray(s_arr, dtype=dtype)
    if not np.isfinite(h_arr).all() or not np.isfinite(s_arr).all():
        raise FixedMuOperatorError("H and S must contain only finite values")
    hermitian_tol = _as_float_scalar("hermitian_tol", hermitian_tol, nonnegative=True)
    if hermitian_tol == 0.0:
        h_ok = np.array_equal(h_arr, np.swapaxes(h_arr.conj(), -1, -2))
        s_ok = np.array_equal(s_arr, np.swapaxes(s_arr.conj(), -1, -2))
    else:
        h_ok = np.allclose(h_arr, np.swapaxes(h_arr.conj(), -1, -2), atol=hermitian_tol, rtol=hermitian_tol)
        s_ok = np.allclose(s_arr, np.swapaxes(s_arr.conj(), -1, -2), atol=hermitian_tol, rtol=hermitian_tol)
    if not h_ok:
        raise FixedMuOperatorError("H must be Hermitian within hermitian_tol")
    if not s_ok:
        raise FixedMuOperatorError("S must be Hermitian within hermitian_tol")
    return hermitian_part(h_arr), hermitian_part(s_arr)


def _canonical_k_axis(k_axis: Optional[int], leading_ndim: int) -> Optional[int]:
    if k_axis is None:
        return None
    if leading_ndim == 0:
        raise FixedMuOperatorError("k_axis is invalid for a single matrix with no leading k dimension")
    axis = int(k_axis)
    if axis < 0:
        axis += leading_ndim
    if axis < 0 or axis >= leading_ndim:
        raise FixedMuOperatorError(f"k_axis={k_axis} outside leading dimensions of rank {leading_ndim}")
    return axis


def _prepare_weights(
    leading_shape: Tuple[int, ...],
    *,
    k_weights: Optional[ArrayLike],
    k_axis: Optional[int],
    normalize_k_weights: bool,
) -> np.ndarray:
    if not leading_shape:
        if k_weights is None:
            return np.asarray(1.0, dtype=np.float64)
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape not in [(), (1,)]:
            raise FixedMuOperatorError("single-matrix k_weights must be scalar")
        return np.asarray(float(weights.reshape(-1)[0]), dtype=np.float64)

    if k_weights is None:
        weights = np.ones(leading_shape, dtype=np.float64)
    else:
        raw = np.asarray(k_weights, dtype=np.float64)
        if k_axis is not None and raw.ndim == 1 and raw.shape[0] == leading_shape[k_axis]:
            shape = [1] * len(leading_shape)
            shape[k_axis] = raw.shape[0]
            raw = raw.reshape(shape)
        try:
            weights = np.broadcast_to(raw, leading_shape).astype(np.float64, copy=True)
        except ValueError as exc:
            raise FixedMuOperatorError(
                f"k_weights shape {raw.shape} cannot broadcast to leading H/S shape {leading_shape}"
            ) from exc

    if not np.isfinite(weights).all():
        raise FixedMuOperatorError("k_weights must be finite")
    if np.any(weights < 0.0):
        raise FixedMuOperatorError("k_weights must be nonnegative")
    if normalize_k_weights and k_axis is not None:
        total = weights.sum(axis=k_axis, keepdims=True)
        if np.any(total <= 0.0):
            raise FixedMuOperatorError("k_weights must have positive sum along k_axis")
        weights = weights / total
    return weights


def _stable_fermi(eps: np.ndarray, mu: float, kT: float) -> Tuple[np.ndarray, np.ndarray]:
    if kT == 0.0:
        below = eps < mu
        equal = eps == mu
        f = below.astype(np.float64) + 0.5 * equal.astype(np.float64)
        response = np.zeros_like(f, dtype=np.float64)
        return f, response

    x = (eps - mu) / kT
    f = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    exp_neg = np.exp(-np.clip(x[pos], 0.0, 745.0))
    f[pos] = exp_neg / (1.0 + exp_neg)
    exp_pos = np.exp(np.clip(x[~pos], -745.0, 0.0))
    f[~pos] = 1.0 / (1.0 + exp_pos)
    response = f * (1.0 - f) / kT
    return f, response


def _check_overlap(
    s: np.ndarray,
    *,
    eig_floor: float,
    max_condition: float,
) -> Tuple[np.ndarray, np.ndarray]:
    se = np.linalg.eigvalsh(s).real
    min_eig = se[..., 0]
    max_eig = se[..., -1]
    if np.any(min_eig <= eig_floor):
        raise OverlapConditionError(
            f"S must be positive definite above eig_floor={eig_floor:g}; min={float(np.min(min_eig)):.6g}"
        )
    cond = max_eig / min_eig
    if np.any(cond > max_condition):
        raise OverlapConditionError(
            f"S condition number exceeds max_condition={max_condition:g}; max={float(np.max(cond)):.6g}"
        )
    return min_eig, cond


def _generalized_eigh_single(h: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_linalg is not None:
        vals, vecs = _scipy_linalg.eigh(h, s, check_finite=False)
        return vals.real, vecs

    chol = np.linalg.cholesky(s)
    inv_l_h = np.linalg.inv(chol.conj().T)
    h_orth = hermitian_part(np.linalg.solve(chol, h) @ inv_l_h)
    vals, q = np.linalg.eigh(h_orth)
    vecs = np.linalg.solve(chol.conj().T, q)
    return vals.real, vecs


def generalized_bands(
    h: ArrayLike,
    s: ArrayLike,
    *,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
) -> GeneralizedBands:
    """Solve ``H C = S C eps`` for a dense Hermitian H/S stack.

    The overlap matrix is not silently regularized: non-positive or badly
    conditioned S raises :class:`OverlapConditionError`.
    """

    eig_floor = _as_float_scalar("eig_floor", eig_floor, positive=True)
    max_condition = _as_float_scalar("max_condition", max_condition, positive=True)
    h_arr, s_arr = _validate_matrix_stack(h, s, hermitian_tol=hermitian_tol)
    min_eig, cond = _check_overlap(s_arr, eig_floor=eig_floor, max_condition=max_condition)
    n = h_arr.shape[-1]
    leading_shape = h_arr.shape[:-2]
    flat_h = h_arr.reshape((-1, n, n))
    flat_s = s_arr.reshape((-1, n, n))
    vals = []
    vecs = []
    for h_one, s_one in zip(flat_h, flat_s):
        eps, c = _generalized_eigh_single(h_one, s_one)
        vals.append(eps)
        vecs.append(c)
    eigvals = np.stack(vals, axis=0).reshape(leading_shape + (n,))
    eigvecs = np.stack(vecs, axis=0).reshape(leading_shape + (n, n))
    return GeneralizedBands(
        eigvals=eigvals,
        eigvecs=eigvecs,
        min_overlap_eig=np.asarray(min_eig),
        overlap_condition=np.asarray(cond),
    )


def fermi_dirac(
    energies: ArrayLike,
    mu: float,
    kT: float,
    *,
    spin_degeneracy: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return occupations and ``d occupation / d mu``.

    ``kT`` is an energy smearing in the same units as ``energies`` and ``mu``.
    At zero temperature the discontinuity uses a half-filled boundary for
    exactly degenerate ``eps == mu`` states and the response is zero.
    """

    mu = _as_float_scalar("mu", mu)
    kT = _as_float_scalar("kT", kT, nonnegative=True)
    spin_degeneracy = _as_float_scalar("spin_degeneracy", spin_degeneracy, positive=True)
    eps = np.asarray(energies, dtype=np.float64)
    if not np.isfinite(eps).all():
        raise FixedMuOperatorError("energies must be finite")
    f, response = _stable_fermi(eps, mu=mu, kT=kT)
    return spin_degeneracy * f, spin_degeneracy * response


def _weighted_sum(values: np.ndarray, weights: np.ndarray, *, k_axis: Optional[int]) -> np.ndarray:
    if k_axis is None:
        return weights * values
    return np.sum(weights * values, axis=k_axis)


def _density_from_vectors(vecs: np.ndarray, occ: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted_occ = occ * weights[..., None]
    return np.einsum("...ni,...i,...mi->...nm", vecs, weighted_occ, vecs.conj(), optimize=True)


def _aggregate_matrix(values: np.ndarray, *, k_axis: Optional[int]) -> np.ndarray:
    if k_axis is None:
        return values
    return np.sum(values, axis=k_axis)


def _density_hermiticity_error(density: np.ndarray) -> np.ndarray:
    anti = density - np.swapaxes(density.conj(), -1, -2)
    denom = np.linalg.norm(density.reshape(density.shape[:-2] + (-1,)), axis=-1)
    denom = np.maximum(denom, 1e-300)
    return np.linalg.norm(anti.reshape(anti.shape[:-2] + (-1,)), axis=-1) / denom


def _trace_last2(x: np.ndarray) -> np.ndarray:
    return np.trace(x, axis1=-2, axis2=-1).real


def _energy_ledger(
    eps: np.ndarray,
    f_base: np.ndarray,
    weights: np.ndarray,
    *,
    mu: float,
    kT: float,
    spin_degeneracy: float,
    k_axis: Optional[int],
) -> EnergyLedger:
    weighted = weights[..., None]
    band_k = spin_degeneracy * np.sum(weighted * eps * f_base, axis=-1)
    if kT == 0.0:
        entropy_k = np.zeros_like(band_k, dtype=np.float64)
        grand_k = spin_degeneracy * np.sum(weighted * (eps - mu) * f_base, axis=-1)
    else:
        f_clip = np.clip(f_base, 1e-300, 1.0 - 1e-16)
        entropy_k = spin_degeneracy * kT * np.sum(
            weighted * (f_clip * np.log(f_clip) + (1.0 - f_clip) * np.log1p(-f_clip)),
            axis=-1,
        )
        x = (mu - eps) / kT
        grand_k = -spin_degeneracy * kT * np.sum(weighted * np.logaddexp(0.0, x), axis=-1)
    free_k = band_k + entropy_k
    return EnergyLedger(
        band_energy=_aggregate_matrix(band_k, k_axis=k_axis),
        entropy_term=_aggregate_matrix(entropy_k, k_axis=k_axis),
        free_energy=_aggregate_matrix(free_k, k_axis=k_axis),
        grand_energy=_aggregate_matrix(grand_k, k_axis=k_axis),
    )


def fixed_mu_observables(
    h: ArrayLike,
    s: ArrayLike,
    *,
    mu: float,
    kT: float = 0.0,
    spin_degeneracy: float = 2.0,
    k_weights: Optional[ArrayLike] = None,
    k_axis: Optional[int] = None,
    normalize_k_weights: bool = True,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
) -> FixedMuResult:
    """Evaluate ``N(mu)``, ``D(mu)``, response, and energies for dense H/S.

    Parameters
    ----------
    h, s
        Dense Hermitian Hamiltonian and positive-definite overlap arrays with
        shape ``[..., norb, norb]``.  Real, complex, and SOC spinor matrices are
        all represented by the same dense shape.
    mu
        Target chemical potential in the same energy unit as ``h``.
    kT
        Fermi-Dirac smearing energy.  ``kT=0`` uses a half-filled boundary for
        states exactly at ``mu``.
    spin_degeneracy
        Occupancy multiplier.  Use ``2`` for spin-degenerate non-SOC matrices
        and ``1`` for explicit spin/SOC spinor matrices.
    k_weights, k_axis
        Optional Brillouin-zone weights and the leading axis to sum over.
        With ``k_axis=None`` no k aggregation is performed.
    """

    mu = _as_float_scalar("mu", mu)
    kT = _as_float_scalar("kT", kT, nonnegative=True)
    spin_degeneracy = _as_float_scalar("spin_degeneracy", spin_degeneracy, positive=True)
    bands = generalized_bands(
        h,
        s,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    h_arr, s_arr = _validate_matrix_stack(h, s, hermitian_tol=hermitian_tol)
    leading_shape = h_arr.shape[:-2]
    k_axis_c = _canonical_k_axis(k_axis, len(leading_shape))
    weights = _prepare_weights(
        leading_shape,
        k_weights=k_weights,
        k_axis=k_axis_c,
        normalize_k_weights=normalize_k_weights,
    )

    f_base, response_base = _stable_fermi(bands.eigvals, mu=mu, kT=kT)
    occupations = spin_degeneracy * f_base
    occupation_response = spin_degeneracy * response_base
    density_k = _density_from_vectors(bands.eigvecs, occupations, weights)
    density_response_k = _density_from_vectors(bands.eigvecs, occupation_response, weights)
    density = _aggregate_matrix(density_k, k_axis=k_axis_c)
    density_response = _aggregate_matrix(density_response_k, k_axis=k_axis_c)

    n_k = weights * np.sum(occupations, axis=-1)
    dos_k = weights * np.sum(occupation_response, axis=-1)
    electron_count = _aggregate_matrix(n_k, k_axis=k_axis_c)
    dos_like_response = _aggregate_matrix(dos_k, k_axis=k_axis_c)
    energies = _energy_ledger(
        bands.eigvals,
        f_base,
        weights,
        mu=mu,
        kT=kT,
        spin_degeneracy=spin_degeneracy,
        k_axis=k_axis_c,
    )

    ne_from_d_k = _trace_last2(density_k @ s_arr)
    band_from_d_k = _trace_last2(density_k @ h_arr)
    ne_from_d = _aggregate_matrix(ne_from_d_k, k_axis=k_axis_c)
    band_from_d = _aggregate_matrix(band_from_d_k, k_axis=k_axis_c)
    conservation = ConservationLedger(
        electron_count_from_density=ne_from_d,
        band_energy_from_density=band_from_d,
        electron_count_residual=ne_from_d - electron_count,
        band_energy_residual=band_from_d - energies.band_energy,
        density_hermiticity_error=_density_hermiticity_error(density),
    )

    return FixedMuResult(
        mu=mu,
        kT=kT,
        spin_degeneracy=spin_degeneracy,
        eigvals=bands.eigvals,
        eigvecs=bands.eigvecs,
        occupations=occupations,
        occupation_response=occupation_response,
        density_k=density_k,
        density_response_k=density_response_k,
        electron_count=electron_count,
        dos_like_response=dos_like_response,
        density=density,
        density_response=density_response,
        energies=energies,
        conservation=conservation,
        k_weights=weights,
        k_axis=k_axis_c,
        min_overlap_eig=bands.min_overlap_eig,
        overlap_condition=bands.overlap_condition,
    )


def validate_conservation(result: FixedMuResult, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    """Fail closed when density-matrix ledgers do not conserve N or band energy."""

    atol = _as_float_scalar("atol", atol, nonnegative=True)
    rtol = _as_float_scalar("rtol", rtol, nonnegative=True)
    cons = result.conservation
    if not np.allclose(cons.electron_count_from_density, result.electron_count, atol=atol, rtol=rtol):
        raise FixedMuOperatorError("Tr(D S) does not match N(mu)")
    if not np.allclose(cons.band_energy_from_density, result.energies.band_energy, atol=atol, rtol=rtol):
        raise FixedMuOperatorError("Tr(D H) does not match the band-energy ledger")
    if np.any(cons.density_hermiticity_error > atol + rtol):
        raise FixedMuOperatorError("D(mu) is not Hermitian within tolerance")
