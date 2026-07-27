"""Fixed-chemical-potential dense H/S postprocess reference utilities.

The routines here are intentionally small NumPy/SciPy CPU reference kernels.
They take predicted dense DPTB Hamiltonian/overlap matrices, solve the
generalized eigenproblem, evaluate fixed-mu Fermi occupations, and return
density-matrix and energy ledgers suitable for downstream conservation checks.
They do not preserve PyTorch autograd graphs.

Units
-----
``mu``, ``kT`` and ``h`` must share one caller-chosen energy unit; nothing in
this module converts between units or records which one was meant.  Every
energy this module returns -- eigenvalues, the band/entropy/free/grand ledger,
``Tr(D H)`` -- comes back in that same unit, and ``density``/``electron_count``
are unitless occupancy quantities.
"""
from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # SciPy gives the most stable Hermitian generalized eigensolver.
    from scipy import linalg as _scipy_linalg
except Exception:  # pragma: no cover - exercised only in minimal runtimes.
    _scipy_linalg = None

ArrayLike = Any

_MACHINE_EPS = float(np.finfo(np.float64).eps)
#: The S-orthonormality and eigenpair residuals of a backward-stable generalized
#: solve grow like ``eps * cond(S)``, so an absolute atol alone cannot certify
#: the conditioning the acceptance gates admit.  Spectral checks therefore use
#: ``max(atol, this many ULPs of eps * cond(S) * scale)``.
_SPECTRAL_CONDITION_ULPS = 16.0
#: Occupancy multipliers this reference understands: 2 for spin-degenerate
#: non-SOC matrices, 1 for explicit spin/SOC spinor matrices.
_ALLOWED_SPIN_DEGENERACY = (1.0, 2.0)
_EXPECTED_UNSET = object()


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

    def __post_init__(self) -> None:
        _freeze_arrays(self, ("eigvals", "eigvecs", "min_overlap_eig", "overlap_condition"))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class EnergyLedger:
    """Single-particle band energetics at fixed chemical potential.

    Every term is a *band-structure* quantity: it comes from the occupied
    single-particle spectrum only, with no double-counting correction and no
    ion-ion contribution, and it carries whatever energy unit ``h`` and ``mu``
    were given in.  The names say so explicitly:

    ``minus_t_s``
        The finite-temperature ``-T*S`` contribution
        ``g * sum_k w_k * kT * [f log f + (1-f) log(1-f)]``, which is ``<= 0``.
    ``band_free_energy``
        ``band_energy + minus_t_s``.
    ``band_grand_energy``
        ``band_free_energy - mu * N``, the Legendre transform of the above.
    """

    band_energy: np.ndarray
    minus_t_s: np.ndarray
    band_free_energy: np.ndarray
    band_grand_energy: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(self, ("band_energy", "minus_t_s", "band_free_energy", "band_grand_energy"))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class ConservationLedger:
    """Hamiltonian/DM downstream conservation diagnostics.

    ``input_h_hermiticity_error`` / ``input_s_hermiticity_error`` are the raw
    ``max |X - X^H|`` of the matrices as supplied, recorded before
    symmetrization.  They cannot be recomputed from the stored H/S snapshot,
    which is already Hermitian by construction; the validator re-checks them
    against the recorded ``hermitian_tol`` instead.
    """

    electron_count_from_density: np.ndarray
    band_energy_from_density: np.ndarray
    electron_count_residual: np.ndarray
    band_energy_residual: np.ndarray
    density_hermiticity_error: np.ndarray
    input_h_hermiticity_error: np.ndarray
    input_s_hermiticity_error: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(
            self,
            (
                "electron_count_from_density",
                "band_energy_from_density",
                "electron_count_residual",
                "band_energy_residual",
                "density_hermiticity_error",
                "input_h_hermiticity_error",
                "input_s_hermiticity_error",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class FixedMuValidationContext:
    """Non-serialized immutable H/S snapshots for fixed-mu validation."""

    hamiltonian: np.ndarray
    overlap: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(self, ("hamiltonian", "overlap"))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class FixedMuResult:
    """Observable bundle for ``H, S`` evaluated at a fixed ``mu``.

    Array fields are defensive read-only copies backed by an immutable buffer,
    so writes cannot be re-enabled with ``setflags(write=True)``.
    Solver-produced results also carry private read-only H/S snapshots so
    conservation validation can recompute ``Tr(D S)`` and ``Tr(D H)`` instead of
    trusting cached ledgers.
    """

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
    normalize_k_weights: bool = True
    eig_floor: float = 1e-10
    max_condition: float = 1e12
    hermitian_tol: float = 1e-8
    validation_context: InitVar[Optional[FixedMuValidationContext]] = None

    def __post_init__(self, validation_context: Optional[FixedMuValidationContext]) -> None:
        if not isinstance(self.energies, EnergyLedger):
            raise FixedMuOperatorError("energies must be an EnergyLedger")
        if not isinstance(self.conservation, ConservationLedger):
            raise FixedMuOperatorError("conservation must be a ConservationLedger")
        object.__setattr__(self, "mu", _as_float_scalar("mu", self.mu))
        object.__setattr__(self, "kT", _as_float_scalar("kT", self.kT, nonnegative=True))
        object.__setattr__(
            self,
            "spin_degeneracy",
            _validate_spin_degeneracy(self.spin_degeneracy),
        )
        object.__setattr__(
            self,
            "normalize_k_weights",
            _as_bool("normalize_k_weights", self.normalize_k_weights),
        )
        object.__setattr__(self, "eig_floor", _as_float_scalar("eig_floor", self.eig_floor, positive=True))
        object.__setattr__(self, "max_condition", _as_float_scalar("max_condition", self.max_condition, positive=True))
        object.__setattr__(
            self,
            "hermitian_tol",
            _as_float_scalar("hermitian_tol", self.hermitian_tol, nonnegative=True),
        )
        for field_name in (
            "eigvals",
            "eigvecs",
            "occupations",
            "occupation_response",
            "density_k",
            "density_response_k",
            "electron_count",
            "dos_like_response",
            "density",
            "density_response",
            "k_weights",
            "min_overlap_eig",
            "overlap_condition",
        ):
            object.__setattr__(self, field_name, _readonly_array(getattr(self, field_name)))
        if validation_context is not None and not isinstance(validation_context, FixedMuValidationContext):
            raise FixedMuOperatorError("validation_context must be a FixedMuValidationContext")
        object.__setattr__(self, "_validation_context", validation_context)

    def __getstate__(self) -> Dict[str, Any]:
        return {field_info.name: getattr(self, field_info.name) for field_info in fields(self)}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self.__post_init__(None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _readonly_array(value: ArrayLike, *, dtype: Any = None) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True, order="C")
    immutable = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(arr.shape)
    immutable.setflags(write=False)
    return immutable


def _freeze_arrays(obj: Any, field_names: Tuple[str, ...]) -> None:
    # Called from both __post_init__ and __setstate__: pickle restores __dict__
    # directly, so the read-only flags would otherwise not survive a round trip.
    for field_name in field_names:
        object.__setattr__(obj, field_name, _readonly_array(getattr(obj, field_name)))


def hermitian_part(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    return 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))


def _looks_like_torch_tensor(x: Any) -> bool:
    typ = type(x)
    module = getattr(typ, "__module__", "")
    return module == "torch" or module.startswith("torch.") or (
        hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy")
    )


def _reject_torch_tensor(name: str, x: Any) -> None:
    if _looks_like_torch_tensor(x):
        raise FixedMuOperatorError(
            f"{name} looks like a torch Tensor. This module is a NumPy CPU "
            "postprocess reference and will not detach/cpu tensors silently. "
            "Use fixed_mu_observables_from_torch(..., detach=True) explicitly "
            "or pass tensor.detach().cpu().numpy()."
        )


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise FixedMuOperatorError(f"{name} must be a boolean, got {value!r}")
    return bool(value)


def _as_float_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    _reject_torch_tensor(name, value)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FixedMuOperatorError(f"{name} must be a scalar float, got {value!r}") from exc
    if not np.isfinite(out):
        raise FixedMuOperatorError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise FixedMuOperatorError(f"{name} must be positive, got {value!r}")
    if nonnegative and out < 0.0:
        raise FixedMuOperatorError(f"{name} must be nonnegative, got {value!r}")
    return out


def _validate_spin_degeneracy(spin_degeneracy: float) -> float:
    value = _as_float_scalar("spin_degeneracy", spin_degeneracy, positive=True)
    if value not in _ALLOWED_SPIN_DEGENERACY:
        raise FixedMuOperatorError(
            f"spin_degeneracy must be one of {_ALLOWED_SPIN_DEGENERACY} (2 for spin-degenerate "
            f"non-SOC matrices, 1 for explicit spin/SOC spinor matrices), got {value!r}"
        )
    return value


def _validate_matrix_stack(
    h: ArrayLike,
    s: ArrayLike,
    *,
    hermitian_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _reject_torch_tensor("H", h)
    _reject_torch_tensor("S", s)
    h_arr = np.asarray(h)
    s_arr = np.asarray(s)
    if h_arr.shape != s_arr.shape:
        raise FixedMuOperatorError(f"H and S shapes must match, got {h_arr.shape} and {s_arr.shape}")
    if h_arr.ndim < 2 or h_arr.shape[-1] != h_arr.shape[-2]:
        raise FixedMuOperatorError(f"H/S must be square matrices or stacks, got shape {h_arr.shape}")
    if h_arr.shape[-1] == 0:
        raise FixedMuOperatorError("H/S matrix size must be positive")
    if 0 in h_arr.shape[:-2]:
        raise FixedMuOperatorError(
            f"H/S leading shape {h_arr.shape[:-2]} must not contain a zero-length dimension"
        )
    if not np.issubdtype(h_arr.dtype, np.number) or not np.issubdtype(s_arr.dtype, np.number):
        raise FixedMuOperatorError("H and S must be numeric arrays")
    dtype = np.result_type(h_arr, s_arr, np.float64)
    h_arr = np.asarray(h_arr, dtype=dtype)
    s_arr = np.asarray(s_arr, dtype=dtype)
    if not np.isfinite(h_arr).all() or not np.isfinite(s_arr).all():
        raise FixedMuOperatorError("H and S must contain only finite values")
    hermitian_tol = _as_float_scalar("hermitian_tol", hermitian_tol, nonnegative=True)
    # Recorded before symmetrization: hermitian_part() below destroys it.
    h_asymmetry = np.max(np.abs(h_arr - np.swapaxes(h_arr.conj(), -1, -2)), axis=(-1, -2)).real
    s_asymmetry = np.max(np.abs(s_arr - np.swapaxes(s_arr.conj(), -1, -2)), axis=(-1, -2)).real
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
    return (
        hermitian_part(h_arr),
        hermitian_part(s_arr),
        np.asarray(h_asymmetry, dtype=np.float64),
        np.asarray(s_asymmetry, dtype=np.float64),
    )


def _hermiticity_budget(matrix: np.ndarray, hermitian_tol: float) -> np.ndarray:
    """Largest asymmetry ``np.allclose(X, X^H, atol=tol, rtol=tol)`` can admit."""

    scale = np.max(np.abs(matrix), axis=(-1, -2)).real
    return hermitian_tol * (1.0 + np.asarray(scale, dtype=np.float64))


def _canonical_k_axis(k_axis: Optional[int], leading_ndim: int) -> Optional[int]:
    if k_axis is None:
        return None
    if leading_ndim == 0:
        raise FixedMuOperatorError("k_axis is invalid for a single matrix with no leading k dimension")
    if isinstance(k_axis, bool) or not isinstance(k_axis, (int, np.integer)):
        raise FixedMuOperatorError(f"k_axis must be an integer, got {k_axis!r}")
    try:
        axis = int(k_axis)
    except (TypeError, ValueError) as exc:
        raise FixedMuOperatorError(f"k_axis must be an integer, got {k_axis!r}") from exc
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
    if k_axis is None and k_weights is not None:
        _reject_torch_tensor("k_weights", k_weights)
        raise FixedMuOperatorError(
            "k_weights requires an explicit k_axis; batch/sample weights are "
            "not accepted by this fixed-mu postprocess reference"
        )
    if not leading_shape:
        if k_weights is None:
            return np.asarray(1.0, dtype=np.float64)
        _reject_torch_tensor("k_weights", k_weights)
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape not in [(), (1,)]:
            raise FixedMuOperatorError("single-matrix k_weights must be scalar")
        weight = float(weights.reshape(-1)[0])
        if not np.isfinite(weight):
            raise FixedMuOperatorError("single-matrix k_weights must be finite")
        if weight < 0.0:
            raise FixedMuOperatorError("single-matrix k_weights must be nonnegative")
        return np.asarray(weight, dtype=np.float64)

    if k_weights is None:
        weights = np.ones(leading_shape, dtype=np.float64)
    else:
        _reject_torch_tensor("k_weights", k_weights)
        raw = np.asarray(k_weights, dtype=np.float64)
        if k_axis is not None and raw.ndim == 1:
            if raw.shape[0] != leading_shape[k_axis]:
                raise FixedMuOperatorError(
                    f"1D k_weights length must equal leading_shape[k_axis]="
                    f"{leading_shape[k_axis]}, got {raw.shape[0]}"
                )
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


def _validate_stored_weights(
    leading_shape: Tuple[int, ...],
    *,
    k_weights: ArrayLike,
    k_axis: Optional[int],
    normalize_k_weights: bool,
    atol: float,
    rtol: float,
) -> np.ndarray:
    _reject_torch_tensor("k_weights", k_weights)
    try:
        weights = np.asarray(k_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FixedMuOperatorError("k_weights must be numeric") from exc
    if not leading_shape:
        if weights.shape not in [(), (1,)]:
            raise FixedMuOperatorError("single-matrix stored k_weights must be scalar")
        weights = np.asarray(float(weights.reshape(-1)[0]), dtype=np.float64)
    elif weights.shape != leading_shape:
        raise FixedMuOperatorError(
            f"stored k_weights shape {weights.shape} must match leading H/S shape {leading_shape}"
        )

    if not np.isfinite(weights).all():
        raise FixedMuOperatorError("k_weights must be finite")
    if np.any(weights < 0.0):
        raise FixedMuOperatorError("k_weights must be nonnegative")
    if normalize_k_weights and k_axis is not None:
        total = weights.sum(axis=k_axis, keepdims=True)
        if np.any(total <= 0.0):
            raise FixedMuOperatorError("k_weights must have positive sum along k_axis")
        if not np.allclose(total, np.ones_like(total), atol=atol, rtol=rtol):
            raise FixedMuOperatorError("normalized stored k_weights must sum to one along k_axis")
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
    h_arr, s_arr, _, _ = _validate_matrix_stack(h, s, hermitian_tol=hermitian_tol)
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
    spin_degeneracy = _validate_spin_degeneracy(spin_degeneracy)
    _reject_torch_tensor("energies", energies)
    eps = np.asarray(energies, dtype=np.float64)
    if not np.isfinite(eps).all():
        raise FixedMuOperatorError("energies must be finite")
    f, response = _stable_fermi(eps, mu=mu, kT=kT)
    return spin_degeneracy * f, spin_degeneracy * response


def _density_from_vectors(vecs: np.ndarray, occ: np.ndarray) -> np.ndarray:
    return np.einsum("...ni,...i,...mi->...nm", vecs, occ, vecs.conj(), optimize=True)


def _aggregate_matrix(values: np.ndarray, *, k_axis: Optional[int], weights: Optional[np.ndarray] = None) -> np.ndarray:
    if k_axis is None:
        return values
    if weights is not None:
        while weights.ndim < values.ndim:
            weights = weights[..., None]
        values = weights * values
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
        minus_t_s=_aggregate_matrix(entropy_k, k_axis=k_axis),
        band_free_energy=_aggregate_matrix(free_k, k_axis=k_axis),
        band_grand_energy=_aggregate_matrix(grand_k, k_axis=k_axis),
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
        ``k_weights`` is accepted only with an explicit ``k_axis`` and is then
        normalized along that axis.  With ``k_axis=None`` no k aggregation is
        performed, batch/sample weights are rejected, and returned densities
        have the same unweighted leading shape as the input matrices.
    normalize_k_weights
        Must be a real ``bool``/``np.bool_``.  Anything merely truthy is
        rejected rather than coerced, because ``bool("false") is True`` would
        silently invert the weight convention at a JSON/YAML boundary.
    """

    mu = _as_float_scalar("mu", mu)
    kT = _as_float_scalar("kT", kT, nonnegative=True)
    spin_degeneracy = _validate_spin_degeneracy(spin_degeneracy)
    normalize_k_weights = _as_bool("normalize_k_weights", normalize_k_weights)
    bands = generalized_bands(
        h,
        s,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    h_arr, s_arr, h_asymmetry, s_asymmetry = _validate_matrix_stack(h, s, hermitian_tol=hermitian_tol)
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
    density_k = _density_from_vectors(bands.eigvecs, occupations)
    density_response_k = _density_from_vectors(bands.eigvecs, occupation_response)
    density = _aggregate_matrix(density_k, k_axis=k_axis_c, weights=weights)
    density_response = _aggregate_matrix(density_response_k, k_axis=k_axis_c, weights=weights)

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

    ne_from_d_k = weights * _trace_last2(density_k @ s_arr)
    band_from_d_k = weights * _trace_last2(density_k @ h_arr)
    ne_from_d = _aggregate_matrix(ne_from_d_k, k_axis=k_axis_c)
    band_from_d = _aggregate_matrix(band_from_d_k, k_axis=k_axis_c)
    conservation = ConservationLedger(
        electron_count_from_density=ne_from_d,
        band_energy_from_density=band_from_d,
        electron_count_residual=ne_from_d - electron_count,
        band_energy_residual=band_from_d - energies.band_energy,
        density_hermiticity_error=_density_hermiticity_error(density),
        input_h_hermiticity_error=h_asymmetry,
        input_s_hermiticity_error=s_asymmetry,
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
        normalize_k_weights=normalize_k_weights,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        validation_context=FixedMuValidationContext(h_arr, s_arr),
    )


def _assert_recomputed_close(
    name: str,
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    atol: ArrayLike,
    rtol: float,
) -> None:
    if stored.shape != recomputed.shape:
        raise FixedMuOperatorError(
            f"fixed-mu stored {name} has shape {stored.shape}, recomputed {recomputed.shape}"
        )
    if not np.all(np.isclose(stored, recomputed, atol=atol, rtol=rtol)):
        raise FixedMuOperatorError(f"fixed-mu stored {name} does not match recomputed value")


def validate_conservation(
    result: FixedMuResult,
    *,
    h: Optional[ArrayLike] = None,
    s: Optional[ArrayLike] = None,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    expected_mu: Optional[float] = None,
    expected_kT: Optional[float] = None,
    expected_spin_degeneracy: Optional[float] = None,
    expected_k_weights: Optional[ArrayLike] = None,
    expected_k_axis: Any = _EXPECTED_UNSET,
    expected_normalize_k_weights: Any = _EXPECTED_UNSET,
    expected_eig_floor: Optional[float] = None,
    expected_max_condition: Optional[float] = None,
    expected_hermitian_tol: Optional[float] = None,
) -> None:
    """Fail closed when density-matrix ledgers do not conserve N or band energy.

    Validation recomputes the k-axis/weight contract, H/S safety checks,
    generalized-eigenpair residuals, density aggregation, ``Tr(D S)``,
    ``Tr(D H)``, residuals, and hermiticity from the current result arrays and
    either caller-supplied H/S arrays or the non-serialized private context
    carried by solver-produced results.  Externally constructed or deserialized
    results without H/S context must pass ``h=`` and ``s=`` explicitly.

    Parameters
    ----------
    expected_mu, expected_kT, expected_spin_degeneracy, expected_k_weights,
    expected_k_axis, expected_normalize_k_weights, expected_eig_floor,
    expected_max_condition, expected_hermitian_tol
        The request and safety policy the caller actually made.  Every conservation identity here
        is self-consistent in the stored scalars -- N, D and the whole energy
        ledger are homogeneous of degree one in ``spin_degeneracy``, so a result
        computed at the wrong SOC/non-SOC convention validates perfectly against
        itself.  Supplying these binds the result to the request instead.

    Notes
    -----
    The S-orthonormality and eigenpair residuals of a backward-stable
    generalized solve grow like ``eps * cond(S)``, which the default
    ``max_condition=1e12`` admits far beyond what ``atol=1e-8`` can certify.
    Those two checks therefore use ``max(atol, 16 * eps * cond(S) * scale)``
    **per leading item**, and the ``Tr(D S)`` / ``Tr(D H)`` budgets that route
    through the same eigenvectors are formed per k/sample item and aggregated
    only along the declared k axis.  A batch-global ``max(cond(S))`` would let
    one pathological overlap relax the certificate of every well-conditioned
    sample beside it, making batch composition decide whether a structure
    validates.  Every other check keeps the caller's ``atol``/``rtol`` verbatim.
    """

    if not isinstance(result, FixedMuResult):
        raise FixedMuOperatorError("result must be a FixedMuResult")
    if not isinstance(result.energies, EnergyLedger):
        raise FixedMuOperatorError("energies must be an EnergyLedger")
    if not isinstance(result.conservation, ConservationLedger):
        raise FixedMuOperatorError("conservation must be a ConservationLedger")
    atol = _as_float_scalar("atol", atol, nonnegative=True)
    rtol = _as_float_scalar("rtol", rtol, nonnegative=True)
    if expected_mu is not None and not np.isclose(
        result.mu, _as_float_scalar("expected_mu", expected_mu), atol=atol, rtol=rtol
    ):
        raise FixedMuOperatorError(
            f"fixed-mu result was computed at mu={result.mu!r}, not the expected {expected_mu!r}"
        )
    if expected_kT is not None and not np.isclose(
        result.kT, _as_float_scalar("expected_kT", expected_kT, nonnegative=True), atol=atol, rtol=rtol
    ):
        raise FixedMuOperatorError(
            f"fixed-mu result was computed at kT={result.kT!r}, not the expected {expected_kT!r}"
        )
    if expected_spin_degeneracy is not None and result.spin_degeneracy != _validate_spin_degeneracy(
        expected_spin_degeneracy
    ):
        raise FixedMuOperatorError(
            f"fixed-mu result was computed at spin_degeneracy={result.spin_degeneracy!r}, not the "
            f"expected {expected_spin_degeneracy!r}; N, D and the energy ledger all scale linearly "
            "with it, so nothing else in this validator can catch the wrong convention"
        )
    if expected_eig_floor is not None:
        expected = _as_float_scalar("expected_eig_floor", expected_eig_floor, positive=True)
        if result.eig_floor != expected:
            raise FixedMuOperatorError(
                f"fixed-mu result used eig_floor={result.eig_floor!r}, not the expected {expected!r}"
            )
    if expected_max_condition is not None:
        expected = _as_float_scalar(
            "expected_max_condition", expected_max_condition, positive=True
        )
        if result.max_condition != expected:
            raise FixedMuOperatorError(
                f"fixed-mu result used max_condition={result.max_condition!r}, "
                f"not the expected {expected!r}"
            )
    if expected_hermitian_tol is not None:
        expected = _as_float_scalar(
            "expected_hermitian_tol", expected_hermitian_tol, nonnegative=True
        )
        if result.hermitian_tol != expected:
            raise FixedMuOperatorError(
                f"fixed-mu result used hermitian_tol={result.hermitian_tol!r}, "
                f"not the expected {expected!r}"
            )
    if (h is None) != (s is None):
        raise FixedMuOperatorError("validate_conservation requires both h and s, or neither")
    if h is None and s is None:
        context = getattr(result, "_validation_context", None)
        if context is None:
            raise FixedMuOperatorError(
                "FixedMuResult is missing H/S context; pass h= and s= for conservation validation"
            )
        h = context.hamiltonian
        s = context.overlap
    h_arr, s_arr, _, _ = _validate_matrix_stack(h, s, hermitian_tol=result.hermitian_tol)
    min_overlap_eig, overlap_condition = _check_overlap(
        s_arr,
        eig_floor=result.eig_floor,
        max_condition=result.max_condition,
    )
    leading_shape = h_arr.shape[:-2]
    k_axis_c = _canonical_k_axis(result.k_axis, len(leading_shape))
    if expected_k_axis is not _EXPECTED_UNSET:
        requested_k_axis = _canonical_k_axis(expected_k_axis, len(leading_shape))
        if k_axis_c != requested_k_axis:
            raise FixedMuOperatorError(
                f"fixed-mu result used k_axis={k_axis_c!r}, not the expected {requested_k_axis!r}"
            )
    if expected_normalize_k_weights is not _EXPECTED_UNSET:
        requested_normalize = _as_bool(
            "expected_normalize_k_weights", expected_normalize_k_weights
        )
        if result.normalize_k_weights != requested_normalize:
            raise FixedMuOperatorError(
                "fixed-mu result used normalize_k_weights="
                f"{result.normalize_k_weights!r}, not the expected {requested_normalize!r}"
            )
    weights = _validate_stored_weights(
        leading_shape,
        k_weights=result.k_weights,
        k_axis=k_axis_c,
        normalize_k_weights=result.normalize_k_weights,
        atol=atol,
        rtol=rtol,
    )
    if expected_k_weights is not None:
        _reject_torch_tensor("expected_k_weights", expected_k_weights)
        if k_axis_c is None:
            # _prepare_weights refuses weights without a k axis, so a result that
            # legitimately carries the implicit unit weights has to be compared
            # directly; there is no normalization to route through either.
            requested = _validate_stored_weights(
                leading_shape,
                k_weights=expected_k_weights,
                k_axis=None,
                normalize_k_weights=False,
                atol=atol,
                rtol=rtol,
            )
        else:
            requested = _prepare_weights(
                leading_shape,
                k_weights=expected_k_weights,
                k_axis=k_axis_c,
                normalize_k_weights=result.normalize_k_weights,
            )
        if requested.shape != weights.shape or not np.allclose(requested, weights, atol=atol, rtol=rtol):
            raise FixedMuOperatorError("fixed-mu stored k_weights do not match the expected k_weights")

    n = h_arr.shape[-1]
    eigvals = np.asarray(result.eigvals)
    eigvecs = np.asarray(result.eigvecs)
    density_k = np.asarray(result.density_k)
    # Without this the S-orthonormality identity below is sized from the stored
    # eigenvector column count, so a truncated m < n basis certifies itself.
    if eigvecs.shape != leading_shape + (n, n):
        raise FixedMuOperatorError(
            f"fixed-mu eigvecs must be a complete {n}x{n} eigenbasis matching H/S, "
            f"got shape {eigvecs.shape}"
        )
    if eigvals.shape != leading_shape + (n,):
        raise FixedMuOperatorError(
            f"fixed-mu eigvals must have shape {leading_shape + (n,)}, got {eigvals.shape}"
        )
    # Keep conditioning budgets local to each leading item.  A single pathological
    # overlap matrix must not relax validation for unrelated well-conditioned
    # samples in the same batch.
    conditioning_floor = (
        _SPECTRAL_CONDITION_ULPS * _MACHINE_EPS * np.asarray(overlap_condition)
    )
    h_scale = np.maximum(np.max(np.abs(h_arr), axis=(-1, -2)), 1.0)
    spectral_atol = np.maximum(atol, conditioning_floor)[..., None, None]
    eigen_atol = np.maximum(atol, conditioning_floor * h_scale)[..., None, None]
    try:
        f_base, response_base = _stable_fermi(eigvals, mu=result.mu, kT=result.kT)
        occupations = result.spin_degeneracy * f_base
        occupation_response = result.spin_degeneracy * response_base
        s_orthonormal = np.swapaxes(eigvecs.conj(), -1, -2) @ s_arr @ eigvecs
        identity = np.broadcast_to(np.eye(n, dtype=s_orthonormal.dtype), s_orthonormal.shape)
        h_c = h_arr @ eigvecs
        s_c_eps = (s_arr @ eigvecs) * eigvals[..., None, :]
        density_k_from_vectors = _density_from_vectors(eigvecs, occupations)
        density_response_k_from_vectors = _density_from_vectors(eigvecs, occupation_response)
        aggregated_density_from_vectors = _aggregate_matrix(
            density_k_from_vectors,
            k_axis=k_axis_c,
            weights=weights,
        )
        aggregated_density_response = _aggregate_matrix(
            density_response_k_from_vectors,
            k_axis=k_axis_c,
            weights=weights,
        )
        electron_count_from_occupations = _aggregate_matrix(
            weights * np.sum(occupations, axis=-1),
            k_axis=k_axis_c,
        )
        dos_like_response = _aggregate_matrix(
            weights * np.sum(occupation_response, axis=-1),
            k_axis=k_axis_c,
        )
        energy_ledger = _energy_ledger(
            eigvals,
            f_base,
            weights,
            mu=result.mu,
            kT=result.kT,
            spin_degeneracy=result.spin_degeneracy,
            k_axis=k_axis_c,
        )
        ne_from_d_k = weights * _trace_last2(density_k @ s_arr)
        band_from_d_k = weights * _trace_last2(density_k @ h_arr)
        ne_from_d = _aggregate_matrix(ne_from_d_k, k_axis=k_axis_c)
        band_from_d = _aggregate_matrix(band_from_d_k, k_axis=k_axis_c)

        count_scale_k = np.maximum(np.sum(np.abs(occupations), axis=-1), 1.0)
        count_floor_k = weights * conditioning_floor * count_scale_k
        count_atol = np.maximum(
            atol, _aggregate_matrix(count_floor_k, k_axis=k_axis_c)
        )
        band_scale_k = np.maximum(
            np.sum(np.abs(eigvals * occupations), axis=-1),
            h_scale * count_scale_k,
        )
        band_floor_k = weights * conditioning_floor * band_scale_k
        band_atol = np.maximum(
            atol, _aggregate_matrix(band_floor_k, k_axis=k_axis_c)
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise FixedMuOperatorError("fixed-mu result arrays are not shape-compatible for conservation validation") from exc

    _assert_recomputed_close(
        "min_overlap_eig",
        np.asarray(result.min_overlap_eig),
        min_overlap_eig,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "overlap_condition",
        np.asarray(result.overlap_condition),
        overlap_condition,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "generalized S-orthonormality",
        s_orthonormal,
        identity,
        atol=spectral_atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "generalized eigen residual",
        h_c,
        s_c_eps,
        atol=eigen_atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "k_weights",
        np.asarray(result.k_weights),
        weights,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "occupations",
        np.asarray(result.occupations),
        occupations,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "occupation_response",
        np.asarray(result.occupation_response),
        occupation_response,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "density_k",
        density_k,
        density_k_from_vectors,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "density_response_k",
        np.asarray(result.density_response_k),
        density_response_k_from_vectors,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "electron_count",
        np.asarray(result.electron_count),
        electron_count_from_occupations,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "dos_like_response",
        np.asarray(result.dos_like_response),
        dos_like_response,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "density_response",
        np.asarray(result.density_response),
        aggregated_density_response,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy band_energy",
        np.asarray(result.energies.band_energy),
        energy_ledger.band_energy,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy minus_t_s",
        np.asarray(result.energies.minus_t_s),
        energy_ledger.minus_t_s,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy band_free_energy",
        np.asarray(result.energies.band_free_energy),
        energy_ledger.band_free_energy,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy band_grand_energy",
        np.asarray(result.energies.band_grand_energy),
        energy_ledger.band_grand_energy,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy band_free_energy closure",
        np.asarray(result.energies.band_free_energy),
        np.asarray(result.energies.band_energy) + np.asarray(result.energies.minus_t_s),
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "energy band_grand_energy closure",
        np.asarray(result.energies.band_grand_energy),
        np.asarray(result.energies.band_free_energy) - result.mu * np.asarray(result.electron_count),
        atol=atol,
        rtol=rtol,
    )

    electron_residual = ne_from_d - result.electron_count
    band_residual = band_from_d - result.energies.band_energy
    density_hermiticity_error = _density_hermiticity_error(result.density)

    # Compared against the from-eigenvector route, so this does not inherit the
    # stored density_k the way a re-aggregation of it would.
    _assert_recomputed_close(
        "density",
        np.asarray(result.density),
        aggregated_density_from_vectors,
        atol=atol,
        rtol=rtol,
    )
    cons = result.conservation
    _assert_recomputed_close(
        "conservation electron_count_from_density",
        np.asarray(cons.electron_count_from_density),
        ne_from_d,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "conservation band_energy_from_density",
        np.asarray(cons.band_energy_from_density),
        band_from_d,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "conservation electron_count_residual",
        np.asarray(cons.electron_count_residual),
        electron_residual,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "conservation band_energy_residual",
        np.asarray(cons.band_energy_residual),
        band_residual,
        atol=atol,
        rtol=rtol,
    )
    _assert_recomputed_close(
        "conservation density_hermiticity_error",
        np.asarray(cons.density_hermiticity_error),
        density_hermiticity_error,
        atol=atol,
        rtol=rtol,
    )

    # The stored H/S snapshot is already Hermitian, so the raw input asymmetry
    # is unrecoverable here; the policy it was accepted under is re-enforced.
    for name, recorded, budget in (
        ("input_h_hermiticity_error", cons.input_h_hermiticity_error, _hermiticity_budget(h_arr, result.hermitian_tol)),
        ("input_s_hermiticity_error", cons.input_s_hermiticity_error, _hermiticity_budget(s_arr, result.hermitian_tol)),
    ):
        recorded_arr = np.asarray(recorded, dtype=np.float64)
        if recorded_arr.shape != leading_shape:
            raise FixedMuOperatorError(
                f"fixed-mu {name} has shape {recorded_arr.shape}, expected {leading_shape}"
            )
        if np.any(~np.isfinite(recorded_arr)) or np.any(recorded_arr < 0.0):
            raise FixedMuOperatorError(f"fixed-mu {name} must be finite and nonnegative")
        if np.any(recorded_arr > budget):
            raise FixedMuOperatorError(
                f"fixed-mu {name} {float(np.max(recorded_arr)):.6g} exceeds what the recorded "
                f"hermitian_tol={result.hermitian_tol:g} admits"
            )

    if not np.all(np.isclose(ne_from_d, result.electron_count, atol=count_atol, rtol=rtol)):
        raise FixedMuOperatorError("Tr(D S) does not match N(mu)")
    if not np.all(
        np.isclose(band_from_d, result.energies.band_energy, atol=band_atol, rtol=rtol)
    ):
        raise FixedMuOperatorError("Tr(D H) does not match the band-energy ledger")
    if np.any(density_hermiticity_error > atol + rtol):
        raise FixedMuOperatorError("D(mu) is not Hermitian within tolerance")


def fixed_mu_observables_from_torch(h: Any, s: Any, *, detach: bool = False, **kwargs: Any) -> FixedMuResult:
    """Explicit PyTorch-to-NumPy CPU wrapper for postprocess use.

    This wrapper always breaks autograd.  Callers must set ``detach=True`` to
    acknowledge the detach/cpu/numpy conversion; otherwise a fail-closed error
    is raised.
    """

    if not detach:
        raise FixedMuOperatorError(
            "fixed_mu_observables_from_torch breaks autograd by converting to "
            "NumPy CPU arrays; call with detach=True explicitly."
        )
    for name, value in [("H", h), ("S", s)]:
        if not (hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")):
            raise FixedMuOperatorError(f"{name} must be a torch Tensor-like object for this wrapper")
    h_np = h.detach().cpu().numpy()
    s_np = s.detach().cpu().numpy()
    converted_kwargs = {}
    for key, value in kwargs.items():
        if _looks_like_torch_tensor(value):
            converted_kwargs[key] = value.detach().cpu().numpy()
        else:
            converted_kwargs[key] = value
    return fixed_mu_observables(h_np, s_np, **converted_kwargs)
