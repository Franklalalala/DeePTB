"""Dense charge-equilibration (QEq) reference utilities.

This module is an independent NumPy CPU reference for the constrained QEq
problem

    E(q) = chi.T @ q + 0.5 * q.T @ J @ q,  sum(q) = total_charge.

It is deliberately separate from the fixed-mu H/S operator: QEq equilibrates a
finite charge vector under a fixed total-charge constraint, while fixed-mu
postprocessing evaluates occupations and density matrices from Hamiltonian and
overlap matrices at a chemical potential.

Units
-----
No conversion is performed anywhere in this module.  ``electronegativity`` and
``hardness_kernel`` are consumed exactly as given, and every derived quantity
(energies, Lagrange multiplier, residuals) comes out in the energy unit implied
by those inputs.  :class:`QEqUnits` is a *declarative label only*: it records
the canonical eV/e convention the DeePTB callers use and is never checked
against the numbers, so passing Hartree-valued inputs yields Hartree-valued
energies still labelled ``eV``.  Convert before calling if you need eV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

ArrayLike = Any

#: Absolute conditioning ceiling this reference is willing to certify.  A
#: recorded or caller-supplied policy may only tighten it, never loosen it.
MODULE_MAX_CONDITION = 1.0e12
#: Absolute ceiling on the recorded stationarity/residual tolerance.
MODULE_MAX_RESIDUAL_TOL = 1.0e-6
#: Absolute ceiling on the recorded kernel-asymmetry tolerance.
MODULE_MAX_SYMMETRY_TOL = 1.0e-6

_MACHINE_EPS = float(np.finfo(np.float64).eps)
#: Residual gates are absolute in the caller's energy unit, but no float64
#: evaluation of ``chi + J q + lambda`` can resolve below the rounding scale of
#: its own intermediates.  Gates therefore admit ``atol`` plus this many ULPs of
#: that arithmetic scale, which is what makes them invariant under the physical
#: gauge freedom ``J -> J + alpha * 11^T``.
_ARITHMETIC_FLOOR_ULPS = 64.0
#: Relative agreement required when a stored magnitude-carrying diagnostic
#: (condition number, eigenvalue, energy) is compared against its recomputation.
_RECOMPUTE_RTOL = 1.0e-9


class QEqOperatorError(ValueError):
    """Base class for fail-closed QEq validation errors."""


class QEqKernelConditionError(QEqOperatorError):
    """Raised when the constrained QEq kernel is unsafe to solve."""


@dataclass(frozen=True)
class QEqUnits:
    """Declarative unit labels for the dense QEq reference kernel.

    This type performs no conversion and is never cross-checked against the
    numbers.  It exists to state, in the result payload, that DeePTB feeds this
    reference eV/e-valued electronegativities and eV/e^2-valued hardness
    kernels; callers working in another consistent energy unit get their own
    unit back with this label unchanged.
    """

    charge: str = "e"
    energy: str = "eV"
    electronegativity: str = "eV/e"
    hardness_kernel: str = "eV/e^2"
    lagrange_multiplier: str = "eV/e"

    def __post_init__(self) -> None:
        expected = {
            "charge": "e",
            "energy": "eV",
            "electronegativity": "eV/e",
            "hardness_kernel": "eV/e^2",
            "lagrange_multiplier": "eV/e",
        }
        for field_name, canonical in expected.items():
            if getattr(self, field_name) != canonical:
                raise QEqOperatorError(
                    "QEqUnits only records the canonical unscaled operator "
                    f"units; {field_name} must be {canonical!r}"
                )


_DIAGNOSTIC_ARRAY_FIELDS = (
    "charge_residual",
    "stationarity_max_abs",
    "stationarity_l2",
    "stationarity_tangent_max_abs",
    "multiplier_residual",
    "energy_identity_residual",
    "kernel_symmetry_error",
    "input_kernel_symmetry_error",
    "constrained_min_eig",
    "constrained_max_eig",
    "constrained_condition",
    "kkt_condition",
)


@dataclass(frozen=True)
class QEqDiagnostics:
    """Constraint, stationarity, identity, and conditioning diagnostics.

    ``stationarity_max_abs`` / ``stationarity_l2`` describe the raw KKT vector
    ``chi + J q + lambda 1``, which is gauge-dependent: adding ``alpha * 11^T``
    to ``J`` leaves the physical solution untouched but injects ``alpha`` into
    every intermediate.  The gauge-invariant split is
    ``stationarity_tangent_max_abs`` (the sum-zero component ``Z^T (chi + J q)``,
    i.e. the equation the solve actually enforces) and ``multiplier_residual``
    (the uniform component, i.e. the equation that *defines* ``lambda``).

    ``input_kernel_symmetry_error`` is the asymmetry of the kernel as supplied,
    measured before symmetrization; ``kernel_symmetry_error`` is the residual
    asymmetry of the stored, already-symmetrized kernel and is therefore
    identically zero for any accepted input.

    ``energy_identity_residual`` is derived telemetry, not an independent
    certificate: ``energy - energy_identity == 0.5 * q . stationarity -
    0.5 * lambda * (sum(q) - Q)`` holds identically, so it can only be nonzero
    when the charge or stationarity residual is nonzero as well.
    """

    charge_residual: np.ndarray
    stationarity_max_abs: np.ndarray
    stationarity_l2: np.ndarray
    stationarity_tangent_max_abs: np.ndarray
    multiplier_residual: np.ndarray
    energy_identity_residual: np.ndarray
    kernel_symmetry_error: np.ndarray
    input_kernel_symmetry_error: np.ndarray
    constrained_min_eig: np.ndarray
    constrained_max_eig: np.ndarray
    constrained_condition: np.ndarray
    kkt_condition: np.ndarray

    def __post_init__(self) -> None:
        for field_name in _DIAGNOSTIC_ARRAY_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _readonly_array(field_name, getattr(self, field_name), dtype=np.float64),
            )

    def __setstate__(self, state: dict) -> None:
        # dataclass/pickle restores __dict__ directly, so the read-only flags
        # set in __post_init__ would otherwise be dropped on unpickling.
        self.__dict__.update(state)
        for field_name in _DIAGNOSTIC_ARRAY_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _readonly_array(field_name, getattr(self, field_name), dtype=np.float64),
            )


_RESULT_ARRAY_FIELDS = (
    "charges",
    "total_charge",
    "electronegativity",
    "hardness_kernel",
    "lagrange_multiplier",
    "stationarity",
    "energy",
    "linear_energy",
    "quadratic_energy",
    "energy_identity",
)


@dataclass(frozen=True)
class QEqResult:
    """Observable bundle for the dense constrained QEq solve.

    Array fields are defensive read-only copies.  This prevents in-place alias
    mutations from making cached diagnostics disagree with the physical fields.

    ``symmetry_tol`` / ``constrained_eig_floor`` / ``max_condition`` /
    ``residual_tol`` persist the safety policy of the solve that produced this
    result so a downstream consumer can re-enforce it.  They are self-declared
    data: :func:`validate_qeq_result` clamps them against the module-level
    ``MODULE_MAX_*`` ceilings, so a serialized payload can tighten the shipped
    policy but never loosen it.
    """

    charges: np.ndarray
    total_charge: np.ndarray
    electronegativity: np.ndarray
    hardness_kernel: np.ndarray
    lagrange_multiplier: np.ndarray
    stationarity: np.ndarray
    energy: np.ndarray
    linear_energy: np.ndarray
    quadratic_energy: np.ndarray
    energy_identity: np.ndarray
    diagnostics: QEqDiagnostics
    units: QEqUnits
    symmetry_tol: float = 1e-10
    constrained_eig_floor: float = 1e-12
    max_condition: float = 1e12
    residual_tol: float = 1e-8

    def __post_init__(self) -> None:
        self._normalize()

    def __setstate__(self, state: dict) -> None:
        # dataclass/pickle restores __dict__ directly, so the read-only flags
        # set in __post_init__ would otherwise be dropped on unpickling.
        self.__dict__.update(state)
        self._normalize()

    def _normalize(self) -> None:
        object.__setattr__(self, "symmetry_tol", _as_float_scalar("symmetry_tol", self.symmetry_tol, nonnegative=True))
        object.__setattr__(self, "constrained_eig_floor", _as_float_scalar("constrained_eig_floor", self.constrained_eig_floor, positive=True))
        object.__setattr__(self, "max_condition", _as_float_scalar("max_condition", self.max_condition, positive=True))
        object.__setattr__(self, "residual_tol", _as_float_scalar("residual_tol", self.residual_tol, nonnegative=True))
        for field_name in _RESULT_ARRAY_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _readonly_array(field_name, getattr(self, field_name), dtype=np.float64),
            )


def _readonly_array(name: str, value: ArrayLike, *, dtype: Any = None) -> np.ndarray:
    _reject_torch_tensor(name, value)
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _looks_like_torch_tensor(x: Any) -> bool:
    typ = type(x)
    module = getattr(typ, "__module__", "")
    return module == "torch" or module.startswith("torch.") or (
        hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy")
    )


def _reject_torch_tensor(name: str, x: Any) -> None:
    if _looks_like_torch_tensor(x):
        raise QEqOperatorError(
            f"{name} looks like a torch Tensor. This NumPy QEq reference will "
            "not detach/cpu tensors silently. Pass explicit NumPy arrays, or "
            "add a differentiable torch QEq backend with a separate API."
        )


def _as_float_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    _reject_torch_tensor(name, value)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise QEqOperatorError(f"{name} must be a scalar float, got {value!r}") from exc
    if not np.isfinite(out):
        raise QEqOperatorError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise QEqOperatorError(f"{name} must be positive, got {value!r}")
    if nonnegative and out < 0.0:
        raise QEqOperatorError(f"{name} must be nonnegative, got {value!r}")
    return out


def _broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    if hasattr(np, "broadcast_shapes"):
        return np.broadcast_shapes(*shapes)
    arrays = [np.empty(shape, dtype=np.uint8) for shape in shapes]
    return np.broadcast(*arrays).shape


def _as_real_numeric_array(name: str, value: ArrayLike) -> np.ndarray:
    _reject_torch_tensor(name, value)
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise QEqOperatorError(f"{name} must be a numeric real array")
    if np.iscomplexobj(arr):
        raise QEqOperatorError(f"{name} must be real; complex QEq kernels are not supported")
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise QEqOperatorError(f"{name} must contain only finite values")
    return arr


def _prepare_total_charge(total_charge: ArrayLike, leading_shape: Tuple[int, ...]) -> np.ndarray:
    _reject_torch_tensor("total_charge", total_charge)
    arr = np.asarray(total_charge)
    if not np.issubdtype(arr.dtype, np.number):
        raise QEqOperatorError("total_charge must be numeric")
    if np.iscomplexobj(arr):
        raise QEqOperatorError("total_charge must be real")
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise QEqOperatorError("total_charge must be finite")
    if not leading_shape and arr.size == 1:
        return np.asarray(float(arr.reshape(-1)[0]), dtype=np.float64)
    try:
        return np.broadcast_to(arr, leading_shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise QEqOperatorError(
            f"total_charge shape {arr.shape} cannot broadcast to leading shape {leading_shape}"
        ) from exc


def _clamp_policy(name: str, value: float, ceiling: float) -> float:
    """Return the tighter of a self-declared policy value and the module ceiling."""

    if value > ceiling:
        raise QEqOperatorError(
            f"{name}={value:g} exceeds the module ceiling {ceiling:g}; this dense "
            "reference does not certify results at that policy"
        )
    return value


def _validate_qeq_inputs(
    electronegativity: ArrayLike,
    hardness_kernel: ArrayLike,
    total_charge: ArrayLike,
    *,
    symmetry_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chi = _as_real_numeric_array("electronegativity", electronegativity)
    kernel = _as_real_numeric_array("hardness_kernel", hardness_kernel)
    if chi.ndim < 1:
        raise QEqOperatorError(f"electronegativity must have shape [..., natom], got {chi.shape}")
    if kernel.ndim < 2 or kernel.shape[-1] != kernel.shape[-2]:
        raise QEqOperatorError(f"hardness_kernel must be square with shape [..., natom, natom], got {kernel.shape}")
    n = chi.shape[-1]
    if n <= 0:
        raise QEqOperatorError("QEq system must contain at least one charge site")
    if kernel.shape[-1] != n:
        raise QEqOperatorError(
            f"electronegativity length and hardness_kernel size must match, got {n} and {kernel.shape[-1]}"
        )
    try:
        leading_shape = _broadcast_shapes(chi.shape[:-1], kernel.shape[:-2])
    except ValueError as exc:
        raise QEqOperatorError(
            f"leading electronegativity shape {chi.shape[:-1]} and hardness_kernel shape "
            f"{kernel.shape[:-2]} cannot broadcast"
        ) from exc
    if 0 in leading_shape:
        raise QEqOperatorError(
            f"QEq leading shape {leading_shape} must not contain a zero-length batch "
            "dimension; an empty batch would satisfy every safety gate vacuously"
        )
    chi = np.broadcast_to(chi, leading_shape + (n,)).astype(np.float64, copy=True)
    kernel = np.broadcast_to(kernel, leading_shape + (n, n)).astype(np.float64, copy=True)

    symmetry_tol = _as_float_scalar("symmetry_tol", symmetry_tol, nonnegative=True)
    asym = kernel - np.swapaxes(kernel, -1, -2)
    input_symmetry_error = np.max(np.abs(asym), axis=(-1, -2))
    if np.any(input_symmetry_error > symmetry_tol):
        raise QEqOperatorError(
            f"hardness_kernel must be symmetric within symmetry_tol={symmetry_tol:g}; "
            f"max asymmetry={float(np.max(input_symmetry_error)):.6g}"
        )
    kernel = 0.5 * (kernel + np.swapaxes(kernel, -1, -2))
    symmetry_error = np.max(np.abs(kernel - np.swapaxes(kernel, -1, -2)), axis=(-1, -2))
    qtot = _prepare_total_charge(total_charge, leading_shape)
    return chi, kernel, qtot, symmetry_error, input_symmetry_error


def _sum_zero_basis(n: int) -> np.ndarray:
    projector = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    eigvals, eigvecs = np.linalg.eigh(projector)
    return eigvecs[:, eigvals > 0.5]


def _constrained_kernel_diagnostics(
    kernel: np.ndarray,
    *,
    constrained_eig_floor: float,
    max_condition: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    leading_shape = kernel.shape[:-2]
    n = kernel.shape[-1]
    if n == 1:
        shape = leading_shape
        return (
            np.full(shape, np.inf, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.ones(shape, dtype=np.float64),
        )

    basis = _sum_zero_basis(n)
    flat_kernel = kernel.reshape((-1, n, n))
    min_eigs = []
    max_eigs = []
    conds = []
    for kernel_one in flat_kernel:
        tangent_kernel = basis.T @ kernel_one @ basis
        eigvals = np.linalg.eigvalsh(tangent_kernel)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        cond = max_eig / min_eig if min_eig > 0.0 else np.inf
        min_eigs.append(min_eig)
        max_eigs.append(max_eig)
        conds.append(cond)

    min_arr = np.asarray(min_eigs, dtype=np.float64).reshape(leading_shape)
    max_arr = np.asarray(max_eigs, dtype=np.float64).reshape(leading_shape)
    cond_arr = np.asarray(conds, dtype=np.float64).reshape(leading_shape)
    if np.any(min_arr <= constrained_eig_floor):
        raise QEqKernelConditionError(
            "hardness_kernel is not positive definite on the fixed-charge "
            f"subspace above constrained_eig_floor={constrained_eig_floor:g}; "
            f"min={float(np.min(min_arr)):.6g}"
        )
    if np.any(~np.isfinite(cond_arr)) or np.any(cond_arr > max_condition):
        raise QEqKernelConditionError(
            f"constrained hardness_kernel condition exceeds max_condition={max_condition:g}; "
            f"max={float(np.max(cond_arr)):.6g}"
        )
    return min_arr, max_arr, cond_arr


def _recompute_constrained_kernel_diagnostics(kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    leading_shape = kernel.shape[:-2]
    n = kernel.shape[-1]
    if n == 1:
        return (
            np.full(leading_shape, np.inf, dtype=np.float64),
            np.zeros(leading_shape, dtype=np.float64),
            np.ones(leading_shape, dtype=np.float64),
        )

    basis = _sum_zero_basis(n)
    flat_kernel = kernel.reshape((-1, n, n))
    min_eigs = []
    max_eigs = []
    conds = []
    for kernel_one in flat_kernel:
        tangent_kernel = basis.T @ kernel_one @ basis
        eigvals = np.linalg.eigvalsh(tangent_kernel)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        cond = max_eig / min_eig if min_eig > 0.0 else np.inf
        min_eigs.append(min_eig)
        max_eigs.append(max_eig)
        conds.append(cond)
    return (
        np.asarray(min_eigs, dtype=np.float64).reshape(leading_shape),
        np.asarray(max_eigs, dtype=np.float64).reshape(leading_shape),
        np.asarray(conds, dtype=np.float64).reshape(leading_shape),
    )


def _recompute_kkt_condition(kernel: np.ndarray) -> np.ndarray:
    leading_shape = kernel.shape[:-2]
    n = kernel.shape[-1]
    flat_kernel = kernel.reshape((-1, n, n))
    conditions = []
    for kernel_one in flat_kernel:
        kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
        kkt[:n, :n] = kernel_one
        kkt[:n, n] = 1.0
        kkt[n, :n] = 1.0
        conditions.append(float(np.linalg.cond(kkt)))
    return np.asarray(conditions, dtype=np.float64).reshape(leading_shape)


def _solve_fixed_charge_nullspace(
    chi: np.ndarray,
    kernel: np.ndarray,
    total_charge: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve only on the physical sum(q)=Q tangent space.

    The raw augmented KKT condition number is intentionally diagnostic-only:
    adding alpha*11.T to J leaves the fixed-charge minimizer unchanged but can
    make that raw condition number arbitrarily large.  Safety is already gated
    by the eigenvalue floor and condition number of Z.T@J@Z.
    """

    leading_shape = chi.shape[:-1]
    n = chi.shape[-1]
    flat_chi = chi.reshape((-1, n))
    flat_kernel = kernel.reshape((-1, n, n))
    flat_qtot = total_charge.reshape(-1)
    basis = None if n == 1 else _sum_zero_basis(n)
    charges = []
    multipliers = []
    kkt_conditions = []
    for chi_one, kernel_one, qtot_one in zip(flat_chi, flat_kernel, flat_qtot):
        q0 = np.full(n, qtot_one / float(n), dtype=np.float64)
        if n == 1:
            q = q0
        else:
            tangent_kernel = basis.T @ kernel_one @ basis
            tangent_rhs = -basis.T @ (chi_one + kernel_one @ q0)
            try:
                tangent_coordinates = np.linalg.solve(tangent_kernel, tangent_rhs)
            except np.linalg.LinAlgError as exc:  # defensive: spectrum was checked above
                raise QEqKernelConditionError(
                    "constrained QEq tangent system is singular"
                ) from exc
            q = q0 + basis @ tangent_coordinates
        stationarity_without_lambda = chi_one + kernel_one @ q
        lagrange_multiplier = -float(np.mean(stationarity_without_lambda))
        charges.append(q)
        multipliers.append(lagrange_multiplier)
        kkt_conditions.append(float(_recompute_kkt_condition(kernel_one)))
    return (
        np.asarray(charges, dtype=np.float64).reshape(leading_shape + (n,)),
        np.asarray(multipliers, dtype=np.float64).reshape(leading_shape),
        np.asarray(kkt_conditions, dtype=np.float64).reshape(leading_shape),
    )


def _qeq_energies(
    chi: np.ndarray,
    kernel: np.ndarray,
    charges: np.ndarray,
    total_charge: np.ndarray,
    lagrange_multiplier: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    linear = np.einsum("...i,...i->...", chi, charges, optimize=True)
    quadratic = 0.5 * np.einsum("...i,...ij,...j->...", charges, kernel, charges, optimize=True)
    total = linear + quadratic
    identity = 0.5 * linear - 0.5 * lagrange_multiplier * total_charge
    return total, linear, quadratic, identity


def _stationarity_parts(
    chi: np.ndarray,
    kernel: np.ndarray,
    charges: np.ndarray,
    lagrange_multiplier: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the KKT stationarity residual into its gauge-invariant components.

    Returns ``(stationarity, tangent_max_abs, multiplier_residual,
    arithmetic_scale)``.  ``tangent_max_abs`` is the sum-zero projection
    ``Z^T (chi + J q)``, which is invariant under ``J -> J + alpha 11^T`` because
    ``Z^T 1 == 0``; ``multiplier_residual`` is the uniform component that defines
    ``lambda``.  ``arithmetic_scale`` is the magnitude of the largest
    intermediate the expression goes through, and therefore the scale below
    which a float64 residual carries no information.
    """

    n = chi.shape[-1]
    gradient = chi + np.einsum("...ij,...j->...i", kernel, charges, optimize=True)
    stationarity = gradient + lagrange_multiplier[..., None]
    if n == 1:
        tangent_max_abs = np.zeros(chi.shape[:-1], dtype=np.float64)
    else:
        basis = _sum_zero_basis(n)
        tangent = np.einsum("ik,...i->...k", basis, gradient, optimize=True)
        tangent_max_abs = np.max(np.abs(tangent), axis=-1)
    multiplier_residual = np.mean(gradient, axis=-1) + lagrange_multiplier
    arithmetic_scale = (
        np.max(np.abs(chi), axis=-1)
        + np.max(np.abs(kernel), axis=(-1, -2)) * np.sum(np.abs(charges), axis=-1)
        + np.abs(lagrange_multiplier)
    )
    return stationarity, tangent_max_abs, multiplier_residual, arithmetic_scale


def _residual_tolerance(atol: float, arithmetic_scale: np.ndarray) -> np.ndarray:
    return atol + _ARITHMETIC_FLOOR_ULPS * _MACHINE_EPS * arithmetic_scale


def solve_qeq(
    electronegativity: ArrayLike,
    hardness_kernel: ArrayLike,
    *,
    total_charge: ArrayLike = 0.0,
    symmetry_tol: float = 1e-10,
    constrained_eig_floor: float = 1e-12,
    max_condition: float = 1e12,
    residual_tol: float = 1e-8,
    units: QEqUnits = QEqUnits(),
) -> QEqResult:
    """Solve the dense constrained QEq problem.

    Parameters
    ----------
    electronegativity
        Real ``[..., natom]`` array in ``eV/e``.
    hardness_kernel
        Real symmetric ``[..., natom, natom]`` kernel in ``eV/e^2``.
    total_charge
        Scalar or broadcastable ``[...]`` total charge in ``e``.

    Returns
    -------
    QEqResult
        Charges in ``e``, energies in ``eV``, and fail-closed diagnostics for
        the KKT stationarity equation ``chi + J q + lambda 1 = 0``.

    Notes
    -----
    ``eV`` here is the DeePTB convention only; the solve is unit-agnostic and
    returns whatever consistent energy unit ``electronegativity`` and
    ``hardness_kernel`` were expressed in.

    ``max_condition``, ``residual_tol`` and ``symmetry_tol`` are capped by the
    module-level ``MODULE_MAX_*`` ceilings so that every accepted policy is one
    :func:`validate_qeq_result` can also certify.
    """

    symmetry_tol = _as_float_scalar("symmetry_tol", symmetry_tol, nonnegative=True)
    symmetry_tol = _clamp_policy("symmetry_tol", symmetry_tol, MODULE_MAX_SYMMETRY_TOL)
    constrained_eig_floor = _as_float_scalar("constrained_eig_floor", constrained_eig_floor, positive=True)
    max_condition = _as_float_scalar("max_condition", max_condition, positive=True)
    max_condition = _clamp_policy("max_condition", max_condition, MODULE_MAX_CONDITION)
    residual_tol = _as_float_scalar("residual_tol", residual_tol, nonnegative=True)
    residual_tol = _clamp_policy("residual_tol", residual_tol, MODULE_MAX_RESIDUAL_TOL)
    if not isinstance(units, QEqUnits):
        raise QEqOperatorError("units must be a QEqUnits instance")

    chi, kernel, qtot, symmetry_error, input_symmetry_error = _validate_qeq_inputs(
        electronegativity,
        hardness_kernel,
        total_charge,
        symmetry_tol=symmetry_tol,
    )
    min_eig, max_eig, constrained_condition = _constrained_kernel_diagnostics(
        kernel,
        constrained_eig_floor=constrained_eig_floor,
        max_condition=max_condition,
    )
    charges, lagrange_multiplier, kkt_condition = _solve_fixed_charge_nullspace(
        chi,
        kernel,
        qtot,
    )

    stationarity, tangent_max_abs, multiplier_residual, _ = _stationarity_parts(
        chi,
        kernel,
        charges,
        lagrange_multiplier,
    )
    charge_residual = np.sum(charges, axis=-1) - qtot
    energy, linear_energy, quadratic_energy, energy_identity = _qeq_energies(
        chi,
        kernel,
        charges,
        qtot,
        lagrange_multiplier,
    )
    energy_identity_residual = energy - energy_identity
    stationarity_max_abs = np.max(np.abs(stationarity), axis=-1)
    stationarity_l2 = np.linalg.norm(stationarity, axis=-1)
    diagnostics = QEqDiagnostics(
        charge_residual=charge_residual,
        stationarity_max_abs=stationarity_max_abs,
        stationarity_l2=stationarity_l2,
        stationarity_tangent_max_abs=tangent_max_abs,
        multiplier_residual=multiplier_residual,
        energy_identity_residual=energy_identity_residual,
        kernel_symmetry_error=symmetry_error,
        input_kernel_symmetry_error=input_symmetry_error,
        constrained_min_eig=min_eig,
        constrained_max_eig=max_eig,
        constrained_condition=constrained_condition,
        kkt_condition=kkt_condition,
    )
    result = QEqResult(
        charges=charges,
        total_charge=qtot,
        electronegativity=chi,
        hardness_kernel=kernel,
        lagrange_multiplier=lagrange_multiplier,
        stationarity=stationarity,
        energy=energy,
        linear_energy=linear_energy,
        quadratic_energy=quadratic_energy,
        energy_identity=energy_identity,
        diagnostics=diagnostics,
        units=units,
        symmetry_tol=symmetry_tol,
        constrained_eig_floor=constrained_eig_floor,
        max_condition=max_condition,
        residual_tol=residual_tol,
    )
    validate_qeq_result(result, atol=residual_tol)
    return result


def _require_result_array(name: str, value: ArrayLike, *, finite: bool = True) -> np.ndarray:
    _reject_torch_tensor(name, value)
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise QEqOperatorError(f"QEq {name} must be numeric")
    if np.iscomplexobj(arr):
        raise QEqOperatorError(f"QEq {name} must be real")
    arr = np.asarray(arr, dtype=np.float64)
    if finite and not np.isfinite(arr).all():
        raise QEqOperatorError(f"QEq {name} must contain only finite values")
    return arr


def _require_same_shape(name: str, actual: np.ndarray, expected_shape: Tuple[int, ...]) -> None:
    if actual.shape != expected_shape:
        raise QEqOperatorError(f"QEq {name} has shape {actual.shape}, expected {expected_shape}")


def _assert_recomputed_close(
    name: str,
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    atol: float,
    rtol: float = 0.0,
    floor: ArrayLike = 0.0,
) -> None:
    if stored.shape != recomputed.shape:
        raise QEqOperatorError(f"QEq stored {name} has shape {stored.shape}, recomputed {recomputed.shape}")
    stored_arr = np.asarray(stored, dtype=np.float64)
    recomputed_arr = np.asarray(recomputed, dtype=np.float64)
    tolerance = np.asarray(floor, dtype=np.float64) + atol + rtol * np.abs(recomputed_arr)
    # constrained_min_eig and kkt_condition are legitimately infinite for n == 1,
    # where the difference is an invalid inf - inf.
    with np.errstate(invalid="ignore"):
        difference = np.abs(stored_arr - recomputed_arr)
    matched = (difference <= tolerance) | (stored_arr == recomputed_arr)
    if not np.all(matched):
        raise QEqOperatorError(f"QEq stored {name} does not match recomputed value")


def _resolve_policy(
    name: str,
    stored: float,
    expected: Optional[float],
    ceiling: float,
    *,
    positive: bool = False,
) -> float:
    if expected is None:
        value = stored
    else:
        value = _as_float_scalar(
            f"expected_{name}",
            expected,
            positive=positive,
            nonnegative=not positive,
        )
    return _clamp_policy(name, value, ceiling)


def validate_qeq_result(
    result: QEqResult,
    *,
    atol: Optional[float] = None,
    expected_symmetry_tol: Optional[float] = None,
    expected_constrained_eig_floor: Optional[float] = None,
    expected_max_condition: Optional[float] = None,
) -> None:
    """Fail closed when a QEq result violates its recomputed contract.

    Validation recomputes charge conservation, KKT stationarity, energy pieces,
    the constrained-kernel spectrum, and KKT conditioning from the current
    result arrays.  Stored result fields and diagnostics must agree with those
    recomputed quantities; cached residuals alone are never trusted.

    Parameters
    ----------
    atol
        Absolute residual tolerance in the caller's energy unit.  ``None``
        (the default) adopts the ``residual_tol`` recorded by the solve, so a
        result carrying a looser *or* tighter recorded policy is re-enforced as
        recorded.  An explicit value always wins.  Every residual gate admits
        ``atol`` plus a float64 arithmetic floor derived from the magnitude of
        the intermediates, which is what makes the gates invariant under the
        gauge freedom ``J -> J + alpha 11^T``.
    expected_symmetry_tol, expected_constrained_eig_floor, expected_max_condition
        Caller-supplied safety policy.  Each overrides the value recorded on
        ``result`` — use them when the result came from an untrusted producer
        and its self-declared policy must not be believed.

    Notes
    -----
    The recorded policy is data, so it is clamped against the module-level
    ``MODULE_MAX_*`` ceilings: a payload may tighten the shipped policy but
    never loosen it.
    """

    if not isinstance(result, QEqResult):
        raise QEqOperatorError("result must be a QEqResult")
    recorded_residual_tol = _clamp_policy(
        "residual_tol", result.residual_tol, MODULE_MAX_RESIDUAL_TOL
    )
    if atol is None:
        atol = recorded_residual_tol
    atol = _as_float_scalar("atol", atol, nonnegative=True)
    atol = _clamp_policy("atol", atol, MODULE_MAX_RESIDUAL_TOL)
    symmetry_tol = _resolve_policy(
        "symmetry_tol", result.symmetry_tol, expected_symmetry_tol, MODULE_MAX_SYMMETRY_TOL
    )
    constrained_eig_floor = _resolve_policy(
        "constrained_eig_floor",
        result.constrained_eig_floor,
        expected_constrained_eig_floor,
        np.inf,
        positive=True,
    )
    max_condition = _resolve_policy(
        "max_condition", result.max_condition, expected_max_condition, MODULE_MAX_CONDITION
    )
    if not isinstance(result.diagnostics, QEqDiagnostics):
        raise QEqOperatorError("QEq diagnostics must be a QEqDiagnostics instance")
    if not isinstance(result.units, QEqUnits):
        raise QEqOperatorError("QEq units must be a QEqUnits instance")

    charges = _require_result_array("charges", result.charges)
    qtot = _require_result_array("total_charge", result.total_charge)
    chi = _require_result_array("electronegativity", result.electronegativity)
    kernel = _require_result_array("hardness_kernel", result.hardness_kernel)
    multiplier = _require_result_array("lagrange_multiplier", result.lagrange_multiplier)

    if charges.ndim < 1:
        raise QEqOperatorError(f"QEq charges must have shape [..., natom], got {charges.shape}")
    n = charges.shape[-1]
    if n <= 0:
        raise QEqOperatorError("QEq result must contain at least one charge site")
    leading_shape = charges.shape[:-1]
    _require_same_shape("total_charge", qtot, leading_shape)
    _require_same_shape("electronegativity", chi, leading_shape + (n,))
    _require_same_shape("hardness_kernel", kernel, leading_shape + (n, n))
    _require_same_shape("lagrange_multiplier", multiplier, leading_shape)

    asym = kernel - np.swapaxes(kernel, -1, -2)
    symmetry_error = np.max(np.abs(asym), axis=(-1, -2))
    if np.any(symmetry_error > atol):
        raise QEqOperatorError("QEq hardness_kernel is not symmetric within tolerance")
    input_symmetry_error = _require_result_array(
        "diagnostics.input_kernel_symmetry_error", result.diagnostics.input_kernel_symmetry_error
    )
    _require_same_shape("diagnostics.input_kernel_symmetry_error", input_symmetry_error, leading_shape)
    if np.any(input_symmetry_error < 0.0):
        raise QEqOperatorError("QEq input_kernel_symmetry_error must be nonnegative")
    # The stored kernel is already symmetrized, so the raw input asymmetry is
    # unrecoverable here; the policy it was accepted under is re-enforced instead.
    if np.any(input_symmetry_error > symmetry_tol):
        raise QEqOperatorError(
            f"QEq input hardness_kernel asymmetry {float(np.max(input_symmetry_error)):.6g} "
            f"exceeds the recorded symmetry_tol={symmetry_tol:g}"
        )

    charge_residual = np.sum(charges, axis=-1) - qtot
    stationarity, tangent_max_abs, multiplier_residual, arithmetic_scale = _stationarity_parts(
        chi,
        kernel,
        charges,
        multiplier,
    )
    stationarity_max_abs = np.max(np.abs(stationarity), axis=-1)
    stationarity_l2 = np.linalg.norm(stationarity, axis=-1)
    residual_floor = _ARITHMETIC_FLOOR_ULPS * _MACHINE_EPS * arithmetic_scale
    stationarity_tolerance = atol + residual_floor
    energy, linear_energy, quadratic_energy, energy_identity = _qeq_energies(
        chi,
        kernel,
        charges,
        qtot,
        multiplier,
    )
    energy_identity_residual = energy - energy_identity
    constrained_min_eig, constrained_max_eig, constrained_condition = _recompute_constrained_kernel_diagnostics(kernel)
    if n > 1 and np.any(constrained_min_eig <= constrained_eig_floor):
        raise QEqKernelConditionError(
            "QEq hardness_kernel violates the recorded constrained_eig_floor"
        )
    if np.any(~np.isfinite(constrained_condition)) or np.any(constrained_condition > max_condition):
        raise QEqKernelConditionError(
            "QEq hardness_kernel violates the recorded constrained max_condition"
        )
    kkt_condition = _recompute_kkt_condition(kernel)

    energy_scale = np.abs(linear_energy) + np.abs(quadratic_energy)
    energy_floor = _ARITHMETIC_FLOOR_ULPS * _MACHINE_EPS * energy_scale
    _assert_recomputed_close(
        "stationarity",
        _require_result_array("stationarity", result.stationarity),
        stationarity,
        atol=atol,
        floor=residual_floor[..., None] if residual_floor.ndim else residual_floor,
    )
    _assert_recomputed_close(
        "energy",
        _require_result_array("energy", result.energy),
        energy,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "linear_energy",
        _require_result_array("linear_energy", result.linear_energy),
        linear_energy,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "quadratic_energy",
        _require_result_array("quadratic_energy", result.quadratic_energy),
        quadratic_energy,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "energy_identity",
        _require_result_array("energy_identity", result.energy_identity),
        energy_identity,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )

    diagnostics = result.diagnostics
    _assert_recomputed_close(
        "diagnostic charge_residual",
        _require_result_array("diagnostics.charge_residual", diagnostics.charge_residual),
        charge_residual,
        atol=atol,
    )
    _assert_recomputed_close(
        "diagnostic stationarity_max_abs",
        _require_result_array("diagnostics.stationarity_max_abs", diagnostics.stationarity_max_abs),
        stationarity_max_abs,
        atol=atol,
        floor=residual_floor,
    )
    _assert_recomputed_close(
        "diagnostic stationarity_l2",
        _require_result_array("diagnostics.stationarity_l2", diagnostics.stationarity_l2),
        stationarity_l2,
        atol=atol,
        floor=residual_floor,
    )
    _assert_recomputed_close(
        "diagnostic stationarity_tangent_max_abs",
        _require_result_array(
            "diagnostics.stationarity_tangent_max_abs", diagnostics.stationarity_tangent_max_abs
        ),
        tangent_max_abs,
        atol=atol,
        floor=residual_floor,
    )
    _assert_recomputed_close(
        "diagnostic multiplier_residual",
        _require_result_array("diagnostics.multiplier_residual", diagnostics.multiplier_residual),
        multiplier_residual,
        atol=atol,
        floor=residual_floor,
    )
    _assert_recomputed_close(
        "diagnostic energy_identity_residual",
        _require_result_array("diagnostics.energy_identity_residual", diagnostics.energy_identity_residual),
        energy_identity_residual,
        atol=atol,
        floor=energy_floor,
    )
    _assert_recomputed_close(
        "diagnostic kernel_symmetry_error",
        _require_result_array("diagnostics.kernel_symmetry_error", diagnostics.kernel_symmetry_error),
        symmetry_error,
        atol=atol,
    )
    _assert_recomputed_close(
        "diagnostic constrained_min_eig",
        _require_result_array("diagnostics.constrained_min_eig", diagnostics.constrained_min_eig, finite=False),
        constrained_min_eig,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "diagnostic constrained_max_eig",
        _require_result_array("diagnostics.constrained_max_eig", diagnostics.constrained_max_eig),
        constrained_max_eig,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "diagnostic constrained_condition",
        _require_result_array("diagnostics.constrained_condition", diagnostics.constrained_condition),
        constrained_condition,
        atol=atol,
        rtol=_RECOMPUTE_RTOL,
    )
    _assert_recomputed_close(
        "diagnostic kkt_condition",
        _require_result_array("diagnostics.kkt_condition", diagnostics.kkt_condition, finite=False),
        kkt_condition,
        atol=max(atol, 1e-10),
        rtol=_RECOMPUTE_RTOL,
    )

    if np.any(np.abs(charge_residual) > atol):
        raise QEqOperatorError("QEq charges do not satisfy sum(q) = total_charge")
    _check_stationarity_gate(
        tangent_max_abs,
        stationarity_tolerance,
        constrained_condition,
        arithmetic_scale,
    )
    if np.any(np.abs(multiplier_residual) > stationarity_tolerance):
        raise QEqOperatorError(
            "QEq lagrange_multiplier is inconsistent with the uniform stationarity component"
        )
    # energy_identity_residual gets no terminal gate of its own: it is identically
    # 0.5 * q . stationarity - 0.5 * lambda * (sum(q) - Q), so any threshold on it
    # is either implied by the charge and stationarity gates or, if set from the
    # energy scale instead of the |q| * residual scale it actually lives on,
    # rejects legitimate ill-conditioned solves. It stays as telemetry, checked
    # above against its recomputation.


def _check_stationarity_gate(
    tangent_max_abs: np.ndarray,
    tolerance: np.ndarray,
    constrained_condition: np.ndarray,
    arithmetic_scale: np.ndarray,
) -> None:
    """Gate the gauge-invariant stationarity residual, naming conditioning when it is the cause."""

    residual = np.atleast_1d(np.asarray(tangent_max_abs, dtype=np.float64))
    tol = np.atleast_1d(np.broadcast_to(np.asarray(tolerance, dtype=np.float64), residual.shape))
    condition = np.atleast_1d(np.broadcast_to(np.asarray(constrained_condition, dtype=np.float64), residual.shape))
    scale = np.atleast_1d(np.broadcast_to(np.asarray(arithmetic_scale, dtype=np.float64), residual.shape))
    failing = residual > tol
    if not np.any(failing):
        return
    # A backward-stable solve on a kernel with tangent condition kappa cannot
    # push the residual below ~ eps * kappa * (magnitude of the intermediates).
    attainable = _MACHINE_EPS * condition * scale
    explained = failing & (attainable >= tol)
    if np.any(explained):
        raise QEqKernelConditionError(
            f"QEq KKT stationarity residual {float(np.max(residual[explained])):.6g} exceeds "
            f"tolerance {float(np.min(tol[explained])):.6g}, but the constrained kernel condition "
            f"{float(np.max(condition[explained])):.6g} only admits ~"
            f"{float(np.max(attainable[explained])):.6g} in double precision. Precondition the "
            "hardness kernel, lower max_condition, or raise residual_tol."
        )
    raise QEqOperatorError("QEq KKT stationarity residual exceeds tolerance")
