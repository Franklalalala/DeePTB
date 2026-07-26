"""Dense charge-equilibration (QEq) reference utilities.

This module is an independent NumPy CPU reference for the constrained QEq
problem

    E(q) = chi.T @ q + 0.5 * q.T @ J @ q,  sum(q) = total_charge.

It is deliberately separate from the fixed-mu H/S operator: QEq equilibrates a
finite charge vector under a fixed total-charge constraint, while fixed-mu
postprocessing evaluates occupations and density matrices from Hamiltonian and
overlap matrices at a chemical potential.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

ArrayLike = Any


class QEqOperatorError(ValueError):
    """Base class for fail-closed QEq validation errors."""


class QEqKernelConditionError(QEqOperatorError):
    """Raised when the constrained QEq kernel is unsafe to solve."""


@dataclass(frozen=True)
class QEqUnits:
    """Units used by the dense QEq reference kernel."""

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


@dataclass(frozen=True)
class QEqDiagnostics:
    """Constraint, stationarity, identity, and conditioning diagnostics."""

    charge_residual: np.ndarray
    stationarity_max_abs: np.ndarray
    stationarity_l2: np.ndarray
    energy_identity_residual: np.ndarray
    kernel_symmetry_error: np.ndarray
    constrained_min_eig: np.ndarray
    constrained_max_eig: np.ndarray
    constrained_condition: np.ndarray
    kkt_condition: np.ndarray


@dataclass(frozen=True)
class QEqResult:
    """Observable bundle for the dense constrained QEq solve."""

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


def _validate_qeq_inputs(
    electronegativity: ArrayLike,
    hardness_kernel: ArrayLike,
    total_charge: ArrayLike,
    *,
    symmetry_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    chi = np.broadcast_to(chi, leading_shape + (n,)).astype(np.float64, copy=True)
    kernel = np.broadcast_to(kernel, leading_shape + (n, n)).astype(np.float64, copy=True)

    symmetry_tol = _as_float_scalar("symmetry_tol", symmetry_tol, nonnegative=True)
    asym = kernel - np.swapaxes(kernel, -1, -2)
    symmetry_error = np.max(np.abs(asym), axis=(-1, -2))
    if np.any(symmetry_error > symmetry_tol):
        raise QEqOperatorError(
            f"hardness_kernel must be symmetric within symmetry_tol={symmetry_tol:g}; "
            f"max asymmetry={float(np.max(symmetry_error)):.6g}"
        )
    kernel = 0.5 * (kernel + np.swapaxes(kernel, -1, -2))
    qtot = _prepare_total_charge(total_charge, leading_shape)
    return chi, kernel, qtot, symmetry_error


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


def _solve_kkt(
    chi: np.ndarray,
    kernel: np.ndarray,
    total_charge: np.ndarray,
    *,
    max_condition: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    leading_shape = chi.shape[:-1]
    n = chi.shape[-1]
    flat_chi = chi.reshape((-1, n))
    flat_kernel = kernel.reshape((-1, n, n))
    flat_qtot = total_charge.reshape(-1)
    charges = []
    multipliers = []
    kkt_conditions = []
    for chi_one, kernel_one, qtot_one in zip(flat_chi, flat_kernel, flat_qtot):
        kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
        kkt[:n, :n] = kernel_one
        kkt[:n, n] = 1.0
        kkt[n, :n] = 1.0
        condition = float(np.linalg.cond(kkt))
        if not np.isfinite(condition) or condition > max_condition:
            raise QEqKernelConditionError(
                f"QEq KKT system condition exceeds max_condition={max_condition:g}; condition={condition:.6g}"
            )
        rhs = np.empty(n + 1, dtype=np.float64)
        rhs[:n] = -chi_one
        rhs[n] = qtot_one
        try:
            solution = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError as exc:
            raise QEqKernelConditionError("QEq KKT system is singular") from exc
        charges.append(solution[:n])
        multipliers.append(solution[n])
        kkt_conditions.append(condition)
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
    """

    constrained_eig_floor = _as_float_scalar("constrained_eig_floor", constrained_eig_floor, positive=True)
    max_condition = _as_float_scalar("max_condition", max_condition, positive=True)
    residual_tol = _as_float_scalar("residual_tol", residual_tol, nonnegative=True)
    if not isinstance(units, QEqUnits):
        raise QEqOperatorError("units must be a QEqUnits instance")

    chi, kernel, qtot, symmetry_error = _validate_qeq_inputs(
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
    charges, lagrange_multiplier, kkt_condition = _solve_kkt(
        chi,
        kernel,
        qtot,
        max_condition=max_condition,
    )

    stationarity = chi + np.einsum("...ij,...j->...i", kernel, charges, optimize=True) + lagrange_multiplier[..., None]
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
        energy_identity_residual=energy_identity_residual,
        kernel_symmetry_error=symmetry_error,
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
    )
    validate_qeq_result(result, atol=residual_tol)
    return result


def validate_qeq_result(result: QEqResult, *, atol: float = 1e-8) -> None:
    """Fail closed when a QEq solve violates its contract beyond ``atol``."""

    if not isinstance(result, QEqResult):
        raise QEqOperatorError("result must be a QEqResult")
    atol = _as_float_scalar("atol", atol, nonnegative=True)
    diagnostics = result.diagnostics
    if np.any(np.abs(diagnostics.charge_residual) > atol):
        raise QEqOperatorError("QEq charges do not satisfy sum(q) = total_charge")
    if np.any(diagnostics.stationarity_max_abs > atol):
        raise QEqOperatorError("QEq KKT stationarity residual exceeds tolerance")
    if np.any(np.abs(diagnostics.energy_identity_residual) > atol):
        raise QEqOperatorError("QEq energy identity residual exceeds tolerance")
