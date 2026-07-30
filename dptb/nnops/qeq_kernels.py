"""Finite-cluster geometry kernels for the dense QEq reference solver.

This module turns Cartesian coordinates in Angstrom and element symbols into a
dense hardness kernel in eV/e^2.  It is deliberately limited to isolated,
finite clusters: there is no periodic minimum image, Ewald/PME, constant-
potential constraint, dipole correction, force, or differentiable backend.

The three available constructions are:

``bare``
    Point Coulomb interactions off diagonal and element hardness on diagonal.
    This is the unscreened limiting form of the QEq electrostatic matrix in
    Rappe and Goddard, J. Phys. Chem. 95, 3358 (1991),
    https://doi.org/10.1021/j100161a070.
``ohno``
    The Ohno/Klopman regularization (K. Ohno, Theor. Chim. Acta 2, 219
    (1964); G. Klopman, JACS 86, 1463 (1964),
    https://doi.org/10.1021/ja01062a001), with the explicit convention
    ``gamma_ij = 2 k_e / (J_ii + J_jj)``.
``gaussian``
    The non-periodic Gaussian pair interaction used by the Hu et al. QEq
    decomposition (Nat. Commun. 16, 7379 (2025),
    https://doi.org/10.1038/s41467-025-62824-5).  It includes the analytic
    Gaussian self term on the diagonal but none of Hu's PME or dipole terms.

Every constructor rejects non-finite coordinates, unsupported elements, and
overlapping atoms.  Returned arrays are bytes-backed and read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from dptb.nnops.qeq_operator import (
    QEqOperatorError,
    QEqResult,
    QEqUnits,
    solve_qeq,
)
from dptb.nnops.qeq_parameters import (
    QEqParameterTable,
    coerce_qeq_parameter_table,
)

ArrayLike = Any

# Coulomb energy of two elementary charges separated by one Angstrom.
COULOMB_EV_ANGSTROM = 14.3996
# Distances at or below this threshold are treated as overlapping.  It is far
# below any physical geometry but avoids silently evaluating 1 / 0 or relying
# on a regularizer to assign two atoms to the same point.
MIN_INTERATOMIC_DISTANCE_ANGSTROM = 1.0e-12
SUPPORTED_QEQ_KERNELS = ("bare", "gaussian", "ohno")


class QEqGeometryError(QEqOperatorError):
    """Raised when geometry cannot define a certified cluster QEq problem."""


def _looks_like_torch_tensor(value: Any) -> bool:
    typ = type(value)
    module = getattr(typ, "__module__", "")
    return module == "torch" or module.startswith("torch.") or (
        hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    )


def _readonly_float64(value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    immutable = np.frombuffer(
        array.tobytes(order="C"), dtype=np.float64
    ).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _canonical_symbols(symbols: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(symbols, (str, bytes)):
        raise QEqGeometryError("symbols must be a sequence of element symbols")
    try:
        raw = tuple(symbols)
    except TypeError as exc:
        raise QEqGeometryError(
            "symbols must be a sequence of element symbols"
        ) from exc
    canonical = []
    for symbol in raw:
        if not isinstance(symbol, str) or not symbol.strip():
            raise QEqGeometryError("element symbols must be non-empty strings")
        stripped = symbol.strip()
        normalized = stripped[0].upper() + stripped[1:].lower()
        if not normalized.isalpha() or len(normalized) > 3:
            raise QEqGeometryError(f"invalid element symbol {symbol!r}")
        canonical.append(normalized)
    return tuple(canonical)


def _prepare_geometry(
    positions: ArrayLike,
    symbols: Sequence[str],
    params: Optional[Any],
) -> Tuple[
    np.ndarray,
    Tuple[str, ...],
    QEqParameterTable,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if _looks_like_torch_tensor(positions):
        raise QEqGeometryError(
            "positions looks like a torch Tensor. This NumPy reference will "
            "not detach/cpu tensors silently; pass an explicit NumPy array."
        )
    array = np.asarray(positions)
    if not np.issubdtype(array.dtype, np.number):
        raise QEqGeometryError("positions must be a numeric real array")
    if np.iscomplexobj(array):
        raise QEqGeometryError("positions must be real")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise QEqGeometryError(
            f"positions must have shape (natom, 3), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise QEqGeometryError("positions must contain at least one atom")
    if not np.isfinite(array).all():
        raise QEqGeometryError("positions must contain only finite values")

    normalized_symbols = _canonical_symbols(symbols)
    if len(normalized_symbols) != array.shape[0]:
        raise QEqGeometryError(
            f"symbols has length {len(normalized_symbols)}, expected "
            f"{array.shape[0]} for positions"
        )

    table = coerce_qeq_parameter_table(params)
    element_values = [table[symbol] for symbol in normalized_symbols]
    electronegativity = np.asarray(
        [value.electronegativity for value in element_values], dtype=np.float64
    )
    hardness = np.asarray(
        [value.hardness for value in element_values], dtype=np.float64
    )
    widths = np.asarray(
        [value.gaussian_width for value in element_values], dtype=np.float64
    )

    with np.errstate(over="ignore", invalid="ignore"):
        displacement = array[:, None, :] - array[None, :, :]
        distances = np.linalg.norm(displacement, axis=-1)
    if not np.isfinite(distances).all():
        raise QEqGeometryError(
            "pair distances overflowed or became non-finite; coordinates are "
            "outside the safe float64 geometry range"
        )
    if array.shape[0] > 1:
        off_diagonal = ~np.eye(array.shape[0], dtype=bool)
        minimum = float(np.min(distances[off_diagonal]))
        if minimum <= MIN_INTERATOMIC_DISTANCE_ANGSTROM:
            raise QEqGeometryError(
                "overlapping atoms are not permitted: minimum pair distance "
                f"{minimum:.6g} Angstrom is not greater than "
                f"{MIN_INTERATOMIC_DISTANCE_ANGSTROM:.6g} Angstrom"
            )

    return (
        np.array(array, copy=True, order="C"),
        normalized_symbols,
        table,
        electronegativity,
        hardness,
        widths,
        distances,
    )


def _finish_kernel(kernel: np.ndarray) -> np.ndarray:
    if not np.isfinite(kernel).all():
        raise QEqGeometryError("constructed hardness kernel is not finite")
    if not np.array_equal(kernel, kernel.T):
        raise QEqGeometryError("internal error: constructed kernel is not symmetric")
    return _readonly_float64(kernel)


def _bare_from_prepared(hardness: np.ndarray, distances: np.ndarray) -> np.ndarray:
    natom = hardness.size
    kernel = np.diag(hardness)
    for i in range(natom):
        for j in range(i):
            value = COULOMB_EV_ANGSTROM / distances[i, j]
            kernel[i, j] = value
            kernel[j, i] = value
    return _finish_kernel(kernel)


def _ohno_from_prepared(hardness: np.ndarray, distances: np.ndarray) -> np.ndarray:
    natom = hardness.size
    kernel = np.diag(hardness)
    for i in range(natom):
        for j in range(i):
            gamma = (
                2.0
                * COULOMB_EV_ANGSTROM
                / (hardness[i] + hardness[j])
            )
            value = COULOMB_EV_ANGSTROM / math.sqrt(
                distances[i, j] ** 2 + gamma**2
            )
            kernel[i, j] = value
            kernel[j, i] = value
    return _finish_kernel(kernel)


def _gaussian_from_prepared(
    hardness: np.ndarray,
    widths: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    natom = hardness.size
    # For sigma_ii = sqrt(2) sigma_i, the r -> 0 limit of
    # k_e erf[r/(sqrt(2) sigma_ii)] / r is k_e/(sqrt(pi) sigma_i).
    gaussian_self = COULOMB_EV_ANGSTROM / (math.sqrt(math.pi) * widths)
    kernel = np.diag(hardness + gaussian_self)
    for i in range(natom):
        for j in range(i):
            sigma_ij = math.sqrt(widths[i] ** 2 + widths[j] ** 2)
            distance = distances[i, j]
            value = (
                COULOMB_EV_ANGSTROM
                * math.erf(distance / (math.sqrt(2.0) * sigma_ij))
                / distance
            )
            kernel[i, j] = value
            kernel[j, i] = value
    return _finish_kernel(kernel)


def bare_qeq_kernel(
    positions: ArrayLike,
    symbols: Sequence[str],
    *,
    params: Optional[Any] = None,
) -> np.ndarray:
    """Build ``J_ii = hardness_i`` and ``J_ij = 14.3996 / r_ij``.

    Positions are in Angstrom and the result is in eV/e^2.  This point-charge
    kernel is singular as two sites approach one another; exact/numerical
    overlaps are rejected before division.
    """

    prepared = _prepare_geometry(positions, symbols, params)
    return _bare_from_prepared(prepared[4], prepared[6])


def ohno_qeq_kernel(
    positions: ArrayLike,
    symbols: Sequence[str],
    *,
    params: Optional[Any] = None,
) -> np.ndarray:
    """Build the selected Ohno/Klopman regularized cluster kernel.

    The convention is

    ``gamma_ij = 14.3996 * 2 / (J_ii + J_jj)`` in Angstrom and
    ``J_ij = 14.3996 / sqrt(r_ij^2 + gamma_ij^2)`` off diagonal.
    The diagonal is the element hardness.  This convention makes the zero-
    separation pair limit the arithmetic mean ``(J_ii + J_jj) / 2``.
    """

    prepared = _prepare_geometry(positions, symbols, params)
    return _ohno_from_prepared(prepared[4], prepared[6])


def gaussian_qeq_kernel(
    positions: ArrayLike,
    symbols: Sequence[str],
    *,
    params: Optional[Any] = None,
) -> np.ndarray:
    """Build the isolated Gaussian-charge kernel corresponding to Hu's SI.

    With ``sigma_ij = sqrt(sigma_i^2 + sigma_j^2)``, the off-diagonal term is

    ``14.3996 erf(r_ij / (sqrt(2) sigma_ij)) / r_ij``.

    The diagonal contains the element on-site hardness plus the analytic
    Gaussian self term ``14.3996 / (sqrt(pi) sigma_i)``.  The original Hu
    method additionally uses periodic PME and a dipole correction; neither is
    present here.
    """

    prepared = _prepare_geometry(positions, symbols, params)
    return _gaussian_from_prepared(prepared[4], prepared[5], prepared[6])


def _normalize_kernel_name(kernel: Any) -> str:
    if not isinstance(kernel, str) or not kernel.strip():
        raise QEqGeometryError(
            f"kernel must be one of {SUPPORTED_QEQ_KERNELS}, got {kernel!r}"
        )
    normalized = kernel.strip().lower()
    if normalized not in SUPPORTED_QEQ_KERNELS:
        raise QEqGeometryError(
            f"unknown QEq geometry kernel {kernel!r}; expected one of "
            f"{SUPPORTED_QEQ_KERNELS}"
        )
    return normalized


def build_qeq_kernel(
    positions: ArrayLike,
    symbols: Sequence[str],
    *,
    kernel: str = "gaussian",
    params: Optional[Any] = None,
) -> np.ndarray:
    """Dispatch to one of the certified finite-cluster kernel constructors."""

    kernel_name = _normalize_kernel_name(kernel)
    prepared = _prepare_geometry(positions, symbols, params)
    if kernel_name == "bare":
        return _bare_from_prepared(prepared[4], prepared[6])
    if kernel_name == "ohno":
        return _ohno_from_prepared(prepared[4], prepared[6])
    return _gaussian_from_prepared(prepared[4], prepared[5], prepared[6])


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _geometry_sha256(
    positions: np.ndarray, symbols: Tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(_array_sha256(positions).encode("ascii"))
    for symbol in symbols:
        encoded = symbol.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class QEqGeometryProvenance:
    """SHA-256 bindings for geometry, constructed kernel, and parameter table."""

    geometry_sha256: str
    kernel_sha256: str
    parameter_table_sha256: str
    kernel_name: str
    hash_algorithm: str = "sha256"
    positions_unit: str = "Angstrom"

    def __post_init__(self) -> None:
        if self.hash_algorithm != "sha256":
            raise QEqGeometryError("only sha256 provenance is supported")
        if self.kernel_name not in SUPPORTED_QEQ_KERNELS:
            raise QEqGeometryError("provenance contains an unknown kernel name")
        for field_name in (
            "geometry_sha256",
            "kernel_sha256",
            "parameter_table_sha256",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise QEqGeometryError(
                    f"provenance {field_name} must be a lowercase SHA-256 digest"
                )
        if self.positions_unit != "Angstrom":
            raise QEqGeometryError("geometry positions_unit must be Angstrom")


@dataclass(frozen=True)
class QEqGeometryResult(QEqResult):
    """A :class:`QEqResult` bound to its finite-cluster construction inputs."""

    positions: Any = field(default=None)
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    parameter_table: Optional[QEqParameterTable] = field(default=None)
    provenance: Optional[QEqGeometryProvenance] = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._normalize_geometry_fields()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._normalize()
        self._normalize_geometry_fields()

    def _normalize_geometry_fields(self) -> None:
        if self.positions is None:
            raise QEqGeometryError("geometry result positions are required")
        positions = _readonly_float64(self.positions)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not np.isfinite(positions).all()
        ):
            raise QEqGeometryError(
                "geometry result positions must be finite with shape (natom, 3)"
            )
        symbols = _canonical_symbols(self.symbols)
        if len(symbols) != positions.shape[0]:
            raise QEqGeometryError(
                "geometry result symbols must match the number of positions"
            )
        if not isinstance(self.parameter_table, QEqParameterTable):
            raise QEqGeometryError(
                "geometry result parameter_table must be QEqParameterTable"
            )
        if not isinstance(self.provenance, QEqGeometryProvenance):
            raise QEqGeometryError(
                "geometry result provenance must be QEqGeometryProvenance"
            )
        if _geometry_sha256(positions, symbols) != self.provenance.geometry_sha256:
            raise QEqGeometryError("geometry provenance hash does not match inputs")
        if (
            _array_sha256(self.hardness_kernel)
            != self.provenance.kernel_sha256
        ):
            raise QEqGeometryError("kernel provenance hash does not match result")
        if (
            self.parameter_table.sha256
            != self.provenance.parameter_table_sha256
        ):
            raise QEqGeometryError(
                "parameter-table provenance hash does not match result"
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "symbols", symbols)


def solve_qeq_from_geometry(
    positions: ArrayLike,
    symbols: Sequence[str],
    total_charge: ArrayLike,
    *,
    kernel: str = "gaussian",
    params: Optional[Any] = None,
    symmetry_tol: float = 1e-10,
    constrained_eig_floor: float = 1e-12,
    max_condition: float = 1e12,
    residual_tol: float = 1e-8,
) -> QEqGeometryResult:
    """Construct ``chi``/``J`` from a cluster and call :func:`solve_qeq`.

    ``solve_qeq`` itself is not modified or bypassed.  In particular, a
    geometrically constructed kernel that is not positive definite on the
    fixed-charge subspace is rejected by the existing solver.

    The returned subclass preserves the complete :class:`QEqResult` API and
    attaches read-only positions, canonical symbols, the resolved parameter
    table, and SHA-256 provenance for geometry, kernel, and parameters.
    """

    kernel_name = _normalize_kernel_name(kernel)
    prepared = _prepare_geometry(positions, symbols, params)
    (
        position_array,
        canonical_symbols,
        table,
        electronegativity,
        hardness,
        widths,
        distances,
    ) = prepared
    if kernel_name == "bare":
        hardness_kernel = _bare_from_prepared(hardness, distances)
    elif kernel_name == "ohno":
        hardness_kernel = _ohno_from_prepared(hardness, distances)
    else:
        hardness_kernel = _gaussian_from_prepared(
            hardness, widths, distances
        )

    base = solve_qeq(
        electronegativity,
        hardness_kernel,
        total_charge=total_charge,
        symmetry_tol=symmetry_tol,
        constrained_eig_floor=constrained_eig_floor,
        max_condition=max_condition,
        residual_tol=residual_tol,
        units=QEqUnits(),
    )
    provenance = QEqGeometryProvenance(
        geometry_sha256=_geometry_sha256(
            position_array, canonical_symbols
        ),
        kernel_sha256=_array_sha256(hardness_kernel),
        parameter_table_sha256=table.sha256,
        kernel_name=kernel_name,
    )
    return QEqGeometryResult(
        charges=base.charges,
        total_charge=base.total_charge,
        electronegativity=base.electronegativity,
        hardness_kernel=base.hardness_kernel,
        lagrange_multiplier=base.lagrange_multiplier,
        stationarity=base.stationarity,
        energy=base.energy,
        linear_energy=base.linear_energy,
        quadratic_energy=base.quadratic_energy,
        energy_identity=base.energy_identity,
        diagnostics=base.diagnostics,
        units=base.units,
        symmetry_tol=base.symmetry_tol,
        constrained_eig_floor=base.constrained_eig_floor,
        max_condition=base.max_condition,
        residual_tol=base.residual_tol,
        positions=position_array,
        symbols=canonical_symbols,
        parameter_table=table,
        provenance=provenance,
    )


__all__ = [
    "COULOMB_EV_ANGSTROM",
    "MIN_INTERATOMIC_DISTANCE_ANGSTROM",
    "QEqGeometryError",
    "QEqGeometryProvenance",
    "QEqGeometryResult",
    "SUPPORTED_QEQ_KERNELS",
    "bare_qeq_kernel",
    "build_qeq_kernel",
    "gaussian_qeq_kernel",
    "ohno_qeq_kernel",
    "solve_qeq_from_geometry",
]
