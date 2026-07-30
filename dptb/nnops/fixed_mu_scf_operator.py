"""Fixed-mu dense molecular electrostatic self-consistency reference.

This module implements the intermediate approximation level
``frozen-H0 + self-consistent Mulliken electrostatics``.  It captures only the
electrostatic response through an overlap-consistent onsite potential; it does
not include occupation-dependent XC response or orbital rehybridization.
Accordingly, it sits above frozen-H fixed-mu postprocessing and below GC-DFT.

The caller supplies a finite-cluster Coulomb kernel in the same atomic-unit
convention as ``h0``, ``mu``, and ``kT``.  Periodic Ewald/PME electrostatics,
electrode boundary conditions, total energies, and forces are outside this v1
reference operator.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    FixedMuResult,
    fixed_mu_observables,
    validate_conservation,
)

ArrayLike = Any

_KERNEL_SYMMETRY_TOL = 1e-12
_ALLOWED_MIXING = ("pdiis", "linear")


class FixedMuSCFError(FixedMuOperatorError):
    """Base class for fail-closed fixed-mu electrostatic SCF errors."""


class FixedMuSCFConvergenceError(FixedMuSCFError):
    """Raised when the electrostatic fixed-point iteration does not converge."""

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        residual_history: ArrayLike,
    ) -> None:
        self.iterations = int(iterations)
        self.residual_history = _readonly_array(residual_history, dtype=np.float64)
        history_text = np.array2string(
            np.asarray(self.residual_history),
            precision=8,
            separator=", ",
            threshold=32,
        )
        super().__init__(
            f"{message}; iterations={self.iterations}; "
            f"residual_history={history_text}"
        )


@dataclass(frozen=True)
class FixedMuSCFResult:
    """Converged fixed-mu Mulliken-electrostatic reference result.

    ``q`` is the Mulliken net charge measured from the terminal
    :class:`FixedMuResult`, and ``phi`` is ``coulomb_kernel @ q``.  The final
    Hamiltonian was built from the preceding SCF input charge.  The last
    ``residual_history`` entry bounds the charge mismatch, while
    ``potential_residual = max(abs(K @ (q_out - q_in)))`` bounds the mismatch
    between the Hamiltonian-building potential and returned ``phi``.

    Array fields are defensive read-only copies backed by immutable byte
    buffers.  Pickle restoration calls :meth:`__post_init__` and restores the
    same non-writeable guarantee.
    """

    q: np.ndarray
    phi: np.ndarray
    iterations: int
    residual_history: np.ndarray
    fixed_mu_result: FixedMuResult
    mixing: str
    mixing_step: float
    n_history: int
    mixing_period: int
    max_iter: int
    charge_tol: float
    divergence_tol: float
    mu: float
    kT: float
    spin_degeneracy: float
    normalize_k_weights: bool
    eig_floor: float
    max_condition: float
    hermitian_tol: float
    potential_residual: float = 0.0
    potential_tol: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_mu_result, FixedMuResult):
            raise FixedMuSCFError("fixed_mu_result must be a FixedMuResult")
        mixing = _validate_mixing(self.mixing)
        iterations = _as_positive_int("iterations", self.iterations)
        max_iter = _as_positive_int("max_iter", self.max_iter)
        if iterations > max_iter:
            raise FixedMuSCFError(
                f"iterations={iterations} cannot exceed max_iter={max_iter}"
            )
        q = _readonly_array(self.q, dtype=np.float64)
        phi = _readonly_array(self.phi, dtype=np.float64)
        residual_history = _readonly_array(
            self.residual_history, dtype=np.float64
        )
        if q.ndim != 1 or phi.shape != q.shape:
            raise FixedMuSCFError(
                f"q and phi must be matching one-dimensional arrays, got "
                f"{q.shape} and {phi.shape}"
            )
        if not np.isfinite(q).all() or not np.isfinite(phi).all():
            raise FixedMuSCFError("q and phi must contain only finite values")
        if residual_history.shape != (iterations,):
            raise FixedMuSCFError(
                "residual_history must contain exactly one scalar per iteration"
            )
        if (
            not np.isfinite(residual_history).all()
            or np.any(residual_history < 0.0)
        ):
            raise FixedMuSCFError(
                "residual_history must be finite and nonnegative"
            )
        charge_tol = _as_float_scalar(
            "charge_tol", self.charge_tol, positive=True
        )
        if residual_history[-1] > charge_tol:
            raise FixedMuSCFError(
                "a converged result must end at residual <= charge_tol"
            )
        potential_residual = _as_float_scalar(
            "potential_residual", self.potential_residual, nonnegative=True
        )
        potential_tol = _as_float_scalar(
            "potential_tol", self.potential_tol, positive=True
        )
        if potential_residual > potential_tol:
            raise FixedMuSCFError(
                "a converged result must end at potential_residual <= potential_tol"
            )

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "residual_history", residual_history)
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(
            self, "mixing_step", _validate_mixing_step(self.mixing_step)
        )
        object.__setattr__(
            self, "n_history", _as_positive_int("n_history", self.n_history)
        )
        object.__setattr__(
            self,
            "mixing_period",
            _as_positive_int("mixing_period", self.mixing_period),
        )
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "charge_tol", charge_tol)
        object.__setattr__(self, "potential_residual", potential_residual)
        object.__setattr__(self, "potential_tol", potential_tol)
        object.__setattr__(
            self,
            "divergence_tol",
            _as_float_scalar(
                "divergence_tol", self.divergence_tol, positive=True
            ),
        )
        object.__setattr__(self, "mu", _as_float_scalar("mu", self.mu))
        object.__setattr__(
            self, "kT", _as_float_scalar("kT", self.kT, nonnegative=True)
        )
        object.__setattr__(
            self,
            "spin_degeneracy",
            _as_float_scalar(
                "spin_degeneracy", self.spin_degeneracy, positive=True
            ),
        )
        object.__setattr__(
            self,
            "normalize_k_weights",
            _as_bool("normalize_k_weights", self.normalize_k_weights),
        )
        object.__setattr__(
            self,
            "eig_floor",
            _as_float_scalar("eig_floor", self.eig_floor, positive=True),
        )
        object.__setattr__(
            self,
            "max_condition",
            _as_float_scalar(
                "max_condition", self.max_condition, positive=True
            ),
        )
        object.__setattr__(
            self,
            "hermitian_tol",
            _as_float_scalar(
                "hermitian_tol", self.hermitian_tol, nonnegative=True
            ),
        )

        nested = self.fixed_mu_result
        if nested.k_axis is not None or np.asarray(nested.eigvals).ndim != 1:
            raise FixedMuSCFError(
                "fixed_mu_result must come from the single-matrix, no-k-axis SCF path"
            )
        for field_name in (
            "mu",
            "kT",
            "spin_degeneracy",
            "normalize_k_weights",
            "eig_floor",
            "max_condition",
            "hermitian_tol",
        ):
            if getattr(self, field_name) != getattr(nested, field_name):
                raise FixedMuSCFError(
                    f"{field_name} does not match fixed_mu_result.{field_name}"
                )

    def __getstate__(self) -> Dict[str, Any]:
        return {
            field_info.name: getattr(self, field_info.name)
            for field_info in fields(self)
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        missing_certificate = {
            "potential_residual",
            "potential_tol",
        }.difference(state)
        if missing_certificate:
            missing = ", ".join(sorted(missing_certificate))
            raise FixedMuSCFError(
                "serialized fixed-mu SCF result lacks the potential-closure "
                f"certificate fields: {missing}"
            )
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self.__post_init__()


def _readonly_array(value: ArrayLike, *, dtype: Any = None) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True, order="C")
    immutable = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(
        arr.shape
    )
    immutable.setflags(write=False)
    return immutable


def _looks_like_torch_tensor(value: Any) -> bool:
    typ = type(value)
    module = getattr(typ, "__module__", "")
    return module == "torch" or module.startswith("torch.") or (
        hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    )


def _reject_torch_tensor(name: str, value: Any) -> None:
    if _looks_like_torch_tensor(value):
        raise FixedMuSCFError(
            f"{name} looks like a torch Tensor. This operator is a NumPy CPU "
            "reference and will not detach/cpu tensors silently."
        )


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise FixedMuSCFError(f"{name} must be a boolean, got {value!r}")
    return bool(value)


def _as_float_scalar(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    _reject_torch_tensor(name, value)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FixedMuSCFError(
            f"{name} must be a scalar float, got {value!r}"
        ) from exc
    if not np.isfinite(out):
        raise FixedMuSCFError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise FixedMuSCFError(f"{name} must be positive, got {value!r}")
    if nonnegative and out < 0.0:
        raise FixedMuSCFError(f"{name} must be nonnegative, got {value!r}")
    return out


def _as_positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise FixedMuSCFError(f"{name} must be a positive integer, got {value!r}")
    out = int(value)
    if out <= 0:
        raise FixedMuSCFError(f"{name} must be a positive integer, got {value!r}")
    return out


def _validate_mixing(value: Any) -> str:
    if not isinstance(value, str):
        raise FixedMuSCFError(
            f"mixing must be one of {_ALLOWED_MIXING}, got {value!r}"
        )
    out = value.lower()
    if out not in _ALLOWED_MIXING:
        raise FixedMuSCFError(
            f"mixing must be one of {_ALLOWED_MIXING}, got {value!r}"
        )
    return out


def _validate_mixing_step(value: Any) -> float:
    out = _as_float_scalar("mixing_step", value, positive=True)
    if out > 1.0:
        raise FixedMuSCFError(
            f"mixing_step must be in (0, 1], got {value!r}"
        )
    return out


def _validate_single_matrix_inputs(
    h0: ArrayLike,
    s: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    _reject_torch_tensor("h0", h0)
    _reject_torch_tensor("s", s)
    try:
        h_arr = np.asarray(h0)
        s_arr = np.asarray(s)
    except (TypeError, ValueError) as exc:
        raise FixedMuSCFError("h0 and s must be numeric arrays") from exc
    if h_arr.ndim != 2 or s_arr.ndim != 2:
        raise FixedMuSCFError(
            "fixed_mu_electrostatic_scf v1 does not support k-stacked or "
            f"leading-batch H/S inputs; expected two-dimensional matrices, "
            f"got {h_arr.shape} and {s_arr.shape}"
        )
    if (
        h_arr.shape != s_arr.shape
        or h_arr.shape[0] != h_arr.shape[1]
        or h_arr.shape[0] == 0
    ):
        raise FixedMuSCFError(
            f"h0 and s must be matching nonempty square matrices, got "
            f"{h_arr.shape} and {s_arr.shape}"
        )
    if not np.issubdtype(h_arr.dtype, np.number) or not np.issubdtype(
        s_arr.dtype, np.number
    ):
        raise FixedMuSCFError("h0 and s must be numeric arrays")
    dtype = np.result_type(h_arr, s_arr, np.float64)
    h_arr = np.asarray(h_arr, dtype=dtype)
    s_arr = np.asarray(s_arr, dtype=dtype)
    if not np.isfinite(h_arr).all() or not np.isfinite(s_arr).all():
        raise FixedMuSCFError("h0 and s must contain only finite values")
    return (
        0.5 * (h_arr + h_arr.conj().T),
        0.5 * (s_arr + s_arr.conj().T),
    )


def _validate_atomic_inputs(
    ao_atom_index: ArrayLike,
    reference_populations: ArrayLike,
    coulomb_kernel: ArrayLike,
    *,
    n_orb: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name, value in (
        ("ao_atom_index", ao_atom_index),
        ("reference_populations", reference_populations),
        ("coulomb_kernel", coulomb_kernel),
    ):
        _reject_torch_tensor(name, value)

    ao = np.asarray(ao_atom_index)
    if ao.ndim != 1 or ao.shape[0] != n_orb:
        raise FixedMuSCFError(
            f"ao_atom_index must be one-dimensional with length n_orb={n_orb}, "
            f"got shape {ao.shape}"
        )
    if not np.issubdtype(ao.dtype, np.integer):
        raise FixedMuSCFError("ao_atom_index must contain integer atom indices")
    ao = np.asarray(ao, dtype=np.int64)

    try:
        reference = np.asarray(reference_populations)
    except (TypeError, ValueError) as exc:
        raise FixedMuSCFError(
            "reference_populations must be a real numeric array"
        ) from exc
    if (
        reference.ndim != 1
        or reference.size == 0
        or not np.issubdtype(reference.dtype, np.number)
        or np.iscomplexobj(reference)
    ):
        raise FixedMuSCFError(
            "reference_populations must be a nonempty one-dimensional real "
            "numeric array"
        )
    reference = np.asarray(reference, dtype=np.float64)
    if not np.isfinite(reference).all():
        raise FixedMuSCFError(
            "reference_populations must contain only finite values"
        )
    n_atom = reference.size
    if np.any(ao < 0) or np.any(ao >= n_atom):
        raise FixedMuSCFError(
            f"ao_atom_index entries must lie in [0, {n_atom}), got "
            f"range [{int(np.min(ao))}, {int(np.max(ao))}]"
        )

    try:
        kernel = np.asarray(coulomb_kernel)
    except (TypeError, ValueError) as exc:
        raise FixedMuSCFError(
            "coulomb_kernel must be a real numeric array"
        ) from exc
    if (
        kernel.shape != (n_atom, n_atom)
        or not np.issubdtype(kernel.dtype, np.number)
        or np.iscomplexobj(kernel)
    ):
        raise FixedMuSCFError(
            f"coulomb_kernel must be a real {n_atom}x{n_atom} matrix, "
            f"got shape {kernel.shape}"
        )
    kernel = np.asarray(kernel, dtype=np.float64)
    if not np.isfinite(kernel).all():
        raise FixedMuSCFError(
            "coulomb_kernel must contain only finite values"
        )
    if not np.allclose(
        kernel,
        kernel.T,
        atol=_KERNEL_SYMMETRY_TOL,
        rtol=_KERNEL_SYMMETRY_TOL,
    ):
        asymmetry = float(np.max(np.abs(kernel - kernel.T)))
        raise FixedMuSCFError(
            "coulomb_kernel must be symmetric within "
            f"{_KERNEL_SYMMETRY_TOL:g}; max asymmetry={asymmetry:.6g}"
        )
    kernel = 0.5 * (kernel + kernel.T)
    return ao, reference, kernel


def _mulliken_net_charges(
    density: np.ndarray,
    overlap: np.ndarray,
    ao_atom_index: np.ndarray,
    reference_populations: np.ndarray,
) -> np.ndarray:
    orbital_populations = np.real(np.diag(density @ overlap))
    populations = np.bincount(
        ao_atom_index,
        weights=orbital_populations,
        minlength=reference_populations.size,
    )
    return reference_populations - populations


def _electrostatic_hamiltonian(
    h0: np.ndarray,
    overlap: np.ndarray,
    ao_atom_index: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Apply the S-consistent Mulliken potential to ``h0``.

    A uniform ``phi=c`` gives exactly ``H=h0-c*S``.  When ``S=I`` only the
    onsite diagonal is shifted.
    """

    phi_ao = phi[ao_atom_index]
    pair_potential = 0.5 * (phi_ao[:, None] + phi_ao[None, :])
    return h0 - overlap * pair_potential


def _pdiis_step(
    q: np.ndarray,
    residual: np.ndarray,
    residual_differences: list[np.ndarray],
    state_differences: list[np.ndarray],
    *,
    mixing_step: float,
) -> np.ndarray:
    """Periodic Pulay step ported from the historical Torch PDIIS routine."""

    if not residual_differences:
        return q + mixing_step * residual
    f_history = np.stack(residual_differences, axis=0)
    r_history = np.stack(state_differences, axis=0)
    gram = f_history @ f_history.T
    coefficients = np.linalg.pinv(gram, rcond=1e-12) @ (
        f_history @ residual
    )
    correction = (
        r_history.T + mixing_step * f_history.T
    ) @ coefficients
    return q + mixing_step * residual - correction


def fixed_mu_electrostatic_scf(
    h0: ArrayLike,
    s: ArrayLike,
    *,
    mu: float,
    kT: float = 0.0,
    ao_atom_index: ArrayLike,
    reference_populations: ArrayLike,
    coulomb_kernel: ArrayLike,
    mixing: str = "pdiis",
    mixing_step: float = 0.2,
    n_history: int = 6,
    mixing_period: int = 3,
    max_iter: int = 100,
    charge_tol: float = 1e-8,
    potential_tol: float = 1e-8,
    divergence_tol: float = 1e6,
    spin_degeneracy: float = 2.0,
    k_weights: Optional[ArrayLike] = None,
    k_axis: Optional[int] = None,
    normalize_k_weights: bool = True,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
) -> FixedMuSCFResult:
    """Solve fixed-mu Mulliken electrostatics for one dense molecular H/S pair.

    This is ``frozen-H0 + self-consistent Mulliken electrostatics``: it captures
    only electrostatic response through the S-consistent onsite update

    ``H[m,n] = H0[m,n] - 0.5*S[m,n]*(phi[a(m)] + phi[a(n)])``.

    It does not include occupation-dependent XC response or orbital
    rehybridization.  The approximation level is above frozen-H fixed-mu
    postprocessing and below GC-DFT.

    ``coulomb_kernel`` is a caller-provided finite-cluster kernel.  The caller
    owns the energy, charge, and length-unit convention.  v1 deliberately
    rejects k stacks and leading batches, and does not implement periodic
    Ewald/PME or electrode boundary conditions.

    Convergence is certified only when both
    ``max(abs(q_out - q_in)) <= charge_tol`` and
    ``max(abs(coulomb_kernel @ (q_out - q_in))) <= potential_tol``.  The second
    gate prevents a small charge residual from hiding a large Hamiltonian shift
    when the supplied kernel has a large norm.  Non-finite iterates, residuals
    above ``divergence_tol``, and exhaustion of ``max_iter`` raise
    :class:`FixedMuSCFConvergenceError` with the iteration count and residual
    history.
    """

    mixing = _validate_mixing(mixing)
    mixing_step = _validate_mixing_step(mixing_step)
    n_history = _as_positive_int("n_history", n_history)
    mixing_period = _as_positive_int("mixing_period", mixing_period)
    max_iter = _as_positive_int("max_iter", max_iter)
    charge_tol = _as_float_scalar("charge_tol", charge_tol, positive=True)
    potential_tol = _as_float_scalar(
        "potential_tol", potential_tol, positive=True
    )
    divergence_tol = _as_float_scalar(
        "divergence_tol", divergence_tol, positive=True
    )
    if divergence_tol <= charge_tol:
        raise FixedMuSCFError(
            "divergence_tol must be strictly greater than charge_tol"
        )
    if k_weights is not None or k_axis is not None:
        raise FixedMuSCFError(
            "fixed_mu_electrostatic_scf v1 does not support k_weights, "
            "k_axis, k-stacked inputs, or leading batches"
        )

    h0_arr, s_arr = _validate_single_matrix_inputs(h0, s)
    ao, reference, kernel = _validate_atomic_inputs(
        ao_atom_index,
        reference_populations,
        coulomb_kernel,
        n_orb=h0_arr.shape[0],
    )

    fixed_mu_kwargs = {
        "mu": mu,
        "kT": kT,
        "spin_degeneracy": spin_degeneracy,
        "k_weights": None,
        "k_axis": None,
        "normalize_k_weights": normalize_k_weights,
        "eig_floor": eig_floor,
        "max_condition": max_condition,
        "hermitian_tol": hermitian_tol,
    }
    initial_result = fixed_mu_observables(h0, s, **fixed_mu_kwargs)
    q = _mulliken_net_charges(
        initial_result.density, s_arr, ao, reference
    )

    residual_history: list[float] = []
    residual_differences: list[np.ndarray] = []
    state_differences: list[np.ndarray] = []
    previous_q: Optional[np.ndarray] = None
    previous_residual: Optional[np.ndarray] = None

    for iteration in range(1, max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            phi_in = kernel @ q
            h_final = _electrostatic_hamiltonian(
                h0_arr, s_arr, ao, phi_in
            )
        if not np.isfinite(phi_in).all() or not np.isfinite(h_final).all():
            raise FixedMuSCFConvergenceError(
                "fixed-mu electrostatic SCF produced a non-finite potential "
                "or Hamiltonian",
                iterations=iteration,
                residual_history=residual_history,
            )

        current_result = fixed_mu_observables(
            h_final, s_arr, **fixed_mu_kwargs
        )
        q_out = _mulliken_net_charges(
            current_result.density, s_arr, ao, reference
        )
        residual_vector = q_out - q
        residual = float(np.max(np.abs(residual_vector)))
        with np.errstate(over="ignore", invalid="ignore"):
            potential_residual = float(
                np.max(np.abs(kernel @ residual_vector))
            )
        residual_history.append(residual)

        if (
            not np.isfinite(residual)
            or not np.isfinite(potential_residual)
            or not np.isfinite(q_out).all()
        ):
            raise FixedMuSCFConvergenceError(
                "fixed-mu electrostatic SCF produced a non-finite charge, "
                "charge residual, or potential residual",
                iterations=iteration,
                residual_history=residual_history,
            )
        if residual > divergence_tol:
            raise FixedMuSCFConvergenceError(
                f"fixed-mu electrostatic SCF residual {residual:.6g} "
                f"exceeded divergence_tol={divergence_tol:.6g}",
                iterations=iteration,
                residual_history=residual_history,
            )
        if residual <= charge_tol and potential_residual <= potential_tol:
            terminal_result = fixed_mu_observables(
                h_final, s_arr, **fixed_mu_kwargs
            )
            final_q = _mulliken_net_charges(
                terminal_result.density, s_arr, ao, reference
            )
            final_phi = kernel @ final_q
            validate_conservation(
                terminal_result,
                h=h_final,
                s=s_arr,
                expected_mu=terminal_result.mu,
                expected_kT=terminal_result.kT,
                expected_spin_degeneracy=terminal_result.spin_degeneracy,
                expected_k_axis=None,
                expected_normalize_k_weights=terminal_result.normalize_k_weights,
                expected_eig_floor=terminal_result.eig_floor,
                expected_max_condition=terminal_result.max_condition,
                expected_hermitian_tol=terminal_result.hermitian_tol,
            )
            return FixedMuSCFResult(
                q=final_q,
                phi=final_phi,
                iterations=iteration,
                residual_history=np.asarray(
                    residual_history, dtype=np.float64
                ),
                fixed_mu_result=terminal_result,
                mixing=mixing,
                mixing_step=mixing_step,
                n_history=n_history,
                mixing_period=mixing_period,
                max_iter=max_iter,
                charge_tol=charge_tol,
                divergence_tol=divergence_tol,
                mu=terminal_result.mu,
                kT=terminal_result.kT,
                spin_degeneracy=terminal_result.spin_degeneracy,
                normalize_k_weights=terminal_result.normalize_k_weights,
                eig_floor=terminal_result.eig_floor,
                max_condition=terminal_result.max_condition,
                hermitian_tol=terminal_result.hermitian_tol,
                potential_residual=potential_residual,
                potential_tol=potential_tol,
            )

        if previous_q is not None and previous_residual is not None:
            residual_differences.append(residual_vector - previous_residual)
            state_differences.append(q - previous_q)
            if len(residual_differences) > n_history:
                residual_differences.pop(0)
                state_differences.pop(0)

        if iteration == max_iter:
            break
        if mixing == "pdiis" and iteration % mixing_period == 0:
            q_next = _pdiis_step(
                q,
                residual_vector,
                residual_differences,
                state_differences,
                mixing_step=mixing_step,
            )
        else:
            q_next = q + mixing_step * residual_vector
        if not np.isfinite(q_next).all():
            raise FixedMuSCFConvergenceError(
                "fixed-mu electrostatic SCF mixing produced a non-finite "
                "charge iterate",
                iterations=iteration,
                residual_history=residual_history,
            )
        previous_q = q.copy()
        previous_residual = residual_vector.copy()
        q = q_next

    raise FixedMuSCFConvergenceError(
        f"fixed-mu electrostatic SCF did not converge within max_iter={max_iter}; "
        f"last_potential_residual={potential_residual:.6g}",
        iterations=max_iter,
        residual_history=residual_history,
    )


__all__ = [
    "FixedMuSCFConvergenceError",
    "FixedMuSCFError",
    "FixedMuSCFResult",
    "fixed_mu_electrostatic_scf",
]
