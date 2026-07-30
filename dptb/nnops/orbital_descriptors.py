"""Dense orbital descriptors derived from a frozen single-particle H/S/D state.

This module provides small NumPy CPU reference kernels.  Eigenvectors follow
the :mod:`dptb.nnops.fixed_mu_operator` convention: columns of ``C`` solve
``H C = S C eps``, obey ``C.conj().T @ S @ C = I``, and form
``D = C @ diag(f) @ C.conj().T``.  The descriptors are frozen-spectrum
single-particle quantities; they are not charge-transfer free energies,
reorganization energies, or electronic couplings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    FixedMuScanResult,
    fermi_dirac,
)

ArrayLike = Any

MODULE_MAX_CONDITION = 1.0e12
MODULE_MAX_HERMITIAN_TOL = 1.0e-6


class OrbitalDescriptorError(FixedMuOperatorError):
    """Raised when orbital-descriptor inputs fail closed validation."""


def _readonly_array(value: ArrayLike, *, dtype: Any = None) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True, order="C")
    immutable = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(arr.shape)
    immutable.setflags(write=False)
    return immutable


def _freeze_arrays(obj: Any, field_names: Tuple[str, ...]) -> None:
    for field_name in field_names:
        object.__setattr__(obj, field_name, _readonly_array(getattr(obj, field_name)))


def _looks_like_torch_tensor(value: Any) -> bool:
    typ = type(value)
    module = getattr(typ, "__module__", "")
    return module == "torch" or module.startswith("torch.") or (
        hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")
    )


def _reject_torch_tensor(name: str, value: Any) -> None:
    if _looks_like_torch_tensor(value):
        raise OrbitalDescriptorError(
            f"{name} looks like a torch Tensor. This module is a NumPy CPU "
            "postprocess reference and will not detach/cpu tensors silently."
        )


def _as_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: Optional[float] = None,
) -> float:
    _reject_torch_tensor(name, value)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise OrbitalDescriptorError(f"{name} must be a scalar float, got {value!r}") from exc
    if not np.isfinite(out):
        raise OrbitalDescriptorError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise OrbitalDescriptorError(f"{name} must be positive, got {value!r}")
    if nonnegative and out < 0.0:
        raise OrbitalDescriptorError(f"{name} must be nonnegative, got {value!r}")
    if maximum is not None and out > maximum:
        raise OrbitalDescriptorError(
            f"{name}={out:g} exceeds the module ceiling {maximum:g}"
        )
    return out


def _validate_policy(
    *,
    eig_floor: float,
    max_condition: float,
    hermitian_tol: float,
    orthonormal_tol: float,
) -> Tuple[float, float, float, float]:
    return (
        _as_float("eig_floor", eig_floor, positive=True),
        _as_float(
            "max_condition",
            max_condition,
            positive=True,
            maximum=MODULE_MAX_CONDITION,
        ),
        _as_float(
            "hermitian_tol",
            hermitian_tol,
            nonnegative=True,
            maximum=MODULE_MAX_HERMITIAN_TOL,
        ),
        _as_float("orthonormal_tol", orthonormal_tol, nonnegative=True),
    )


def _validate_hermitian_matrix(
    name: str,
    value: ArrayLike,
    *,
    hermitian_tol: float,
) -> np.ndarray:
    _reject_torch_tensor(name, value)
    raw = np.asarray(value)
    if raw.ndim < 2 or raw.shape[-1] != raw.shape[-2] or raw.shape[-1] == 0:
        raise OrbitalDescriptorError(
            f"{name} must be a square matrix or stack, got shape {raw.shape}"
        )
    if 0 in raw.shape[:-2]:
        raise OrbitalDescriptorError(
            f"{name} leading shape {raw.shape[:-2]} must not contain a zero dimension"
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise OrbitalDescriptorError(f"{name} must be numeric")
    arr = np.asarray(raw, dtype=np.result_type(raw, np.float64))
    if not np.isfinite(arr).all():
        raise OrbitalDescriptorError(f"{name} must contain only finite values")
    adjoint = np.swapaxes(arr.conj(), -1, -2)
    if hermitian_tol == 0.0:
        ok = np.array_equal(arr, adjoint)
    else:
        ok = np.allclose(arr, adjoint, atol=hermitian_tol, rtol=hermitian_tol)
    if not ok:
        raise OrbitalDescriptorError(f"{name} must be Hermitian within hermitian_tol")
    return 0.5 * (arr + adjoint)


def _validate_overlap(
    overlap: ArrayLike,
    *,
    eig_floor: float,
    max_condition: float,
    hermitian_tol: float,
) -> np.ndarray:
    s = _validate_hermitian_matrix("overlap", overlap, hermitian_tol=hermitian_tol)
    eigenvalues = np.linalg.eigvalsh(s).real
    minimum = eigenvalues[..., 0]
    condition = eigenvalues[..., -1] / minimum
    if np.any(minimum <= eig_floor):
        raise OrbitalDescriptorError(
            "overlap must be positive definite above "
            f"eig_floor={eig_floor:g}; min={float(np.min(minimum)):.6g}"
        )
    if np.any(condition > max_condition):
        raise OrbitalDescriptorError(
            "overlap condition number exceeds "
            f"max_condition={max_condition:g}; max={float(np.max(condition)):.6g}"
        )
    return s


def _validate_mapping(name: str, value: ArrayLike, norb: int) -> np.ndarray:
    _reject_torch_tensor(name, value)
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != norb:
        raise OrbitalDescriptorError(
            f"{name} must be one-dimensional with length {norb}, got shape {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise OrbitalDescriptorError(f"{name} must contain integer labels")
    mapping = np.asarray(raw, dtype=np.int64)
    if np.any(mapping < 0):
        raise OrbitalDescriptorError(f"{name} labels must be nonnegative")
    return mapping


def _prepare_groups(
    ao_atom_index: ArrayLike,
    ao_l_index: Optional[ArrayLike],
    norb: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atom_map = _validate_mapping("ao_atom_index", ao_atom_index, norb)
    atom_labels = np.unique(atom_map)
    if ao_l_index is None:
        return (
            atom_map,
            atom_labels,
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
        )
    l_map = _validate_mapping("ao_l_index", ao_l_index, norb)
    resolved_labels = np.unique(np.stack((atom_map, l_map), axis=-1), axis=0)
    return atom_map, atom_labels, l_map, resolved_labels


def _reduce_last_ao(
    values: np.ndarray,
    atom_map: np.ndarray,
    atom_labels: np.ndarray,
    l_map: np.ndarray,
    resolved_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    atomic = np.stack(
        [np.sum(values[..., atom_map == atom], axis=-1) for atom in atom_labels],
        axis=-1,
    )
    if resolved_labels.size == 0:
        return atomic, np.empty(values.shape[:-1] + (0,), dtype=values.dtype)
    resolved = np.stack(
        [
            np.sum(
                values[..., (atom_map == atom) & (l_map == angular)],
                axis=-1,
            )
            for atom, angular in resolved_labels
        ],
        axis=-1,
    )
    return atomic, resolved


def _spectral_sqrt(overlap: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    scaled = eigenvectors * np.sqrt(eigenvalues)[..., None, :]
    return scaled @ np.swapaxes(eigenvectors.conj(), -1, -2)


def _validate_density_and_overlap(
    density_name: str,
    density: ArrayLike,
    overlap: ArrayLike,
    *,
    eig_floor: float,
    max_condition: float,
    hermitian_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    d = _validate_hermitian_matrix(
        density_name,
        density,
        hermitian_tol=hermitian_tol,
    )
    s = _validate_overlap(
        overlap,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    if d.shape != s.shape:
        raise OrbitalDescriptorError(
            f"{density_name} and overlap shapes must match, got {d.shape} and {s.shape}"
        )
    return d, s


@dataclass(frozen=True)
class PopulationResult:
    """Mulliken and symmetric-Löwdin electron populations.

    ``mulliken`` uses ``diag(D S)`` as introduced by Mulliken,
    J. Chem. Phys. **23**, 1833 (1955), DOI 10.1063/1.1740588.
    ``lowdin`` uses ``diag(S**(1/2) D S**(1/2))`` and the positive symmetric
    square root of Löwdin, J. Chem. Phys. **18**, 365 (1950),
    DOI 10.1063/1.1747632.
    """

    atom_labels: np.ndarray
    mulliken: np.ndarray
    lowdin: np.ndarray
    resolved_labels: np.ndarray
    resolved_mulliken: np.ndarray
    resolved_lowdin: np.ndarray
    electron_count: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(
            self,
            (
                "atom_labels",
                "mulliken",
                "lowdin",
                "resolved_labels",
                "resolved_mulliken",
                "resolved_lowdin",
                "electron_count",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDOSResult:
    """Mulliken-projected state weights and fragment band centers.

    State weights are the per-state form of Mulliken population analysis
    (Mulliken, J. Chem. Phys. **23**, 1833 (1955),
    DOI 10.1063/1.1740588).  The weighted first spectral moment is the usual
    projected-DOS band-center construction.  Stenlid and Žguns,
    ACS Energy Lett. **9**, 3608 (2024), DOI 10.1021/acsenergylett.4c01375,
    discuss DOS, charge, and bonding descriptors in the CIET context.
    """

    atom_labels: np.ndarray
    weights: np.ndarray
    band_centers: np.ndarray
    resolved_labels: np.ndarray
    resolved_weights: np.ndarray
    resolved_band_centers: np.ndarray
    window_weights: np.ndarray
    window_policy: str
    energy_window: np.ndarray

    def __post_init__(self) -> None:
        if self.window_policy not in ("all", "occupied", "energy"):
            raise OrbitalDescriptorError(
                f"invalid stored window_policy {self.window_policy!r}"
            )
        _freeze_arrays(
            self,
            (
                "atom_labels",
                "weights",
                "band_centers",
                "resolved_labels",
                "resolved_weights",
                "resolved_band_centers",
                "window_weights",
                "energy_window",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BondEnergyResult:
    """Atom-pair one-electron energy partition analogous to integrated COHP.

    Dronskowski and Blöchl, J. Phys. Chem. **97**, 8617 (1993),
    DOI 10.1021/j100135a014, derive COHP by partitioning band energy into
    orbital-pair Hamilton populations.  Here
    ``directed_energy[A,B] = sum_(mu in A,nu in B)
    Re[D_(nu,mu) H_(mu,nu)]``.  ``pair_energy`` stores each unordered pair
    once in its strict upper triangle as the sum of both directed blocks, so
    ``sum(onsite_energy) + sum(pair_energy) == Re Tr(D H)``.

    This partition is energy-zero dependent.  Under ``H -> H + c S`` its
    directed block changes by
    ``c sum_(mu in A,nu in B) Re[D_(nu,mu) S_(mu,nu)]``; it is not generally
    gauge invariant, while the total changes by ``c Tr(D S)``.
    """

    atom_labels: np.ndarray
    directed_energy: np.ndarray
    onsite_energy: np.ndarray
    pair_energy: np.ndarray
    trace_energy: np.ndarray
    closure_residual: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(
            self,
            (
                "atom_labels",
                "directed_energy",
                "onsite_energy",
                "pair_energy",
                "trace_energy",
                "closure_residual",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChargeResponseResult:
    """Fixed-spectrum fragment population derivatives ``d p / d mu``.

    The projection is the linear derivative of Mulliken/Löwdin population
    analysis (Mulliken, DOI 10.1063/1.1740588; Löwdin,
    DOI 10.1063/1.1747632) applied to a supplied ``dD/dmu``.  It omits
    self-consistent spectral, ionic, solvent, and double-layer response and is
    therefore not a complete electrochemical capacitance.
    """

    atom_labels: np.ndarray
    mulliken: np.ndarray
    lowdin: np.ndarray
    resolved_labels: np.ndarray
    resolved_mulliken: np.ndarray
    resolved_lowdin: np.ndarray
    total_response: np.ndarray

    def __post_init__(self) -> None:
        _freeze_arrays(
            self,
            (
                "atom_labels",
                "mulliken",
                "lowdin",
                "resolved_labels",
                "resolved_mulliken",
                "resolved_lowdin",
                "total_response",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrbitalDescriptorsResult:
    """Combined frozen-spectrum population, pDOS, bond, and response record."""

    occupations: np.ndarray
    density: np.ndarray
    populations: PopulationResult
    pdos: PDOSResult
    bond_energy: BondEnergyResult
    charge_response: Optional[ChargeResponseResult]

    def __post_init__(self) -> None:
        if not isinstance(self.populations, PopulationResult):
            raise OrbitalDescriptorError("populations must be a PopulationResult")
        if not isinstance(self.pdos, PDOSResult):
            raise OrbitalDescriptorError("pdos must be a PDOSResult")
        if not isinstance(self.bond_energy, BondEnergyResult):
            raise OrbitalDescriptorError("bond_energy must be a BondEnergyResult")
        if self.charge_response is not None and not isinstance(
            self.charge_response, ChargeResponseResult
        ):
            raise OrbitalDescriptorError(
                "charge_response must be a ChargeResponseResult or None"
            )
        _freeze_arrays(self, ("occupations", "density"))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FragmentPopulationScanResult:
    """Mulliken fragment population and fixed-spectrum response versus ``mu``."""

    mu_grid: np.ndarray
    atom_labels: np.ndarray
    populations: np.ndarray
    responses: np.ndarray
    resolved_labels: np.ndarray
    resolved_populations: np.ndarray
    resolved_responses: np.ndarray
    k_axis: Optional[int]

    def __post_init__(self) -> None:
        if self.k_axis is not None and (
            isinstance(self.k_axis, bool) or not isinstance(self.k_axis, (int, np.integer))
        ):
            raise OrbitalDescriptorError("k_axis must be an integer or None")
        _freeze_arrays(
            self,
            (
                "mu_grid",
                "atom_labels",
                "populations",
                "responses",
                "resolved_labels",
                "resolved_populations",
                "resolved_responses",
            ),
        )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def mulliken_lowdin_populations(
    density: ArrayLike,
    overlap: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    ao_l_index: Optional[ArrayLike] = None,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
) -> PopulationResult:
    """Compute atomic and optional ``(atom, l)`` Mulliken/Löwdin populations."""

    eig_floor, max_condition, hermitian_tol, _ = _validate_policy(
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=0.0,
    )
    d, s = _validate_density_and_overlap(
        "density",
        density,
        overlap,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    atom_map, atom_labels, l_map, resolved_labels = _prepare_groups(
        ao_atom_index,
        ao_l_index,
        d.shape[-1],
    )
    mulliken_ao = np.real(np.diagonal(d @ s, axis1=-2, axis2=-1))
    sqrt_s = _spectral_sqrt(s)
    lowdin_ao = np.real(
        np.diagonal(sqrt_s @ d @ sqrt_s, axis1=-2, axis2=-1)
    )
    mulliken, resolved_mulliken = _reduce_last_ao(
        mulliken_ao,
        atom_map,
        atom_labels,
        l_map,
        resolved_labels,
    )
    lowdin, resolved_lowdin = _reduce_last_ao(
        lowdin_ao,
        atom_map,
        atom_labels,
        l_map,
        resolved_labels,
    )
    electron_count = np.trace(d @ s, axis1=-2, axis2=-1).real
    return PopulationResult(
        atom_labels=atom_labels,
        mulliken=mulliken,
        lowdin=lowdin,
        resolved_labels=resolved_labels,
        resolved_mulliken=resolved_mulliken,
        resolved_lowdin=resolved_lowdin,
        electron_count=electron_count,
    )


def _validate_eigensystem(
    eigvals: ArrayLike,
    eigvecs: ArrayLike,
    overlap: ArrayLike,
    *,
    eig_floor: float,
    max_condition: float,
    hermitian_tol: float,
    orthonormal_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _reject_torch_tensor("eigvals", eigvals)
    _reject_torch_tensor("eigvecs", eigvecs)
    eps_raw = np.asarray(eigvals)
    c_raw = np.asarray(eigvecs)
    if not np.issubdtype(eps_raw.dtype, np.number) or np.iscomplexobj(eps_raw):
        raise OrbitalDescriptorError("eigvals must be a real numeric array")
    eps = np.asarray(eps_raw, dtype=np.float64)
    if eps.ndim < 1 or eps.shape[-1] == 0 or not np.isfinite(eps).all():
        raise OrbitalDescriptorError("eigvals must be finite with a nonempty state axis")
    if not np.issubdtype(c_raw.dtype, np.number):
        raise OrbitalDescriptorError("eigvecs must be numeric")
    c = np.asarray(c_raw, dtype=np.result_type(c_raw, np.float64))
    nstate = eps.shape[-1]
    expected_c_shape = eps.shape[:-1] + (nstate, nstate)
    if c.shape != expected_c_shape:
        raise OrbitalDescriptorError(
            "eigvecs must be a complete square column-eigenvector stack with "
            f"shape {expected_c_shape}, got {c.shape}"
        )
    if not np.isfinite(c).all():
        raise OrbitalDescriptorError("eigvecs must contain only finite values")
    s = _validate_overlap(
        overlap,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    if s.shape != c.shape:
        raise OrbitalDescriptorError(
            f"overlap shape must match eigvecs, got {s.shape} and {c.shape}"
        )
    identity = np.eye(nstate, dtype=np.result_type(c, s))
    gram = np.swapaxes(c.conj(), -1, -2) @ s @ c
    if not np.allclose(gram, identity, atol=orthonormal_tol, rtol=orthonormal_tol):
        error = float(np.max(np.abs(gram - identity)))
        raise OrbitalDescriptorError(
            "eigvecs must be S-orthonormal within orthonormal_tol; "
            f"max error={error:.6g}"
        )
    return eps, c, s


def _prepare_occupations(
    eigvals: np.ndarray,
    *,
    occupations: Optional[ArrayLike],
    mu: Optional[float],
    kT: Optional[float],
    spin_degeneracy: Optional[float],
) -> np.ndarray:
    explicit = occupations is not None
    tuple_items = (mu, kT, spin_degeneracy)
    any_tuple = any(value is not None for value in tuple_items)
    all_tuple = all(value is not None for value in tuple_items)
    if explicit and any_tuple:
        raise OrbitalDescriptorError(
            "supply occupations or (mu, kT, spin_degeneracy), not both"
        )
    if not explicit and not all_tuple:
        raise OrbitalDescriptorError(
            "supply occupations or the complete (mu, kT, spin_degeneracy) triple"
        )
    if explicit:
        _reject_torch_tensor("occupations", occupations)
        raw = np.asarray(occupations)
        if (
            raw.shape != eigvals.shape
            or not np.issubdtype(raw.dtype, np.number)
            or np.iscomplexobj(raw)
        ):
            raise OrbitalDescriptorError(
                f"occupations must be a real numeric array with shape {eigvals.shape}"
            )
        occ = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(occ).all() or np.any(occ < 0.0):
            raise OrbitalDescriptorError(
                "occupations must contain finite nonnegative values"
            )
        return occ
    try:
        occ, _ = fermi_dirac(
            eigvals,
            mu=mu,
            kT=kT,
            spin_degeneracy=spin_degeneracy,
        )
    except FixedMuOperatorError as exc:
        raise OrbitalDescriptorError(str(exc)) from exc
    return np.asarray(occ)


def _state_fragment_weights(
    eigvecs: np.ndarray,
    overlap: np.ndarray,
    atom_map: np.ndarray,
    atom_labels: np.ndarray,
    l_map: np.ndarray,
    resolved_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sc = overlap @ eigvecs
    ao_state = np.real(eigvecs.conj() * sc)
    state_ao = np.swapaxes(ao_state, -1, -2)
    return _reduce_last_ao(
        state_ao,
        atom_map,
        atom_labels,
        l_map,
        resolved_labels,
    )


def _prepare_window(
    eigvals: np.ndarray,
    occupations: np.ndarray,
    *,
    window: str,
    energy_window: Optional[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(window, str) or window not in ("all", "occupied", "energy"):
        raise OrbitalDescriptorError(
            "window must be exactly one of 'all', 'occupied', or 'energy'"
        )
    if window == "all":
        if energy_window is not None:
            raise OrbitalDescriptorError("energy_window is accepted only with window='energy'")
        return np.ones_like(eigvals), np.empty((0,), dtype=np.float64)
    if window == "occupied":
        if energy_window is not None:
            raise OrbitalDescriptorError("energy_window is accepted only with window='energy'")
        return occupations, np.empty((0,), dtype=np.float64)
    if energy_window is None:
        raise OrbitalDescriptorError("window='energy' requires energy_window=(emin, emax)")
    _reject_torch_tensor("energy_window", energy_window)
    raw = np.asarray(energy_window)
    if (
        raw.shape != (2,)
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise OrbitalDescriptorError("energy_window must be two finite real bounds")
    bounds = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise OrbitalDescriptorError(
            "energy_window must satisfy finite emin <= emax"
        )
    omega = ((eigvals >= bounds[0]) & (eigvals <= bounds[1])).astype(np.float64)
    return omega, bounds


def _band_centers(
    eigvals: np.ndarray,
    weights: np.ndarray,
    window_weights: np.ndarray,
) -> np.ndarray:
    weighted = weights * window_weights[..., :, None]
    denominator = np.sum(weighted, axis=-2)
    scale = np.sum(np.abs(weighted), axis=-2)
    threshold = 64.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
    if np.any(np.abs(denominator) <= threshold):
        raise OrbitalDescriptorError(
            "band center is undefined because a fragment has zero window weight"
        )
    numerator = np.sum(weighted * eigvals[..., :, None], axis=-2)
    return numerator / denominator


def fragment_pdos(
    eigvals: ArrayLike,
    eigvecs: ArrayLike,
    overlap: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    occupations: Optional[ArrayLike] = None,
    mu: Optional[float] = None,
    kT: Optional[float] = None,
    spin_degeneracy: Optional[float] = None,
    ao_l_index: Optional[ArrayLike] = None,
    window: str = "all",
    energy_window: Optional[ArrayLike] = None,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
    orthonormal_tol: float = 1e-8,
) -> PDOSResult:
    """Compute state-resolved Mulliken fragment weights and band centers.

    ``window='all'`` uses unit spectral weights, ``'occupied'`` uses the
    supplied or generated occupations, and ``'energy'`` uses an inclusive
    ``energy_window=(emin, emax)``.  The chosen policy is stored in the result.
    """

    eig_floor, max_condition, hermitian_tol, orthonormal_tol = _validate_policy(
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    eps, c, s = _validate_eigensystem(
        eigvals,
        eigvecs,
        overlap,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    occ = _prepare_occupations(
        eps,
        occupations=occupations,
        mu=mu,
        kT=kT,
        spin_degeneracy=spin_degeneracy,
    )
    atom_map, atom_labels, l_map, resolved_labels = _prepare_groups(
        ao_atom_index,
        ao_l_index,
        c.shape[-2],
    )
    weights, resolved_weights = _state_fragment_weights(
        c,
        s,
        atom_map,
        atom_labels,
        l_map,
        resolved_labels,
    )
    normalization = np.sum(weights, axis=-1)
    if not np.allclose(
        normalization,
        np.ones_like(normalization),
        atol=orthonormal_tol,
        rtol=orthonormal_tol,
    ):
        raise OrbitalDescriptorError(
            "complete atomic fragment weights do not sum to one per state"
        )
    omega, bounds = _prepare_window(
        eps,
        occ,
        window=window,
        energy_window=energy_window,
    )
    centers = _band_centers(eps, weights, omega)
    if resolved_labels.size == 0:
        resolved_centers = np.empty(eps.shape[:-1] + (0,), dtype=np.float64)
    else:
        resolved_centers = _band_centers(eps, resolved_weights, omega)
    return PDOSResult(
        atom_labels=atom_labels,
        weights=weights,
        band_centers=centers,
        resolved_labels=resolved_labels,
        resolved_weights=resolved_weights,
        resolved_band_centers=resolved_centers,
        window_weights=omega,
        window_policy=window,
        energy_window=bounds,
    )


def bond_energy_partition(
    density: ArrayLike,
    hamiltonian: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    hermitian_tol: float = 1e-8,
) -> BondEnergyResult:
    """Partition ``Re Tr(D H)`` into on-site and once-counted atom pairs."""

    hermitian_tol = _as_float(
        "hermitian_tol",
        hermitian_tol,
        nonnegative=True,
        maximum=MODULE_MAX_HERMITIAN_TOL,
    )
    d = _validate_hermitian_matrix(
        "density",
        density,
        hermitian_tol=hermitian_tol,
    )
    h = _validate_hermitian_matrix(
        "hamiltonian",
        hamiltonian,
        hermitian_tol=hermitian_tol,
    )
    if d.shape != h.shape:
        raise OrbitalDescriptorError(
            f"density and hamiltonian shapes must match, got {d.shape} and {h.shape}"
        )
    atom_map, atom_labels, _, _ = _prepare_groups(
        ao_atom_index,
        None,
        d.shape[-1],
    )
    orbital_energy = np.real(np.swapaxes(d, -1, -2) * h)
    rows = []
    for atom_a in atom_labels:
        cols = []
        mask_a = atom_map == atom_a
        for atom_b in atom_labels:
            mask_b = atom_map == atom_b
            cols.append(np.sum(orbital_energy[..., mask_a, :][..., mask_b], axis=(-2, -1)))
        rows.append(np.stack(cols, axis=-1))
    directed = np.stack(rows, axis=-2)
    onsite = np.diagonal(directed, axis1=-2, axis2=-1)
    pair = np.triu(directed + np.swapaxes(directed, -1, -2), k=1)
    trace_energy = np.trace(d @ h, axis1=-2, axis2=-1).real
    partition_total = np.sum(onsite, axis=-1) + np.sum(pair, axis=(-2, -1))
    return BondEnergyResult(
        atom_labels=atom_labels,
        directed_energy=directed,
        onsite_energy=onsite,
        pair_energy=pair,
        trace_energy=trace_energy,
        closure_residual=partition_total - trace_energy,
    )


def fragment_charge_response(
    density_response: ArrayLike,
    overlap: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    ao_l_index: Optional[ArrayLike] = None,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
) -> ChargeResponseResult:
    """Project a fixed-spectrum ``dD/dmu`` into fragment responses."""

    result = mulliken_lowdin_populations(
        density_response,
        overlap,
        ao_atom_index,
        ao_l_index=ao_l_index,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    return ChargeResponseResult(
        atom_labels=result.atom_labels,
        mulliken=result.mulliken,
        lowdin=result.lowdin,
        resolved_labels=result.resolved_labels,
        resolved_mulliken=result.resolved_mulliken,
        resolved_lowdin=result.resolved_lowdin,
        total_response=result.electron_count,
    )


def _density_from_eigensystem(eigvecs: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    return np.einsum(
        "...ni,...i,...mi->...nm",
        eigvecs,
        occupations,
        eigvecs.conj(),
        optimize=True,
    )


def orbital_descriptors(
    eigvals: ArrayLike,
    eigvecs: ArrayLike,
    overlap: ArrayLike,
    hamiltonian: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    density: Optional[ArrayLike] = None,
    density_response: Optional[ArrayLike] = None,
    occupations: Optional[ArrayLike] = None,
    mu: Optional[float] = None,
    kT: Optional[float] = None,
    spin_degeneracy: Optional[float] = None,
    ao_l_index: Optional[ArrayLike] = None,
    window: str = "all",
    energy_window: Optional[ArrayLike] = None,
    eig_floor: float = 1e-10,
    max_condition: float = 1e12,
    hermitian_tol: float = 1e-8,
    orthonormal_tol: float = 1e-8,
    consistency_atol: float = 1e-10,
    consistency_rtol: float = 1e-10,
) -> OrbitalDescriptorsResult:
    """Evaluate all four frozen-spectrum descriptor families.

    A supplied ``density`` is checked against ``C/f`` rather than trusted as
    an unrelated state.  ``density_response`` is optional because a zero-K
    occupation step has no ordinary pointwise derivative.
    """

    eig_floor, max_condition, hermitian_tol, orthonormal_tol = _validate_policy(
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    consistency_atol = _as_float(
        "consistency_atol", consistency_atol, nonnegative=True
    )
    consistency_rtol = _as_float(
        "consistency_rtol", consistency_rtol, nonnegative=True
    )
    eps, c, s = _validate_eigensystem(
        eigvals,
        eigvecs,
        overlap,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    occ = _prepare_occupations(
        eps,
        occupations=occupations,
        mu=mu,
        kT=kT,
        spin_degeneracy=spin_degeneracy,
    )
    rebuilt_density = _density_from_eigensystem(c, occ)
    if density is None:
        d = rebuilt_density
    else:
        d = _validate_hermitian_matrix(
            "density",
            density,
            hermitian_tol=hermitian_tol,
        )
        if d.shape != rebuilt_density.shape:
            raise OrbitalDescriptorError(
                f"density shape must be {rebuilt_density.shape}, got {d.shape}"
            )
        if not np.allclose(
            d,
            rebuilt_density,
            atol=consistency_atol,
            rtol=consistency_rtol,
        ):
            raise OrbitalDescriptorError(
                "supplied density is inconsistent with eigvecs and occupations"
            )
    h = _validate_hermitian_matrix(
        "hamiltonian",
        hamiltonian,
        hermitian_tol=hermitian_tol,
    )
    if h.shape != s.shape:
        raise OrbitalDescriptorError(
            f"hamiltonian shape must be {s.shape}, got {h.shape}"
        )
    populations = mulliken_lowdin_populations(
        d,
        s,
        ao_atom_index,
        ao_l_index=ao_l_index,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
    )
    pdos = fragment_pdos(
        eps,
        c,
        s,
        ao_atom_index,
        occupations=occ,
        ao_l_index=ao_l_index,
        window=window,
        energy_window=energy_window,
        eig_floor=eig_floor,
        max_condition=max_condition,
        hermitian_tol=hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    bond_energy = bond_energy_partition(
        d,
        h,
        ao_atom_index,
        hermitian_tol=hermitian_tol,
    )
    response = None
    if density_response is not None:
        response = fragment_charge_response(
            density_response,
            s,
            ao_atom_index,
            ao_l_index=ao_l_index,
            eig_floor=eig_floor,
            max_condition=max_condition,
            hermitian_tol=hermitian_tol,
        )
    return OrbitalDescriptorsResult(
        occupations=occ,
        density=d,
        populations=populations,
        pdos=pdos,
        bond_energy=bond_energy,
        charge_response=response,
    )


def _aggregate_scan_fragments(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    k_axis: Optional[int],
) -> np.ndarray:
    if k_axis is None:
        return values
    expanded_weights = weights[None, ..., None]
    return np.sum(values * expanded_weights, axis=k_axis + 1)


def fixed_mu_scan_fragment_populations(
    scan: FixedMuScanResult,
    eigvecs: ArrayLike,
    overlap: ArrayLike,
    ao_atom_index: ArrayLike,
    *,
    ao_l_index: Optional[ArrayLike] = None,
    orthonormal_tol: float = 1e-8,
    consistency_atol: float = 1e-10,
    consistency_rtol: float = 1e-10,
) -> FragmentPopulationScanResult:
    """Build Mulliken fragment population/response curves from a fixed-μ scan.

    The scan's stored occupations and occupation responses are contracted with
    state-resolved Mulliken weights from the supplied ``C/S``.  Its stored
    k-weight and k-axis policy is then applied exactly once.
    """

    if not isinstance(scan, FixedMuScanResult):
        raise OrbitalDescriptorError("scan must be a FixedMuScanResult")
    orthonormal_tol = _as_float(
        "orthonormal_tol", orthonormal_tol, nonnegative=True
    )
    consistency_atol = _as_float(
        "consistency_atol", consistency_atol, nonnegative=True
    )
    consistency_rtol = _as_float(
        "consistency_rtol", consistency_rtol, nonnegative=True
    )
    eps, c, s = _validate_eigensystem(
        scan.eigvals,
        eigvecs,
        overlap,
        eig_floor=scan.eig_floor,
        max_condition=scan.max_condition,
        hermitian_tol=scan.hermitian_tol,
        orthonormal_tol=orthonormal_tol,
    )
    if not np.allclose(
        eps,
        scan.eigvals,
        atol=consistency_atol,
        rtol=consistency_rtol,
    ):
        raise OrbitalDescriptorError("supplied eigensystem does not match scan eigvals")
    expected_shape = (scan.mu_grid.size,) + eps.shape
    if scan.occupations.shape != expected_shape:
        raise OrbitalDescriptorError(
            f"scan occupations must have shape {expected_shape}"
        )
    if scan.occupation_response.shape != expected_shape:
        raise OrbitalDescriptorError(
            f"scan occupation_response must have shape {expected_shape}"
        )
    atom_map, atom_labels, l_map, resolved_labels = _prepare_groups(
        ao_atom_index,
        ao_l_index,
        c.shape[-2],
    )
    state_weights, resolved_state_weights = _state_fragment_weights(
        c,
        s,
        atom_map,
        atom_labels,
        l_map,
        resolved_labels,
    )
    normalization = np.sum(state_weights, axis=-1)
    if not np.allclose(
        normalization,
        np.ones_like(normalization),
        atol=orthonormal_tol,
        rtol=orthonormal_tol,
    ):
        raise OrbitalDescriptorError(
            "complete atomic fragment weights do not sum to one per state"
        )
    populations = np.sum(
        scan.occupations[..., :, None] * state_weights[None, ...],
        axis=-2,
    )
    responses = np.sum(
        scan.occupation_response[..., :, None] * state_weights[None, ...],
        axis=-2,
    )
    if resolved_labels.size == 0:
        shape = populations.shape[:-1] + (0,)
        resolved_populations = np.empty(shape, dtype=np.float64)
        resolved_responses = np.empty(shape, dtype=np.float64)
    else:
        resolved_populations = np.sum(
            scan.occupations[..., :, None] * resolved_state_weights[None, ...],
            axis=-2,
        )
        resolved_responses = np.sum(
            scan.occupation_response[..., :, None]
            * resolved_state_weights[None, ...],
            axis=-2,
        )
    weights = np.asarray(scan.k_weights)
    populations = _aggregate_scan_fragments(
        populations,
        weights=weights,
        k_axis=scan.k_axis,
    )
    responses = _aggregate_scan_fragments(
        responses,
        weights=weights,
        k_axis=scan.k_axis,
    )
    resolved_populations = _aggregate_scan_fragments(
        resolved_populations,
        weights=weights,
        k_axis=scan.k_axis,
    )
    resolved_responses = _aggregate_scan_fragments(
        resolved_responses,
        weights=weights,
        k_axis=scan.k_axis,
    )
    return FragmentPopulationScanResult(
        mu_grid=scan.mu_grid,
        atom_labels=atom_labels,
        populations=populations,
        responses=responses,
        resolved_labels=resolved_labels,
        resolved_populations=resolved_populations,
        resolved_responses=resolved_responses,
        k_axis=scan.k_axis,
    )


__all__ = [
    "BondEnergyResult",
    "ChargeResponseResult",
    "FragmentPopulationScanResult",
    "OrbitalDescriptorError",
    "OrbitalDescriptorsResult",
    "PDOSResult",
    "PopulationResult",
    "bond_energy_partition",
    "fixed_mu_scan_fragment_populations",
    "fragment_charge_response",
    "fragment_pdos",
    "mulliken_lowdin_populations",
    "orbital_descriptors",
]
