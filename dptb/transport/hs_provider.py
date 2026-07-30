"""Validated dense H/S input contract for optional zero-bias transport.

The provider deliberately exposes only the two duck-typed methods consumed by
the pinned dpnegf M1 path:

``get_hs_device(kpoint, V, block_tridiagonal, only_subblocks=False)``
    Return the dense device as one block (or the upstream dense form).
``get_hs_lead(kpoint, tab, v)``
    Return ``H00-v*S00, H01-v*S01, HD0, S00, S01, SD0``.

All Hamiltonians and energy references use eV.  Energies passed to dpnegf are
relative energies, with ``E_abs = E_relative + E_ref``.  This M1 provider is a
frozen-H, zero-bias contract; nonzero potentials are rejected rather than
silently presenting a finite-bias result.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch


class TransportContractError(ValueError):
    """Raised when transport H/S data or conventions fail closed."""


_SPIN_CONVENTIONS = {
    1.0: "explicit-spin/SOC",
    2.0: "non-SOC spin-degenerate",
}
_POTENTIAL_CONVENTION = "H(V)=H-V*S; E_abs=E_relative+E_ref"


def _immutable_array(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Return a defensive, bytes-backed, genuinely read-only array."""

    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _complex_stack(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != np.dtype(np.complex128):
        raise TransportContractError(f"{name} must have dtype complex128; got {array.dtype}.")
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise TransportContractError(
            f"{name} must have shape [nk,n,m] or [n,m]; got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise TransportContractError(f"{name} contains NaN or infinity.")
    return _immutable_array(array)


def _float_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise TransportContractError(f"{name} must be {ndim}-D; got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise TransportContractError(f"{name} contains NaN or infinity.")
    return _immutable_array(array)


def _validate_square(array: np.ndarray, name: str) -> int:
    if array.shape[-2] != array.shape[-1]:
        raise TransportContractError(f"{name} must be square; got {array.shape[-2:]}.")
    return int(array.shape[-1])


def _validate_hermitian(array: np.ndarray, name: str, atol: float) -> None:
    error = float(np.max(np.abs(array - np.swapaxes(array.conj(), -1, -2))))
    if error > atol:
        raise TransportContractError(
            f"{name} must be Hermitian within {atol:.3e}; max error is {error:.3e}."
        )


def _validate_positive_definite(array: np.ndarray, name: str, eig_floor: float) -> None:
    min_eig = float(np.min(np.linalg.eigvalsh(array)))
    if min_eig <= eig_floor:
        raise TransportContractError(
            f"{name} must be positive definite above {eig_floor:.3e}; "
            f"minimum eigenvalue is {min_eig:.3e}."
        )


def _array_hash(array: np.ndarray) -> str:
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TransportConventions:
    """Explicit AO, k, spin, energy-reference, and potential conventions."""

    energy_unit: str
    E_ref: Optional[float]
    ao_basis: str
    device_ao_labels: Tuple[str, ...]
    atom_orbital_map: Tuple[int, ...]
    m_order: str
    kpoints: np.ndarray
    k_weights: np.ndarray
    kpoint_convention: str
    spin_degeneracy: float
    spin_convention: str
    transport_direction: int
    potential_convention: str

    def __post_init__(self) -> None:
        if self.energy_unit != "eV":
            raise TransportContractError(
                f"energy_unit must be exactly 'eV'; got {self.energy_unit!r}."
            )
        if self.E_ref is None:
            raise TransportContractError("E_ref is required for relative-energy transport.")
        e_ref = float(self.E_ref)
        if not np.isfinite(e_ref):
            raise TransportContractError("E_ref must be finite.")
        object.__setattr__(self, "E_ref", e_ref)

        for field_name in ("ao_basis", "m_order", "kpoint_convention"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise TransportContractError(f"{field_name} must be a non-empty string.")

        labels = tuple(self.device_ao_labels)
        if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise TransportContractError("device_ao_labels must contain non-empty strings.")
        if len(set(labels)) != len(labels):
            raise TransportContractError("device_ao_labels must uniquely identify each device AO.")
        object.__setattr__(self, "device_ao_labels", labels)

        atom_map = tuple(int(index) for index in self.atom_orbital_map)
        if len(atom_map) != len(labels) or any(index < 0 for index in atom_map):
            raise TransportContractError(
                "atom_orbital_map must contain one non-negative atom index per device AO."
            )
        object.__setattr__(self, "atom_orbital_map", atom_map)

        kpoints = _float_array(self.kpoints, "kpoints", 2)
        if kpoints.shape[1:] != (3,) or kpoints.shape[0] == 0:
            raise TransportContractError(f"kpoints must have non-empty shape [nk,3]; got {kpoints.shape}.")
        weights = _float_array(self.k_weights, "k_weights", 1)
        if weights.shape != (kpoints.shape[0],):
            raise TransportContractError(
                f"k_weights must have shape ({kpoints.shape[0]},); got {weights.shape}."
            )
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise TransportContractError("k_weights must be non-negative with at least one positive value.")
        weight_sum = float(weights.sum())
        if not np.isclose(weight_sum, 1.0, atol=1.0e-12, rtol=0.0):
            raise TransportContractError(
                f"k_weights must already be normalized to 1; got {weight_sum:.17g}."
            )
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "k_weights", weights)

        spin = float(self.spin_degeneracy)
        expected_spin_convention = _SPIN_CONVENTIONS.get(spin)
        if expected_spin_convention is None:
            raise TransportContractError("spin_degeneracy must be 1 or 2.")
        if self.spin_convention != expected_spin_convention:
            raise TransportContractError(
                f"spin_degeneracy={spin:g} requires spin_convention="
                f"{expected_spin_convention!r}; got {self.spin_convention!r}."
            )
        object.__setattr__(self, "spin_degeneracy", spin)

        if self.transport_direction not in (0, 1, 2):
            raise TransportContractError("transport_direction must be Cartesian axis 0, 1, or 2.")
        if self.potential_convention != _POTENTIAL_CONVENTION:
            raise TransportContractError(
                "Unsupported potential_convention. Expected "
                f"{_POTENTIAL_CONVENTION!r}."
            )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class LeadPrincipalLayer:
    """Dense k-resolved principal-layer and device-lead coupling matrices."""

    h00: np.ndarray
    s00: np.ndarray
    h01: np.ndarray
    s01: np.ndarray
    hd0: np.ndarray
    sd0: np.ndarray
    ao_labels: Tuple[str, ...]
    hermitian_atol: float = 1.0e-10
    overlap_eig_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("h00", "s00", "h01", "s01", "hd0", "sd0"):
            arrays[name] = _complex_stack(getattr(self, name), name)
            object.__setattr__(self, name, arrays[name])

        n_lead = _validate_square(arrays["h00"], "h00")
        for name in ("s00", "h01", "s01"):
            if arrays[name].shape[-2:] != (n_lead, n_lead):
                raise TransportContractError(
                    f"{name} must have trailing shape {(n_lead, n_lead)}; "
                    f"got {arrays[name].shape[-2:]}."
                )
        nk = arrays["h00"].shape[0]
        if any(array.shape[0] != nk for array in arrays.values()):
            raise TransportContractError("All lead matrices must use the same nk.")
        if arrays["hd0"].shape[-1] != n_lead:
            raise TransportContractError(
                f"hd0 lead dimension must be {n_lead}; got {arrays['hd0'].shape[-1]}."
            )
        if arrays["sd0"].shape != arrays["hd0"].shape:
            raise TransportContractError(
                f"sd0 shape {arrays['sd0'].shape} must match hd0 {arrays['hd0'].shape}."
            )

        hermitian_atol = float(self.hermitian_atol)
        overlap_eig_floor = float(self.overlap_eig_floor)
        if not np.isfinite(hermitian_atol) or hermitian_atol <= 0.0:
            raise TransportContractError("hermitian_atol must be finite and positive.")
        if not np.isfinite(overlap_eig_floor) or overlap_eig_floor <= 0.0:
            raise TransportContractError("overlap_eig_floor must be finite and positive.")
        object.__setattr__(self, "hermitian_atol", hermitian_atol)
        object.__setattr__(self, "overlap_eig_floor", overlap_eig_floor)
        _validate_hermitian(arrays["h00"], "h00", hermitian_atol)
        _validate_hermitian(arrays["s00"], "s00", hermitian_atol)
        _validate_positive_definite(arrays["s00"], "s00", overlap_eig_floor)

        labels = tuple(self.ao_labels)
        if len(labels) != n_lead or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise TransportContractError(f"ao_labels must contain one non-empty label per lead AO ({n_lead}).")
        if len(set(labels)) != len(labels):
            raise TransportContractError("Lead ao_labels must uniquely identify each AO.")
        object.__setattr__(self, "ao_labels", labels)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class DenseHSProvider:
    """Frozen, validated dense H/S provider implementing dpnegf's duck protocol."""

    device_h: np.ndarray
    device_s: np.ndarray
    lead_left: LeadPrincipalLayer
    lead_right: LeadPrincipalLayer
    conventions: TransportConventions
    hermitian_atol: float = 1.0e-10
    overlap_eig_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        device_h = _complex_stack(self.device_h, "device_h")
        device_s = _complex_stack(self.device_s, "device_s")
        object.__setattr__(self, "device_h", device_h)
        object.__setattr__(self, "device_s", device_s)

        n_device = _validate_square(device_h, "device_h")
        if device_s.shape != device_h.shape:
            raise TransportContractError(
                f"device_s shape {device_s.shape} must match device_h {device_h.shape}."
            )
        nk = device_h.shape[0]
        if self.conventions.kpoints.shape[0] != nk:
            raise TransportContractError(
                f"Device matrices use nk={nk}, but conventions contain "
                f"{self.conventions.kpoints.shape[0]} kpoints."
            )
        if len(self.conventions.device_ao_labels) != n_device:
            raise TransportContractError(
                f"device_ao_labels has length {len(self.conventions.device_ao_labels)}, "
                f"but device dimension is {n_device}."
            )

        hermitian_atol = float(self.hermitian_atol)
        overlap_eig_floor = float(self.overlap_eig_floor)
        if not np.isfinite(hermitian_atol) or hermitian_atol <= 0.0:
            raise TransportContractError("hermitian_atol must be finite and positive.")
        if not np.isfinite(overlap_eig_floor) or overlap_eig_floor <= 0.0:
            raise TransportContractError("overlap_eig_floor must be finite and positive.")
        object.__setattr__(self, "hermitian_atol", hermitian_atol)
        object.__setattr__(self, "overlap_eig_floor", overlap_eig_floor)
        _validate_hermitian(device_h, "device_h", hermitian_atol)
        _validate_hermitian(device_s, "device_s", hermitian_atol)
        _validate_positive_definite(device_s, "device_s", overlap_eig_floor)

        for side, lead in (("left", self.lead_left), ("right", self.lead_right)):
            if lead.h00.shape[0] != nk:
                raise TransportContractError(
                    f"{side} lead uses nk={lead.h00.shape[0]}, but device uses nk={nk}."
                )
            if lead.hd0.shape[-2] != n_device:
                raise TransportContractError(
                    f"{side} hd0 device dimension must be {n_device}; "
                    f"got {lead.hd0.shape[-2]}."
                )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def matrix_hashes(self) -> Mapping[str, str]:
        """SHA-256 hashes over dtype, shape, and bytes of every input matrix."""

        matrices = {
            "device.H": self.device_h,
            "device.S": self.device_s,
            "lead_L.H00": self.lead_left.h00,
            "lead_L.S00": self.lead_left.s00,
            "lead_L.H01": self.lead_left.h01,
            "lead_L.S01": self.lead_left.s01,
            "lead_L.HD0": self.lead_left.hd0,
            "lead_L.SD0": self.lead_left.sd0,
            "lead_R.H00": self.lead_right.h00,
            "lead_R.S00": self.lead_right.s00,
            "lead_R.H01": self.lead_right.h01,
            "lead_R.S01": self.lead_right.s01,
            "lead_R.HD0": self.lead_right.hd0,
            "lead_R.SD0": self.lead_right.sd0,
        }
        return {name: _array_hash(array) for name, array in matrices.items()}

    def _k_index(self, kpoint: Any) -> int:
        if torch.is_tensor(kpoint):
            kpoint = kpoint.detach().cpu().numpy()
        point = np.asarray(kpoint, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise TransportContractError(f"kpoint must be a finite length-3 vector; got {point}.")
        errors = np.max(np.abs(self.conventions.kpoints - point), axis=1)
        matches = np.flatnonzero(errors <= 1.0e-8)
        if matches.size != 1:
            raise TransportContractError(
                f"kpoint {point.tolist()} must match exactly one declared kpoint; "
                f"found {matches.size}."
            )
        return int(matches[0])

    @staticmethod
    def _require_zero_potential(value: Any, name: str) -> None:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise TransportContractError(f"{name} must be finite.")
        if np.any(np.abs(array) > 1.0e-15):
            raise TransportContractError(
                f"{name} must be zero in the M1 zero-bias frozen-H provider."
            )

    def get_hs_device(
        self,
        kpoint: Any = (0.0, 0.0, 0.0),
        V: Any = 0.0,
        block_tridiagonal: bool = True,
        only_subblocks: bool = False,
    ):
        """Return device H/S using dpnegf's current consumption protocol."""

        index = self._k_index(kpoint)
        self._require_zero_potential(V, "V")
        if only_subblocks:
            return np.asarray([self.device_h.shape[-1]], dtype=np.int64)

        h = torch.tensor(self.device_h[index], dtype=torch.complex128)
        s = torch.tensor(self.device_s[index], dtype=torch.complex128)
        if block_tridiagonal:
            return [h], [s], [], [], [], []
        return h, s, [], [], [], []

    def get_hs_lead(self, kpoint: Any, tab: str, v: Any):
        """Return lead H/S and coupling using dpnegf's current tuple order."""

        index = self._k_index(kpoint)
        self._require_zero_potential(v, "v")
        aliases = {
            "L": self.lead_left,
            "left": self.lead_left,
            "lead_L": self.lead_left,
            "R": self.lead_right,
            "right": self.lead_right,
            "lead_R": self.lead_right,
        }
        try:
            lead = aliases[tab]
        except KeyError as exc:
            raise TransportContractError(
                f"tab must identify the left or right lead; got {tab!r}."
            ) from exc

        tensors = (
            lead.h00[index],
            lead.h01[index],
            lead.hd0[index],
            lead.s00[index],
            lead.s01[index],
            lead.sd0[index],
        )
        return tuple(torch.tensor(array, dtype=torch.complex128) for array in tensors)


__all__ = [
    "DenseHSProvider",
    "LeadPrincipalLayer",
    "TransportContractError",
    "TransportConventions",
]
