"""Pinned dpnegf bridge for M1 zero-bias equilibrium transmission.

This module has no import-time dpnegf dependency.  Install the optional backend
at the audited commit without changing DeePTB's core dependency metadata:

``python -m pip install "git+https://github.com/deepmodeling/dpnegf.git@9b5da1296b7f0ca952b2e38742095d1d78b44434"``

The implementation intentionally follows only this narrow upstream path:
``surface_green.selfEnergy`` -> ``recursive_green_cal.recursive_gf`` ->
``DeviceProperty._cal_tc_`` (Caroli transmission).  It does not import or use
the legacy ``dpnegf.negf.transport`` module, finite-bias current, density, or
Poisson-NEGF functionality.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from .hs_provider import DenseHSProvider, TransportContractError, TransportConventions


DPNEGF_PIN_COMMIT = "9b5da1296b7f0ca952b2e38742095d1d78b44434"
DPNEGF_PIN_SHA = DPNEGF_PIN_COMMIT[:7]
DPNEGF_INSTALL_HINT = (
    'python -m pip install "git+https://github.com/deepmodeling/'
    f'dpnegf.git@{DPNEGF_PIN_COMMIT}"'
)


def _immutable_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class TransmissionProvenance:
    """Immutable provenance attached to every bridge transmission result."""

    matrix_hashes: Tuple[Tuple[str, str], ...]
    dpnegf_version: str
    dpnegf_sha: str
    energy_grid: np.ndarray
    eta_lead: float
    eta_device: float
    conventions: TransportConventions
    backend_path: str = (
        "surface_green.selfEnergy -> recursive_green_cal.recursive_gf -> "
        "DeviceProperty._cal_tc_"
    )

    def __post_init__(self) -> None:
        hashes = tuple((str(name), str(value)) for name, value in self.matrix_hashes)
        if not hashes or any(len(value) != 64 for _, value in hashes):
            raise TransportContractError("matrix_hashes must contain SHA-256 hex digests.")
        object.__setattr__(self, "matrix_hashes", hashes)
        object.__setattr__(
            self,
            "energy_grid",
            _immutable_array(self.energy_grid, dtype=np.dtype(np.float64)),
        )

    @property
    def matrix_hash_map(self) -> Mapping[str, str]:
        """Return a defensive mapping view of the recorded matrix hashes."""

        return dict(self.matrix_hashes)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


@dataclass(frozen=True)
class TransmissionResult:
    """k-resolved and k-averaged zero-bias Caroli transmission."""

    energy_grid: np.ndarray
    transmission: np.ndarray
    transmission_k: np.ndarray
    provenance: TransmissionProvenance

    def __post_init__(self) -> None:
        energy_grid = _immutable_array(self.energy_grid, dtype=np.dtype(np.float64))
        transmission = _immutable_array(self.transmission, dtype=np.dtype(np.float64))
        transmission_k = _immutable_array(self.transmission_k, dtype=np.dtype(np.float64))
        if energy_grid.ndim != 1 or transmission.shape != energy_grid.shape:
            raise TransportContractError("Transmission and energy_grid must be matching 1-D arrays.")
        if transmission_k.shape != (
            self.provenance.conventions.kpoints.shape[0],
            energy_grid.size,
        ):
            raise TransportContractError(
                "transmission_k must have shape [nk,nE] matching provenance."
            )
        if not np.isfinite(transmission).all() or not np.isfinite(transmission_k).all():
            raise TransportContractError("Transmission output contains NaN or infinity.")
        object.__setattr__(self, "energy_grid", energy_grid)
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "transmission_k", transmission_k)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()


def _load_dpnegf_backend():
    """Load only the audited modules and reject an unpinned dpnegf build."""

    try:
        package = importlib.import_module("dpnegf")
        surface_green = importlib.import_module("dpnegf.negf.surface_green")
        recursive_green = importlib.import_module("dpnegf.negf.recursive_green_cal")
        device_property = importlib.import_module("dpnegf.negf.device_property")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The optional dpnegf backend is required for zero-bias transport. "
            f"Install audited commit {DPNEGF_PIN_SHA} with:\n{DPNEGF_INSTALL_HINT}"
        ) from exc

    version = str(getattr(package, "__version__", "unknown"))
    if DPNEGF_PIN_SHA not in version:
        raise RuntimeError(
            f"Unsupported dpnegf version {version!r}; this bridge is pinned to "
            f"{DPNEGF_PIN_SHA}. Install it with:\n{DPNEGF_INSTALL_HINT}"
        )
    return (
        version,
        surface_green.selfEnergy,
        recursive_green.recursive_gf,
        device_property.DeviceProperty,
    )


def _validate_energy_grid(energy_grid: Any) -> np.ndarray:
    energies = np.asarray(energy_grid, dtype=np.float64)
    if energies.ndim != 1 or energies.size == 0:
        raise TransportContractError("energy_grid must be a non-empty 1-D array.")
    if not np.isfinite(energies).all():
        raise TransportContractError("energy_grid contains NaN or infinity.")
    if energies.size > 1 and np.any(np.diff(energies) <= 0.0):
        raise TransportContractError("energy_grid must be strictly increasing.")
    return energies


def _validate_eta(value: Any, name: str, *, strictly_positive: bool) -> float:
    eta = float(value)
    invalid = eta <= 0.0 if strictly_positive else eta < 0.0
    if not np.isfinite(eta) or invalid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise TransportContractError(f"{name} must be finite and {qualifier}.")
    return eta


def _caroli_via_device_property(
    g_trans: torch.Tensor,
    sigma_left: torch.Tensor,
    sigma_right: torch.Tensor,
    device_property_class: Any,
) -> torch.Tensor:
    """Evaluate the exact Caroli implementation isolated in DeviceProperty."""

    gamma_left = 1j * (sigma_left - sigma_left.mH)
    gamma_right = 1j * (sigma_right - sigma_right.mH)
    carrier = SimpleNamespace(
        g_trans=g_trans,
        lead_L=SimpleNamespace(gamma=gamma_left),
        lead_R=SimpleNamespace(gamma=gamma_right),
        rgf_device=torch.device("cpu"),
        cdtype=torch.complex128,
    )
    return device_property_class._cal_tc_(carrier)


def zero_bias_transmission(
    provider: DenseHSProvider,
    energy_grid: Any,
    *,
    eta_lead: float = 1.0e-5,
    eta_device: float = 0.0,
    surface_method: str = "Lopez-Sancho",
) -> TransmissionResult:
    """Compute frozen-H zero-bias transmission on every declared kpoint.

    ``energy_grid`` is relative to ``provider.conventions.E_ref`` and all
    energies are in eV.  The returned scalar channel count does not multiply
    by ``spin_degeneracy``; that convention is recorded in provenance so
    downstream conductance code can apply it exactly once.
    """

    if not isinstance(provider, DenseHSProvider):
        raise TransportContractError("provider must be a validated DenseHSProvider.")
    energies = _validate_energy_grid(energy_grid)
    eta_lead = _validate_eta(eta_lead, "eta_lead", strictly_positive=True)
    eta_device = _validate_eta(eta_device, "eta_device", strictly_positive=False)
    if surface_method != "Lopez-Sancho":
        raise TransportContractError(
            "M1 supports only the audited surface_method='Lopez-Sancho'."
        )

    version, self_energy, recursive_gf, device_property_class = _load_dpnegf_backend()
    energy_tensor = torch.tensor(energies, dtype=torch.complex128)
    transmission_k = []

    for kpoint in provider.conventions.kpoints:
        hd, sd, hl, su, sl, hu = provider.get_hs_device(
            kpoint,
            V=0.0,
            block_tridiagonal=True,
        )
        lead_left = provider.get_hs_lead(kpoint, "lead_L", 0.0)
        lead_right = provider.get_hs_lead(kpoint, "lead_R", 0.0)

        sigma_left = []
        sigma_right = []
        for energy in energies:
            se_left, _ = self_energy(
                lead_left[0],
                lead_left[1],
                lead_left[3],
                lead_left[4],
                float(energy),
                hDL=lead_left[2],
                sDL=lead_left[5],
                etaLead=eta_lead,
                E_ref=provider.conventions.E_ref,
                dtype=np.complex128,
                device="cpu",
                method=surface_method,
            )
            se_right, _ = self_energy(
                lead_right[0],
                lead_right[1],
                lead_right[3],
                lead_right[4],
                float(energy),
                hDL=lead_right[2],
                sDL=lead_right[5],
                etaLead=eta_lead,
                E_ref=provider.conventions.E_ref,
                dtype=np.complex128,
                device="cpu",
                method=surface_method,
            )
            sigma_left.append(se_left)
            sigma_right.append(se_right)

        sigma_left_tensor = torch.stack(sigma_left)
        sigma_right_tensor = torch.stack(sigma_right)
        answer = recursive_gf(
            energy_tensor,
            hl=hl,
            hd=hd,
            hu=hu,
            sd=sd,
            su=su,
            sl=sl,
            left_se=sigma_left_tensor,
            right_se=sigma_right_tensor,
            seP=None,
            E_ref=provider.conventions.E_ref,
            eta=eta_device,
            need_lesser=False,
            need_greater=False,
            need_gr_lc=False,
            keep_gr_left=False,
        )
        tc = _caroli_via_device_property(
            answer[0],
            sigma_left_tensor,
            sigma_right_tensor,
            device_property_class,
        )
        tc_numpy = tc.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(tc_numpy).all():
            raise TransportContractError("dpnegf returned non-finite transmission.")
        minimum = float(np.min(tc_numpy))
        if minimum < -1.0e-10:
            raise TransportContractError(
                f"Caroli transmission violates non-negativity: minimum {minimum:.3e}."
            )
        transmission_k.append(np.maximum(tc_numpy, 0.0))

    transmission_k_array = np.stack(transmission_k)
    transmission = np.einsum(
        "k,ke->e",
        provider.conventions.k_weights,
        transmission_k_array,
    )
    provenance = TransmissionProvenance(
        matrix_hashes=tuple(sorted(provider.matrix_hashes.items())),
        dpnegf_version=version,
        dpnegf_sha=DPNEGF_PIN_COMMIT,
        energy_grid=energies,
        eta_lead=eta_lead,
        eta_device=eta_device,
        conventions=provider.conventions,
    )
    return TransmissionResult(
        energy_grid=energies,
        transmission=transmission,
        transmission_k=transmission_k_array,
        provenance=provenance,
    )


__all__ = [
    "DPNEGF_INSTALL_HINT",
    "DPNEGF_PIN_COMMIT",
    "DPNEGF_PIN_SHA",
    "TransmissionProvenance",
    "TransmissionResult",
    "zero_bias_transmission",
]
