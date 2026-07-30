"""M1 acceptance tests for the optional pinned dpnegf transport bridge."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("dpnegf")

import dptb.transport.negf_bridge as bridge_module
from dptb.transport import (
    DPNEGF_PIN_COMMIT,
    DPNEGF_PIN_SHA,
    DenseHSProvider,
    LeadPrincipalLayer,
    TransportContractError,
    TransportConventions,
    zero_bias_transmission,
)


_E_REF = -13.638587951660156
_HOPPING = 0.6095958948135376
_POTENTIAL_CONVENTION = "H(V)=H-V*S; E_abs=E_relative+E_ref"


def _carbon_matrices():
    device_h = np.diag(np.full(4, _E_REF)).astype(np.complex128)
    device_h += np.diag(np.full(3, _HOPPING), 1)
    device_h += np.diag(np.full(3, _HOPPING), -1)
    device_s = np.eye(4, dtype=np.complex128)

    h00 = np.array(
        [[_E_REF, _HOPPING], [_HOPPING, _E_REF]],
        dtype=np.complex128,
    )
    s00 = np.eye(2, dtype=np.complex128)
    zeros_22 = np.zeros((2, 2), dtype=np.complex128)
    zeros_42 = np.zeros((4, 2), dtype=np.complex128)

    h01_left = zeros_22.copy()
    h01_left[0, 1] = _HOPPING
    hd0_left = zeros_42.copy()
    hd0_left[0, 1] = _HOPPING

    h01_right = zeros_22.copy()
    h01_right[1, 0] = _HOPPING
    hd0_right = zeros_42.copy()
    hd0_right[3, 0] = _HOPPING

    return {
        "device_h": device_h,
        "device_s": device_s,
        "left": (h00, s00, h01_left, zeros_22, hd0_left, zeros_42),
        "right": (h00, s00, h01_right, zeros_22, hd0_right, zeros_42),
    }


def _conventions(E_ref=_E_REF, *, energy_unit="eV"):
    return TransportConventions(
        energy_unit=energy_unit,
        E_ref=E_ref,
        ao_basis="C:2s",
        device_ao_labels=("D0:2s", "D1:2s", "D2:2s", "D3:2s"),
        atom_orbital_map=(0, 1, 2, 3),
        m_order="single s orbital",
        kpoints=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        k_weights=np.array([1.0], dtype=np.float64),
        kpoint_convention="fractional reciprocal coordinates",
        spin_degeneracy=2.0,
        spin_convention="non-SOC spin-degenerate",
        transport_direction=0,
        potential_convention=_POTENTIAL_CONVENTION,
    )


def _provider(**overrides):
    matrices = _carbon_matrices()
    device_h = overrides.pop("device_h", matrices["device_h"])
    device_s = overrides.pop("device_s", matrices["device_s"])
    conventions = overrides.pop("conventions", _conventions())
    if overrides:
        raise AssertionError(f"Unknown fixture overrides: {sorted(overrides)}")
    lead_left = LeadPrincipalLayer(
        h00=matrices["left"][0],
        s00=matrices["left"][1],
        h01=matrices["left"][2],
        s01=matrices["left"][3],
        hd0=matrices["left"][4],
        sd0=matrices["left"][5],
        ao_labels=("L0:2s", "L1:2s"),
    )
    lead_right = LeadPrincipalLayer(
        h00=matrices["right"][0],
        s00=matrices["right"][1],
        h01=matrices["right"][2],
        s01=matrices["right"][3],
        hd0=matrices["right"][4],
        sd0=matrices["right"][5],
        ao_labels=("R0:2s", "R1:2s"),
    )
    return DenseHSProvider(
        device_h=device_h,
        device_s=device_s,
        lead_left=lead_left,
        lead_right=lead_right,
        conventions=conventions,
    )


def _write_native_h5_fixture(path: Path) -> None:
    """Write the audited dpnegf carbon-chain H/S values in its native schema."""

    matrices = _carbon_matrices()
    with h5py.File(path / "HS_device.h5", "w") as handle:
        handle["HD"] = matrices["device_h"][np.newaxis, np.newaxis, ...]
        handle["SD"] = matrices["device_s"][np.newaxis, np.newaxis, ...]
        handle["kpoints"] = np.array([[0, 0, 0]], dtype=np.int32)
        handle["subblocks"] = np.array([4], dtype=np.int32)
        handle["block_tridiagonal"] = False

    for tab, key in (("L", "left"), ("R", "right")):
        with h5py.File(path / f"HS_lead_{tab}.h5", "w") as handle:
            handle["kpoints"] = np.array([[0, 0, 0]], dtype=np.int32)
            handle["kpoints_bloch"] = np.bytes_("None")
            handle["bloch_factor"] = np.bytes_("None")
            handle["useBloch"] = False
            for name, value in zip(
                ("HL", "SL", "HLL", "SLL", "HDL", "SDL"),
                (
                    matrices[key][0],
                    matrices[key][1],
                    matrices[key][2],
                    matrices[key][3],
                    matrices[key][4],
                    matrices[key][5],
                ),
            ):
                handle[f"{name}_real"] = value.real[np.newaxis, ...]
                handle[f"{name}_imag"] = value.imag[np.newaxis, ...]


def _native_h5_transmission(path: Path, energies: np.ndarray) -> np.ndarray:
    """Reference path through dpnegf's native H5 reader and numerical kernels."""

    from dpnegf.negf.device_property import DeviceProperty
    from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
    from dpnegf.negf.recursive_green_cal import recursive_gf
    from dpnegf.negf.surface_green import selfEnergy

    native = NEGFHamiltonianInit.__new__(NEGFHamiltonianInit)
    native.saved_HS_path = str(path)
    native.results_path = str(path)
    native.torch_device = torch.device("cpu")
    kpoint = np.array([0.0, 0.0, 0.0])

    device_h, device_s, _, _, _, _ = native.get_hs_device(
        kpoint,
        V=0.0,
        block_tridiagonal=False,
    )
    hd = list(device_h)
    sd = list(device_s)
    left = native.get_hs_lead(kpoint, "lead_L", 0.0)
    right = native.get_hs_lead(kpoint, "lead_R", 0.0)

    sigma_left = []
    sigma_right = []
    for energy in energies:
        se_left, _ = selfEnergy(
            left[0],
            left[1],
            left[3],
            left[4],
            float(energy),
            hDL=left[2],
            sDL=left[5],
            etaLead=1.0e-5,
            E_ref=_E_REF,
        )
        se_right, _ = selfEnergy(
            right[0],
            right[1],
            right[3],
            right[4],
            float(energy),
            hDL=right[2],
            sDL=right[5],
            etaLead=1.0e-5,
            E_ref=_E_REF,
        )
        sigma_left.append(se_left)
        sigma_right.append(se_right)

    sigma_left_tensor = torch.stack(sigma_left)
    sigma_right_tensor = torch.stack(sigma_right)
    answer = recursive_gf(
        torch.tensor(energies, dtype=torch.complex128),
        hl=[],
        hd=hd,
        hu=[],
        sd=sd,
        su=[],
        sl=[],
        left_se=sigma_left_tensor,
        right_se=sigma_right_tensor,
        E_ref=_E_REF,
        eta=0.0,
        need_gr_lc=False,
        keep_gr_left=False,
    )
    carrier = SimpleNamespace(
        g_trans=answer[0],
        lead_L=SimpleNamespace(gamma=1j * (sigma_left_tensor - sigma_left_tensor.mH)),
        lead_R=SimpleNamespace(gamma=1j * (sigma_right_tensor - sigma_right_tensor.mH)),
        rgf_device=torch.device("cpu"),
        cdtype=torch.complex128,
    )
    return DeviceProperty._cal_tc_(carrier).detach().cpu().numpy()


def test_carbon_chain_provider_matches_native_h5_all_40_points(tmp_path):
    energies = np.linspace(-0.2, 0.2, 40, dtype=np.float64)
    _write_native_h5_fixture(tmp_path)
    native = _native_h5_transmission(tmp_path, energies)
    result = zero_bias_transmission(
        _provider(),
        energies,
        eta_lead=1.0e-5,
        eta_device=0.0,
    )

    assert result.transmission.shape == (40,)
    assert abs(result.transmission[len(energies) // 2] - 1.0) < 1.0e-5
    max_delta = float(np.max(np.abs(result.transmission - native)))
    assert max_delta < 1.0e-8

    assert result.provenance.dpnegf_sha == DPNEGF_PIN_COMMIT
    assert DPNEGF_PIN_SHA in result.provenance.dpnegf_version
    assert len(result.provenance.matrix_hash_map) == 14
    np.testing.assert_array_equal(result.provenance.energy_grid, energies)
    assert result.provenance.eta_lead == 1.0e-5
    assert result.provenance.eta_device == 0.0
    assert result.provenance.conventions.E_ref == _E_REF
    assert result.provenance.conventions.energy_unit == "eV"
    assert not result.transmission.flags.writeable
    assert not result.provenance.energy_grid.flags.writeable


def test_single_level_wide_band_limit_matches_analytic_caroli():
    _, _, recursive_gf, device_property_class = bridge_module._load_dpnegf_backend()
    epsilon = 0.17
    gamma_left = 0.4
    gamma_right = 0.7
    energies = np.linspace(-1.0, 1.0, 81)
    batch = energies.size

    sigma_left = torch.full(
        (batch, 1, 1),
        -0.5j * gamma_left,
        dtype=torch.complex128,
    )
    sigma_right = torch.full(
        (batch, 1, 1),
        -0.5j * gamma_right,
        dtype=torch.complex128,
    )
    answer = recursive_gf(
        torch.tensor(energies, dtype=torch.complex128),
        hl=[],
        hd=[torch.tensor([[epsilon]], dtype=torch.complex128)],
        hu=[],
        sd=[torch.eye(1, dtype=torch.complex128)],
        su=[],
        sl=[],
        left_se=sigma_left,
        right_se=sigma_right,
        E_ref=0.0,
        eta=0.0,
        need_gr_lc=False,
        keep_gr_left=False,
    )
    numeric = bridge_module._caroli_via_device_property(
        answer[0],
        sigma_left,
        sigma_right,
        device_property_class,
    ).detach().cpu().numpy()
    analytic = gamma_left * gamma_right / (
        (energies - epsilon) ** 2
        + ((gamma_left + gamma_right) / 2.0) ** 2
    )
    np.testing.assert_allclose(numeric, analytic, atol=1.0e-12, rtol=1.0e-12)


def test_provider_rejects_wrong_shape():
    with pytest.raises(TransportContractError, match="device_s shape"):
        _provider(device_s=np.eye(3, dtype=np.complex128))


def test_provider_rejects_non_hermitian_input():
    device_h = _carbon_matrices()["device_h"].copy()
    device_h[0, 1] += 0.1j
    with pytest.raises(TransportContractError, match="device_h must be Hermitian"):
        _provider(device_h=device_h)


def test_provider_rejects_missing_energy_reference():
    with pytest.raises(TransportContractError, match="E_ref is required"):
        _conventions(E_ref=None)


def test_provider_rejects_wrong_energy_unit_and_nonfinite_matrix():
    with pytest.raises(TransportContractError, match="energy_unit"):
        _conventions(energy_unit="Hartree")

    device_h = _carbon_matrices()["device_h"].copy()
    device_h[0, 0] = np.nan
    with pytest.raises(TransportContractError, match="NaN or infinity"):
        _provider(device_h=device_h)


def test_provider_protocol_is_complex128_and_zero_bias_only():
    provider = _provider()
    hd, sd, hl, su, sl, hu = provider.get_hs_device(
        [0.0, 0.0, 0.0],
        V=0.0,
        block_tridiagonal=True,
    )
    assert len(hd) == len(sd) == 1
    assert hd[0].dtype == sd[0].dtype == torch.complex128
    assert hl == su == sl == hu == []
    lead = provider.get_hs_lead([0.0, 0.0, 0.0], "lead_L", 0.0)
    assert len(lead) == 6
    assert all(matrix.dtype == torch.complex128 for matrix in lead)

    with pytest.raises(TransportContractError, match="zero"):
        provider.get_hs_device([0.0, 0.0, 0.0], V=0.1)
    with pytest.raises(TransportContractError, match="zero"):
        provider.get_hs_lead([0.0, 0.0, 0.0], "lead_L", -0.1)


def test_missing_dpnegf_has_pinned_actionable_error(monkeypatch):
    real_import_module = importlib.import_module

    def hide_dpnegf(name, package=None):
        if name == "dpnegf":
            raise ModuleNotFoundError("dpnegf hidden for optional-dependency test")
        return real_import_module(name, package)

    monkeypatch.setattr(bridge_module.importlib, "import_module", hide_dpnegf)
    with pytest.raises(RuntimeError) as error:
        zero_bias_transmission(_provider(), np.array([0.0]))
    message = str(error.value)
    assert DPNEGF_PIN_SHA in message
    assert "pip install" in message

