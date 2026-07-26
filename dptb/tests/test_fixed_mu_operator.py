import numpy as np
import pytest

import dptb.nnops.fixed_mu_operator as fixed_mu_module
from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    OverlapConditionError,
    fermi_dirac,
    fixed_mu_observables,
    generalized_bands,
    validate_conservation,
)


def test_generalized_bands_are_s_orthonormal_nonorthogonal():
    h = np.diag([-1.0, 0.25, 2.0])
    s = np.array(
        [
            [1.4, 0.1, 0.0],
            [0.1, 1.1, 0.05],
            [0.0, 0.05, 0.9],
        ],
        dtype=float,
    )
    bands = generalized_bands(h, s)
    c = bands.eigvecs
    assert np.all(np.diff(bands.eigvals) >= 0.0)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(h @ c, s @ c @ np.diag(bands.eigvals), atol=1e-10)


def test_generalized_bands_numpy_fallback_matches_scipy_path(monkeypatch):
    h = np.array([[0.2, 0.03], [0.03, 0.8]], dtype=float)
    s = np.array([[1.3, 0.06], [0.06, 0.9]], dtype=float)
    expected = generalized_bands(h, s)
    monkeypatch.setattr(fixed_mu_module, "_scipy_linalg", None)
    fallback = generalized_bands(h, s)
    np.testing.assert_allclose(fallback.eigvals, expected.eigvals, atol=1e-12)
    np.testing.assert_allclose(fallback.eigvecs.T @ s @ fallback.eigvecs, np.eye(2), atol=1e-12)


def test_overlap_fail_closed_for_nonpositive_or_ill_conditioned_s():
    h = np.eye(2)
    with pytest.raises(OverlapConditionError):
        generalized_bands(h, np.diag([1e-14, 1.0]), eig_floor=1e-10)
    with pytest.raises(OverlapConditionError):
        generalized_bands(h, np.diag([1e-8, 1.0]), eig_floor=1e-12, max_condition=1e4)


def test_fermi_dirac_zero_temperature_boundary_is_half_filled():
    occ, response = fermi_dirac(np.array([-1.0, 0.0, 1.0]), mu=0.0, kT=0.0, spin_degeneracy=2.0)
    np.testing.assert_allclose(occ, np.array([2.0, 1.0, 0.0]))
    np.testing.assert_allclose(response, np.zeros(3))


def test_fermi_dirac_extreme_finite_temperature_inputs_are_stable():
    occ, response = fermi_dirac(np.array([-1e6, 0.0, 1e6]), mu=0.0, kT=0.01, spin_degeneracy=1.0)
    assert np.isfinite(occ).all()
    assert np.isfinite(response).all()
    np.testing.assert_allclose(occ[[0, 2]], np.array([1.0, 0.0]), atol=1e-300)
    assert occ[1] == pytest.approx(0.5)


def test_density_and_energy_ledgers_conserve_nonorthogonal_hamiltonian():
    h = np.diag([-1.0, -0.5, 0.75])
    s = np.array(
        [
            [1.2, 0.05, 0.0],
            [0.05, 0.9, 0.02],
            [0.0, 0.02, 1.1],
        ],
        dtype=float,
    )
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    validate_conservation(result, atol=1e-10)
    np.testing.assert_allclose(result.electron_count, 4.0, atol=1e-10)
    np.testing.assert_allclose(result.conservation.electron_count_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.conservation.band_energy_residual, 0.0, atol=1e-10)


def test_finite_temperature_dos_and_density_response_match_finite_difference():
    h = np.diag([-0.5, 0.2, 1.0])
    s = np.eye(3)
    mu = 0.13
    step = 1e-6
    result = fixed_mu_observables(h, s, mu=mu, kT=0.2, spin_degeneracy=2.0)
    plus = fixed_mu_observables(h, s, mu=mu + step, kT=0.2, spin_degeneracy=2.0)
    minus = fixed_mu_observables(h, s, mu=mu - step, kT=0.2, spin_degeneracy=2.0)
    fd_n = (plus.electron_count - minus.electron_count) / (2.0 * step)
    fd_d = (plus.density - minus.density) / (2.0 * step)
    np.testing.assert_allclose(result.dos_like_response, fd_n, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result.density_response, fd_d, rtol=1e-6, atol=1e-8)


def test_grand_potential_uses_free_energy_minus_mu_n_without_double_subtraction():
    h = np.diag([-0.4, 0.1, 0.7])
    s = np.eye(3)
    mu = 0.23
    result = fixed_mu_observables(h, s, mu=mu, kT=0.15, spin_degeneracy=2.0)
    np.testing.assert_allclose(
        result.energies.grand_energy,
        result.energies.free_energy - mu * result.electron_count,
        atol=1e-12,
    )
    assert result.energies.entropy_term < 0.0


def test_batch_kpoint_weights_aggregate_only_over_requested_k_axis():
    h = np.zeros((2, 3, 2, 2), dtype=float)
    h[0, :, 0, 0] = [-1.0, -0.5, 0.1]
    h[0, :, 1, 1] = [0.4, 0.6, 0.8]
    h[1, :, 0, 0] = [-0.2, 0.2, 0.6]
    h[1, :, 1, 1] = [0.5, 0.7, 0.9]
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    weights = np.array([0.2, 0.3, 0.5])
    result = fixed_mu_observables(
        h,
        s,
        mu=0.0,
        kT=0.0,
        spin_degeneracy=2.0,
        k_weights=weights,
        k_axis=1,
    )
    assert result.electron_count.shape == (2,)
    assert result.density.shape == (2, 2, 2)
    np.testing.assert_allclose(result.electron_count, np.array([1.0, 0.4]))
    np.testing.assert_allclose(result.k_weights[0], weights)
    validate_conservation(result, atol=1e-12)


def test_batch_without_k_axis_keeps_samples_independent():
    h = np.zeros((2, 2, 2), dtype=float)
    h[0] = np.diag([-1.0, 0.5])
    h[1] = np.diag([-0.3, -0.1])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    np.testing.assert_allclose(result.k_weights, np.ones(2))
    np.testing.assert_allclose(result.electron_count, np.array([2.0, 4.0]))
    assert result.density.shape == (2, 2, 2)


def test_nonorthogonal_kweighted_grand_ledger_matches_mu_legendre_transform():
    h = np.array(
        [
            [[-0.7, 0.02], [0.02, 0.4]],
            [[-0.2, 0.03], [0.03, 0.8]],
        ],
        dtype=float,
    )
    s = np.array(
        [
            [[1.1, 0.04], [0.04, 0.95]],
            [[0.9, 0.02], [0.02, 1.2]],
        ],
        dtype=float,
    )
    mu = 0.11
    result = fixed_mu_observables(
        h,
        s,
        mu=mu,
        kT=0.07,
        spin_degeneracy=2.0,
        k_weights=np.array([2.0, 1.0]),
        k_axis=0,
    )
    validate_conservation(result, atol=1e-10)
    np.testing.assert_allclose(result.k_weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    np.testing.assert_allclose(
        result.energies.grand_energy,
        result.energies.band_energy + result.energies.entropy_term - mu * result.electron_count,
        atol=1e-12,
    )


def test_complex_soc_spinor_case_uses_explicit_spin_degeneracy_one():
    h = np.array([[0.1, 0.2j], [-0.2j, 0.4]], dtype=np.complex128)
    s = np.array([[1.2, 0.05 - 0.02j], [0.05 + 0.02j, 1.1]], dtype=np.complex128)
    result = fixed_mu_observables(h, s, mu=0.25, kT=0.05, spin_degeneracy=1.0)
    assert result.occupations.max() <= 1.0
    np.testing.assert_allclose(result.density, result.density.conj().T, atol=1e-12)
    validate_conservation(result, atol=1e-10)


def test_invalid_inputs_fail_closed():
    with pytest.raises(FixedMuOperatorError):
        fixed_mu_observables(np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2), mu=0.0)
    with pytest.raises(FixedMuOperatorError):
        fixed_mu_observables(np.array([[np.nan]]), np.eye(1), mu=0.0)
    with pytest.raises(FixedMuOperatorError):
        fixed_mu_observables(np.eye(2), np.eye(2), mu=0.0, spin_degeneracy=0.0)
