import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

import dptb.nnops.fixed_mu_operator as fixed_mu_module
from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    OverlapConditionError,
    fermi_dirac,
    fixed_mu_observables,
    fixed_mu_observables_from_torch,
    generalized_bands,
    validate_conservation,
)


def _replace_result(result, **changes):
    return replace(result, validation_context=getattr(result, "_validation_context", None), **changes)


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


def test_fixed_mu_result_arrays_are_read_only_defensive_copies():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)

    with pytest.raises(ValueError, match="read-only"):
        result.density[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.energies.band_energy[...] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.conservation.electron_count_residual[...] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        result._validation_context.hamiltonian[0, 0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        result._validation_context.hamiltonian.setflags(write=True)

    density_alias = result.density.copy()
    rebuilt = _replace_result(result, density=density_alias)
    density_alias[0, 0] += 10.0
    np.testing.assert_allclose(rebuilt.density, result.density, atol=1e-12)


def test_validate_conservation_recomputes_density_and_ledgers_from_hs_snapshots():
    h = np.array([[-1.0, 0.05], [0.05, 0.6]], dtype=float)
    s = np.array([[1.1, 0.02], [0.02, 0.9]], dtype=float)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)

    bad_density = result.density.copy()
    bad_density[0, 0] += 0.1
    with pytest.raises(FixedMuOperatorError, match="density"):
        validate_conservation(_replace_result(result, density=bad_density), atol=1e-10)

    with pytest.raises(FixedMuOperatorError, match="electron_count"):
        validate_conservation(_replace_result(result, electron_count=result.electron_count + 0.5), atol=1e-10)

    bad_energies = replace(result.energies, band_energy=result.energies.band_energy + 0.5)
    with pytest.raises(FixedMuOperatorError, match="energy band_energy"):
        validate_conservation(_replace_result(result, energies=bad_energies), atol=1e-10)

    bad_conservation = replace(
        result.conservation,
        band_energy_from_density=result.conservation.band_energy_from_density + 0.5,
    )
    with pytest.raises(FixedMuOperatorError, match="conservation band_energy_from_density"):
        validate_conservation(_replace_result(result, conservation=bad_conservation), atol=1e-10)

    with pytest.raises(FixedMuOperatorError, match="H/S context"):
        validate_conservation(replace(result), atol=1e-10)


def test_fixed_mu_validation_context_is_excluded_from_public_serialization():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)

    public = asdict(result)
    assert "_validation_context" not in public
    assert "validation_context" not in public
    assert "_hamiltonian" not in public
    assert "_overlap" not in public
    assert "_validation_context" not in result.to_dict()

    restored = pickle.loads(pickle.dumps(result))
    with pytest.raises(FixedMuOperatorError, match="H/S context"):
        validate_conservation(restored, atol=1e-10)
    validate_conservation(restored, h=h, s=s, atol=1e-10)


def test_validate_conservation_revalidates_k_axis_and_weight_contract():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.25, 0.75])])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    result = fixed_mu_observables(
        h,
        s,
        mu=0.0,
        kT=0.0,
        spin_degeneracy=2.0,
        k_weights=np.array([0.25, 0.75]),
        k_axis=0,
    )

    bad_weights = result.k_weights.copy()
    bad_weights[0] = -0.1
    with pytest.raises(FixedMuOperatorError, match="nonnegative"):
        validate_conservation(_replace_result(result, k_weights=bad_weights), atol=1e-10)

    bad_shape_weights = np.ones((2, 1))
    with pytest.raises(FixedMuOperatorError, match="stored k_weights shape"):
        validate_conservation(_replace_result(result, k_weights=bad_shape_weights), atol=1e-10)

    unnormalized_weights = np.array([2.0, 1.0])
    with pytest.raises(FixedMuOperatorError, match="sum to one"):
        validate_conservation(_replace_result(result, k_weights=unnormalized_weights), atol=1e-10)

    with pytest.raises(FixedMuOperatorError, match="k_axis must be an integer"):
        validate_conservation(_replace_result(result, k_axis=object()), atol=1e-10)


def test_positive_unnormalized_k_weights_are_preserved_with_provenance():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.25, 0.75])])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    weights = np.array([2.0, 1.0])
    result = fixed_mu_observables(
        h,
        s,
        mu=0.1,
        kT=0.05,
        spin_degeneracy=2.0,
        k_weights=weights,
        k_axis=0,
        normalize_k_weights=False,
    )

    assert result.normalize_k_weights is False
    np.testing.assert_allclose(result.k_weights, weights, atol=1e-12)
    validate_conservation(result, atol=1e-10)
    np.testing.assert_allclose(
        result.energies.band_grand_energy,
        result.energies.band_free_energy - result.mu * result.electron_count,
        atol=1e-12,
    )


def test_validate_conservation_rechecks_hs_context_and_nested_types():
    h = np.array([[-1.0, 0.05], [0.05, 0.6]], dtype=float)
    s = np.array([[1.1, 0.02], [0.02, 0.9]], dtype=float)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)

    bad_h = h.copy()
    bad_h[0, 1] += 0.2
    bad_h[1, 0] += 0.2
    with pytest.raises(FixedMuOperatorError, match="generalized eigen residual"):
        validate_conservation(result, h=bad_h, s=s, atol=1e-10)

    bad_s = s.copy()
    bad_s[0, 1] += 0.2
    with pytest.raises(FixedMuOperatorError, match="Hermitian"):
        validate_conservation(result, h=h, s=bad_s, atol=1e-10)

    with pytest.raises(FixedMuOperatorError, match="energies must be an EnergyLedger"):
        _replace_result(result, energies=object())
    with pytest.raises(FixedMuOperatorError, match="conservation must be a ConservationLedger"):
        _replace_result(result, conservation=object())


def test_complex_soc_kpoint_batch_validates_fixed_mu_contract():
    h = np.array(
        [
            [[0.1, 0.2j], [-0.2j, 0.4]],
            [[-0.2, 0.05 + 0.03j], [0.05 - 0.03j, 0.7]],
        ],
        dtype=np.complex128,
    )
    s = np.array(
        [
            [[1.2, 0.05 - 0.02j], [0.05 + 0.02j, 1.1]],
            [[1.1, -0.04j], [0.04j, 1.3]],
        ],
        dtype=np.complex128,
    )
    result = fixed_mu_observables(
        h,
        s,
        mu=0.25,
        kT=0.05,
        spin_degeneracy=1.0,
        k_weights=np.array([0.4, 0.6]),
        k_axis=0,
    )

    assert result.density.shape == (2, 2)
    assert np.iscomplexobj(result.density)
    validate_conservation(result, atol=1e-9)


def test_validate_conservation_recomputes_fixed_mu_occupations_and_energy_closures():
    h = np.diag([-0.4, 0.1, 0.7])
    s = np.eye(3)
    result = fixed_mu_observables(h, s, mu=0.23, kT=0.15, spin_degeneracy=2.0)

    bad_occ = result.occupations.copy()
    bad_occ[0] += 0.1
    with pytest.raises(FixedMuOperatorError, match="occupations"):
        validate_conservation(_replace_result(result, occupations=bad_occ), atol=1e-10)

    bad_eigvals = result.eigvals.copy()
    bad_eigvals[0] += 0.1
    with pytest.raises(FixedMuOperatorError, match="generalized eigen residual"):
        validate_conservation(_replace_result(result, eigvals=bad_eigvals), atol=1e-10)

    for field_name in ("minus_t_s", "band_free_energy", "band_grand_energy"):
        bad_energies = replace(result.energies, **{field_name: getattr(result.energies, field_name) + 0.25})
        with pytest.raises(FixedMuOperatorError, match=f"energy {field_name}"):
            validate_conservation(_replace_result(result, energies=bad_energies), atol=1e-10)


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


def test_band_grand_energy_uses_band_free_energy_minus_mu_n_without_double_subtraction():
    h = np.diag([-0.4, 0.1, 0.7])
    s = np.eye(3)
    mu = 0.23
    result = fixed_mu_observables(h, s, mu=mu, kT=0.15, spin_degeneracy=2.0)
    np.testing.assert_allclose(
        result.energies.band_grand_energy,
        result.energies.band_free_energy - mu * result.electron_count,
        atol=1e-12,
    )
    assert result.energies.minus_t_s < 0.0


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


def test_raw_1d_k_weights_must_match_requested_k_axis_length():
    h = np.zeros((3, 2, 2, 2), dtype=float)
    h[..., 0, 0] = -1.0
    h[..., 1, 1] = 1.0
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    with pytest.raises(FixedMuOperatorError, match="1D k_weights length"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=np.array([0.4, 0.6]), k_axis=0)

    full_shape_weights = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    result = fixed_mu_observables(h, s, mu=0.0, k_weights=full_shape_weights, k_axis=0)
    np.testing.assert_allclose(result.k_weights.sum(axis=0), np.ones(2))
    np.testing.assert_allclose(result.electron_count, np.array([2.0, 2.0]))


def test_density_k_and_response_k_are_raw_per_k_not_weighted():
    h = np.array(
        [
            [[-1.0, 0.0], [0.0, 0.5]],
            [[0.25, 0.0], [0.0, 0.75]],
        ],
        dtype=float,
    )
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    weights = np.array([0.25, 0.75])
    result = fixed_mu_observables(
        h,
        s,
        mu=0.0,
        kT=0.2,
        spin_degeneracy=2.0,
        k_weights=weights,
        k_axis=0,
    )
    unweighted_first = fixed_mu_observables(h[0], s[0], mu=0.0, kT=0.2, spin_degeneracy=2.0)
    unweighted_second = fixed_mu_observables(h[1], s[1], mu=0.0, kT=0.2, spin_degeneracy=2.0)
    np.testing.assert_allclose(result.density_k[0], unweighted_first.density, atol=1e-12)
    np.testing.assert_allclose(result.density_k[1], unweighted_second.density, atol=1e-12)
    np.testing.assert_allclose(result.density_response_k[0], unweighted_first.density_response, atol=1e-12)
    np.testing.assert_allclose(result.density_response_k[1], unweighted_second.density_response, atol=1e-12)
    np.testing.assert_allclose(
        result.density,
        weights[0] * result.density_k[0] + weights[1] * result.density_k[1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.density_response,
        weights[0] * result.density_response_k[0] + weights[1] * result.density_response_k[1],
        atol=1e-12,
    )


def test_batch_without_k_axis_keeps_samples_independent():
    h = np.zeros((2, 2, 2), dtype=float)
    h[0] = np.diag([-1.0, 0.5])
    h[1] = np.diag([-0.3, -0.1])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    np.testing.assert_allclose(result.k_weights, np.ones(2))
    np.testing.assert_allclose(result.electron_count, np.array([2.0, 4.0]))
    assert result.density.shape == (2, 2, 2)


def test_k_weights_require_explicit_k_axis_and_density_trace_matches_n():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    with pytest.raises(FixedMuOperatorError, match="k_weights requires an explicit k_axis"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=2.0)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    np.testing.assert_allclose(result.density, np.diag([2.0, 0.0]))
    np.testing.assert_allclose(result.electron_count, 2.0)
    np.testing.assert_allclose(np.trace(result.density @ s).real, result.electron_count)
    np.testing.assert_allclose(result.conservation.electron_count_from_density, result.electron_count)
    validate_conservation(result, atol=1e-12)


def test_k_weight_values_are_checked_when_k_axis_is_explicit():
    h = np.stack([np.diag([-1.0, 1.0]), np.diag([-0.5, 0.5])])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    with pytest.raises(FixedMuOperatorError, match="finite"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=np.array([np.nan, 1.0]), k_axis=0)
    with pytest.raises(FixedMuOperatorError, match="nonnegative"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=np.array([-1.0, 2.0]), k_axis=0)


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
        result.energies.band_grand_energy,
        result.energies.band_energy + result.energies.minus_t_s - mu * result.electron_count,
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
    with pytest.raises(FixedMuOperatorError, match="matrix size must be positive"):
        fixed_mu_observables(np.empty((0, 0)), np.empty((0, 0)), mu=0.0)


def test_invalid_scalar_conversion_errors_are_fixed_mu_operator_errors():
    h = np.eye(1)
    s = np.eye(1)
    with pytest.raises(FixedMuOperatorError, match="mu must be a scalar float"):
        fixed_mu_observables(h, s, mu=object())
    with pytest.raises(FixedMuOperatorError, match="kT must be a scalar float"):
        fermi_dirac(np.array([0.0]), mu=0.0, kT=object())


class _FakeTorchTensor:
    __module__ = "torch"

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_torch_like_inputs_are_rejected_without_silent_detach():
    h = _FakeTorchTensor(np.diag([-1.0, 1.0]))
    s = _FakeTorchTensor(np.eye(2))
    with pytest.raises(FixedMuOperatorError, match="will not detach/cpu tensors silently"):
        fixed_mu_observables(h, s, mu=0.0)
    with pytest.raises(FixedMuOperatorError, match="detach=True"):
        fixed_mu_observables_from_torch(h, s, mu=0.0)
    result = fixed_mu_observables_from_torch(h, s, detach=True, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    np.testing.assert_allclose(result.electron_count, 2.0)


def test_torch_like_weights_and_energies_do_not_silently_convert():
    h = np.stack([np.diag([-1.0, 1.0]), np.diag([-0.5, 0.5])])
    s = np.broadcast_to(np.eye(2), h.shape).copy()
    weights = _FakeTorchTensor(np.array([0.25, 0.75]))
    with pytest.raises(FixedMuOperatorError, match="k_weights looks like a torch Tensor"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=weights, k_axis=0)
    with pytest.raises(FixedMuOperatorError, match="energies looks like a torch Tensor"):
        fermi_dirac(_FakeTorchTensor(np.array([-1.0, 1.0])), mu=0.0, kT=0.1)

    result = fixed_mu_observables_from_torch(
        _FakeTorchTensor(h),
        _FakeTorchTensor(s),
        detach=True,
        mu=0.0,
        k_weights=weights,
        k_axis=0,
    )
    np.testing.assert_allclose(result.k_weights, np.array([0.25, 0.75]))
    np.testing.assert_allclose(result.electron_count, 2.0)


def test_single_matrix_fake_torch_scalar_weight_is_rejected_before_numpy():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    weight = _FakeTorchTensor(np.array(1.0))
    with pytest.raises(FixedMuOperatorError, match="k_weights looks like a torch Tensor"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=weight)
    with pytest.raises(FixedMuOperatorError, match="k_weights requires an explicit k_axis"):
        fixed_mu_observables_from_torch(
            _FakeTorchTensor(h),
            _FakeTorchTensor(s),
            detach=True,
            mu=0.0,
            k_weights=weight,
        )
    result = fixed_mu_observables_from_torch(
        _FakeTorchTensor(h),
        _FakeTorchTensor(s),
        detach=True,
        mu=0.0,
    )
    np.testing.assert_allclose(result.electron_count, 2.0)


def test_single_matrix_fake_torch_shape_one_weight_is_rejected_before_numpy():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    weight = _FakeTorchTensor(np.array([1.0]))
    with pytest.raises(FixedMuOperatorError, match="k_weights looks like a torch Tensor"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=weight)
    with pytest.raises(FixedMuOperatorError, match="k_weights requires an explicit k_axis"):
        fixed_mu_observables_from_torch(
            _FakeTorchTensor(h),
            _FakeTorchTensor(s),
            detach=True,
            mu=0.0,
            k_weights=weight,
        )
    result = fixed_mu_observables_from_torch(
        _FakeTorchTensor(h),
        _FakeTorchTensor(s),
        detach=True,
        mu=0.0,
    )
    np.testing.assert_allclose(result.electron_count, 2.0)


def _ill_conditioned_pair(n=6, cond=1.0e10, seed=0):
    """Hermitian H and an S whose spectrum spans exactly ``cond``."""

    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = fixed_mu_module.hermitian_part(q @ np.diag(np.logspace(0.0, np.log10(cond), n)) @ q.T)
    a = rng.normal(size=(n, n))
    return fixed_mu_module.hermitian_part(a + a.T), s


def test_validate_conservation_rejects_a_truncated_eigenbasis():
    h = np.diag([-1.0, -0.5, 0.5, 1.0])
    s = np.eye(4)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    validate_conservation(result)
    assert float(result.electron_count) == pytest.approx(4.0)

    # Keep only the two occupied columns and rebuild every derived field from
    # them, exactly as an external producer with a truncated solver would.
    keep = slice(0, 2)
    vecs = np.asarray(result.eigvecs)[:, keep]
    vals = np.asarray(result.eigvals)[keep]
    occ = np.asarray(result.occupations)[keep]
    occ_response = np.asarray(result.occupation_response)[keep]
    density = fixed_mu_module._density_from_vectors(vecs, occ)
    density_response = fixed_mu_module._density_from_vectors(vecs, occ_response)
    energies = fixed_mu_module._energy_ledger(
        vals,
        occ / 2.0,
        np.asarray(1.0),
        mu=0.0,
        kT=0.0,
        spin_degeneracy=2.0,
        k_axis=None,
    )
    electron_count = float(np.sum(occ))
    conservation = fixed_mu_module.ConservationLedger(
        electron_count_from_density=np.asarray(np.trace(density @ s).real),
        band_energy_from_density=np.asarray(np.trace(density @ h).real),
        electron_count_residual=np.asarray(np.trace(density @ s).real - electron_count),
        band_energy_residual=np.asarray(np.trace(density @ h).real - energies.band_energy),
        density_hermiticity_error=fixed_mu_module._density_hermiticity_error(density),
        input_h_hermiticity_error=np.asarray(0.0),
        input_s_hermiticity_error=np.asarray(0.0),
    )
    truncated = _replace_result(
        result,
        eigvals=vals,
        eigvecs=vecs,
        occupations=occ,
        occupation_response=occ_response,
        density_k=density,
        density_response_k=density_response,
        density=density,
        density_response=density_response,
        electron_count=np.asarray(electron_count),
        dos_like_response=np.asarray(float(np.sum(occ_response))),
        energies=energies,
        conservation=conservation,
    )

    with pytest.raises(FixedMuOperatorError, match="complete 4x4 eigenbasis"):
        validate_conservation(truncated)


def test_validate_conservation_rejects_a_truncated_eigenvalue_vector():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    with pytest.raises(FixedMuOperatorError, match="eigvals must have shape"):
        validate_conservation(_replace_result(result, eigvals=np.asarray(result.eigvals)[:1]))


def test_expected_request_parameters_bind_the_result_to_the_caller():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.1, kT=0.05, spin_degeneracy=2.0)

    validate_conservation(result, expected_mu=0.1, expected_kT=0.05, expected_spin_degeneracy=2.0)
    with pytest.raises(FixedMuOperatorError, match="computed at mu="):
        validate_conservation(result, expected_mu=0.4)
    with pytest.raises(FixedMuOperatorError, match="computed at kT="):
        validate_conservation(result, expected_kT=0.2)
    with pytest.raises(FixedMuOperatorError, match="computed at spin_degeneracy="):
        validate_conservation(result, expected_spin_degeneracy=1.0)


def test_expected_k_weights_bind_the_brillouin_zone_request():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.8, 0.6])])
    s = np.stack([np.eye(2), np.eye(2)])
    result = fixed_mu_observables(
        h, s, mu=0.0, kT=0.1, spin_degeneracy=2.0, k_weights=np.array([2.0, 1.0]), k_axis=0
    )

    validate_conservation(result, expected_k_weights=np.array([2.0 / 3.0, 1.0 / 3.0]))
    with pytest.raises(FixedMuOperatorError, match="do not match the expected k_weights"):
        validate_conservation(result, expected_k_weights=np.array([0.5, 0.5]))


def test_spin_degeneracy_is_constrained_to_the_conventions_this_module_knows():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    for good in (1.0, 2.0):
        fixed_mu_observables(h, s, mu=0.0, spin_degeneracy=good)
    with pytest.raises(FixedMuOperatorError, match="spin_degeneracy must be one of"):
        fixed_mu_observables(h, s, mu=0.0, spin_degeneracy=3.0)
    with pytest.raises(FixedMuOperatorError, match="spin_degeneracy must be one of"):
        fermi_dirac(np.array([0.0]), mu=0.0, kT=0.0, spin_degeneracy=1.5)


def test_untampered_ill_conditioned_result_self_validates_but_stays_strict():
    h, s = _ill_conditioned_pair(cond=1.0e10)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.1, spin_degeneracy=2.0)
    assert float(result.overlap_condition) > 1.0e9

    # Attainable orthonormality error here is ~1e-7, unreachable for atol=1e-8.
    validate_conservation(result)

    # Strictness for genuinely wrong data survives: rotating one eigenvector
    # breaks S-orthonormality far beyond the conditioning budget.
    broken = np.asarray(result.eigvecs).copy()
    broken[:, 0] *= 1.5
    with pytest.raises(FixedMuOperatorError, match="S-orthonormality"):
        validate_conservation(_replace_result(result, eigvecs=broken))


def test_well_conditioned_results_keep_the_plain_absolute_tolerance():
    h, s = _ill_conditioned_pair(cond=1.0e2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.1, spin_degeneracy=2.0)

    nudged = np.asarray(result.eigvecs).copy()
    nudged[0, 0] += 1.0e-6
    with pytest.raises(FixedMuOperatorError, match="S-orthonormality"):
        validate_conservation(_replace_result(result, eigvecs=nudged))


def test_input_hermiticity_errors_are_recorded_before_symmetrization():
    h = np.array([[0.0, 0.2 + 4.0e-10], [0.2, 1.0]])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0, hermitian_tol=1e-8)

    assert float(result.conservation.input_h_hermiticity_error) == pytest.approx(4.0e-10, rel=1e-6)
    assert float(result.conservation.input_s_hermiticity_error) == 0.0
    # The stored snapshot is symmetrized, so this cannot be recomputed; the
    # validator re-enforces the recorded hermitian_tol against it instead.
    np.testing.assert_allclose(
        np.asarray(result._validation_context.hamiltonian),
        fixed_mu_module.hermitian_part(h),
        atol=0.0,
    )
    validate_conservation(result)

    forged = replace(result.conservation, input_h_hermiticity_error=np.asarray(1.0))
    with pytest.raises(FixedMuOperatorError, match="exceeds what the recorded hermitian_tol"):
        validate_conservation(_replace_result(result, conservation=forged))
    negative = replace(result.conservation, input_s_hermiticity_error=np.asarray(-1.0))
    with pytest.raises(FixedMuOperatorError, match="must be finite and nonnegative"):
        validate_conservation(_replace_result(result, conservation=negative))


def test_k_axis_rejects_non_integral_values():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.8, 0.6])])
    s = np.stack([np.eye(2), np.eye(2)])

    with pytest.raises(FixedMuOperatorError, match="k_axis must be an integer"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=np.array([0.25, 0.75]), k_axis=0.9)
    with pytest.raises(FixedMuOperatorError, match="k_axis must be an integer"):
        fixed_mu_observables(h, s, mu=0.0, k_weights=np.array([0.25, 0.75]), k_axis=True)
    # Integral numpy scalars stay acceptable.
    result = fixed_mu_observables(
        h, s, mu=0.0, k_weights=np.array([0.25, 0.75]), k_axis=np.int64(0)
    )
    assert result.k_axis == 0


def test_zero_length_leading_batch_fails_closed():
    empty = np.zeros((0, 2, 2))
    with pytest.raises(FixedMuOperatorError, match="zero-length dimension"):
        generalized_bands(empty, empty)
    with pytest.raises(FixedMuOperatorError, match="zero-length dimension"):
        fixed_mu_observables(empty, empty, mu=0.0)


def test_forged_density_is_rejected_by_the_from_vectors_route():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)

    bad = np.asarray(result.density).copy()
    bad[0, 0] += 1.0e-3
    # The aggregate density is now compared against C f C^H rather than against
    # a re-aggregation of the stored per-k density, so it no longer inherits it.
    with pytest.raises(FixedMuOperatorError, match="stored density does not match"):
        validate_conservation(_replace_result(result, density=bad))
    # Forging density_k to keep the two mutually consistent is caught upstream.
    with pytest.raises(FixedMuOperatorError, match="stored density_k"):
        validate_conservation(_replace_result(result, density=bad, density_k=bad))


def test_pickle_round_trip_preserves_the_read_only_invariant():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    result = fixed_mu_observables(h, s, mu=0.0, kT=0.0, spin_degeneracy=2.0)
    restored = pickle.loads(pickle.dumps(result))

    assert restored.eigvals.flags.writeable is False
    assert restored.density.flags.writeable is False
    assert restored.energies.band_free_energy.flags.writeable is False
    assert restored.conservation.input_h_hermiticity_error.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        restored.density[0, 0] = 1.0
    # The H/S context is deliberately not serialized.
    with pytest.raises(FixedMuOperatorError, match="missing H/S context"):
        validate_conservation(restored)
    validate_conservation(restored, h=h, s=s)


def test_energy_ledger_names_state_what_the_terms_are():
    h = np.diag([-1.0, 0.5])
    s = np.eye(2)
    mu = 0.1
    result = fixed_mu_observables(h, s, mu=mu, kT=0.2, spin_degeneracy=2.0)
    ledger = result.energies

    assert float(ledger.minus_t_s) < 0.0
    np.testing.assert_allclose(
        ledger.band_free_energy, ledger.band_energy + ledger.minus_t_s, atol=1e-14
    )
    np.testing.assert_allclose(
        ledger.band_grand_energy,
        ledger.band_free_energy - mu * result.electron_count,
        atol=1e-12,
    )
    assert {field.name for field in fields(ledger)} == {
        "band_energy",
        "minus_t_s",
        "band_free_energy",
        "band_grand_energy",
    }
