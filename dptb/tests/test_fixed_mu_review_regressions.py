from dataclasses import replace

import numpy as np
import pytest

import dptb.nnops.fixed_mu_operator as fixed_mu


def test_expected_raw_weights_are_normalized_like_the_request():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.8, 0.6])])
    s = np.stack([np.eye(2), np.eye(2)])
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, kT=0.1, k_weights=np.array([2.0, 1.0]), k_axis=0
    )
    fixed_mu.validate_conservation(result, expected_k_weights=np.array([2.0, 1.0]))


def test_mixed_batch_condition_does_not_relax_good_item():
    h = np.stack([np.diag([-1.0, 1.0]), np.diag([-1.0e-12, 1.0])])
    s = np.stack([np.eye(2), np.diag([1.0e-12, 1.0])])
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, eig_floor=1.0e-13, max_condition=1.0e12
    )
    eigvecs = np.asarray(result.eigvecs).copy()
    eigvecs[0, :, 0] *= 1.0001
    density_k = fixed_mu._density_from_vectors(eigvecs, np.asarray(result.occupations))
    density_response_k = fixed_mu._density_from_vectors(
        eigvecs, np.asarray(result.occupation_response)
    )
    ne = fixed_mu._trace_last2(density_k @ s)
    band = fixed_mu._trace_last2(density_k @ h)
    conservation = fixed_mu.ConservationLedger(
        electron_count_from_density=ne,
        band_energy_from_density=band,
        electron_count_residual=ne - np.asarray(result.electron_count),
        band_energy_residual=band - np.asarray(result.energies.band_energy),
        density_hermiticity_error=fixed_mu._density_hermiticity_error(density_k),
        input_h_hermiticity_error=np.zeros(2),
        input_s_hermiticity_error=np.zeros(2),
    )
    forged = replace(
        result,
        validation_context=result._validation_context,
        eigvecs=eigvecs,
        density_k=density_k,
        density_response_k=density_response_k,
        density=density_k,
        density_response=density_response_k,
        conservation=conservation,
    )
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="S-orthonormality"):
        fixed_mu.validate_conservation(forged)


def test_request_policy_can_bind_axis_normalization_and_safety():
    h = np.zeros((2, 2, 1, 1))
    h[..., 0, 0] = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    s = np.ones_like(h)
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, k_axis=1, normalize_k_weights=True
    )
    fixed_mu.validate_conservation(
        result,
        expected_k_axis=1,
        expected_normalize_k_weights=True,
        expected_eig_floor=1.0e-10,
        expected_max_condition=1.0e12,
        expected_hermitian_tol=1.0e-8,
    )
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="k_axis"):
        fixed_mu.validate_conservation(result, expected_k_axis=0)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="normalize_k_weights"):
        fixed_mu.validate_conservation(result, expected_normalize_k_weights=False)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="max_condition"):
        fixed_mu.validate_conservation(result, expected_max_condition=1.0e6)


def test_normalize_k_weights_rejects_truthy_non_boolean_values():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="must be a boolean"):
        fixed_mu.fixed_mu_observables(h, s, mu=0.0, normalize_k_weights="false")
