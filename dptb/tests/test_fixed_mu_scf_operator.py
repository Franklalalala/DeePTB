from dataclasses import fields, is_dataclass
import pickle

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from dptb.nnops.fixed_mu_operator import fixed_mu_observables
from dptb.nnops.fixed_mu_scf_operator import (
    FixedMuSCFConvergenceError,
    FixedMuSCFError,
    _electrostatic_hamiltonian,
    fixed_mu_electrostatic_scf,
)


def _assert_dataclass_fields_exact(actual, expected):
    assert type(actual) is type(expected)
    assert is_dataclass(actual)
    for field_info in fields(actual):
        actual_value = getattr(actual, field_info.name)
        expected_value = getattr(expected, field_info.name)
        if isinstance(actual_value, np.ndarray):
            assert np.array_equal(
                actual_value, expected_value
            ), field_info.name
        elif is_dataclass(actual_value):
            _assert_dataclass_fields_exact(actual_value, expected_value)
        else:
            assert actual_value == expected_value, field_info.name


def _one_level_scf(
    *,
    epsilon=0.1,
    mu=0.0,
    kT=0.2,
    gamma=0.8,
    reference_population=1.0,
    spin_degeneracy=2.0,
    mixing="linear",
    mixing_step=0.5,
    charge_tol=1e-13,
    max_iter=1000,
):
    return fixed_mu_electrostatic_scf(
        np.array([[epsilon]], dtype=np.float64),
        np.eye(1),
        mu=mu,
        kT=kT,
        ao_atom_index=np.array([0]),
        reference_populations=np.array([reference_population]),
        coulomb_kernel=np.array([[gamma]]),
        spin_degeneracy=spin_degeneracy,
        mixing=mixing,
        mixing_step=mixing_step,
        charge_tol=charge_tol,
        max_iter=max_iter,
    )


def test_zero_kernel_matches_frozen_fixed_mu_field_by_field_exactly():
    h0 = np.array([[-0.35, 0.08], [0.08, 0.42]])
    s = np.array([[1.0, 0.12], [0.12, 1.1]])
    kwargs = {
        "mu": 0.07,
        "kT": 0.11,
        "spin_degeneracy": 2.0,
        "normalize_k_weights": False,
        "eig_floor": 1e-11,
        "max_condition": 1e10,
        "hermitian_tol": 1e-10,
    }
    frozen = fixed_mu_observables(h0, s, **kwargs)

    scf = fixed_mu_electrostatic_scf(
        h0,
        s,
        ao_atom_index=[0, 1],
        reference_populations=[1.0, 1.0],
        coulomb_kernel=np.zeros((2, 2)),
        mixing="pdiis",
        mixing_step=0.37,
        n_history=4,
        mixing_period=2,
        max_iter=17,
        charge_tol=1e-14,
        divergence_tol=1234.0,
        **kwargs,
    )

    assert scf.iterations == 1
    assert scf.residual_history[0] == 0.0
    assert np.array_equal(scf.phi, np.zeros(2))
    _assert_dataclass_fields_exact(scf.fixed_mu_result, frozen)
    assert scf.mixing == "pdiis"
    assert scf.mixing_step == 0.37
    assert scf.n_history == 4
    assert scf.mixing_period == 2
    assert scf.max_iter == 17
    assert scf.charge_tol == 1e-14
    assert scf.divergence_tol == 1234.0
    assert scf.normalize_k_weights is False


def test_uniform_potential_is_exact_h_minus_c_s_gauge_shift():
    h0 = np.array([[-0.4, 0.07], [0.07, 0.3]])
    s = np.array([[1.0, 0.125], [0.125, 1.25]])
    ao_atom_index = np.array([0, 1])
    c = 0.25

    shifted = _electrostatic_hamiltonian(
        h0, s, ao_atom_index, np.array([c, c])
    )

    assert np.array_equal(shifted, h0 - c * s)


def test_uniform_gauge_shift_with_mu_shift_preserves_fixed_mu_scf_state():
    h0 = np.array([[-0.2, 0.07], [0.07, 0.4]])
    s = np.array([[1.0, 0.1], [0.1, 1.2]])
    common = {
        "kT": 0.13,
        "ao_atom_index": [0, 1],
        "reference_populations": [1.0, 1.0],
        "coulomb_kernel": np.zeros((2, 2)),
        "spin_degeneracy": 2.0,
        "charge_tol": 1e-14,
    }
    c = 0.25
    baseline = fixed_mu_electrostatic_scf(h0, s, mu=0.05, **common)
    shifted = fixed_mu_electrostatic_scf(
        h0 - c * s, s, mu=0.05 - c, **common
    )

    np.testing.assert_allclose(shifted.q, baseline.q, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        shifted.fixed_mu_result.occupations,
        baseline.fixed_mu_result.occupations,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        shifted.fixed_mu_result.density,
        baseline.fixed_mu_result.density,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        shifted.fixed_mu_result.electron_count,
        baseline.fixed_mu_result.electron_count,
        atol=2e-14,
        rtol=2e-14,
    )


def test_identity_overlap_reduces_update_to_pure_onsite_shift():
    h0 = np.array([[-0.3, 0.11], [0.11, 0.4]])
    phi = np.array([0.2, -0.1])

    shifted = _electrostatic_hamiltonian(
        h0, np.eye(2), np.array([0, 1]), phi
    )

    expected = h0.copy()
    expected[0, 0] -= phi[0]
    expected[1, 1] -= phi[1]
    assert np.array_equal(shifted, expected)


def test_one_level_scf_matches_independent_analytic_root():
    epsilon = 0.1
    mu = 0.0
    kT = 0.2
    gamma = 0.8
    n_ref = 1.0
    degeneracy = 2.0

    def equation(n):
        effective_energy = epsilon + gamma * (n - n_ref)
        return n - degeneracy * expit(
            -(effective_energy - mu) / kT
        )

    analytic_n = brentq(equation, 0.0, degeneracy, xtol=1e-14)
    scf = _one_level_scf(
        epsilon=epsilon,
        mu=mu,
        kT=kT,
        gamma=gamma,
        reference_population=n_ref,
        spin_degeneracy=degeneracy,
    )
    frozen = fixed_mu_observables(
        np.array([[epsilon]]),
        np.eye(1),
        mu=mu,
        kT=kT,
        spin_degeneracy=degeneracy,
    )

    scf_n = float(scf.fixed_mu_result.electron_count)
    frozen_n = float(frozen.electron_count)
    assert abs(scf_n - analytic_n) <= 1e-10
    assert abs(frozen_n - scf_n) > 0.1
    assert scf.q[0] == pytest.approx(n_ref - scf_n, abs=1e-14)


def test_asymmetric_diatomic_charge_transfer_and_symmetric_limit():
    common = {
        "s": np.eye(2),
        "mu": 0.0,
        "kT": 0.12,
        "ao_atom_index": [0, 1],
        "reference_populations": [0.5, 0.5],
        "coulomb_kernel": [[0.4, 0.1], [0.1, 0.4]],
        "spin_degeneracy": 1.0,
        "mixing": "linear",
        "mixing_step": 0.5,
        "charge_tol": 1e-13,
        "max_iter": 1000,
    }
    asymmetric = fixed_mu_electrostatic_scf(
        np.diag([-0.3, 0.3]), **common
    )
    symmetric = fixed_mu_electrostatic_scf(
        np.diag([0.1, 0.1]), **common
    )

    populations = np.array([0.5, 0.5]) - asymmetric.q
    assert populations[0] > populations[1]
    assert asymmetric.q[0] < asymmetric.q[1]
    assert symmetric.q[0] == symmetric.q[1]


def test_small_gamma_matches_frozen_linear_response_to_gamma_squared():
    epsilon = 0.15
    mu = 0.0
    kT = 0.3
    gamma = 1e-4
    n_ref = 0.8
    degeneracy = 2.0
    frozen = fixed_mu_observables(
        np.array([[epsilon]]),
        np.eye(1),
        mu=mu,
        kT=kT,
        spin_degeneracy=degeneracy,
    )
    q_frozen = n_ref - float(frozen.electron_count)
    frozen_d_p_d_mu = float(frozen.dos_like_response)
    first_order_q = q_frozen * (1.0 - gamma * frozen_d_p_d_mu)

    scf = _one_level_scf(
        epsilon=epsilon,
        mu=mu,
        kT=kT,
        gamma=gamma,
        reference_population=n_ref,
        spin_degeneracy=degeneracy,
        charge_tol=1e-14,
    )

    assert abs(scf.q[0] - first_order_q) <= 2.0 * gamma**2


def test_pdiis_and_linear_converge_to_same_fixed_point():
    common = {
        "epsilon": 0.1,
        "mu": 0.0,
        "kT": 0.2,
        "gamma": 0.8,
        "reference_population": 1.0,
        "spin_degeneracy": 2.0,
        "mixing_step": 0.35,
        "charge_tol": 1e-11,
        "max_iter": 1000,
    }
    linear = _one_level_scf(mixing="linear", **common)
    pdiis = _one_level_scf(mixing="pdiis", **common)

    np.testing.assert_allclose(pdiis.q, linear.q, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        pdiis.fixed_mu_result.density,
        linear.fixed_mu_result.density,
        atol=1e-8,
        rtol=1e-8,
    )


def test_max_iter_failure_carries_iterations_and_residual_history():
    with pytest.raises(FixedMuSCFConvergenceError) as caught:
        _one_level_scf(
            gamma=2.0,
            mixing="linear",
            mixing_step=0.1,
            charge_tol=1e-15,
            max_iter=1,
        )

    error = caught.value
    assert error.iterations == 1
    assert error.residual_history.shape == (1,)
    assert error.residual_history[0] > 1e-15
    assert not error.residual_history.flags.writeable
    assert "iterations=1" in str(error)
    assert "residual_history=" in str(error)


@pytest.mark.parametrize(
    "kernel, match",
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "must be symmetric"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite values"),
    ],
)
def test_invalid_coulomb_kernel_is_rejected(kernel, match):
    with pytest.raises(FixedMuSCFError, match=match):
        fixed_mu_electrostatic_scf(
            np.eye(2),
            np.eye(2),
            mu=0.0,
            kT=0.1,
            ao_atom_index=[0, 1],
            reference_populations=[1.0, 1.0],
            coulomb_kernel=kernel,
        )


def test_wrong_ao_atom_index_length_is_rejected():
    with pytest.raises(FixedMuSCFError, match="length n_orb=2"):
        fixed_mu_electrostatic_scf(
            np.eye(2),
            np.eye(2),
            mu=0.0,
            kT=0.1,
            ao_atom_index=[0],
            reference_populations=[1.0],
            coulomb_kernel=np.eye(1),
        )


@pytest.mark.parametrize(
    "h, s",
    [
        (np.zeros((2, 2, 2)), np.broadcast_to(np.eye(2), (2, 2, 2))),
        (np.zeros((1, 2, 2)), np.broadcast_to(np.eye(2), (1, 2, 2))),
    ],
)
def test_k_stacks_and_leading_batches_are_rejected_with_v1_message(h, s):
    with pytest.raises(
        FixedMuSCFError, match="v1 does not support.*leading-batch"
    ):
        fixed_mu_electrostatic_scf(
            h,
            s,
            mu=0.0,
            kT=0.1,
            ao_atom_index=[0, 1],
            reference_populations=[1.0, 1.0],
            coulomb_kernel=np.eye(2),
        )


def test_k_weight_request_is_rejected_with_v1_message():
    with pytest.raises(
        FixedMuSCFError, match="v1 does not support k_weights"
    ):
        fixed_mu_electrostatic_scf(
            np.eye(1),
            np.eye(1),
            mu=0.0,
            kT=0.1,
            ao_atom_index=[0],
            reference_populations=[1.0],
            coulomb_kernel=np.eye(1),
            k_weights=[1.0],
            k_axis=0,
        )


def test_result_and_pickle_roundtrip_arrays_are_bytes_backed_read_only():
    result = _one_level_scf(gamma=0.3, charge_tol=1e-11)
    restored = pickle.loads(pickle.dumps(result))

    for value in (
        result.q,
        result.phi,
        result.residual_history,
        restored.q,
        restored.phi,
        restored.residual_history,
        restored.fixed_mu_result.density,
        restored.fixed_mu_result.eigvals,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    np.testing.assert_array_equal(restored.q, result.q)
    np.testing.assert_array_equal(restored.phi, result.phi)
    np.testing.assert_array_equal(
        restored.residual_history, result.residual_history
    )
