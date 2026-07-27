import pickle

import numpy as np
import pytest

import dptb.nnops.qeq_operator as qeq

#: Uniform gauge amplitudes the SHIP gate requires solve and validate to be
#: adversarially exercised at.
GAUGE_ALPHAS = (1.0e9, 1.0e12, 1.0e14)


def _self_consistent_qeq_result(
    chi,
    kernel,
    charges,
    total_charge,
    *,
    multiplier=None,
    symmetry_tol=1.0e-10,
    constrained_eig_floor=1.0e-12,
    max_condition=1.0e12,
    residual_tol=1.0e-8,
):
    chi = np.asarray(chi, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    charges = np.asarray(charges, dtype=float)
    total_charge = np.asarray(float(total_charge))
    if multiplier is None:
        multiplier = np.asarray(-float(np.mean(chi + kernel @ charges)))
    else:
        multiplier = np.asarray(float(multiplier))

    stationarity, tangent, multiplier_residual, _ = qeq._stationarity_parts(
        chi, kernel, charges, multiplier
    )
    energy, linear, quadratic, identity = qeq._qeq_energies(
        chi, kernel, charges, total_charge, multiplier
    )
    min_eig, max_eig, condition = qeq._recompute_constrained_kernel_diagnostics(kernel)
    diagnostics = qeq.QEqDiagnostics(
        charge_residual=np.sum(charges, axis=-1) - total_charge,
        stationarity_max_abs=np.max(np.abs(stationarity), axis=-1),
        stationarity_l2=np.linalg.norm(stationarity, axis=-1),
        stationarity_tangent_max_abs=tangent,
        multiplier_residual=multiplier_residual,
        energy_identity_residual=energy - identity,
        kernel_symmetry_error=np.max(
            np.abs(kernel - np.swapaxes(kernel, -1, -2)), axis=(-1, -2)
        ),
        input_kernel_symmetry_error=np.asarray(0.0),
        constrained_min_eig=min_eig,
        constrained_max_eig=max_eig,
        constrained_condition=condition,
        kkt_condition=qeq._recompute_kkt_condition(kernel),
    )
    return qeq.QEqResult(
        charges=charges,
        total_charge=total_charge,
        electronegativity=chi,
        hardness_kernel=kernel,
        lagrange_multiplier=multiplier,
        stationarity=stationarity,
        energy=energy,
        linear_energy=linear,
        quadratic_energy=quadratic,
        energy_identity=identity,
        diagnostics=diagnostics,
        units=qeq.QEqUnits(),
        symmetry_tol=symmetry_tol,
        constrained_eig_floor=constrained_eig_floor,
        max_condition=max_condition,
        residual_tol=residual_tol,
    )


def test_uniform_kernel_gauge_does_not_relax_tangent_gate():
    chi = np.array([1.0, 3.0])
    base_kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    kernel = base_kernel + 1.0e14 * np.ones((2, 2))
    solved = qeq.solve_qeq(chi, kernel)
    baseline = qeq.solve_qeq(chi, base_kernel)
    np.testing.assert_allclose(solved.charges, baseline.charges, atol=1.0e-12)
    np.testing.assert_allclose(
        solved.lagrange_multiplier, baseline.lagrange_multiplier, atol=1.0e-12
    )
    np.testing.assert_allclose(solved.stationarity, np.zeros(2), atol=1.0e-12)

    forged_charges = np.asarray(solved.charges) + np.array([0.1, -0.1])
    forged = _self_consistent_qeq_result(
        chi,
        kernel,
        forged_charges,
        0.0,
        symmetry_tol=solved.symmetry_tol,
        constrained_eig_floor=solved.constrained_eig_floor,
        max_condition=solved.max_condition,
        residual_tol=solved.residual_tol,
    )
    with pytest.raises(qeq.QEqOperatorError, match="stationarity residual"):
        qeq.validate_qeq_result(forged)


@pytest.mark.parametrize("alpha", GAUGE_ALPHAS)
@pytest.mark.parametrize("total_charge", [0.0, 1.25])
def test_gauge_sweep_solves_exactly_and_rejects_a_forged_charge_vector(alpha, total_charge):
    chi = np.array([1.0, 3.0])
    base_kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    baseline = qeq.solve_qeq(chi, base_kernel, total_charge=total_charge)
    shifted = qeq.solve_qeq(
        chi, base_kernel + alpha * np.ones((2, 2)), total_charge=total_charge
    )
    np.testing.assert_allclose(shifted.charges, baseline.charges, atol=1.0e-12)
    qeq.validate_qeq_result(shifted)

    forged = _self_consistent_qeq_result(
        chi,
        base_kernel + alpha * np.ones((2, 2)),
        np.asarray(shifted.charges) + np.array([0.1, -0.1]),
        total_charge,
    )
    with pytest.raises(qeq.QEqOperatorError, match="stationarity residual"):
        qeq.validate_qeq_result(forged)


@pytest.mark.parametrize("alpha", GAUGE_ALPHAS)
@pytest.mark.parametrize("total_charge", [0.0, 1.25])
def test_neutral_gauge_shift_does_not_reject_a_correct_solve(alpha, total_charge):
    """The uniform component's floor must cover its own cancellation.

    ``mean(J) * sum(q)`` cancels to roundoff for a neutral system, so a floor
    keyed to the cancelled result rejects the solve that produced it.
    """

    chi = np.array([0.5, -1.0, 2.0, 0.25, -0.75])
    kernel = np.diag([4.0, 5.0, 6.0, 7.0, 8.0]) + 0.2
    baseline = qeq.solve_qeq(chi, kernel, total_charge=total_charge)
    shifted = qeq.solve_qeq(
        chi, kernel + alpha * np.ones((5, 5)), total_charge=total_charge
    )
    np.testing.assert_allclose(shifted.charges, baseline.charges, atol=1.0e-9)


@pytest.mark.parametrize("alpha", GAUGE_ALPHAS)
def test_externally_computed_multiplier_survives_a_large_gauge(alpha):
    """An external producer spelling ``lambda = -mean(chi + J q)`` directly pays
    ``eps * max|J| * sum|q|`` for the dense matvec; the uniform gate must admit it."""

    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]]) + alpha * np.ones((2, 2))
    solved = qeq.solve_qeq(chi, kernel)
    external = _self_consistent_qeq_result(chi, kernel, np.asarray(solved.charges), 0.0)
    qeq.validate_qeq_result(external)


def test_stored_kernel_uses_symmetry_policy_not_residual_atol():
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0 + 5.0e-9], [1.0, 7.0]])
    basis = qeq._sum_zero_basis(2)
    charges = basis @ np.linalg.solve(basis.T @ kernel @ basis, -basis.T @ chi)
    forged = _self_consistent_qeq_result(
        chi,
        kernel,
        charges,
        0.0,
        symmetry_tol=1.0e-10,
    )
    with pytest.raises(qeq.QEqOperatorError, match="symmetry_tol"):
        qeq.validate_qeq_result(forged)


def test_result_arrays_cannot_reenable_writes():
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    result = qeq.solve_qeq(chi, kernel)
    with pytest.raises(ValueError):
        result.charges.setflags(write=True)


def test_result_arrays_stay_frozen_across_a_pickle_round_trip():
    result = qeq.solve_qeq(np.array([1.0, 3.0]), np.array([[5.0, 1.0], [1.0, 7.0]]))
    restored = pickle.loads(pickle.dumps(result))
    for field_name in qeq._RESULT_ARRAY_FIELDS:
        with pytest.raises(ValueError):
            getattr(restored, field_name).setflags(write=True)
    for field_name in qeq._DIAGNOSTIC_ARRAY_FIELDS:
        with pytest.raises(ValueError):
            getattr(restored.diagnostics, field_name).setflags(write=True)
    qeq.validate_qeq_result(restored)
