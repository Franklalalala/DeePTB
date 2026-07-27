import numpy as np
import pytest

import dptb.nnops.qeq_operator as qeq


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
