from dataclasses import replace

import numpy as np
import pytest

import dptb.nnops.qeq_operator as qeq_module
from dptb.nnops import QEqResult as ExportedQEqResult
from dptb.nnops import fixed_mu_observables as exported_fixed_mu_observables
from dptb.nnops import solve_qeq as exported_solve_qeq
from dptb.nnops.qeq_operator import (
    QEqKernelConditionError,
    QEqOperatorError,
    QEqResult,
    QEqUnits,
    solve_qeq,
    validate_qeq_result,
)


def test_two_atom_neutral_solution_matches_analytic_formula():
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    result = solve_qeq(chi, kernel)

    expected_q0 = (chi[1] - chi[0]) / (kernel[0, 0] + kernel[1, 1] - 2.0 * kernel[0, 1])
    expected = np.array([expected_q0, -expected_q0])
    np.testing.assert_allclose(result.charges, expected, atol=1e-12)
    np.testing.assert_allclose(result.total_charge, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.stationarity, np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(result.diagnostics.charge_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.energy, result.energy_identity, atol=1e-12)
    validate_qeq_result(result, atol=1e-12)


def test_two_atom_charged_solution_matches_analytic_formula_and_units():
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    total_charge = 1.0
    result = solve_qeq(chi, kernel, total_charge=total_charge)

    denom = kernel[0, 0] + kernel[1, 1] - 2.0 * kernel[0, 1]
    expected_q0 = (chi[1] - chi[0] - (kernel[0, 1] - kernel[1, 1]) * total_charge) / denom
    expected = np.array([expected_q0, total_charge - expected_q0])
    np.testing.assert_allclose(result.charges, expected, atol=1e-12)
    np.testing.assert_allclose(result.charges.sum(), total_charge, atol=1e-12)
    assert result.units == QEqUnits()
    assert result.units.charge == "e"
    assert result.units.energy == "eV"
    assert result.units.electronegativity == "eV/e"
    assert result.units.hardness_kernel == "eV/e^2"


def test_chi_gauge_shift_preserves_charges_and_shifts_energy_by_total_charge():
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    total_charge = 1.25
    shift = 4.0
    base = solve_qeq(chi, kernel, total_charge=total_charge)
    shifted = solve_qeq(chi + shift, kernel, total_charge=total_charge)

    np.testing.assert_allclose(shifted.charges, base.charges, atol=1e-12)
    np.testing.assert_allclose(shifted.lagrange_multiplier, base.lagrange_multiplier - shift, atol=1e-12)
    np.testing.assert_allclose(shifted.energy - base.energy, shift * total_charge, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0e3, 1.0e6])
def test_fixed_charge_solution_is_invariant_to_uniform_kernel_gauge(alpha):
    chi = np.array([1.0, 3.0])
    kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    total_charge = 0.0
    baseline = solve_qeq(chi, kernel, total_charge=total_charge)

    shifted_kernel = kernel + alpha * np.ones_like(kernel)
    shifted = solve_qeq(
        chi,
        shifted_kernel,
        total_charge=total_charge,
        max_condition=1.0e12,
    )

    # Z.T@(alpha*11.T)@Z == 0, so the physical tangent problem is identical.
    np.testing.assert_allclose(shifted.charges, baseline.charges, atol=1e-10)
    np.testing.assert_allclose(
        shifted.diagnostics.constrained_condition,
        baseline.diagnostics.constrained_condition,
        atol=1e-12,
    )
    validate_qeq_result(shifted, atol=1e-8)


def test_permutation_equivariance_for_dense_kernel():
    chi = np.array([0.2, -0.3, 0.5])
    kernel = np.array(
        [
            [4.0, 0.2, -0.1],
            [0.2, 3.5, 0.4],
            [-0.1, 0.4, 5.0],
        ]
    )
    total_charge = -0.4
    perm = np.array([2, 0, 1])
    base = solve_qeq(chi, kernel, total_charge=total_charge)
    permuted = solve_qeq(chi[perm], kernel[np.ix_(perm, perm)], total_charge=total_charge)

    np.testing.assert_allclose(permuted.charges, base.charges[perm], atol=1e-12)
    np.testing.assert_allclose(permuted.energy, base.energy, atol=1e-12)
    np.testing.assert_allclose(permuted.diagnostics.constrained_min_eig, base.diagnostics.constrained_min_eig, atol=1e-12)


def test_batch_broadcasting_keeps_samples_independent():
    chi = np.array([[1.0, 3.0, 0.5], [0.2, -0.1, 0.8]])
    kernel = np.array(
        [
            [5.0, 1.0, 0.0],
            [1.0, 7.0, 0.2],
            [0.0, 0.2, 4.0],
        ],
        dtype=float,
    )
    total_charge = np.array([0.0, 1.0])
    result = solve_qeq(chi, kernel, total_charge=total_charge)

    assert result.charges.shape == (2, 3)
    assert result.energy.shape == (2,)
    np.testing.assert_allclose(result.charges.sum(axis=-1), total_charge, atol=1e-12)
    np.testing.assert_allclose(result.diagnostics.stationarity_max_abs, np.zeros(2), atol=1e-12)

    first = solve_qeq(chi[0], kernel, total_charge=0.0)
    second = solve_qeq(chi[1], kernel, total_charge=1.0)
    np.testing.assert_allclose(result.charges[0], first.charges, atol=1e-12)
    np.testing.assert_allclose(result.charges[1], second.charges, atol=1e-12)


def test_batch_kernel_and_charge_broadcast_shapes():
    chi = np.array([1.0, 2.0, -0.5])
    kernels = np.stack(
        [
            np.diag([4.0, 5.0, 6.0]),
            np.array([[5.0, 0.3, 0.1], [0.3, 4.0, 0.2], [0.1, 0.2, 3.5]]),
        ]
    )
    result = solve_qeq(chi, kernels, total_charge=np.array([0.0, -1.0]))

    assert result.electronegativity.shape == (2, 3)
    assert result.hardness_kernel.shape == (2, 3, 3)
    np.testing.assert_allclose(result.charges.sum(axis=-1), np.array([0.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(result.energy, result.linear_energy + result.quadratic_energy, atol=1e-12)


def test_one_atom_qeq_is_fixed_by_total_charge():
    result = solve_qeq(np.array([2.0]), np.array([[0.0]]), total_charge=-0.75)

    np.testing.assert_allclose(result.charges, np.array([-0.75]), atol=1e-12)
    np.testing.assert_allclose(result.lagrange_multiplier, -2.0, atol=1e-12)
    np.testing.assert_allclose(result.energy, -1.5, atol=1e-12)
    assert np.isinf(result.diagnostics.constrained_min_eig)
    np.testing.assert_allclose(result.diagnostics.constrained_condition, 1.0, atol=1e-12)


def test_public_export_does_not_require_fixed_mu_api():
    result = exported_solve_qeq(np.array([1.0, 3.0]), np.array([[5.0, 1.0], [1.0, 7.0]]))

    assert isinstance(result, ExportedQEqResult)
    assert isinstance(result, QEqResult)
    assert hasattr(qeq_module, "solve_qeq")

    fixed_mu = exported_fixed_mu_observables(
        np.diag([-1.0, 1.0]),
        np.eye(2),
        mu=0.0,
        kT=0.0,
        spin_degeneracy=1.0,
    )
    np.testing.assert_allclose(fixed_mu.electron_count, 1.0, atol=1e-12)


def test_qeq_units_reject_unimplemented_unit_relabeling():
    with pytest.raises(QEqOperatorError, match="canonical unscaled"):
        QEqUnits(charge="coulomb")


def test_validate_qeq_result_fails_for_mutated_result():
    result = solve_qeq(np.array([1.0, 3.0]), np.array([[5.0, 1.0], [1.0, 7.0]]))
    bad_diag = qeq_module.QEqDiagnostics(
        charge_residual=np.asarray(1e-3),
        stationarity_max_abs=result.diagnostics.stationarity_max_abs,
        stationarity_l2=result.diagnostics.stationarity_l2,
        energy_identity_residual=result.diagnostics.energy_identity_residual,
        kernel_symmetry_error=result.diagnostics.kernel_symmetry_error,
        constrained_min_eig=result.diagnostics.constrained_min_eig,
        constrained_max_eig=result.diagnostics.constrained_max_eig,
        constrained_condition=result.diagnostics.constrained_condition,
        kkt_condition=result.diagnostics.kkt_condition,
    )
    bad = QEqResult(
        charges=result.charges,
        total_charge=result.total_charge,
        electronegativity=result.electronegativity,
        hardness_kernel=result.hardness_kernel,
        lagrange_multiplier=result.lagrange_multiplier,
        stationarity=result.stationarity,
        energy=result.energy,
        linear_energy=result.linear_energy,
        quadratic_energy=result.quadratic_energy,
        energy_identity=result.energy_identity,
        diagnostics=bad_diag,
        units=result.units,
    )

    with pytest.raises(QEqOperatorError, match="diagnostic charge_residual"):
        validate_qeq_result(bad, atol=1e-8)


def test_qeq_result_arrays_are_read_only_defensive_copies():
    result = solve_qeq(np.array([1.0, 3.0]), np.array([[5.0, 1.0], [1.0, 7.0]]))

    with pytest.raises(ValueError, match="read-only"):
        result.charges[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        result.diagnostics.charge_residual[...] = 1.0

    charges_alias = result.charges.copy()
    rebuilt = replace(result, charges=charges_alias)
    charges_alias[0] += 10.0
    np.testing.assert_allclose(rebuilt.charges, result.charges, atol=1e-12)

    diag_alias = result.diagnostics.stationarity_max_abs.copy()
    rebuilt_diag = replace(result.diagnostics, stationarity_max_abs=diag_alias)
    diag_alias[...] = 10.0
    np.testing.assert_allclose(rebuilt_diag.stationarity_max_abs, result.diagnostics.stationarity_max_abs, atol=1e-12)


def test_validate_qeq_result_recomputes_current_arrays_not_stale_cache():
    result = solve_qeq(np.array([1.0, 3.0]), np.array([[5.0, 1.0], [1.0, 7.0]]))

    bad_charges = result.charges.copy()
    bad_charges[0] += 0.1
    with pytest.raises(QEqOperatorError):
        validate_qeq_result(replace(result, charges=bad_charges), atol=1e-8)

    bad_energy = result.energy + 1.0
    with pytest.raises(QEqOperatorError, match="stored energy"):
        validate_qeq_result(replace(result, energy=bad_energy), atol=1e-8)

    bad_multiplier = result.lagrange_multiplier + 0.25
    with pytest.raises(QEqOperatorError, match="stationarity"):
        validate_qeq_result(replace(result, lagrange_multiplier=bad_multiplier), atol=1e-8)

    bad_kernel = result.hardness_kernel.copy()
    bad_kernel[0, 0] += 0.25
    with pytest.raises(QEqOperatorError):
        validate_qeq_result(replace(result, hardness_kernel=bad_kernel), atol=1e-8)

    bad_diag = replace(
        result.diagnostics,
        stationarity_max_abs=result.diagnostics.stationarity_max_abs + 1.0,
    )
    with pytest.raises(QEqOperatorError, match="diagnostic stationarity_max_abs"):
        validate_qeq_result(replace(result, diagnostics=bad_diag), atol=1e-8)

    bad_symmetry_diag = replace(
        result.diagnostics,
        kernel_symmetry_error=result.diagnostics.kernel_symmetry_error + 1.0,
    )
    with pytest.raises(QEqOperatorError, match="diagnostic kernel_symmetry_error"):
        validate_qeq_result(replace(result, diagnostics=bad_symmetry_diag), atol=1e-8)


def test_validator_preserves_solver_condition_policy_after_serialization():
    import pickle

    result = solve_qeq(
        np.array([1.0, 2.0, 3.0]),
        np.diag([1.0, 10.0, 100.0]),
        max_condition=1.0e4,
        residual_tol=1.0e-9,
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.max_condition == pytest.approx(1.0e4)
    assert restored.residual_tol == pytest.approx(1.0e-9)
    validate_qeq_result(restored)

    with pytest.raises(QEqKernelConditionError, match="recorded constrained"):
        validate_qeq_result(replace(restored, max_condition=1.1))


def test_validate_qeq_result_wraps_zero_site_results():
    result = solve_qeq(np.array([1.0]), np.array([[1.0]]))
    zero_site = replace(
        result,
        charges=np.empty((0,)),
        total_charge=np.asarray(0.0),
        electronegativity=np.empty((0,)),
        hardness_kernel=np.empty((0, 0)),
        lagrange_multiplier=np.asarray(0.0),
        stationarity=np.empty((0,)),
    )

    with pytest.raises(QEqOperatorError, match="at least one charge site"):
        validate_qeq_result(zero_site, atol=1e-8)


def test_invalid_shapes_fail_closed():
    with pytest.raises(QEqOperatorError, match="electronegativity"):
        solve_qeq(np.array(1.0), np.eye(1))
    with pytest.raises(QEqOperatorError, match="square"):
        solve_qeq(np.array([1.0, 2.0]), np.ones((2, 3)))
    with pytest.raises(QEqOperatorError, match="must match"):
        solve_qeq(np.array([1.0, 2.0, 3.0]), np.eye(2))
    with pytest.raises(QEqOperatorError, match="at least one"):
        solve_qeq(np.empty((0,)), np.empty((0, 0)))
    with pytest.raises(QEqOperatorError, match="cannot broadcast"):
        solve_qeq(np.ones((2, 3)), np.broadcast_to(np.eye(3), (4, 3, 3)))
    with pytest.raises(QEqOperatorError, match="total_charge shape"):
        solve_qeq(np.ones((2, 3)), np.eye(3), total_charge=np.ones(3))


def test_nonfinite_complex_and_nonsymmetric_inputs_fail_closed():
    with pytest.raises(QEqOperatorError, match="finite"):
        solve_qeq(np.array([np.nan, 0.0]), np.eye(2))
    with pytest.raises(QEqOperatorError, match="finite"):
        solve_qeq(np.array([0.0, 1.0]), np.array([[1.0, np.inf], [np.inf, 1.0]]))
    with pytest.raises(QEqOperatorError, match="complex"):
        solve_qeq(np.array([1.0 + 0.1j, 2.0]), np.eye(2))
    with pytest.raises(QEqOperatorError, match="symmetric"):
        solve_qeq(np.array([1.0, 2.0]), np.array([[2.0, 0.5], [0.0, 2.0]]), symmetry_tol=1e-12)
    with pytest.raises(QEqOperatorError, match="total_charge must be finite"):
        solve_qeq(np.array([1.0, 2.0]), np.eye(2), total_charge=np.nan)


def test_singular_nonconvex_and_ill_conditioned_kernels_fail_closed():
    with pytest.raises(QEqKernelConditionError, match="positive definite"):
        solve_qeq(np.array([1.0, 2.0]), np.ones((2, 2)))
    with pytest.raises(QEqKernelConditionError, match="positive definite"):
        solve_qeq(np.array([1.0, 2.0, 3.0]), np.diag([-1.0, 2.0, 2.0]))
    with pytest.raises(QEqKernelConditionError, match="condition"):
        solve_qeq(np.array([1.0, 2.0, 3.0]), np.diag([1.0, 100.0, 10000.0]), max_condition=10.0)


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
    with pytest.raises(QEqOperatorError, match="will not detach/cpu tensors silently"):
        solve_qeq(_FakeTorchTensor(np.array([1.0, 2.0])), np.eye(2))
    with pytest.raises(QEqOperatorError, match="will not detach/cpu tensors silently"):
        solve_qeq(np.array([1.0, 2.0]), _FakeTorchTensor(np.eye(2)))
    with pytest.raises(QEqOperatorError, match="will not detach/cpu tensors silently"):
        solve_qeq(np.array([1.0, 2.0]), np.eye(2), total_charge=_FakeTorchTensor(np.array(0.0)))
