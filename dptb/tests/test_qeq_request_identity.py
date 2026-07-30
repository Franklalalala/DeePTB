from dataclasses import replace

import numpy as np
import pytest

from dptb.nnops.qeq_operator import (
    QEqOperatorError,
    QEqUnits,
    solve_qeq,
    validate_qeq_result,
)


def test_request_binding_is_independent_of_stationarity_tolerance():
    tangent_eigenvalue = 2.0e-12
    z = np.array([1.0, -1.0]) / np.sqrt(2.0)
    kernel = tangent_eigenvalue * np.outer(z, z)
    submitted_chi = np.array([5.0e-9, -5.0e-9])

    result = solve_qeq(
        submitted_chi,
        kernel,
        total_charge=0.0,
        residual_tol=1.0e-8,
    )
    assert np.max(np.abs(result.charges)) > 2.4e3

    with pytest.raises(QEqOperatorError, match="electronegativity request"):
        validate_qeq_result(
            result,
            expected_electronegativity=np.zeros(2),
            expected_hardness_kernel=kernel,
            expected_total_charge=0.0,
        )

    # Approximate request matching remains possible, but only by explicit opt-in.
    validate_qeq_result(
        result,
        request_atol=1.0e-8,
        expected_electronegativity=np.zeros(2),
        expected_hardness_kernel=kernel,
        expected_total_charge=0.0,
    )


def test_validator_can_bind_declarative_units():
    requested_units = QEqUnits()
    result = solve_qeq(
        np.array([0.2, -0.1]),
        np.array([[2.0, 0.2], [0.2, 1.7]]),
        total_charge=0.0,
        units=requested_units,
    )
    relabelled = replace(result, units=replace(requested_units, energy="Ha"))

    with pytest.raises(QEqOperatorError, match="units do not match"):
        validate_qeq_result(relabelled, expected_units=requested_units)

    validate_qeq_result(result, expected_units=requested_units)


def test_total_charge_request_binding_is_exact_by_default():
    result = solve_qeq(
        electronegativity=[0.0, 0.0],
        hardness_kernel=[[2.0, 0.0], [0.0, 2.0]],
        total_charge=1.0e-14,
    )

    with pytest.raises(QEqOperatorError, match="total_charge request"):
        validate_qeq_result(result, expected_total_charge=0.0)

    validate_qeq_result(
        result,
        request_atol=1.0e-13,
        expected_total_charge=0.0,
    )
