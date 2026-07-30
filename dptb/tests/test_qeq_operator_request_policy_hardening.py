import copy

import numpy as np
import pytest

from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    fixed_mu_observables,
    validate_conservation,
)
from dptb.nnops.qeq_operator import (
    QEqOperatorError,
    QEqUnits,
    solve_qeq,
    validate_qeq_result,
)


def test_qeq_validator_can_bind_the_original_request():
    requested_chi = np.array([1.0, 3.0])
    requested_kernel = np.array([[5.0, 1.0], [1.0, 7.0]])
    requested_total_charge = 0.5

    # This is a valid result, but for a different request.  Internal
    # self-consistency alone cannot distinguish the two problems.
    wrong_result = solve_qeq(
        requested_chi + 10.0,
        2.0 * requested_kernel,
        total_charge=-1.0,
    )
    validate_qeq_result(wrong_result)

    with pytest.raises(QEqOperatorError, match="electronegativity request"):
        validate_qeq_result(
            wrong_result,
            expected_electronegativity=requested_chi,
            expected_hardness_kernel=requested_kernel,
            expected_total_charge=requested_total_charge,
        )

    result = solve_qeq(
        requested_chi,
        requested_kernel,
        total_charge=requested_total_charge,
    )
    validate_qeq_result(
        result,
        expected_electronegativity=requested_chi,
        expected_hardness_kernel=requested_kernel,
        expected_total_charge=requested_total_charge,
    )
    with pytest.raises(QEqOperatorError, match="hardness_kernel request"):
        validate_qeq_result(
            result,
            expected_hardness_kernel=requested_kernel + np.eye(2),
        )
    with pytest.raises(QEqOperatorError, match="total_charge request"):
        validate_qeq_result(result, expected_total_charge=-requested_total_charge)


def test_qeq_total_charge_binding_does_not_use_energy_tolerance():
    result = solve_qeq(
        np.array([0.1, 0.2]),
        np.array([[0.5, 0.1], [0.1, 0.6]]),
        total_charge=0.3,
        residual_tol=1.0e-6,
    )

    with pytest.raises(QEqOperatorError, match="total_charge request"):
        validate_qeq_result(result, expected_total_charge=0.3 + 5.0e-9)


def test_qeq_request_binding_broadcasts_with_custom_unit_labels():
    rng = np.random.default_rng(20260730)
    chi = rng.normal(size=(2, 1, 3))
    raw = rng.normal(size=(4, 3, 3))
    kernel = np.swapaxes(raw, -1, -2) @ raw + np.eye(3)
    total_charge = np.array([[0.2], [-0.4]])
    units = QEqUnits(
        charge="electron",
        energy="Ha",
        electronegativity="Ha/electron",
        hardness_kernel="Ha/electron^2",
        lagrange_multiplier="Ha/electron",
    )

    result = solve_qeq(
        chi,
        kernel,
        total_charge=total_charge,
        units=units,
    )
    validate_qeq_result(
        result,
        expected_electronegativity=chi,
        expected_hardness_kernel=kernel,
        expected_total_charge=total_charge,
    )
    assert result.charges.shape == (2, 4, 3)
    assert result.units == units

    wrong_chi = np.array(chi, copy=True)
    wrong_chi[1, 0, 2] += 1.0
    with pytest.raises(QEqOperatorError, match="electronegativity request"):
        validate_qeq_result(result, expected_electronegativity=wrong_chi)


@pytest.mark.parametrize(
    ("factor", "accepted"),
    [
        (1.0 - 1.0e-10, True),
        (1.0, True),
        (1.0 + 1.0e-10, False),
    ],
)
def test_fixed_mu_condition_ceiling_boundary(factor, accepted):
    condition = 1.0e12 * factor
    h = np.diag([-1.0, 1.0])
    s = np.diag([1.0 / condition, 1.0])

    if accepted:
        result = fixed_mu_observables(
            h,
            s,
            mu=0.0,
            eig_floor=1.0e-15,
            max_condition=1.0e12,
        )
        assert float(result.overlap_condition) <= 1.0e12
    else:
        with pytest.raises(FixedMuOperatorError, match="condition number"):
            fixed_mu_observables(
                h,
                s,
                mu=0.0,
                eig_floor=1.0e-15,
                max_condition=1.0e12,
            )


def test_fixed_mu_rejects_self_declared_unsafe_policy():
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)

    with pytest.raises(FixedMuOperatorError, match="module ceiling"):
        fixed_mu_observables(h, s, mu=0.0, max_condition=1.0e30)
    with pytest.raises(FixedMuOperatorError, match="module ceiling"):
        fixed_mu_observables(h, s, mu=0.0, hermitian_tol=1.0)

    result = fixed_mu_observables(h, s, mu=0.0)
    forged = copy.copy(result)
    object.__setattr__(forged, "max_condition", 1.0e30)
    with pytest.raises(FixedMuOperatorError, match="module ceiling"):
        validate_conservation(forged)
