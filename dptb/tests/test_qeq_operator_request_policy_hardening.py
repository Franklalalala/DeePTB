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
