import numpy as np
import pytest

from dptb.nnops.fixed_mu_operator import (
    FixedMuOperatorError,
    fixed_mu_observables,
    validate_conservation,
)


def test_mu_request_binding_is_independent_of_conservation_tolerance():
    h = np.array([[2.0e-9]])
    s = np.eye(1)
    result = fixed_mu_observables(
        h,
        s,
        mu=5.0e-9,
        kT=0.0,
        spin_degeneracy=2.0,
    )
    assert float(result.electron_count) == 2.0

    with pytest.raises(FixedMuOperatorError, match="not the expected"):
        validate_conservation(
            result,
            h=h,
            s=s,
            expected_mu=0.0,
        )

    # Approximate request matching remains available only by explicit opt-in.
    validate_conservation(
        result,
        h=h,
        s=s,
        request_atol=1.0e-8,
        expected_mu=0.0,
    )
