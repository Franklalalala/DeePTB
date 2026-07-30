import numpy as np
import pytest

from dptb.nnops import (
    FixedMuOperatorError,
    fixed_mu_scan,
    validate_fixed_mu_scan,
)
import dptb.nnops.fixed_mu_operator as fixed_mu_module


def _two_k_problem():
    h = np.array(
        [
            [[-0.70, 0.12], [0.12, 0.45]],
            [[-0.50, -0.08], [-0.08, 0.80]],
        ]
    )
    s = np.array(
        [
            [[1.00, 0.09], [0.09, 1.00]],
            [[1.00, -0.06], [-0.06, 1.00]],
        ]
    )
    return h, s


def test_true_scan_is_bound_to_original_hs_and_request():
    h, s = _two_k_problem()
    mu_grid = np.array([-0.35, 0.05, 0.40])
    requested_weights = np.array([2.0, 1.0])
    result = fixed_mu_scan(
        h,
        s,
        mu_grid,
        kT=0.08,
        spin_degeneracy=2.0,
        k_weights=requested_weights,
        k_axis=0,
        normalize_k_weights=True,
    )

    validate_fixed_mu_scan(
        result,
        h=h,
        s=s,
        expected_mu_grid=mu_grid,
        expected_kT=0.08,
        expected_spin_degeneracy=2.0,
        expected_k_weights=requested_weights,
        expected_k_axis=0,
        expected_normalize_k_weights=True,
        expected_eig_floor=1.0e-10,
        expected_max_condition=1.0e12,
        expected_hermitian_tol=1.0e-8,
    )


def test_self_consistent_forged_eigvals_are_rejected_by_hs_anchor():
    expected_h = np.array([[-0.45, 0.08], [0.08, 0.70]])
    expected_s = np.array([[1.00, 0.12], [0.12, 1.00]])
    forged_h = expected_h + np.array([[0.20, -0.03], [-0.03, -0.10]])
    forged = fixed_mu_scan(
        forged_h,
        expected_s,
        [-0.30, 0.00, 0.35],
        kT=0.08,
        spin_degeneracy=2.0,
    )

    # Patch 0002 certifies this genuine scan for forged_h against its own
    # stored eigvals and payload. It has no way to know expected_h.
    fixed_mu_module._validate_fixed_mu_scan_payload(forged)
    with pytest.raises(
        FixedMuOperatorError,
        match="stored eigvals do not match caller-supplied H/S",
    ):
        validate_fixed_mu_scan(forged, h=expected_h, s=expected_s)


def test_one_micro_unit_h_perturbation_is_detected():
    h, s = _two_k_problem()
    result = fixed_mu_scan(
        h,
        s,
        [-0.25, 0.10],
        kT=0.05,
        k_weights=[0.4, 0.6],
        k_axis=0,
    )
    perturbed_h = h + 1.0e-6 * s

    with pytest.raises(
        FixedMuOperatorError,
        match="stored eigvals do not match caller-supplied H/S",
    ):
        validate_fixed_mu_scan(result, h=perturbed_h, s=s)


def test_gauge_shifted_h_and_mu_grid_validate_covariantly():
    h, s = _two_k_problem()
    mu_grid = np.array([-0.25, 0.10, 0.55])
    c = 0.37
    weights = np.array([0.4, 0.6])
    original = fixed_mu_scan(
        h,
        s,
        mu_grid,
        kT=0.07,
        k_weights=weights,
        k_axis=0,
    )
    shifted = fixed_mu_scan(
        h + c * s,
        s,
        mu_grid + c,
        kT=0.07,
        k_weights=weights,
        k_axis=0,
    )

    validate_fixed_mu_scan(
        shifted,
        h=h + c * s,
        s=s,
        expected_mu_grid=mu_grid + c,
        expected_kT=0.07,
        expected_k_weights=weights,
        expected_k_axis=0,
    )
    np.testing.assert_allclose(
        shifted.electron_count,
        original.electron_count,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        shifted.dos_like_response,
        original.dos_like_response,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        shifted.energies.band_energy,
        original.energies.band_energy + c * original.electron_count,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        shifted.energies.band_grand_energy,
        original.energies.band_grand_energy,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("wrong_request", "message"),
    [
        ({"expected_kT": 0.080000000001}, "not the expected"),
        ({"expected_spin_degeneracy": 1.0}, "spin_degeneracy"),
        (
            {"expected_k_weights": np.array([2.000000000001, 1.0])},
            "expected k_weights",
        ),
    ],
)
def test_expected_thermal_spin_and_weight_requests_are_exact_by_default(
    wrong_request,
    message,
):
    h, s = _two_k_problem()
    result = fixed_mu_scan(
        h,
        s,
        [-0.30, 0.20],
        kT=0.08,
        spin_degeneracy=2.0,
        k_weights=[2.0, 1.0],
        k_axis=0,
    )

    with pytest.raises(FixedMuOperatorError, match=message):
        validate_fixed_mu_scan(
            result,
            h=h,
            s=s,
            atol=1.0e-2,
            **wrong_request,
        )


def test_mu_grid_request_tolerance_requires_explicit_opt_in():
    h = np.diag([-0.4, 0.7])
    s = np.eye(2)
    result = fixed_mu_scan(h, s, [0.0, 0.3], kT=0.1)
    nearby = np.array([0.0, 0.300000000001])

    with pytest.raises(FixedMuOperatorError, match="expected mu_grid"):
        validate_fixed_mu_scan(
            result,
            h=h,
            s=s,
            atol=1.0,
            expected_mu_grid=nearby,
        )
    validate_fixed_mu_scan(
        result,
        h=h,
        s=s,
        expected_mu_grid=nearby,
        request_atol=2.0e-12,
    )


def test_condition_number_near_one_e10_is_not_falsely_rejected():
    rng = np.random.default_rng(20260730)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    s = fixed_mu_module.hermitian_part(
        q @ np.diag(np.logspace(0.0, 10.0, 6)) @ q.T
    )
    raw_h = rng.normal(size=(6, 6))
    h = fixed_mu_module.hermitian_part(raw_h + raw_h.T)
    result = fixed_mu_scan(
        h,
        s,
        [-0.20, 0.15],
        kT=0.1,
        max_condition=1.0e12,
    )

    assert 5.0e9 < float(result.overlap_condition) < 2.0e10
    validate_fixed_mu_scan(result, h=h, s=s)


def test_external_hs_hermiticity_and_overlap_gates_are_reapplied():
    h = np.diag([-0.4, 0.7])
    s = np.eye(2)
    result = fixed_mu_scan(h, s, [0.0], kT=0.1)

    nonhermitian_h = h.copy()
    nonhermitian_h[0, 1] = 1.0e-4
    with pytest.raises(FixedMuOperatorError, match="H must be Hermitian"):
        validate_fixed_mu_scan(result, h=nonhermitian_h, s=s)

    with pytest.raises(fixed_mu_module.OverlapConditionError, match="eig_floor"):
        validate_fixed_mu_scan(
            result,
            h=h,
            s=np.diag([1.0e-14, 1.0]),
        )
