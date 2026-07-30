import pickle

import numpy as np
import pytest

from dptb.nnops.fixed_mu_scf_operator import (
    FixedMuSCFConvergenceError,
    FixedMuSCFError,
    _pdiis_gram_condition,
    fixed_mu_electrostatic_scf,
)


def _one_level(
    *,
    epsilon=0.1,
    mu=0.0,
    kT=0.2,
    gamma=0.8,
    mixing_step=0.35,
    **overrides,
):
    kwargs = {
        "h0": np.array([[epsilon]], dtype=np.float64),
        "s": np.eye(1),
        "mu": mu,
        "kT": kT,
        "ao_atom_index": [0],
        "reference_populations": [1.0],
        "coulomb_kernel": np.array([[gamma]], dtype=np.float64),
        "mixing": "pdiis",
        "mixing_step": mixing_step,
        "n_history": 6,
        "mixing_period": 3,
        "max_iter": 1000,
        "charge_tol": 1e-10,
        "potential_tol": 1e-10,
    }
    kwargs.update(overrides)
    return fixed_mu_electrostatic_scf(**kwargs)


def test_review_p1_3_failure_is_safeguarded_and_legacy_is_reproducible():
    kwargs = {
        "epsilon": 1.67,
        "mu": -0.04,
        "kT": 0.012,
        "gamma": 1.72,
        "mixing_step": 0.2,
        "max_iter": 200,
        "charge_tol": 1e-8,
        "potential_tol": 1e-8,
    }

    safeguarded = _one_level(**kwargs)

    assert safeguarded.iterations == 19
    assert safeguarded.pdiis_safeguard is True
    assert safeguarded.pdiis_gram_condition_threshold == 1e10
    assert safeguarded.pdiis_step_ratio_threshold == 40.0
    assert safeguarded.pdiis_residual_growth_threshold == 1.0
    assert safeguarded.pdiis_gram_fallbacks == 0
    assert safeguarded.pdiis_step_ratio_fallbacks == 0
    assert safeguarded.pdiis_residual_growth_fallbacks == 2

    with pytest.raises(FixedMuSCFConvergenceError) as caught:
        _one_level(**kwargs, pdiis_safeguard=False)

    assert caught.value.iterations == 200
    assert caught.value.residual_history[-1] > 0.1


def test_gram_condition_fallback_uses_linear_candidate_and_resets_history():
    # A one-level numerical Gram range has condition one.  A deliberately
    # lower explicit threshold forces this branch without relying on NaNs.
    result = _one_level(
        pdiis_gram_condition_threshold=0.5,
        pdiis_step_ratio_threshold=1e100,
        pdiis_residual_growth_threshold=1e100,
    )

    assert result.pdiis_gram_fallbacks > 0
    assert result.pdiis_step_ratio_fallbacks == 0
    assert result.pdiis_residual_growth_fallbacks == 0


def test_step_ratio_fallback_uses_linear_candidate_and_resets_history():
    result = _one_level(
        pdiis_gram_condition_threshold=1e100,
        pdiis_step_ratio_threshold=0.01,
        pdiis_residual_growth_threshold=1e100,
    )

    assert result.pdiis_gram_fallbacks == 0
    assert result.pdiis_step_ratio_fallbacks > 0
    assert result.pdiis_residual_growth_fallbacks == 0


def test_effective_gram_condition_excludes_structural_nullspace():
    # Two collinear history rows in a one-dimensional state make the raw 2x2
    # Gram singular, but the numerical range used by pinv is well conditioned.
    condition = _pdiis_gram_condition(
        [np.array([1.0]), np.array([2.0])]
    )

    assert condition == pytest.approx(1.0)


@pytest.mark.parametrize(
    "epsilon, mu, kT, gamma, mixing_step",
    [
        (-1.5, -0.5, 0.005, 1.2, 0.03),
        (-1.5, 0.5, 0.005, 2.0, 0.1),
        (-1.5, 0.5, 0.01, 2.0, 0.2),
        (-0.5, 0.5, 0.005, 1.2, 0.03),
        (0.5, -0.5, 0.005, 1.2, 0.03),
        (1.5, -0.5, 0.005, 2.0, 0.1),
        (1.5, -0.5, 0.01, 2.0, 0.2),
        (1.5, 0.5, 0.005, 1.2, 0.03),
    ],
)
def test_safeguard_retains_sampled_reviewer_linear_only_cases(
    epsilon, mu, kT, gamma, mixing_step
):
    result = _one_level(
        epsilon=epsilon,
        mu=mu,
        kT=kT,
        gamma=gamma,
        mixing_step=mixing_step,
        max_iter=200,
        charge_tol=1e-8,
        potential_tol=1e-8,
    )

    assert result.residual_history[-1] <= result.charge_tol
    assert result.potential_residual <= result.potential_tol


def test_safeguard_strategy_pickle_roundtrip_and_missing_fields_fail_closed():
    result = _one_level()
    restored = pickle.loads(pickle.dumps(result))

    for field_name in (
        "pdiis_safeguard",
        "pdiis_gram_condition_threshold",
        "pdiis_step_ratio_threshold",
        "pdiis_residual_growth_threshold",
        "pdiis_gram_fallbacks",
        "pdiis_step_ratio_fallbacks",
        "pdiis_residual_growth_fallbacks",
    ):
        assert getattr(restored, field_name) == getattr(result, field_name)

    incomplete_state = result.__getstate__()
    del incomplete_state["pdiis_step_ratio_threshold"]
    incomplete = object.__new__(type(result))
    with pytest.raises(FixedMuSCFError, match="lacks the safeguarded-PDIIS"):
        incomplete.__setstate__(incomplete_state)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"pdiis_safeguard": 1}, "must be a boolean"),
        (
            {"pdiis_gram_condition_threshold": 0.0},
            "pdiis_gram_condition_threshold must be positive",
        ),
        (
            {"pdiis_step_ratio_threshold": np.inf},
            "pdiis_step_ratio_threshold must be finite",
        ),
        (
            {"pdiis_residual_growth_threshold": -1.0},
            "pdiis_residual_growth_threshold must be positive",
        ),
    ],
)
def test_invalid_safeguard_parameters_fail_closed(override, match):
    with pytest.raises(FixedMuSCFError, match=match):
        _one_level(**override)
