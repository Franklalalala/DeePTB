from dataclasses import replace

import numpy as np
import pytest

from dptb.nnops.fixed_mu_scf_operator import (
    FixedMuSCFConvergenceError,
    FixedMuSCFError,
    fixed_mu_electrostatic_scf,
)


def _one_level(**overrides):
    kwargs = {
        "h0": np.array([[0.1]]),
        "s": np.eye(1),
        "mu": 0.0,
        "kT": 0.2,
        "ao_atom_index": [0],
        "reference_populations": [1.0],
        "coulomb_kernel": np.array([[0.8]]),
        "mixing": "linear",
        "mixing_step": 0.5,
        "max_iter": 1000,
        "charge_tol": 1e-12,
        "potential_tol": 1e-11,
    }
    kwargs.update(overrides)
    return fixed_mu_electrostatic_scf(**kwargs)


def test_charge_tolerance_cannot_hide_large_potential_mismatch():
    # The first fixed-point charge residual is < 2 e, but K amplifies it into a
    # > 10-energy-unit Hamiltonian mismatch.  The old charge-only gate returned
    # a cross-iterate result from this single iteration.
    with pytest.raises(FixedMuSCFConvergenceError, match="last_potential_residual"):
        _one_level(
            coulomb_kernel=np.array([[10.0]]),
            charge_tol=2.0,
            potential_tol=0.1,
            max_iter=1,
        )


def test_default_charge_tolerance_cannot_hide_amplified_potential_mismatch():
    # The current input contract accepts this finite symmetric kernel and the
    # default 1e-8 charge tolerance.  Charge-only convergence occurs after 85
    # linear steps, while K @ delta_q is still about 9e3 energy units.
    with pytest.raises(FixedMuSCFConvergenceError, match="last_potential_residual"):
        _one_level(
            reference_populations=[3.0],
            coulomb_kernel=np.array([[1.0e12]]),
            charge_tol=1.0e-8,
            potential_tol=0.1,
            max_iter=100,
        )


def test_converged_result_records_both_closure_certificates():
    gamma = 0.8
    result = _one_level(coulomb_kernel=np.array([[gamma]]))

    assert result.residual_history[-1] <= result.charge_tol
    assert result.potential_residual <= result.potential_tol
    assert result.potential_residual == pytest.approx(
        gamma * result.residual_history[-1], rel=1e-13, abs=1e-15
    )

    with pytest.raises(FixedMuSCFError, match="potential_residual"):
        replace(
            result,
            potential_residual=2.0 * result.potential_tol,
        )

def test_scf_result_metadata_cannot_be_relabelled():
    result = _one_level()

    with pytest.raises(FixedMuSCFError, match="mu does not match"):
        replace(result, mu=result.mu + 1.0e-3)

    with pytest.raises(FixedMuSCFError, match="spin_degeneracy does not match"):
        replace(result, spin_degeneracy=1.0)
