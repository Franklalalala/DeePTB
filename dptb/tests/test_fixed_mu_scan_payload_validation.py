from dataclasses import replace

import numpy as np
import pytest

from dptb.nnops.fixed_mu_operator import (
    EnergyLedger,
    FixedMuOperatorError,
    OverlapConditionError,
    fixed_mu_scan,
)


def _result():
    return fixed_mu_scan(
        np.diag([-0.4, 0.8]),
        np.array([[1.0, 0.1], [0.1, 1.0]]),
        [-0.2, 0.3],
        kT=0.1,
    )


def test_scan_result_rejects_relabelled_observables():
    result = _result()

    with pytest.raises(FixedMuOperatorError, match="electron_count"):
        replace(result, electron_count=result.electron_count + 0.25)

    bad_energies = EnergyLedger(
        band_energy=result.energies.band_energy,
        minus_t_s=result.energies.minus_t_s,
        band_free_energy=result.energies.band_free_energy,
        band_grand_energy=result.energies.band_grand_energy + 1.0,
    )
    with pytest.raises(FixedMuOperatorError, match="band_grand_energy"):
        replace(result, energies=bad_energies)


def test_scan_result_rejects_nonfinite_and_impossible_diagnostics():
    result = _result()

    with pytest.raises(FixedMuOperatorError, match="finite"):
        replace(
            result,
            dos_like_response=np.full_like(result.dos_like_response, np.nan),
        )

    with pytest.raises(FixedMuOperatorError, match="k_weights"):
        replace(result, k_weights=np.asarray(-1.0))

    with pytest.raises(FixedMuOperatorError, match="implicit unit weights"):
        replace(result, k_weights=np.asarray(2.0))

    with pytest.raises(OverlapConditionError, match="eig_floor"):
        replace(result, min_overlap_eig=np.asarray(result.eig_floor))


def test_unrepresentable_finite_temperature_response_fails_closed():
    with pytest.raises(FixedMuOperatorError, match="kT is too small"):
        fixed_mu_scan(
            np.array([[0.0]]),
            np.eye(1),
            [0.0],
            kT=1.0e-309,
        )
