"""Self-consistency guard unit tests (hrebuild, plan §3.4-1).

Calibration fixtures are the measured values from the 2026-07-03 production
repair-line test (998933 CFM iter100000, three large-gap SOC cases):
  case_0154 residual_mean 2.7e-4 eV (healthy, floor-limited)  -> skip (below gain floor)
  case_0193 residual_mean 1.9e-3 eV (sick, inside basin)      -> accept
  case_0008 residual_mean 7.0e-2 eV (outside basin, blew up)  -> reject
"""
import numpy as np
import pytest

from dptb.postprocess.hrebuild import (
    SCGuardConfig,
    _coerce_sc_guard,
    evaluate_sc_guard,
    parse_scf_energies_ev,
    self_consistency_residual,
)


# Verbatim row shapes from ABACUS running_scf.log final-energy tables; each
# row carries the SAME energy in (Ry, eV) columns -- the parser must take the
# eV column and derive the Harris/KS gap across rows, not across columns.
LOG_0008 = """
  E_KohnSham     -380.3861913450      -5175.4196428107
  E_Harris       -355.5499559313      -4837.5053243140
  E_Fermi        -0.1106552797        -1.5055423183
"""

LOG_0154 = """
  E_KohnSham     -286.9922200589      -3904.7294744705
  E_Harris       -286.9760728400      -3904.5097802873
  E_Fermi        0.3681094976         5.0083866547
"""


def test_parse_scf_energies_takes_ev_column_and_row_gap():
    e = parse_scf_energies_ev(LOG_0008)
    assert e["e_kohnsham_ev"] == pytest.approx(-5175.4196428107)
    assert e["e_harris_ev"] == pytest.approx(-4837.5053243140)
    assert e["harris_ks_gap_ev"] == pytest.approx(337.914, abs=1e-2)

    e2 = parse_scf_energies_ev(LOG_0154)
    assert e2["harris_ks_gap_ev"] == pytest.approx(0.2197, abs=1e-3)


def test_parse_scf_energies_missing_rows():
    e = parse_scf_energies_ev("no energies here")
    assert e["e_kohnsham_ev"] is None and e["harris_ks_gap_ev"] is None


def test_parse_scf_energies_takes_last_occurrence():
    text = LOG_0154 + "\n" + LOG_0008
    e = parse_scf_energies_ev(text)
    assert e["harris_ks_gap_ev"] == pytest.approx(337.914, abs=1e-2)


def test_self_consistency_residual_units_and_common_keys():
    a = {"0_0_0_0_0": np.eye(2), "0_1_0_0_0": np.zeros((2, 2))}
    b = {"0_0_0_0_0": np.eye(2) + 0.01, "1_1_0_0_0": np.eye(2)}
    r = self_consistency_residual(a, b, unit="eV")
    assert r["n_common"] == 1
    assert r["residual_mean_ev"] == pytest.approx(0.01)
    # Hartree-unit blocks are converted to eV
    r_ha = self_consistency_residual(a, b, unit="Ha")
    assert r_ha["residual_mean_ev"] == pytest.approx(0.01 * 27.211386245988)


def test_self_consistency_residual_no_common():
    r = self_consistency_residual({"a": np.eye(1)}, {"b": np.eye(1)})
    assert r["n_common"] == 0 and np.isnan(r["residual_mean_ev"])


@pytest.mark.parametrize(
    "residual,expect_ok,reason_word",
    [
        (2.7e-4, False, "gain floor"),   # case_0154: repair gain below floor
        (1.9e-3, True, None),            # case_0193: inside basin -> accept
        (7.0e-2, False, "basin"),        # case_0008: outside basin -> reject
    ],
)
def test_guard_calibration_three_cases(residual, expect_ok, reason_word):
    ok, reason = evaluate_sc_guard(residual, SCGuardConfig())
    assert ok is expect_ok
    if reason_word:
        assert reason_word in reason


def test_guard_nan_rejects():
    ok, reason = evaluate_sc_guard(float("nan"), SCGuardConfig())
    assert not ok and "undefined" in reason


def test_coerce_sc_guard_forms():
    assert _coerce_sc_guard(None) is None
    assert _coerce_sc_guard(False) is None
    assert isinstance(_coerce_sc_guard(True), SCGuardConfig)
    cfg = _coerce_sc_guard({"max_residual_mean_ev": 0.5})
    assert cfg.max_residual_mean_ev == 0.5
    with pytest.raises(TypeError):
        _coerce_sc_guard(3.14)
