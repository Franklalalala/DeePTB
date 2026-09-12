"""Physical constants and unit conversions.

CODATA 2018 values as adopted by NIST:
  1 bohr = 0.529177210903 angstrom
  1 hartree = 27.211386245988 eV
The Rydberg energy unit used by ABACUS/QE is half a hartree.
"""
from __future__ import annotations

import numpy as np

BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV: float = 27.211386245988
RY_TO_EV: float = HARTREE_TO_EV / 2.0
EV_TO_RY: float = 1.0 / RY_TO_EV
FOUR_PI: float = 4.0 * np.pi
E2_RY_BOHR: float = 2.0  # e^2 in Rydberg atomic units


def length_to_bohr(value, unit: str):
    unit_l = unit.lower()
    if unit_l in {"bohr", "a.u.", "au"}:
        return np.asarray(value, dtype=float)
    if unit_l in {"angstrom", "ang", "å", "a"}:
        return np.asarray(value, dtype=float) * ANGSTROM_TO_BOHR
    raise ValueError(f"Unsupported length unit: {unit!r}")


def energy_from_ry(value, unit: str):
    unit_l = unit.lower()
    arr = np.asarray(value)
    if unit_l in {"ry", "rydberg"}:
        return arr
    if unit_l in {"ev"}:
        return arr * RY_TO_EV
    if unit_l in {"ha", "hartree"}:
        return arr / 2.0
    raise ValueError(f"Unsupported energy unit: {unit!r}")
