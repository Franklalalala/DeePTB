"""Geometry-identity and reverse-edge helpers for the durable materializers.

These verify that a compact record's stored geometry is the same structure as
its authoritative ``STRU`` file (modulo periodic images and serialization
round-off), and provide the directed-edge reverse permutation used when
rotating packed AO block tensors between the ABACUS and DeePTB harmonic gauges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import ase.data
import numpy as np

from dptb.utils.constants import Bohr2Ang
import dptb.data.AtomicDataDict as AtomicDataDict

from .hashing import sha256_file


# Compact records and STRU cells are serialized through different decimal
# paths.  Raw200 contains legitimate cell round-off just above 5e-5 Angstrom;
# 1e-4 remains far below any physically meaningful geometry displacement.
GEOMETRY_TOLERANCE_ANGSTROM = 1.0e-4


def reverse_edge_rows(edge_index: Any, edge_cell_shift: Any) -> np.ndarray:
    """Return, for every directed edge row, the row of its reverse ``(j,i,-R)``.

    The result is validated to be an exact involution over the main directed
    graph, which is required before any reverse-transpose gauge symmetrization.
    """

    edge = np.asarray(edge_index, dtype=np.int64)
    shift_raw = np.asarray(edge_cell_shift, dtype=np.float64)
    shift = np.rint(shift_raw).astype(np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2 or shift.shape != (edge.shape[1], 3):
        raise ValueError("edge graph has invalid shape for gauge transformation.")
    if float(np.max(np.abs(shift_raw - shift), initial=0.0)) > 1.0e-6:
        raise ValueError("edge shifts are not integer-valued.")
    lookup: dict[tuple[int, int, int, int, int], int] = {}
    for row, ((i, j), cell_shift) in enumerate(zip(edge.T, shift)):
        key = (int(i), int(j), *(int(value) for value in cell_shift))
        if key in lookup:
            raise ValueError(f"duplicate directed graph row {key}.")
        lookup[key] = row
    reverse = np.empty(edge.shape[1], dtype=np.int64)
    for row, ((i, j), cell_shift) in enumerate(zip(edge.T, shift)):
        key = (int(j), int(i), *(-cell_shift).tolist())
        if key not in lookup:
            raise ValueError(f"edge row {row} has no reverse row {key}.")
        reverse[row] = lookup[key]
    if not np.array_equal(reverse[reverse], np.arange(len(reverse))):
        raise ValueError("reverse-edge map is not an involution.")
    return reverse


def structure_geometry_bohr(
    *,
    record: Mapping[str, Any],
    case_path: Path,
    gate1: Any,
    table_species: Mapping[str, Any],
    tolerance_angstrom: float = GEOMETRY_TOLERANCE_ANGSTROM,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse ``STRU`` and assert it is the compact record's geometry (in Bohr).

    Positions are compared modulo lattice vectors on periodic axes (an
    ABACUS/ASE conversion may wrap an atom into another image), while cells and
    non-periodic axes are compared exactly up to ``tolerance_angstrom``.
    """

    parsed = gate1.parse_stru(case_path / "STRU")
    structure = parsed.structure
    symbols = [str(atom.species) for atom in structure.atoms]
    numbers = np.asarray(
        [ase.data.atomic_numbers[symbol] for symbol in symbols], dtype=np.int64
    )
    stored_numbers = np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]).reshape(-1)
    if not np.array_equal(numbers, stored_numbers.astype(np.int64)):
        raise ValueError(f"{case_path.name}: STRU atom order differs from compact record.")
    missing_species = sorted(set(symbols) - set(table_species))
    if missing_species:
        raise KeyError(f"P23 table lacks structure species {missing_species}.")

    positions_bohr = np.asarray(structure.cart_positions, dtype=np.float64)
    cell_bohr = np.asarray(structure.cell_bohr, dtype=np.float64)
    stored_positions = np.asarray(
        record[AtomicDataDict.POSITIONS_KEY], dtype=np.float64
    ).reshape(-1, 3)
    stored_cell = np.asarray(record[AtomicDataDict.CELL_KEY], dtype=np.float64).reshape(3, 3)
    if positions_bohr.shape != stored_positions.shape or cell_bohr.shape != (3, 3):
        raise ValueError(f"{case_path.name}: STRU/compact geometry shapes differ.")
    stru_positions_angstrom = positions_bohr * Bohr2Ang
    stru_cell_angstrom = cell_bohr * Bohr2Ang
    raw_position_error = float(
        np.max(np.abs(stru_positions_angstrom - stored_positions), initial=0.0)
    )
    # ABACUS/ASE conversion may wrap an atom into another image of the same
    # periodic cell.  Compare positions modulo lattice vectors instead of
    # requiring the same Cartesian representative.  Non-periodic directions
    # remain exact and therefore cannot be hidden by this normalization.
    pbc = np.asarray(record[AtomicDataDict.PBC_KEY], dtype=bool).reshape(-1)
    if pbc.size == 1:
        pbc = np.repeat(pbc, 3)
    if pbc.size != 3:
        raise ValueError(f"{case_path.name}: compact PBC field is not length 1 or 3.")
    try:
        fractional_delta = (
            stru_positions_angstrom - stored_positions
        ) @ np.linalg.inv(stru_cell_angstrom)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{case_path.name}: STRU cell is singular.") from exc
    image_shift = np.zeros_like(fractional_delta)
    image_shift[:, pbc] = np.rint(fractional_delta[:, pbc])
    minimum_image_delta = (fractional_delta - image_shift) @ stru_cell_angstrom
    position_error = float(
        np.max(np.abs(minimum_image_delta), initial=0.0)
    )
    cell_error = float(
        np.max(np.abs(cell_bohr * Bohr2Ang - stored_cell), initial=0.0)
    )
    if max(position_error, cell_error) > float(tolerance_angstrom):
        raise ValueError(
            f"{case_path.name}: compact geometry is not the STRU geometry in "
            f"Angstrom (position error={position_error:.3e}, "
            f"cell error={cell_error:.3e} Angstrom)."
        )
    return symbols, positions_bohr, cell_bohr, {
        "compact_length_unit": "angstrom",
        "assembler_length_unit": "bohr",
        "bohr_to_angstrom": float(Bohr2Ang),
        "max_position_identity_error_angstrom": position_error,
        "max_raw_position_image_error_angstrom": raw_position_error,
        "position_identity_semantics": "minimum_image_on_periodic_axes",
        "max_cell_identity_error_angstrom": cell_error,
        "stru_sha256": sha256_file(case_path / "STRU"),
    }
