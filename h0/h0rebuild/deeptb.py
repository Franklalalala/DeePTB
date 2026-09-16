"""Fail-closed bridge from :mod:`h0rebuild` AO blocks to DeePTB graphs.

The functions in this module do not import DeePTB or PyTorch.  They validate
artifact provenance, units, structure identity, graph coverage, block shapes,
and the real-space Hermiticity relation before DeePTB's ``block_to_feature`` is
allowed to pack the complex AO matrices into its real RME/SOC layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .constants import ANGSTROM_TO_BOHR, EV_TO_RY
from .io import load_blocks, validate_artifact
from .models import BlockKey


DEEPNET_BLOCK_KEY_SCHEMA = "i_j_R1_R2_R3/zero_based"
DEEPNET_RME_REPRESENTATION = "deeptb.rme_soc_real_imag/v1"
_EXPECTED_SPIN_ORDER = "spin-block-major-per-atom:[up_spatial,down_spatial]"
_SUPPORTED_ENERGY_UNITS = frozenset({"eV", "Ry", "Ha"})


@dataclass(frozen=True)
class DeePTBArtifact:
    """Validated blocks in ABACUS AO order; convert harmonics before packing."""

    blocks: dict[str, np.ndarray]
    metadata: dict
    source_energy_unit: str
    target_energy_unit: str
    energy_scale: float
    orbital_counts: tuple[int, ...]


def _normalize_energy_unit(unit: str) -> str:
    text = str(unit).strip().lower()
    aliases = {
        "ev": "eV",
        "ry": "Ry",
        "rydberg": "Ry",
        "ha": "Ha",
        "hartree": "Ha",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported energy unit {unit!r}; expected one of {sorted(_SUPPORTED_ENERGY_UNITS)}"
        ) from exc


def energy_unit_scale(source: str, target: str) -> float:
    """Multiplicative conversion from ``source`` to ``target`` energy units."""
    src = _normalize_energy_unit(source)
    dst = _normalize_energy_unit(target)
    # Convert one source unit to Rydberg, then Rydberg to the target unit.
    to_ry = {"Ry": 1.0, "eV": EV_TO_RY, "Ha": 2.0}[src]
    from_ry = {"Ry": 1.0, "eV": 1.0 / EV_TO_RY, "Ha": 0.5}[dst]
    return float(to_ry * from_ry)


def deeptb_block_key(key: BlockKey | Sequence[int]) -> str:
    """Return DeePTB's zero-based ``i_j_R1_R2_R3`` key string."""
    if isinstance(key, BlockKey):
        values = key.as_tuple()
    else:
        values = tuple(int(v) for v in key)
    if len(values) != 5:
        raise ValueError(f"Block key must contain five integers, got {values!r}")
    return "_".join(str(int(v)) for v in values)


def parse_deeptb_block_key(key: str) -> tuple[int, int, int, int, int]:
    parts = str(key).split("_")
    if len(parts) != 5:
        raise ValueError(f"Malformed DeePTB block key {key!r}")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Malformed DeePTB block key {key!r}") from exc


def load_deeptb_artifact(
    path: str | Path,
    *,
    target_energy_unit: str = "eV",
    verify_hash: bool = True,
) -> DeePTBArtifact:
    """Load, verify, convert, and key a sparse ``h0rebuild`` Hamiltonian.

    The returned arrays remain complex AO matrices.  They must be packed by the
    audited DeePTB ``block_to_feature`` path; taking ``.real`` is forbidden for
    SOC because it deletes the imaginary spin-orbit channels.
    """
    path = Path(path)
    metadata = validate_artifact(path, verify_hash=verify_hash)
    if int(metadata.get("atom_index_base", -1)) != 0:
        raise ValueError("Only zero-based h0rebuild artifacts are supported")
    nspin = int(metadata.get('nspin',4))
    if nspin not in (1,4):
        raise ValueError('Unsupported artifact nspin')
    expected_spin_order = 'scalar-spatial-per-atom' if nspin == 1 else _EXPECTED_SPIN_ORDER
    if metadata.get("spin_order") != expected_spin_order:
        raise ValueError(
            "Incompatible spin order in h0rebuild artifact: "
            f"{metadata.get('spin_order')!r}; expected {expected_spin_order!r}"
        )
    source_unit = _normalize_energy_unit(metadata.get("output_energy_unit", ""))
    target_unit = _normalize_energy_unit(target_energy_unit)
    scale = energy_unit_scale(source_unit, target_unit)

    with np.load(path, allow_pickle=False) as archive:
        counts = tuple(int(v) for v in np.asarray(archive["orbital_counts"]).reshape(-1))
    if not counts or any(v <= 0 for v in counts):
        raise ValueError(f"Artifact orbital_counts must be positive, got {counts}")

    raw = load_blocks(path, prefix="h", validate=False)
    blocks: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        name = deeptb_block_key(key)
        array = np.asarray(value, dtype=np.complex128) * scale
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Hamiltonian block {name} is not a finite rank-2 matrix")
        if not (0 <= key.i < len(counts) and 0 <= key.j < len(counts)):
            raise ValueError(f'Atom index outside orbital_counts: {name}')
        factor = 1 if nspin == 1 else 2
        if array.shape != (factor*counts[key.i],factor*counts[key.j]):
            raise ValueError(f'Block spin dimensions disagree with nspin for {name}')
        if name in blocks:
            raise ValueError(f"Duplicate Hamiltonian block key {name}")
        blocks[name] = np.ascontiguousarray(array)
    return DeePTBArtifact(
        blocks=blocks,
        metadata=metadata,
        source_energy_unit=source_unit,
        target_energy_unit=target_unit,
        energy_scale=scale,
        orbital_counts=counts,
    )


def _numpy(value) -> np.ndarray:
    """Convert NumPy/CPU-Torch-like values without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _integer_edge_shifts(value) -> np.ndarray:
    shifts = _numpy(value)
    if shifts.ndim != 2 or shifts.shape[1] != 3:
        raise ValueError(f"edge_cell_shift must have shape [n_edge,3], got {shifts.shape}")
    rounded = np.rint(shifts)
    if not np.allclose(shifts, rounded, atol=1.0e-8, rtol=0.0):
        raise ValueError("edge_cell_shift contains non-integer lattice translations")
    return rounded.astype(np.int64)


def _block_shape(orbital_counts: Sequence[int], i: int, j: int, has_soc: bool) -> tuple[int, int]:
    factor = 2 if has_soc else 1
    return factor * int(orbital_counts[i]), factor * int(orbital_counts[j])


def preflight_deeptb_graph(
    blocks: Mapping[str, np.ndarray],
    *,
    edge_index,
    edge_cell_shift,
    orbital_counts: Sequence[int],
    has_soc: bool,
    hermitian_atol: float = 1.0e-8,
    hermitian_rtol: float = 1.0e-7,
) -> dict[str, object]:
    """Validate every onsite/edge block before DeePTB feature packing.

    DeePTB's current ``block_to_feature`` fills a missing edge block with zeros.
    This guard deliberately fails instead.  A reverse ``(j,i,-R)`` block is
    accepted because ``block_to_feature`` takes its conjugate transpose.
    """
    counts = [int(v) for v in orbital_counts]
    if not counts or any(v <= 0 for v in counts):
        raise ValueError(f"orbital_counts must contain positive integers, got {counts}")
    edges = _numpy(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,n_edge], got {edges.shape}")
    edges = edges.astype(np.int64, copy=False)
    shifts = _integer_edge_shifts(edge_cell_shift)
    if shifts.shape[0] != edges.shape[1]:
        raise ValueError(
            f"edge row mismatch: edge_index has {edges.shape[1]} columns, "
            f"edge_cell_shift has {shifts.shape[0]} rows"
        )
    n_atom = len(counts)
    if edges.size and (edges.min() < 0 or edges.max() >= n_atom):
        raise ValueError("edge_index references an atom outside orbital_counts")

    def checked(name: str, expected: tuple[int, int]) -> np.ndarray:
        array = np.asarray(blocks[name])
        if array.shape != expected:
            raise ValueError(f"Block {name} has shape {array.shape}, expected {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"Block {name} contains NaN or infinity")
        return array

    onsite_count = 0
    for atom in range(n_atom):
        name = f"{atom}_{atom}_0_0_0"
        if name not in blocks:
            raise KeyError(f"Missing required onsite Hamiltonian block {name}")
        array = checked(name, _block_shape(counts, atom, atom, has_soc))
        residual = array - array.conj().T
        if not np.allclose(array, array.conj().T, atol=hermitian_atol, rtol=hermitian_rtol):
            raise ValueError(
                f"Onsite block {name} is not Hermitian; max residual="
                f"{float(np.max(np.abs(residual))):.3e}"
            )
        onsite_count += 1

    direct_count = 0
    reverse_count = 0
    both_count = 0
    max_pair_residual = 0.0
    for row in range(edges.shape[1]):
        i, j = (int(edges[0, row]), int(edges[1, row]))
        rx, ry, rz = (int(v) for v in shifts[row])
        direct = f"{i}_{j}_{rx}_{ry}_{rz}"
        reverse = f"{j}_{i}_{-rx}_{-ry}_{-rz}"
        expected = _block_shape(counts, i, j, has_soc)
        reverse_expected = (expected[1], expected[0])
        has_direct = direct in blocks
        has_reverse = reverse in blocks
        if not has_direct and not has_reverse:
            raise KeyError(
                f"Missing edge Hamiltonian block for row {row}: neither {direct} nor {reverse} exists"
            )
        direct_array = checked(direct, expected) if has_direct else None
        reverse_array = checked(reverse, reverse_expected) if has_reverse else None
        if has_direct:
            direct_count += 1
        else:
            reverse_count += 1
        if direct_array is not None and reverse_array is not None:
            both_count += 1
            residual = direct_array - reverse_array.conj().T
            max_pair_residual = max(max_pair_residual, float(np.max(np.abs(residual))))
            if not np.allclose(
                direct_array,
                reverse_array.conj().T,
                atol=hermitian_atol,
                rtol=hermitian_rtol,
            ):
                raise ValueError(
                    f"Hermitian counterpart mismatch for {direct}/{reverse}; "
                    f"max residual={float(np.max(np.abs(residual))):.3e}"
                )

    # Validate dimensions of every artifact block whose atom indices belong to
    # this structure.  This catches a stale artifact even when the active graph
    # happens not to request the malformed block.
    for name, value in blocks.items():
        i, j, _rx, _ry, _rz = parse_deeptb_block_key(name)
        if i < 0 or j < 0 or i >= n_atom or j >= n_atom:
            raise ValueError(f"Artifact block {name} references atom outside [0,{n_atom})")
        checked(name, _block_shape(counts, i, j, has_soc))

    return {
        "block_key_schema": DEEPNET_BLOCK_KEY_SCHEMA,
        "has_soc": bool(has_soc),
        "n_atom": n_atom,
        "n_edge": int(edges.shape[1]),
        "onsite_blocks_checked": onsite_count,
        "edge_rows_using_direct_block": direct_count,
        "edge_rows_using_reverse_block": reverse_count,
        "edge_rows_with_both_blocks": both_count,
        "max_counterpart_residual": max_pair_residual,
        "hermitian_atol": float(hermitian_atol),
        "hermitian_rtol": float(hermitian_rtol),
    }


def validate_structure_against_artifact(
    metadata: Mapping[str, object],
    *,
    cell,
    positions,
    species: Sequence[str],
    data_length_unit: str,
    cell_atol_bohr: float = 1.0e-8,
    frac_atol: float = 1.0e-8,
) -> dict[str, object]:
    """Verify atom order, cell, and periodic fractional coordinates exactly enough.

    DeePTB/ASE records normally store Angstrom, while ``h0rebuild`` metadata is
    in bohr.  The caller must state the record unit explicitly; no heuristic unit
    detection is used.
    """
    structure = metadata.get("structure")
    if not isinstance(structure, Mapping):
        raise KeyError("h0rebuild artifact metadata lacks the structure payload")
    ref_cell = np.asarray(structure.get("cell_bohr"), dtype=float)
    ref_frac = np.asarray(structure.get("fractional_coordinates"), dtype=float)
    ref_species = [str(item) for item in structure.get("species", [])]
    got_species = [str(item) for item in species]
    if ref_species != got_species:
        raise ValueError(f"Species/order mismatch: artifact={ref_species}, data={got_species}")

    unit = str(data_length_unit).strip().lower()
    factor = 1.0 if unit in {"bohr", "au", "a.u."} else ANGSTROM_TO_BOHR if unit in {
        "angstrom", "ang", "a", "å"
    } else None
    if factor is None:
        raise ValueError("data_length_unit must be explicitly 'angstrom' or 'bohr'")
    got_cell = _numpy(cell).reshape(3, 3).astype(float) * factor
    got_pos = _numpy(positions).reshape(-1, 3).astype(float) * factor
    if ref_cell.shape != (3, 3) or ref_frac.shape != (len(ref_species), 3):
        raise ValueError("Malformed structure payload in h0rebuild metadata")
    if got_pos.shape != (len(ref_species), 3):
        raise ValueError(
            f"Position shape {got_pos.shape} does not match {len(ref_species)} artifact atoms"
        )
    max_cell = float(np.max(np.abs(got_cell - ref_cell)))
    if not np.allclose(got_cell, ref_cell, atol=cell_atol_bohr, rtol=1.0e-10):
        raise ValueError(f"Cell mismatch against h0rebuild artifact; max |delta|={max_cell:.3e} bohr")
    got_frac = got_pos @ np.linalg.inv(got_cell)
    delta = got_frac - ref_frac
    delta -= np.rint(delta)
    max_frac = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_frac > frac_atol:
        raise ValueError(
            f"Fractional-coordinate/order mismatch against artifact; max periodic |delta|={max_frac:.3e}"
        )
    return {
        "data_length_unit": "bohr" if factor == 1.0 else "angstrom",
        "max_cell_delta_bohr": max_cell,
        "max_periodic_fractional_delta": max_frac,
        "cell_atol_bohr": float(cell_atol_bohr),
        "frac_atol": float(frac_atol),
    }
