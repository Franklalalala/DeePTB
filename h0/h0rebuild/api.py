from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .assemble import AssemblyResult, assemble_h0
from .constants import length_to_bohr
from .models import Atom, SpeciesData, Structure
from .orb import read_abacus_orb
from .upf import read_upf


def _load_by_element(paths: Sequence[str | Path], reader, kind: str):
    out = {}
    for path in paths:
        obj = reader(path)
        key = obj.element
        if key in out:
            raise ValueError(f"Duplicate {kind} element {key!r}: {path}")
        out[key] = obj
    return out


def reconstruct_from_files(
    upf_files: Mapping[str, str | Path] | Sequence[str | Path],
    orb_files: Mapping[str, str | Path] | Sequence[str | Path],
    cell: np.ndarray,
    fractional_coordinates: np.ndarray,
    species: Sequence[str],
    *,
    length_unit: str = "angstrom",
    **assemble_options,
) -> AssemblyResult:
    """Convenience API matching the declared artifact interface.

    ``upf_files`` and ``orb_files`` may be mappings keyed by the structure's
    species labels, or lists whose entries are paired automatically by the
    element names stored inside the files. ``cell`` contains row lattice
    vectors. The returned matrices use Rydberg internally; :func:`save_result`
    writes explicit eV/Ry/Ha output.
    """
    if isinstance(upf_files, Mapping):
        upfs = {str(k): read_upf(v) for k, v in upf_files.items()}
    else:
        upfs = _load_by_element(list(upf_files), read_upf, "UPF")
    if isinstance(orb_files, Mapping):
        orbs = {str(k): read_abacus_orb(v) for k, v in orb_files.items()}
    else:
        orbs = _load_by_element(list(orb_files), read_abacus_orb, ".orb")

    labels = set(str(x) for x in species)
    missing_upf = labels - set(upfs)
    missing_orb = labels - set(orbs)
    if missing_upf or missing_orb:
        raise KeyError(
            f"Unpaired species files; missing UPF={sorted(missing_upf)}, "
            f"missing .orb={sorted(missing_orb)}"
        )
    coords = np.asarray(fractional_coordinates, dtype=float)
    if coords.shape != (len(species), 3):
        raise ValueError(
            f"fractional_coordinates must have shape ({len(species)}, 3), got {coords.shape}"
        )
    structure = Structure(
        cell_bohr=length_to_bohr(cell, length_unit),
        atoms=[Atom(str(symbol), coords[i]) for i, symbol in enumerate(species)],
    )
    data = {symbol: SpeciesData(orbs[symbol], upfs[symbol]) for symbol in sorted(labels)}
    return assemble_h0(structure, data, **assemble_options)
