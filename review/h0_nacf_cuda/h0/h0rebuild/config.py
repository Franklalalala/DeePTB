from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import length_to_bohr
from .models import Atom, SpeciesData, Structure
from .orb import read_abacus_orb
from .upf import read_upf


def load_config(path: str | Path):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    length_unit = cfg.get("length_unit", "angstrom")
    cell = length_to_bohr(cfg["cell"], length_unit)
    atoms = [Atom(str(a["species"]), np.asarray(a["frac"], dtype=float)) for a in cfg["atoms"]]
    structure = Structure(cell_bohr=cell, atoms=atoms)

    species_data: dict[str, SpeciesData] = {}
    if "species" in cfg:
        entries = cfg["species"]
    else:
        upfs = cfg.get("upf_files", {})
        orbs = cfg.get("orb_files", {})
        if set(upfs) != set(orbs):
            raise KeyError(
                "upf_files and orb_files must contain the same species labels; "
                f"UPF-only={sorted(set(upfs) - set(orbs))}, "
                f"ORB-only={sorted(set(orbs) - set(upfs))}"
            )
        entries = {s: {"upf": upfs[s], "orb": orbs[s]} for s in sorted(upfs)}
    for symbol in sorted(entries):
        entry = entries[symbol]
        upf_path = Path(entry["upf"])
        orb_path = Path(entry["orb"])
        if not upf_path.is_absolute():
            upf_path = base / upf_path
        if not orb_path.is_absolute():
            orb_path = base / orb_path
        species_data[symbol] = SpeciesData(orb=read_abacus_orb(orb_path), upf=read_upf(upf_path))

    numerics = dict(cfg.get("numerics", {}))
    physics = dict(cfg.get("physics", {}))
    output = dict(cfg.get("output", {}))
    return structure, species_data, numerics, physics, output, cfg
