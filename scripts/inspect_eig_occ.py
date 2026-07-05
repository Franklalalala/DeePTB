#!/usr/bin/env python3
"""Inspect an ABACUS-style eig_occ.txt for n_occ and HOMO-LUMO gaps.

Absorbed (lightly adapted) from the external grassmann_m_b patch package -- a reusable,
dependency-light parser so the product-flow ``n_occ`` / ``min_gap`` config values come from
the real DFT reference instead of being guessed.

The parser is conservative: comments and blank lines are skipped, numeric columns are
loaded, and the caller chooses which columns hold k-index, energy, and occupation.

NOTE on ABACUS ``eig_occ.txt`` layout: rows are ``band_index energy occupation`` inside a
per-k header block, so use ``--energy-col 1 --occ-col 2`` (the defaults 0/1 are for a bare
two-column ``energy occupation`` file).  For a SOC (nspin=4) calculation pass ``--soc`` so
the reported ``electron_count_effective`` uses spin degeneracy 1 (one electron per spinor
band), i.e. n_occ == electron count (NOT electron count / 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _load_numeric(path: Path) -> np.ndarray:
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        vals = []
        ok = True
        for part in line.replace(",", " ").split():
            try:
                vals.append(float(part))
            except ValueError:
                ok = False
                break
        if ok and vals:
            rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    width = max(len(r) for r in rows)
    padded = [r + [np.nan] * (width - len(r)) for r in rows]
    return np.asarray(padded, dtype=float)


def _groups(data: np.ndarray, k_col: Optional[int]):
    if k_col is None:
        yield None, data
        return
    keys = data[:, k_col]
    for key in np.unique(keys[~np.isnan(keys)]):
        yield int(key) if float(key).is_integer() else float(key), data[keys == key]


def inspect(path: Path, *, energy_col: int, occ_col: int, k_col: Optional[int],
            occ_threshold: float, spin_degeneracy: float):
    data = _load_numeric(path)
    out = {"file": str(path), "spin_degeneracy": spin_degeneracy, "kpoints": []}
    gaps = []
    occs = []
    for key, block in _groups(data, k_col):
        block = block[np.isfinite(block[:, energy_col]) & np.isfinite(block[:, occ_col])]
        if block.size == 0:
            continue
        order = np.argsort(block[:, energy_col])
        e = block[order, energy_col]
        occ = block[order, occ_col]
        occupied = occ > occ_threshold
        n_occ = int(np.count_nonzero(occupied))
        gap = None
        homo = lumo = None
        if 0 < n_occ < len(e):
            homo = float(e[n_occ - 1])
            lumo = float(e[n_occ])
            gap = lumo - homo
            gaps.append(gap)
        occs.append(n_occ)
        out["kpoints"].append({
            "k": key, "n_occ": n_occ,
            "electron_count_effective": float(n_occ * spin_degeneracy),
            "homo": homo, "lumo": lumo, "gap": gap,
        })
    if not occs:
        raise ValueError("could not derive any occupied counts")
    vals, counts = np.unique(np.asarray(occs), return_counts=True)
    out["n_occ_mode"] = int(vals[np.argmax(counts)])
    out["n_occ_min"] = int(np.min(occs))
    out["n_occ_max"] = int(np.max(occs))
    out["gap_min"] = None if not gaps else float(np.min(gaps))
    out["gap_median"] = None if not gaps else float(np.median(gaps))
    out["num_kpoints"] = len(out["kpoints"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--energy-col", type=int, default=0)
    p.add_argument("--occ-col", type=int, default=1)
    p.add_argument("--k-col", type=int, default=None)
    p.add_argument("--occ-threshold", type=float, default=0.5)
    p.add_argument("--soc", action="store_true",
                   help="SOC spinor bands: one state per band; spin degeneracy defaults to 1")
    p.add_argument("--spin-degeneracy", type=float, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    spin = args.spin_degeneracy if args.spin_degeneracy is not None else (1.0 if args.soc else 2.0)
    result = inspect(args.path, energy_col=args.energy_col, occ_col=args.occ_col, k_col=args.k_col,
                     occ_threshold=args.occ_threshold, spin_degeneracy=spin)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
