"""Sparse periodic linked cells for row-vector, including triclinic, lattices.

No minimum-image approximation: every intersecting periodic image is retained.
Memory is O(N); query cost is local under bounded density, cutoff and cell shape.
The object is a geometry snapshot and must be rebuilt when positions/cell change.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np

from .lattice import pair_translations
from .models import Structure


@dataclass(frozen=True)
class AtomImage:
    atom_index: int
    translation: tuple[int, int, int]
    center_bohr: np.ndarray


class PeriodicAtomIndex:
    """Hash occupied fractional bins; bound Cartesian balls with the dual cell.

    For ||dx|| <= r, |(dx @ inv(cell))[d]| <= r*||inv(cell)[:,d]||.
    Enumerating *unwrapped* bins in that box and doing an exact Cartesian
    distance check supports skew cells, multi-image cutoffs and unwrapped atoms.
    Translations are relative to the original (not wrapped) atom coordinates.
    """

    def __init__(self, structure: Structure, bin_radius_bohr: float):
        radius = float(bin_radius_bohr)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("bin_radius_bohr must be finite and positive")
        self.cell = np.array(structure.cell_bohr, dtype=float, copy=True)
        self.inv_cell = np.linalg.inv(self.cell)
        self.dual_norm = np.linalg.norm(self.inv_cell, axis=0)
        frac = np.asarray([a.frac for a in structure.atoms], dtype=float).reshape(-1, 3)
        self.positions = frac @ self.cell
        self.winding = np.floor(frac).astype(np.int64)
        wrapped = frac - self.winding
        # A cap only coarsens a bin, never discards an image. Avoid integer
        # overflow for otherwise valid but extremely large/vacuum cells.
        self.nbins = np.maximum(
            1, np.floor(np.minimum(1.0 / (radius * self.dual_norm), 2**30))
        ).astype(np.int64)
        self._bins: dict[tuple[int, int, int], list[int]] = {}
        keys = np.floor(wrapped * self.nbins).astype(np.int64) % self.nbins
        for atom_index, key in enumerate(keys):
            self._bins.setdefault(tuple(int(x) for x in key), []).append(atom_index)
        self._stats = dict(queries=0, bin_visits=0, candidate_tests=0, returned_images=0)

    def query(self, target_bohr: np.ndarray, radius_bohr: float) -> list[AtomImage]:
        target = np.asarray(target_bohr, dtype=float)
        radius = float(radius_bohr)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_bohr must be a finite length-3 vector")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius_bohr must be finite and non-negative")
        frac = target @ self.inv_cell
        winding = np.floor(frac).astype(np.int64)
        home = frac - winding
        # Expand only the candidate box for floating conversion at bin edges.
        delta = (radius + 1e-12) * self.dual_norm
        delta += 32 * np.finfo(float).eps * (1 + np.abs(frac))
        lower = np.floor((home - delta) * self.nbins).astype(np.int64)
        upper = np.floor((home + delta) * self.nbins).astype(np.int64)
        self._stats["queries"] += 1
        hits: list[AtomImage] = []
        ranges = [range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper)]
        for raw in product(*ranges):
            raw_array = np.asarray(raw, dtype=np.int64)
            key = tuple(int(x) for x in raw_array % self.nbins)
            self._stats["bin_visits"] += 1
            atom_ids = self._bins.get(key)
            if atom_ids is None:
                continue
            image_winding = raw_array // self.nbins + winding
            for atom_index in atom_ids:
                self._stats["candidate_tests"] += 1
                R = image_winding - self.winding[atom_index]
                center = self.positions[atom_index] + R @ self.cell
                if np.linalg.norm(center - target) <= radius + 1e-12:
                    hits.append(AtomImage(atom_index, tuple(int(x) for x in R), center))
        # Match reference atom-major / translation-lexicographic accumulation.
        hits.sort(key=lambda h: (h.atom_index, h.translation))
        self._stats["returned_images"] += len(hits)
        return hits

    def stats(self) -> dict[str, object]:
        return {
            **self._stats,
            "atoms": len(self.positions),
            "bins_per_axis": self.nbins.tolist(),
            "occupied_bins": len(self._bins),
            "stored_atom_references": sum(map(len, self._bins.values())),
            "maximum_bin_population": max(map(len, self._bins.values()), default=0),
        }


def iter_pair_images(
    structure: Structure,
    orbital_cutoffs: np.ndarray,
    *,
    index: PeriodicAtomIndex | None = None,
    extra_cutoff_bohr: float = 0.0,
) -> Iterator[tuple[int, int, tuple[int, int, int], np.ndarray, np.ndarray]]:
    """Enumerate the same conservative pair support in either backend.

    extra_cutoff=2*max_projector_cutoff also includes disjoint AO supports that
    couple through a common KB projector. This is a conservative superset,
    not an approximation to a projector or a distance-based pruning scheme.
    """
    positions = structure.cart_positions
    cutoffs = np.asarray(orbital_cutoffs, dtype=float)
    if cutoffs.shape != (len(structure.atoms),) or not np.isfinite(cutoffs).all():
        raise ValueError("orbital_cutoffs must contain one finite value per atom")
    if np.any(cutoffs < 0) or not np.isfinite(extra_cutoff_bohr) or extra_cutoff_bohr < 0:
        raise ValueError("cutoffs must be non-negative")
    largest = float(np.max(cutoffs, initial=0.0))
    for i, ci in enumerate(positions):
        if index is None:
            for j, cj_home in enumerate(positions):
                for R in pair_translations(
                    structure.cell_bohr, ci, cj_home,
                    cutoffs[i] + cutoffs[j] + extra_cutoff_bohr,
                ):
                    yield i, j, R, ci, cj_home + np.asarray(R) @ structure.cell_bohr
        else:
            radius = cutoffs[i] + largest + extra_cutoff_bohr
            for hit in index.query(ci, radius):
                j = hit.atom_index
                if np.linalg.norm(hit.center_bohr - ci) <= cutoffs[i] + cutoffs[j] + extra_cutoff_bohr + 1e-12:
                    yield i, j, hit.translation, ci, hit.center_bohr
