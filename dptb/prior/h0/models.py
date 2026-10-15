from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dptb.prior.common.pporb import (OrbitalBasis, OrbitalChannel, OrbitalDescriptor, Projector, SpeciesData,
                                      UPFData, abacus_m_order)


@dataclass(frozen=True)
class Atom:
    species: str
    frac: np.ndarray


@dataclass
class Structure:
    cell_bohr: np.ndarray  # row vectors
    atoms: list[Atom]

    def __post_init__(self) -> None:
        self.cell_bohr = np.asarray(self.cell_bohr, dtype=float)
        if self.cell_bohr.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        if not np.isfinite(self.cell_bohr).all():
            raise ValueError("cell contains non-finite values")
        if abs(np.linalg.det(self.cell_bohr)) < 1e-12:
            raise ValueError("cell is singular")
        normalized_atoms: list[Atom] = []
        for atom in self.atoms:
            frac = np.asarray(atom.frac, dtype=float)
            if frac.shape != (3,) or not np.isfinite(frac).all():
                raise ValueError(
                    f"fractional coordinate for species {atom.species!r} must be a finite length-3 vector"
                )
            normalized_atoms.append(Atom(str(atom.species), frac))
        self.atoms = normalized_atoms

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell_bohr)))

    @property
    def cart_positions(self) -> np.ndarray:
        return np.asarray([a.frac for a in self.atoms], dtype=float) @ self.cell_bohr

    @property
    def reciprocal_rows(self) -> np.ndarray:
        return 2.0 * np.pi * np.linalg.inv(self.cell_bohr).T


@dataclass(frozen=True)
class BlockKey:
    i: int
    j: int
    R: tuple[int, int, int]

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.i, self.j, *self.R)


def iter_species(atoms: Iterable[Atom]) -> set[str]:
    return {a.species for a in atoms}
