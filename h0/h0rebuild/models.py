from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OrbitalChannel:
    l: int
    zeta: int
    radial: np.ndarray
    # Position of this channel in the .orb file.  Keeping it makes basis-order
    # provenance explicit and prevents an apparently harmless sort from silently
    # permuting AO/RME channels.
    source_index: int = -1


@dataclass
class OrbitalBasis:
    element: str
    ecut_ry: float
    r: np.ndarray
    dr: float
    channels: list[OrbitalChannel]
    source: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def rcut(self) -> float:
        return float(self.r[-1])

    @property
    def lmax(self) -> int:
        return max((c.l for c in self.channels), default=-1)

    @property
    def norb(self) -> int:
        return sum(2 * c.l + 1 for c in self.channels)

    def descriptors(self) -> list["OrbitalDescriptor"]:
        out: list[OrbitalDescriptor] = []
        for ic, channel in enumerate(self.channels):
            for m in abacus_m_order(channel.l):
                out.append(OrbitalDescriptor(ic, channel.l, channel.zeta, m))
        return out


@dataclass(frozen=True)
class OrbitalDescriptor:
    channel_index: int
    l: int
    zeta: int
    m: int


@dataclass(frozen=True)
class Projector:
    index: int
    l: int
    j: float | None
    radial_u: np.ndarray  # UPF PP_BETA stores u_l(r)=r*beta_l(r)
    cutoff_index: int
    cutoff_radius: float


@dataclass
class UPFData:
    element: str
    z_valence: float
    r: np.ndarray
    rab: np.ndarray
    vloc_ry: np.ndarray
    rhoatom_q: np.ndarray  # q(r)=4*pi*r^2*rho_atom(r)
    dij_ry: np.ndarray
    projectors: list[Projector]
    has_so: bool
    functional: str = ""
    nlcc: np.ndarray | None = None
    nlcc_is_radial_density: bool = True
    source: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def max_projector_cutoff(self) -> float:
        return max((p.cutoff_radius for p in self.projectors), default=0.0)


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


@dataclass
class SpeciesData:
    orb: OrbitalBasis
    upf: UPFData


@dataclass(frozen=True)
class BlockKey:
    i: int
    j: int
    R: tuple[int, int, int]

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.i, self.j, *self.R)


def abacus_m_order(l: int) -> list[int]:
    """ABACUS real-harmonic order: 0,+1,-1,+2,-2,..."""
    order = [0]
    for m in range(1, l + 1):
        order.extend((m, -m))
    return order


def iter_species(atoms: Iterable[Atom]) -> set[str]:
    return {a.species for a in atoms}
