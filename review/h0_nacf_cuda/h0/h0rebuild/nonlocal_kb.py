from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from .harmonics import complex_ylm, mj_values, spinor_cg
from .lattice import nearby_atom_images
from .models import SpeciesData, Structure, UPFData
from .quadrature import intersection_grid
from .radial import OrbitalEvaluator, projector_beta_values


def _projector_overlaps(
    upf: UPFData,
    projector_positions: list[int],
    evaluator: OrbitalEvaluator,
    basis_center: np.ndarray,
    projector_center: np.ndarray,
    step_bohr: float,
    chunk_size: int = 100_000,
) -> dict[int, np.ndarray]:
    """Return ``<beta_{p,l,m}|phi_mu>`` for complex ``m=-l..l``."""
    result: dict[int, np.ndarray] = {}
    for ppos in projector_positions:
        proj = upf.projectors[ppos]
        grid = intersection_grid(
            projector_center,
            proj.cutoff_radius,
            basis_center,
            evaluator.basis.rcut,
            step_bohr,
        )
        q = np.zeros((2 * proj.l + 1, evaluator.norb), dtype=complex)
        if len(grid.points) == 0:
            result[ppos] = q
            continue
        for start in range(0, len(grid.points), chunk_size):
            pts = grid.points[start : start + chunk_size]
            rel = pts - projector_center
            radii = np.linalg.norm(rel, axis=1)
            beta = projector_beta_values(
                upf.r, proj.radial_u, proj.cutoff_radius, radii
            )
            phi = evaluator.values(pts, basis_center)
            for row, m in enumerate(range(-proj.l, proj.l + 1)):
                angular_bra = np.conjugate(complex_ylm(proj.l, m, rel))
                q[row] += (beta * angular_bra) @ phi * grid.weight
        result[ppos] = q
    return result


def _scalar_group_block(
    ppos_i: list[int],
    ppos_j: list[int],
    d: np.ndarray,
    qi: dict[int, np.ndarray],
    qj: dict[int, np.ndarray],
) -> np.ndarray:
    """Contract one scalar KB ``(l)`` group with independent bra/ket sets."""
    if d.shape != (len(ppos_i), len(ppos_j)):
        raise ValueError(
            f"D subblock shape {d.shape} != ({len(ppos_i)}, {len(ppos_j)})"
        )
    ni = qi[ppos_i[0]].shape[1]
    nj = qj[ppos_j[0]].shape[1]
    out = np.zeros((ni, nj), dtype=complex)
    for ia, p in enumerate(ppos_i):
        for ib, q in enumerate(ppos_j):
            if abs(d[ia, ib]) < 1e-18:
                continue
            out += d[ia, ib] * np.conjugate(qi[p]).T @ qj[q]
    return out


def _spinor_group_block(
    ppos_i: list[int],
    ppos_j: list[int],
    l: int,
    j: float,
    d: np.ndarray,
    qi: dict[int, np.ndarray],
    qj: dict[int, np.ndarray],
) -> np.ndarray:
    """Contract one fully relativistic ``(l,j)`` projector group.

    The bra and ket active projector lists are intentionally independent.  An
    off-diagonal ``D_pq`` remains valid when projector ``p`` overlaps only the
    bra basis and projector ``q`` overlaps only the ket basis.
    """
    if d.shape != (len(ppos_i), len(ppos_j)):
        raise ValueError(
            f"D subblock shape {d.shape} != ({len(ppos_i)}, {len(ppos_j)})"
        )
    ni = qi[ppos_i[0]].shape[1]
    nj = qj[ppos_j[0]].shape[1]
    mjs = mj_values(j)
    bi = np.zeros((len(ppos_i), len(mjs), 2, ni), dtype=complex)
    bj = np.zeros((len(ppos_j), len(mjs), 2, nj), dtype=complex)
    for a, p in enumerate(ppos_i):
        for imj, mj in enumerate(mjs):
            for spin in (0, 1):
                ml, coeff = spinor_cg(l, j, float(mj), spin)
                if coeff and -l <= ml <= l:
                    bi[a, imj, spin] = coeff * qi[p][ml + l]
    for b, p in enumerate(ppos_j):
        for imj, mj in enumerate(mjs):
            for spin in (0, 1):
                ml, coeff = spinor_cg(l, j, float(mj), spin)
                if coeff and -l <= ml <= l:
                    bj[b, imj, spin] = coeff * qj[p][ml + l]
    # Output axes are (bra_spin, bra_orb, ket_spin, ket_orb).
    out4 = np.einsum("pmsa,pq,qmtb->satb", np.conjugate(bi), d, bj, optimize=True)
    return out4.reshape(2 * ni, 2 * nj)


def _group_projectors(
    upf: UPFData,
    positions: list[int],
) -> dict[tuple[int, float | None], list[int]]:
    groups: dict[tuple[int, float | None], list[int]] = defaultdict(list)
    for ppos in positions:
        projector = upf.projectors[ppos]
        groups[(projector.l, projector.j if upf.has_so else None)].append(ppos)
    return groups


def _reference_projector_centers(
    structure: Structure,
    species_data: Mapping[str, SpeciesData],
    center_i: np.ndarray,
    center_j: np.ndarray,
    orbital_radius: float,
):
    """Original all-atom candidate traversal, retained as the reference path."""
    midpoint = 0.5 * (center_i + center_j)
    separation = np.linalg.norm(center_i - center_j)
    positions = structure.cart_positions
    for atom_index, atom in enumerate(structure.atoms):
        upf = species_data[atom.species].upf
        if not upf.projectors:
            continue
        radius = upf.max_projector_cutoff + orbital_radius + 0.5 * separation
        for _, pcenter in nearby_atom_images(
            structure.cell_bohr, positions[atom_index], midpoint, radius
        ):
            yield atom_index, pcenter


def nonlocal_pair_block(
    structure: Structure,
    species_data: Mapping[str, SpeciesData],
    evaluator_i: OrbitalEvaluator,
    evaluator_j: OrbitalEvaluator,
    center_i: np.ndarray,
    center_j: np.ndarray,
    step_bohr: float,
    *,
    _candidate_centers: Iterable[tuple[int, np.ndarray]] | None = None,
    nspin: int = 4,
) -> np.ndarray:
    """Full norm-conserving KB block, including projector-level SOC.

    _candidate_centers is an internal conservative candidate iterator from the
    assembly's spatial index. It must include every possible common center.
    The unchanged active-projector tests below define the physical support.
    """
    ni, nj = evaluator_i.norb, evaluator_j.norb
    if nspin not in (1, 4):
        raise ValueError('nspin must be 1 or 4')
    multiplier = 2 if nspin == 4 else 1
    total = np.zeros((multiplier * ni, multiplier * nj), dtype=complex)
    if _candidate_centers is None:
        _candidate_centers = _reference_projector_centers(
            structure, species_data, center_i, center_j,
            max(evaluator_i.basis.rcut, evaluator_j.basis.rcut),
        )
    for atom_index, pcenter in _candidate_centers:
        upf = species_data[structure.atoms[atom_index].species].upf
        if nspin == 1 and upf.has_so:
            raise ValueError('nspin=1 requires scalarize_upf before KB contraction')
        active_i = [
            pos
            for pos, projector in enumerate(upf.projectors)
            if np.linalg.norm(pcenter - center_i)
            <= projector.cutoff_radius + evaluator_i.basis.rcut
        ]
        active_j = [
            pos
            for pos, projector in enumerate(upf.projectors)
            if np.linalg.norm(pcenter - center_j)
            <= projector.cutoff_radius + evaluator_j.basis.rcut
        ]
        if not active_i or not active_j:
            continue
        qi = _projector_overlaps(
            upf, active_i, evaluator_i, center_i, pcenter, step_bohr
        )
        qj = _projector_overlaps(
            upf, active_j, evaluator_j, center_j, pcenter, step_bohr
        )
        groups_i = _group_projectors(upf, active_i)
        groups_j = _group_projectors(upf, active_j)

        for group in sorted(set(groups_i) & set(groups_j), key=str):
            l, j = group
            ppos_i = groups_i[group]
            ppos_j = groups_j[group]
            d = upf.dij_ry[np.ix_(ppos_i, ppos_j)]
            if not np.any(np.abs(d) > 1e-18):
                continue
            if j is None:
                scalar = _scalar_group_block(ppos_i, ppos_j, d, qi, qj)
                total[:ni, :nj] += scalar
                if nspin == 4:
                    total[ni:, nj:] += scalar
            else:
                total += _spinor_group_block(
                    ppos_i, ppos_j, l, float(j), d, qi, qj
                )
    return total
