from __future__ import annotations

import itertools

import numpy as np


def translation_bounds(cell_bohr: np.ndarray, radius: float) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(cell_bohr, dtype=float))
    # For a Cartesian row vector x, fractional component i is x dot inv[:,i].
    return np.ceil(radius * np.linalg.norm(inv, axis=0)).astype(int) + 1


def integer_translations(cell_bohr: np.ndarray, radius: float):
    bounds = translation_bounds(cell_bohr, radius)
    for n in itertools.product(*(range(-b, b + 1) for b in bounds)):
        yield np.asarray(n, dtype=int)


def pair_translations(
    cell_bohr: np.ndarray,
    center_i: np.ndarray,
    center_j_home: np.ndarray,
    cutoff_sum: float,
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    cell = np.asarray(cell_bohr, dtype=float)
    for n in integer_translations(cell, cutoff_sum + np.linalg.norm(center_j_home - center_i)):
        cj = center_j_home + n @ cell
        if np.linalg.norm(cj - center_i) <= cutoff_sum + 1e-12:
            out.append(tuple(int(x) for x in n))
    return sorted(set(out))


def nearby_atom_images(
    cell_bohr: np.ndarray,
    base_center: np.ndarray,
    target: np.ndarray,
    radius: float,
):
    """Yield (integer translation, Cartesian image) near a target point."""
    cell = np.asarray(cell_bohr, dtype=float)
    inv = np.linalg.inv(cell)
    guess = np.rint((target - base_center) @ inv).astype(int)
    bounds = translation_bounds(cell, radius)
    for off in itertools.product(*(range(-b, b + 1) for b in bounds)):
        n = guess + np.asarray(off, dtype=int)
        center = base_center + n @ cell
        if np.linalg.norm(center - target) <= radius + 1e-12:
            yield tuple(int(x) for x in n), center
