"""Shared input and provider checks of the full NACF candidate prior (NumPy only, no Torch).

Raw structure arrays are validated before any cast (:func:`validated_geometry`), pair-XC table headers are
compared with the P2 shell definitions (:func:`pair_shell_problems`), declared source hashes are compared across
families (:func:`species_source_problems`) and the onsite provider's XC declaration is resolved against the recipe
vocabulary (:func:`declared_xc`). Nothing here reads table contents, evaluates physics or infers what an
arbitrary callable computes: these are declaration and input checks, and the boundary of what they can prove is
stated on each function.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .candidate_policy import (XC_CONTRADICTION_TOKENS, XC_LABEL_TOKENS, CandidateIdentityError, CandidateRecipe,
                               label_denies_potential, xc_label_tokens)

SOURCE_KEYS = ("upf_sha256", "orbital_sha256", "source_sha256")
GRAPH_INTEGER_LIMIT = 2**31 - 1      # the native topology core accepts exact integers of at most this magnitude


class CandidateInputError(ValueError):
    """Malformed structure input: shape, finiteness, integrality or index range of the caller's arrays."""


# --------------------------------------------------------------------------- structure input
def integer_graph_array(value, name: str) -> np.ndarray:
    """An ``int64`` copy of a graph array, validated on the raw values before the cast.

    Integer dtypes pass through; floating arrays are accepted only when finite and exactly integral (a legal
    integer-valued float graph). Fractional cell shifts or indices are refused instead of being truncated to a
    different graph; the supported magnitude is that of the native topology core. Row order is preserved.
    """
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise CandidateInputError(f"{name} must be an integer (or exactly integral floating) array, got dtype {raw.dtype}")
    if raw.dtype.kind == "f":
        if not np.isfinite(raw).all():
            raise CandidateInputError(f"{name} must be finite")
        if not np.array_equal(raw, np.floor(raw)):
            raise CandidateInputError(f"{name} must hold exact integers; fractional entries are neither cell translations nor atom indices")
    # Compare Python scalars: NumPy may otherwise round INT32_MAX to 2**31 in float32.
    if raw.size and (raw.min().item() < -GRAPH_INTEGER_LIMIT or raw.max().item() > GRAPH_INTEGER_LIMIT):
        raise CandidateInputError(f"{name} exceeds the supported integer magnitude {GRAPH_INTEGER_LIMIT}")
    return np.array(raw, dtype=np.int64, copy=True)


def validated_geometry(g: Mapping[str, Any]) -> dict[str, Any]:
    """Own copies of one structure (``symbols``, ``positions_bohr``, ``cell_bohr``, ``edge_index``, ``edge_cell_shift``,
    ``pbc``) after checking the raw arrays: finite ``[n, 3]`` positions and a nondegenerate ``[3, 3]`` cell, three
    boolean ``pbc`` flags, exact-integer ``[2, E]`` indices inside the structure and ``[E, 3]`` shifts. Edge rows keep
    the caller's order; nothing is sorted or deduplicated here (the assembly plan enforces reverse closure).
    """
    symbols = [str(s) for s in g["symbols"]]
    if not symbols:
        raise CandidateInputError("empty structures are not supported")
    positions = np.array(g["positions_bohr"], dtype=np.float64, copy=True)
    cell = np.array(g["cell_bohr"], dtype=np.float64, copy=True)
    if positions.shape != (len(symbols), 3):
        raise CandidateInputError(f"positions_bohr must be [{len(symbols)}, 3] to match symbols, got {positions.shape}")
    if cell.shape != (3, 3):
        raise CandidateInputError(f"cell_bohr must be [3, 3], got {cell.shape}")
    if not (np.isfinite(positions).all() and np.isfinite(cell).all()):
        raise CandidateInputError("positions_bohr and cell_bohr must be finite")
    if not abs(float(np.linalg.det(cell))) > 0:
        raise CandidateInputError("degenerate cell")
    pbc_raw = np.asarray(g.get("pbc", (True, True, True)))
    if pbc_raw.shape != (3,) or not np.isin(pbc_raw, [0, 1]).all():
        raise CandidateInputError("pbc must contain three booleans")
    pbc = tuple(bool(x) for x in pbc_raw)
    edge_index = integer_graph_array(g["edge_index"], "edge_index")
    edge_cell_shift = integer_graph_array(g["edge_cell_shift"], "edge_cell_shift")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise CandidateInputError(f"edge_index must be [2, E], got {edge_index.shape}")
    if edge_cell_shift.shape != (edge_index.shape[1], 3):
        raise CandidateInputError(f"edge_cell_shift must be [{edge_index.shape[1]}, 3], got {edge_cell_shift.shape}")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(symbols)):
        raise CandidateInputError("edge_index refers to an atom outside the structure")
    return dict(symbols=symbols, positions_bohr=positions, cell_bohr=cell, edge_index=edge_index, edge_cell_shift=edge_cell_shift, pbc=pbc)


# --------------------------------------------------------------------------- provider declarations
def normalize_sources(row: Mapping[str, Any] | None) -> dict[str, str]:
    """Species source hashes in one vocabulary: ``upf_sha256``, ``orbital_sha256``, ``source_sha256``."""
    if not row:
        return {}
    out = {}
    for key in SOURCE_KEYS:
        if row.get(key):
            out[key] = str(row[key])
    nested = row.get("sha256")
    if isinstance(nested, Mapping):
        for name, key in (("upf", "upf_sha256"), ("orbital", "orbital_sha256")):
            if nested.get(name):
                out[key] = str(nested[name])
    for name, key in (("upf", "upf_sha256"), ("orbital", "orbital_sha256")):
        if isinstance(row.get(name), str) and len(row[name]) == 64:
            out[key] = row[name]
    return out


def species_source_problems(symbol: str, declared: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Source hashes one species declares in every family (``declared[family] = normalized sources``).

    ``declared['p2']`` is the reference: every family must carry the P2 keys (UPF and ORB, or the single
    ``source_sha256`` of synthetic sources) and every shared key must agree. Disjoint key sets, or a family without
    any hash, do not establish identity and are reported as problems. Hashes prove provenance only; shell
    compatibility is checked separately from the table headers.
    """
    problems = []
    reference = declared["p2"]
    required = set(reference) & {"upf_sha256", "orbital_sha256"}
    if not required and "source_sha256" in reference:
        required = {"source_sha256"}
    for name, src in declared.items():
        if not src:
            problems.append(f"{symbol}: family {name} declares no source hash")
        elif not required or not required.issubset(src):
            problems.append(f"{symbol}: family {name} has no comparable complete P2 source identity; required {sorted(required)}")
    for key in SOURCE_KEYS:
        values = {name: src[key] for name, src in declared.items() if key in src}
        if len(set(values.values())) > 1:
            problems.append(f"{symbol}: {key} differs across families {values}")
    return problems


def pair_shell_problems(tables: Mapping[tuple[str, str], Any], p2_species: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Ordered left/right shell sequences of every pair-XC table header against the P2 species shells.

    Equal AO counts do not establish the same shell decomposition: two s shells plus one p shell and a single d
    shell are both five AOs, yet a table in the other gauge would be added index by index to the P2 blocks without
    any shape error. Species unknown to P2 are skipped here; a structure using them is refused by the coverage
    check. ``support_bohr`` is not compared: production pair tables may legitimately use their own grid extent.
    """
    problems = []
    for (a, b), table in sorted(tables.items()):
        for species, side, declared in ((a, "left", table.left_shells), (b, "right", table.right_shells)):
            row = p2_species.get(species)
            if row is None:
                continue
            expected = tuple(int(l) for l in row["orbital_shells"])
            actual = tuple(int(l) for l in declared)
            if actual != expected:
                problems.append(f"pair XC {a}|{b} {side} shells {actual} vs P2 {species} {expected}")
    return problems


def onsite_quadrature_problems(symbol: str, quadrature: Any, expected_norb: int) -> list[str]:
    """Check a selected onsite grid before global AO padding can hide a species mismatch.

    Shape checks establish AO count and point alignment, not shell order, phases, weights or provenance.
    The caller supplies the quadrature object; this function does not construct grids or evaluate physics.
    """
    xyz = tuple(getattr(getattr(quadrature, "xyz", None), "shape", ()))
    basis = tuple(getattr(getattr(quadrature, "basis", None), "shape", ()))
    if len(xyz) != 2 or xyz[1] != 3 or xyz[0] < 1:
        return [f"onsite {symbol}: quadrature xyz must have nonempty shape [points, 3], got {xyz}"]
    expected = (xyz[0], int(expected_norb))
    if basis != expected:
        return [f"onsite {symbol}: quadrature basis shape {basis} vs expected {expected} from P2 AO count"]
    return []


def declared_xc(onsite_identity: Mapping[str, Any], recipe: CandidateRecipe) -> str:
    """Canonical XC key the onsite provider declares, checked against the recipe.

    ``onsite_identity['xc_functional']`` (the canonical key or the recipe label) is the contract; it must name the
    recipe functional. ``onsite_identity['potential']`` stays a free implementation label kept for provenance: it is
    accepted as the declaration only when it recognizably names the implemented functional (``'pz81'`` token), and
    it is rejected when it recognizably contradicts the recipe (another functional, or no potential at all). What
    the injected callable actually computes is not inferred here; that remains the caller's responsibility.
    """
    key, label = recipe.xc_key, recipe.xc_functional
    potential = onsite_identity.get("potential")
    tokens = xc_label_tokens(potential) if potential is not None else set()
    conflict = sorted(tokens & XC_CONTRADICTION_TOKENS)
    if conflict or (potential is not None and label_denies_potential(potential)):
        raise CandidateIdentityError(f"onsite potential label {potential!r} contradicts the recipe functional {key!r} ({label})"
                                     + (f": it names {conflict}" if conflict else ": it declares no XC potential"))
    explicit = onsite_identity.get("xc_functional")
    if explicit is not None:
        if explicit not in (key, label):
            raise CandidateIdentityError(f"onsite_identity declares xc_functional {explicit!r}; the recipe implements {key!r} ({label})")
        return key
    if tokens & XC_LABEL_TOKENS:
        return key
    raise CandidateIdentityError(f"onsite_identity must declare xc_functional={key!r}; the implementation label {potential!r} "
                                 "does not recognizably name the implemented functional")


__all__ = ["SOURCE_KEYS", "GRAPH_INTEGER_LIMIT", "CandidateInputError", "integer_graph_array", "validated_geometry",
           "normalize_sources", "species_source_problems", "pair_shell_problems", "onsite_quadrature_problems", "declared_xc"]
