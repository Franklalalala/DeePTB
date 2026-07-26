from __future__ import annotations

"""Pure physical-prior primitives shared by the online flow prior and the
offline EMolFlow preprocess tool.

These functions build the extended-Hückel / basis-onsite initial-Hamiltonian
guess from DeePTB's ``onsite_energy_database`` and an ``OrbitalMapper``-like
``idp``.  They are intentionally free of any ``HamiltonianCFM`` (or other
trainer) dependency so the online prior in :mod:`dptb.nnops.flow` and the
offline dataset preprocessing in the EMolFlow repo can share a single
implementation and never drift.  Divergence between the two used to silently
desynchronize offline datasets from the online prior; this module is the one
source of truth for:

* :func:`basis_onsite_energy`   -- onsite level lookup with the starred /
  highest-``n`` fallback rules,
* :func:`basis_onsite_table`    -- the ``[num_types, raw_dim]`` diagonal onsite
  table in the raw (layout-agnostic) orbpair convention,
* :func:`basis_onsite_type_mean`-- per-type mean onsite level, and
* :func:`huckel_edge_energy`    -- the per-edge ``0.5 * (mean_src + mean_dst)``
  Wolfsberg-Helmholz endpoint energy, including the strict-basis raise behavior.

Callers that need a compressed / SOC feature layout (the flow prior) project the
raw :func:`basis_onsite_table` output into their target layout afterwards; the
table produced here is deliberately layout-agnostic.
"""

import re
from typing import Any, Optional

import torch

from dptb.nnops.onsite_database import onsite_energy_database


def orbital_l(orbital: str) -> int:
    """Angular momentum ``l`` of an orbital label (``s``->0, ``p``->1, ...)."""
    letters = re.findall(r"[A-Za-z]", str(orbital))
    if not letters:
        return 0
    return {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}.get(
        letters[-1].lower(),
        0,
    )


def basis_onsite_energy(symbol: str, orbital: str, missing: float = 0.0) -> float:
    """Onsite energy for ``symbol``/``orbital`` from ``onsite_energy_database``.

    Falls back, in order, to a starred ``l*`` entry (for starred orbitals) and
    then to the highest principal-quantum-number entry of the same angular
    momentum.  ``missing`` is returned when nothing matches.
    """
    db = onsite_energy_database.get(str(symbol), {})
    orbital = str(orbital)
    if orbital in db:
        return float(db[orbital])

    letters = re.findall(r"[A-Za-z]", orbital)
    if not letters:
        return float(missing)
    angular = letters[-1].lower()
    if "*" in orbital:
        starred = f"{angular}*"
        if starred in db:
            return float(db[starred])

    candidates = []
    for key, value in db.items():
        if "*" in key:
            continue
        match = re.fullmatch(r"(\d+)([A-Za-z])", str(key))
        if match is not None and match.group(2).lower() == angular:
            candidates.append((int(match.group(1)), float(value)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return float(missing)


def basis_onsite_table(
    idp: Any,
    *,
    device: Any,
    dtype: torch.dtype,
    scale: float = 1.0,
    missing: float = 0.0,
) -> Optional[torch.Tensor]:
    """Build the ``[num_types, raw_dim]`` diagonal basis-onsite table.

    The onsite energies are written onto the diagonal of each
    ``full_orbital``-``full_orbital`` orbpair block in the raw orbpair
    convention.  Returns ``None`` when ``idp`` lacks the required
    ``basis`` / ``basis_to_full_basis`` / ``orbpair_maps`` attributes (the raw,
    layout-agnostic table; callers that need a compressed/SOC layout project it
    afterwards).
    """
    basis = getattr(idp, "basis", None)
    type_names = getattr(idp, "type_names", None)
    chemical_symbol_to_type = getattr(idp, "chemical_symbol_to_type", None)
    basis_to_full_basis = getattr(idp, "basis_to_full_basis", None)
    orbpair_maps = getattr(idp, "orbpair_maps", None)
    if callable(getattr(idp, "get_orbpair_maps", None)) and orbpair_maps is None:
        orbpair_maps = idp.get_orbpair_maps()
    if (
        not isinstance(basis, dict)
        or not isinstance(basis_to_full_basis, dict)
        or not isinstance(orbpair_maps, dict)
    ):
        return None
    if chemical_symbol_to_type is None:
        if type_names is None:
            return None
        chemical_symbol_to_type = {str(symbol): idx for idx, symbol in enumerate(type_names)}

    raw_dim = int(getattr(idp, "reduced_matrix_element", 0))
    for slc in orbpair_maps.values():
        raw_dim = max(raw_dim, int(getattr(slc, "stop", 0)))
    num_types = 0
    for type_idx in chemical_symbol_to_type.values():
        num_types = max(num_types, int(type_idx) + 1)
    table = torch.zeros(
        num_types,
        raw_dim,
        device=device,
        dtype=dtype,
    )
    for symbol, type_idx in chemical_symbol_to_type.items():
        orbitals = basis.get(symbol, ())
        full_map = basis_to_full_basis.get(symbol, {})
        if not isinstance(full_map, dict):
            continue
        for orbital in orbitals:
            full_orbital = full_map.get(orbital)
            if full_orbital is None:
                continue
            block = orbpair_maps.get(f"{full_orbital}-{full_orbital}")
            if block is None:
                continue
            width = 2 * orbital_l(full_orbital) + 1
            diag = torch.arange(width, device=device, dtype=torch.long)
            diag = int(block.start) + diag * width + diag
            diag = diag[diag < int(block.stop)]
            if diag.numel() == 0:
                continue
            energy = basis_onsite_energy(str(symbol), str(orbital), missing=missing)
            table[int(type_idx), diag] = float(scale) * energy
    return table


def basis_onsite_type_mean(table: torch.Tensor, fallback: float = 0.0) -> torch.Tensor:
    """Per-type mean over the non-zero (active) onsite levels of ``table``.

    Types with no active level get ``fallback``.
    """
    active = table.abs() > 0
    count = active.sum(dim=-1).clamp_min(1)
    mean = (table * active.to(dtype=table.dtype)).sum(dim=-1) / count.to(dtype=table.dtype)
    fallback_t = torch.full_like(mean, float(fallback))
    return torch.where(active.any(dim=-1), mean, fallback_t)


def huckel_pair_energy_table(
    idp: Any,
    *,
    device: Any,
    dtype: torch.dtype,
    missing: float = 0.0,
) -> Optional[torch.Tensor]:
    """Per-bond-type, per-orbital-pair Wolfsberg-Helmholz endpoint energy table.

    Returns ``[num_bond_types, raw_dim]`` where the columns of orbpair block
    ``fo1-fo2`` of bond type ``"Src-Dst"`` hold ``0.5*(eps_Src(fo1)+eps_Dst(fo2))``
    -- the orbital-resolved analogue of :func:`huckel_edge_energy`'s per-type
    mean (classic extended-Hueckel uses the orbital energies; the per-type mean
    collapses every radial/angular channel of an edge onto one scalar).  Blocks
    whose orbitals are absent from the respective species' basis stay 0 (those
    columns are masked by ``mask_to_erme`` downstream).  Layout-agnostic raw
    orbpair convention, like :func:`basis_onsite_table`; callers project to
    compressed/SOC layouts.  Returns ``None`` when ``idp`` lacks the required
    attributes.
    """
    basis = getattr(idp, "basis", None)
    bond_to_type = getattr(idp, "bond_to_type", None)
    basis_to_full_basis = getattr(idp, "basis_to_full_basis", None)
    orbpair_maps = getattr(idp, "orbpair_maps", None)
    if callable(getattr(idp, "get_orbpair_maps", None)) and orbpair_maps is None:
        orbpair_maps = idp.get_orbpair_maps()
    if (
        not isinstance(basis, dict)
        or not isinstance(bond_to_type, dict)
        or not isinstance(basis_to_full_basis, dict)
        or not isinstance(orbpair_maps, dict)
    ):
        return None

    raw_dim = int(getattr(idp, "reduced_matrix_element", 0))
    for slc in orbpair_maps.values():
        raw_dim = max(raw_dim, int(getattr(slc, "stop", 0)))
    num_bond = 0
    for btype in bond_to_type.values():
        num_bond = max(num_bond, int(btype) + 1)
    if num_bond == 0 or raw_dim == 0:
        return None
    table = torch.zeros(num_bond, raw_dim, device=device, dtype=dtype)
    full2orb = {
        sym: {fo: orb for orb, fo in (basis_to_full_basis.get(sym) or {}).items()}
        for sym in basis
    }
    for bond, btype in bond_to_type.items():
        parts = str(bond).split("-")
        if len(parts) != 2:
            continue
        sym_i, sym_j = parts
        fi = full2orb.get(sym_i, {})
        fj = full2orb.get(sym_j, {})
        for name, slc in orbpair_maps.items():
            pair = str(name).split("-")
            if len(pair) != 2:
                continue
            orb_i = fi.get(pair[0])
            orb_j = fj.get(pair[1])
            if orb_i is None or orb_j is None:
                continue
            energy = 0.5 * (
                basis_onsite_energy(sym_i, orb_i, missing=missing)
                + basis_onsite_energy(sym_j, orb_j, missing=missing)
            )
            table[int(btype), int(slc.start):int(slc.stop)] = energy
    return table


def huckel_edge_energy(
    type_mean: torch.Tensor,
    edge_index: torch.Tensor,
    atom_types: torch.Tensor,
    n_edge: int,
    *,
    fallback: float,
    strict: bool,
) -> torch.Tensor:
    """Per-edge Wolfsberg-Helmholz endpoint energy ``0.5*(mean_src+mean_dst)``.

    ``type_mean`` is the per-type mean onsite level (already on the desired
    device/dtype); ``edge_index`` is ``[2, n_cols]`` and ``atom_types`` maps atom
    rows to types.  Returns a 1-D ``[n_edge]`` tensor (the fallback value fills
    edges whose endpoint type is out of range).

    Under ``strict=True`` a mismatch between the edge-index column count and
    ``n_edge``, or an endpoint atom index outside ``atom_types``, raises instead
    of being silently padded/truncated/clamped -- that misalignment is exactly
    the bug class the strict overlap-Huckel prior is meant to surface.
    """
    device = type_mean.device
    dtype = type_mean.dtype
    count = int(n_edge)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    atom_types = atom_types.to(device=device, dtype=torch.long).reshape(-1)
    n_atoms = int(atom_types.numel())
    src = edge_index[0].reshape(-1)
    dst = edge_index[1].reshape(-1)
    n_cols = int(src.numel())
    fallback_energy = torch.full((count,), float(fallback), device=device, dtype=dtype)
    if n_cols != count:
        if strict:
            raise ValueError(
                "flow_options.prior='overlap_huckel' expects one edge_index column "
                f"per edge overlap row, but edge_index has {n_cols} columns while the "
                f"edge overlap/feature tensor has {count} rows. Refusing to pad or "
                "truncate under huckel_strict_basis=True; align edge_index with the "
                "edge overlap rows or set huckel_strict_basis=False for the legacy path."
            )
        if n_cols < count:
            pad = src.new_zeros(count - n_cols)
            src = torch.cat([src, pad], dim=0)
            dst = torch.cat([dst, pad], dim=0)
        src = src[:count]
        dst = dst[:count]
    if strict:
        in_range = (src >= 0) & (src < n_atoms) & (dst >= 0) & (dst < n_atoms)
        if not bool(in_range.all().item()):
            raise ValueError(
                "flow_options.prior='overlap_huckel' edge_index references endpoint "
                f"atom rows outside atom_types (natoms={n_atoms}). Refusing to clamp "
                "under huckel_strict_basis=True; check edge_index/atom_types alignment "
                "or set huckel_strict_basis=False for the legacy path."
            )
    else:
        max_idx = max(n_atoms - 1, 0)
        src = src.clamp(min=0, max=max_idx)
        dst = dst.clamp(min=0, max=max_idx)
    src_type = atom_types.index_select(0, src)
    dst_type = atom_types.index_select(0, dst)
    valid = (
        (src_type >= 0)
        & (src_type < type_mean.shape[0])
        & (dst_type >= 0)
        & (dst_type < type_mean.shape[0])
    )
    energy = fallback_energy.clone()
    if valid.any():
        rows = torch.arange(count, device=device, dtype=torch.long)[valid]
        energy[rows] = 0.5 * (
            type_mean.index_select(0, src_type[valid])
            + type_mean.index_select(0, dst_type[valid])
        )
    return energy
