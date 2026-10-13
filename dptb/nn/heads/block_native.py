"""Experimental block-native output heads for DeePTB Plan B.

These heads write explicit AO-block tensors instead of DeePTB RME feature
vectors.  They intentionally do not call Wigner/CG routines and must not be
fed into E3Hamiltonian.
"""

from __future__ import annotations

from typing import Union

import torch
from e3nn import o3


class BlockNativeLinearHead(torch.nn.Module):
    """Project final equivariant features directly to padded AO blocks.

    This is a minimal block-native decoder used to validate the Plan-B routing
    contract.  It is intentionally a debug/control baseline: the dense linear
    projection is not an SO(3)/O(3) AO decoder.  Onsite blocks are symmetrized
    explicitly; directed edge blocks are left directed.
    """

    performs_angular_coupling = False
    output_contract = "ao_block"
    bypasses_rme = True
    bypasses_e3hamiltonian = True
    uses_ict = False
    is_equivariant_output = False
    debug_only = True

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        max_norb: int,
        *,
        symmetrize: bool,
        init: float = 0.0,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.max_norb = int(max_norb)
        self.symmetrize = bool(symmetrize)
        self.output_shape = (self.max_norb, self.max_norb)

        if self.max_norb <= 0:
            raise ValueError(f"max_norb must be positive, got {max_norb}.")
        if init < 0:
            raise ValueError(f"init must be non-negative, got {init}.")

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        self.proj = torch.nn.Linear(
            self.irreps_in.dim,
            self.max_norb * self.max_norb,
            bias=True,
            **factory_kwargs,
        )
        self.reset_parameters(float(init))

    def reset_parameters(self, init: float) -> None:
        if init == 0.0:
            torch.nn.init.zeros_(self.proj.weight)
        else:
            torch.nn.init.normal_(self.proj.weight, mean=0.0, std=init)
        if self.proj.bias is not None:
            torch.nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected last dimension {self.irreps_in.dim}, got "
                f"{features.shape[-1]}."
            )
        blocks = self.proj(features).reshape(*features.shape[:-1], *self.output_shape)
        if self.symmetrize:
            blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
        return blocks


def apply_ao_basis_mask(
    blocks: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero padded AO rows/columns outside each atom/bond basis."""
    if col_mask is None:
        col_mask = row_mask
    mask = row_mask.to(dtype=torch.bool).unsqueeze(-1) & col_mask.to(dtype=torch.bool).unsqueeze(-2)
    return blocks * mask.to(dtype=blocks.dtype, device=blocks.device)


def species_compact_index(mask_to_basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Union-slot -> species-contiguous gather maps for padded AO canvases.

    Block heads lay out ``[max_norb, max_norb]`` canvases by full/union-basis
    shell offsets, while blockwise targets, H0 blocks, and the blockwise loss
    masks pack each species' orbitals contiguously from the top-left (the
    ``orbital_maps``/``feature_tensors_to_block_tensors`` convention).  The two
    layouts only coincide when a species' basis is a prefix of the union basis;
    any skipped shell (e.g. H ``2s1p`` inside an O ``3s2p1d`` union) misaligns
    every affected row/column.

    Returns ``(index, norb)`` where ``index[s, :norb[s]]`` lists species ``s``'s
    union-slot positions in ascending order (tail padded with 0) and ``norb[s]``
    is its orbital count.
    """
    mask = mask_to_basis.to(dtype=torch.bool)
    n_species, max_norb = mask.shape
    index = torch.zeros((n_species, max_norb), dtype=torch.long, device=mask.device)
    norb = mask.sum(dim=-1).to(dtype=torch.long)
    for s in range(n_species):
        pos = torch.nonzero(mask[s], as_tuple=False).flatten()
        index[s, : pos.numel()] = pos
    return index, norb


def compact_blocks_to_species_layout(
    blocks: torch.Tensor,
    row_index: torch.Tensor,
    row_norb: torch.Tensor,
    col_index: torch.Tensor | None = None,
    col_norb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather union-slot AO blocks into the species-contiguous data layout.

    ``blocks`` is ``[N, max_norb, max_norb]`` in union full-basis slot layout;
    ``row_index``/``col_index`` are per-block ``[N, max_norb]`` gather positions
    and ``row_norb``/``col_norb`` the per-block valid counts (both typically
    ``species_compact_index`` outputs indexed by atom type).  Entries beyond the
    valid ``(row_norb, col_norb)`` rectangle are zeroed so the result matches
    ``block_mask_from_shapes`` semantics exactly.
    """
    if col_index is None:
        col_index = row_index
    if col_norb is None:
        col_norb = row_norb
    max_norb = blocks.shape[-1]
    rows = blocks.gather(-2, row_index.unsqueeze(-1).expand(-1, -1, max_norb))
    compact = rows.gather(-1, col_index.unsqueeze(-2).expand(-1, max_norb, -1))
    ar = torch.arange(max_norb, device=blocks.device)
    valid = (ar.view(1, -1, 1) < row_norb.view(-1, 1, 1)) & (
        ar.view(1, 1, -1) < col_norb.view(-1, 1, 1)
    )
    return compact * valid.to(dtype=blocks.dtype)
