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
    contract.  Onsite blocks are symmetrized explicitly; directed edge blocks
    are left directed.
    """

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
