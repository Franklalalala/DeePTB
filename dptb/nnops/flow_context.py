from __future__ import annotations

"""Shared dataclasses and dtype helper for Hamiltonian flow training.

Extracted from :mod:`dptb.nnops.flow` so the CFM and pixel-meanflow modules
can share the interpolation contexts without importing each other.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch


def _to_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return torch.float32


@dataclass
class CFMContext:
    t: torch.Tensor
    node_t: Optional[torch.Tensor]
    edge_t: Optional[torch.Tensor]
    node_base: Optional[torch.Tensor]
    edge_base: Optional[torch.Tensor]
    node_target: Optional[torch.Tensor]
    edge_target: Optional[torch.Tensor]
    node_current: Optional[torch.Tensor]
    edge_current: Optional[torch.Tensor]
    node_prior: Optional[torch.Tensor]
    edge_prior: Optional[torch.Tensor]
    block_target_semantics: Optional[str] = None


@dataclass
class PixelMFContext:
    r: torch.Tensor
    t: torch.Tensor
    fm_mask: torch.Tensor
    node_r: Optional[torch.Tensor]
    node_t: Optional[torch.Tensor]
    edge_r: Optional[torch.Tensor]
    edge_t: Optional[torch.Tensor]
    node_base: Optional[torch.Tensor]
    edge_base: Optional[torch.Tensor]
    node_clean: Optional[torch.Tensor]
    edge_clean: Optional[torch.Tensor]
    node_state: Optional[torch.Tensor]
    edge_state: Optional[torch.Tensor]
    node_prior: Optional[torch.Tensor]
    edge_prior: Optional[torch.Tensor]
