from __future__ import annotations

import torch

from dptb.nn.so2_moe_persistent_grouped import (
    _PersistentGroupedP1Function,
    _mainloop_kind,
    _prepare_route_layout,
    _wigner_tensor_and_mode,
)


SO2CudaSchedulerFunction = _PersistentGroupedP1Function
"""Neutral SO2 scheduler façade over the historical persistent kernel.

The C++ symbol names still carry the persistent/P1 lineage, but non-MoE SO2
callers should depend on this module so the public Python path is about
schedule descriptors, not MoE route plumbing.
"""


def mainloop_kind(name: str | None = None) -> int:
    return _mainloop_kind(name)


def prepare_single_route_layout(
    graph_index: torch.Tensor,
    *,
    n_routes: int,
    n_problems: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    raw_pair_tiles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _prepare_route_layout(
        graph_index,
        int(n_routes),
        int(n_problems),
        int(block_m),
        int(block_n),
        out_ptr,
        raw_pair_tiles=bool(raw_pair_tiles),
    )


def wigner_tensor_and_mode(module, wigner_D_all, x: torch.Tensor):
    return _wigner_tensor_and_mode(module, wigner_D_all, x)
