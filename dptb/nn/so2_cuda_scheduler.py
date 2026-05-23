from __future__ import annotations

from collections import OrderedDict

import torch

from dptb.nn.so2_moe_persistent_grouped import (
    _PersistentGroupedP1Function as _HistoricalPersistentGroupedP1Function,
    _load_extension,
    _mainloop_kind,
    _prepare_route_layout,
    _wigner_tensor_and_mode,
)


class SO2CudaSchedulerFunction(_HistoricalPersistentGroupedP1Function):
    """Neutral SO2 scheduler facade over the historical persistent kernel.

    The implementation still reuses the historical autograd body, but non-MoE
    callers depend on this module so the public Python path is about SO2
    schedule descriptors instead of MoE route plumbing.
    """


def load_scheduler_extension():
    return _load_extension()


_SINGLE_ROUTE_LAYOUT_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_SINGLE_ROUTE_LAYOUT_CACHE_MAX = 64


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


def prepare_so2_single_route_layout(
    *,
    num_rows: int,
    n_problems: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    raw_pair_tiles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        str(out_ptr.device),
        int(num_rows),
        int(n_problems),
        int(block_m),
        int(block_n),
        int(out_ptr.data_ptr()),
        int(out_ptr.numel()),
        int(getattr(out_ptr, "_version", 0)),
        bool(raw_pair_tiles),
    )
    cached = _SINGLE_ROUTE_LAYOUT_CACHE.get(key)
    if cached is not None:
        _SINGLE_ROUTE_LAYOUT_CACHE.move_to_end(key)
        return cached

    ext = load_scheduler_extension()
    edge_order, route_ptr, problem_tile_prefix = ext.so2_single_route_layout(
        int(num_rows),
        int(n_problems),
        int(block_m),
        int(block_n),
        out_ptr,
        bool(raw_pair_tiles),
    )
    cached = edge_order.contiguous(), route_ptr.contiguous(), problem_tile_prefix.contiguous()
    _SINGLE_ROUTE_LAYOUT_CACHE[key] = cached
    while len(_SINGLE_ROUTE_LAYOUT_CACHE) > _SINGLE_ROUTE_LAYOUT_CACHE_MAX:
        _SINGLE_ROUTE_LAYOUT_CACHE.popitem(last=False)
    return cached


def wigner_tensor_and_mode(module, wigner_D_all, x: torch.Tensor):
    return _wigner_tensor_and_mode(module, wigner_D_all, x)
