from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SegmentLayout:
    order: Optional[torch.Tensor]
    unorder: Optional[torch.Tensor]
    sorted_index: torch.Tensor
    ptr_cpu: torch.Tensor

    def as_legacy_tuple(self):
        return self.order, self.unorder, self.sorted_index, self.ptr_cpu


_LAYOUT_CACHES: dict[str, OrderedDict[tuple, SegmentLayout]] = {}


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False", "FALSE", "no", "No", "NO")


def _cache_for(name: str) -> OrderedDict[tuple, SegmentLayout]:
    cache = _LAYOUT_CACHES.get(name)
    if cache is None:
        cache = OrderedDict()
        _LAYOUT_CACHES[name] = cache
    return cache


def repeated_segment_layout(
    graph_index: torch.Tensor,
    num_routes: int,
    *,
    repeat: int,
    assume_sorted: bool = False,
    cache_name: str = "default",
    max_entries: int = 64,
) -> SegmentLayout:
    """Return permutation and CPU ptr for repeated route-index segments."""

    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    graph_index = graph_index.reshape(-1).to(dtype=torch.long)
    fresh_layout = _truthy_env("DPTB_SO2_DISABLE_ROUTE_LAYOUT_CACHE", "0") or (
        (not torch.is_grad_enabled()) and _truthy_env("DPTB_SO2_EVAL_FRESH_ROUTE_LAYOUT", "1")
    )
    key = None
    cache = None
    if not fresh_layout:
        key = (
            int(repeat),
            graph_index.device.type,
            graph_index.device.index,
            int(graph_index.data_ptr()),
            int(graph_index.numel()),
            int(getattr(graph_index, "_version", 0)),
            int(num_routes),
            bool(assume_sorted),
        )
        cache = _cache_for(cache_name)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    if repeat == 1:
        flat_graph = graph_index.contiguous()
    else:
        flat_graph = graph_index.reshape(-1, 1).expand(-1, int(repeat)).reshape(-1).contiguous()

    if assume_sorted or flat_graph.numel() <= 1:
        order = None
        unorder = None
        sorted_graph = flat_graph
    elif torch.all(flat_graph[1:] >= flat_graph[:-1]).item():
        order = None
        unorder = None
        sorted_graph = flat_graph
    else:
        order = torch.argsort(flat_graph, stable=True)
        sorted_graph = flat_graph.index_select(0, order)
        unorder = torch.empty_like(order)
        unorder.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))

    counts = torch.bincount(sorted_graph, minlength=int(num_routes))
    ptr = torch.zeros(int(num_routes) + 1, dtype=torch.long, device=counts.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    layout = SegmentLayout(
        order=order,
        unorder=unorder,
        sorted_index=sorted_graph,
        ptr_cpu=ptr.to(device="cpu", dtype=torch.long).contiguous(),
    )
    if cache is not None and key is not None:
        cache[key] = layout
        while len(cache) > int(max_entries):
            cache.popitem(last=False)
    return layout


def row_segment_layout(
    graph_index: torch.Tensor,
    num_routes: int,
    *,
    assume_sorted: bool = False,
    cache_name: str = "default",
) -> SegmentLayout:
    return repeated_segment_layout(
        graph_index,
        num_routes,
        repeat=1,
        assume_sorted=assume_sorted,
        cache_name=cache_name,
    )
