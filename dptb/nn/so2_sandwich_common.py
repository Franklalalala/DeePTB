from __future__ import annotations

from typing import Any

import torch


def _entry_values(entry: Any) -> tuple[int, int, slice]:
    if hasattr(entry, "l"):
        return int(entry.l), int(entry.mul), entry.slice_info
    l, mul, slice_info, _group_start = entry
    return int(l), int(mul), slice_info


def so2_pair_maps(module, m: int, device: torch.device, *, cache_attr: str = "_so2_sandwich_pair_maps"):
    cache = getattr(module, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(module, cache_attr, cache)

    key = (int(m), str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached

    in_base = []
    in_l = []
    for entry in module._in_entries_by_m[m]:
        l, mul, slice_info = _entry_values(entry)
        dim = 2 * l + 1
        start = int(slice_info.start)
        for idx in range(mul):
            in_base.append(start + idx * dim)
            in_l.append(l)

    out_base = []
    out_l = []
    for entry in module._out_entries_by_m[m]:
        l, mul, slice_info = _entry_values(entry)
        dim = 2 * l + 1
        start = int(slice_info.start)
        for idx in range(mul):
            out_base.append(start + idx * dim)
            out_l.append(l)

    offsets = [int(module.offsets[l]) for l in range(module.l_max + 1)]
    cached = (
        torch.tensor(in_base, dtype=torch.long, device=device).contiguous(),
        torch.tensor(in_l, dtype=torch.long, device=device).contiguous(),
        torch.tensor(out_base, dtype=torch.long, device=device).contiguous(),
        torch.tensor(out_l, dtype=torch.long, device=device).contiguous(),
        torch.tensor(offsets, dtype=torch.long, device=device).contiguous(),
    )
    cache[key] = cached
    return cached
