from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn

from dptb.nn.so2_moe_persistent_grouped import (
    _PersistentGroupedP1Function as _ScheduledSandwichFunction,
    _mainloop_kind,
    _prepare_route_layout,
    _wigner_tensor_and_mode,
)
from dptb.nn.so2_sandwich_common import so2_m0_output, so2_pair_maps

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _parse_tile(value: str) -> Optional[tuple[int, int]]:
    value = value.lower().replace(" ", "")
    for sep in ("x", ",", ":"):
        if sep in value:
            left, right = value.split(sep, 1)
            try:
                return int(left), int(right)
            except ValueError:
                return None
    return None


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.long, device=device)


def _scheduled_metadata(module, device: torch.device):
    cache = getattr(module, "_so2_scheduled_sandwich_metadata_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_so2_scheduled_sandwich_metadata_cache", cache)
    key = str(device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    m_values: list[int] = []
    in_ptr = [0]
    out_ptr = [0]
    in_base_parts = []
    in_l_parts = []
    out_base_parts = []
    out_l_parts = []
    offsets = None

    for m, m_module in zip(range(1, module.irreps_out.lmax + 1), module.m_linear):
        fc = getattr(m_module, "fc", None)
        if not isinstance(fc, nn.Linear):
            _warn_once("non_linear_m_fallback", "scheduled_sandwich requires nn.Linear m>0 blocks; falling back.")
            return None
        in_base, in_l, out_base, out_l, offsets_m = so2_pair_maps(
            module,
            m,
            device,
            cache_attr="_so2_scheduled_sandwich_pair_maps_cache",
        )
        cin = int(in_base.numel())
        cout = int(out_base.numel())
        if cin == 0 or cout == 0:
            continue
        if fc.in_features != cin or fc.out_features != 2 * cout:
            _warn_once("shape_mismatch_fallback", "scheduled_sandwich m block shapes do not match SO2 maps; falling back.")
            return None
        m_values.append(int(m))
        in_base_parts.append(in_base)
        in_l_parts.append(in_l)
        out_base_parts.append(out_base)
        out_l_parts.append(out_l)
        in_ptr.append(in_ptr[-1] + cin)
        out_ptr.append(out_ptr[-1] + cout)
        offsets = offsets_m

    if not m_values:
        return None

    cat = lambda xs: torch.cat(xs, dim=0).contiguous() if xs else _empty_long(device)
    cached = (
        torch.tensor(m_values, dtype=torch.long, device=device).contiguous(),
        torch.tensor(in_ptr, dtype=torch.long, device=device).contiguous(),
        cat(in_base_parts),
        cat(in_l_parts),
        torch.tensor(out_ptr, dtype=torch.long, device=device).contiguous(),
        cat(out_base_parts),
        cat(out_l_parts),
        offsets if offsets is not None else _empty_long(device),
    )
    cache[key] = cached
    return cached


def _scheduled_weights(module, m_values: torch.Tensor, out_ptr: torch.Tensor, in_ptr: torch.Tensor):
    m_list = [int(v) for v in m_values.detach().cpu().tolist()]
    weight_parts = []
    weight_offsets = []
    cursor = 0
    for m_idx, m in enumerate(m_list):
        fc = module.m_linear[m - 1].fc
        weight = fc.weight.unsqueeze(0).contiguous()
        cin = int((in_ptr[m_idx + 1] - in_ptr[m_idx]).item())
        cout = int((out_ptr[m_idx + 1] - out_ptr[m_idx]).item())
        if tuple(weight.shape) != (1, 2 * cout, cin):
            _warn_once("weight_shape_fallback", "scheduled_sandwich weight shape does not match descriptor; falling back.")
            return None
        weight_offsets.append(cursor)
        flat = weight.reshape(-1).contiguous()
        weight_parts.append(flat)
        cursor += int(flat.numel())

    weight_flat = torch.cat(weight_parts, dim=0).contiguous() if weight_parts else torch.empty((0,), dtype=torch.float32, device=out_ptr.device)
    return (
        weight_flat,
        torch.tensor(weight_offsets, dtype=torch.long, device=out_ptr.device).contiguous(),
        torch.empty((0,), dtype=torch.float32, device=out_ptr.device),
        torch.full((len(m_list),), -1, dtype=torch.long, device=out_ptr.device).contiguous(),
        1,
    )


def _tile_shape(mainloop_kind: int) -> tuple[int, int]:
    if int(mainloop_kind) == 3:
        spec = os.environ.get("DPTB_SO2_SCHEDULED_SANDWICH_TILE", "64x32")
        parsed = _parse_tile(spec)
        if parsed != (64, 32):
            if parsed is not None:
                _warn_once("cutlass_tile_fallback", f"scheduled_sandwich cutlass_native currently uses 64x32, got {spec!r}.")
            return 64, 32
        return parsed
    return (
        max(1, _int_env("DPTB_SO2_SCHEDULED_SANDWICH_BLOCK_M", 8)),
        max(1, _int_env("DPTB_SO2_SCHEDULED_SANDWICH_BLOCK_N", 8)),
    )


def try_forward_so2_scheduled_sandwich(module, x: torch.Tensor, weights, wigner_D_all):
    strict = os.environ.get("DPTB_SO2_SCHEDULED_SANDWICH_STRICT", "0").lower() in ("1", "true", "yes", "on")
    try:
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return None
        if module.irreps_out.lmax < 1:
            return None

        wigner_info = _wigner_tensor_and_mode(module, wigner_D_all, x)
        if wigner_info is None:
            return None
        wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

        metadata = _scheduled_metadata(module, x.device)
        if metadata is None:
            return None
        m_values, in_ptr, in_base, in_l, out_ptr, out_base, out_l, offsets = metadata

        packed_weights = _scheduled_weights(module, m_values, out_ptr, in_ptr)
        if packed_weights is None:
            return None
        weight_flat, weight_offsets, bias_flat, bias_offsets, n_routes = packed_weights

        mainloop_name = os.environ.get("DPTB_SO2_SCHEDULED_SANDWICH_MAINLOOP", "warp_collective")
        mainloop_kind = _mainloop_kind(mainloop_name)
        block_m, block_n = _tile_shape(mainloop_kind)
        active_blocks = _int_env("DPTB_SO2_SCHEDULED_SANDWICH_ACTIVE_BLOCKS", 0)

        graph_index = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        edge_order, route_ptr, problem_tile_prefix = _prepare_route_layout(
            graph_index,
            int(n_routes),
            int(m_values.numel()),
            int(block_m),
            int(block_n),
            out_ptr,
            raw_pair_tiles=(int(mainloop_kind) == 3),
        )

        radial_all = weights.contiguous() if module.radial_emb else x.new_empty((0,))
        m_in_index = (
            torch.as_tensor(module.m_in_index, dtype=torch.long, device=x.device).contiguous()
            if module.radial_emb
            else _empty_long(x.device)
        )

        out = _ScheduledSandwichFunction.apply(
            x.contiguous(),
            wigner,
            edge_order,
            route_ptr,
            problem_tile_prefix,
            graph_index,
            weight_flat,
            weight_offsets,
            bias_flat,
            bias_offsets,
            m_values,
            in_ptr,
            in_base,
            in_l,
            out_ptr,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            radial_all,
            m_in_index,
            int(module.irreps_out.dim),
            int(n_routes),
            bool(module.rotate_in),
            bool(module.rotate_out),
            bool(module.front),
            int(wigner_mode),
            int(wigner_stride),
            int(mainloop_kind),
            int(block_m),
            int(block_n),
            int(active_blocks),
        )

        return (out + so2_m0_output(module, x, weights, wigner_D_all)).contiguous(), wigner_D_all
    except Exception:
        if strict:
            raise
        return None
