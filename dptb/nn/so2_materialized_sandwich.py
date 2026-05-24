from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from dptb.nn.cuda_ops.grouped_gemm import indexed_sandwich_multi_gemm
from dptb.nn.so2_moe_fused_p0 import (
    _PackPairsMultiFunction,
    _ScatterPairOutputFunction,
    _ScatterRawPairOutputFunction,
    _ScatterRawPairsMultiOutputMajorFunction,
    _multi_output_entry_map,
    _wigner_tensor_and_mode,
)
from dptb.nn.so2_sandwich_common import so2_m0_output, so2_pair_maps

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _normalize_strategy(strategy: str) -> str:
    aliases = {
        "cublas_grouped": "grouped",
        "grouped_cublas": "grouped",
        "dense": "block_dense",
        "single_dense": "block_dense",
        "block": "block_dense",
        "persistent": "block_dense",
        "persistent_block": "block_dense",
    }
    return aliases.get(strategy, strategy)


def _strategy() -> str:
    return _normalize_strategy(os.environ.get("DPTB_SO2_MATERIALIZED_GEMM_STRATEGY", "block_dense").lower())


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.long, device=device)


def _materialized_metadata(module, device: torch.device):
    cache = getattr(module, "_so2_materialized_sandwich_metadata_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_so2_materialized_sandwich_metadata_cache", cache)
    key = str(device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    m_values: list[int] = []
    in_bases: list[torch.Tensor] = []
    in_ls: list[torch.Tensor] = []
    out_bases: list[torch.Tensor] = []
    out_ls: list[torch.Tensor] = []
    cin_prefix = [0]
    cout_prefix = [0]
    offsets = None

    for m, m_module in zip(range(1, module.irreps_out.lmax + 1), module.m_linear):
        fc = getattr(m_module, "fc", None)
        if not isinstance(fc, nn.Linear):
            _warn_once("non_linear_m_fallback", "materialized_sandwich requires nn.Linear m>0 blocks; falling back.")
            return None
        in_base, in_l, out_base, out_l, offsets_m = so2_pair_maps(
            module,
            m,
            device,
            cache_attr="_so2_materialized_sandwich_pair_maps_cache",
        )
        cin = int(in_base.numel())
        cout = int(out_base.numel())
        if cin == 0 or cout == 0:
            continue
        if fc.in_features != cin or fc.out_features != 2 * cout:
            _warn_once("shape_mismatch_fallback", "materialized_sandwich m block shapes do not match SO2 maps; falling back.")
            return None
        m_values.append(int(m))
        in_bases.append(in_base)
        in_ls.append(in_l)
        out_bases.append(out_base)
        out_ls.append(out_l)
        cin_prefix.append(cin_prefix[-1] + cin)
        cout_prefix.append(cout_prefix[-1] + cout)
        offsets = offsets_m

    if not m_values:
        return None

    cached = (
        m_values,
        in_bases,
        in_ls,
        out_bases,
        out_ls,
        torch.tensor(cin_prefix, dtype=torch.long, device=device).contiguous(),
        torch.tensor(cout_prefix, dtype=torch.long, device=device).contiguous(),
        torch.tensor(m_values, dtype=torch.long, device=device).contiguous(),
        offsets if offsets is not None else _empty_long(device),
        tuple(int(v) for v in cin_prefix),
        tuple(int(v) for v in cout_prefix),
    )
    cache[key] = cached
    return cached


def _apply_radial_front(module, packed_all: torch.Tensor, weights, m_values: list[int], cin_prefix_host: tuple[int, ...]) -> torch.Tensor:
    if not module.radial_emb or not bool(module.front):
        return packed_all
    pieces = []
    for i, m in enumerate(m_values):
        start = cin_prefix_host[i]
        end = cin_prefix_host[i + 1]
        radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1)
        pieces.append(packed_all[:, :, start:end] * radial)
    return torch.cat(pieces, dim=2).contiguous() if pieces else packed_all


def _single_route_pair_ptr(module, n_rows: int) -> torch.Tensor:
    cache = getattr(module, "_so2_materialized_single_route_pair_ptr_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_so2_materialized_single_route_pair_ptr_cache", cache)
    cached = cache.get(int(n_rows))
    if cached is None:
        cached = torch.tensor([0, int(n_rows) * 2], dtype=torch.long, device="cpu")
        cache[int(n_rows)] = cached
    return cached


def _grouped_raw_outputs(module, packed_eff: torch.Tensor, m_values: list[int], cin_prefix_host: tuple[int, ...]):
    pair_inputs = []
    ptr = _single_route_pair_ptr(module, int(packed_eff.shape[0]))
    ptrs = []
    weights = []
    for i, m in enumerate(m_values):
        start = cin_prefix_host[i]
        end = cin_prefix_host[i + 1]
        pair = packed_eff[:, :, start:end]
        pair_inputs.append(pair.contiguous())
        ptrs.append(ptr)
        weights.append(module.m_linear[m - 1].fc.weight.unsqueeze(0).contiguous())
    return indexed_sandwich_multi_gemm(pair_inputs, ptrs, weights)


def _block_dense_raw_outputs(
    module,
    packed_eff: torch.Tensor,
    m_values: list[int],
    cin_prefix_host: tuple[int, ...],
    cout_prefix_host: tuple[int, ...],
):
    total_cin = cin_prefix_host[-1]
    weight_rows = []
    for i, m in enumerate(m_values):
        left = cin_prefix_host[i]
        right = cin_prefix_host[i + 1]
        weight = module.m_linear[m - 1].fc.weight
        weight_rows.append(F.pad(weight, (left, total_cin - right)))

    block_weight = torch.cat(weight_rows, dim=0).contiguous()
    raw_all = packed_eff.reshape(-1, total_cin).matmul(block_weight.t())
    raw_all = raw_all.reshape(packed_eff.shape[0], packed_eff.shape[1], -1)

    raw_tensors = []
    for i in range(len(m_values)):
        start = 2 * cout_prefix_host[i]
        end = 2 * cout_prefix_host[i + 1]
        raw_tensors.append(raw_all[:, :, start:end].contiguous())
    return raw_tensors


def _raw_output_major_contribution(
    module,
    raw_tensors,
    m_values: list[int],
    out_bases: list[torch.Tensor],
    out_ls: list[torch.Tensor],
    wigner: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    cout_prefix: torch.Tensor,
    m_values_t: torch.Tensor,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    entry_offsets, entry_m, entry_channel, entry_d, entry_l = _multi_output_entry_map(
        module,
        m_values,
        out_bases,
        out_ls,
        int(module.irreps_out.dim),
        wigner.device,
    )
    return _ScatterRawPairsMultiOutputMajorFunction.apply(
        wigner,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values_t,
        entry_offsets,
        entry_m,
        entry_channel,
        entry_d,
        entry_l,
        int(module.irreps_out.dim),
        bool(module.rotate_out),
        int(wigner_mode),
        int(wigner_stride),
        len(raw_tensors),
        *raw_tensors,
        *out_bases,
        *out_ls,
    )


def try_forward_so2_materialized_sandwich(
    module,
    x: torch.Tensor,
    weights,
    wigner_D_all,
    *,
    strategy_override: str | None = None,
    strict_override: bool | None = None,
):
    strict = (
        bool(strict_override)
        if strict_override is not None
        else os.environ.get("DPTB_SO2_MATERIALIZED_STRICT", "0").lower() in ("1", "true", "yes", "on")
    )
    try:
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return None
        if module.irreps_out.lmax < 1:
            return None

        wigner_info = _wigner_tensor_and_mode(module, wigner_D_all, x)
        if wigner_info is None:
            return None
        wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

        metadata = _materialized_metadata(module, x.device)
        if metadata is None:
            return None
        (
            m_values,
            in_bases,
            in_ls,
            out_bases,
            out_ls,
            cin_prefix,
            cout_prefix,
            m_values_t,
            offsets,
            cin_prefix_host,
            cout_prefix_host,
        ) = metadata

        packed_all = _PackPairsMultiFunction.apply(
            x.contiguous(),
            wigner,
            in_bases,
            in_ls,
            offsets,
            compact_offsets,
            cin_prefix,
            m_values_t,
            bool(module.rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        packed_eff = _apply_radial_front(module, packed_all, weights, m_values, cin_prefix_host)

        strategy = _normalize_strategy(strategy_override.lower()) if strategy_override is not None else _strategy()
        if strategy == "grouped":
            raw_tensors = _grouped_raw_outputs(module, packed_eff, m_values, cin_prefix_host)
        elif strategy == "block_dense":
            raw_tensors = _block_dense_raw_outputs(module, packed_eff, m_values, cin_prefix_host, cout_prefix_host)
        else:
            _warn_once("unknown_strategy_fallback", f"unknown DPTB_SO2_MATERIALIZED_GEMM_STRATEGY={strategy!r}; falling back.")
            return None

        epilogue_schedule = os.environ.get("DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE", "per_m").lower()
        raw_output_epilogue = not (module.radial_emb and not bool(module.front))
        if epilogue_schedule == "output_major" and raw_output_epilogue:
            contribution = _raw_output_major_contribution(
                module,
                raw_tensors,
                m_values,
                out_bases,
                out_ls,
                wigner,
                offsets,
                compact_offsets,
                cout_prefix,
                m_values_t,
                int(wigner_mode),
                int(wigner_stride),
            )
        else:
            contribution = None
            for raw, m, out_base, out_l in zip(raw_tensors, m_values, out_bases, out_ls):
                if module.radial_emb and not bool(module.front):
                    radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1)
                    pair_out = module.m_linear[m - 1]._finish_linear_output(raw) * radial
                    part = _ScatterPairOutputFunction.apply(
                        pair_out.contiguous(),
                        wigner,
                        out_base,
                        out_l,
                        offsets,
                        compact_offsets,
                        int(module.irreps_out.dim),
                        int(m),
                        bool(module.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                    )
                else:
                    part = _ScatterRawPairOutputFunction.apply(
                        raw.contiguous(),
                        wigner,
                        out_base,
                        out_l,
                        offsets,
                        compact_offsets,
                        int(module.irreps_out.dim),
                        int(m),
                        bool(module.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                    )
                contribution = part if contribution is None else contribution + part
        if contribution is None:
            return None

        return (so2_m0_output(module, x, weights, wigner_D_all) + contribution).contiguous(), wigner_D_all
    except Exception:
        if strict:
            raise
        return None
