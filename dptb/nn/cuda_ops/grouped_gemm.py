from __future__ import annotations

from typing import Optional

import torch


def grouped_gemm(
    x: torch.Tensor,
    ptr: torch.Tensor,
    weight: torch.Tensor,
    *,
    fast_tf32: Optional[bool] = None,
) -> torch.Tensor:
    from dptb.nn.cublas_grouped_gemm import grouped_gemm as _grouped_gemm

    return _grouped_gemm(x, ptr, weight, fast_tf32=fast_tf32)


def grouped_gemm_multi(
    xs: list[torch.Tensor],
    ptrs: list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    from dptb.nn.cublas_grouped_gemm import grouped_gemm_multi as _grouped_gemm_multi

    return _grouped_gemm_multi(xs, ptrs, weights, fast_tf32=fast_tf32)


def indexed_sandwich_multi_gemm(
    pair_inputs: list[torch.Tensor],
    ptrs: torch.Tensor | list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    permute_idx: torch.Tensor | None = None,
    unpermute_idx: torch.Tensor | None = None,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    """Shared middle GEMM for indexed_sandwich_multi-style SO2 paths."""
    ptr_list = [ptrs] * len(pair_inputs) if isinstance(ptrs, torch.Tensor) else list(ptrs)
    flat_inputs = []
    for pair in pair_inputs:
        flat = pair.reshape(-1, pair.shape[-1])
        if permute_idx is not None:
            flat = flat.index_select(0, permute_idx)
        flat_inputs.append(flat.contiguous())

    flat_outputs = grouped_gemm_multi(flat_inputs, ptr_list, weights, fast_tf32=fast_tf32)
    outputs = []
    for flat_out, pair in zip(flat_outputs, pair_inputs):
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        outputs.append(flat_out.reshape(*pair.shape[:-1], flat_out.shape[-1]).contiguous())
    return outputs
