
import math
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def _force_disable_triton() -> bool:
    return os.environ.get("DPTB_TRITON_LINEAR_DISABLE", "0") == "1"


def _require_triton() -> bool:
    return os.environ.get("DPTB_TRITON_LINEAR_REQUIRE", "0") == "1"


def _use_triton_for_linear(x: torch.Tensor, mixed_weights: torch.Tensor) -> bool:
    if _force_disable_triton():
        return False
    if not _TRITON_AVAILABLE:
        return False
    if x.device.type != "cuda" or mixed_weights.device.type != "cuda":
        return False
    if x.dtype != torch.float32:
        return False
    if mixed_weights.dtype != x.dtype:
        return False
    return True


def _get_num_sms(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


_META_CACHE = {}


def _canonical_split_sizes(split_sizes: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in split_sizes)


def _flatten_grouped_rows(x: torch.Tensor, split_sizes: Sequence[int]):
    split_sizes = _canonical_split_sizes(split_sizes)
    lead = 1
    if x.ndim > 2:
        for dim in x.shape[1:-1]:
            lead *= int(dim)
    rows_per_group = tuple(int(v) * lead for v in split_sizes)
    total_rows = sum(rows_per_group)
    flat_x = x.reshape(total_rows, x.shape[-1]).contiguous()
    return flat_x, rows_per_group, x.shape[:-1], x.shape[-1]


def _meta_tensors(rows_per_group: Sequence[int], device: torch.device):
    rows_per_group = _canonical_split_sizes(rows_per_group)
    key = (str(device), rows_per_group)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    row_sizes = torch.tensor(rows_per_group, device=device, dtype=torch.int32)
    row_offsets = torch.zeros(len(rows_per_group), device=device, dtype=torch.int32)
    if len(rows_per_group) > 1:
        row_offsets[1:] = torch.cumsum(row_sizes, dim=0)[:-1]
    cached = (row_offsets, row_sizes)
    _META_CACHE[key] = cached
    return cached


def _torch_grouped_linear_forward(flat_x: torch.Tensor,
                                  mixed_weights: torch.Tensor,
                                  mixed_bias: Optional[torch.Tensor],
                                  rows_per_group: Sequence[int]) -> torch.Tensor:
    out_parts = []
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xg = flat_x[start:start + rows]
        wg = mixed_weights[g]
        bg = mixed_bias[g] if mixed_bias is not None else None
        out_parts.append(F.linear(xg, wg, bg))
        start += rows
    return torch.cat(out_parts, dim=0)


if _TRITON_AVAILABLE:
    _DEFAULT_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ]

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN", "HAS_BIAS"],
    )
    @triton.jit
    def _grouped_linear_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wg,
        stride_wk,
        stride_wn,
        stride_bg,
        stride_bn,
        stride_ym,
        stride_yn,
        N_OUT,
        K_IN,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        tiles_n = tl.cdiv(N_OUT, BLOCK_N)
        total_tiles = 0
        for g_scan in range(num_groups):
            rows_scan = tl.load(row_sizes_ptr + g_scan)
            total_tiles += tl.cdiv(rows_scan, BLOCK_M) * tiles_n

        tile_id = pid
        while tile_id < total_tiles:
            remaining = tile_id
            selected_g = tl.full((), 0, tl.int32)
            selected_remaining = remaining
            found = tl.full((), False, tl.int1)
            for g_scan in range(num_groups):
                rows_g = tl.load(row_sizes_ptr + g_scan)
                tiles_g = tl.cdiv(rows_g, BLOCK_M) * tiles_n
                take = (remaining < tiles_g) & (~found)
                selected_g = tl.where(take, g_scan, selected_g)
                selected_remaining = tl.where(take, remaining, selected_remaining)
                remaining = tl.where(found | take, remaining, remaining - tiles_g)
                found = found | take

            g = selected_g
            remaining = selected_remaining
            rows = tl.load(row_sizes_ptr + g)
            row_start = tl.load(row_offsets_ptr + g)
            pid_m = remaining // tiles_n
            pid_n = remaining % tiles_n

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_m = offs_m < rows
            mask_n = offs_n < N_OUT

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            k0 = 0
            while k0 < K_IN:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K_IN

                a = tl.load(
                    x_ptr + (row_start + offs_m[:, None]) * stride_xm + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                b = tl.load(
                    w_ptr + g * stride_wg + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                acc += tl.dot(a, b, input_precision="ieee")
                k0 += BLOCK_K

            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + g * stride_bg + offs_n * stride_bn,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                acc += bias[None, :]

            tl.store(
                y_ptr + (row_start + offs_m[:, None]) * stride_ym + offs_n[None, :] * stride_yn,
                acc,
                mask=mask_m[:, None] & mask_n[None, :],
            )
            tile_id += num_programs


def _triton_grouped_linear_forward(flat_x: torch.Tensor,
                                   mixed_weights_t: torch.Tensor,
                                   mixed_bias: Optional[torch.Tensor],
                                   rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _use_triton_for_linear(flat_x, mixed_weights_t):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped linear backend is unavailable.")
        w = mixed_weights_t.transpose(1, 2).contiguous()
        return _torch_grouped_linear_forward(flat_x, w, mixed_bias, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    total_rows = int(sum(int(v) for v in rows_per_group))
    n_out = int(mixed_weights_t.shape[2])

    out = torch.empty((total_rows, n_out), device=flat_x.device, dtype=flat_x.dtype)
    bias = mixed_bias if mixed_bias is not None else torch.empty((1, 1), device=flat_x.device, dtype=flat_x.dtype)

    num_sms = _get_num_sms(flat_x.device)
    cta_factor = int(os.environ.get("DPTB_TRITON_LINEAR_PERSISTENT_FACTOR", "2"))

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_n = meta["BLOCK_N"]
        tiles_n = triton.cdiv(n_out, block_n)
        total_tiles = sum(triton.cdiv(int(v), block_m) * tiles_n for v in rows_per_group)
        return (max(1, min(total_tiles, num_sms * cta_factor)),)

    _grouped_linear_kernel[grid](
        flat_x,
        mixed_weights_t,
        bias,
        out,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        flat_x.stride(0),
        flat_x.stride(1),
        mixed_weights_t.stride(0),
        mixed_weights_t.stride(1),
        mixed_weights_t.stride(2),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        n_out,
        int(flat_x.shape[1]),
        HAS_BIAS=mixed_bias is not None,
    )
    return out


class _GroupedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                mixed_weights: torch.Tensor,
                mixed_bias: Optional[torch.Tensor],
                row_splits_tensor: torch.Tensor):
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        flat_x, rows_per_group, leading_shape, _ = _flatten_grouped_rows(x, split_sizes)

        mixed_weights_t = mixed_weights.transpose(1, 2).contiguous()
        flat_out = _triton_grouped_linear_forward(flat_x, mixed_weights_t, mixed_bias, rows_per_group)

        ctx.save_for_backward(flat_x, mixed_weights, mixed_bias if mixed_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype), row_splits_tensor)
        ctx.orig_shape = tuple(int(v) for v in x.shape)
        ctx.has_bias = mixed_bias is not None
        ctx.leading_shape = tuple(int(v) for v in x.shape[:-1])
        return flat_out.reshape(*x.shape[:-1], mixed_weights.shape[1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        flat_x, mixed_weights, bias_saved, row_splits_tensor = ctx.saved_tensors
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        flat_grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()

        rows_per_group = []
        lead = int(flat_grad.shape[0] // sum(split_sizes)) if sum(split_sizes) > 0 else 1
        for s in split_sizes:
            rows_per_group.append(int(s) * lead)

        # grad_x uses the same grouped kernel with weights interpreted as [group, K=out, N=in]
        flat_grad_x = _triton_grouped_linear_forward(flat_grad, mixed_weights.contiguous(), None, rows_per_group)
        grad_x = flat_grad_x.reshape(ctx.orig_shape)

        grad_w = torch.zeros_like(mixed_weights)
        grad_b = torch.zeros((mixed_weights.shape[0], mixed_weights.shape[1]), device=flat_grad.device, dtype=flat_grad.dtype) if ctx.has_bias else None

        start = 0
        acc_dtype = torch.float32 if flat_grad.dtype in (torch.float16, torch.bfloat16) else flat_grad.dtype
        for g, rows in enumerate(rows_per_group):
            rows = int(rows)
            xg = flat_x[start:start + rows]
            gog = flat_grad[start:start + rows]
            grad_w[g] = gog.to(acc_dtype).transpose(0, 1).matmul(xg.to(acc_dtype)).to(grad_w.dtype)
            if grad_b is not None:
                grad_b[g] = gog.to(acc_dtype).sum(dim=0).to(grad_b.dtype)
            start += rows

        if not ctx.has_bias:
            grad_b_out = None
        else:
            grad_b_out = grad_b

        return grad_x, grad_w, grad_b_out, None


def grouped_linear_apply(x: torch.Tensor,
                         mixed_weights: torch.Tensor,
                         mixed_bias: Optional[torch.Tensor],
                         split_sizes: Sequence[int]) -> torch.Tensor:
    row_splits_tensor = torch.tensor(_canonical_split_sizes(split_sizes), device=x.device, dtype=torch.long)
    return _GroupedLinearFn.apply(x, mixed_weights, mixed_bias, row_splits_tensor)
