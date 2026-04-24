# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Triton graph-persistent exact MoE V3 overlay for DeePTB.

V3 builds on ``so2_triton_exact_gp_v2`` and keeps the same exact graph-mix
semantics, but changes the aggressive backward reduce kernel.  V2 launches one
Triton program per ``(graph, expert, row_chunk, out_tile, in_tile)``.  That is
simple and memory-light, but it reloads the same ``x`` and ``grad_y`` tile for
every expert.  V3 instead launches one program per
``(graph, row_chunk, out_tile, in_tile)`` and loops over experts inside the
program.  This keeps the no-``grad_mixed_w`` property while reducing x/grad_y
traffic, launch tiles, and shared-weight/shared-bias atomics.

The implementation is intentionally opt-in and can fall back to V2 or the exact
Torch backward at runtime.

Environment switches
--------------------
DPTB_TRITON_EXACT_GP_V3=1
    Enable real-valued V3.
DPTB_TRITON_COMPLEX_EXACT_GP_V3=1
    Enable complex SO2_m V3.  If unset, it inherits
    DPTB_TRITON_EXACT_GP_V3.
DPTB_TRITON_EXACT_GP_V3_BWD=expert_loop|v2_atomic|torch
    Use the V3 expert-loop atomic reduce, the V2 atomic reduce, or the exact
    Torch reduce.  ``expert_loop`` is the default.
DPTB_TRITON_EXACT_GP_V3_REQUIRE=1
    Raise on CPU / non-fp32 / missing Triton instead of silently falling back.

Tile knobs use V3 names first and fall back to V2 names for compatibility:
DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_M, _N, _K.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch

from .so2_triton_exact_gp_v2 import (  # type: ignore[attr-defined]
    _TRITON_AVAILABLE,
    _can_use_triton,
    _canonical_split_sizes,
    _cdiv,
    _check_complex_shapes,
    _check_real_shapes,
    _empty,
    _env_flag,
    _env_int,
    _flatten_grouped_rows,
    _launch_complex_dw_dc_atomic,
    _launch_complex_dx,
    _launch_complex_fwd,
    _launch_real_dw_dc_atomic,
    _launch_real_dx,
    _launch_real_fwd,
    _meta_tensors,
    _real_torch_backward,
    _complex_torch_backward,
    reference_complex_exact_moe_linear,
    reference_exact_moe_linear,
    tl,
    triton,
)


def use_exact_gp_v3() -> bool:
    """Whether the real-valued V3 route is requested."""

    return _env_flag("DPTB_TRITON_EXACT_GP_V3", "0")


def use_complex_exact_gp_v3() -> bool:
    """Whether the complex V3 route is requested."""

    return _env_flag(
        "DPTB_TRITON_COMPLEX_EXACT_GP_V3",
        os.environ.get("DPTB_TRITON_EXACT_GP_V3", "0"),
    )


def _require_v3() -> bool:
    return _env_flag(
        "DPTB_TRITON_EXACT_GP_V3_REQUIRE",
        os.environ.get("DPTB_TRITON_EXACT_GP_V2_REQUIRE", "0"),
    )


def _env_int_v3(name: str, v2_name: str, default: int) -> int:
    if os.environ.get(name) not in (None, ""):
        return _env_int(name, default)
    return _env_int(v2_name, default)


def _bwd_mode() -> str:
    return os.environ.get("DPTB_TRITON_EXACT_GP_V3_BWD", "expert_loop").strip().lower()


if _TRITON_AVAILABLE:  # pragma: no cover - compiled only on CUDA/Triton machines

    @triton.jit
    def _real_dw_dc_expert_loop_atomic_kernel(
        x_ptr,
        gy_ptr,
        coeff_ptr,
        w_ptr,
        bias_ptr,
        grad_c_ptr,
        grad_w_ptr,
        grad_b_ptr,
        grad_sw_ptr,
        grad_sb_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        TILES_N: tl.constexpr,
        TILES_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        HAS_SHARED_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid = tl.program_id(1)
        pid_k = pid % TILES_K
        tmp = pid // TILES_K
        pid_n = tmp % TILES_N
        pid_chunk = tmp // TILES_N

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT
        mask_k = offs_k < K_IN
        mask_nk = mask_n[:, None] & mask_k[None, :]

        x = tl.load(
            x_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        gy = tl.load(
            gy_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        grad_mixed = tl.dot(tl.trans(gy), x, input_precision="ieee")

        if HAS_SHARED_W:
            tl.atomic_add(
                grad_sw_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                grad_mixed,
                sem="relaxed",
                mask=mask_nk,
            )

        gb = tl.sum(gy, axis=0)
        only_once_per_n_tile = pid_k == 0
        if HAS_SHARED_B:
            tl.atomic_add(
                grad_sb_ptr + offs_n,
                gb,
                sem="relaxed",
                mask=mask_n & only_once_per_n_tile,
            )

        for e in range(NUM_EXPERTS):
            coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
            w = tl.load(
                w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                mask=mask_nk,
                other=0.0,
            )
            tl.atomic_add(
                grad_w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                coeff * grad_mixed,
                sem="relaxed",
                mask=mask_nk,
            )

            grad_c_sum = tl.sum(tl.sum(grad_mixed * w, axis=0), axis=0)
            if HAS_BIAS:
                b = tl.load(bias_ptr + e * N_OUT + offs_n, mask=mask_n, other=0.0)
                tl.atomic_add(
                    grad_b_ptr + e * N_OUT + offs_n,
                    coeff * gb,
                    sem="relaxed",
                    mask=mask_n & only_once_per_n_tile,
                )
                grad_c_sum += tl.where(only_once_per_n_tile, tl.sum(gb * b, axis=0), 0.0)
            tl.atomic_add(grad_c_ptr + g * NUM_EXPERTS + e, grad_c_sum, sem="relaxed")

    @triton.jit
    def _complex_dw_dc_expert_loop_atomic_kernel(
        x_ptr,
        gy_ptr,
        coeff_ptr,
        w_ptr,
        grad_c_ptr,
        grad_w_ptr,
        grad_sw_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        TILES_N: tl.constexpr,
        TILES_K: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid = tl.program_id(1)
        pid_k = pid % TILES_K
        tmp = pid // TILES_K
        pid_n = tmp % TILES_N
        pid_chunk = tmp // TILES_N

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT
        mask_k = offs_k < K_IN
        mask_nk = mask_n[:, None] & mask_k[None, :]

        xr = tl.load(
            x_ptr + (row_start + offs_m[:, None]) * (2 * K_IN) + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        xi = tl.load(
            x_ptr + (row_start + offs_m[:, None]) * (2 * K_IN) + K_IN + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        gyr = tl.load(
            gy_ptr + (row_start + offs_m[:, None]) * (2 * N_OUT) + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        gyi = tl.load(
            gy_ptr + (row_start + offs_m[:, None]) * (2 * N_OUT) + N_OUT + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        grad_wr = tl.dot(tl.trans(gyr), xr, input_precision="ieee") + tl.dot(
            tl.trans(gyi), xi, input_precision="ieee"
        )
        grad_wi = -tl.dot(tl.trans(gyr), xi, input_precision="ieee") + tl.dot(
            tl.trans(gyi), xr, input_precision="ieee"
        )

        if HAS_SHARED_W:
            tl.atomic_add(
                grad_sw_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                grad_wr,
                sem="relaxed",
                mask=mask_nk,
            )
            tl.atomic_add(
                grad_sw_ptr + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                grad_wi,
                sem="relaxed",
                mask=mask_nk,
            )

        for e in range(NUM_EXPERTS):
            coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
            wr = tl.load(
                w_ptr + e * (2 * N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                mask=mask_nk,
                other=0.0,
            )
            wi = tl.load(
                w_ptr + e * (2 * N_OUT * K_IN) + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                mask=mask_nk,
                other=0.0,
            )
            tl.atomic_add(
                grad_w_ptr + e * (2 * N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                coeff * grad_wr,
                sem="relaxed",
                mask=mask_nk,
            )
            tl.atomic_add(
                grad_w_ptr + e * (2 * N_OUT * K_IN) + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                coeff * grad_wi,
                sem="relaxed",
                mask=mask_nk,
            )
            grad_c_sum = tl.sum(tl.sum(grad_wr * wr + grad_wi * wi, axis=0), axis=0)
            tl.atomic_add(grad_c_ptr + g * NUM_EXPERTS + e, grad_c_sum, sem="relaxed")


def _launch_real_dw_dc_expert_loop_atomic(
    flat_x: torch.Tensor,
    flat_grad: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    rows_per_group: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not _can_use_triton(flat_x, flat_grad, coefficients, weight_experts, bias_experts, shared_weight, shared_bias):
        if _require_v3():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but real V3 expert-loop backward needs CUDA float32 tensors and Triton."
            )
        _, grad_c, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
            flat_x,
            flat_grad,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            rows_per_group,
        )
        return grad_c, grad_w, grad_b, grad_sw, grad_sb

    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    block_m = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_M", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M", 64)
    block_n = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_N", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N", 16)
    block_k = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_K", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1])
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    tiles_k = _cdiv(k_in, block_k)
    max_chunks = max(1, max(_cdiv(int(v), block_m) for v in rows_per_group))

    grad_c = torch.zeros_like(coefficients)
    grad_w = torch.zeros_like(weight_experts)
    grad_b = torch.zeros_like(bias_experts) if bias_experts is not None else None
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None
    grad_sb = torch.zeros_like(shared_bias) if shared_bias is not None else None

    grid = (len(rows_per_group), max_chunks * tiles_n * tiles_k)
    _real_dw_dc_expert_loop_atomic_kernel[grid](
        flat_x.contiguous(),
        flat_grad.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        bias_experts.contiguous() if bias_experts is not None else _empty(flat_x.device, flat_x.dtype),
        grad_c,
        grad_w,
        grad_b if grad_b is not None else _empty(flat_x.device, flat_x.dtype),
        grad_sw if grad_sw is not None else _empty(flat_x.device, flat_x.dtype),
        grad_sb if grad_sb is not None else _empty(flat_x.device, flat_x.dtype),
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        TILES_N=tiles_n,
        TILES_K=tiles_k,
        HAS_BIAS=bias_experts is not None,
        HAS_SHARED_W=shared_weight is not None,
        HAS_SHARED_B=shared_bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return grad_c, grad_w, grad_b, grad_sw, grad_sb


def _launch_complex_dw_dc_expert_loop_atomic(
    x_pair: torch.Tensor,
    grad_out: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_weight):
        if _require_v3():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but complex V3 expert-loop backward needs CUDA float32 tensors and Triton."
            )
        _, grad_c, grad_w, grad_sw = _complex_torch_backward(
            x_pair, grad_out, coefficients, weight_experts, shared_weight, split_sizes
        )
        return grad_c, grad_w, grad_sw

    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(split_sizes, x_pair.device)
    block_m = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_M", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M", 64)
    block_n = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_N", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N", 16)
    block_k = _env_int_v3("DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_K", "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    tiles_k = _cdiv(k_in, block_k)
    max_chunks = max(1, max(_cdiv(int(v), block_m) for v in split_sizes))

    grad_c = torch.zeros_like(coefficients)
    grad_w = torch.zeros_like(weight_experts)
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None

    grid = (len(split_sizes), max_chunks * tiles_n * tiles_k)
    _complex_dw_dc_expert_loop_atomic_kernel[grid](
        x_pair.contiguous(),
        grad_out.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        grad_c,
        grad_w,
        grad_sw if grad_sw is not None else _empty(x_pair.device, x_pair.dtype),
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        TILES_N=tiles_n,
        TILES_K=tiles_k,
        HAS_SHARED_W=shared_weight is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return grad_c, grad_w, grad_sw


class _ExactMoELinearV3Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        coefficients: torch.Tensor,
        weight_experts: torch.Tensor,
        bias_experts: torch.Tensor,
        shared_weight: torch.Tensor,
        shared_bias: torch.Tensor,
        split_sizes: Tuple[int, ...],
    ) -> torch.Tensor:
        bias_arg = bias_experts if bias_experts.numel() > 0 else None
        shared_w_arg = shared_weight if shared_weight.numel() > 0 else None
        shared_b_arg = shared_bias if shared_bias.numel() > 0 else None
        _check_real_shapes(x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, split_sizes)
        flat_x, rows_per_group = _flatten_grouped_rows(x, split_sizes)
        if use_exact_gp_v3() and _can_use_triton(flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg):
            flat_out = _launch_real_fwd(flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, rows_per_group)
        elif use_exact_gp_v3() and _require_v3():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but real V3 forward needs CUDA float32 tensors and Triton.")
        else:
            flat_out = reference_exact_moe_linear(
                flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, rows_per_group
            )
        ctx.save_for_backward(flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias)
        ctx.rows_per_group = rows_per_group
        ctx.orig_shape = tuple(int(v) for v in x.shape)
        ctx.has_bias = bias_arg is not None
        ctx.has_shared_weight = shared_w_arg is not None
        ctx.has_shared_bias = shared_b_arg is not None
        return flat_out.reshape(*x.shape[:-1], weight_experts.shape[1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias = ctx.saved_tensors
        bias_arg = bias_experts if ctx.has_bias else None
        shared_w_arg = shared_weight if ctx.has_shared_weight else None
        shared_b_arg = shared_bias if ctx.has_shared_bias else None
        flat_grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        use_cuda = use_exact_gp_v3() and _can_use_triton(
            flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg
        )
        if use_cuda:
            grad_x = _launch_real_dx(flat_grad, coefficients, weight_experts, shared_w_arg, ctx.rows_per_group)
            mode = _bwd_mode()
            if mode in {"expert_loop", "atomic", "v3", "v3_atomic"}:
                grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _launch_real_dw_dc_expert_loop_atomic(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
            elif mode in {"v2", "v2_atomic"}:
                grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _launch_real_dw_dc_atomic(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
            elif mode == "torch":
                _, grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
            else:
                raise ValueError("DPTB_TRITON_EXACT_GP_V3_BWD must be expert_loop, v2_atomic, or torch")
        elif use_exact_gp_v3() and _require_v3():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but real V3 backward needs CUDA float32 tensors and Triton.")
        else:
            grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
                flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
            )
        return grad_x.reshape(ctx.orig_shape), grad_coeff, grad_w, grad_b, grad_sw, grad_sb, None


def exact_moe_linear_v3(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact MoE linear with V3 expert-loop backward fusion."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ExactMoELinearV3Fn.apply(
        x,
        coefficients,
        weight_experts,
        bias_experts if bias_experts is not None else _empty(x.device, x.dtype),
        shared_weight if shared_weight is not None else _empty(x.device, x.dtype),
        shared_bias if shared_bias is not None else _empty(x.device, x.dtype),
        split_sizes,
    )


class _ComplexExactMoELinearV3Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_pair: torch.Tensor,
        coefficients: torch.Tensor,
        weight_experts: torch.Tensor,
        shared_weight: torch.Tensor,
        split_sizes: Tuple[int, ...],
    ) -> torch.Tensor:
        shared_w_arg = shared_weight if shared_weight.numel() > 0 else None
        _check_complex_shapes(x_pair, coefficients, weight_experts, shared_w_arg, split_sizes)
        if use_complex_exact_gp_v3() and _can_use_triton(x_pair, coefficients, weight_experts, shared_w_arg):
            out = _launch_complex_fwd(x_pair, coefficients, weight_experts, shared_w_arg, split_sizes)
        elif use_complex_exact_gp_v3() and _require_v3():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but complex V3 forward needs CUDA float32 tensors and Triton.")
        else:
            out = reference_complex_exact_moe_linear(x_pair, coefficients, weight_experts, shared_w_arg, split_sizes)
        ctx.save_for_backward(x_pair.contiguous(), coefficients, weight_experts, shared_weight)
        ctx.split_sizes = split_sizes
        ctx.has_shared_weight = shared_w_arg is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_pair, coefficients, weight_experts, shared_weight = ctx.saved_tensors
        shared_w_arg = shared_weight if ctx.has_shared_weight else None
        grad_out = grad_out.contiguous()
        use_cuda = use_complex_exact_gp_v3() and _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_w_arg)
        if use_cuda:
            grad_x = _launch_complex_dx(grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes)
            mode = _bwd_mode()
            if mode in {"expert_loop", "atomic", "v3", "v3_atomic"}:
                grad_coeff, grad_w, grad_sw = _launch_complex_dw_dc_expert_loop_atomic(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
            elif mode in {"v2", "v2_atomic"}:
                grad_coeff, grad_w, grad_sw = _launch_complex_dw_dc_atomic(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
            elif mode == "torch":
                _, grad_coeff, grad_w, grad_sw = _complex_torch_backward(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
            else:
                raise ValueError("DPTB_TRITON_EXACT_GP_V3_BWD must be expert_loop, v2_atomic, or torch")
        elif use_complex_exact_gp_v3() and _require_v3():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V3_REQUIRE=1 but complex V3 backward needs CUDA float32 tensors and Triton.")
        else:
            grad_x, grad_coeff, grad_w, grad_sw = _complex_torch_backward(
                x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
            )
        return grad_x, grad_coeff, grad_w, grad_sw, None


def complex_exact_moe_linear_v3(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact complex MoE linear with V3 expert-loop backward fusion."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ComplexExactMoELinearV3Fn.apply(
        x_pair,
        coefficients,
        weight_experts,
        shared_weight if shared_weight is not None else _empty(x_pair.device, x_pair.dtype),
        split_sizes,
    )
