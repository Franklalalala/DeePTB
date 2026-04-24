# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Triton graph-persistent exact MoE V4 overlay for DeePTB.

V4 builds on the V3 exact graph-persistent path.  V3's reduce kernel already
loads each ``x`` / ``grad_y`` tile once and loops over experts inside the
program, but it still atomically accumulates ``grad_coeff[g, e]`` from every
``(row_chunk, out_tile, in_tile)`` program.  That scalar atomic can become a hot
spot when the number of row chunks and N/K tiles is large.

V4 keeps the memory-light fused ``dW`` / ``dBias`` / shared-gradient atomics, but
moves the coefficient gradient to a two-stage path:

1. the main reduce kernel stores one small partial ``dCoeff`` value per
   ``(graph, expert, tile)`` instead of atomic-adding into ``grad_coeff``;
2. a second tiny Triton kernel reduces those partials to the final
   ``grad_coeff``.

The scratch tensor is intentionally bounded by
``DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB`` and the implementation can
fall back to V3/V2/Torch at runtime.  This route is experimental and opt-in.

Environment switches
--------------------
DPTB_TRITON_EXACT_GP_V4=1
    Enable real-valued V4.
DPTB_TRITON_COMPLEX_EXACT_GP_V4=1
    Enable complex SO2_m V4.  If unset, it inherits
    DPTB_TRITON_EXACT_GP_V4.
DPTB_TRITON_EXACT_GP_V4_BWD=split_coeff|v3_atomic|v2_atomic|torch
    Use the two-stage coefficient reduce, V3 expert-loop atomic reduce,
    V2 atomic reduce, or exact Torch backward.  ``split_coeff`` is the default.
DPTB_TRITON_EXACT_GP_V4_REQUIRE=1
    Raise on CPU / non-fp32 / missing Triton instead of silently falling back.
DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB=128
    Upper bound for the temporary coefficient-partials buffer.  Set to 0 to
    disable the limit.

Tile knobs use V4 names first, then V3 names, then V2 names:
DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_M, _N, _K.
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
from .so2_triton_exact_gp_v3 import (  # type: ignore[attr-defined]
    _launch_complex_dw_dc_expert_loop_atomic,
    _launch_real_dw_dc_expert_loop_atomic,
)


def use_exact_gp_v4() -> bool:
    """Whether the real-valued V4 route is requested."""

    return _env_flag("DPTB_TRITON_EXACT_GP_V4", "0")


def use_complex_exact_gp_v4() -> bool:
    """Whether the complex V4 route is requested."""

    return _env_flag(
        "DPTB_TRITON_COMPLEX_EXACT_GP_V4",
        os.environ.get("DPTB_TRITON_EXACT_GP_V4", "0"),
    )


def _require_v4() -> bool:
    return _env_flag(
        "DPTB_TRITON_EXACT_GP_V4_REQUIRE",
        os.environ.get("DPTB_TRITON_EXACT_GP_V3_REQUIRE", os.environ.get("DPTB_TRITON_EXACT_GP_V2_REQUIRE", "0")),
    )


def _env_int_cascade(names: Sequence[str], default: int) -> int:
    for name in names:
        if os.environ.get(name) not in (None, ""):
            return _env_int(name, default)
    return int(default)


def _bwd_mode() -> str:
    return os.environ.get("DPTB_TRITON_EXACT_GP_V4_BWD", "split_coeff").strip().lower()


def _scratch_limit_bytes() -> int:
    value = os.environ.get("DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB", "128")
    try:
        mb = float(value)
    except ValueError:
        mb = 128.0
    if mb <= 0:
        return 0
    return int(mb * 1024 * 1024)


def _scratch_fits(num_groups: int, num_experts: int, total_tiles: int, dtype: torch.dtype) -> bool:
    limit = _scratch_limit_bytes()
    if limit <= 0:
        return True
    elem_size = torch.empty((), dtype=dtype).element_size()
    needed = int(num_groups) * int(num_experts) * int(total_tiles) * elem_size
    return needed <= limit


def _reduce_blocks() -> tuple[int, int, int]:
    block_m = _env_int_cascade(
        (
            "DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_M",
            "DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_M",
            "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M",
        ),
        64,
    )
    block_n = _env_int_cascade(
        (
            "DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_N",
            "DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_N",
            "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N",
        ),
        16,
    )
    block_k = _env_int_cascade(
        (
            "DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_K",
            "DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_K",
            "DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K",
        ),
        32,
    )
    return int(block_m), int(block_n), int(block_k)


if _TRITON_AVAILABLE:  # pragma: no cover - compiled only on CUDA/Triton machines

    @triton.jit
    def _reduce_coeff_partials_kernel(
        partial_ptr,
        grad_c_ptr,
        TOTAL_TILES: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        g = tl.program_id(0)
        e = tl.program_id(1)
        offs = tl.arange(0, BLOCK_T)
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        base = (g * NUM_EXPERTS + e) * TOTAL_TILES
        for t0 in range(0, TOTAL_TILES, BLOCK_T):
            idx = t0 + offs
            vals = tl.load(partial_ptr + base + idx, mask=idx < TOTAL_TILES, other=0.0)
            acc += vals
        total = tl.sum(acc, axis=0)
        tl.store(grad_c_ptr + g * NUM_EXPERTS + e, total)

    @triton.jit
    def _real_dw_dc_split_coeff_kernel(
        x_ptr,
        gy_ptr,
        coeff_ptr,
        w_ptr,
        bias_ptr,
        coeff_partials_ptr,
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
        TOTAL_TILES: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        HAS_SHARED_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        flat_pid = tl.program_id(0)
        g = flat_pid // TOTAL_TILES
        pid = flat_pid - g * TOTAL_TILES
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
            tl.store(coeff_partials_ptr + (g * NUM_EXPERTS + e) * TOTAL_TILES + pid, grad_c_sum)

    @triton.jit
    def _complex_dw_dc_split_coeff_kernel(
        x_ptr,
        gy_ptr,
        coeff_ptr,
        w_ptr,
        coeff_partials_ptr,
        grad_w_ptr,
        grad_sw_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        TILES_N: tl.constexpr,
        TILES_K: tl.constexpr,
        TOTAL_TILES: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        flat_pid = tl.program_id(0)
        g = flat_pid // TOTAL_TILES
        pid = flat_pid - g * TOTAL_TILES
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
            tl.store(coeff_partials_ptr + (g * NUM_EXPERTS + e) * TOTAL_TILES + pid, grad_c_sum)


def _launch_reduce_coeff_partials(coeff_partials: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    assert triton is not None
    grad_c = torch.empty_like(coefficients)
    total_tiles = int(coeff_partials.shape[2])
    block_t = _env_int_cascade(("DPTB_TRITON_EXACT_GP_V4_COEFF_REDUCE_BLOCK_T",), 1024)
    _reduce_coeff_partials_kernel[(int(coefficients.shape[0]), int(coefficients.shape[1]))](
        coeff_partials,
        grad_c,
        TOTAL_TILES=total_tiles,
        NUM_EXPERTS=int(coefficients.shape[1]),
        BLOCK_T=block_t,
    )
    return grad_c


def _launch_real_dw_dc_split_coeff(
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
        if _require_v4():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but real V4 split-coeff backward needs CUDA float32 tensors and Triton."
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
    block_m, block_n, block_k = _reduce_blocks()
    n_out = int(weight_experts.shape[1])
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    tiles_k = _cdiv(k_in, block_k)
    max_chunks = max(1, max(_cdiv(int(v), block_m) for v in rows_per_group))
    total_tiles = int(max_chunks * tiles_n * tiles_k)
    if not _scratch_fits(len(rows_per_group), int(coefficients.shape[1]), total_tiles, coefficients.dtype):
        if _require_v4():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4 coefficient scratch exceeds DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB."
            )
        return _launch_real_dw_dc_expert_loop_atomic(
            flat_x, flat_grad, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, rows_per_group
        )

    coeff_partials = torch.empty(
        (len(rows_per_group), int(coefficients.shape[1]), total_tiles),
        device=flat_x.device,
        dtype=flat_x.dtype,
    )
    grad_w = torch.zeros_like(weight_experts)
    grad_b = torch.zeros_like(bias_experts) if bias_experts is not None else None
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None
    grad_sb = torch.zeros_like(shared_bias) if shared_bias is not None else None

    grid = (len(rows_per_group) * total_tiles,)
    _real_dw_dc_split_coeff_kernel[grid](
        flat_x.contiguous(),
        flat_grad.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        bias_experts.contiguous() if bias_experts is not None else _empty(flat_x.device, flat_x.dtype),
        coeff_partials,
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
        TOTAL_TILES=total_tiles,
        HAS_BIAS=bias_experts is not None,
        HAS_SHARED_W=shared_weight is not None,
        HAS_SHARED_B=shared_bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    grad_c = _launch_reduce_coeff_partials(coeff_partials, coefficients)
    return grad_c, grad_w, grad_b, grad_sw, grad_sb


def _launch_complex_dw_dc_split_coeff(
    x_pair: torch.Tensor,
    grad_out: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_weight):
        if _require_v4():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but complex V4 split-coeff backward needs CUDA float32 tensors and Triton."
            )
        _, grad_c, grad_w, grad_sw = _complex_torch_backward(
            x_pair, grad_out, coefficients, weight_experts, shared_weight, split_sizes
        )
        return grad_c, grad_w, grad_sw

    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(split_sizes, x_pair.device)
    block_m, block_n, block_k = _reduce_blocks()
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    tiles_k = _cdiv(k_in, block_k)
    max_chunks = max(1, max(_cdiv(int(v), block_m) for v in split_sizes))
    total_tiles = int(max_chunks * tiles_n * tiles_k)
    if not _scratch_fits(len(split_sizes), int(coefficients.shape[1]), total_tiles, coefficients.dtype):
        if _require_v4():
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4 coefficient scratch exceeds DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB."
            )
        return _launch_complex_dw_dc_expert_loop_atomic(x_pair, grad_out, coefficients, weight_experts, shared_weight, split_sizes)

    coeff_partials = torch.empty(
        (len(split_sizes), int(coefficients.shape[1]), total_tiles),
        device=x_pair.device,
        dtype=x_pair.dtype,
    )
    grad_w = torch.zeros_like(weight_experts)
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None

    grid = (len(split_sizes) * total_tiles,)
    _complex_dw_dc_split_coeff_kernel[grid](
        x_pair.contiguous(),
        grad_out.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        coeff_partials,
        grad_w,
        grad_sw if grad_sw is not None else _empty(x_pair.device, x_pair.dtype),
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        TILES_N=tiles_n,
        TILES_K=tiles_k,
        TOTAL_TILES=total_tiles,
        HAS_SHARED_W=shared_weight is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    grad_c = _launch_reduce_coeff_partials(coeff_partials, coefficients)
    return grad_c, grad_w, grad_sw


class _ExactMoELinearV4Fn(torch.autograd.Function):
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
        if use_exact_gp_v4() and _can_use_triton(flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg):
            flat_out = _launch_real_fwd(flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, rows_per_group)
        elif use_exact_gp_v4() and _require_v4():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but real V4 forward needs CUDA float32 tensors and Triton.")
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
        use_cuda = use_exact_gp_v4() and _can_use_triton(
            flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg
        )
        if use_cuda:
            grad_x = _launch_real_dx(flat_grad, coefficients, weight_experts, shared_w_arg, ctx.rows_per_group)
            mode = _bwd_mode()
            if mode in {"split_coeff", "split", "v4", "v4_split_coeff"}:
                grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _launch_real_dw_dc_split_coeff(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
            elif mode in {"v3", "v3_atomic", "expert_loop", "atomic"}:
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
                raise ValueError("DPTB_TRITON_EXACT_GP_V4_BWD must be split_coeff, v3_atomic, v2_atomic, or torch")
        elif use_exact_gp_v4() and _require_v4():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but real V4 backward needs CUDA float32 tensors and Triton.")
        else:
            grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
                flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
            )
        return grad_x.reshape(ctx.orig_shape), grad_coeff, grad_w, grad_b, grad_sw, grad_sb, None


def exact_moe_linear_v4(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact MoE linear with V4 split-coefficient backward fusion."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ExactMoELinearV4Fn.apply(
        x,
        coefficients,
        weight_experts,
        bias_experts if bias_experts is not None else _empty(x.device, x.dtype),
        shared_weight if shared_weight is not None else _empty(x.device, x.dtype),
        shared_bias if shared_bias is not None else _empty(x.device, x.dtype),
        split_sizes,
    )


class _ComplexExactMoELinearV4Fn(torch.autograd.Function):
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
        if use_complex_exact_gp_v4() and _can_use_triton(x_pair, coefficients, weight_experts, shared_w_arg):
            out = _launch_complex_fwd(x_pair, coefficients, weight_experts, shared_w_arg, split_sizes)
        elif use_complex_exact_gp_v4() and _require_v4():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but complex V4 forward needs CUDA float32 tensors and Triton.")
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
        use_cuda = use_complex_exact_gp_v4() and _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_w_arg)
        if use_cuda:
            grad_x = _launch_complex_dx(grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes)
            mode = _bwd_mode()
            if mode in {"split_coeff", "split", "v4", "v4_split_coeff"}:
                grad_coeff, grad_w, grad_sw = _launch_complex_dw_dc_split_coeff(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
            elif mode in {"v3", "v3_atomic", "expert_loop", "atomic"}:
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
                raise ValueError("DPTB_TRITON_EXACT_GP_V4_BWD must be split_coeff, v3_atomic, v2_atomic, or torch")
        elif use_complex_exact_gp_v4() and _require_v4():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V4_REQUIRE=1 but complex V4 backward needs CUDA float32 tensors and Triton.")
        else:
            grad_x, grad_coeff, grad_w, grad_sw = _complex_torch_backward(
                x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
            )
        return grad_x, grad_coeff, grad_w, grad_sw, None


def complex_exact_moe_linear_v4(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact complex MoE linear with V4 split-coefficient backward fusion."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ComplexExactMoELinearV4Fn.apply(
        x_pair,
        coefficients,
        weight_experts,
        shared_weight if shared_weight is not None else _empty(x_pair.device, x_pair.dtype),
        split_sizes,
    )
