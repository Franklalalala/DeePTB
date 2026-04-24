# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Aggressive graph-persistent exact MoE Triton kernels for DeePTB.

This module is intentionally opt-in.  It provides drop-in forward/backward
implementations for the exact graph-mix MoE route used by
``so2_triton_grouped_linear_ops.py`` without materialising graph-mixed weights.

Environment switches
--------------------
DPTB_TRITON_EXACT_GP_V2=1
    Enable the real-valued exact MoE v2 path.
DPTB_TRITON_COMPLEX_EXACT_GP_V2=1
    Enable the complex SO2_m exact MoE v2 path.  If unset, this inherits
    DPTB_TRITON_EXACT_GP_V2.
DPTB_TRITON_EXACT_GP_V2_BWD=atomic|torch
    Use the atomic fused dW/dCoeff reduce, or fall back to the torch reduce.
DPTB_TRITON_EXACT_GP_V2_REQUIRE=1
    Raise if v2 is requested but Triton/CUDA conditions are not met.

The CPU / non-Triton fallback is exact and differentiable through an explicit
custom backward, which makes this file safe to import and test on developer
machines without Triton installed.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_TRITON_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def use_exact_gp_v2() -> bool:
    """Whether the real-valued graph-persistent v2 route is requested."""

    return _env_flag("DPTB_TRITON_EXACT_GP_V2", "0")


def use_complex_exact_gp_v2() -> bool:
    """Whether the complex graph-persistent v2 route is requested."""

    return _env_flag(
        "DPTB_TRITON_COMPLEX_EXACT_GP_V2",
        os.environ.get("DPTB_TRITON_EXACT_GP_V2", "0"),
    )


def _require_v2() -> bool:
    return _env_flag("DPTB_TRITON_EXACT_GP_V2_REQUIRE", "0")


def _canonical_split_sizes(split_sizes: Sequence[int]) -> Tuple[int, ...]:
    out = tuple(int(v) for v in split_sizes)
    if any(v < 0 for v in out):
        raise ValueError(f"split_sizes must be non-negative, got {out!r}")
    return out


def _empty(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=dtype)


_META_CACHE: dict[tuple[str, Tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}


def _meta_tensors(rows_per_group: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _canonical_split_sizes(rows_per_group)
    key = (str(device), rows)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    row_sizes = torch.tensor(rows, device=device, dtype=torch.int32)
    row_offsets = torch.zeros(len(rows), device=device, dtype=torch.int32)
    if len(rows) > 1:
        row_offsets[1:] = torch.cumsum(row_sizes, dim=0)[:-1]
    cached = (row_offsets, row_sizes)
    _META_CACHE[key] = cached
    return cached


def _flatten_grouped_rows(x: torch.Tensor, split_sizes: Sequence[int]) -> tuple[torch.Tensor, Tuple[int, ...]]:
    split_sizes = _canonical_split_sizes(split_sizes)
    if int(sum(split_sizes)) != int(x.shape[0]):
        raise ValueError(f"split sizes sum to {sum(split_sizes)}, but x has {x.shape[0]} rows.")
    lead = 1
    if x.ndim > 2:
        for dim in x.shape[1:-1]:
            lead *= int(dim)
    rows_per_group = tuple(int(v) * lead for v in split_sizes)
    flat_x = x.reshape(sum(rows_per_group), x.shape[-1]).contiguous()
    return flat_x, rows_per_group


def _check_real_shapes(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> Tuple[int, int, int, int]:
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {tuple(x.shape)}")
    if coefficients.ndim != 2:
        raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
    if weight_experts.ndim != 3:
        raise ValueError(f"weight_experts must have shape [E, O, I], got {tuple(weight_experts.shape)}")
    split_sizes = _canonical_split_sizes(split_sizes)
    groups = len(split_sizes)
    experts, n_out, k_in = (int(weight_experts.shape[0]), int(weight_experts.shape[1]), int(weight_experts.shape[2]))
    if int(coefficients.shape[0]) != groups:
        raise ValueError(f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {groups} groups.")
    if int(coefficients.shape[1]) != experts:
        raise ValueError(f"coefficients has {coefficients.shape[1]} experts, but weight_experts has {experts} experts.")
    if int(x.shape[-1]) != k_in:
        raise ValueError(f"weight_experts in_features={k_in} does not match x last dim={x.shape[-1]}.")
    if bias_experts is not None and tuple(bias_experts.shape) != (experts, n_out):
        raise ValueError(f"bias_experts must have shape {(experts, n_out)}, got {tuple(bias_experts.shape)}")
    if shared_weight is not None and tuple(shared_weight.shape) != (n_out, k_in):
        raise ValueError(f"shared_weight must have shape {(n_out, k_in)}, got {tuple(shared_weight.shape)}")
    if shared_bias is not None and tuple(shared_bias.shape) != (n_out,):
        raise ValueError(f"shared_bias must have shape {(n_out,)}, got {tuple(shared_bias.shape)}")
    return groups, experts, n_out, k_in


def _check_complex_shapes(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> Tuple[int, int, int, int]:
    if x_pair.ndim != 3 or int(x_pair.shape[1]) != 2:
        raise ValueError(f"x_pair must have shape [N, 2, Cin], got {tuple(x_pair.shape)}")
    if coefficients.ndim != 2:
        raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
    if weight_experts.ndim != 3 or int(weight_experts.shape[1]) % 2 != 0:
        raise ValueError(f"weight_experts must have shape [E, 2*Cout, Cin], got {tuple(weight_experts.shape)}")
    split_sizes = _canonical_split_sizes(split_sizes)
    groups = len(split_sizes)
    experts = int(weight_experts.shape[0])
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    if int(sum(split_sizes)) != int(x_pair.shape[0]):
        raise ValueError(f"split sizes sum to {sum(split_sizes)}, but x_pair has {x_pair.shape[0]} rows.")
    if int(coefficients.shape[0]) != groups:
        raise ValueError(f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {groups} groups.")
    if int(coefficients.shape[1]) != experts:
        raise ValueError(f"coefficients has {coefficients.shape[1]} experts, but weight_experts has {experts} experts.")
    if int(x_pair.shape[2]) != k_in:
        raise ValueError(f"weight_experts Cin={k_in} does not match x_pair Cin={x_pair.shape[2]}.")
    if shared_weight is not None and tuple(shared_weight.shape) != (2 * n_out, k_in):
        raise ValueError(f"shared_weight must have shape {(2 * n_out, k_in)}, got {tuple(shared_weight.shape)}")
    return groups, experts, n_out, k_in


def _can_use_triton(*tensors: Optional[torch.Tensor], dtype: torch.dtype = torch.float32) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    live = [t for t in tensors if t is not None and t.numel() > 0]
    if not live:
        return False
    return all(t.device.type == "cuda" and t.dtype == dtype for t in live)


def _num_sms(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _cdiv(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _mix_real(
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    groups = int(coefficients.shape[0])
    mixed_w = coefficients.matmul(weight_experts.reshape(weight_experts.shape[0], -1))
    mixed_w = mixed_w.reshape(groups, weight_experts.shape[1], weight_experts.shape[2])
    if shared_weight is not None:
        mixed_w = mixed_w + shared_weight.unsqueeze(0)
    mixed_b = None
    if bias_experts is not None:
        mixed_b = coefficients.matmul(bias_experts)
        if shared_bias is not None:
            mixed_b = mixed_b + shared_bias.unsqueeze(0)
    elif shared_bias is not None:
        mixed_b = shared_bias.unsqueeze(0).expand(groups, -1)
    return mixed_w.contiguous(), mixed_b.contiguous() if mixed_b is not None else None


def reference_exact_moe_linear(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Torch reference for exact graph-mix real MoE linear."""

    split_sizes = _canonical_split_sizes(split_sizes)
    _check_real_shapes(x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, split_sizes)
    flat_x, rows_per_group = _flatten_grouped_rows(x, split_sizes)
    mixed_w, mixed_b = _mix_real(coefficients, weight_experts, bias_experts, shared_weight, shared_bias)
    outs = []
    start = 0
    for g, rows in enumerate(rows_per_group):
        xg = flat_x[start : start + int(rows)]
        bg = mixed_b[g] if mixed_b is not None else None
        outs.append(F.linear(xg, mixed_w[g], bg))
        start += int(rows)
    flat_out = torch.cat(outs, dim=0) if outs else flat_x.new_empty((0, weight_experts.shape[1]))
    return flat_out.reshape(*x.shape[:-1], weight_experts.shape[1])


def _real_torch_backward(
    flat_x: torch.Tensor,
    flat_grad: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    rows_per_group: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    mixed_w, _ = _mix_real(coefficients, weight_experts, bias_experts, shared_weight, shared_bias)
    grad_x = torch.empty((flat_x.shape[0], weight_experts.shape[2]), device=flat_x.device, dtype=flat_x.dtype)
    grad_mixed_w = torch.zeros((len(rows_per_group), weight_experts.shape[1], weight_experts.shape[2]), device=flat_x.device, dtype=flat_x.dtype)
    grad_mixed_b = torch.zeros((len(rows_per_group), weight_experts.shape[1]), device=flat_x.device, dtype=flat_x.dtype) if (bias_experts is not None or shared_bias is not None) else None
    acc_dtype = torch.float32 if flat_x.dtype in (torch.float16, torch.bfloat16) else flat_x.dtype
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xg = flat_x[start : start + rows]
        gg = flat_grad[start : start + rows]
        grad_x[start : start + rows] = gg.matmul(mixed_w[g]).to(flat_x.dtype)
        grad_mixed_w[g] = gg.to(acc_dtype).transpose(0, 1).matmul(xg.to(acc_dtype)).to(flat_x.dtype)
        if grad_mixed_b is not None:
            grad_mixed_b[g] = gg.to(acc_dtype).sum(dim=0).to(flat_x.dtype)
        start += rows

    grad_mixed_w_flat = grad_mixed_w.reshape(grad_mixed_w.shape[0], -1)
    weight_flat = weight_experts.reshape(weight_experts.shape[0], -1)
    grad_coeff = grad_mixed_w_flat.matmul(weight_flat.transpose(0, 1))
    grad_w = coefficients.transpose(0, 1).matmul(grad_mixed_w_flat).reshape_as(weight_experts)
    grad_b = None
    if bias_experts is not None and grad_mixed_b is not None:
        grad_coeff = grad_coeff + grad_mixed_b.matmul(bias_experts.transpose(0, 1))
        grad_b = coefficients.transpose(0, 1).matmul(grad_mixed_b)
    grad_sw = grad_mixed_w.sum(dim=0) if shared_weight is not None else None
    grad_sb = grad_mixed_b.sum(dim=0) if shared_bias is not None and grad_mixed_b is not None else None
    return grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb


def reference_complex_exact_moe_linear(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Torch reference for exact graph-mix complex MoE linear.

    ``weight_experts[:, :Cout]`` are real weights and ``weight_experts[:, Cout:]``
    are imaginary weights.
    """

    split_sizes = _canonical_split_sizes(split_sizes)
    _check_complex_shapes(x_pair, coefficients, weight_experts, shared_weight, split_sizes)
    cout = int(weight_experts.shape[1] // 2)
    mixed, _ = _mix_real(coefficients, weight_experts, None, shared_weight, None)
    out = x_pair.new_empty((x_pair.shape[0], 2, cout))
    start = 0
    for g, rows in enumerate(split_sizes):
        rows = int(rows)
        xr = x_pair[start : start + rows, 0, :]
        xi = x_pair[start : start + rows, 1, :]
        wr = mixed[g, :cout, :]
        wi = mixed[g, cout:, :]
        out[start : start + rows, 0, :] = xr.matmul(wr.transpose(0, 1)) - xi.matmul(wi.transpose(0, 1))
        out[start : start + rows, 1, :] = xr.matmul(wi.transpose(0, 1)) + xi.matmul(wr.transpose(0, 1))
        start += rows
    return out


def _complex_torch_backward(
    x_pair: torch.Tensor,
    grad_out: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    split_sizes = _canonical_split_sizes(split_sizes)
    cout = int(weight_experts.shape[1] // 2)
    mixed, _ = _mix_real(coefficients, weight_experts, None, shared_weight, None)
    grad_x = torch.empty_like(x_pair)
    grad_mixed = torch.zeros((len(split_sizes), 2 * cout, weight_experts.shape[2]), device=x_pair.device, dtype=x_pair.dtype)
    acc_dtype = torch.float32 if x_pair.dtype in (torch.float16, torch.bfloat16) else x_pair.dtype
    start = 0
    for g, rows in enumerate(split_sizes):
        rows = int(rows)
        xr = x_pair[start : start + rows, 0, :]
        xi = x_pair[start : start + rows, 1, :]
        gyr = grad_out[start : start + rows, 0, :]
        gyi = grad_out[start : start + rows, 1, :]
        wr = mixed[g, :cout, :]
        wi = mixed[g, cout:, :]
        grad_x[start : start + rows, 0, :] = gyr.matmul(wr) + gyi.matmul(wi)
        grad_x[start : start + rows, 1, :] = -gyr.matmul(wi) + gyi.matmul(wr)
        xr_a = xr.to(acc_dtype)
        xi_a = xi.to(acc_dtype)
        gyr_a = gyr.to(acc_dtype)
        gyi_a = gyi.to(acc_dtype)
        grad_wr = gyr_a.transpose(0, 1).matmul(xr_a) + gyi_a.transpose(0, 1).matmul(xi_a)
        grad_wi = -gyr_a.transpose(0, 1).matmul(xi_a) + gyi_a.transpose(0, 1).matmul(xr_a)
        grad_mixed[g, :cout, :] = grad_wr.to(x_pair.dtype)
        grad_mixed[g, cout:, :] = grad_wi.to(x_pair.dtype)
        start += rows
    grad_mixed_flat = grad_mixed.reshape(grad_mixed.shape[0], -1)
    weight_flat = weight_experts.reshape(weight_experts.shape[0], -1)
    grad_coeff = grad_mixed_flat.matmul(weight_flat.transpose(0, 1))
    grad_w = coefficients.transpose(0, 1).matmul(grad_mixed_flat).reshape_as(weight_experts)
    grad_sw = grad_mixed.sum(dim=0) if shared_weight is not None else None
    return grad_x, grad_coeff, grad_w, grad_sw


if _TRITON_AVAILABLE:  # pragma: no cover - compiled only on CUDA/Triton machines

    @triton.jit
    def _real_fwd_persistent_kernel(
        x_ptr,
        coeff_ptr,
        w_ptr,
        bias_ptr,
        shared_w_ptr,
        shared_b_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        TOTAL_TILES: tl.constexpr,
        TILES_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        HAS_SHARED_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        remaining = tl.program_id(0)
        selected_g = tl.full((), 0, tl.int32)
        selected_remaining = remaining
        found = tl.full((), False, tl.int1)
        for g_scan in range(NUM_GROUPS):
            rows_g = tl.load(row_sizes_ptr + g_scan)
            tiles_g = tl.cdiv(rows_g, BLOCK_M) * TILES_N
            take = (remaining < tiles_g) & (~found)
            selected_g = tl.where(take, g_scan, selected_g)
            selected_remaining = tl.where(take, remaining, selected_remaining)
            remaining = tl.where(found | take, remaining, remaining - tiles_g)
            found = found | take

        g = selected_g
        rem = selected_remaining
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        pid_m = rem // TILES_N
        pid_n = rem - pid_m * TILES_N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K_IN, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_IN
            x = tl.load(
                x_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                c = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += c * w
            if HAS_SHARED_W:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw
            acc += tl.dot(x, tl.trans(mixed_w), input_precision="ieee")

        if HAS_BIAS:
            mixed_b = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                c = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                b = tl.load(bias_ptr + e * N_OUT + offs_n, mask=mask_n, other=0.0)
                mixed_b += c * b
            acc += mixed_b[None, :]
        if HAS_SHARED_B:
            sb = tl.load(shared_b_ptr + offs_n, mask=mask_n, other=0.0)
            acc += sb[None, :]

        tl.store(
            y_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _real_dx_persistent_kernel(
        gy_ptr,
        coeff_ptr,
        w_ptr,
        shared_w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        TOTAL_TILES: tl.constexpr,
        TILES_K: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        remaining = tl.program_id(0)
        selected_g = tl.full((), 0, tl.int32)
        selected_remaining = remaining
        found = tl.full((), False, tl.int1)
        for g_scan in range(NUM_GROUPS):
            rows_g = tl.load(row_sizes_ptr + g_scan)
            tiles_g = tl.cdiv(rows_g, BLOCK_M) * TILES_K
            take = (remaining < tiles_g) & (~found)
            selected_g = tl.where(take, g_scan, selected_g)
            selected_remaining = tl.where(take, remaining, selected_remaining)
            remaining = tl.where(found | take, remaining, remaining - tiles_g)
            found = found | take

        g = selected_g
        rem = selected_remaining
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        pid_m = rem // TILES_K
        pid_k = rem - pid_m * TILES_K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_k = offs_k < K_IN
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n0 in range(0, N_OUT, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_OUT
            gy = tl.load(
                gy_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                c = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += c * w
            if HAS_SHARED_W:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw
            acc += tl.dot(gy, mixed_w, input_precision="ieee")

        tl.store(
            gx_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
            acc,
            mask=mask_m[:, None] & mask_k[None, :],
        )

    @triton.jit
    def _real_dw_dc_atomic_kernel(
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
        e = tl.program_id(1)
        pid = tl.program_id(2)
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
        coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
        w = tl.load(
            w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        mask_nk = mask_n[:, None] & mask_k[None, :]
        tl.atomic_add(
            grad_w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
            coeff * grad_mixed,
            sem="relaxed",
            mask=mask_nk,
        )
        grad_c_tile = grad_mixed * w
        grad_c_sum = tl.sum(tl.sum(grad_c_tile, axis=0), axis=0)
        tl.atomic_add(grad_c_ptr + g * NUM_EXPERTS + e, grad_c_sum, sem="relaxed")

        if HAS_SHARED_W:
            tl.atomic_add(
                grad_sw_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                grad_mixed,
                sem="relaxed",
                mask=mask_nk & (e == 0),
            )
        if HAS_BIAS or HAS_SHARED_B:
            gb = tl.sum(gy, axis=0)
            only_once_per_n_tile = pid_k == 0
            if HAS_BIAS:
                b = tl.load(bias_ptr + e * N_OUT + offs_n, mask=mask_n, other=0.0)
                tl.atomic_add(
                    grad_b_ptr + e * N_OUT + offs_n,
                    coeff * gb,
                    sem="relaxed",
                    mask=mask_n & only_once_per_n_tile,
                )
                gb_c = tl.sum(gb * b, axis=0)
                tl.atomic_add(
                    grad_c_ptr + g * NUM_EXPERTS + e,
                    gb_c,
                    sem="relaxed",
                    mask=only_once_per_n_tile,
                )
            if HAS_SHARED_B:
                tl.atomic_add(
                    grad_sb_ptr + offs_n,
                    gb,
                    sem="relaxed",
                    mask=mask_n & only_once_per_n_tile & (e == 0),
                )

    @triton.jit
    def _complex_fwd_persistent_kernel(
        x_ptr,
        coeff_ptr,
        w_ptr,
        shared_w_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        TOTAL_TILES: tl.constexpr,
        TILES_N: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        remaining = tl.program_id(0)
        selected_g = tl.full((), 0, tl.int32)
        selected_remaining = remaining
        found = tl.full((), False, tl.int1)
        for g_scan in range(NUM_GROUPS):
            rows_g = tl.load(row_sizes_ptr + g_scan)
            tiles_g = tl.cdiv(rows_g, BLOCK_M) * TILES_N
            take = (remaining < tiles_g) & (~found)
            selected_g = tl.where(take, g_scan, selected_g)
            selected_remaining = tl.where(take, remaining, selected_remaining)
            remaining = tl.where(found | take, remaining, remaining - tiles_g)
            found = found | take
        g = selected_g
        rem = selected_remaining
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        pid_m = rem // TILES_N
        pid_n = rem - pid_m * TILES_N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT
        acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K_IN, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_IN
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
            wr = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            wi = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                c = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                wr_e = tl.load(
                    w_ptr + e * (2 * N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wi_e = tl.load(
                    w_ptr + e * (2 * N_OUT * K_IN) + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wr += c * wr_e
                wi += c * wi_e
            if HAS_SHARED_W:
                swr = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                swi = tl.load(
                    shared_w_ptr + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wr += swr
                wi += swi
            acc_r += tl.dot(xr, tl.trans(wr), input_precision="ieee")
            acc_r -= tl.dot(xi, tl.trans(wi), input_precision="ieee")
            acc_i += tl.dot(xr, tl.trans(wi), input_precision="ieee")
            acc_i += tl.dot(xi, tl.trans(wr), input_precision="ieee")
        tl.store(
            y_ptr + (row_start + offs_m[:, None]) * (2 * N_OUT) + offs_n[None, :],
            acc_r,
            mask=mask_m[:, None] & mask_n[None, :],
        )
        tl.store(
            y_ptr + (row_start + offs_m[:, None]) * (2 * N_OUT) + N_OUT + offs_n[None, :],
            acc_i,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _complex_dx_persistent_kernel(
        gy_ptr,
        coeff_ptr,
        w_ptr,
        shared_w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        TOTAL_TILES: tl.constexpr,
        TILES_K: tl.constexpr,
        HAS_SHARED_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        remaining = tl.program_id(0)
        selected_g = tl.full((), 0, tl.int32)
        selected_remaining = remaining
        found = tl.full((), False, tl.int1)
        for g_scan in range(NUM_GROUPS):
            rows_g = tl.load(row_sizes_ptr + g_scan)
            tiles_g = tl.cdiv(rows_g, BLOCK_M) * TILES_K
            take = (remaining < tiles_g) & (~found)
            selected_g = tl.where(take, g_scan, selected_g)
            selected_remaining = tl.where(take, remaining, selected_remaining)
            remaining = tl.where(found | take, remaining, remaining - tiles_g)
            found = found | take
        g = selected_g
        rem = selected_remaining
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        pid_m = rem // TILES_K
        pid_k = rem - pid_m * TILES_K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_k = offs_k < K_IN
        acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n0 in range(0, N_OUT, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_OUT
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
            wr = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            wi = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                c = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                wr_e = tl.load(
                    w_ptr + e * (2 * N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wi_e = tl.load(
                    w_ptr + e * (2 * N_OUT * K_IN) + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wr += c * wr_e
                wi += c * wi_e
            if HAS_SHARED_W:
                swr = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                swi = tl.load(
                    shared_w_ptr + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wr += swr
                wi += swi
            acc_r += tl.dot(gyr, wr, input_precision="ieee")
            acc_r += tl.dot(gyi, wi, input_precision="ieee")
            acc_i -= tl.dot(gyr, wi, input_precision="ieee")
            acc_i += tl.dot(gyi, wr, input_precision="ieee")
        tl.store(
            gx_ptr + (row_start + offs_m[:, None]) * (2 * K_IN) + offs_k[None, :],
            acc_r,
            mask=mask_m[:, None] & mask_k[None, :],
        )
        tl.store(
            gx_ptr + (row_start + offs_m[:, None]) * (2 * K_IN) + K_IN + offs_k[None, :],
            acc_i,
            mask=mask_m[:, None] & mask_k[None, :],
        )

    @triton.jit
    def _complex_dw_dc_atomic_kernel(
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
        e = tl.program_id(1)
        pid = tl.program_id(2)
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
        grad_wr = tl.dot(tl.trans(gyr), xr, input_precision="ieee") + tl.dot(tl.trans(gyi), xi, input_precision="ieee")
        grad_wi = -tl.dot(tl.trans(gyr), xi, input_precision="ieee") + tl.dot(tl.trans(gyi), xr, input_precision="ieee")
        coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
        wr = tl.load(
            w_ptr + e * (2 * N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        wi = tl.load(
            w_ptr + e * (2 * N_OUT * K_IN) + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        mask_nk = mask_n[:, None] & mask_k[None, :]
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
        grad_c_tile = grad_wr * wr + grad_wi * wi
        grad_c_sum = tl.sum(tl.sum(grad_c_tile, axis=0), axis=0)
        tl.atomic_add(grad_c_ptr + g * NUM_EXPERTS + e, grad_c_sum, sem="relaxed")
        if HAS_SHARED_W:
            tl.atomic_add(
                grad_sw_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                grad_wr,
                sem="relaxed",
                mask=mask_nk & (e == 0),
            )
            tl.atomic_add(
                grad_sw_ptr + (N_OUT + offs_n[:, None]) * K_IN + offs_k[None, :],
                grad_wi,
                sem="relaxed",
                mask=mask_nk & (e == 0),
            )


def _launch_real_fwd(
    flat_x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    rows_per_group: Sequence[int],
) -> torch.Tensor:
    if not _can_use_triton(flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias):
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but real v2 forward needs CUDA float32 tensors and Triton.")
        return reference_exact_moe_linear(
            flat_x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            rows_per_group,
        )
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_M", 128)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_N", 64)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1])
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    total_tiles = sum(_cdiv(int(v), block_m) * tiles_n for v in rows_per_group)
    out = torch.empty((flat_x.shape[0], n_out), device=flat_x.device, dtype=flat_x.dtype)
    grid = (max(1, int(total_tiles)),)
    _real_fwd_persistent_kernel[grid](
        flat_x.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        bias_experts.contiguous() if bias_experts is not None else _empty(flat_x.device, flat_x.dtype),
        shared_weight.contiguous() if shared_weight is not None else _empty(flat_x.device, flat_x.dtype),
        shared_bias.contiguous() if shared_bias is not None else _empty(flat_x.device, flat_x.dtype),
        out,
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        NUM_GROUPS=len(rows_per_group),
        TOTAL_TILES=int(total_tiles),
        TILES_N=tiles_n,
        HAS_BIAS=bias_experts is not None,
        HAS_SHARED_W=shared_weight is not None,
        HAS_SHARED_B=shared_bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _launch_real_dx(
    flat_grad: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    rows_per_group: Sequence[int],
) -> torch.Tensor:
    if not _can_use_triton(flat_grad, coefficients, weight_experts, shared_weight):
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but real v2 dX needs CUDA float32 tensors and Triton.")
        # Caller will use the full torch backward, this branch is for direct smoke use.
        raise RuntimeError("real dX v2 cannot launch without Triton; use _real_torch_backward instead")
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_grad.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_M", 128)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_N", 64)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1])
    k_in = int(weight_experts.shape[2])
    tiles_k = _cdiv(k_in, block_k)
    total_tiles = sum(_cdiv(int(v), block_m) * tiles_k for v in rows_per_group)
    grad_x = torch.empty((flat_grad.shape[0], k_in), device=flat_grad.device, dtype=flat_grad.dtype)
    grid = (max(1, int(total_tiles)),)
    _real_dx_persistent_kernel[grid](
        flat_grad.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        shared_weight.contiguous() if shared_weight is not None else _empty(flat_grad.device, flat_grad.dtype),
        grad_x,
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        NUM_GROUPS=len(rows_per_group),
        TOTAL_TILES=int(total_tiles),
        TILES_K=tiles_k,
        HAS_SHARED_W=shared_weight is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return grad_x


def _launch_real_dw_dc_atomic(
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
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but real v2 atomic backward needs CUDA float32 tensors and Triton.")
        _, grad_c, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
            flat_x, flat_grad, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, rows_per_group
        )
        return grad_c, grad_w, grad_b, grad_sw, grad_sb
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M", 64)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N", 32)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K", 32)
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
    grid = (len(rows_per_group), int(coefficients.shape[1]), max_chunks * tiles_n * tiles_k)
    _real_dw_dc_atomic_kernel[grid](
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


def _launch_complex_fwd(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    if not _can_use_triton(x_pair, coefficients, weight_experts, shared_weight):
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but complex v2 forward needs CUDA float32 tensors and Triton.")
        return reference_complex_exact_moe_linear(x_pair, coefficients, weight_experts, shared_weight, split_sizes)
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(split_sizes, x_pair.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_M", 128)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_N", 64)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    total_tiles = sum(_cdiv(int(v), block_m) * tiles_n for v in split_sizes)
    out = torch.empty((x_pair.shape[0], 2, n_out), device=x_pair.device, dtype=x_pair.dtype)
    grid = (max(1, int(total_tiles)),)
    _complex_fwd_persistent_kernel[grid](
        x_pair.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        shared_weight.contiguous() if shared_weight is not None else _empty(x_pair.device, x_pair.dtype),
        out,
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        NUM_GROUPS=len(split_sizes),
        TOTAL_TILES=int(total_tiles),
        TILES_N=tiles_n,
        HAS_SHARED_W=shared_weight is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _launch_complex_dx(
    grad_out: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    if not _can_use_triton(grad_out, coefficients, weight_experts, shared_weight):
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but complex v2 dX needs CUDA float32 tensors and Triton.")
        raise RuntimeError("complex dX v2 cannot launch without Triton; use _complex_torch_backward instead")
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(split_sizes, grad_out.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_M", 128)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_N", 64)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    tiles_k = _cdiv(k_in, block_k)
    total_tiles = sum(_cdiv(int(v), block_m) * tiles_k for v in split_sizes)
    grad_x = torch.empty((grad_out.shape[0], 2, k_in), device=grad_out.device, dtype=grad_out.dtype)
    grid = (max(1, int(total_tiles)),)
    _complex_dx_persistent_kernel[grid](
        grad_out.contiguous(),
        coefficients.contiguous(),
        weight_experts.contiguous(),
        shared_weight.contiguous() if shared_weight is not None else _empty(grad_out.device, grad_out.dtype),
        grad_x,
        row_offsets,
        row_sizes,
        N_OUT=n_out,
        K_IN=k_in,
        NUM_EXPERTS=int(coefficients.shape[1]),
        NUM_GROUPS=len(split_sizes),
        TOTAL_TILES=int(total_tiles),
        TILES_K=tiles_k,
        HAS_SHARED_W=shared_weight is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return grad_x


def _launch_complex_dw_dc_atomic(
    x_pair: torch.Tensor,
    grad_out: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_weight):
        if _require_v2():
            raise RuntimeError("DPTB_TRITON_EXACT_GP_V2_REQUIRE=1 but complex v2 atomic backward needs CUDA float32 tensors and Triton.")
        _, grad_c, grad_w, grad_sw = _complex_torch_backward(x_pair, grad_out, coefficients, weight_experts, shared_weight, split_sizes)
        return grad_c, grad_w, grad_sw
    assert triton is not None
    row_offsets, row_sizes = _meta_tensors(split_sizes, x_pair.device)
    block_m = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M", 64)
    block_n = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N", 32)
    block_k = _env_int("DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K", 32)
    n_out = int(weight_experts.shape[1] // 2)
    k_in = int(weight_experts.shape[2])
    tiles_n = _cdiv(n_out, block_n)
    tiles_k = _cdiv(k_in, block_k)
    max_chunks = max(1, max(_cdiv(int(v), block_m) for v in split_sizes))
    grad_c = torch.zeros_like(coefficients)
    grad_w = torch.zeros_like(weight_experts)
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None
    grid = (len(split_sizes), int(coefficients.shape[1]), max_chunks * tiles_n * tiles_k)
    _complex_dw_dc_atomic_kernel[grid](
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


class _ExactMoELinearV2Fn(torch.autograd.Function):
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
        if use_exact_gp_v2():
            flat_out = _launch_real_fwd(flat_x, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, rows_per_group)
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
        use_cuda = use_exact_gp_v2() and _can_use_triton(flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg)
        if use_cuda:
            grad_x = _launch_real_dx(flat_grad, coefficients, weight_experts, shared_w_arg, ctx.rows_per_group)
            if os.environ.get("DPTB_TRITON_EXACT_GP_V2_BWD", "atomic").strip().lower() == "atomic":
                grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _launch_real_dw_dc_atomic(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
            else:
                _, grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
                    flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
                )
        else:
            grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _real_torch_backward(
                flat_x, flat_grad, coefficients, weight_experts, bias_arg, shared_w_arg, shared_b_arg, ctx.rows_per_group
            )
        return grad_x.reshape(ctx.orig_shape), grad_coeff, grad_w, grad_b, grad_sw, grad_sb, None


def exact_moe_linear_v2(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    bias_experts: Optional[torch.Tensor],
    shared_weight: Optional[torch.Tensor],
    shared_bias: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact MoE linear with optional Triton graph-persistent v2 kernels."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ExactMoELinearV2Fn.apply(
        x,
        coefficients,
        weight_experts,
        bias_experts if bias_experts is not None else _empty(x.device, x.dtype),
        shared_weight if shared_weight is not None else _empty(x.device, x.dtype),
        shared_bias if shared_bias is not None else _empty(x.device, x.dtype),
        split_sizes,
    )


class _ComplexExactMoELinearV2Fn(torch.autograd.Function):
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
        if use_complex_exact_gp_v2():
            out = _launch_complex_fwd(x_pair, coefficients, weight_experts, shared_w_arg, split_sizes)
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
        use_cuda = use_complex_exact_gp_v2() and _can_use_triton(x_pair, grad_out, coefficients, weight_experts, shared_w_arg)
        if use_cuda:
            grad_x = _launch_complex_dx(grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes)
            if os.environ.get("DPTB_TRITON_EXACT_GP_V2_BWD", "atomic").strip().lower() == "atomic":
                grad_coeff, grad_w, grad_sw = _launch_complex_dw_dc_atomic(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
            else:
                _, grad_coeff, grad_w, grad_sw = _complex_torch_backward(
                    x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
                )
        else:
            grad_x, grad_coeff, grad_w, grad_sw = _complex_torch_backward(
                x_pair, grad_out, coefficients, weight_experts, shared_w_arg, ctx.split_sizes
            )
        return grad_x, grad_coeff, grad_w, grad_sw, None


def complex_exact_moe_linear_v2(
    x_pair: torch.Tensor,
    coefficients: torch.Tensor,
    weight_experts: torch.Tensor,
    shared_weight: Optional[torch.Tensor],
    split_sizes: Sequence[int],
) -> torch.Tensor:
    """Drop-in exact complex MoE linear with optional Triton graph-persistent v2 kernels."""

    split_sizes = _canonical_split_sizes(split_sizes)
    return _ComplexExactMoELinearV2Fn.apply(
        x_pair,
        coefficients,
        weight_experts,
        shared_weight if shared_weight is not None else _empty(x_pair.device, x_pair.dtype),
        split_sizes,
    )
