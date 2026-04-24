
import math
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F



# Triton exact graph-persistent V4 overlay
try:
    from .so2_triton_exact_gp_v4 import (
        complex_exact_moe_linear_v4 as _complex_exact_moe_linear_v4,
        exact_moe_linear_v4 as _exact_moe_linear_v4,
        use_complex_exact_gp_v4 as _use_complex_exact_gp_v4,
        use_exact_gp_v4 as _use_exact_gp_v4,
    )
except Exception:  # pragma: no cover - additive experimental route must not break default imports
    _exact_moe_linear_v4 = None
    _complex_exact_moe_linear_v4 = None

    def _use_exact_gp_v4() -> bool:
        return os.environ.get("DPTB_TRITON_EXACT_GP_V4", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _use_complex_exact_gp_v4() -> bool:
        return os.environ.get(
            "DPTB_TRITON_COMPLEX_EXACT_GP_V4",
            os.environ.get("DPTB_TRITON_EXACT_GP_V4", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}

# Triton exact graph-persistent V3 overlay
try:
    from .so2_triton_exact_gp_v3 import (
        complex_exact_moe_linear_v3 as _complex_exact_moe_linear_v3,
        exact_moe_linear_v3 as _exact_moe_linear_v3,
        use_complex_exact_gp_v3 as _use_complex_exact_gp_v3,
        use_exact_gp_v3 as _use_exact_gp_v3,
    )
except Exception:  # pragma: no cover - additive experimental route must not break default imports
    _exact_moe_linear_v3 = None
    _complex_exact_moe_linear_v3 = None

    def _use_exact_gp_v3() -> bool:
        return os.environ.get("DPTB_TRITON_EXACT_GP_V3", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _use_complex_exact_gp_v3() -> bool:
        return os.environ.get(
            "DPTB_TRITON_COMPLEX_EXACT_GP_V3",
            os.environ.get("DPTB_TRITON_EXACT_GP_V3", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}

# Triton exact graph-persistent V2 overlay
try:
    from .so2_triton_exact_gp_v2 import (
        complex_exact_moe_linear_v2 as _complex_exact_moe_linear_v2,
        exact_moe_linear_v2 as _exact_moe_linear_v2,
        use_complex_exact_gp_v2 as _use_complex_exact_gp_v2,
        use_exact_gp_v2 as _use_exact_gp_v2,
    )
except Exception:  # pragma: no cover - additive experimental route must not break default imports
    _exact_moe_linear_v2 = None
    _complex_exact_moe_linear_v2 = None

    def _use_exact_gp_v2() -> bool:
        return os.environ.get("DPTB_TRITON_EXACT_GP_V2", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _use_complex_exact_gp_v2() -> bool:
        return os.environ.get(
            "DPTB_TRITON_COMPLEX_EXACT_GP_V2",
            os.environ.get("DPTB_TRITON_EXACT_GP_V2", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}

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


def _enable_fused_expert_triton() -> bool:
    return os.environ.get("DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT", "0") == "1"


def _enable_exact_graph_persistent_triton() -> bool:
    return os.environ.get("DPTB_TRITON_EXACT_USE_GRAPH_PERSISTENT", "0") == "1"


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


def _empty(device: torch.device, dtype: torch.dtype):
    return torch.empty(0, device=device, dtype=dtype)


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


def _mix_moe_weights_and_bias(coefficients: torch.Tensor,
                              weight_experts: torch.Tensor,
                              bias_experts: Optional[torch.Tensor],
                              shared_weight: Optional[torch.Tensor],
                              shared_bias: Optional[torch.Tensor]):
    num_graphs = int(coefficients.shape[0])
    mixed_weights = coefficients.matmul(weight_experts.reshape(weight_experts.shape[0], -1))
    mixed_weights = mixed_weights.reshape(num_graphs, weight_experts.shape[1], weight_experts.shape[2])
    if shared_weight is not None:
        mixed_weights = mixed_weights + shared_weight.unsqueeze(0)

    mixed_bias = None
    if bias_experts is not None:
        mixed_bias = coefficients.matmul(bias_experts)
        if shared_bias is not None:
            mixed_bias = mixed_bias + shared_bias.unsqueeze(0)

    return mixed_weights.contiguous(), mixed_bias.contiguous() if mixed_bias is not None else None


def _torch_grouped_linear_dw_db(flat_x: torch.Tensor,
                                flat_grad: torch.Tensor,
                                rows_per_group: Sequence[int],
                                *,
                                has_bias: bool):
    num_groups = len(rows_per_group)
    grad_w = flat_grad.new_zeros((num_groups, flat_grad.shape[1], flat_x.shape[1]))
    grad_b = flat_grad.new_zeros((num_groups, flat_grad.shape[1])) if has_bias else None
    start = 0
    acc_dtype = torch.float32 if flat_grad.dtype in (torch.float16, torch.bfloat16) else flat_grad.dtype
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xg = flat_x[start:start + rows].to(acc_dtype)
        gog = flat_grad[start:start + rows].to(acc_dtype)
        grad_w[g] = gog.transpose(0, 1).matmul(xg).to(grad_w.dtype)
        if grad_b is not None:
            grad_b[g] = gog.sum(dim=0).to(grad_b.dtype)
        start += rows
    return grad_w, grad_b


def _mixed_grads_to_moe_grads(grad_mixed_w: torch.Tensor,
                              grad_mixed_b: Optional[torch.Tensor],
                              coefficients: torch.Tensor,
                              weight_experts: torch.Tensor,
                              bias_experts: Optional[torch.Tensor],
                              shared_weight: Optional[torch.Tensor],
                              shared_bias: Optional[torch.Tensor]):
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
    return grad_coeff, grad_w, grad_b, grad_sw, grad_sb


def _torch_grouped_moe_forward(flat_x: torch.Tensor,
                               coefficients: torch.Tensor,
                               weight_experts: torch.Tensor,
                               bias_experts: Optional[torch.Tensor],
                               shared_weight: Optional[torch.Tensor],
                               shared_bias: Optional[torch.Tensor],
                               rows_per_group: Sequence[int]) -> torch.Tensor:
    out_parts = []
    start = 0
    acc_dtype = torch.float32 if flat_x.dtype in (torch.float16, torch.bfloat16) else flat_x.dtype
    weight_acc = weight_experts.to(acc_dtype)
    bias_acc = bias_experts.to(acc_dtype) if bias_experts is not None else None
    shared_weight_acc = shared_weight.to(acc_dtype) if shared_weight is not None else None
    shared_bias_acc = shared_bias.to(acc_dtype) if shared_bias is not None else None

    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xg = flat_x[start:start + rows].to(acc_dtype)
        coeff_g = coefficients[g].to(acc_dtype)
        mixed_w = torch.einsum("e,eoi->oi", coeff_g, weight_acc)
        if shared_weight_acc is not None:
            mixed_w = mixed_w + shared_weight_acc

        mixed_b = None
        if bias_acc is not None:
            mixed_b = torch.einsum("e,eo->o", coeff_g, bias_acc)
            if shared_bias_acc is not None:
                mixed_b = mixed_b + shared_bias_acc

        out_parts.append(F.linear(xg, mixed_w, mixed_b).to(flat_x.dtype))
        start += rows

    return torch.cat(out_parts, dim=0)


def _torch_grouped_moe_dx(flat_grad: torch.Tensor,
                          coefficients: torch.Tensor,
                          weight_experts: torch.Tensor,
                          shared_weight: Optional[torch.Tensor],
                          rows_per_group: Sequence[int]) -> torch.Tensor:
    grad_parts = []
    start = 0
    acc_dtype = torch.float32 if flat_grad.dtype in (torch.float16, torch.bfloat16) else flat_grad.dtype
    weight_acc = weight_experts.to(acc_dtype)
    shared_weight_acc = shared_weight.to(acc_dtype) if shared_weight is not None else None

    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        gog = flat_grad[start:start + rows].to(acc_dtype)
        coeff_g = coefficients[g].to(acc_dtype)
        mixed_w = torch.einsum("e,eoi->oi", coeff_g, weight_acc)
        if shared_weight_acc is not None:
            mixed_w = mixed_w + shared_weight_acc
        grad_parts.append(gog.matmul(mixed_w).to(flat_grad.dtype))
        start += rows

    return torch.cat(grad_parts, dim=0)


def _torch_grouped_moe_dw_db_dc(flat_x: torch.Tensor,
                                flat_grad: torch.Tensor,
                                coefficients: torch.Tensor,
                                weight_experts: torch.Tensor,
                                bias_experts: Optional[torch.Tensor],
                                shared_weight: Optional[torch.Tensor],
                                shared_bias: Optional[torch.Tensor],
                                rows_per_group: Sequence[int]):
    grad_coeff = torch.zeros_like(coefficients)
    grad_w = torch.zeros_like(weight_experts)
    grad_b = torch.zeros_like(bias_experts) if bias_experts is not None else None
    grad_sw = torch.zeros_like(shared_weight) if shared_weight is not None else None
    grad_sb = torch.zeros_like(shared_bias) if shared_bias is not None else None

    start = 0
    acc_dtype = torch.float32 if flat_grad.dtype in (torch.float16, torch.bfloat16) else flat_grad.dtype
    weight_acc = weight_experts.to(acc_dtype)
    bias_acc = bias_experts.to(acc_dtype) if bias_experts is not None else None

    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xg = flat_x[start:start + rows].to(acc_dtype)
        gog = flat_grad[start:start + rows].to(acc_dtype)
        coeff_g = coefficients[g].to(acc_dtype)

        grad_base_w = gog.transpose(0, 1).matmul(xg)
        grad_w += (coeff_g[:, None, None] * grad_base_w.unsqueeze(0)).to(grad_w.dtype)
        if grad_b is not None:
            grad_base_b = gog.sum(dim=0)
            grad_b += (coeff_g[:, None] * grad_base_b.unsqueeze(0)).to(grad_b.dtype)
        if grad_sw is not None:
            grad_sw += grad_base_w.to(grad_sw.dtype)
        if grad_sb is not None:
            grad_sb += gog.sum(dim=0).to(grad_sb.dtype)

        acts = torch.einsum("mi,eoi->meo", xg, weight_acc)
        if bias_acc is not None:
            acts = acts + bias_acc.unsqueeze(0)
        grad_coeff[g] = torch.einsum("meo,mo->e", acts, gog).to(grad_coeff.dtype)
        start += rows

    return grad_coeff, grad_w, grad_b, grad_sw, grad_sb


def _torch_grouped_complex_forward(x_pair: torch.Tensor,
                                   mixed_weights: torch.Tensor,
                                   rows_per_group: Sequence[int]) -> torch.Tensor:
    cout = int(mixed_weights.shape[1] // 2)
    out = x_pair.new_empty((x_pair.shape[0], 2, cout))
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xr = x_pair[start:start + rows, 0, :]
        xi = x_pair[start:start + rows, 1, :]
        wr = mixed_weights[g, :cout, :]
        wi = mixed_weights[g, cout:, :]
        out[start:start + rows, 0, :] = xr.matmul(wr.transpose(0, 1)) - xi.matmul(wi.transpose(0, 1))
        out[start:start + rows, 1, :] = xr.matmul(wi.transpose(0, 1)) + xi.matmul(wr.transpose(0, 1))
        start += rows
    return out


def _torch_grouped_complex_dx(grad_out: torch.Tensor,
                              mixed_weights: torch.Tensor,
                              rows_per_group: Sequence[int]) -> torch.Tensor:
    cin = int(mixed_weights.shape[2])
    cout = int(mixed_weights.shape[1] // 2)
    grad_x = grad_out.new_empty((grad_out.shape[0], 2, cin))
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        gyr = grad_out[start:start + rows, 0, :]
        gyi = grad_out[start:start + rows, 1, :]
        wr = mixed_weights[g, :cout, :]
        wi = mixed_weights[g, cout:, :]
        grad_x[start:start + rows, 0, :] = gyr.matmul(wr) + gyi.matmul(wi)
        grad_x[start:start + rows, 1, :] = -gyr.matmul(wi) + gyi.matmul(wr)
        start += rows
    return grad_x


def _torch_grouped_complex_dw(x_pair: torch.Tensor,
                              grad_out: torch.Tensor,
                              mixed_weights: torch.Tensor,
                              rows_per_group: Sequence[int]) -> torch.Tensor:
    grad_w = torch.zeros_like(mixed_weights)
    cout = int(mixed_weights.shape[1] // 2)
    start = 0
    acc_dtype = torch.float32 if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out.dtype
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xr = x_pair[start:start + rows, 0, :].to(acc_dtype)
        xi = x_pair[start:start + rows, 1, :].to(acc_dtype)
        gyr = grad_out[start:start + rows, 0, :].to(acc_dtype)
        gyi = grad_out[start:start + rows, 1, :].to(acc_dtype)
        grad_wr = gyr.transpose(0, 1).matmul(xr) + gyi.transpose(0, 1).matmul(xi)
        grad_wi = -gyr.transpose(0, 1).matmul(xi) + gyi.transpose(0, 1).matmul(xr)
        grad_w[g, :cout, :] = grad_wr.to(grad_w.dtype)
        grad_w[g, cout:, :] = grad_wi.to(grad_w.dtype)
        start += rows
    return grad_w


def _torch_grouped_complex_moe_forward(x_pair: torch.Tensor,
                                       coefficients: torch.Tensor,
                                       weight_experts: torch.Tensor,
                                       rows_per_group: Sequence[int]) -> torch.Tensor:
    cout = int(weight_experts.shape[1] // 2)
    out = x_pair.new_empty((x_pair.shape[0], 2, cout))
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xr = x_pair[start:start + rows, 0, :]
        xi = x_pair[start:start + rows, 1, :]
        mixed = torch.einsum("e,eoi->oi", coefficients[g], weight_experts)
        wr = mixed[:cout, :]
        wi = mixed[cout:, :]
        out[start:start + rows, 0, :] = xr.matmul(wr.transpose(0, 1)) - xi.matmul(wi.transpose(0, 1))
        out[start:start + rows, 1, :] = xr.matmul(wi.transpose(0, 1)) + xi.matmul(wr.transpose(0, 1))
        start += rows
    return out


def _torch_grouped_complex_moe_dx(grad_out: torch.Tensor,
                                  coefficients: torch.Tensor,
                                  weight_experts: torch.Tensor,
                                  rows_per_group: Sequence[int]) -> torch.Tensor:
    cin = int(weight_experts.shape[2])
    cout = int(weight_experts.shape[1] // 2)
    grad_x = grad_out.new_empty((grad_out.shape[0], 2, cin))
    start = 0
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        gyr = grad_out[start:start + rows, 0, :]
        gyi = grad_out[start:start + rows, 1, :]
        mixed = torch.einsum("e,eoi->oi", coefficients[g], weight_experts)
        wr = mixed[:cout, :]
        wi = mixed[cout:, :]
        grad_x[start:start + rows, 0, :] = gyr.matmul(wr) + gyi.matmul(wi)
        grad_x[start:start + rows, 1, :] = -gyr.matmul(wi) + gyi.matmul(wr)
        start += rows
    return grad_x


def _torch_grouped_complex_moe_dw_dc(x_pair: torch.Tensor,
                                     grad_out: torch.Tensor,
                                     coefficients: torch.Tensor,
                                     weight_experts: torch.Tensor,
                                     rows_per_group: Sequence[int]):
    grad_w = torch.zeros_like(weight_experts)
    grad_c = torch.zeros_like(coefficients)
    cout = int(weight_experts.shape[1] // 2)
    start = 0
    acc_dtype = torch.float32 if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out.dtype
    for g, rows in enumerate(rows_per_group):
        rows = int(rows)
        xr = x_pair[start:start + rows, 0, :].to(acc_dtype)
        xi = x_pair[start:start + rows, 1, :].to(acc_dtype)
        gyr = grad_out[start:start + rows, 0, :].to(acc_dtype)
        gyi = grad_out[start:start + rows, 1, :].to(acc_dtype)
        grad_wr = gyr.transpose(0, 1).matmul(xr) + gyi.transpose(0, 1).matmul(xi)
        grad_wi = -gyr.transpose(0, 1).matmul(xi) + gyi.transpose(0, 1).matmul(xr)
        coeff_g = coefficients[g].to(acc_dtype)
        for e in range(int(coefficients.shape[1])):
            grad_w[e, :cout, :] += (coeff_g[e] * grad_wr).to(grad_w.dtype)
            grad_w[e, cout:, :] += (coeff_g[e] * grad_wi).to(grad_w.dtype)

            wr = weight_experts[e, :cout, :].to(acc_dtype)
            wi = weight_experts[e, cout:, :].to(acc_dtype)
            yr = xr.matmul(wr.transpose(0, 1)) - xi.matmul(wi.transpose(0, 1))
            yi = xr.matmul(wi.transpose(0, 1)) + xi.matmul(wr.transpose(0, 1))
            grad_c[g, e] = (gyr * yr + gyi * yi).sum().to(grad_c.dtype)
        start += rows
    return grad_w, grad_c


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

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN", "HAS_BIAS"],
    )
    @triton.jit
    def _grouped_linear_dw_kernel(
        x_ptr,
        grad_ptr,
        grad_w_ptr,
        grad_b_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        stride_xm,
        stride_xk,
        stride_gm,
        stride_gn,
        stride_wg,
        stride_wn,
        stride_wk,
        stride_bg,
        stride_bn,
        N_OUT,
        K_IN,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = offs_n < N_OUT
        mask_k = offs_k < K_IN

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        bacc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        m0 = 0
        while m0 < rows:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < rows
            grad = tl.load(
                grad_ptr + (row_start + offs_m[:, None]) * stride_gm + offs_n[None, :] * stride_gn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            x = tl.load(
                x_ptr + (row_start + offs_m[:, None]) * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc += tl.dot(tl.trans(grad), x, input_precision="ieee")
            if HAS_BIAS:
                bacc += tl.where(pid_k == 0, tl.sum(grad, axis=0), 0.0)
            m0 += BLOCK_M

        tl.store(
            grad_w_ptr + g * stride_wg + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            acc,
            mask=mask_n[:, None] & mask_k[None, :],
        )
        if HAS_BIAS:
            tl.store(
                grad_b_ptr + g * stride_bg + offs_n * stride_bn,
                bacc,
                mask=mask_n & (pid_k == 0),
            )

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN"],
    )
    @triton.jit
    def _grouped_complex_forward_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
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

            acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            k0 = 0
            while k0 < K_IN:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K_IN

                xr = tl.load(
                    x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                xi = tl.load(
                    x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + K_IN + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wr = tl.load(
                    w_ptr + g * (2 * N_OUT * K_IN) + (offs_n[:, None] * K_IN) + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wi = tl.load(
                    w_ptr + g * (2 * N_OUT * K_IN) + ((N_OUT + offs_n[:, None]) * K_IN) + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc_r += tl.dot(xr, tl.trans(wr), input_precision="ieee")
                acc_r -= tl.dot(xi, tl.trans(wi), input_precision="ieee")
                acc_i += tl.dot(xr, tl.trans(wi), input_precision="ieee")
                acc_i += tl.dot(xi, tl.trans(wr), input_precision="ieee")
                k0 += BLOCK_K

            tl.store(
                y_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + offs_n[None, :],
                acc_r,
                mask=mask_m[:, None] & mask_n[None, :],
            )
            tl.store(
                y_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + N_OUT + offs_n[None, :],
                acc_i,
                mask=mask_m[:, None] & mask_n[None, :],
            )
            tile_id += num_programs

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN"],
    )
    @triton.jit
    def _grouped_complex_dx_kernel(
        gy_ptr,
        w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        tiles_k = tl.cdiv(K_IN, BLOCK_K)
        total_tiles = 0
        for g_scan in range(num_groups):
            rows_scan = tl.load(row_sizes_ptr + g_scan)
            total_tiles += tl.cdiv(rows_scan, BLOCK_M) * tiles_k

        tile_id = pid
        while tile_id < total_tiles:
            remaining = tile_id
            selected_g = tl.full((), 0, tl.int32)
            selected_remaining = remaining
            found = tl.full((), False, tl.int1)
            for g_scan in range(num_groups):
                rows_g = tl.load(row_sizes_ptr + g_scan)
                tiles_g = tl.cdiv(rows_g, BLOCK_M) * tiles_k
                take = (remaining < tiles_g) & (~found)
                selected_g = tl.where(take, g_scan, selected_g)
                selected_remaining = tl.where(take, remaining, selected_remaining)
                remaining = tl.where(found | take, remaining, remaining - tiles_g)
                found = found | take

            g = selected_g
            remaining = selected_remaining
            rows = tl.load(row_sizes_ptr + g)
            row_start = tl.load(row_offsets_ptr + g)
            pid_m = remaining // tiles_k
            pid_k = remaining % tiles_k

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_m = offs_m < rows
            mask_k = offs_k < K_IN

            acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            n0 = 0
            while n0 < N_OUT:
                offs_n = n0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_OUT
                gyr = tl.load(
                    gy_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                )
                gyi = tl.load(
                    gy_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + N_OUT + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                )
                wr = tl.load(
                    w_ptr + g * (2 * N_OUT * K_IN) + (offs_n[:, None] * K_IN) + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wi = tl.load(
                    w_ptr + g * (2 * N_OUT * K_IN) + ((N_OUT + offs_n[:, None]) * K_IN) + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc_r += tl.dot(gyr, wr, input_precision="ieee")
                acc_r += tl.dot(gyi, wi, input_precision="ieee")
                acc_i -= tl.dot(gyr, wi, input_precision="ieee")
                acc_i += tl.dot(gyi, wr, input_precision="ieee")
                n0 += BLOCK_N

            tl.store(
                gx_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + offs_k[None, :],
                acc_r,
                mask=mask_m[:, None] & mask_k[None, :],
            )
            tl.store(
                gx_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + K_IN + offs_k[None, :],
                acc_i,
                mask=mask_m[:, None] & mask_k[None, :],
            )
            tile_id += num_programs

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN"],
    )
    @triton.jit
    def _grouped_complex_dw_kernel(
        x_ptr,
        grad_ptr,
        grad_w_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        N_OUT,
        K_IN,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = offs_n < N_OUT
        mask_k = offs_k < K_IN

        acc_r = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        m0 = 0
        while m0 < rows:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < rows
            xr = tl.load(
                x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            xi = tl.load(
                x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + K_IN + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            gyr = tl.load(
                grad_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            gyi = tl.load(
                grad_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + N_OUT + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )

            acc_r += tl.dot(tl.trans(gyr), xr, input_precision="ieee")
            acc_r += tl.dot(tl.trans(gyi), xi, input_precision="ieee")
            acc_i -= tl.dot(tl.trans(gyr), xi, input_precision="ieee")
            acc_i += tl.dot(tl.trans(gyi), xr, input_precision="ieee")
            m0 += BLOCK_M

        tl.store(
            grad_w_ptr + g * (2 * N_OUT * K_IN) + (offs_n[:, None] * K_IN) + offs_k[None, :],
            acc_r,
            mask=mask_n[:, None] & mask_k[None, :],
        )
        tl.store(
            grad_w_ptr + g * (2 * N_OUT * K_IN) + ((N_OUT + offs_n[:, None]) * K_IN) + offs_k[None, :],
            acc_i,
            mask=mask_n[:, None] & mask_k[None, :],
        )

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN", "NUM_EXPERTS"],
    )
    @triton.jit
    def _grouped_complex_moe_forward_kernel(
        x_ptr,
        coeff_ptr,
        w_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
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

            acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            k0 = 0
            while k0 < K_IN:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K_IN
                xr = tl.load(
                    x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                xi = tl.load(
                    x_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + K_IN + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )

                wr_mix = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                wi_mix = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                for e in range(NUM_EXPERTS):
                    coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                    wr = tl.load(
                        w_ptr + e * (2 * N_OUT * K_IN) + (offs_n[:, None] * K_IN) + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    wi = tl.load(
                        w_ptr + e * (2 * N_OUT * K_IN) + ((N_OUT + offs_n[:, None]) * K_IN) + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    wr_mix += coeff * wr
                    wi_mix += coeff * wi
                acc_r += tl.dot(xr, tl.trans(wr_mix), input_precision="ieee")
                acc_r -= tl.dot(xi, tl.trans(wi_mix), input_precision="ieee")
                acc_i += tl.dot(xr, tl.trans(wi_mix), input_precision="ieee")
                acc_i += tl.dot(xi, tl.trans(wr_mix), input_precision="ieee")
                k0 += BLOCK_K

            tl.store(
                y_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + offs_n[None, :],
                acc_r,
                mask=mask_m[:, None] & mask_n[None, :],
            )
            tl.store(
                y_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + N_OUT + offs_n[None, :],
                acc_i,
                mask=mask_m[:, None] & mask_n[None, :],
            )
            tile_id += num_programs

    @triton.autotune(
        configs=_DEFAULT_CONFIGS,
        key=["N_OUT", "K_IN", "NUM_EXPERTS"],
    )
    @triton.jit
    def _grouped_complex_moe_dx_kernel(
        gy_ptr,
        coeff_ptr,
        w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        tiles_k = tl.cdiv(K_IN, BLOCK_K)
        total_tiles = 0
        for g_scan in range(num_groups):
            rows_scan = tl.load(row_sizes_ptr + g_scan)
            total_tiles += tl.cdiv(rows_scan, BLOCK_M) * tiles_k

        tile_id = pid
        while tile_id < total_tiles:
            remaining = tile_id
            selected_g = tl.full((), 0, tl.int32)
            selected_remaining = remaining
            found = tl.full((), False, tl.int1)
            for g_scan in range(num_groups):
                rows_g = tl.load(row_sizes_ptr + g_scan)
                tiles_g = tl.cdiv(rows_g, BLOCK_M) * tiles_k
                take = (remaining < tiles_g) & (~found)
                selected_g = tl.where(take, g_scan, selected_g)
                selected_remaining = tl.where(take, remaining, selected_remaining)
                remaining = tl.where(found | take, remaining, remaining - tiles_g)
                found = found | take

            g = selected_g
            remaining = selected_remaining
            rows = tl.load(row_sizes_ptr + g)
            row_start = tl.load(row_offsets_ptr + g)
            pid_m = remaining // tiles_k
            pid_k = remaining % tiles_k

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_m = offs_m < rows
            mask_k = offs_k < K_IN

            acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            n0 = 0
            while n0 < N_OUT:
                offs_n = n0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N_OUT
                gyr = tl.load(
                    gy_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                )
                gyi = tl.load(
                    gy_ptr + ((row_start + offs_m[:, None]) * 2 * N_OUT) + N_OUT + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                )
                wr_mix = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                wi_mix = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                for e in range(NUM_EXPERTS):
                    coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                    wr = tl.load(
                        w_ptr + e * (2 * N_OUT * K_IN) + (offs_n[:, None] * K_IN) + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    wi = tl.load(
                        w_ptr + e * (2 * N_OUT * K_IN) + ((N_OUT + offs_n[:, None]) * K_IN) + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    wr_mix += coeff * wr
                    wi_mix += coeff * wi
                acc_r += tl.dot(gyr, wr_mix, input_precision="ieee")
                acc_r += tl.dot(gyi, wi_mix, input_precision="ieee")
                acc_i -= tl.dot(gyr, wi_mix, input_precision="ieee")
                acc_i += tl.dot(gyi, wr_mix, input_precision="ieee")
                n0 += BLOCK_N

            tl.store(
                gx_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + offs_k[None, :],
                acc_r,
                mask=mask_m[:, None] & mask_k[None, :],
            )
            tl.store(
                gx_ptr + ((row_start + offs_m[:, None]) * 2 * K_IN) + K_IN + offs_k[None, :],
                acc_i,
                mask=mask_m[:, None] & mask_k[None, :],
            )
            tile_id += num_programs

    @triton.jit
    def _grouped_moe_forward_kernel(
        x_ptr,
        coeff_ptr,
        w_ptr,
        bias_ptr,
        shared_w_ptr,
        shared_b_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SHARED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k0 = 0
        while k0 < K_IN:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_IN
            x = tl.load(
                x_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += coeff * w

            if HAS_SHARED:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw

            acc += tl.dot(x, tl.trans(mixed_w), input_precision="ieee")
            k0 += BLOCK_K

        if HAS_BIAS:
            mixed_b = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                b = tl.load(
                    bias_ptr + e * N_OUT + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                mixed_b += coeff * b
            if HAS_SHARED:
                sb = tl.load(shared_b_ptr + offs_n, mask=mask_n, other=0.0)
                mixed_b += sb
            acc += mixed_b[None, :]

        tl.store(
            y_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _grouped_moe_dx_kernel(
        gy_ptr,
        coeff_ptr,
        w_ptr,
        shared_w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
        HAS_SHARED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_k = offs_k < K_IN

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        n0 = 0
        while n0 < N_OUT:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_OUT
            gy = tl.load(
                gy_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )

            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += coeff * w
            if HAS_SHARED:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw
            acc += tl.dot(gy, mixed_w, input_precision="ieee")
            n0 += BLOCK_N

        tl.store(
            gx_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
            acc,
            mask=mask_m[:, None] & mask_k[None, :],
        )

    @triton.jit
    def _grouped_moe_forward_graph_persistent_kernel(
        x_ptr,
        coeff_ptr,
        w_ptr,
        bias_ptr,
        shared_w_ptr,
        shared_b_ptr,
        y_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SHARED: tl.constexpr,
        BLOCK_M_SUPER: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # One CTA owns a larger row supertile for a fixed graph/output tile.
        # The mixed W[n,k] tile is formed once and reused across all rows in
        # the supertile, avoiding the per-row-tile expert mixing overhead of the
        # older fused-expert prototype.
        g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_m * BLOCK_M_SUPER + tl.arange(0, BLOCK_M_SUPER)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < N_OUT

        acc = tl.zeros((BLOCK_M_SUPER, BLOCK_N), dtype=tl.float32)
        k0 = 0
        while k0 < K_IN:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_IN
            x = tl.load(
                x_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += coeff * w
            if HAS_SHARED:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw

            acc += tl.dot(x, tl.trans(mixed_w), input_precision="ieee")
            k0 += BLOCK_K

        if HAS_BIAS:
            mixed_b = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                b = tl.load(
                    bias_ptr + e * N_OUT + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                mixed_b += coeff * b
            if HAS_SHARED:
                sb = tl.load(shared_b_ptr + offs_n, mask=mask_n, other=0.0)
                mixed_b += sb
            acc += mixed_b[None, :]

        tl.store(
            y_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _grouped_moe_dx_graph_persistent_kernel(
        gy_ptr,
        coeff_ptr,
        w_ptr,
        shared_w_ptr,
        gx_ptr,
        row_offsets_ptr,
        row_sizes_ptr,
        num_groups: tl.constexpr,
        N_OUT,
        K_IN,
        NUM_EXPERTS: tl.constexpr,
        HAS_SHARED: tl.constexpr,
        BLOCK_M_SUPER: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        rows = tl.load(row_sizes_ptr + g)
        row_start = tl.load(row_offsets_ptr + g)
        offs_m = pid_m * BLOCK_M_SUPER + tl.arange(0, BLOCK_M_SUPER)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < rows
        mask_k = offs_k < K_IN

        acc = tl.zeros((BLOCK_M_SUPER, BLOCK_K), dtype=tl.float32)
        n0 = 0
        while n0 < N_OUT:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_OUT
            gy = tl.load(
                gy_ptr + (row_start + offs_m[:, None]) * N_OUT + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )

            mixed_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for e in range(NUM_EXPERTS):
                coeff = tl.load(coeff_ptr + g * NUM_EXPERTS + e).to(tl.float32)
                w = tl.load(
                    w_ptr + e * (N_OUT * K_IN) + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += coeff * w
            if HAS_SHARED:
                sw = tl.load(
                    shared_w_ptr + offs_n[:, None] * K_IN + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                mixed_w += sw

            acc += tl.dot(gy, mixed_w, input_precision="ieee")
            n0 += BLOCK_N

        tl.store(
            gx_ptr + (row_start + offs_m[:, None]) * K_IN + offs_k[None, :],
            acc,
            mask=mask_m[:, None] & mask_k[None, :],
        )


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


def _triton_grouped_linear_dw_db(flat_x: torch.Tensor,
                                 flat_grad: torch.Tensor,
                                 rows_per_group: Sequence[int],
                                 *,
                                 has_bias: bool):
    if (
        _force_disable_triton()
        or not _TRITON_AVAILABLE
        or flat_x.device.type != "cuda"
        or flat_grad.device.type != "cuda"
        or flat_x.dtype != torch.float32
        or flat_grad.dtype != flat_x.dtype
    ):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped linear reduce is unavailable.")
        return _torch_grouped_linear_dw_db(flat_x, flat_grad, rows_per_group, has_bias=has_bias)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    n_out = int(flat_grad.shape[1])
    k_in = int(flat_x.shape[1])
    grad_w = torch.empty((len(rows_per_group), n_out, k_in), device=flat_x.device, dtype=flat_x.dtype)
    grad_b = torch.empty((len(rows_per_group), n_out), device=flat_x.device, dtype=flat_x.dtype) if has_bias else None
    bias = grad_b if grad_b is not None else torch.empty((1, 1), device=flat_x.device, dtype=flat_x.dtype)

    def grid(meta):
        return (
            len(rows_per_group),
            triton.cdiv(n_out, meta["BLOCK_N"]),
            triton.cdiv(k_in, meta["BLOCK_K"]),
        )

    _grouped_linear_dw_kernel[grid](
        flat_x.contiguous(),
        flat_grad.contiguous(),
        grad_w,
        bias,
        row_offsets,
        row_sizes,
        flat_x.stride(0),
        flat_x.stride(1),
        flat_grad.stride(0),
        flat_grad.stride(1),
        grad_w.stride(0),
        grad_w.stride(1),
        grad_w.stride(2),
        bias.stride(0),
        bias.stride(1),
        n_out,
        k_in,
        HAS_BIAS=has_bias,
    )
    return grad_w, grad_b


def _triton_grouped_complex_forward(x_pair: torch.Tensor,
                                    mixed_weights: torch.Tensor,
                                    rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _use_triton_for_linear(x_pair, mixed_weights):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex backend is unavailable.")
        return _torch_grouped_complex_forward(x_pair, mixed_weights, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, x_pair.device)
    n_out = int(mixed_weights.shape[1] // 2)
    out = torch.empty((x_pair.shape[0], 2, n_out), device=x_pair.device, dtype=x_pair.dtype)

    num_sms = _get_num_sms(x_pair.device)
    cta_factor = int(os.environ.get("DPTB_TRITON_LINEAR_PERSISTENT_FACTOR", "2"))

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_n = meta["BLOCK_N"]
        tiles_n = triton.cdiv(n_out, block_n)
        total_tiles = sum(triton.cdiv(int(v), block_m) * tiles_n for v in rows_per_group)
        return (max(1, min(total_tiles, num_sms * cta_factor)),)

    _grouped_complex_forward_kernel[grid](
        x_pair,
        mixed_weights,
        out,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        int(x_pair.shape[2]),
    )
    return out


def _triton_grouped_complex_dx(grad_out: torch.Tensor,
                               mixed_weights: torch.Tensor,
                               rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _use_triton_for_linear(grad_out, mixed_weights):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex grad_x backend is unavailable.")
        return _torch_grouped_complex_dx(grad_out, mixed_weights, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, grad_out.device)
    cin = int(mixed_weights.shape[2])
    n_out = int(mixed_weights.shape[1] // 2)
    grad_x = torch.empty((grad_out.shape[0], 2, cin), device=grad_out.device, dtype=grad_out.dtype)

    num_sms = _get_num_sms(grad_out.device)
    cta_factor = int(os.environ.get("DPTB_TRITON_LINEAR_PERSISTENT_FACTOR", "2"))

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_k = meta["BLOCK_K"]
        tiles_k = triton.cdiv(cin, block_k)
        total_tiles = sum(triton.cdiv(int(v), block_m) * tiles_k for v in rows_per_group)
        return (max(1, min(total_tiles, num_sms * cta_factor)),)

    _grouped_complex_dx_kernel[grid](
        grad_out,
        mixed_weights,
        grad_x,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        cin,
    )
    return grad_x


def _triton_grouped_complex_dw(x_pair: torch.Tensor,
                               grad_out: torch.Tensor,
                               mixed_weights: torch.Tensor,
                               rows_per_group: Sequence[int]) -> torch.Tensor:
    if (
        not _use_triton_for_linear(x_pair, mixed_weights)
        or grad_out.device.type != "cuda"
        or grad_out.dtype != x_pair.dtype
    ):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex grad_weight backend is unavailable.")
        return _torch_grouped_complex_dw(x_pair, grad_out, mixed_weights, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, x_pair.device)
    n_out = int(mixed_weights.shape[1] // 2)
    cin = int(mixed_weights.shape[2])
    grad_w = torch.empty_like(mixed_weights)

    def grid(meta):
        return (
            len(rows_per_group),
            triton.cdiv(n_out, meta["BLOCK_N"]),
            triton.cdiv(cin, meta["BLOCK_K"]),
        )

    _grouped_complex_dw_kernel[grid](
        x_pair.contiguous(),
        grad_out.contiguous(),
        grad_w,
        row_offsets,
        row_sizes,
        n_out,
        cin,
    )
    return grad_w


def _triton_grouped_complex_moe_dw_dc(x_pair: torch.Tensor,
                                      grad_out: torch.Tensor,
                                      coefficients: torch.Tensor,
                                      weight_experts: torch.Tensor,
                                      rows_per_group: Sequence[int]):
    if (
        not _use_triton_for_linear(x_pair, weight_experts)
        or grad_out.device.type != "cuda"
        or coefficients.device.type != "cuda"
        or grad_out.dtype != x_pair.dtype
        or coefficients.dtype != x_pair.dtype
    ):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex MoE reduce backend is unavailable.")
        return _torch_grouped_complex_moe_dw_dc(x_pair, grad_out, coefficients, weight_experts, rows_per_group)

    mixed_weights, _ = _mix_moe_weights_and_bias(
        coefficients,
        weight_experts,
        None,
        None,
        None,
    )
    grad_mixed_w = _triton_grouped_complex_dw(x_pair, grad_out, mixed_weights, rows_per_group)
    grad_mixed_flat = grad_mixed_w.reshape(grad_mixed_w.shape[0], -1)
    weight_flat = weight_experts.reshape(weight_experts.shape[0], -1)
    grad_c = grad_mixed_flat.matmul(weight_flat.transpose(0, 1))
    grad_w = coefficients.transpose(0, 1).matmul(grad_mixed_flat).reshape_as(weight_experts)
    return grad_w, grad_c


def _triton_grouped_complex_moe_forward(x_pair: torch.Tensor,
                                        coefficients: torch.Tensor,
                                        weight_experts: torch.Tensor,
                                        rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _use_triton_for_linear(x_pair, weight_experts) or coefficients.device.type != "cuda" or coefficients.dtype != x_pair.dtype:
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex MoE backend is unavailable.")
        return _torch_grouped_complex_moe_forward(x_pair, coefficients, weight_experts, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, x_pair.device)
    n_out = int(weight_experts.shape[1] // 2)
    out = torch.empty((x_pair.shape[0], 2, n_out), device=x_pair.device, dtype=x_pair.dtype)

    num_sms = _get_num_sms(x_pair.device)
    cta_factor = int(os.environ.get("DPTB_TRITON_LINEAR_PERSISTENT_FACTOR", "2"))

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_n = meta["BLOCK_N"]
        tiles_n = triton.cdiv(n_out, block_n)
        total_tiles = sum(triton.cdiv(int(v), block_m) * tiles_n for v in rows_per_group)
        return (max(1, min(total_tiles, num_sms * cta_factor)),)

    _grouped_complex_moe_forward_kernel[grid](
        x_pair,
        coefficients,
        weight_experts,
        out,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        int(x_pair.shape[2]),
        NUM_EXPERTS=int(coefficients.shape[1]),
    )
    return out


def _triton_grouped_complex_moe_dx(grad_out: torch.Tensor,
                                  coefficients: torch.Tensor,
                                  weight_experts: torch.Tensor,
                                  rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _use_triton_for_linear(grad_out, weight_experts) or coefficients.device.type != "cuda" or coefficients.dtype != grad_out.dtype:
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped complex MoE grad_x backend is unavailable.")
        return _torch_grouped_complex_moe_dx(grad_out, coefficients, weight_experts, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, grad_out.device)
    cin = int(weight_experts.shape[2])
    n_out = int(weight_experts.shape[1] // 2)
    grad_x = torch.empty((grad_out.shape[0], 2, cin), device=grad_out.device, dtype=grad_out.dtype)

    num_sms = _get_num_sms(grad_out.device)
    cta_factor = int(os.environ.get("DPTB_TRITON_LINEAR_PERSISTENT_FACTOR", "2"))

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_k = meta["BLOCK_K"]
        tiles_k = triton.cdiv(cin, block_k)
        total_tiles = sum(triton.cdiv(int(v), block_m) * tiles_k for v in rows_per_group)
        return (max(1, min(total_tiles, num_sms * cta_factor)),)

    _grouped_complex_moe_dx_kernel[grid](
        grad_out,
        coefficients,
        weight_experts,
        grad_x,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        cin,
        NUM_EXPERTS=int(coefficients.shape[1]),
    )
    return grad_x


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

        grad_w, grad_b = _triton_grouped_linear_dw_db(
            flat_x,
            flat_grad,
            rows_per_group,
            has_bias=ctx.has_bias,
        )

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


def _triton_grouped_moe_forward(flat_x: torch.Tensor,
                                coefficients: torch.Tensor,
                                weight_experts: torch.Tensor,
                                bias_experts: Optional[torch.Tensor],
                                shared_weight: Optional[torch.Tensor],
                                shared_bias: Optional[torch.Tensor],
                                rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _enable_fused_expert_triton():
        if _require_triton():
            raise RuntimeError(
                "DPTB_TRITON_LINEAR_REQUIRE=1 but fused-expert Triton execution is disabled. "
                "Set DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT=1 only for isolated experiments."
            )
        return _torch_grouped_moe_forward(
            flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, rows_per_group
        )
    if (
        not _use_triton_for_linear(flat_x, weight_experts)
        or coefficients.device.type != "cuda"
        or coefficients.dtype != flat_x.dtype
    ):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped MoE backend is unavailable.")
        return _torch_grouped_moe_forward(
            flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, rows_per_group
        )

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    weight = weight_experts.contiguous()
    coeff = coefficients.contiguous()
    bias = bias_experts.contiguous() if bias_experts is not None else _empty(flat_x.device, flat_x.dtype)
    shared_w = shared_weight.contiguous() if shared_weight is not None else _empty(flat_x.device, flat_x.dtype)
    shared_b = shared_bias.contiguous() if shared_bias is not None else _empty(flat_x.device, flat_x.dtype)
    n_out = int(weight.shape[1])
    out = torch.empty((flat_x.shape[0], n_out), device=flat_x.device, dtype=flat_x.dtype)

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_n = meta["BLOCK_N"]
        tiles_n = triton.cdiv(n_out, block_n)
        max_tiles_m = max(triton.cdiv(int(v), block_m) for v in rows_per_group)
        return (len(rows_per_group), max_tiles_m, tiles_n)

    _grouped_moe_forward_kernel[grid](
        flat_x.contiguous(),
        coeff,
        weight,
        bias,
        shared_w,
        shared_b,
        out,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        int(flat_x.shape[1]),
        NUM_EXPERTS=int(coeff.shape[1]),
        HAS_BIAS=bias_experts is not None,
        HAS_SHARED=shared_weight is not None,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return out


def _triton_grouped_moe_dx(flat_grad: torch.Tensor,
                           coefficients: torch.Tensor,
                           weight_experts: torch.Tensor,
                           shared_weight: Optional[torch.Tensor],
                           rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _enable_fused_expert_triton():
        if _require_triton():
            raise RuntimeError(
                "DPTB_TRITON_LINEAR_REQUIRE=1 but fused-expert Triton grad_x execution is disabled. "
                "Set DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT=1 only for isolated experiments."
            )
        return _torch_grouped_moe_dx(flat_grad, coefficients, weight_experts, shared_weight, rows_per_group)
    if (
        not _use_triton_for_linear(flat_grad, weight_experts)
        or coefficients.device.type != "cuda"
        or coefficients.dtype != flat_grad.dtype
    ):
        if _require_triton():
            raise RuntimeError("DPTB_TRITON_LINEAR_REQUIRE=1 but Triton grouped MoE grad_x backend is unavailable.")
        return _torch_grouped_moe_dx(flat_grad, coefficients, weight_experts, shared_weight, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_grad.device)
    weight = weight_experts.contiguous()
    coeff = coefficients.contiguous()
    shared_w = shared_weight.contiguous() if shared_weight is not None else _empty(flat_grad.device, flat_grad.dtype)
    k_in = int(weight.shape[2])
    grad_x = torch.empty((flat_grad.shape[0], k_in), device=flat_grad.device, dtype=flat_grad.dtype)

    def grid(meta):
        block_m = meta["BLOCK_M"]
        block_k = meta["BLOCK_K"]
        tiles_k = triton.cdiv(k_in, block_k)
        max_tiles_m = max(triton.cdiv(int(v), block_m) for v in rows_per_group)
        return (len(rows_per_group), max_tiles_m, tiles_k)

    _grouped_moe_dx_kernel[grid](
        flat_grad.contiguous(),
        coeff,
        weight,
        shared_w,
        grad_x,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        int(flat_grad.shape[1]),
        k_in,
        NUM_EXPERTS=int(coeff.shape[1]),
        HAS_SHARED=shared_weight is not None,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return grad_x


def _triton_grouped_moe_forward_graph_persistent(flat_x: torch.Tensor,
                                                 coefficients: torch.Tensor,
                                                 weight_experts: torch.Tensor,
                                                 bias_experts: Optional[torch.Tensor],
                                                 shared_weight: Optional[torch.Tensor],
                                                 shared_bias: Optional[torch.Tensor],
                                                 rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _enable_exact_graph_persistent_triton():
        raise RuntimeError("graph-persistent exact MoE Triton path is not enabled")
    if (
        not _use_triton_for_linear(flat_x, weight_experts)
        or coefficients.device.type != "cuda"
        or coefficients.dtype != flat_x.dtype
    ):
        if _require_triton():
            raise RuntimeError(
                "DPTB_TRITON_LINEAR_REQUIRE=1 but graph-persistent exact MoE forward is unavailable."
            )
        return _torch_grouped_moe_forward(
            flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, rows_per_group
        )

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_x.device)
    weight = weight_experts.contiguous()
    coeff = coefficients.contiguous()
    bias = bias_experts.contiguous() if bias_experts is not None else _empty(flat_x.device, flat_x.dtype)
    shared_w = shared_weight.contiguous() if shared_weight is not None else _empty(flat_x.device, flat_x.dtype)
    shared_b = shared_bias.contiguous() if shared_bias is not None else _empty(flat_x.device, flat_x.dtype)
    n_out = int(weight.shape[1])
    out = torch.empty((flat_x.shape[0], n_out), device=flat_x.device, dtype=flat_x.dtype)
    block_m = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_M", "128"))
    block_n = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_N", "64"))
    block_k = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_K", "32"))

    grid = (
        len(rows_per_group),
        max(triton.cdiv(int(v), block_m) for v in rows_per_group),
        triton.cdiv(n_out, block_n),
    )
    _grouped_moe_forward_graph_persistent_kernel[grid](
        flat_x.contiguous(),
        coeff,
        weight,
        bias,
        shared_w,
        shared_b,
        out,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        n_out,
        int(flat_x.shape[1]),
        NUM_EXPERTS=int(coeff.shape[1]),
        HAS_BIAS=bias_experts is not None,
        HAS_SHARED=shared_weight is not None,
        BLOCK_M_SUPER=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _triton_grouped_moe_dx_graph_persistent(flat_grad: torch.Tensor,
                                            coefficients: torch.Tensor,
                                            weight_experts: torch.Tensor,
                                            shared_weight: Optional[torch.Tensor],
                                            rows_per_group: Sequence[int]) -> torch.Tensor:
    if not _enable_exact_graph_persistent_triton():
        raise RuntimeError("graph-persistent exact MoE Triton path is not enabled")
    if (
        not _use_triton_for_linear(flat_grad, weight_experts)
        or coefficients.device.type != "cuda"
        or coefficients.dtype != flat_grad.dtype
    ):
        if _require_triton():
            raise RuntimeError(
                "DPTB_TRITON_LINEAR_REQUIRE=1 but graph-persistent exact MoE grad_x is unavailable."
            )
        return _torch_grouped_moe_dx(flat_grad, coefficients, weight_experts, shared_weight, rows_per_group)

    row_offsets, row_sizes = _meta_tensors(rows_per_group, flat_grad.device)
    weight = weight_experts.contiguous()
    coeff = coefficients.contiguous()
    shared_w = shared_weight.contiguous() if shared_weight is not None else _empty(flat_grad.device, flat_grad.dtype)
    k_in = int(weight.shape[2])
    grad_x = torch.empty((flat_grad.shape[0], k_in), device=flat_grad.device, dtype=flat_grad.dtype)
    block_m = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_M", "128"))
    block_n = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_N", "64"))
    block_k = int(os.environ.get("DPTB_TRITON_EXACT_GP_BLOCK_K", "32"))

    grid = (
        len(rows_per_group),
        max(triton.cdiv(int(v), block_m) for v in rows_per_group),
        triton.cdiv(k_in, block_k),
    )
    _grouped_moe_dx_graph_persistent_kernel[grid](
        flat_grad.contiguous(),
        coeff,
        weight,
        shared_w,
        grad_x,
        row_offsets,
        row_sizes,
        len(rows_per_group),
        int(flat_grad.shape[1]),
        k_in,
        NUM_EXPERTS=int(coeff.shape[1]),
        HAS_SHARED=shared_weight is not None,
        BLOCK_M_SUPER=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return grad_x


class _GroupedMoEFusedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                coefficients: torch.Tensor,
                weight_experts: torch.Tensor,
                bias_experts: torch.Tensor,
                shared_weight: torch.Tensor,
                shared_bias: torch.Tensor,
                row_splits_tensor: torch.Tensor):
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        if int(sum(split_sizes)) != int(x.shape[0]):
            raise ValueError(
                f"split sizes sum to {sum(split_sizes)}, but x has {x.shape[0]} rows."
            )
        if coefficients.ndim != 2:
            raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
        if weight_experts.ndim != 3:
            raise ValueError(f"weight_experts must have shape [E, O, I], got {tuple(weight_experts.shape)}")
        if int(coefficients.shape[0]) != len(split_sizes):
            raise ValueError(
                f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {len(split_sizes)} groups."
            )
        if int(coefficients.shape[1]) != int(weight_experts.shape[0]):
            raise ValueError(
                f"coefficients has {coefficients.shape[1]} experts, but weight_experts has "
                f"{weight_experts.shape[0]} experts."
            )
        if int(weight_experts.shape[2]) != int(x.shape[-1]):
            raise ValueError(
                f"weight_experts in_features={weight_experts.shape[2]} does not match x last dim={x.shape[-1]}."
            )

        bias_arg = bias_experts if bias_experts.numel() > 0 else None
        shared_w_arg = shared_weight if shared_weight.numel() > 0 else None
        shared_b_arg = shared_bias if shared_bias.numel() > 0 else None
        flat_x, rows_per_group, _, _ = _flatten_grouped_rows(x, split_sizes)
        flat_out = _triton_grouped_moe_forward(
            flat_x,
            coefficients,
            weight_experts,
            bias_arg,
            shared_w_arg,
            shared_b_arg,
            rows_per_group,
        )

        ctx.save_for_backward(
            flat_x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            row_splits_tensor,
        )
        ctx.orig_shape = tuple(int(v) for v in x.shape)
        ctx.rows_per_group = rows_per_group
        ctx.has_bias = bias_arg is not None
        ctx.has_shared_weight = shared_w_arg is not None
        ctx.has_shared_bias = shared_b_arg is not None
        return flat_out.reshape(*x.shape[:-1], weight_experts.shape[1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        flat_x, coefficients, weight_experts, bias_experts, shared_weight, shared_bias, row_splits_tensor = ctx.saved_tensors
        bias_arg = bias_experts if ctx.has_bias else None
        shared_w_arg = shared_weight if ctx.has_shared_weight else None
        shared_b_arg = shared_bias if ctx.has_shared_bias else None
        flat_grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()

        grad_x = _triton_grouped_moe_dx(
            flat_grad,
            coefficients,
            weight_experts,
            shared_w_arg,
            ctx.rows_per_group,
        ).reshape(ctx.orig_shape)
        grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _torch_grouped_moe_dw_db_dc(
            flat_x,
            flat_grad,
            coefficients,
            weight_experts,
            bias_arg,
            shared_w_arg,
            shared_b_arg,
            ctx.rows_per_group,
        )

        return grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb, None


def grouped_moe_fused_linear(x: torch.Tensor,
                             coefficients: torch.Tensor,
                             weight_experts: torch.Tensor,
                             bias_experts: Optional[torch.Tensor],
                             shared_weight: Optional[torch.Tensor],
                             shared_bias: Optional[torch.Tensor],
                             split_sizes: Sequence[int]) -> torch.Tensor:
    row_splits_tensor = torch.tensor(_canonical_split_sizes(split_sizes), device=x.device, dtype=torch.long)
    return _GroupedMoEFusedLinearFn.apply(
        x,
        coefficients,
        weight_experts,
        bias_experts if bias_experts is not None else _empty(x.device, x.dtype),
        shared_weight if shared_weight is not None else _empty(x.device, x.dtype),
        shared_bias if shared_bias is not None else _empty(x.device, x.dtype),
        row_splits_tensor,
    )


class _GroupedExactMoELinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                coefficients: torch.Tensor,
                weight_experts: torch.Tensor,
                bias_experts: torch.Tensor,
                shared_weight: torch.Tensor,
                shared_bias: torch.Tensor,
                split_sizes: Tuple[int, ...]):
        if int(sum(split_sizes)) != int(x.shape[0]):
            raise ValueError(f"split sizes sum to {sum(split_sizes)}, but x has {x.shape[0]} rows.")
        if coefficients.ndim != 2:
            raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
        if weight_experts.ndim != 3:
            raise ValueError(f"weight_experts must have shape [E, O, I], got {tuple(weight_experts.shape)}")
        if int(coefficients.shape[0]) != len(split_sizes):
            raise ValueError(
                f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {len(split_sizes)} groups."
            )
        if int(coefficients.shape[1]) != int(weight_experts.shape[0]):
            raise ValueError(
                f"coefficients has {coefficients.shape[1]} experts, but weight_experts has "
                f"{weight_experts.shape[0]} experts."
            )
        if int(weight_experts.shape[2]) != int(x.shape[-1]):
            raise ValueError(
                f"weight_experts in_features={weight_experts.shape[2]} does not match x last dim={x.shape[-1]}."
            )

        bias_arg = bias_experts if bias_experts.numel() > 0 else None
        shared_w_arg = shared_weight if shared_weight.numel() > 0 else None
        shared_b_arg = shared_bias if shared_bias.numel() > 0 else None
        flat_x, rows_per_group, _, _ = _flatten_grouped_rows(x, split_sizes)

        if _enable_exact_graph_persistent_triton():
            flat_out = _triton_grouped_moe_forward_graph_persistent(
                flat_x,
                coefficients,
                weight_experts,
                bias_arg,
                shared_w_arg,
                shared_b_arg,
                rows_per_group,
            )
        else:
            mixed_weights, mixed_bias = _mix_moe_weights_and_bias(
                coefficients,
                weight_experts,
                bias_arg,
                shared_w_arg,
                shared_b_arg,
            )
            flat_out = _triton_grouped_linear_forward(
                flat_x,
                mixed_weights.transpose(1, 2).contiguous(),
                mixed_bias,
                rows_per_group,
            )

        # Do not save mixed_weights: backward recomputes it and only saves the
        # graph-level coefficients / expert banks plus x for grouped reduce.
        ctx.save_for_backward(
            flat_x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
        )
        ctx.orig_shape = tuple(int(v) for v in x.shape)
        ctx.rows_per_group = rows_per_group
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

        if _enable_exact_graph_persistent_triton():
            grad_x = _triton_grouped_moe_dx_graph_persistent(
                flat_grad,
                coefficients,
                weight_experts,
                shared_w_arg,
                ctx.rows_per_group,
            ).reshape(ctx.orig_shape)
        else:
            mixed_weights, _ = _mix_moe_weights_and_bias(
                coefficients,
                weight_experts,
                bias_arg,
                shared_w_arg,
                shared_b_arg,
            )
            grad_x = _triton_grouped_linear_forward(
                flat_grad,
                mixed_weights.contiguous(),
                None,
                ctx.rows_per_group,
            ).reshape(ctx.orig_shape)

        grad_mixed_w, grad_mixed_b = _triton_grouped_linear_dw_db(
            flat_x,
            flat_grad,
            ctx.rows_per_group,
            has_bias=ctx.has_bias,
        )
        grad_coeff, grad_w, grad_b, grad_sw, grad_sb = _mixed_grads_to_moe_grads(
            grad_mixed_w,
            grad_mixed_b,
            coefficients,
            weight_experts,
            bias_arg,
            shared_w_arg,
            shared_b_arg,
        )

        return grad_x, grad_coeff, grad_w, grad_b, grad_sw, grad_sb, None


def grouped_exact_moe_linear(x: torch.Tensor,
                             coefficients: torch.Tensor,
                             weight_experts: torch.Tensor,
                             bias_experts: Optional[torch.Tensor],
                             shared_weight: Optional[torch.Tensor],
                             shared_bias: Optional[torch.Tensor],
                             split_sizes: Sequence[int]) -> torch.Tensor:
    split_sizes = _canonical_split_sizes(split_sizes)
    if _use_exact_gp_v4():
        if _exact_moe_linear_v4 is None:
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4 is enabled, but dptb.nn.so2_triton_exact_gp_v4 could not be imported."
            )
        return _exact_moe_linear_v4(
            x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            split_sizes,
        )
    if _use_exact_gp_v3():
        if _exact_moe_linear_v3 is None:
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V3 is enabled, but dptb.nn.so2_triton_exact_gp_v3 could not be imported."
            )
        return _exact_moe_linear_v3(
            x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            split_sizes,
        )
    if _use_exact_gp_v2():
        if _exact_moe_linear_v2 is None:
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V2 is enabled, but dptb.nn.so2_triton_exact_gp_v2 could not be imported."
            )
        return _exact_moe_linear_v2(
            x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            split_sizes,
        )
    return _GroupedExactMoELinearFn.apply(
        x,
        coefficients,
        weight_experts,
        bias_experts if bias_experts is not None else _empty(x.device, x.dtype),
        shared_weight if shared_weight is not None else _empty(x.device, x.dtype),
        shared_bias if shared_bias is not None else _empty(x.device, x.dtype),
        split_sizes,
    )


class _GroupedComplexLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x_pair: torch.Tensor,
                mixed_weights: torch.Tensor,
                row_splits_tensor: torch.Tensor):
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        if x_pair.ndim != 3 or int(x_pair.shape[1]) != 2:
            raise ValueError(f"x_pair must have shape [N, 2, C], got {tuple(x_pair.shape)}")
        if int(sum(split_sizes)) != int(x_pair.shape[0]):
            raise ValueError(
                f"split sizes sum to {sum(split_sizes)}, but x_pair has {x_pair.shape[0]} rows."
            )
        x_pair = x_pair.contiguous()
        mixed_weights = mixed_weights.contiguous()
        out = _triton_grouped_complex_forward(x_pair, mixed_weights, split_sizes)
        ctx.save_for_backward(x_pair, mixed_weights, row_splits_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_pair, mixed_weights, row_splits_tensor = ctx.saved_tensors
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        grad_out = grad_out.contiguous()
        grad_x = _triton_grouped_complex_dx(grad_out, mixed_weights, split_sizes)
        grad_w = _triton_grouped_complex_dw(x_pair, grad_out, mixed_weights, split_sizes)
        return grad_x, grad_w, None


def grouped_complex_linear(x_pair: torch.Tensor,
                           mixed_weights: torch.Tensor,
                           split_sizes: Sequence[int]) -> torch.Tensor:
    row_splits_tensor = torch.tensor(_canonical_split_sizes(split_sizes), device=x_pair.device, dtype=torch.long)
    return _GroupedComplexLinearFn.apply(x_pair, mixed_weights, row_splits_tensor)


class _GroupedComplexMoEFusedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x_pair: torch.Tensor,
                coefficients: torch.Tensor,
                weight_experts: torch.Tensor,
                row_splits_tensor: torch.Tensor):
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        if x_pair.ndim != 3 or int(x_pair.shape[1]) != 2:
            raise ValueError(f"x_pair must have shape [N, 2, C], got {tuple(x_pair.shape)}")
        if int(sum(split_sizes)) != int(x_pair.shape[0]):
            raise ValueError(
                f"split sizes sum to {sum(split_sizes)}, but x_pair has {x_pair.shape[0]} rows."
            )
        if coefficients.ndim != 2:
            raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
        if weight_experts.ndim != 3:
            raise ValueError(f"weight_experts must have shape [E, 2*Cout, Cin], got {tuple(weight_experts.shape)}")
        if int(coefficients.shape[0]) != len(split_sizes):
            raise ValueError(
                f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {len(split_sizes)} groups."
            )
        if int(coefficients.shape[1]) != int(weight_experts.shape[0]):
            raise ValueError(
                f"coefficients has {coefficients.shape[1]} experts, but weight_experts has {weight_experts.shape[0]} experts."
            )
        if int(weight_experts.shape[1]) % 2 != 0:
            raise ValueError("weight_experts second dimension must be 2*Cout.")
        if int(weight_experts.shape[2]) != int(x_pair.shape[2]):
            raise ValueError(
                f"weight_experts Cin={weight_experts.shape[2]} does not match x_pair Cin={x_pair.shape[2]}."
            )

        x_pair = x_pair.contiguous()
        coefficients = coefficients.contiguous()
        weight_experts = weight_experts.contiguous()
        out = _triton_grouped_complex_moe_forward(x_pair, coefficients, weight_experts, split_sizes)
        ctx.save_for_backward(x_pair, coefficients, weight_experts, row_splits_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_pair, coefficients, weight_experts, row_splits_tensor = ctx.saved_tensors
        split_sizes = tuple(int(v) for v in row_splits_tensor.detach().cpu().tolist())
        grad_out = grad_out.contiguous()
        grad_x = _triton_grouped_complex_moe_dx(grad_out, coefficients, weight_experts, split_sizes)
        grad_w, grad_c = _triton_grouped_complex_moe_dw_dc(x_pair, grad_out, coefficients, weight_experts, split_sizes)
        return grad_x, grad_c, grad_w, None


def grouped_complex_moe_fused_linear(x_pair: torch.Tensor,
                                     coefficients: torch.Tensor,
                                     weight_experts: torch.Tensor,
                                     split_sizes: Sequence[int]) -> torch.Tensor:
    row_splits_tensor = torch.tensor(_canonical_split_sizes(split_sizes), device=x_pair.device, dtype=torch.long)
    return _GroupedComplexMoEFusedLinearFn.apply(x_pair, coefficients, weight_experts, row_splits_tensor)


class _GroupedComplexExactMoELinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x_pair: torch.Tensor,
                coefficients: torch.Tensor,
                weight_experts: torch.Tensor,
                shared_weight: torch.Tensor,
                split_sizes: Tuple[int, ...]):
        if x_pair.ndim != 3 or int(x_pair.shape[1]) != 2:
            raise ValueError(f"x_pair must have shape [N, 2, C], got {tuple(x_pair.shape)}")
        if int(sum(split_sizes)) != int(x_pair.shape[0]):
            raise ValueError(
                f"split sizes sum to {sum(split_sizes)}, but x_pair has {x_pair.shape[0]} rows."
            )
        if coefficients.ndim != 2:
            raise ValueError(f"coefficients must have shape [G, E], got {tuple(coefficients.shape)}")
        if weight_experts.ndim != 3:
            raise ValueError(f"weight_experts must have shape [E, 2*Cout, Cin], got {tuple(weight_experts.shape)}")
        if int(coefficients.shape[0]) != len(split_sizes):
            raise ValueError(
                f"coefficients has {coefficients.shape[0]} groups, but split_sizes has {len(split_sizes)} groups."
            )
        if int(coefficients.shape[1]) != int(weight_experts.shape[0]):
            raise ValueError(
                f"coefficients has {coefficients.shape[1]} experts, but weight_experts has {weight_experts.shape[0]} experts."
            )
        if int(weight_experts.shape[1]) % 2 != 0:
            raise ValueError("weight_experts second dimension must be 2*Cout.")
        if int(weight_experts.shape[2]) != int(x_pair.shape[2]):
            raise ValueError(
                f"weight_experts Cin={weight_experts.shape[2]} does not match x_pair Cin={x_pair.shape[2]}."
            )

        shared_w_arg = shared_weight if shared_weight.numel() > 0 else None
        mixed_weights, _ = _mix_moe_weights_and_bias(
            coefficients,
            weight_experts,
            None,
            shared_w_arg,
            None,
        )
        x_pair = x_pair.contiguous()
        out = _triton_grouped_complex_forward(x_pair, mixed_weights, split_sizes)

        ctx.save_for_backward(x_pair, coefficients, weight_experts, shared_weight)
        ctx.split_sizes = split_sizes
        ctx.has_shared_weight = shared_w_arg is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_pair, coefficients, weight_experts, shared_weight = ctx.saved_tensors
        split_sizes = ctx.split_sizes
        shared_w_arg = shared_weight if ctx.has_shared_weight else None
        grad_out = grad_out.contiguous()

        mixed_weights, _ = _mix_moe_weights_and_bias(
            coefficients,
            weight_experts,
            None,
            shared_w_arg,
            None,
        )
        grad_x = _triton_grouped_complex_dx(grad_out, mixed_weights, split_sizes)
        grad_mixed_w = _triton_grouped_complex_dw(x_pair, grad_out, mixed_weights, split_sizes)

        grad_mixed_flat = grad_mixed_w.reshape(grad_mixed_w.shape[0], -1)
        weight_flat = weight_experts.reshape(weight_experts.shape[0], -1)
        grad_coeff = grad_mixed_flat.matmul(weight_flat.transpose(0, 1))
        grad_w = coefficients.transpose(0, 1).matmul(grad_mixed_flat).reshape_as(weight_experts)
        grad_sw = grad_mixed_w.sum(dim=0) if shared_w_arg is not None else None
        return grad_x, grad_coeff, grad_w, grad_sw, None


def grouped_complex_exact_moe_linear(x_pair: torch.Tensor,
                                     coefficients: torch.Tensor,
                                     weight_experts: torch.Tensor,
                                     shared_weight: Optional[torch.Tensor],
                                     split_sizes: Sequence[int]) -> torch.Tensor:
    split_sizes = _canonical_split_sizes(split_sizes)
    if _use_complex_exact_gp_v4():
        if _complex_exact_moe_linear_v4 is None:
            raise RuntimeError(
                "DPTB_TRITON_COMPLEX_EXACT_GP_V4 is enabled, but dptb.nn.so2_triton_exact_gp_v4 could not be imported."
            )
        return _complex_exact_moe_linear_v4(
            x_pair,
            coefficients,
            weight_experts,
            shared_weight,
            split_sizes,
        )
    if _use_complex_exact_gp_v3():
        if _complex_exact_moe_linear_v3 is None:
            raise RuntimeError(
                "DPTB_TRITON_COMPLEX_EXACT_GP_V3 is enabled, but dptb.nn.so2_triton_exact_gp_v3 could not be imported."
            )
        return _complex_exact_moe_linear_v3(
            x_pair,
            coefficients,
            weight_experts,
            shared_weight,
            split_sizes,
        )
    if _use_complex_exact_gp_v2():
        if _complex_exact_moe_linear_v2 is None:
            raise RuntimeError(
                "DPTB_TRITON_COMPLEX_EXACT_GP_V2 is enabled, but dptb.nn.so2_triton_exact_gp_v2 could not be imported."
            )
        return _complex_exact_moe_linear_v2(
            x_pair,
            coefficients,
            weight_experts,
            shared_weight,
            split_sizes,
        )
    return _GroupedComplexExactMoELinearFn.apply(
        x_pair,
        coefficients,
        weight_experts,
        shared_weight if shared_weight is not None else _empty(x_pair.device, x_pair.dtype),
        split_sizes,
    )
