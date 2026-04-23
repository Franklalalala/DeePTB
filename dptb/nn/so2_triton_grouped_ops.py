import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_TRITON_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


# --------------------------------------------------------------------------------------
# Runtime guards
# --------------------------------------------------------------------------------------

def _force_disable_grouped_triton() -> bool:
    return os.environ.get("DPTB_SO2_TRITON_GROUPED_DISABLE", "0") == "1"


def _allow_triton_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.float32


def _grouped_triton_runtime_ok(tensors: Iterable[torch.Tensor]) -> bool:
    if _force_disable_grouped_triton() or not _TRITON_AVAILABLE:
        return False
    tensors = tuple(tensors)
    if not tensors:
        return False
    if any((not t.is_cuda) for t in tensors):
        return False
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) != 1:
        return False
    dtype = next(iter(dtypes))
    return _allow_triton_dtype(dtype)


def _dtype_flags(dtype: torch.dtype) -> Tuple[bool, bool, bool]:
    return dtype == torch.float16, dtype == torch.bfloat16, dtype == torch.float32


# --------------------------------------------------------------------------------------
# Torch reference helpers
# --------------------------------------------------------------------------------------

def _torch_pack_m0(x_group: torch.Tensor, rot_block: torch.Tensor, col: int) -> torch.Tensor:
    return torch.einsum("ncd,nd->nc", x_group, rot_block[:, :, col])


def _torch_pack_pair(x_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int) -> torch.Tensor:
    return torch.einsum("ncd,ndp->npc", x_group, rot_block[:, :, [col0, col1]])


def _torch_scatter_m0(y_group: torch.Tensor, rot_block: torch.Tensor, col: int) -> torch.Tensor:
    return y_group.unsqueeze(-1) * rot_block[:, :, col].unsqueeze(1)


def _torch_scatter_pair(y_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int) -> torch.Tensor:
    return torch.einsum("npc,ndp->ncd", y_group, rot_block[:, :, [col0, col1]])


# --------------------------------------------------------------------------------------
# Triton grouped kernels.
# Best-practice changes compared with the previous per-group kernels:
#   * one launch per m for all active l groups (grouped GEMM style static scheduling);
#   * fixed number of CTAs ~= number of SMs (persistent / device-side work distribution);
#   * BLOCK_N x BLOCK_C tiling instead of one row per program;
#   * fp32 accumulation for low precision input/output.
# --------------------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    _GROUPED_CONFIGS = [
        triton.Config({"BLOCK_N": 1, "BLOCK_C": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 2, "BLOCK_C": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 2, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 4, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 4, "BLOCK_C": 128}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_GROUPED_CONFIGS, key=["group_size", "DMAX", "MAX_C"])
    @triton.jit
    def _grouped_pack_m0_kernel(
        x_ptrs,
        rot_ptrs,
        out_ptrs,
        g_meta,     # [group_size, 4] -> (N, C, D, col)
        group_size: tl.constexpr,
        NUM_SMS: tl.constexpr,
        MAX_C: tl.constexpr,
        DMAX: tl.constexpr,
        IS_FP16: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dtype = tl.float32
        if IS_FP16:
            dtype = tl.float16
        elif IS_BF16:
            dtype = tl.bfloat16
        elif IS_FP32:
            dtype = tl.float32

        tile_idx = pid
        last_problem_end = 0
        for g in range(group_size):
            N = tl.load(g_meta + g * 4 + 0)
            C = tl.load(g_meta + g * 4 + 1)
            D = tl.load(g_meta + g * 4 + 2)
            col = tl.load(g_meta + g * 4 + 3)
            num_n_tiles = tl.cdiv(N, BLOCK_N)
            num_c_tiles = tl.cdiv(C, BLOCK_C)
            num_tiles = num_n_tiles * num_c_tiles

            x_ptr = tl.load(x_ptrs + g).to(tl.pointer_type(dtype))
            rot_ptr = tl.load(rot_ptrs + g).to(tl.pointer_type(dtype))
            out_ptr = tl.load(out_ptrs + g).to(tl.pointer_type(dtype))

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                local_idx = tile_idx - last_problem_end
                tile_n_idx = local_idx // num_c_tiles
                tile_c_idx = local_idx % num_c_tiles

                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_c = tile_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
                mask_n = offs_n < N
                mask_c = offs_c < C

                acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
                for d in tl.static_range(0, DMAX):
                    dmask = d < D
                    x_ptrs_tile = x_ptr + offs_n[:, None] * (C * D) + offs_c[None, :] * D + d
                    x = tl.load(x_ptrs_tile, mask=(mask_n[:, None] & mask_c[None, :] & dmask), other=0.0)
                    w_ptrs = rot_ptr + offs_n * (D * D) + d * D + col
                    w = tl.load(w_ptrs, mask=(mask_n & dmask), other=0.0)
                    acc += x.to(tl.float32) * w[:, None].to(tl.float32)

                out_ptrs_tile = out_ptr + offs_n[:, None] * C + offs_c[None, :]
                tl.store(out_ptrs_tile, acc, mask=mask_n[:, None] & mask_c[None, :])
                tile_idx += NUM_SMS

            last_problem_end += num_tiles

    @triton.autotune(configs=_GROUPED_CONFIGS, key=["group_size", "DMAX", "MAX_C"])
    @triton.jit
    def _grouped_pack_pair_kernel(
        x_ptrs,
        rot_ptrs,
        out_ptrs,
        g_meta,     # [group_size, 5] -> (N, C, D, col0, col1)
        group_size: tl.constexpr,
        NUM_SMS: tl.constexpr,
        MAX_C: tl.constexpr,
        DMAX: tl.constexpr,
        IS_FP16: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dtype = tl.float32
        if IS_FP16:
            dtype = tl.float16
        elif IS_BF16:
            dtype = tl.bfloat16
        elif IS_FP32:
            dtype = tl.float32

        tile_idx = pid
        last_problem_end = 0
        for g in range(group_size):
            N = tl.load(g_meta + g * 5 + 0)
            C = tl.load(g_meta + g * 5 + 1)
            D = tl.load(g_meta + g * 5 + 2)
            col0 = tl.load(g_meta + g * 5 + 3)
            col1 = tl.load(g_meta + g * 5 + 4)
            num_n_tiles = tl.cdiv(N, BLOCK_N)
            num_c_tiles = tl.cdiv(C, BLOCK_C)
            num_tiles = num_n_tiles * num_c_tiles

            x_ptr = tl.load(x_ptrs + g).to(tl.pointer_type(dtype))
            rot_ptr = tl.load(rot_ptrs + g).to(tl.pointer_type(dtype))
            out_ptr = tl.load(out_ptrs + g).to(tl.pointer_type(dtype))

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                local_idx = tile_idx - last_problem_end
                tile_n_idx = local_idx // num_c_tiles
                tile_c_idx = local_idx % num_c_tiles

                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_c = tile_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
                mask_n = offs_n < N
                mask_c = offs_c < C

                acc0 = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
                acc1 = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
                for d in tl.static_range(0, DMAX):
                    dmask = d < D
                    x_ptrs_tile = x_ptr + offs_n[:, None] * (C * D) + offs_c[None, :] * D + d
                    x = tl.load(x_ptrs_tile, mask=(mask_n[:, None] & mask_c[None, :] & dmask), other=0.0)
                    w0_ptrs = rot_ptr + offs_n * (D * D) + d * D + col0
                    w1_ptrs = rot_ptr + offs_n * (D * D) + d * D + col1
                    w0 = tl.load(w0_ptrs, mask=(mask_n & dmask), other=0.0)
                    w1 = tl.load(w1_ptrs, mask=(mask_n & dmask), other=0.0)
                    xf = x.to(tl.float32)
                    acc0 += xf * w0[:, None].to(tl.float32)
                    acc1 += xf * w1[:, None].to(tl.float32)

                base_ptrs = out_ptr + offs_n[:, None] * (2 * C) + offs_c[None, :]
                tl.store(base_ptrs + 0 * C, acc0, mask=mask_n[:, None] & mask_c[None, :])
                tl.store(base_ptrs + 1 * C, acc1, mask=mask_n[:, None] & mask_c[None, :])
                tile_idx += NUM_SMS

            last_problem_end += num_tiles

    @triton.autotune(configs=_GROUPED_CONFIGS, key=["group_size", "DMAX", "MAX_C"])
    @triton.jit
    def _grouped_scatter_m0_kernel(
        y_ptrs,
        rot_ptrs,
        out_ptrs,
        g_meta,     # [group_size, 4] -> (N, C, D, col)
        group_size: tl.constexpr,
        NUM_SMS: tl.constexpr,
        MAX_C: tl.constexpr,
        DMAX: tl.constexpr,
        IS_FP16: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dtype = tl.float32
        if IS_FP16:
            dtype = tl.float16
        elif IS_BF16:
            dtype = tl.bfloat16
        elif IS_FP32:
            dtype = tl.float32

        tile_idx = pid
        last_problem_end = 0
        for g in range(group_size):
            N = tl.load(g_meta + g * 4 + 0)
            C = tl.load(g_meta + g * 4 + 1)
            D = tl.load(g_meta + g * 4 + 2)
            col = tl.load(g_meta + g * 4 + 3)
            num_n_tiles = tl.cdiv(N, BLOCK_N)
            num_c_tiles = tl.cdiv(C, BLOCK_C)
            num_tiles = num_n_tiles * num_c_tiles

            y_ptr = tl.load(y_ptrs + g).to(tl.pointer_type(dtype))
            rot_ptr = tl.load(rot_ptrs + g).to(tl.pointer_type(dtype))
            out_ptr = tl.load(out_ptrs + g).to(tl.pointer_type(dtype))

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                local_idx = tile_idx - last_problem_end
                tile_n_idx = local_idx // num_c_tiles
                tile_c_idx = local_idx % num_c_tiles

                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_c = tile_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
                mask_n = offs_n < N
                mask_c = offs_c < C

                y_ptrs_tile = y_ptr + offs_n[:, None] * C + offs_c[None, :]
                y = tl.load(y_ptrs_tile, mask=mask_n[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
                for d in tl.static_range(0, DMAX):
                    dmask = d < D
                    w_ptrs = rot_ptr + offs_n * (D * D) + d * D + col
                    w = tl.load(w_ptrs, mask=(mask_n & dmask), other=0.0).to(tl.float32)
                    out_ptrs_tile = out_ptr + offs_n[:, None] * (C * D) + offs_c[None, :] * D + d
                    tl.store(out_ptrs_tile, y * w[:, None], mask=mask_n[:, None] & mask_c[None, :] & dmask)
                tile_idx += NUM_SMS

            last_problem_end += num_tiles

    @triton.autotune(configs=_GROUPED_CONFIGS, key=["group_size", "DMAX", "MAX_C"])
    @triton.jit
    def _grouped_scatter_pair_kernel(
        y_ptrs,
        rot_ptrs,
        out_ptrs,
        g_meta,     # [group_size, 5] -> (N, C, D, col0, col1)
        group_size: tl.constexpr,
        NUM_SMS: tl.constexpr,
        MAX_C: tl.constexpr,
        DMAX: tl.constexpr,
        IS_FP16: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dtype = tl.float32
        if IS_FP16:
            dtype = tl.float16
        elif IS_BF16:
            dtype = tl.bfloat16
        elif IS_FP32:
            dtype = tl.float32

        tile_idx = pid
        last_problem_end = 0
        for g in range(group_size):
            N = tl.load(g_meta + g * 5 + 0)
            C = tl.load(g_meta + g * 5 + 1)
            D = tl.load(g_meta + g * 5 + 2)
            col0 = tl.load(g_meta + g * 5 + 3)
            col1 = tl.load(g_meta + g * 5 + 4)
            num_n_tiles = tl.cdiv(N, BLOCK_N)
            num_c_tiles = tl.cdiv(C, BLOCK_C)
            num_tiles = num_n_tiles * num_c_tiles

            y_ptr = tl.load(y_ptrs + g).to(tl.pointer_type(dtype))
            rot_ptr = tl.load(rot_ptrs + g).to(tl.pointer_type(dtype))
            out_ptr = tl.load(out_ptrs + g).to(tl.pointer_type(dtype))

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                local_idx = tile_idx - last_problem_end
                tile_n_idx = local_idx // num_c_tiles
                tile_c_idx = local_idx % num_c_tiles

                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_c = tile_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
                mask_n = offs_n < N
                mask_c = offs_c < C

                base_y_ptrs = y_ptr + offs_n[:, None] * (2 * C) + offs_c[None, :]
                y0 = tl.load(base_y_ptrs + 0 * C, mask=mask_n[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
                y1 = tl.load(base_y_ptrs + 1 * C, mask=mask_n[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
                for d in tl.static_range(0, DMAX):
                    dmask = d < D
                    w0_ptrs = rot_ptr + offs_n * (D * D) + d * D + col0
                    w1_ptrs = rot_ptr + offs_n * (D * D) + d * D + col1
                    w0 = tl.load(w0_ptrs, mask=(mask_n & dmask), other=0.0).to(tl.float32)
                    w1 = tl.load(w1_ptrs, mask=(mask_n & dmask), other=0.0).to(tl.float32)
                    out_ptrs_tile = out_ptr + offs_n[:, None] * (C * D) + offs_c[None, :] * D + d
                    tl.store(out_ptrs_tile, y0 * w0[:, None] + y1 * w1[:, None],
                             mask=mask_n[:, None] & mask_c[None, :] & dmask)
                tile_idx += NUM_SMS

            last_problem_end += num_tiles


def _num_sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _launch_grouped_pack_m0(xs: Sequence[torch.Tensor], rots: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if not xs:
        return []
    if not _grouped_triton_runtime_ok(tuple(xs) + tuple(rots)):
        return [_torch_pack_m0(x, r, r.shape[1] // 2) for x, r in zip(xs, rots)]

    device = xs[0].device
    dtype = xs[0].dtype
    is_fp16, is_bf16, is_fp32 = _dtype_flags(dtype)

    outs = [torch.empty((x.shape[0], x.shape[1]), device=device, dtype=dtype) for x in xs]
    x_ptrs = torch.tensor([x.data_ptr() for x in xs], device=device, dtype=torch.int64)
    rot_ptrs = torch.tensor([r.data_ptr() for r in rots], device=device, dtype=torch.int64)
    out_ptrs = torch.tensor([o.data_ptr() for o in outs], device=device, dtype=torch.int64)
    meta = []
    max_c = 0
    dmax = 0
    for x, r in zip(xs, rots):
        n, c, d = x.shape
        col = d // 2
        meta.extend([n, c, d, col])
        max_c = max(max_c, c)
        dmax = max(dmax, d)
    meta_t = torch.tensor(meta, device=device, dtype=torch.int32)
    grid = (_num_sms(device),)
    _grouped_pack_m0_kernel[grid](
        x_ptrs,
        rot_ptrs,
        out_ptrs,
        meta_t,
        len(xs),
        _num_sms(device),
        max_c,
        DMAX=dmax,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        IS_FP32=is_fp32,
    )
    return outs


def _launch_grouped_pack_pair(xs: Sequence[torch.Tensor], rots: Sequence[torch.Tensor], ls: Sequence[int], m: int) -> List[torch.Tensor]:
    if not xs:
        return []
    if not _grouped_triton_runtime_ok(tuple(xs) + tuple(rots)):
        return [_torch_pack_pair(x, r, l - m, l + m) for x, r, l in zip(xs, rots, ls)]

    device = xs[0].device
    dtype = xs[0].dtype
    is_fp16, is_bf16, is_fp32 = _dtype_flags(dtype)

    outs = [torch.empty((x.shape[0], 2, x.shape[1]), device=device, dtype=dtype) for x in xs]
    x_ptrs = torch.tensor([x.data_ptr() for x in xs], device=device, dtype=torch.int64)
    rot_ptrs = torch.tensor([r.data_ptr() for r in rots], device=device, dtype=torch.int64)
    out_ptrs = torch.tensor([o.data_ptr() for o in outs], device=device, dtype=torch.int64)
    meta = []
    max_c = 0
    dmax = 0
    for x, r, l in zip(xs, rots, ls):
        n, c, d = x.shape
        meta.extend([n, c, d, int(l - m), int(l + m)])
        max_c = max(max_c, c)
        dmax = max(dmax, d)
    meta_t = torch.tensor(meta, device=device, dtype=torch.int32)
    grid = (_num_sms(device),)
    _grouped_pack_pair_kernel[grid](
        x_ptrs,
        rot_ptrs,
        out_ptrs,
        meta_t,
        len(xs),
        _num_sms(device),
        max_c,
        DMAX=dmax,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        IS_FP32=is_fp32,
    )
    return outs


def _launch_grouped_scatter_m0(ys: Sequence[torch.Tensor], rots: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if not ys:
        return []
    if not _grouped_triton_runtime_ok(tuple(ys) + tuple(rots)):
        return [_torch_scatter_m0(y, r, r.shape[1] // 2) for y, r in zip(ys, rots)]

    device = ys[0].device
    dtype = ys[0].dtype
    is_fp16, is_bf16, is_fp32 = _dtype_flags(dtype)

    outs = [torch.empty((y.shape[0], y.shape[1], r.shape[1]), device=device, dtype=dtype) for y, r in zip(ys, rots)]
    y_ptrs = torch.tensor([y.data_ptr() for y in ys], device=device, dtype=torch.int64)
    rot_ptrs = torch.tensor([r.data_ptr() for r in rots], device=device, dtype=torch.int64)
    out_ptrs = torch.tensor([o.data_ptr() for o in outs], device=device, dtype=torch.int64)
    meta = []
    max_c = 0
    dmax = 0
    for y, r in zip(ys, rots):
        n, c = y.shape
        d = r.shape[1]
        col = d // 2
        meta.extend([n, c, d, col])
        max_c = max(max_c, c)
        dmax = max(dmax, d)
    meta_t = torch.tensor(meta, device=device, dtype=torch.int32)
    grid = (_num_sms(device),)
    _grouped_scatter_m0_kernel[grid](
        y_ptrs,
        rot_ptrs,
        out_ptrs,
        meta_t,
        len(ys),
        _num_sms(device),
        max_c,
        DMAX=dmax,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        IS_FP32=is_fp32,
    )
    return outs


def _launch_grouped_scatter_pair(ys: Sequence[torch.Tensor], rots: Sequence[torch.Tensor], ls: Sequence[int], m: int) -> List[torch.Tensor]:
    if not ys:
        return []
    if not _grouped_triton_runtime_ok(tuple(ys) + tuple(rots)):
        return [_torch_scatter_pair(y, r, l - m, l + m) for y, r, l in zip(ys, rots, ls)]

    device = ys[0].device
    dtype = ys[0].dtype
    is_fp16, is_bf16, is_fp32 = _dtype_flags(dtype)

    outs = [torch.empty((y.shape[0], y.shape[2], r.shape[1]), device=device, dtype=dtype) for y, r in zip(ys, rots)]
    y_ptrs = torch.tensor([y.data_ptr() for y in ys], device=device, dtype=torch.int64)
    rot_ptrs = torch.tensor([r.data_ptr() for r in rots], device=device, dtype=torch.int64)
    out_ptrs = torch.tensor([o.data_ptr() for o in outs], device=device, dtype=torch.int64)
    meta = []
    max_c = 0
    dmax = 0
    for y, r, l in zip(ys, rots, ls):
        n, _, c = y.shape
        d = r.shape[1]
        meta.extend([n, c, d, int(l - m), int(l + m)])
        max_c = max(max_c, c)
        dmax = max(dmax, d)
    meta_t = torch.tensor(meta, device=device, dtype=torch.int32)
    grid = (_num_sms(device),)
    _grouped_scatter_pair_kernel[grid](
        y_ptrs,
        rot_ptrs,
        out_ptrs,
        meta_t,
        len(ys),
        _num_sms(device),
        max_c,
        DMAX=dmax,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        IS_FP32=is_fp32,
    )
    return outs


# --------------------------------------------------------------------------------------
# Grouped autograd wrappers. Forward uses grouped Triton launches when available;
# backward keeps a compact torch formula for correctness and lower implementation risk.
# --------------------------------------------------------------------------------------

class _GroupedPackM0Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l_keys, *tensor_args):
        num_tasks = len(l_keys)
        xs = [tensor_args[2 * i].contiguous() for i in range(num_tasks)]
        rots = [tensor_args[2 * i + 1].contiguous() for i in range(num_tasks)]
        ctx.l_keys = tuple(int(v) for v in l_keys)
        ctx.save_for_backward(*xs, *rots)
        outs = _launch_grouped_pack_m0(xs, rots)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_tasks = len(ctx.l_keys)
        saved = ctx.saved_tensors
        xs = saved[:num_tasks]
        rots = saved[num_tasks:]
        grads = []
        for x, rot, grad_out in zip(xs, rots, grad_outputs):
            col = rot.shape[1] // 2
            grad_x = _torch_scatter_m0(grad_out, rot, col)
            grad_rot = rot.new_zeros(rot.shape)
            grad_rot[:, :, col] = torch.einsum("ncd,nc->nd", x, grad_out)
            grads.extend([grad_x, grad_rot])
        return (None, *grads)


class _GroupedPackPairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l_keys, m, *tensor_args):
        num_tasks = len(l_keys)
        xs = [tensor_args[2 * i].contiguous() for i in range(num_tasks)]
        rots = [tensor_args[2 * i + 1].contiguous() for i in range(num_tasks)]
        ctx.l_keys = tuple(int(v) for v in l_keys)
        ctx.m = int(m)
        ctx.save_for_backward(*xs, *rots)
        outs = _launch_grouped_pack_pair(xs, rots, ctx.l_keys, ctx.m)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_tasks = len(ctx.l_keys)
        saved = ctx.saved_tensors
        xs = saved[:num_tasks]
        rots = saved[num_tasks:]
        grads = []
        for l, x, rot, grad_out in zip(ctx.l_keys, xs, rots, grad_outputs):
            col0, col1 = int(l - ctx.m), int(l + ctx.m)
            grad_x = _torch_scatter_pair(grad_out, rot, col0, col1)
            grad_rot = rot.new_zeros(rot.shape)
            grad_rot[:, :, col0] = torch.einsum("ncd,nc->nd", x, grad_out[:, 0, :])
            grad_rot[:, :, col1] = torch.einsum("ncd,nc->nd", x, grad_out[:, 1, :])
            grads.extend([grad_x, grad_rot])
        return (None, None, *grads)


class _GroupedScatterM0Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l_keys, *tensor_args):
        num_tasks = len(l_keys)
        ys = [tensor_args[2 * i].contiguous() for i in range(num_tasks)]
        rots = [tensor_args[2 * i + 1].contiguous() for i in range(num_tasks)]
        ctx.l_keys = tuple(int(v) for v in l_keys)
        ctx.save_for_backward(*ys, *rots)
        outs = _launch_grouped_scatter_m0(ys, rots)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_tasks = len(ctx.l_keys)
        saved = ctx.saved_tensors
        ys = saved[:num_tasks]
        rots = saved[num_tasks:]
        grads = []
        for y, rot, grad_out in zip(ys, rots, grad_outputs):
            col = rot.shape[1] // 2
            grad_y = _torch_pack_m0(grad_out, rot, col)
            grad_rot = rot.new_zeros(rot.shape)
            grad_rot[:, :, col] = torch.einsum("ncd,nc->nd", grad_out, y)
            grads.extend([grad_y, grad_rot])
        return (None, *grads)


class _GroupedScatterPairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l_keys, m, *tensor_args):
        num_tasks = len(l_keys)
        ys = [tensor_args[2 * i].contiguous() for i in range(num_tasks)]
        rots = [tensor_args[2 * i + 1].contiguous() for i in range(num_tasks)]
        ctx.l_keys = tuple(int(v) for v in l_keys)
        ctx.m = int(m)
        ctx.save_for_backward(*ys, *rots)
        outs = _launch_grouped_scatter_pair(ys, rots, ctx.l_keys, ctx.m)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_tasks = len(ctx.l_keys)
        saved = ctx.saved_tensors
        ys = saved[:num_tasks]
        rots = saved[num_tasks:]
        grads = []
        for l, y, rot, grad_out in zip(ctx.l_keys, ys, rots, grad_outputs):
            col0, col1 = int(l - ctx.m), int(l + ctx.m)
            grad_y = _torch_pack_pair(grad_out, rot, col0, col1)
            grad_rot = rot.new_zeros(rot.shape)
            grad_rot[:, :, col0] = torch.einsum("ncd,nc->nd", grad_out, y[:, 0, :])
            grad_rot[:, :, col1] = torch.einsum("ncd,nc->nd", grad_out, y[:, 1, :])
            grads.extend([grad_y, grad_rot])
        return (None, None, *grads)


# --------------------------------------------------------------------------------------
# Public grouped helpers used by the SO2 overlay.
# These helpers keep route-level semantics identical to the existing streamed path,
# but collapse all active l groups for a given m into one Triton launch.
# --------------------------------------------------------------------------------------

def grouped_pack_m0(
    input_groups: Dict[int, torch.Tensor],
    rot_blocks: Dict[int, Optional[torch.Tensor]],
    l_keys: Sequence[int],
    *,
    rotate_in: bool,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    triton_ls: List[int] = []
    triton_xs: List[torch.Tensor] = []
    triton_rs: List[torch.Tensor] = []

    for l in l_keys:
        x = input_groups[l]
        r = rot_blocks.get(l)
        if x.numel() == 0:
            out[l] = x.new_empty((x.shape[0], x.shape[1]))
        elif l == 0 or (not rotate_in) or r is None:
            out[l] = x[:, :, l]
        else:
            triton_ls.append(int(l))
            triton_xs.append(x.contiguous())
            triton_rs.append(r.contiguous())

    if triton_ls:
        flat_args: List[torch.Tensor] = []
        for x, r in zip(triton_xs, triton_rs):
            flat_args.extend([x, r])
        packed = _GroupedPackM0Fn.apply(tuple(triton_ls), *flat_args)
        for l, y in zip(triton_ls, packed):
            out[l] = y
    return out


def grouped_pack_pair(
    input_groups: Dict[int, torch.Tensor],
    rot_blocks: Dict[int, Optional[torch.Tensor]],
    l_keys: Sequence[int],
    m: int,
    *,
    rotate_in: bool,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    triton_ls: List[int] = []
    triton_xs: List[torch.Tensor] = []
    triton_rs: List[torch.Tensor] = []

    for l in l_keys:
        x = input_groups[l]
        r = rot_blocks.get(l)
        col0, col1 = int(l - m), int(l + m)
        if x.numel() == 0:
            out[l] = x.new_empty((x.shape[0], 2, x.shape[1]))
        elif (not rotate_in) or r is None:
            out[l] = x[:, :, [col0, col1]].transpose(1, 2).contiguous()
        else:
            triton_ls.append(int(l))
            triton_xs.append(x.contiguous())
            triton_rs.append(r.contiguous())

    if triton_ls:
        flat_args: List[torch.Tensor] = []
        for x, r in zip(triton_xs, triton_rs):
            flat_args.extend([x, r])
        packed = _GroupedPackPairFn.apply(tuple(triton_ls), int(m), *flat_args)
        for l, y in zip(triton_ls, packed):
            out[l] = y
    return out


def grouped_scatter_m0(
    grouped_y: Dict[int, torch.Tensor],
    rot_blocks: Dict[int, Optional[torch.Tensor]],
    l_keys: Sequence[int],
    *,
    rotate_out: bool,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    triton_ls: List[int] = []
    triton_ys: List[torch.Tensor] = []
    triton_rs: List[torch.Tensor] = []

    for l in l_keys:
        y = grouped_y.get(l)
        if y is None:
            continue
        r = rot_blocks.get(l)
        out_dim = 2 * l + 1
        if y.numel() == 0:
            out[l] = y.new_zeros((y.shape[0], y.shape[1], out_dim))
        elif l == 0 or (not rotate_out) or r is None:
            buf = y.new_zeros((y.shape[0], y.shape[1], out_dim))
            buf[:, :, l] = y
            out[l] = buf
        else:
            triton_ls.append(int(l))
            triton_ys.append(y.contiguous())
            triton_rs.append(r.contiguous())

    if triton_ls:
        flat_args: List[torch.Tensor] = []
        for y, r in zip(triton_ys, triton_rs):
            flat_args.extend([y, r])
        scattered = _GroupedScatterM0Fn.apply(tuple(triton_ls), *flat_args)
        for l, block in zip(triton_ls, scattered):
            out[l] = block
    return out


def grouped_scatter_pair(
    grouped_y: Dict[int, torch.Tensor],
    rot_blocks: Dict[int, Optional[torch.Tensor]],
    l_keys: Sequence[int],
    m: int,
    *,
    rotate_out: bool,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    triton_ls: List[int] = []
    triton_ys: List[torch.Tensor] = []
    triton_rs: List[torch.Tensor] = []

    for l in l_keys:
        y = grouped_y.get(l)
        if y is None:
            continue
        r = rot_blocks.get(l)
        out_dim = 2 * l + 1
        col0, col1 = int(l - m), int(l + m)
        if y.numel() == 0:
            out[l] = y.new_zeros((y.shape[0], y.shape[2], out_dim))
        elif (not rotate_out) or r is None:
            buf = y.new_zeros((y.shape[0], y.shape[2], out_dim))
            buf[:, :, [col0, col1]] = y.transpose(1, 2)
            out[l] = buf
        else:
            triton_ls.append(int(l))
            triton_ys.append(y.contiguous())
            triton_rs.append(r.contiguous())

    if triton_ls:
        flat_args: List[torch.Tensor] = []
        for y, r in zip(triton_ys, triton_rs):
            flat_args.extend([y, r])
        scattered = _GroupedScatterPairFn.apply(tuple(triton_ls), int(m), *flat_args)
        for l, block in zip(triton_ls, scattered):
            out[l] = block
    return out


def triton_grouped_runtime_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available() and not _force_disable_grouped_triton()
