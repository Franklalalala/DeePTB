import os
from typing import Optional

import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def _use_triton_for_tensor(x: torch.Tensor) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if x.device.type != "cuda":
        return False
    # Keep the first production route fp32-only. fp64 is left on the torch
    # fallback path so correctness tests keep strict tolerances, and half
    # precision is intentionally out of scope for this branch.
    if x.dtype != torch.float32:
        return False
    return True


def _force_disable_triton() -> bool:
    return os.environ.get("DPTB_SO2_TRITON_DISABLE", "0") == "1"


if _TRITON_AVAILABLE:
    _BLOCK_C = 64

    @triton.jit
    def _pack_m0_kernel(
        x_ptr,
        rot_ptr,
        out_ptr,
        stride_xn,
        stride_xc,
        stride_xd,
        stride_rn,
        stride_rd0,
        stride_rd1,
        stride_on,
        stride_oc,
        C,
        COL,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for d in tl.static_range(0, D):
            x = tl.load(x_ptr + pid_n * stride_xn + offs_c * stride_xc + d * stride_xd, mask=mask_c, other=0.0)
            w = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL * stride_rd1)
            acc += x.to(tl.float32) * w.to(tl.float32)
        tl.store(out_ptr + pid_n * stride_on + offs_c * stride_oc, acc, mask=mask_c)

    @triton.jit
    def _pack_pair_kernel(
        x_ptr,
        rot_ptr,
        out_ptr,
        stride_xn,
        stride_xc,
        stride_xd,
        stride_rn,
        stride_rd0,
        stride_rd1,
        stride_on,
        stride_op,
        stride_oc,
        C,
        COL0,
        COL1,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for d in tl.static_range(0, D):
            x = tl.load(x_ptr + pid_n * stride_xn + offs_c * stride_xc + d * stride_xd, mask=mask_c, other=0.0)
            w0 = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL0 * stride_rd1)
            w1 = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL1 * stride_rd1)
            xf = x.to(tl.float32)
            acc0 += xf * w0.to(tl.float32)
            acc1 += xf * w1.to(tl.float32)
        tl.store(out_ptr + pid_n * stride_on + 0 * stride_op + offs_c * stride_oc, acc0, mask=mask_c)
        tl.store(out_ptr + pid_n * stride_on + 1 * stride_op + offs_c * stride_oc, acc1, mask=mask_c)

    @triton.jit
    def _scatter_m0_kernel(
        y_ptr,
        rot_ptr,
        out_ptr,
        stride_yn,
        stride_yc,
        stride_rn,
        stride_rd0,
        stride_rd1,
        stride_on,
        stride_oc,
        stride_od,
        C,
        COL,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        y = tl.load(y_ptr + pid_n * stride_yn + offs_c * stride_yc, mask=mask_c, other=0.0).to(tl.float32)
        for d in tl.static_range(0, D):
            w = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL * stride_rd1).to(tl.float32)
            tl.store(out_ptr + pid_n * stride_on + offs_c * stride_oc + d * stride_od, y * w, mask=mask_c)

    @triton.jit
    def _scatter_pair_kernel(
        y_ptr,
        rot_ptr,
        out_ptr,
        stride_yn,
        stride_yp,
        stride_yc,
        stride_rn,
        stride_rd0,
        stride_rd1,
        stride_on,
        stride_oc,
        stride_od,
        C,
        COL0,
        COL1,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        y0 = tl.load(y_ptr + pid_n * stride_yn + 0 * stride_yp + offs_c * stride_yc, mask=mask_c, other=0.0).to(tl.float32)
        y1 = tl.load(y_ptr + pid_n * stride_yn + 1 * stride_yp + offs_c * stride_yc, mask=mask_c, other=0.0).to(tl.float32)
        for d in tl.static_range(0, D):
            w0 = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL0 * stride_rd1).to(tl.float32)
            w1 = tl.load(rot_ptr + pid_n * stride_rn + d * stride_rd0 + COL1 * stride_rd1).to(tl.float32)
            tl.store(out_ptr + pid_n * stride_on + offs_c * stride_oc + d * stride_od, y0 * w0 + y1 * w1, mask=mask_c)


def _torch_pack_m0(x_group: torch.Tensor, rot_block: torch.Tensor, col: int) -> torch.Tensor:
    return torch.einsum("ncd,nd->nc", x_group, rot_block[:, :, col])


def _torch_pack_pair(x_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int) -> torch.Tensor:
    return torch.einsum("ncd,ndp->npc", x_group, rot_block[:, :, [col0, col1]])


def _torch_scatter_m0(y_group: torch.Tensor, rot_block: torch.Tensor, col: int) -> torch.Tensor:
    return y_group.unsqueeze(-1) * rot_block[:, :, col].unsqueeze(1)


def _torch_scatter_pair(y_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int) -> torch.Tensor:
    return torch.einsum("npc,ndp->ncd", y_group, rot_block[:, :, [col0, col1]])


def pack_m0_fwd(x_group: torch.Tensor, rot_block: torch.Tensor, col: int) -> torch.Tensor:
    if _force_disable_triton() or not _use_triton_for_tensor(x_group):
        return _torch_pack_m0(x_group, rot_block, col)
    n, c, d = x_group.shape
    out = torch.empty((n, c), device=x_group.device, dtype=x_group.dtype)
    grid = (n, triton.cdiv(c, _BLOCK_C))
    _pack_m0_kernel[grid](
        x_group, rot_block, out,
        x_group.stride(0), x_group.stride(1), x_group.stride(2),
        rot_block.stride(0), rot_block.stride(1), rot_block.stride(2),
        out.stride(0), out.stride(1),
        c, col, D=d, BLOCK_C=_BLOCK_C,
    )
    return out


def pack_pair_fwd(x_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int) -> torch.Tensor:
    if _force_disable_triton() or not _use_triton_for_tensor(x_group):
        return _torch_pack_pair(x_group, rot_block, col0, col1)
    n, c, d = x_group.shape
    out = torch.empty((n, 2, c), device=x_group.device, dtype=x_group.dtype)
    grid = (n, triton.cdiv(c, _BLOCK_C))
    _pack_pair_kernel[grid](
        x_group, rot_block, out,
        x_group.stride(0), x_group.stride(1), x_group.stride(2),
        rot_block.stride(0), rot_block.stride(1), rot_block.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        c, col0, col1, D=d, BLOCK_C=_BLOCK_C,
    )
    return out


def scatter_m0_fwd(y_group: torch.Tensor, rot_block: torch.Tensor, col: int, out_dim: int) -> torch.Tensor:
    if _force_disable_triton() or not _use_triton_for_tensor(y_group):
        return _torch_scatter_m0(y_group, rot_block, col)
    n, c = y_group.shape
    out = torch.empty((n, c, out_dim), device=y_group.device, dtype=y_group.dtype)
    grid = (n, triton.cdiv(c, _BLOCK_C))
    _scatter_m0_kernel[grid](
        y_group, rot_block, out,
        y_group.stride(0), y_group.stride(1),
        rot_block.stride(0), rot_block.stride(1), rot_block.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        c, col, D=out_dim, BLOCK_C=_BLOCK_C,
    )
    return out


def scatter_pair_fwd(y_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int, out_dim: int) -> torch.Tensor:
    if _force_disable_triton() or not _use_triton_for_tensor(y_group):
        return _torch_scatter_pair(y_group, rot_block, col0, col1)
    n, _, c = y_group.shape
    out = torch.empty((n, c, out_dim), device=y_group.device, dtype=y_group.dtype)
    grid = (n, triton.cdiv(c, _BLOCK_C))
    _scatter_pair_kernel[grid](
        y_group, rot_block, out,
        y_group.stride(0), y_group.stride(1), y_group.stride(2),
        rot_block.stride(0), rot_block.stride(1), rot_block.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        c, col0, col1, D=out_dim, BLOCK_C=_BLOCK_C,
    )
    return out


class _PackM0Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_group: torch.Tensor, rot_block: torch.Tensor, col: int):
        ctx.col = int(col)
        ctx.save_for_backward(x_group, rot_block)
        return pack_m0_fwd(x_group, rot_block, int(col))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_group, rot_block = ctx.saved_tensors
        col = ctx.col
        grad_x = scatter_m0_fwd(grad_out.contiguous(), rot_block, col, x_group.shape[2])
        grad_rot = rot_block.new_zeros(rot_block.shape)
        grad_rot[:, :, col] = torch.einsum("ncd,nc->nd", x_group, grad_out)
        return grad_x, grad_rot, None


class _PackPairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int):
        ctx.col0 = int(col0)
        ctx.col1 = int(col1)
        ctx.save_for_backward(x_group, rot_block)
        return pack_pair_fwd(x_group, rot_block, int(col0), int(col1))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_group, rot_block = ctx.saved_tensors
        col0, col1 = ctx.col0, ctx.col1
        grad_x = scatter_pair_fwd(grad_out.contiguous(), rot_block, col0, col1, x_group.shape[2])
        grad_rot = rot_block.new_zeros(rot_block.shape)
        grad_rot[:, :, col0] = torch.einsum("ncd,nc->nd", x_group, grad_out[:, 0, :])
        grad_rot[:, :, col1] = torch.einsum("ncd,nc->nd", x_group, grad_out[:, 1, :])
        return grad_x, grad_rot, None, None


class _ScatterM0Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_group: torch.Tensor, rot_block: torch.Tensor, col: int, out_dim: int):
        ctx.col = int(col)
        ctx.out_dim = int(out_dim)
        ctx.save_for_backward(y_group, rot_block)
        return scatter_m0_fwd(y_group, rot_block, int(col), int(out_dim))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        y_group, rot_block = ctx.saved_tensors
        col = ctx.col
        grad_y = pack_m0_fwd(grad_out.contiguous(), rot_block, col)
        grad_rot = rot_block.new_zeros(rot_block.shape)
        grad_rot[:, :, col] = torch.einsum("ncd,nc->nd", grad_out, y_group)
        return grad_y, grad_rot, None, None


class _ScatterPairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_group: torch.Tensor, rot_block: torch.Tensor, col0: int, col1: int, out_dim: int):
        ctx.col0 = int(col0)
        ctx.col1 = int(col1)
        ctx.out_dim = int(out_dim)
        ctx.save_for_backward(y_group, rot_block)
        return scatter_pair_fwd(y_group, rot_block, int(col0), int(col1), int(out_dim))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        y_group, rot_block = ctx.saved_tensors
        col0, col1 = ctx.col0, ctx.col1
        grad_y = pack_pair_fwd(grad_out.contiguous(), rot_block, col0, col1)
        grad_rot = rot_block.new_zeros(rot_block.shape)
        grad_rot[:, :, col0] = torch.einsum("ncd,nc->nd", grad_out, y_group[:, 0, :])
        grad_rot[:, :, col1] = torch.einsum("ncd,nc->nd", grad_out, y_group[:, 1, :])
        return grad_y, grad_rot, None, None, None


def triton_pack_group_m0(x_group: torch.Tensor, rot_block: Optional[torch.Tensor], l: int, rotate_in: bool) -> torch.Tensor:
    if x_group.numel() == 0:
        return x_group.new_empty((x_group.shape[0], x_group.shape[1]))
    if l == 0 or (not rotate_in) or rot_block is None:
        return x_group[:, :, l]
    return _PackM0Fn.apply(x_group.contiguous(), rot_block.contiguous(), int(l))


def triton_pack_group_pair(x_group: torch.Tensor, rot_block: Optional[torch.Tensor], l: int, m: int, rotate_in: bool) -> torch.Tensor:
    if x_group.numel() == 0:
        return x_group.new_empty((x_group.shape[0], 2, x_group.shape[1]))
    col0, col1 = int(l - m), int(l + m)
    if (not rotate_in) or rot_block is None:
        return x_group[:, :, [col0, col1]].transpose(1, 2).contiguous()
    return _PackPairFn.apply(x_group.contiguous(), rot_block.contiguous(), col0, col1)


def triton_scatter_group_m0(y_group: torch.Tensor, rot_block: Optional[torch.Tensor], l: int, out_dim: int, rotate_out: bool) -> torch.Tensor:
    if y_group.numel() == 0:
        return y_group.new_zeros((y_group.shape[0], y_group.shape[1], out_dim))
    if l == 0 or (not rotate_out) or rot_block is None:
        out = y_group.new_zeros((y_group.shape[0], y_group.shape[1], out_dim))
        out[:, :, l] = y_group
        return out
    return _ScatterM0Fn.apply(y_group.contiguous(), rot_block.contiguous(), int(l), int(out_dim))


def triton_scatter_group_pair(y_group: torch.Tensor, rot_block: Optional[torch.Tensor], l: int, m: int, out_dim: int, rotate_out: bool) -> torch.Tensor:
    if y_group.numel() == 0:
        return y_group.new_zeros((y_group.shape[0], y_group.shape[2], out_dim))
    col0, col1 = int(l - m), int(l + m)
    if (not rotate_out) or rot_block is None:
        out = y_group.new_zeros((y_group.shape[0], y_group.shape[2], out_dim))
        out[:, :, [col0, col1]] = y_group.transpose(1, 2)
        return out
    return _ScatterPairFn.apply(y_group.contiguous(), rot_block.contiguous(), col0, col1, int(out_dim))


def triton_runtime_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()
