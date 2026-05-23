from pathlib import Path
from typing import Optional

import torch

from dptb.nn.cuda_ops.extension_loader import load_cuda_extension, truthy_env


_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    src = Path(__file__).resolve().parent / "csrc" / "cublas_grouped_gemm.cpp"
    _EXT = load_cuda_extension(
        name="dptb_cublas_grouped_gemm",
        source_files=[src],
        build_dir_env="DPTB_CUBLAS_GROUPED_BUILD_DIR",
        default_build_dir=Path.home() / ".cache" / "dptb_cublas_grouped",
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublas"],
        verbose_env="DPTB_CUBLAS_GROUPED_VERBOSE",
    )
    return _EXT


def _fast_tf32_enabled() -> bool:
    return truthy_env("DPTB_CUBLAS_GROUPED_FAST_TF32")


class _GroupedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, ptr: torch.Tensor, weight: torch.Tensor, fast_tf32: bool):
        if x.dtype != torch.float32 or weight.dtype != torch.float32:
            raise RuntimeError(f"cublas_grouped_gemm currently requires float32, got x={x.dtype}, weight={weight.dtype}")
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("cublas_grouped_gemm requires CUDA tensors")
        ptr_cpu = ptr.detach().to(device="cpu", dtype=torch.long).contiguous()
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        out = _load_extension().grouped_gemm_forward_fp32(
            x_contig,
            ptr_cpu,
            weight_contig,
            bool(fast_tf32),
        )
        ctx.save_for_backward(x_contig, ptr_cpu, weight_contig)
        ctx.fast_tf32 = bool(fast_tf32)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, ptr_cpu, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        ext = _load_extension()
        grad_x = ext.grouped_gemm_forward_fp32(
            grad_out,
            ptr_cpu,
            weight.transpose(1, 2).contiguous(),
            ctx.fast_tf32,
        )
        grad_weight = ext.grouped_gemm_backward_weight_fp32(
            grad_out,
            x,
            ptr_cpu,
            int(weight.shape[0]),
            ctx.fast_tf32,
        )
        return grad_x, None, grad_weight, None


def grouped_gemm(x: torch.Tensor, ptr: torch.Tensor, weight: torch.Tensor, *, fast_tf32: Optional[bool] = None) -> torch.Tensor:
    """Apply per-segment row-major GEMM: out[start:end] = x[start:end] @ weight[g].T.

    Args:
        x: Contiguous or strided CUDA fp32 tensor with shape ``[N, in_features]``.
        ptr: CPU or CUDA int64 offsets with shape ``[num_groups + 1]``.
        weight: CUDA fp32 tensor with shape ``[num_groups, out_features, in_features]``.
        fast_tf32: Optional override for Tensor Core TF32 math. Defaults to the
            ``DPTB_CUBLAS_GROUPED_FAST_TF32`` environment flag.
    """
    if fast_tf32 is None:
        fast_tf32 = _fast_tf32_enabled()
    return _GroupedGemmFunction.apply(x, ptr, weight, bool(fast_tf32))


class _GroupedGemmMultiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fast_tf32: bool, num_problems: int, *args):
        xs = [arg.contiguous() for arg in args[:num_problems]]
        ptrs = [
            arg.detach().to(device="cpu", dtype=torch.long).contiguous()
            for arg in args[num_problems:2 * num_problems]
        ]
        weights = [arg.contiguous() for arg in args[2 * num_problems:]]
        if len(weights) != num_problems:
            raise RuntimeError("grouped_gemm_multi received inconsistent input lengths")
        for x, weight in zip(xs, weights):
            if x.dtype != torch.float32 or weight.dtype != torch.float32:
                raise RuntimeError(f"cublas_grouped_gemm currently requires float32, got x={x.dtype}, weight={weight.dtype}")
            if not x.is_cuda or not weight.is_cuda:
                raise RuntimeError("cublas_grouped_gemm requires CUDA tensors")
        outputs = _load_extension().grouped_gemm_multi_forward_fp32(xs, ptrs, weights, bool(fast_tf32))
        ctx.num_problems = int(num_problems)
        ctx.fast_tf32 = bool(fast_tf32)
        ctx.save_for_backward(*(xs + ptrs + weights))
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        num_problems = ctx.num_problems
        saved = ctx.saved_tensors
        xs = list(saved[:num_problems])
        ptrs = list(saved[num_problems:2 * num_problems])
        weights = list(saved[2 * num_problems:])
        grad_outputs = [grad.contiguous() for grad in grad_outputs]
        ext = _load_extension()
        grad_xs = ext.grouped_gemm_multi_forward_fp32(
            grad_outputs,
            ptrs,
            [weight.transpose(1, 2).contiguous() for weight in weights],
            ctx.fast_tf32,
        )
        grad_weights = ext.grouped_gemm_multi_backward_weight_fp32(
            grad_outputs,
            xs,
            ptrs,
            ctx.fast_tf32,
        )
        return (None, None, *grad_xs, *([None] * num_problems), *grad_weights)


def grouped_gemm_multi(
    xs: list[torch.Tensor],
    ptrs: list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    """Apply several per-segment GEMM problem sets in one cuBLAS grouped call.

    Each entry follows ``out[start:end] = x[start:end] @ weight[g].T`` and may
    have a different input/output feature shape. All problems must be CUDA fp32
    tensors on the same device.
    """
    if not (len(xs) == len(ptrs) == len(weights)):
        raise RuntimeError("xs, ptrs, and weights must have the same length")
    if not xs:
        return []
    if fast_tf32 is None:
        fast_tf32 = _fast_tf32_enabled()
    outputs = _GroupedGemmMultiFunction.apply(
        bool(fast_tf32),
        len(xs),
        *(list(xs) + list(ptrs) + list(weights)),
    )
    return list(outputs)
