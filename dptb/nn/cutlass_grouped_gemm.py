from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT = None
_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _cutlass_root() -> Path:
    root = (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT")
    )
    if not root:
        raise RuntimeError(
            "CUTLASS grouped GEMM backend requires DPTB_CUTLASS_ROOT or "
            "DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT."
        )
    root_path = Path(root)
    if not (root_path / "include" / "cutlass").exists():
        raise RuntimeError(f"CUTLASS include directory not found under {root_path}")
    return root_path


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    here = Path(__file__).resolve().parent
    root = _cutlass_root()
    build_dir = Path(
        os.environ.get(
            "DPTB_CUTLASS_GROUPED_BUILD_DIR",
            Path.home() / ".cache" / "dptb_cutlass_grouped_gemm",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_flags = ["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    if _flag("DPTB_CUTLASS_GROUPED_LINEINFO"):
        cuda_flags.append("-lineinfo")

    _EXT = load(
        name="dptb_cutlass_grouped_gemm",
        sources=[str(here / "csrc" / "cutlass_grouped_gemm.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=_flag("DPTB_CUTLASS_GROUPED_VERBOSE"),
    )
    return _EXT


def grouped_gemm(x: torch.Tensor, ptr: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CUTLASS grouped fp32 GEMM: out[start:end] = x[start:end] @ weight[g].T."""

    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise RuntimeError(f"cutlass_grouped_gemm requires float32, got x={x.dtype}, weight={weight.dtype}")
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("cutlass_grouped_gemm requires CUDA tensors")
    ptr_cpu = ptr.detach().to(device="cpu", dtype=torch.long).contiguous()
    return _load_extension().grouped_gemm_forward_fp32(x.contiguous(), ptr_cpu, weight.contiguous())


def grouped_gemm_backward_weight(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    ptr: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """CUTLASS grouped fp32 weight gradient: grad_w[g] = grad_out_g.T @ x_g."""

    if grad_out.dtype != torch.float32 or x.dtype != torch.float32:
        raise RuntimeError(
            f"cutlass_grouped_gemm grad weight requires float32, got grad_out={grad_out.dtype}, x={x.dtype}"
        )
    if not grad_out.is_cuda or not x.is_cuda:
        raise RuntimeError("cutlass_grouped_gemm grad weight requires CUDA tensors")
    ptr_cpu = ptr.detach().to(device="cpu", dtype=torch.long).contiguous()
    return _load_extension().grouped_gemm_backward_weight_fp32(
        grad_out.contiguous(),
        x.contiguous(),
        ptr_cpu,
        int(groups),
    )


class _GroupedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, ptr: torch.Tensor, weight: torch.Tensor):
        ptr_cpu = ptr.detach().to(device="cpu", dtype=torch.long).contiguous()
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        out = _load_extension().grouped_gemm_forward_fp32(x_contig, ptr_cpu, weight_contig)
        ctx.save_for_backward(x_contig, ptr_cpu, weight_contig)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, ptr_cpu, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        ext = _load_extension()
        grad_x = ext.grouped_gemm_grad_x_fp32(
            grad_out,
            ptr_cpu,
            weight,
        )
        grad_weight = ext.grouped_gemm_backward_weight_fp32(
            grad_out,
            x,
            ptr_cpu,
            int(weight.shape[0]),
        )
        return grad_x, None, grad_weight


def grouped_gemm_autograd(x: torch.Tensor, ptr: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise RuntimeError(f"cutlass_grouped_gemm requires float32, got x={x.dtype}, weight={weight.dtype}")
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("cutlass_grouped_gemm requires CUDA tensors")
    return _GroupedGemmFunction.apply(x, ptr, weight)
