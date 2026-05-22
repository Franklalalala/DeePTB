from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _cutlass_root() -> Path:
    root = (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT")
    )
    if not root:
        raise RuntimeError(
            "CUTLASS GemmUniversal smoke requires DPTB_CUTLASS_ROOT, "
            "DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT, or DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT."
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
            "DPTB_CUTLASS_SO2_GEMM_SMOKE_BUILD_DIR",
            Path.home() / ".cache" / "dptb_cutlass_so2_gemm_universal_smoke",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_flags = ["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    if _flag("DPTB_CUTLASS_SO2_GEMM_SMOKE_LINEINFO"):
        cuda_flags.append("-lineinfo")

    _EXT = load(
        name="dptb_cutlass_so2_gemm_universal_smoke",
        sources=[str(here / "csrc" / "cutlass_so2_gemm_universal_smoke.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=_flag("DPTB_CUTLASS_SO2_GEMM_SMOKE_VERBOSE"),
    )
    return _EXT


def gemm_universal_smoke(a: torch.Tensor, b_row_major_nk: torch.Tensor) -> torch.Tensor:
    """Single-problem CUTLASS GemmUniversal smoke: ``a @ b_row_major_nk.T``.

    This is intentionally isolated from the SO2 production path. It proves that
    Liyue's SM89 toolchain can build and run a CUTLASS GemmUniversal kernel from
    the DeePTB extension loader before we attach a DeePTB-specific epilogue and
    custom A-loader.
    """

    if a.dtype != torch.float32 or b_row_major_nk.dtype != torch.float32:
        raise RuntimeError(f"gemm_universal_smoke requires fp32, got {a.dtype} and {b_row_major_nk.dtype}")
    if not a.is_cuda or not b_row_major_nk.is_cuda:
        raise RuntimeError("gemm_universal_smoke requires CUDA tensors")
    if a.dim() != 2 or b_row_major_nk.dim() != 2:
        raise RuntimeError("gemm_universal_smoke expects a [M,K] and b [N,K]")
    if a.shape[1] != b_row_major_nk.shape[1]:
        raise RuntimeError(f"K mismatch: a={tuple(a.shape)} b={tuple(b_row_major_nk.shape)}")
    return _load_extension().gemm_universal_smoke_fp32(a.contiguous(), b_row_major_nk.contiguous())


def gemm_universal_pair_epilogue_smoke(
    pair: torch.Tensor,
    weight: torch.Tensor,
    wigner_l1: torch.Tensor,
) -> torch.Tensor:
    """CUTLASS mainloop + custom epilogue smoke for one fixed SO2 pair.

    Inputs are already packed as ``pair [E,2,Cin]``. The CUTLASS GEMM computes
    raw columns ``[rr, ii]``; the custom epilogue does not store that raw tensor.
    Instead, each accumulator contributes directly to a dense ``l=1,m=1``
    Wigner-rotated output ``[E, 3*Cout]``.
    """

    if pair.dtype != torch.float32 or weight.dtype != torch.float32 or wigner_l1.dtype != torch.float32:
        raise RuntimeError("gemm_universal_pair_epilogue_smoke requires fp32 tensors")
    if not pair.is_cuda or not weight.is_cuda or not wigner_l1.is_cuda:
        raise RuntimeError("gemm_universal_pair_epilogue_smoke requires CUDA tensors")
    if pair.dim() != 3 or pair.shape[1] != 2:
        raise RuntimeError(f"pair must be [E,2,Cin], got {tuple(pair.shape)}")
    if weight.dim() != 2 or weight.shape[0] % 2 != 0 or weight.shape[1] != pair.shape[2]:
        raise RuntimeError(f"weight must be [2*Cout,Cin], got {tuple(weight.shape)} for pair {tuple(pair.shape)}")
    if wigner_l1.shape != (pair.shape[0], 3, 3):
        raise RuntimeError(f"wigner_l1 must be [E,3,3], got {tuple(wigner_l1.shape)}")
    return _load_extension().gemm_universal_pair_epilogue_smoke_fp32(
        pair.contiguous(),
        weight.contiguous(),
        wigner_l1.contiguous(),
    )


def gemm_universal_raw_a_loader_pair_epilogue_smoke(
    x: torch.Tensor,
    weight: torch.Tensor,
    wigner_l1: torch.Tensor,
) -> torch.Tensor:
    """CUTLASS mainloop + SO2 raw-A loader + custom epilogue smoke.

    This is the Stage-3 bridge from a packed-A prototype to a DeePTB-shaped
    loader: logical A is ``[2*E,Cin]``, but values are loaded directly from
    ``x [E,3*Cin]`` and compact ``l=1`` Wigner blocks. The custom epilogue is
    the same direct Wigner output scatter used by
    :func:`gemm_universal_pair_epilogue_smoke`.
    """

    if x.dtype != torch.float32 or weight.dtype != torch.float32 or wigner_l1.dtype != torch.float32:
        raise RuntimeError("gemm_universal_raw_a_loader_pair_epilogue_smoke requires fp32 tensors")
    if not x.is_cuda or not weight.is_cuda or not wigner_l1.is_cuda:
        raise RuntimeError("gemm_universal_raw_a_loader_pair_epilogue_smoke requires CUDA tensors")
    if x.dim() != 2 or x.shape[1] % 3 != 0:
        raise RuntimeError(f"x must be [E,3*Cin], got {tuple(x.shape)}")
    cin = x.shape[1] // 3
    if weight.dim() != 2 or weight.shape[0] % 2 != 0 or weight.shape[1] != cin:
        raise RuntimeError(f"weight must be [2*Cout,Cin], got {tuple(weight.shape)} for x {tuple(x.shape)}")
    if wigner_l1.shape != (x.shape[0], 3, 3):
        raise RuntimeError(f"wigner_l1 must be [E,3,3], got {tuple(wigner_l1.shape)}")
    return _load_extension().gemm_universal_raw_a_loader_pair_epilogue_smoke_fp32(
        x.contiguous(),
        weight.contiguous(),
        wigner_l1.contiguous(),
    )
