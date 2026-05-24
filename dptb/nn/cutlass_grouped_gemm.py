"""Compatibility shim for optional CUTLASS grouped GEMM support."""

from so2_cuda_ops import _cutlass_grouped_gemm as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
