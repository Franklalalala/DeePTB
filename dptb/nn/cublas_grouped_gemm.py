"""Compatibility shim for the external SO2 CUDA backend package."""

from so2_cuda_ops import _cublas_grouped_gemm as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
