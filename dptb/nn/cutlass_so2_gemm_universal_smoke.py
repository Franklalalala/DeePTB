"""Compatibility shim for optional CUTLASS SO2 smoke kernels."""

from so2_cuda_ops import _cutlass_so2_gemm_universal_smoke as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
