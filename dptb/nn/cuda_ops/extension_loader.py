"""Compatibility shim for the external SO2 CUDA extension loader."""

from so2_cuda_ops import _extension_loader as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
