"""DeePTB adapter for the external SO2 CUDA materialized scheduler backend."""

from so2_cuda_ops import _scheduler_backend as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
