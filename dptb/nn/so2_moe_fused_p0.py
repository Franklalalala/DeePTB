"""DeePTB adapter for SO2 CUDA pack/scatter and indexed sandwich kernels."""

from so2_cuda_ops import tensor_product as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
