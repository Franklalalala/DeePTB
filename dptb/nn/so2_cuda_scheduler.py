"""Lazy adapter for the optional external SO2 CUDA scheduler backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _backend():
    try:
        from so2_cuda_ops import scheduler as backend
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SO2 CUDA scheduler backend requires the optional `so2_cuda_ops` package. "
            "The legacy production SO2 paths use the in-repo fused/grouped extension "
            "loaders instead; install `so2_cuda_ops` only for the scheduler experiment."
        ) from exc
    return backend


def mainloop_kind(*args: Any, **kwargs: Any):
    return _backend().mainloop_kind(*args, **kwargs)


def prepare_so2_single_route_layout(*args: Any, **kwargs: Any):
    return _backend().prepare_so2_single_route_layout(*args, **kwargs)


def wigner_tensor_and_mode(*args: Any, **kwargs: Any):
    return _backend().wigner_tensor_and_mode(*args, **kwargs)


class SO2CudaSchedulerFunction:
    @staticmethod
    def apply(*args: Any, **kwargs: Any):
        return _backend().SO2CudaSchedulerFunction.apply(*args, **kwargs)
