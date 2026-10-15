"""Skip conditions for tests that need optional hardware or native components.

A test that needs one of these runs where it is present and is reported as skipped,
with the reason, where it is not; it never fails for the missing component itself.

"""
import importlib.util

import pytest
import torch


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_CUDA = torch.cuda.is_available()
HAS_SO2_CUDA = HAS_CUDA and _module_available("so2_cuda_ops")

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
requires_multi_gpu = pytest.mark.skipif(not HAS_CUDA or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
requires_so2_cuda = pytest.mark.skipif(not HAS_SO2_CUDA, reason="needs CUDA and SO2CUDA (so2_cuda_ops)")

def requires_module(name, reason=None):
    return pytest.mark.skipif(not _module_available(name), reason=reason or f"needs the optional package {name}")
