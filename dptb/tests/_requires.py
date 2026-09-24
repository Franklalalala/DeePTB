"""Skip conditions for tests that need optional hardware or native components.

A test that needs one of these runs where it is present and is reported as skipped,
with the reason, where it is not; it never fails for the missing component itself.

The NACF native components are checked lazily: mark a test (or a parameter) with
``requires_nacf_prebuilt`` / ``requires_nacf_topology`` and conftest.py skips it when the
component is missing or stale.
"""
import functools
import importlib.util
import os
from pathlib import Path

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

# resolved by conftest.py when a marked test is collected
requires_nacf_prebuilt = pytest.mark.nacf_prebuilt
requires_nacf_topology = pytest.mark.nacf_topology


def requires_module(name, reason=None):
    return pytest.mark.skipif(not _module_available(name), reason=reason or f"needs the optional package {name}")


@functools.lru_cache(maxsize=None)
def nacf_missing(kind):
    """Why the NACF native component `kind` cannot be used here, or None."""
    try:
        if kind == "nacf_prebuilt":
            if not HAS_CUDA:
                return "needs CUDA"
            from dptb.nacf.precompiled import load, verify

            verify(torch.device("cuda", torch.cuda.current_device()))
            absent = [s for s in ("pack_out", "onsite_density") if not hasattr(load(), s)]
        elif kind == "nacf_topology":
            path = os.environ.get("DPTB_NACF_TOPOLOGY_LIBRARY")
            if not path:
                return "needs DPTB_NACF_TOPOLOGY_LIBRARY (build tools/build_nacf_topology.py)"
            from dptb.nacf.topology import _library

            lib = _library(str(Path(path).resolve(strict=True)))
            absent = [s for s in ("nacf_topology_build_mode", "nacf_onsite_neighbours") if not hasattr(lib, s)]
        else:
            raise ValueError(kind)
    except (OSError, RuntimeError, ImportError) as error:
        return f"{kind} unavailable: {error}"
    return f"{kind} is stale (lacks {absent}); rebuild it" if absent else None
