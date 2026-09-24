"""h0/tests_h0fast test infrastructure.

- ``h0_extension(name)`` marks a test as needing a verified, device-matching precompiled H0 CUDA
  extension (``h0rebuild.precompiled.verify(name, check_device=True)``); it is skipped, with the
  reason, wherever CUDA or that extension is missing or stale. Use it instead of a bare
  ``torch.cuda.is_available()`` check whenever the test path actually loads a compiled
  ``h0rebuild._cuda_*`` module -- CUDA can be present while the extension itself is not built,
  in which case the bare check does not skip and the test fails instead.
- ``h0_reference`` marks an opt-in real-ABACUS-data regression: skipped unless ``H0_REFERENCE_ROOT``
  points at an existing directory and ``pyabacus`` imports. Tests under this marker resolve their
  case directories from the ``h0_reference_root`` fixture, never from a hard-coded absolute path.
"""
import functools
import importlib
import os
from pathlib import Path

import pytest
import torch

H0_REFERENCE_ROOT = os.environ.get("H0_REFERENCE_ROOT")


def pytest_configure(config):
    config.addinivalue_line("markers", "h0_extension(name): needs CUDA and the verified H0 prebuilt extension `name`")
    config.addinivalue_line("markers", "h0_reference: opt-in real-ABACUS-data regression (H0_REFERENCE_ROOT + pyabacus)")


@functools.lru_cache(maxsize=None)
def _h0_extension_missing(name):
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        from h0rebuild.precompiled import verify
        verify(name, check_device=True)
    except (OSError, RuntimeError) as error:
        return f"H0 extension {name!r} unavailable: {error}"
    return None


@functools.lru_cache(maxsize=None)
def _h0_reference_missing():
    if not H0_REFERENCE_ROOT or not Path(H0_REFERENCE_ROOT).is_dir():
        return "set H0_REFERENCE_ROOT to a directory holding the real-ABACUS reference cases"
    try:
        importlib.import_module("pyabacus")
    except ImportError as error:
        return f"needs pyabacus: {error}"
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in item.iter_markers(name="h0_extension"):
            reason = _h0_extension_missing(marker.args[0])
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
        if item.get_closest_marker("h0_reference"):
            reason = _h0_reference_missing()
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def h0_reference_root():
    """Root of the opt-in reference-case tree, or None (tests using this are h0_reference-marked,
    so they are already skipped before this fixture would be evaluated with a None root)."""
    return Path(H0_REFERENCE_ROOT) if H0_REFERENCE_ROOT else None
