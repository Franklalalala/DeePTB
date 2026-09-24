"""Suite-wide test infrastructure.

- Default dtype guard: a test module must leave torch's default dtype as it found it.
  Modules that compute in float64 set it in a fixture and restore it; a module that changes
  it at import (seen during collection) or leaks it past its tests would turn later float32
  modules into mixed-dtype runs, so the guard fails that module.
- Deterministic-algorithms guard: a module that switches on
  ``torch.use_deterministic_algorithms`` must switch it back; the guard restores the
  previous mode and fails that module.  ``CUBLAS_WORKSPACE_CONFIG`` is set before CUDA
  starts so a deterministic CUDA test works regardless of which module ran first.
- Native-component markers: tests marked ``nacf_prebuilt`` / ``nacf_topology`` (see
  _requires.py) are skipped, with the reason, where that component is missing or stale.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402

_DTYPE_BEFORE_IMPORT = {}
_NATIVE_MARKERS = {
    "nacf_prebuilt": "needs CUDA and the verified dptb/nacf prebuilt binary (python -m dptb.nacf.precompile)",
    "nacf_topology": "needs a current NACF topology library in DPTB_NACF_TOPOLOGY_LIBRARY",
}


def pytest_configure(config):
    for name, doc in _NATIVE_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {doc}")


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _DTYPE_BEFORE_IMPORT[collector.nodeid] = torch.get_default_dtype()


@pytest.hookimpl(tryfirst=True)
def pytest_collectreport(report):
    before = _DTYPE_BEFORE_IMPORT.pop(report.nodeid, None)
    after = torch.get_default_dtype()
    if before is not None and after != before:
        torch.set_default_dtype(before)
        report.outcome = "failed"
        report.longrepr = f"{report.nodeid} set the default dtype to {after} at import (was {before})"


def pytest_collection_modifyitems(config, items):
    marked = [(item, kind) for item in items for kind in _NATIVE_MARKERS if item.get_closest_marker(kind)]
    if not marked:
        return
    from dptb.tests._requires import nacf_missing

    for item, kind in marked:
        reason = nacf_missing(kind)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True, scope="module")
def _default_dtype_is_restored(request):
    before = torch.get_default_dtype()
    yield
    after = torch.get_default_dtype()
    if after != before:
        torch.set_default_dtype(before)
        pytest.fail(f"{request.module.__name__} left the default dtype at {after} (was {before})", pytrace=False)


@pytest.fixture(autouse=True, scope="module")
def _deterministic_mode_is_restored(request):
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    yield
    after = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    if after != before:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        pytest.fail(f"{request.module.__name__} left torch.use_deterministic_algorithms at {after} (was {before})",
                    pytrace=False)
