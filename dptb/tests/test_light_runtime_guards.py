from __future__ import annotations

import pytest

from dptb.data.build import _validate_cutoff_coverage
from dptb.nn.cuda_ops import extension_loader


def test_dictionary_cutoff_coverage_accepts_valid_values():
    _validate_cutoff_coverage(
        "r_max",
        {"H-H": 5.0, "H-C": 6.0},
        {"H-H": 4.0, "H-C": 6.0},
    )


def test_dictionary_cutoff_coverage_rejects_small_value():
    with pytest.raises(ValueError, match="offending values"):
        _validate_cutoff_coverage(
            "r_max",
            {"H-H": 3.5},
            {"H-H": 4.0},
        )


def test_dictionary_cutoff_coverage_rejects_missing_key():
    with pytest.raises(ValueError, match="missing model cutoff keys"):
        _validate_cutoff_coverage(
            "r_max",
            {"H-H": 5.0},
            {"H-H": 4.0, "H-C": 4.5},
        )


def test_scalar_cutoff_coverage_accepts_int_or_float():
    _validate_cutoff_coverage("r_max", 5, 4.5)


def test_scalar_cutoff_coverage_rejects_small_value():
    with pytest.raises(ValueError, match="smaller than model"):
        _validate_cutoff_coverage("r_max", 4.0, 5.0)


def _missing_module(name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named {name!r}")
    exc.name = name
    return exc


def test_cuda_loader_falls_back_when_optional_package_is_absent(monkeypatch):
    def fake_import(name):
        assert name == "so2_cuda_ops._extension_loader"
        raise _missing_module("so2_cuda_ops")

    monkeypatch.setattr(extension_loader.importlib, "import_module", fake_import)
    assert extension_loader._load_external_backend() is None


def test_cuda_loader_does_not_mask_missing_backend_submodule(monkeypatch):
    def fake_import(name):
        assert name == "so2_cuda_ops._extension_loader"
        raise _missing_module("so2_cuda_ops._extension_loader")

    monkeypatch.setattr(extension_loader.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="so2_cuda_ops._extension_loader"):
        extension_loader._load_external_backend()


def test_cuda_loader_does_not_mask_transitive_missing_dependency(monkeypatch):
    def fake_import(name):
        assert name == "so2_cuda_ops._extension_loader"
        raise _missing_module("backend_runtime_dependency")

    monkeypatch.setattr(extension_loader.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="backend_runtime_dependency"):
        extension_loader._load_external_backend()


def test_cuda_loader_returns_installed_backend(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        extension_loader.importlib,
        "import_module",
        lambda name: marker,
    )
    assert extension_loader._load_external_backend() is marker
