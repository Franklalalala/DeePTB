from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


def _installer_module():
    path = Path(__file__).resolve().parents[2] / "docs" / "auto_install_torch_scatter.py"
    spec = importlib.util.spec_from_file_location("dptb_torch_scatter_installer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("torch_version", "cuda_version", "hip_version", "expected"),
    [
        ("2.5.1+cu121", None, None, "https://data.pyg.org/whl/torch-2.5.0+cu121.html"),
        ("2.5.1", "12.1", None, "https://data.pyg.org/whl/torch-2.5.0+cu121.html"),
        ("2.5.1+cpu", "12.1", None, "https://data.pyg.org/whl/torch-2.5.0+cpu.html"),
        ("2.5.1", None, None, "https://data.pyg.org/whl/torch-2.5.0+cpu.html"),
    ],
)
def test_torch_scatter_wheel_url(torch_version, cuda_version, hip_version, expected):
    installer = _installer_module()
    assert installer.torch_scatter_wheel_url(torch_version, cuda_version, hip_version) == expected


def test_rocm_requires_an_explicit_supported_install_path():
    installer = _installer_module()
    with pytest.raises(RuntimeError, match="ROCm torch-scatter wheels"):
        installer.torch_scatter_wheel_url("2.5.1+rocm6.2", None, "6.2")


def test_required_runtime_dependency_is_declared():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'torch-scatter = "2.1.2"' in text
    assert 'openequivariance = { version = "0.6.8", python = ">=3.10", optional = true }' in text
    # module-scope import in dptb/postprocess/write_abacus_csr_file.py
    assert re.search(r'^dftio = ">=', text, flags=re.M)
    # a torch bound must reach the resolver, not only e3nn's transitive >=1.8.0
    assert re.search(r'^torch = ">=', text, flags=re.M)


def test_test_requirements_are_pip_visible():
    """Poetry groups are invisible to `pip install .`, so the documented
    validation command needs a real extra."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert re.search(r'^test = \["pytest"\]', text, flags=re.M)
    assert "[tool.poetry.group.dev.dependencies]" not in text
    # repo-root `tools` package must be importable however pytest is invoked
    assert 'pythonpath = ["."]' in text

    ut = (Path(__file__).resolve().parents[2] / "ut.sh").read_text(encoding="utf-8")
    assert 'pip install ".[test]"' in ut
    assert "python -m pytest" in ut


def test_unused_dependency_is_not_declared():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert "opt-einsum" not in pyproject.read_text(encoding="utf-8")
