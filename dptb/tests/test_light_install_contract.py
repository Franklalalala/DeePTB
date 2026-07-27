from __future__ import annotations

import ast
import importlib.util
import re
import sys
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
    # a torch bound must reach the resolver, not only e3nn's transitive >=1.8.0
    assert re.search(r'^torch = ">=', text, flags=re.M)


def test_no_dependency_is_declared_that_pypi_cannot_resolve():
    """`pip install .` must resolve from a bare index. dftio has no PyPI
    release at all, so declaring it -- as a requirement or as an extra whose
    source is a git URL -- breaks the documented install for everyone."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert not re.search(r"^dftio\s*=", text, flags=re.M)
    assert not re.search(r'^\w+\s*=\s*\[[^\]]*"dftio"', text, flags=re.M)


DFTIO_LAZY_MODULES = (
    "dptb/postprocess/write_abacus_csr_file.py",
    "dptb/postprocess/hrebuild_abacus_io.py",
)


@pytest.mark.parametrize("relative_path", DFTIO_LAZY_MODULES)
def test_dftio_is_never_imported_at_module_scope(relative_path):
    source = (Path(__file__).resolve().parents[2] / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dftio"), relative_path
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("dftio"), relative_path


def test_missing_dftio_names_the_feature_and_the_install_command():
    from dptb.postprocess.write_abacus_csr_file import DFTIO_MISSING_MESSAGE

    assert "hrebuild" in DFTIO_MISSING_MESSAGE
    assert "ABACUS CSR" in DFTIO_MISSING_MESSAGE
    assert "pip install git+https://github.com/deepmodeling/dftio.git" in DFTIO_MISSING_MESSAGE


def test_the_abacus_transform_fails_closed_when_dftio_is_absent(monkeypatch):
    """The lazy import must surface the actionable message at the use site,
    not a bare `ModuleNotFoundError: No module named 'dftio'`."""
    from dptb.postprocess import write_abacus_csr_file as csr

    monkeypatch.setitem(sys.modules, "dftio", None)
    monkeypatch.setitem(sys.modules, "dftio.constants", None)
    csr.abacus2dftio_matrices.cache_clear()
    csr.dftio2abacus_matrices.cache_clear()
    try:
        with pytest.raises(ImportError, match="dftio has no PyPI release"):
            csr.transform_2_ABACUS(None, [0], [0])
    finally:
        csr.abacus2dftio_matrices.cache_clear()
        csr.dftio2abacus_matrices.cache_clear()


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
