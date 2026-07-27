"""Optional CUDA/OEQ backends must fail with actionable errors, not noise."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# L12: SO(2) persistent-grouped backward backend
# ---------------------------------------------------------------------------
def test_missing_so2_backward_backend_raises_an_actionable_runtime_error(monkeypatch):
    from dptb.nn import so2_moe_fused_p0, so2_moe_persistent_grouped as pg

    # Reproduce the advertised install shape: the module imports fine but the
    # segmented backward entry points are not defined.
    monkeypatch.delattr(so2_moe_fused_p0, "_segmented_m0_backward", raising=False)
    monkeypatch.delattr(so2_moe_fused_p0, "_segmented_pair_backward", raising=False)

    with pytest.raises(RuntimeError, match=r'pip install "\.\[so2\]"'):
        pg._segmented_m0_backward()
    with pytest.raises(RuntimeError, match="_segmented_pair_backward"):
        pg._segmented_pair_backward()


def test_so2_backward_guard_does_not_swallow_a_present_backend(monkeypatch):
    from dptb.nn import so2_moe_fused_p0, so2_moe_persistent_grouped as pg

    monkeypatch.setattr(
        so2_moe_fused_p0, "_segmented_m0_backward", lambda *a, **k: "called", raising=False
    )
    assert pg._segmented_m0_backward() == "called"


# ---------------------------------------------------------------------------
# L13: openequivariance import-time print spam
# ---------------------------------------------------------------------------
OEQ_MODULES = (
    "dptb/nn/embedding/emoles.py",
    "dptb/nn/embedding/emoles_norm.py",
    "dptb/nn/embedding/emoles_norm_v2.py",
    "dptb/nn/embedding/lem_in_frame.py",
)


@pytest.mark.parametrize("relative_path", OEQ_MODULES)
def test_oeq_modules_do_not_print_at_import(relative_path):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "print(\"Warning: openequivariance" not in source
    assert "from dptb.nn.embedding.oeq_backend import" in source


def test_importing_the_embedding_package_is_silent():
    result = subprocess.run(
        [sys.executable, "-c", "import dptb.nn.embedding"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_missing_backend_warning_is_emitted_once(caplog, monkeypatch):
    from dptb.nn.embedding import oeq_backend

    monkeypatch.setattr(oeq_backend, "oeq", None)
    monkeypatch.setattr(oeq_backend, "_warned", False)

    with caplog.at_level(logging.WARNING, logger="dptb.nn.embedding.oeq_backend"):
        oeq_backend.warn_openequivariance_missing()
        oeq_backend.warn_openequivariance_missing()

    matching = [
        record
        for record in caplog.records
        if "openequivariance is not installed" in record.getMessage()
    ]
    assert len(matching) == 1
    assert '".[openequi]"' in matching[0].getMessage()


def test_no_warning_when_the_backend_is_present(caplog, monkeypatch):
    from dptb.nn.embedding import oeq_backend

    monkeypatch.setattr(oeq_backend, "oeq", object())
    monkeypatch.setattr(oeq_backend, "_warned", False)
    with caplog.at_level(logging.WARNING, logger="dptb.nn.embedding.oeq_backend"):
        oeq_backend.warn_openequivariance_missing()
    assert caplog.records == []
