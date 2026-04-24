# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if p != str(_ROOT)]
sys.path.insert(0, str(_ROOT))

from dptb.nn.so2_triton_exact_gp_v2 import (
    reference_complex_exact_moe_linear,
    reference_exact_moe_linear,
)
from dptb.nn.so2_triton_exact_gp_v4 import (
    _scratch_fits,
    complex_exact_moe_linear_v4,
    exact_moe_linear_v4,
    use_complex_exact_gp_v4,
    use_exact_gp_v4,
)


def _clone_requires_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def _clear_require_flags(monkeypatch) -> None:
    # Match the latest branch-side V3 test isolation fix: fallback tests must not
    # inherit REQUIRE=1 from benchmark shells or parent CI jobs.  V4 cascades to
    # older REQUIRE flags, so clear the whole family.
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V4_REQUIRE", raising=False)
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", raising=False)
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V2_REQUIRE", raising=False)


def test_v4_flags(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    monkeypatch.delenv("DPTB_TRITON_COMPLEX_EXACT_GP_V4", raising=False)
    assert use_exact_gp_v4()
    assert use_complex_exact_gp_v4()
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V4", "0")
    assert not use_complex_exact_gp_v4()


def test_v4_scratch_limit(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB", "1")
    assert _scratch_fits(1, 1, 128, torch.float32)
    assert not _scratch_fits(1024, 1024, 1024, torch.float32)
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB", "0")
    assert _scratch_fits(1024, 1024, 1024, torch.float32)


def test_real_v4_cpu_fallback_matches_reference_forward_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    _clear_require_flags(monkeypatch)
    torch.manual_seed(201)
    split = (2, 0, 3, 1)
    x0 = torch.randn(sum(split), 5, dtype=torch.float64)
    c0 = torch.randn(len(split), 4, dtype=torch.float64)
    w0 = torch.randn(4, 7, 5, dtype=torch.float64)
    b0 = torch.randn(4, 7, dtype=torch.float64)
    sw0 = torch.randn(7, 5, dtype=torch.float64)
    sb0 = torch.randn(7, dtype=torch.float64)
    gout = torch.randn(sum(split), 7, dtype=torch.float64)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_ref = reference_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v4 = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_v4 = exact_moe_linear_v4(*args_v4, split)
    out_v4.backward(gout)

    torch.testing.assert_close(out_v4, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v4, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_real_v4_cpu_fallback_no_bias_no_shared(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    _clear_require_flags(monkeypatch)
    torch.manual_seed(202)
    split = (1, 4)
    x = torch.randn(sum(split), 3, dtype=torch.float32, requires_grad=True)
    c = torch.randn(len(split), 2, dtype=torch.float32, requires_grad=True)
    w = torch.randn(2, 6, 3, dtype=torch.float32, requires_grad=True)
    out = exact_moe_linear_v4(x, c, w, None, None, None, split)
    out.square().mean().backward()
    assert out.shape == (sum(split), 6)
    assert x.grad is not None and c.grad is not None and w.grad is not None


def test_complex_v4_cpu_fallback_matches_reference_forward_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    monkeypatch.delenv("DPTB_TRITON_COMPLEX_EXACT_GP_V4", raising=False)
    _clear_require_flags(monkeypatch)
    torch.manual_seed(203)
    split = (3, 2, 1)
    x0 = torch.randn(sum(split), 2, 4, dtype=torch.float64)
    c0 = torch.randn(len(split), 3, dtype=torch.float64)
    w0 = torch.randn(3, 2 * 5, 4, dtype=torch.float64)
    sw0 = torch.randn(2 * 5, 4, dtype=torch.float64)
    gout = torch.randn(sum(split), 2, 5, dtype=torch.float64)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_ref = reference_complex_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v4 = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_v4 = complex_exact_moe_linear_v4(*args_v4, split)
    out_v4.backward(gout)

    torch.testing.assert_close(out_v4, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v4, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_v4_require_fails_on_cpu(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4_REQUIRE", "1")
    split = (1,)
    x = torch.randn(1, 2)
    c = torch.randn(1, 1)
    w = torch.randn(1, 3, 2)
    with pytest.raises(RuntimeError, match="V4 forward needs CUDA"):
        exact_moe_linear_v4(x, c, w, None, None, None, split)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton V4 smoke test")
def test_cuda_v4_smoke_if_available(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4", "1")
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V4", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V4_BWD", os.environ.get("DPTB_TRITON_EXACT_GP_V4_BWD", "split_coeff"))
    device = torch.device("cuda")
    split = (5, 4)
    x = torch.randn(sum(split), 16, device=device, dtype=torch.float32, requires_grad=True)
    c = torch.randn(len(split), 4, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(4, 32, 16, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 32, device=device, dtype=torch.float32, requires_grad=True)
    sw = torch.randn(32, 16, device=device, dtype=torch.float32, requires_grad=True)
    sb = torch.randn(32, device=device, dtype=torch.float32, requires_grad=True)

    out = exact_moe_linear_v4(x, c, w, b, sw, sb, split)
    out.square().mean().backward()
    assert out.shape == (sum(split), 32)
    assert x.grad is not None and c.grad is not None and w.grad is not None
