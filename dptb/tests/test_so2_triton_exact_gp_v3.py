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
from dptb.nn.so2_triton_exact_gp_v3 import (
    complex_exact_moe_linear_v3,
    exact_moe_linear_v3,
    use_complex_exact_gp_v3,
    use_exact_gp_v3,
)


def _clone_requires_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def test_v3_flags(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.delenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", raising=False)
    assert use_exact_gp_v3()
    assert use_complex_exact_gp_v3()
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", "0")
    assert not use_complex_exact_gp_v3()


def test_real_v3_cpu_fallback_matches_reference_forward_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", raising=False)
    torch.manual_seed(101)
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

    args_v3 = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_v3 = exact_moe_linear_v3(*args_v3, split)
    out_v3.backward(gout)

    torch.testing.assert_close(out_v3, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v3, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_real_v3_cpu_fallback_no_bias_no_shared(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", raising=False)
    torch.manual_seed(102)
    split = (1, 4)
    x = torch.randn(sum(split), 3, dtype=torch.float32, requires_grad=True)
    c = torch.randn(len(split), 2, dtype=torch.float32, requires_grad=True)
    w = torch.randn(2, 6, 3, dtype=torch.float32, requires_grad=True)
    out = exact_moe_linear_v3(x, c, w, None, None, None, split)
    out.square().mean().backward()
    assert out.shape == (sum(split), 6)
    assert x.grad is not None and c.grad is not None and w.grad is not None


def test_complex_v3_cpu_fallback_matches_reference_forward_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.delenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", raising=False)
    monkeypatch.delenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", raising=False)
    torch.manual_seed(103)
    split = (3, 2, 1)
    x0 = torch.randn(sum(split), 2, 4, dtype=torch.float64)
    c0 = torch.randn(len(split), 3, dtype=torch.float64)
    w0 = torch.randn(3, 2 * 5, 4, dtype=torch.float64)
    sw0 = torch.randn(2 * 5, 4, dtype=torch.float64)
    gout = torch.randn(sum(split), 2, 5, dtype=torch.float64)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_ref = reference_complex_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v3 = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_v3 = complex_exact_moe_linear_v3(*args_v3, split)
    out_v3.backward(gout)

    torch.testing.assert_close(out_v3, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v3, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_v3_require_fails_on_cpu(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", "1")
    split = (1,)
    x = torch.randn(1, 2)
    c = torch.randn(1, 1)
    w = torch.randn(1, 3, 2)
    with pytest.raises(RuntimeError, match="V3 forward needs CUDA"):
        exact_moe_linear_v3(x, c, w, None, None, None, split)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton V3 smoke test")
def test_cuda_v3_smoke_if_available(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_BWD", os.environ.get("DPTB_TRITON_EXACT_GP_V3_BWD", "expert_loop"))
    device = torch.device("cuda")
    split = (5, 4)
    x = torch.randn(sum(split), 16, device=device, dtype=torch.float32, requires_grad=True)
    c = torch.randn(len(split), 4, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(4, 32, 16, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 32, device=device, dtype=torch.float32, requires_grad=True)
    sw = torch.randn(32, 16, device=device, dtype=torch.float32, requires_grad=True)
    sb = torch.randn(32, device=device, dtype=torch.float32, requires_grad=True)

    out = exact_moe_linear_v3(x, c, w, b, sw, sb, split)
    out.square().mean().backward()
    assert out.shape == (sum(split), 32)
    assert x.grad is not None and c.grad is not None and w.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton V3 parity test")
@pytest.mark.parametrize("bwd_mode", ("expert_loop", "v2_atomic", "torch"))
def test_cuda_real_v3_matches_reference_forward_and_backward(monkeypatch, bwd_mode):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_BWD", bwd_mode)
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", "1")
    torch.manual_seed(301)
    device = torch.device("cuda")
    split = (3, 0, 4)
    x0 = torch.randn(sum(split), 6, device=device, dtype=torch.float32)
    c0 = torch.randn(len(split), 3, device=device, dtype=torch.float32)
    w0 = torch.randn(3, 8, 6, device=device, dtype=torch.float32)
    b0 = torch.randn(3, 8, device=device, dtype=torch.float32)
    sw0 = torch.randn(8, 6, device=device, dtype=torch.float32)
    sb0 = torch.randn(8, device=device, dtype=torch.float32)
    gout = torch.randn(sum(split), 8, device=device, dtype=torch.float32)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_ref = reference_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v3 = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_v3 = exact_moe_linear_v3(*args_v3, split)
    out_v3.backward(gout)

    torch.testing.assert_close(out_v3, out_ref, rtol=1e-4, atol=2e-4)
    for got, exp in zip(args_v3, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=2e-4, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton V3 parity test")
@pytest.mark.parametrize("bwd_mode", ("expert_loop", "v2_atomic", "torch"))
def test_cuda_complex_v3_matches_reference_forward_and_backward(monkeypatch, bwd_mode):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V3", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_BWD", bwd_mode)
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V3_REQUIRE", "1")
    torch.manual_seed(302)
    device = torch.device("cuda")
    split = (2, 5)
    x0 = torch.randn(sum(split), 2, 5, device=device, dtype=torch.float32)
    c0 = torch.randn(len(split), 3, device=device, dtype=torch.float32)
    w0 = torch.randn(3, 2 * 7, 5, device=device, dtype=torch.float32)
    sw0 = torch.randn(2 * 7, 5, device=device, dtype=torch.float32)
    gout = torch.randn(sum(split), 2, 7, device=device, dtype=torch.float32)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_ref = reference_complex_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v3 = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_v3 = complex_exact_moe_linear_v3(*args_v3, split)
    out_v3.backward(gout)

    torch.testing.assert_close(out_v3, out_ref, rtol=1e-4, atol=2e-4)
    for got, exp in zip(args_v3, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=2e-4, atol=5e-4)
