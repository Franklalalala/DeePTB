# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if p != str(_ROOT)]
sys.path.insert(0, str(_ROOT))

try:
    from dptb.nn.so2_triton_exact_gp_v2 import (
        complex_exact_moe_linear_v2,
        exact_moe_linear_v2,
        reference_complex_exact_moe_linear,
        reference_exact_moe_linear,
    )
except Exception:  # pragma: no cover - lets this test run in overlay mode too
    _PATH = _ROOT / "dptb" / "nn" / "so2_triton_exact_gp_v2.py"
    spec = importlib.util.spec_from_file_location("so2_triton_exact_gp_v2", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exact_moe_linear_v2 = mod.exact_moe_linear_v2
    complex_exact_moe_linear_v2 = mod.complex_exact_moe_linear_v2
    reference_exact_moe_linear = mod.reference_exact_moe_linear
    reference_complex_exact_moe_linear = mod.reference_complex_exact_moe_linear


def _clone_requires_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def _assert_grads_close(got_args, exp_args, *, rtol: float, atol: float) -> None:
    for got, exp in zip(got_args, exp_args):
        torch.testing.assert_close(got.grad, exp.grad, rtol=rtol, atol=atol)


def test_real_cpu_fallback_matches_reference_forward_and_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2", "0")
    torch.manual_seed(7)
    split = (3, 2, 4)
    x0 = torch.randn(sum(split), 5, dtype=torch.float64)
    c0 = torch.randn(len(split), 4, dtype=torch.float64)
    w0 = torch.randn(4, 6, 5, dtype=torch.float64)
    b0 = torch.randn(4, 6, dtype=torch.float64)
    sw0 = torch.randn(6, 5, dtype=torch.float64)
    sb0 = torch.randn(6, dtype=torch.float64)
    gout = torch.randn(sum(split), 6, dtype=torch.float64)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_ref = reference_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v2 = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_v2 = exact_moe_linear_v2(*args_v2, split)
    out_v2.backward(gout)

    torch.testing.assert_close(out_v2, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v2, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_grouped_real_dispatch_fails_fast_when_v2_import_is_missing(monkeypatch):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setattr(ops, "_exact_moe_linear_v2", None)
    monkeypatch.setattr(ops, "_use_exact_gp_v2", lambda: True)

    split = (1,)
    x = torch.randn(1, 3)
    c = torch.randn(1, 2)
    w = torch.randn(2, 4, 3)

    with pytest.raises(RuntimeError, match="DPTB_TRITON_EXACT_GP_V2"):
        ops.grouped_exact_moe_linear(x, c, w, None, None, None, split)


def test_real_cpu_fallback_supports_no_bias_no_shared(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2", "0")
    torch.manual_seed(11)
    split = (1, 4)
    x = torch.randn(sum(split), 3, dtype=torch.float32, requires_grad=True)
    c = torch.randn(len(split), 2, dtype=torch.float32, requires_grad=True)
    w = torch.randn(2, 7, 3, dtype=torch.float32, requires_grad=True)
    out = exact_moe_linear_v2(x, c, w, None, None, None, split)
    loss = out.square().mean()
    loss.backward()
    assert out.shape == (sum(split), 7)
    assert x.grad is not None and c.grad is not None and w.grad is not None


def test_complex_cpu_fallback_matches_reference_forward_and_backward(monkeypatch):
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2", "0")
    torch.manual_seed(13)
    split = (2, 3, 1)
    x0 = torch.randn(sum(split), 2, 4, dtype=torch.float64)
    c0 = torch.randn(len(split), 3, dtype=torch.float64)
    w0 = torch.randn(3, 2 * 5, 4, dtype=torch.float64)
    sw0 = torch.randn(2 * 5, 4, dtype=torch.float64)
    gout = torch.randn(sum(split), 2, 5, dtype=torch.float64)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_ref = reference_complex_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v2 = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_v2 = complex_exact_moe_linear_v2(*args_v2, split)
    out_v2.backward(gout)

    torch.testing.assert_close(out_v2, out_ref, rtol=1e-10, atol=1e-10)
    for got, exp in zip(args_v2, args_ref):
        torch.testing.assert_close(got.grad, exp.grad, rtol=1e-10, atol=1e-10)


def test_grouped_complex_dispatch_fails_fast_when_v2_import_is_missing(monkeypatch):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setattr(ops, "_complex_exact_moe_linear_v2", None)
    monkeypatch.setattr(ops, "_use_complex_exact_gp_v2", lambda: True)

    split = (1,)
    x = torch.randn(1, 2, 3)
    c = torch.randn(1, 2)
    w = torch.randn(2, 2 * 4, 3)

    with pytest.raises(RuntimeError, match="DPTB_TRITON_COMPLEX_EXACT_GP_V2"):
        ops.grouped_complex_exact_moe_linear(x, c, w, None, split)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton smoke test")
@pytest.mark.parametrize("bwd_mode", ("torch", "atomic"))
def test_cuda_real_v2_matches_reference_forward_and_backward(monkeypatch, bwd_mode):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2_BWD", bwd_mode)
    device = torch.device("cuda")
    torch.manual_seed(17)
    split = (3, 2, 4)
    x0 = torch.randn(sum(split), 7, device=device, dtype=torch.float32)
    c0 = torch.randn(len(split), 3, device=device, dtype=torch.float32)
    w0 = torch.randn(3, 8, 7, device=device, dtype=torch.float32)
    b0 = torch.randn(3, 8, device=device, dtype=torch.float32)
    sw0 = torch.randn(8, 7, device=device, dtype=torch.float32)
    sb0 = torch.randn(8, device=device, dtype=torch.float32)
    gout = torch.randn(sum(split), 8, device=device, dtype=torch.float32)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_ref = reference_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v2 = [_clone_requires_grad(t) for t in (x0, c0, w0, b0, sw0, sb0)]
    out_v2 = exact_moe_linear_v2(*args_v2, split)
    out_v2.backward(gout)

    torch.testing.assert_close(out_v2, out_ref, rtol=2e-4, atol=2e-4)
    _assert_grads_close(args_v2, args_ref, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton smoke test")
@pytest.mark.parametrize("bwd_mode", ("torch", "atomic"))
def test_cuda_complex_v2_matches_reference_forward_and_backward(monkeypatch, bwd_mode):
    pytest.importorskip("triton")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2", "1")
    monkeypatch.setenv("DPTB_TRITON_COMPLEX_EXACT_GP_V2", "1")
    monkeypatch.setenv("DPTB_TRITON_EXACT_GP_V2_BWD", bwd_mode)
    device = torch.device("cuda")
    torch.manual_seed(19)
    split = (2, 3)
    x0 = torch.randn(sum(split), 2, 6, device=device, dtype=torch.float32)
    c0 = torch.randn(len(split), 3, device=device, dtype=torch.float32)
    w0 = torch.randn(3, 2 * 7, 6, device=device, dtype=torch.float32)
    sw0 = torch.randn(2 * 7, 6, device=device, dtype=torch.float32)
    gout = torch.randn(sum(split), 2, 7, device=device, dtype=torch.float32)

    args_ref = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_ref = reference_complex_exact_moe_linear(*args_ref, split)
    out_ref.backward(gout)

    args_v2 = [_clone_requires_grad(t) for t in (x0, c0, w0, sw0)]
    out_v2 = complex_exact_moe_linear_v2(*args_v2, split)
    out_v2.backward(gout)

    torch.testing.assert_close(out_v2, out_ref, rtol=2e-4, atol=2e-4)
    _assert_grads_close(args_v2, args_ref, rtol=3e-4, atol=3e-4)
