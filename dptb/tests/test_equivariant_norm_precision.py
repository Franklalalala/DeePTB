"""Precision contract of EquivariantMergedRMSNormFlat.

A float64 model must stay float64 through the norm: the previous
implementation hard-cast activations to float32 inside forward, silently
capping every pass (forward and backward) at ~1e-7 relative accuracy even
when common_options.dtype=float64.
"""

import torch
from e3nn import o3

from dptb.nn.embedding.eqv3_grid_helpers import (
    EquivariantMergedRMSNormFlat,
    _get_grid_mats,
)


IRREPS = o3.Irreps("4x0e+4x1o+4x2e")


def _norm(dtype):
    return EquivariantMergedRMSNormFlat(
        IRREPS,
        eps=1e-12,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def test_norm_float64_retains_double_precision():
    torch.manual_seed(0)
    x64 = torch.randn(6, IRREPS.dim, dtype=torch.float64)
    # Perturbation far below float32 resolution but well above float64's.
    delta = 1e-9
    y_base = _norm(torch.float64)(x64)
    y_pert = _norm(torch.float64)(x64 + delta)
    diff = (y_pert - y_base).abs().max().item()
    assert y_base.dtype == torch.float64
    # A float32 internal cast rounds the 1e-9 perturbation away entirely
    # (relative eps ~1.2e-7 on O(1) values); float64 must propagate it.
    assert 0.0 < diff < 1e-6
    assert diff > 1e-11


def test_norm_float64_matches_float32_at_float32_scale():
    torch.manual_seed(1)
    x = torch.randn(5, IRREPS.dim)
    out32 = _norm(torch.float32)(x)
    out64 = _norm(torch.float64)(x.to(torch.float64))
    assert out32.dtype == torch.float32
    assert torch.allclose(out32.to(torch.float64), out64, atol=1e-5)


def test_norm_half_inputs_still_upcast_to_float32():
    x = torch.randn(4, IRREPS.dim, dtype=torch.float16)
    out = _norm(torch.float32)(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_grid_mat_cache_is_dtype_keyed():
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        to32, _ = _get_grid_mats(2, 2, "integral", (8, 8))
        torch.set_default_dtype(torch.float64)
        to64, _ = _get_grid_mats(2, 2, "integral", (8, 8))
    finally:
        torch.set_default_dtype(prior)
    assert to32.dtype == torch.float32
    assert to64.dtype == torch.float64
