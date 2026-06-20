from pathlib import Path

import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.rme_nocg_fusion_head import (
    RMENoCGFusionHead,
    normalize_rme_head_mode,
)


def _make_head(init=0.0, dtype=torch.float64):
    return RMENoCGFusionHead(
        "4x0e + 3x1o + 2x2e",
        "3x0e + 2x1o + 2x2e",
        rank=5,
        init=init,
        dtype=dtype,
    )


def test_mode_normalization_and_validation():
    assert normalize_rme_head_mode(None) == "legacy_linear"
    assert normalize_rme_head_mode("nocg") == "rme_nocg_fusion"
    with pytest.raises(ValueError):
        normalize_rme_head_mode("expansion_uuw")


def test_zero_init_is_exact_legacy_projection():
    torch.manual_seed(7)
    head = _make_head(init=0.0)
    x = torch.randn(11, head.irreps_in.dim, dtype=torch.float64)
    actual = head(x)
    expected = head.legacy(x)
    assert torch.equal(actual, expected)


def test_residual_update_is_trainable_and_shape_preserving():
    torch.manual_seed(11)
    head = _make_head(init=1.0e-2)
    x = torch.randn(9, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    y = head(x)
    assert y.shape == (9, head.irreps_out.dim)
    loss = y.square().mean()
    loss.backward()
    assert head.scale_up.weight.grad is not None
    assert torch.isfinite(head.scale_up.weight.grad).all()


def test_rotation_equivariance():
    torch.manual_seed(19)
    head = _make_head(init=1.0e-2)
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    d_in = head.irreps_in.D_from_matrix(rotation)
    d_out = head.irreps_out.D_from_matrix(rotation)
    lhs = head(x @ d_in.T)
    rhs = head(x) @ d_out.T
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-9, atol=2.0e-9)


def test_strict_load_accepts_legacy_linear_state_dict():
    irreps_in = o3.Irreps("4x0e + 3x1o + 2x2e")
    irreps_out = o3.Irreps("3x0e + 2x1o + 2x2e")
    torch.manual_seed(23)
    legacy = o3.Linear(irreps_in, irreps_out, biases=True).to(dtype=torch.float64)
    head = RMENoCGFusionHead(
        irreps_in, irreps_out, rank=5, init=0.0, dtype=torch.float64
    )
    head.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(4, irreps_in.dim, dtype=torch.float64)
    assert torch.equal(head(x), legacy(x))


def test_new_head_source_has_no_coupling_decoder_call():
    source = Path(__file__).parents[1] / "nn" / "embedding" / "rme_nocg_fusion_head.py"
    text = source.read_text(encoding="utf-8")
    forbidden = "wigner" + "_3j"
    assert forbidden not in text
