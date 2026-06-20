from pathlib import Path

import torch
from e3nn import o3

from dptb.nn.embedding.block_native_head import (
    BlockNativeLinearHead,
    apply_ao_basis_mask,
)
from dptb.nn.embedding.rme_nocg_fusion_head import normalize_rme_head_mode


def test_block_native_mode_aliases():
    assert normalize_rme_head_mode("block_native") == "block_native_linear"
    assert normalize_rme_head_mode("block_linear") == "block_native_linear"


def test_block_native_head_shapes_and_onsite_symmetry():
    irreps = o3.Irreps("3x0e+2x1o")
    head = BlockNativeLinearHead(irreps, max_norb=4, symmetrize=True, init=0.01)
    features = torch.randn(5, irreps.dim)
    blocks = head(features)
    assert blocks.shape == (5, 4, 4)
    assert torch.allclose(blocks, blocks.transpose(-1, -2), atol=1e-6)


def test_block_native_edge_head_is_directed():
    irreps = o3.Irreps("2x0e+1x1o")
    head = BlockNativeLinearHead(irreps, max_norb=3, symmetrize=False, init=0.01)
    features = torch.randn(7, irreps.dim)
    blocks = head(features)
    assert blocks.shape == (7, 3, 3)


def test_apply_ao_basis_mask_zeros_padding():
    blocks = torch.ones(2, 4, 4)
    row_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    col_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
    masked = apply_ao_basis_mask(blocks, row_mask, col_mask)
    expected = row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
    assert torch.equal(masked.bool(), expected)


def test_block_native_head_has_no_wigner_3j_call():
    source = Path(__file__).parents[1] / "nn" / "embedding" / "block_native_head.py"
    text = source.read_text(encoding="utf-8")
    forbidden = "wigner" + "_3j"
    assert forbidden not in text
