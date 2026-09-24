"""block_native_linear head: mode aliases, output shapes and species-compact slot remapping."""
import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.block_native_head import (
    BlockNativeLinearHead,
    compact_blocks_to_species_layout,
    species_compact_index,
)
from dptb.nn.embedding.rme_nocg_fusion_head import normalize_rme_head_mode

# H-like species skips the union 3s slot (water 2s1p inside 3s2p1d); O uses the full basis.
_SPECIES_MASK = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
_H_SLOTS = [0, 1, 3, 4, 5]


@pytest.mark.parametrize("alias", ["block_native", "block_linear"])
def test_block_native_mode_aliases(alias):
    assert normalize_rme_head_mode(alias) == "block_native_linear"


@pytest.mark.parametrize("symmetrize", [True, False], ids=["onsite", "edge"])
def test_block_native_head_shapes_and_onsite_symmetry(symmetrize):
    irreps = o3.Irreps("3x0e+2x1o")
    head = BlockNativeLinearHead(irreps, max_norb=4, symmetrize=symmetrize, init=0.01)
    blocks = head(torch.randn(5, irreps.dim, generator=torch.Generator().manual_seed(0)))
    assert blocks.shape == (5, 4, 4)
    if symmetrize:
        torch.testing.assert_close(blocks, blocks.transpose(-1, -2), rtol=0.0, atol=1e-6)


def test_species_compact_index_orders_union_slots():
    index, norb = species_compact_index(_SPECIES_MASK)
    assert norb.tolist() == [5, 6]
    assert index[0, :5].tolist() == _H_SLOTS
    assert index[1].tolist() == [0, 1, 2, 3, 4, 5]


def test_compact_blocks_move_skipped_shell_rows_and_zero_the_tail():
    index, norb = species_compact_index(_SPECIES_MASK)
    n = _SPECIES_MASK.shape[1]
    base = torch.arange(n, dtype=torch.float32)
    blocks = (base.view(1, n, 1) * 10.0 + base.view(1, 1, n)).repeat(2, 1, 1)  # distinct value per slot pair
    expected_h = torch.tensor(_H_SLOTS, dtype=torch.float32)

    onsite = compact_blocks_to_species_layout(blocks, index[[0, 1]], norb[[0, 1]])
    assert torch.equal(onsite[0, :5, :5], expected_h.view(-1, 1) * 10.0 + expected_h.view(1, -1))
    assert torch.count_nonzero(onsite[0, 5:, :]) == 0
    assert torch.count_nonzero(onsite[0, :, 5:]) == 0
    assert torch.equal(onsite[1], blocks[1])

    # Directed edge: H rows against O columns.
    edge = compact_blocks_to_species_layout(blocks[:1], index[[0]], norb[[0]], index[[1]], norb[[1]])
    assert torch.equal(edge[0, :5, :], expected_h.view(-1, 1) * 10.0 + base.view(1, -1))
    assert torch.count_nonzero(edge[0, 5:, :]) == 0
