from pathlib import Path

import torch
from e3nn import o3

from dptb.nn.embedding.block_native_head import (
    BlockNativeLinearHead,
    apply_ao_basis_mask,
    compact_blocks_to_species_layout,
    species_compact_index,
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


def test_block_native_linear_is_marked_debug_nonequivariant():
    head = BlockNativeLinearHead("3x0e+2x1o", max_norb=4, symmetrize=True)
    assert head.output_contract == "ao_block"
    assert head.bypasses_rme is True
    assert head.bypasses_e3hamiltonian is True
    assert head.uses_ict is False
    assert head.is_equivariant_output is False
    assert head.debug_only is True


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


def test_species_compact_index_orders_union_slots():
    # H-like species skips the union 3s slot (water 2s1p inside 3s2p1d).
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1],  # H: slots {0,1,3,4,5}, norb 5
            [1, 1, 1, 1, 1, 1],  # O: full basis, norb 6
        ],
        dtype=torch.bool,
    )
    index, norb = species_compact_index(mask)
    assert norb.tolist() == [5, 6]
    assert index[0, :5].tolist() == [0, 1, 3, 4, 5]
    assert index[1].tolist() == [0, 1, 2, 3, 4, 5]


def test_compact_blocks_moves_skipped_shell_rows_and_zeroes_tail():
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    index, norb = species_compact_index(mask)
    n = mask.shape[1]
    # Distinct values per (row, col) union slot so misplacement is detectable.
    base = torch.arange(n, dtype=torch.float32)
    blocks = base.view(1, n, 1) * 10.0 + base.view(1, 1, n)
    blocks = blocks.repeat(2, 1, 1)

    # Onsite-style: both sides use the H map for block 0, O map for block 1.
    row_index = index[torch.tensor([0, 1])]
    row_norb = norb[torch.tensor([0, 1])]
    compact = compact_blocks_to_species_layout(blocks, row_index, row_norb)

    # H block: contiguous rows/cols are union slots [0,1,3,4,5].
    expected_h = torch.tensor([0.0, 1.0, 3.0, 4.0, 5.0])
    assert torch.equal(compact[0, :5, :5], expected_h.view(-1, 1) * 10.0 + expected_h.view(1, -1))
    assert torch.count_nonzero(compact[0, 5:, :]) == 0
    assert torch.count_nonzero(compact[0, :, 5:]) == 0
    # O block: identity remap.
    assert torch.equal(compact[1], blocks[1])

    # Directed-edge-style: H rows against O columns.
    edge = compact_blocks_to_species_layout(
        blocks[:1],
        index[torch.tensor([0])],
        norb[torch.tensor([0])],
        index[torch.tensor([1])],
        norb[torch.tensor([1])],
    )
    assert torch.equal(edge[0, :5, :], expected_h.view(-1, 1) * 10.0 + base.view(1, -1))
    assert torch.count_nonzero(edge[0, 5:, :]) == 0


def test_block_native_head_has_no_wigner_3j_call():
    source = Path(__file__).parents[1] / "nn" / "embedding" / "block_native_head.py"
    text = source.read_text(encoding="utf-8")
    forbidden = "wigner" + "_3j"
    assert forbidden not in text
