from __future__ import annotations

import pytest
import torch

from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    canonical_block_tensors_to_feature_tensors,
    feature_tensors_to_block_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import _inverse_contract_cg_hr
from dptb.nnops.block_flow_codec import BlockStateCodec


FP32_CODEC_ATOL = 2.0e-5
SUPPORTED_L = ("s", "p", "d", "f", "g", "h")
SUPPORTED_PAIRTYPES = tuple(
    f"{left}-{right}"
    for left_index, left in enumerate(SUPPORTED_L)
    for right in SUPPORTED_L[left_index:]
)


def _single_atom_reverse_pair(idp: OrbitalMapper, dtype: torch.dtype) -> dict:
    atom_type = idp.chemical_symbol_to_type["H"]
    edge_type = idp.bond_to_type["H-H"]
    return {
        "pos": torch.zeros((1, 3), dtype=dtype),
        "cell": torch.eye(3, dtype=dtype).unsqueeze(0),
        "pbc": torch.tensor([True, False, False]),
        "batch": torch.zeros((1,), dtype=torch.long),
        "atom_types": torch.tensor([atom_type], dtype=torch.long),
        "edge_type": torch.tensor([edge_type, edge_type], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor(
            [[1, 0, 0], [-1, 0, 0]], dtype=dtype
        ),
    }


def _shell_slot(idp: OrbitalMapper, shell: str) -> slice:
    return idp.orbital_maps["H"][shell]


def test_repeated_shell_canonical_slots_are_gathered_in_mapper_order():
    """Use explicit sentinels so pack/gather cannot hide a shared permutation."""
    idp = OrbitalMapper({"H": "3s2p1d"}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    data = _single_atom_reverse_pair(idp, torch.float64)
    width = int(idp.reduced_matrix_element)

    node_features = torch.arange(1, width + 1, dtype=torch.float64).unsqueeze(0)
    edge_features = torch.stack(
        (
            torch.arange(1001, 1001 + width, dtype=torch.float64),
            torch.arange(2001, 2001 + width, dtype=torch.float64),
        )
    )
    packed = feature_tensors_to_block_tensors(
        data,
        idp,
        node_features=node_features,
        edge_features=edge_features,
        symmetrize_onsite=True,
        complete_edges=True,
        strict_complete_edges=True,
    )

    # These hand-selected slots cross repeated s/s and p/p chunks as well as
    # different angular-momentum pair types.  Check both the direct write and
    # the transpose-completion side of the reverse edge explicitly.
    for shell_i, shell_j in (
        ("1s", "2s"),
        ("2s", "3s"),
        ("3s", "1p"),
        ("1p", "2p"),
        ("2p", "1d"),
    ):
        row = _shell_slot(idp, shell_i)
        col = _shell_slot(idp, shell_j)
        feature = idp.orbpair_maps[f"{shell_i}-{shell_j}"]
        height = row.stop - row.start
        width_slot = col.stop - col.start

        expected_node = node_features[0, feature].reshape(height, width_slot)
        torch.testing.assert_close(
            packed.node_blocks[0, row, col], expected_node, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            packed.node_blocks[0, col, row],
            expected_node.transpose(0, 1),
            rtol=0.0,
            atol=0.0,
        )

        expected_edge_0 = edge_features[0, feature].reshape(height, width_slot)
        expected_edge_1 = edge_features[1, feature].reshape(height, width_slot)
        torch.testing.assert_close(
            packed.edge_blocks[0, row, col], expected_edge_0, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            packed.edge_blocks[1, row, col], expected_edge_1, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            packed.edge_blocks[0, col, row],
            expected_edge_1.transpose(0, 1),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            packed.edge_blocks[1, col, row],
            expected_edge_0.transpose(0, 1),
            rtol=0.0,
            atol=0.0,
        )

    gathered = canonical_block_tensors_to_feature_tensors(
        data,
        idp,
        node_blocks=packed.node_blocks,
        edge_blocks=packed.edge_blocks,
        node_shapes=packed.node_shapes,
        edge_shapes=packed.edge_shapes,
        mode="strict",
        atol=0.0,
    )
    torch.testing.assert_close(
        gathered.node_features, node_features, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gathered.edge_features, edge_features, rtol=0.0, atol=0.0
    )


@pytest.fixture(scope="module")
def all_l_float32_case():
    idp = OrbitalMapper(
        {"H": ["1s", "2p", "3d", "4f", "5g", "6h"]},
        method="e3tb",
        device="cpu",
    )
    idp.get_orbital_maps()
    data = _single_atom_reverse_pair(idp, torch.float32)
    codec = BlockStateCodec(
        idp, dtype=torch.float32, inverse_mode="strict", atol=FP32_CODEC_ATOL
    )
    return idp, data, codec


@pytest.mark.parametrize("pairtype", SUPPORTED_PAIRTYPES)
def test_s_through_h_float32_standard_basis_inverse_for_every_pairtype(
    all_l_float32_case,
    pairtype,
):
    """Attack every coupled coordinate without reusing the forward CG helper."""
    _, _, codec = all_l_float32_case
    assert set(codec._contract.cgbasis) == set(SUPPORTED_PAIRTYPES)
    basis = codec._contract.cgbasis[pairtype]
    matrix = basis.reshape(-1, basis.shape[-1])
    assert matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]
    identity = torch.eye(size, dtype=torch.float32)

    # Each row is a distinct coupled-coordinate basis vector.  Build the
    # product-space columns manually instead of deriving an oracle through the
    # same forward contraction that the codec uses.
    product = matrix.transpose(0, 1).reshape(
        size, 1, basis.shape[0], basis.shape[1]
    )
    restored = _inverse_contract_cg_hr(basis, product)
    torch.testing.assert_close(
        restored.squeeze(-1), identity, rtol=0.0, atol=2.0e-6
    )


def test_s_through_h_float32_full_codec_round_trip(all_l_float32_case):
    idp, data, codec = all_l_float32_case
    norb = int(idp.norbs["H"])
    generator = torch.Generator().manual_seed(20260718)
    raw_node = torch.randn(norb, norb, dtype=torch.float32, generator=generator)
    node = (0.5 * (raw_node + raw_node.transpose(0, 1))).unsqueeze(0)
    edge_0 = torch.randn(norb, norb, dtype=torch.float32, generator=generator)
    edge = torch.stack((edge_0, edge_0.transpose(0, 1)))
    state = BlockTensorResult(
        node_blocks=node,
        edge_blocks=edge,
        node_shapes=torch.tensor([[norb, norb]], dtype=torch.long),
        edge_shapes=torch.tensor([[norb, norb], [norb, norb]], dtype=torch.long),
    )
    node_rme, edge_rme = codec.blocks_to_rme(data, state)
    rebuilt = codec.rme_to_blocks(data, node_rme, edge_rme)
    torch.testing.assert_close(
        rebuilt.node_blocks, node, rtol=0.0, atol=FP32_CODEC_ATOL
    )
    torch.testing.assert_close(
        rebuilt.edge_blocks, edge, rtol=0.0, atol=FP32_CODEC_ATOL
    )


def test_inverse_cg_rejects_rectangular_and_singular_bases():
    rectangular = torch.zeros((2, 3, 2), dtype=torch.float64)
    rectangular_product = torch.zeros((1, 1, 2, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="square complete basis"):
        _inverse_contract_cg_hr(rectangular, rectangular_product)

    singular = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64
    )
    singular_product = torch.ones((1, 1, 1, 2), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="singular/rank deficient"):
        _inverse_contract_cg_hr(singular, singular_product)
