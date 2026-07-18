from __future__ import annotations

import pytest
import torch

from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    canonical_block_tensors_to_feature_tensors,
    strict_reverse_edge_index,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state


FP64_FLOOR = 1e-10


def _tol(x: torch.Tensor, dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return max(FP64_FLOOR, 1e-12 * float(x.abs().max().item()))
    return max(2e-5, 2e-5 * float(x.abs().max().item()))


def _mixed_case(dtype: torch.dtype = torch.float64):
    # C has only a p shell.  Its compact AO indices start at zero although the
    # union canvas contains H's s shell first: this is the non-prefix case.
    idp = OrbitalMapper(
        {"H": ["1s"], "C": ["2p"]}, method="e3tb", device="cpu"
    )
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    atom_types = torch.tensor(
        [
            idp.chemical_symbol_to_type["H"],
            idp.chemical_symbol_to_type["C"],
            idp.chemical_symbol_to_type["H"],
            idp.chemical_symbol_to_type["C"],
        ],
        dtype=torch.long,
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_types = torch.tensor(
        [
            idp.bond_to_type["H-C"],
            idp.bond_to_type["C-H"],
            idp.bond_to_type["H-C"],
            idp.bond_to_type["C-H"],
        ],
        dtype=torch.long,
    )
    data = {
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 1.0, 0.0]],
            dtype=dtype,
        ),
        "cell": torch.stack([torch.eye(3, dtype=dtype) * 5.0] * 2),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "edge_index": edge_index,
        "edge_cell_shift": torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=dtype
        ),
        "atom_types": atom_types,
        "edge_types": edge_types,
    }
    return idp, data


def _canonical_random_rme(idp, data, dtype):
    generator = torch.Generator().manual_seed(20260718)
    node = torch.randn(
        len(data["atom_types"]), idp.reduced_matrix_element, generator=generator, dtype=dtype
    )
    edge = torch.randn(
        data["edge_index"].shape[1], idp.reduced_matrix_element, generator=generator, dtype=dtype
    )
    node *= idp.mask_to_nrme[data["atom_types"]].to(dtype=dtype)
    edge *= idp.mask_to_erme[data["edge_types"]].to(dtype=dtype)
    # For H-C only the H->C row owns the global s-p canonical slice; reverse
    # rows are the transpose-completion side, hence carry no independent RME.
    edge[1::2] = 0
    return node, edge


def _random_state(idp, data, dtype=torch.float64, requires_grad=False):
    node_shapes = torch.tensor([[1, 1], [3, 3], [1, 1], [3, 3]], dtype=torch.long)
    edge_shapes = torch.tensor([[1, 3], [3, 1], [1, 3], [3, 1]], dtype=torch.long)
    generator = torch.Generator().manual_seed(718)
    node = torch.randn(4, 3, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
    edge = torch.randn(4, 3, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
    return BlockTensorResult(node, edge, node_shapes, edge_shapes)


def test_all_supported_cg_matrices_rank_orthogonality_and_condition_number():
    idp = OrbitalMapper(
        {"Si": ["3s", "3p", "3d"]}, method="e3tb", device="cpu"
    )
    module = E3Hamiltonian(idp=idp, dtype=torch.float64)
    assert module.cgbasis
    for pairtype, basis in module.cgbasis.items():
        matrix = basis.reshape(-1, basis.shape[-1])
        identity = torch.eye(matrix.shape[0], dtype=torch.float64)
        assert matrix.shape[0] == matrix.shape[1], pairtype
        assert torch.linalg.matrix_rank(matrix, atol=1e-12, rtol=1e-12) == matrix.shape[0]
        assert (matrix.T @ matrix - identity).abs().max().item() <= FP64_FLOOR
        assert (matrix @ matrix.T - identity).abs().max().item() <= FP64_FLOOR
        assert torch.linalg.cond(matrix).item() <= 1.0 + FP64_FLOOR


def test_inverse_cg_random_and_exhaustive_standard_basis_round_trip_fp64():
    idp = OrbitalMapper(
        {"Si": ["3s", "3p", "3d"]}, method="e3tb", device="cpu"
    )
    expand = E3Hamiltonian(idp=idp, dtype=torch.float64, decompose=False)
    contract = E3Hamiltonian(idp=idp, dtype=torch.float64, decompose=True)
    width = idp.reduced_matrix_element
    eye = torch.eye(width, dtype=torch.float64)
    random = torch.randn(width, width, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    for values in (eye, random):
        data = {
            "pos": torch.zeros(width, 3, dtype=torch.float64),
            "edge_index": torch.stack([torch.arange(width), torch.arange(width)]),
            "node_features": values.clone(),
            "edge_features": values.clone(),
        }
        restored = contract(expand(data))
        assert (restored["node_features"] - values).abs().max().item() <= _tol(values, torch.float64)
        assert (restored["edge_features"] - values).abs().max().item() <= _tol(values, torch.float64)


def test_inverse_cg_soc_is_explicitly_rejected():
    idp = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    inverse = E3Hamiltonian(idp=idp, dtype=torch.float64, decompose=True)
    inverse.soc = True
    data = {
        "pos": torch.zeros(1, 3, dtype=torch.float64),
        "edge_index": torch.tensor([[0], [0]]),
        "node_features": torch.zeros(1, idp.reduced_matrix_element, dtype=torch.float64),
        "edge_features": torch.zeros(1, idp.reduced_matrix_element, dtype=torch.float64),
    }
    with pytest.raises(NotImplementedError, match="non-SOC"):
        inverse(data)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_species_codec_round_trip_nonprefix_rectangular_and_padding(dtype):
    idp, data = _mixed_case(dtype)
    assert idp.full_basis == ["1s", "1p"]
    assert idp.basis_to_full_basis["C"]["2p"] == "1p"
    assert idp.orbital_maps["C"]["2p"].start == 0
    node, edge = _canonical_random_rme(idp, data, dtype)
    codec = BlockStateCodec(idp, dtype=dtype)
    blocks = codec.rme_to_blocks(data, node, edge)
    node2, edge2 = codec.blocks_to_rme(data, blocks)
    assert (node2 - node).abs().max().item() <= _tol(node, dtype)
    assert (edge2 - edge).abs().max().item() <= _tol(edge, dtype)
    assert blocks.edge_shapes.tolist() == [[1, 3], [3, 1], [1, 3], [3, 1]]
    node_mask = block_mask_from_shapes(blocks.node_shapes, tuple(blocks.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(blocks.edge_shapes, tuple(blocks.edge_blocks.shape[-2:]))
    assert torch.count_nonzero(blocks.node_blocks[~node_mask]) == 0
    assert torch.count_nonzero(blocks.edge_blocks[~edge_mask]) == 0


def test_independently_projected_physical_blocks_round_trip_both_directions():
    idp, data = _mixed_case()
    codec = BlockStateCodec(idp, dtype=torch.float64)
    allowed = project_block_state(data, idp, _random_state(idp, data))
    node_rme, edge_rme = codec.blocks_to_rme(data, allowed)
    rebuilt = codec.rme_to_blocks(data, node_rme, edge_rme)
    assert (rebuilt.node_blocks - allowed.node_blocks).abs().max().item() <= FP64_FLOOR
    assert (rebuilt.edge_blocks - allowed.edge_blocks).abs().max().item() <= FP64_FLOOR


def test_strict_rejects_off_image_and_project_reports_residual():
    idp = OrbitalMapper(
        {"H": ["1s"], "C": ["2s", "2p"]}, method="e3tb", device="cpu"
    )
    idp.get_irreps(no_parity=False)
    data = {
        "atom_types": torch.tensor(
            [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]]
        ),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_cell_shift": torch.zeros(2, 3),
    }
    node_shapes = torch.tensor([[1, 1], [4, 4]])
    edge_shapes = torch.tensor([[1, 4], [4, 1]])
    node = torch.zeros(2, 4, 4, dtype=torch.float64)
    edge = torch.zeros(2, 4, 4, dtype=torch.float64)
    node[1, 1, 0] = 1e-3  # non-canonical transpose-copy side
    node[0, 3, 3] = 1e-3  # padding
    with pytest.raises(ValueError, match="outside the canonical packer image"):
        canonical_block_tensors_to_feature_tensors(
            data, idp, node_blocks=node, edge_blocks=edge,
            node_shapes=node_shapes, edge_shapes=edge_shapes, mode="strict", atol=FP64_FLOOR,
        )
    result = canonical_block_tensors_to_feature_tensors(
        data, idp, node_blocks=node, edge_blocks=edge,
        node_shapes=node_shapes, edge_shapes=edge_shapes, mode="project", atol=FP64_FLOOR,
    )
    assert result.node_projection_residual.max().item() >= 2.5e-4
    assert result.projected_node_blocks[0, 3, 3].item() == 0.0


def test_pbc_reverse_metadata_missing_duplicate_and_batched_isolation():
    idp, data = _mixed_case()
    del idp
    assert strict_reverse_edge_index(data).tolist() == [1, 0, 3, 2]

    missing = dict(data)
    missing["edge_index"] = data["edge_index"][:, :3]
    missing["edge_cell_shift"] = data["edge_cell_shift"][:3]
    with pytest.raises(ValueError, match="missing reverse"):
        strict_reverse_edge_index(missing)

    duplicate = dict(data)
    duplicate["edge_index"] = torch.cat([data["edge_index"], data["edge_index"][:, :1]], dim=1)
    duplicate["edge_cell_shift"] = torch.cat(
        [data["edge_cell_shift"], data["edge_cell_shift"][:1]], dim=0
    )
    with pytest.raises(ValueError, match="Duplicate directed edge key"):
        strict_reverse_edge_index(duplicate)


def test_projector_is_idempotent_and_enforces_all_invariants():
    idp, data = _mixed_case()
    once = project_block_state(data, idp, _random_state(idp, data))
    twice = project_block_state(data, idp, once)
    assert (twice.node_blocks - once.node_blocks).abs().max().item() <= FP64_FLOOR
    assert (twice.edge_blocks - once.edge_blocks).abs().max().item() <= FP64_FLOOR
    assert (once.node_blocks - once.node_blocks.transpose(-1, -2)).abs().max().item() <= FP64_FLOOR
    rev = strict_reverse_edge_index(data)
    assert (
        once.edge_blocks - once.edge_blocks.index_select(0, rev).transpose(-1, -2)
    ).abs().max().item() <= FP64_FLOOR
    node_mask = block_mask_from_shapes(once.node_shapes, tuple(once.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(once.edge_shapes, tuple(once.edge_blocks.shape[-2:]))
    assert torch.count_nonzero(once.node_blocks[~node_mask]) == 0
    assert torch.count_nonzero(once.edge_blocks[~edge_mask]) == 0


def test_projector_gradcheck():
    idp, data = _mixed_case()
    state = _random_state(idp, data, requires_grad=True)

    def fn(node, edge):
        projected = project_block_state(
            data,
            idp,
            BlockTensorResult(node, edge, state.node_shapes, state.edge_shapes),
        )
        return projected.node_blocks, projected.edge_blocks

    assert torch.autograd.gradcheck(fn, (state.node_blocks, state.edge_blocks), eps=1e-6, atol=1e-5)


def test_endpoint_semantics_absolute_and_residual_add_h0_exactly_once():
    idp, data = _mixed_case()
    state = project_block_state(data, idp, _random_state(idp, data))
    h0 = BlockTensorResult(
        torch.full_like(state.node_blocks, 3.0),
        torch.full_like(state.edge_blocks, 3.0),
        state.node_shapes,
        state.edge_shapes,
    )
    endpoint = BlockTensorResult(
        torch.full_like(state.node_blocks, 2.0),
        torch.full_like(state.edge_blocks, 2.0),
        state.node_shapes,
        state.edge_shapes,
    )
    absolute = BlockStateCodec(
        idp, dtype=torch.float64, target_semantics="absolute_full_h"
    ).endpoint_to_full(endpoint, h0)
    residual = BlockStateCodec(
        idp, dtype=torch.float64, target_semantics="residual_dh"
    ).endpoint_to_full(endpoint, h0)
    assert torch.equal(absolute.node_blocks, endpoint.node_blocks)
    assert torch.equal(absolute.edge_blocks, endpoint.edge_blocks)
    assert torch.all(residual.node_blocks == 5.0)
    assert torch.all(residual.edge_blocks == 5.0)
