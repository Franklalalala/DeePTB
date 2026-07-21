from __future__ import annotations

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import (
    canonical_block_tensors_to_feature_tensors,
    feature_tensors_to_block_tensors,
    infer_block_shapes,
    strict_reverse_edge_index,
)
from dptb.data.transforms import OrbitalMapper


def _mapper() -> OrbitalMapper:
    mapper = OrbitalMapper({"H": ["1s"], "C": ["2s"]}, method="e3tb", device="cpu")
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    return mapper


def _two_atom_graph(*, periodic: bool = False) -> dict:
    return {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1, 6], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0, 0], dtype=torch.long),
        _keys.CELL_KEY: torch.eye(3).unsqueeze(0),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[1, 0, 0], [-1, 0, 0]] if periodic else [[0, 0, 0], [0, 0, 0]],
            dtype=torch.long,
        ),
        _keys.PBC_KEY: torch.tensor([periodic, False, False]),
    }


@pytest.mark.parametrize(
    ("edge_index", "match"),
    [
        (torch.tensor([[0.5, 1.0], [1.0, 0.5]]), "non-integer"),
        (torch.tensor([[float("nan"), 1.0], [1.0, 0.0]]), "finite integer"),
        (torch.tensor([[-1, 0], [0, -1]]), "negative"),
        (torch.tensor([[0, 2], [2, 0]]), "outside"),
    ],
)
def test_strict_graph_validates_edge_index_before_cast(edge_index, match):
    data = _two_atom_graph()
    data[_keys.EDGE_INDEX_KEY] = edge_index
    with pytest.raises(ValueError, match=match):
        strict_reverse_edge_index(data)


def test_strict_graph_rejects_missing_periodic_shift_and_nonperiodic_shift():
    data = _two_atom_graph(periodic=True)
    data.pop(_keys.EDGE_CELL_SHIFT_KEY)
    with pytest.raises(ValueError, match="require explicit edge_cell_shift"):
        strict_reverse_edge_index(data)

    data = _two_atom_graph(periodic=True)
    data[_keys.EDGE_CELL_SHIFT_KEY] = torch.tensor(
        [[0, 1, 0], [0, -1, 0]], dtype=torch.long
    )
    with pytest.raises(ValueError, match="non-periodic axes"):
        strict_reverse_edge_index(data)


def test_strict_graph_requires_cell_and_rejects_shift_int64_overflow():
    data = _two_atom_graph(periodic=True)
    data.pop(_keys.CELL_KEY)
    with pytest.raises(ValueError, match="requires an explicit cell"):
        strict_reverse_edge_index(data)

    data = _two_atom_graph()
    data[_keys.EDGE_CELL_SHIFT_KEY] = torch.tensor(
        [[2**64 - 1, 0, 0], [0, 0, 0]], dtype=torch.uint64
    )
    with pytest.raises(ValueError, match="outside the int64 range"):
        strict_reverse_edge_index(data)

    data = _two_atom_graph(periodic=True)
    data.pop(_keys.PBC_KEY)
    with pytest.raises(ValueError, match="requires explicit pbc axes"):
        strict_reverse_edge_index(data)


def test_strict_graph_rejects_cross_batch_edges():
    data = _two_atom_graph()
    data[_keys.BATCH_KEY] = torch.tensor([0, 1], dtype=torch.long)
    data[_keys.PBC_KEY] = torch.zeros((2, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="same batch graph"):
        strict_reverse_edge_index(data)


@pytest.mark.parametrize("empty_edges", [True, False])
def test_strict_graph_rejects_zero_row_pbc_with_contract_error(empty_edges):
    data = _two_atom_graph()
    data[_keys.PBC_KEY] = torch.empty((0, 3), dtype=torch.bool)
    if empty_edges:
        data[_keys.EDGE_INDEX_KEY] = torch.empty((2, 0), dtype=torch.long)
        data[_keys.EDGE_CELL_SHIFT_KEY] = torch.empty((0, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="pbc has 0 graph rows"):
        strict_reverse_edge_index(data)


def test_strict_graph_validates_explicit_edge_type_against_mapper():
    mapper = _mapper()
    data = _two_atom_graph()
    data[_keys.EDGE_TYPE_KEY] = torch.tensor(
        [mapper.bond_to_type["H-C"], mapper.bond_to_type["C-H"]],
        dtype=torch.long,
    )
    assert strict_reverse_edge_index(data, idp=mapper).tolist() == [1, 0]

    data[_keys.EDGE_TYPE_KEY] = data[_keys.EDGE_TYPE_KEY].flip(0)
    with pytest.raises(ValueError, match="mapper/species"):
        strict_reverse_edge_index(data, idp=mapper)

    data[_keys.EDGE_TYPE_KEY] = torch.tensor([0.5, 1.0])
    with pytest.raises(ValueError, match="non-integer"):
        strict_reverse_edge_index(data, idp=mapper)


def test_empty_edge_graph_keeps_rank_two_shapes_through_strict_codec():
    mapper = _mapper()
    width = int(mapper.reduced_matrix_element)
    data = {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.empty((0, 3), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.empty((0,), dtype=torch.long),
        _keys.PBC_KEY: torch.tensor([False, False, False]),
    }
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    assert node_shapes.shape == (1, 2)
    assert edge_shapes.shape == (0, 2)
    assert strict_reverse_edge_index(data, idp=mapper).shape == (0,)

    packed = feature_tensors_to_block_tensors(
        data,
        mapper,
        node_features=torch.zeros((1, width), dtype=torch.float64),
        edge_features=torch.zeros((0, width), dtype=torch.float64),
        strict_complete_edges=True,
    )
    assert packed.edge_blocks.shape[0] == 0
    assert packed.edge_shapes.shape == (0, 2)

    inverse = canonical_block_tensors_to_feature_tensors(
        data,
        mapper,
        node_blocks=packed.node_blocks,
        edge_blocks=packed.edge_blocks,
        node_shapes=packed.node_shapes,
        edge_shapes=packed.edge_shapes,
        mode="strict",
        atol=1e-10,
    )
    assert inverse.edge_features.shape == (0, width)
    assert inverse.edge_projection_residual.shape == (0,)


def test_with_edge_vectors_accepts_integer_cell_shifts_with_float_cell():
    data = {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.5, 0.0]], dtype=torch.float64
        ),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float64) * 2.0,
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[1, 0, 0], [-1, 0, 0]], dtype=torch.long
        ),
    }

    out = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

    expected = torch.tensor([[2.25, 0.5, 0.0], [-2.25, -0.5, 0.0]], dtype=torch.float64)
    assert out[_keys.EDGE_VECTORS_KEY].dtype == torch.float64
    assert out[_keys.EDGE_LENGTH_KEY].dtype == torch.float64
    torch.testing.assert_close(out[_keys.EDGE_VECTORS_KEY], expected)
