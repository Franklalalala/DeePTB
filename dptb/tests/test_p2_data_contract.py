from __future__ import annotations

import numpy as np
import pytest
import torch

from dptb.data import _keys
from dptb.data.dataset.lmdb_dataset import assert_absolute_full_h_target_contract
from dptb.data.interfaces.blockwise_tensor import (
    block_dict_to_ordered_tensors,
    validate_packed_non_soc_blocks,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    EDGE_GRAPH_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    assert_record_fingerprint,
    canonical_edge_graph,
    edge_graph_fingerprint,
    fingerprint_fields,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
    mapper_basis_fingerprint,
)
from dptb.data.transforms_upper_triangle import OrbitalMapper


def _mixed_species_case():
    idp = OrbitalMapper(
        {"H": ["1s"], "O": ["1s", "2p"]}, method="e3tb", device="cpu"
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.long
    )
    shifts = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0]],
        dtype=torch.float32,
    )
    data = {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1, 8]),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.EDGE_CELL_SHIFT_KEY: shifts,
    }
    node = torch.zeros(2, 4, 4)
    node[0, 0, 0] = 1.0
    node[1, :4, :4] = torch.eye(4)
    edge = torch.zeros(4, 4, 4)
    edge[0, 0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    edge[2, :4, 0] = edge[0, 0, :4]
    edge[1, 0, :4] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    edge[3, :4, 0] = edge[1, 0, :4]
    node_shapes = torch.tensor([[1, 1], [4, 4]])
    edge_shapes = torch.tensor([[1, 4], [1, 4], [4, 1], [4, 1]])
    return idp, data, node, edge, node_shapes, edge_shapes


def test_periodic_edge_graph_is_integer_unique_and_order_fingerprinted():
    idp, data, *_ = _mixed_species_case()
    edge, shifts = canonical_edge_graph(
        data[_keys.ATOMIC_NUMBERS_KEY],
        data[_keys.EDGE_INDEX_KEY],
        data[_keys.EDGE_CELL_SHIFT_KEY],
    )
    assert edge.shape == (2, 4)
    assert shifts.dtype == np.int64
    basis = mapper_basis_fingerprint(idp)
    original = edge_graph_fingerprint(
        data[_keys.ATOMIC_NUMBERS_KEY], edge, shifts, basis_fingerprint=basis
    )
    order = np.asarray([1, 0, 2, 3])
    shuffled = edge_graph_fingerprint(
        data[_keys.ATOMIC_NUMBERS_KEY],
        edge[:, order],
        shifts[order],
        basis_fingerprint=basis,
    )
    assert shuffled != original

    bad_shift = data[_keys.EDGE_CELL_SHIFT_KEY].clone()
    bad_shift[1, 0] = 0.999
    with pytest.raises(ValueError, match="integer-valued"):
        canonical_edge_graph(
            data[_keys.ATOMIC_NUMBERS_KEY], data[_keys.EDGE_INDEX_KEY], bad_shift
        )

    duplicate_edge = torch.cat(
        [data[_keys.EDGE_INDEX_KEY], data[_keys.EDGE_INDEX_KEY][:, :1]], dim=1
    )
    duplicate_shift = torch.cat(
        [data[_keys.EDGE_CELL_SHIFT_KEY], data[_keys.EDGE_CELL_SHIFT_KEY][:1]], dim=0
    )
    with pytest.raises(ValueError, match="duplicate"):
        canonical_edge_graph(
            data[_keys.ATOMIC_NUMBERS_KEY], duplicate_edge, duplicate_shift
        )


def test_mixed_species_packed_blocks_fail_on_shape_padding_and_hermiticity():
    idp, data, node, edge, node_shapes, edge_shapes = _mixed_species_case()
    validate_packed_non_soc_blocks(
        data,
        idp,
        node,
        edge,
        node_shapes,
        edge_shapes,
        label="P2",
    )

    wrong_shapes = edge_shapes.clone()
    wrong_shapes[[0, 2]] = wrong_shapes[[2, 0]]
    with pytest.raises(ValueError, match="bond species"):
        validate_packed_non_soc_blocks(
            data, idp, node, edge, node_shapes, wrong_shapes, label="P2"
        )

    bad_padding = edge.clone()
    bad_padding[0, 3, 3] = 1.0
    with pytest.raises(ValueError, match="non-zero padding"):
        validate_packed_non_soc_blocks(
            data, idp, node, bad_padding, node_shapes, edge_shapes, label="P2"
        )

    bad_onsite = node.clone()
    bad_onsite[1, 0, 1] = 0.5
    with pytest.raises(ValueError, match="not symmetric"):
        validate_packed_non_soc_blocks(
            data, idp, bad_onsite, edge, node_shapes, edge_shapes, label="P2"
        )

    bad_reverse = edge.clone()
    bad_reverse[2, 0, 0] += 0.5
    with pytest.raises(ValueError, match="Hermiticity mismatch"):
        validate_packed_non_soc_blocks(
            data, idp, node, bad_reverse, node_shapes, edge_shapes, label="P2"
        )


def test_raw_block_packing_rejects_species_shape_truncation():
    idp, data, *_ = _mixed_species_case()
    blocks = {
        "0_0_0_0_0": np.zeros((4, 4), dtype=np.float32),  # H must be 1x1
        "1_1_0_0_0": np.zeros((4, 4), dtype=np.float32),
        "0_1_0_0_0": np.zeros((1, 4), dtype=np.float32),
        "1_0_0_0_0": np.zeros((4, 1), dtype=np.float32),
        "0_1_1_0_0": np.zeros((1, 4), dtype=np.float32),
        "1_0_-1_0_0": np.zeros((4, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="refusing silent truncation"):
        block_dict_to_ordered_tensors(
            data, idp, blocks, complete_edges=False, start_id=0
        )


def test_full_h_semantics_never_inferred_from_delta_named_fields():
    valid = {
        SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "raw_hamiltonian",
        "hamiltonian": {"0_0_0_0_0": np.ones((1, 1), dtype=np.float32)},
    }
    assert_absolute_full_h_target_contract(valid)
    missing_marker = dict(valid)
    del missing_marker[TARGET_SEMANTICS_KEY]
    with pytest.raises(ValueError, match="requires hamiltonian_target_semantics"):
        assert_absolute_full_h_target_contract(missing_marker)

    dedicated = {
        SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "dedicated_full_h_blocks",
        _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros((1, 1, 1)),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros((0, 1, 1)),
        _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((1, 2)),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.empty((0, 2)),
        _keys.NODE_DELTA_HAMIL_BLOCKS_KEY: np.zeros((1, 1, 1)),
    }
    with pytest.raises(ValueError, match="delta-named"):
        assert_absolute_full_h_target_contract(dedicated)


def test_fingerprint_fields_detects_same_count_row_permutation():
    record = {
        "node": np.asarray([[1.0], [2.0]], dtype=np.float32),
        "edge": np.asarray([[3.0], [4.0]], dtype=np.float32),
    }
    original = fingerprint_fields(record, ("node", "edge"))
    record["edge"] = record["edge"][::-1].copy()
    assert fingerprint_fields(record, ("node", "edge")) != original


def test_row_aligned_bundle_binds_h0_and_targets_to_graph_order():
    idp, data, *_ = _mixed_species_case()
    basis = mapper_basis_fingerprint(idp)
    graph = edge_graph_fingerprint(
        data[_keys.ATOMIC_NUMBERS_KEY],
        data[_keys.EDGE_INDEX_KEY],
        data[_keys.EDGE_CELL_SHIFT_KEY],
        basis_fingerprint=basis,
    )
    record = {
        _keys.NODE_H0_KEY: np.zeros((2, 3), dtype=np.float32),
        _keys.EDGE_H0_KEY: np.arange(12, dtype=np.float32).reshape(4, 3),
        _keys.NODE_DELTA_HAMIL_BLOCKS_KEY: np.zeros((2, 4, 4), dtype=np.float32),
        _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY: np.arange(
            64, dtype=np.float32
        ).reshape(4, 4, 4),
        BASIS_FINGERPRINT_KEY: basis,
        EDGE_GRAPH_FINGERPRINT_KEY: graph,
    }
    row_hash = fingerprint_present_row_aligned_fields(record)
    record[ROW_ALIGNED_DATA_FINGERPRINT_KEY] = row_hash
    record[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ),
    )

    order = np.asarray([1, 0, 2, 3])
    record[_keys.EDGE_H0_KEY] = record[_keys.EDGE_H0_KEY][order].copy()
    with pytest.raises(ValueError, match="row_aligned_data_fingerprint mismatch"):
        assert_record_fingerprint(
            record,
            field=ROW_ALIGNED_DATA_FINGERPRINT_KEY,
            actual=fingerprint_present_row_aligned_fields(record),
        )
