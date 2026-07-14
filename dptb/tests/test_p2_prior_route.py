from __future__ import annotations

from copy import deepcopy
import hashlib
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
import yaml

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset import lmdb_dataset as lmdb_dataset_module
from dptb.data.dataset.lmdb_dataset import (
    LMDBDataset,
    validate_non_soc_p2_blocks,
    validate_non_soc_p2_block_tensors,
    validate_p2_feature_pair,
)
from dptb.data.interfaces import blockwise_tensor as blockwise_tensor_module
from dptb.data.interfaces.blockwise_tensor import (
    EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    EDGE_P2_BLOCKS_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    NODE_P2_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    feature_tensors_to_block_tensors,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    P2_BLOCK_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_RME_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P2_SOURCE_FINGERPRINT_KEY,
    P23_BLOCK_FINGERPRINT_KEY,
    P23_BUNDLE_FINGERPRINT_KEY,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    P23_RME_FINGERPRINT_KEY,
    P23_SOURCE_FINGERPRINT_KEY,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    edge_graph_fingerprint,
    fingerprint_fields,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
    mapper_basis_fingerprint,
)
from dptb.data.transforms import OrbitalMapper as ProductionOrbitalMapper
from dptb.nn.blockwise_hamiltonian import (
    attach_full_hamiltonian_from_h0,
    attach_full_hamiltonian_from_prior,
)
from dptb.nn.build import build_model
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.utils.argcheck import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strict_p2_full_h_config_and_semantic_guards():
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "p2_prior_non_soc_full_h_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize(deepcopy(payload))
    assert normalized["data_options"]["train"]["residual_hamiltonian"] is False
    assert normalized["data_options"]["train"]["require_full_h_target"] is True
    assert normalized["data_options"]["train"]["require_p2_blocks"] is True
    assert (
        normalized["data_options"]["train"][
            "allow_unbound_prior_source_fingerprint"
        ]
        is True
    )
    assert normalized["model_options"]["embedding"]["method"] == "lem_moe_v3_prior"
    assert normalized["train_options"]["loss_options"]["train"][
        "pred_node_block_key"
    ] == "node_full_hamil_blocks"

    residual = deepcopy(payload)
    residual["data_options"]["train"]["residual_hamiltonian"] = True
    with pytest.raises(ValueError, match="absolute Full H"):
        normalize(residual)

    correction_loss = deepcopy(payload)
    correction_loss["train_options"]["loss_options"]["train"][
        "pred_node_block_key"
    ] = "node_hamil_blocks"
    with pytest.raises(ValueError, match="reconstructed Full-H"):
        normalize(correction_loss)

    p2_as_target = deepcopy(payload)
    p2_as_target["train_options"]["loss_options"]["train"].update(
        {
            "target_node_block_key": "node_p2_blocks",
            "target_edge_block_key": "edge_p2_blocks",
            "target_node_shape_key": "node_p2_block_shape",
            "target_edge_shape_key": "edge_p2_block_shape",
        }
    )
    with pytest.raises(ValueError, match="explicit absolute Full-H"):
        normalize(p2_as_target)


def _p23_config(*, direct: bool = False):
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "p2_prior_non_soc_full_h_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    embedding = payload["model_options"]["embedding"]
    embedding.update(
        {
            "prior_kind": "p23",
            "prior_node_key": "node_p23",
            "prior_edge_key": "edge_p23",
        }
    )
    prediction = payload["model_options"]["prediction"]
    prediction.update(
        {
            "prior_node_block_field": "node_p23_blocks",
            "prior_edge_block_field": "edge_p23_blocks",
            "prior_label": "P23",
        }
    )
    for split_options in payload["data_options"].values():
        if isinstance(split_options, dict) and "get_P2" in split_options:
            split_options["prior_kind"] = "p23"
            split_options["p2_key"] = "hamiltonian_p23"
            split_options["require_p2_blocks"] = not direct
    if direct:
        prediction["add_prior"] = False
        for split_loss in payload["train_options"]["loss_options"].values():
            split_loss["pred_node_block_key"] = "node_hamil_blocks"
            split_loss["pred_edge_block_key"] = "edge_hamil_blocks"
    return payload


def test_p23_residual_and_direct_configs_fail_closed_on_cross_wiring():
    residual = normalize(_p23_config(direct=False))
    assert residual["model_options"]["prediction"]["add_prior"] is True
    assert residual["data_options"]["train"]["prior_kind"] == "p23"

    direct = normalize(_p23_config(direct=True))
    assert direct["model_options"]["prediction"]["add_prior"] is False
    assert direct["data_options"]["train"]["require_p2_blocks"] is False
    assert direct["train_options"]["loss_options"]["train"][
        "pred_node_block_key"
    ] == "node_hamil_blocks"

    wrong_rme = _p23_config(direct=False)
    wrong_rme["model_options"]["embedding"]["prior_edge_key"] = "edge_p2"
    with pytest.raises(ValueError, match="mix P2 and P23"):
        normalize(wrong_rme)

    wrong_raw = _p23_config(direct=False)
    wrong_raw["data_options"]["train"]["p2_key"] = "hamiltonian_p2"
    with pytest.raises(ValueError, match="requires p2_key='hamiltonian_p23'"):
        normalize(wrong_raw)

    wrong_blocks = _p23_config(direct=False)
    wrong_blocks["model_options"]["prediction"][
        "prior_node_block_field"
    ] = "node_p2_blocks"
    with pytest.raises(ValueError, match="requires prior_node_block_field"):
        normalize(wrong_blocks)

    direct_with_blocks = _p23_config(direct=True)
    direct_with_blocks["data_options"]["train"]["require_p2_blocks"] = True
    with pytest.raises(ValueError, match="direct Full-H head"):
        normalize(direct_with_blocks)

    direct_delta_target = _p23_config(direct=True)
    direct_delta_target["train_options"]["loss_options"]["train"][
        "target_node_block_key"
    ] = "node_delta_hamil_blocks"
    with pytest.raises(ValueError, match="explicit absolute Full-H"):
        normalize(direct_delta_target)


def test_p2_feature_and_block_contract_fails_closed():
    node = torch.zeros(2, 4)
    edge = torch.zeros(3, 4)
    got_node, got_edge = validate_p2_feature_pair(
        node,
        edge,
        num_nodes=2,
        num_edges=3,
        feature_dim=4,
    )
    assert got_node is node and got_edge is edge

    with pytest.raises(ValueError, match="both node_p2 and edge_p2"):
        validate_p2_feature_pair(node, None)
    with pytest.raises(TypeError, match="floating-point"):
        validate_p2_feature_pair(node.to(torch.long), edge.to(torch.long))
    with pytest.raises(ValueError, match="mapper RME width"):
        validate_p2_feature_pair(node, edge, feature_dim=5)
    with pytest.raises(NotImplementedError, match="non-SOC"):
        validate_p2_feature_pair(node.to(torch.complex64), edge.to(torch.complex64))
    with pytest.raises(ValueError, match="NaN"):
        validate_p2_feature_pair(node.fill_(torch.nan), edge)

    validate_non_soc_p2_blocks({"0_0_0_0_0": torch.eye(2).numpy()})
    with pytest.raises(NotImplementedError, match="non-SOC"):
        validate_non_soc_p2_blocks(
            {"0_0_0_0_0": torch.eye(2, dtype=torch.complex64).numpy()}
        )
    with pytest.raises(ValueError, match="invalid AO-block key"):
        validate_non_soc_p2_blocks({"bad-key": torch.eye(2).numpy()})

    validate_non_soc_p2_block_tensors(
        torch.zeros(2, 3, 3),
        torch.zeros(3, 3, 3),
        torch.tensor([[1, 1], [3, 3]]),
        torch.tensor([[1, 1], [2, 2], [3, 3]]),
        num_nodes=2,
        num_edges=3,
    )
    with pytest.raises(ValueError, match="exceeds packed canvas"):
        validate_non_soc_p2_block_tensors(
            torch.zeros(2, 3, 3),
            torch.zeros(3, 3, 3),
            torch.tensor([[1, 1], [4, 3]]),
            torch.tensor([[1, 1], [2, 2], [3, 3]]),
            num_nodes=2,
            num_edges=3,
        )


def test_full_h_reconstruction_uses_explicit_output_and_preserves_h0_soc():
    pred_node = torch.randn(2, 3, 3, requires_grad=True)
    pred_edge = torch.randn(4, 3, 3, requires_grad=True)
    p2_node = torch.randn_like(pred_node)
    p2_edge = torch.randn_like(pred_edge)
    data = {
        NODE_PRED_HAMIL_BLOCKS_KEY: pred_node,
        EDGE_PRED_HAMIL_BLOCKS_KEY: pred_edge,
        NODE_P2_BLOCKS_KEY: p2_node,
        EDGE_P2_BLOCKS_KEY: p2_edge,
    }
    attach_full_hamiltonian_from_prior(
        data,
        prior_node_field=NODE_P2_BLOCKS_KEY,
        prior_edge_field=EDGE_P2_BLOCKS_KEY,
        prior_label="P2",
        non_soc_only=True,
    )
    torch.testing.assert_close(data["node_full_hamil_blocks"], p2_node + pred_node)
    torch.testing.assert_close(data["edge_full_hamil_blocks"], p2_edge + pred_edge)
    assert data[NODE_PRED_HAMIL_BLOCKS_KEY] is pred_node
    assert data[EDGE_PRED_HAMIL_BLOCKS_KEY] is pred_edge
    (data["node_full_hamil_blocks"].sum() + data["edge_full_hamil_blocks"].sum()).backward()
    assert torch.equal(pred_node.grad, torch.ones_like(pred_node))
    assert torch.equal(pred_edge.grad, torch.ones_like(pred_edge))

    complex_data = dict(data)
    complex_data[NODE_P2_BLOCKS_KEY] = p2_node.to(torch.complex64)
    complex_data[EDGE_P2_BLOCKS_KEY] = p2_edge.to(torch.complex64)
    with pytest.raises(TypeError, match="non-SOC"):
        attach_full_hamiltonian_from_prior(
            complex_data,
            prior_node_field=NODE_P2_BLOCKS_KEY,
            prior_edge_field=EDGE_P2_BLOCKS_KEY,
            non_soc_only=True,
        )

    # The generic refactor must not narrow the historical H0/SOC route.
    h0_data = {
        NODE_PRED_HAMIL_BLOCKS_KEY: pred_node.detach().to(torch.complex64),
        EDGE_PRED_HAMIL_BLOCKS_KEY: pred_edge.detach().to(torch.complex64),
        "node_h0_blocks": p2_node.to(torch.complex64) * (1 + 1j),
        "edge_h0_blocks": p2_edge.to(torch.complex64) * (1 + 1j),
    }
    attach_full_hamiltonian_from_h0(h0_data)
    assert torch.is_complex(h0_data["node_full_hamil_blocks"])


def test_lmdb_raw_full_h_plus_p2_contract_end_to_end(tmp_path):
    lmdb_name = "p2-smoke.lmdb"
    lmdb_path = tmp_path / lmdb_name
    h_blocks = {
        "0_0_0_0_0": np.asarray([[1.2]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.4]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.3]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.3]], dtype=np.float32),
    }
    p2_blocks = {
        "0_0_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.1]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.2]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.2]], dtype=np.float32),
    }
    record = {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_p2": p2_blocks,
        SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "raw_hamiltonian",
    }
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    env.close()

    dataset = DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix="p2-smoke",
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        require_p2_blocks=True,
        require_full_h_target=True,
        audit_p2_representations=True,
        residual_hamiltonian=False,
    )
    assert isinstance(dataset, LMDBDataset)
    sample = dataset.get(0)

    torch.testing.assert_close(sample[_keys.NODE_P2_KEY], torch.tensor([[1.0], [1.1]]))
    torch.testing.assert_close(sample[_keys.EDGE_P2_KEY], torch.tensor([[0.2], [0.2]]))
    assert sample[NODE_P2_BLOCKS_KEY].shape == (2, 1, 1)
    assert sample[EDGE_P2_BLOCKS_KEY].shape == (2, 1, 1)
    torch.testing.assert_close(
        sample[NODE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(), torch.tensor([1.2, 1.4])
    )
    torch.testing.assert_close(
        sample[EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(), torch.tensor([0.3, 0.3])
    )

    raw_dataset = DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix="p2-smoke",
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        prefer_precomputed_p2=False,
        require_p2_blocks=True,
        require_full_h_target=True,
        audit_p2_representations=True,
        residual_hamiltonian=False,
    )
    raw_sample = raw_dataset.get(0)
    torch.testing.assert_close(
        raw_sample[_keys.NODE_P2_KEY], torch.tensor([[1.0], [1.1]])
    )
    torch.testing.assert_close(
        raw_sample[_keys.EDGE_P2_KEY], torch.tensor([[0.2], [0.2]])
    )


def _compact_p2_record():
    mapper = ProductionOrbitalMapper({"H": "1s"}, method="e3tb", device="cpu")
    basis_fingerprint = mapper_basis_fingerprint(mapper)
    p2_source = hashlib.sha256(b"unit-test-p2-source").hexdigest()
    record = {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        # Deliberately reversed relative to a typical graph builder order.
        _keys.EDGE_INDEX_KEY: np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        _keys.NODE_FEATURES_KEY: np.asarray([[1.2], [1.4]], dtype=np.float32),
        _keys.EDGE_FEATURES_KEY: np.asarray([[0.3], [0.3]], dtype=np.float32),
        _keys.NODE_P2_KEY: np.asarray([[1.0], [1.1]], dtype=np.float32),
        _keys.EDGE_P2_KEY: np.asarray([[0.2], [0.2]], dtype=np.float32),
        _keys.NODE_P2_BLOCKS_KEY: np.asarray([[[1.0]], [[1.1]]], dtype=np.float32),
        _keys.EDGE_P2_BLOCKS_KEY: np.asarray([[[0.2]], [[0.2]]], dtype=np.float32),
        _keys.NODE_P2_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.EDGE_P2_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.asarray(
            [[[1.2]], [[1.4]]], dtype=np.float32
        ),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.asarray(
            [[[0.3]], [[0.3]]], dtype=np.float32
        ),
        _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "dedicated_full_h_blocks",
        BASIS_FINGERPRINT_KEY: basis_fingerprint,
        P2_SOURCE_FINGERPRINT_KEY: p2_source,
    }
    record[EDGE_GRAPH_FINGERPRINT_KEY] = edge_graph_fingerprint(
        record[_keys.ATOMIC_NUMBERS_KEY],
        record[_keys.EDGE_INDEX_KEY],
        record[_keys.EDGE_CELL_SHIFT_KEY],
        basis_fingerprint=basis_fingerprint,
    )
    return _refresh_compact_fingerprints(record), p2_source


def _refresh_compact_fingerprints(record):
    record[P2_RME_FINGERPRINT_KEY] = fingerprint_fields(
        record, (_keys.NODE_P2_KEY, _keys.EDGE_P2_KEY)
    )
    record[P2_BLOCK_FINGERPRINT_KEY] = fingerprint_fields(
        record,
        (
            _keys.NODE_P2_BLOCKS_KEY,
            _keys.EDGE_P2_BLOCKS_KEY,
            _keys.NODE_P2_BLOCK_SHAPE_KEY,
            _keys.EDGE_P2_BLOCK_SHAPE_KEY,
        ),
    )
    record[P2_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            P2_SOURCE_FINGERPRINT_KEY,
            P2_RME_FINGERPRINT_KEY,
            P2_BLOCK_FINGERPRINT_KEY,
        ),
    )
    record[FULL_H_TARGET_FINGERPRINT_KEY] = fingerprint_fields(
        record,
        (
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
        ),
    )
    record[ROW_ALIGNED_DATA_FINGERPRINT_KEY] = (
        fingerprint_present_row_aligned_fields(record)
    )
    record[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ),
    )
    return record


def _compact_dual_prior_record():
    record, p2_source = _compact_p2_record()
    p23_source = hashlib.sha256(b"unit-test-p23-source").hexdigest()
    record.update(
        {
            SAMPLE_SCHEMA_KEY: DUAL_PRIOR_SAMPLE_SCHEMA,
            _keys.NODE_P23_KEY: np.asarray([[2.0], [2.1]], dtype=np.float32),
            _keys.EDGE_P23_KEY: np.asarray([[0.4], [0.4]], dtype=np.float32),
            _keys.NODE_P23_BLOCKS_KEY: np.asarray(
                [[[2.0]], [[2.1]]], dtype=np.float32
            ),
            _keys.EDGE_P23_BLOCKS_KEY: np.asarray(
                [[[0.4]], [[0.4]]], dtype=np.float32
            ),
            _keys.NODE_P23_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
            _keys.EDGE_P23_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
            P23_SOURCE_FINGERPRINT_KEY: p23_source,
            P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY: record[
                P2_BUNDLE_FINGERPRINT_KEY
            ],
        }
    )
    return _refresh_dual_prior_fingerprints(record), p2_source, p23_source


def _refresh_dual_prior_fingerprints(record):
    _refresh_compact_fingerprints(record)
    record[P23_RME_FINGERPRINT_KEY] = fingerprint_fields(
        record, (_keys.NODE_P23_KEY, _keys.EDGE_P23_KEY)
    )
    record[P23_BLOCK_FINGERPRINT_KEY] = fingerprint_fields(
        record,
        (
            _keys.NODE_P23_BLOCKS_KEY,
            _keys.EDGE_P23_BLOCKS_KEY,
            _keys.NODE_P23_BLOCK_SHAPE_KEY,
            _keys.EDGE_P23_BLOCK_SHAPE_KEY,
        ),
    )
    record[P23_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            P23_SOURCE_FINGERPRINT_KEY,
            P23_RME_FINGERPRINT_KEY,
            P23_BLOCK_FINGERPRINT_KEY,
            P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
        ),
    )
    record[ROW_ALIGNED_DATA_FINGERPRINT_KEY] = (
        fingerprint_present_row_aligned_fields(record)
    )
    record[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ),
    )
    return record


def _write_single_lmdb(root, prefix, record):
    path = root / f"{prefix}.lmdb"
    env = lmdb.open(str(path), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    env.close()


def _compact_dataset(root, prefix, p2_source):
    return DatasetBuilder()(
        root=str(root),
        r_max=2.0,
        er_max=3.0,
        type="LMDBDataset",
        prefix=prefix,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        require_p2_blocks=True,
        require_full_h_target=True,
        audit_p2_representations=True,
        expected_p2_source_fingerprint=p2_source,
        residual_hamiltonian=False,
    )


def _compact_selected_prior_dataset(
    root,
    prefix,
    source_fingerprint,
    *,
    prior_kind,
    require_blocks,
    audit=False,
):
    return DatasetBuilder()(
        root=str(root),
        r_max=2.0,
        er_max=3.0,
        type="LMDBDataset",
        prefix=prefix,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        prior_kind=prior_kind,
        p2_key=f"hamiltonian_{prior_kind}",
        require_p2_blocks=require_blocks,
        require_full_h_target=True,
        audit_p2_representations=audit,
        expected_p2_source_fingerprint=source_fingerprint,
        residual_hamiltonian=False,
    )


def test_dual_prior_lmdb_selects_p23_and_direct_skips_ao_blocks(tmp_path):
    record, p2_source, p23_source = _compact_dual_prior_record()
    _write_single_lmdb(tmp_path, "dual-prior", record)

    p2_direct = _compact_selected_prior_dataset(
        tmp_path,
        "dual-prior",
        p2_source,
        prior_kind="p2",
        require_blocks=False,
    ).get(0)
    torch.testing.assert_close(
        p2_direct[_keys.NODE_P2_KEY], torch.tensor([[1.0], [1.1]])
    )
    assert _keys.NODE_P23_KEY not in p2_direct
    assert _keys.NODE_P2_BLOCKS_KEY not in p2_direct

    direct = _compact_selected_prior_dataset(
        tmp_path,
        "dual-prior",
        p23_source,
        prior_kind="p23",
        require_blocks=False,
    ).get(0)
    torch.testing.assert_close(
        direct[_keys.NODE_P23_KEY], torch.tensor([[2.0], [2.1]])
    )
    torch.testing.assert_close(
        direct[_keys.EDGE_P23_KEY], torch.tensor([[0.4], [0.4]])
    )
    assert _keys.NODE_P2_KEY not in direct
    assert _keys.EDGE_P2_KEY not in direct
    for field in (
        _keys.NODE_P23_BLOCKS_KEY,
        _keys.EDGE_P23_BLOCKS_KEY,
        _keys.NODE_P23_BLOCK_SHAPE_KEY,
        _keys.EDGE_P23_BLOCK_SHAPE_KEY,
    ):
        assert field not in direct

    residual = _compact_selected_prior_dataset(
        tmp_path,
        "dual-prior",
        p23_source,
        prior_kind="p23",
        require_blocks=True,
        audit=True,
    ).get(0)
    torch.testing.assert_close(
        residual[_keys.NODE_P23_BLOCKS_KEY].flatten(),
        torch.tensor([2.0, 2.1]),
    )
    torch.testing.assert_close(
        residual[_keys.EDGE_P23_BLOCKS_KEY].flatten(),
        torch.tensor([0.4, 0.4]),
    )
    assert _keys.NODE_P2_BLOCKS_KEY not in residual

    wrong_parent = deepcopy(record)
    wrong_parent[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY] = hashlib.sha256(
        b"another-record-p2-bundle"
    ).hexdigest()
    _refresh_dual_prior_fingerprints(wrong_parent)
    _write_single_lmdb(tmp_path, "dual-prior-wrong-parent", wrong_parent)
    with pytest.raises(ValueError, match="parent P2 bundle does not match"):
        _compact_selected_prior_dataset(
            tmp_path,
            "dual-prior-wrong-parent",
            p23_source,
            prior_kind="p23",
            require_blocks=False,
        ).get(0)

    wrong_schema = deepcopy(record)
    wrong_schema[SAMPLE_SCHEMA_KEY] = P2_SAMPLE_SCHEMA
    _refresh_dual_prior_fingerprints(wrong_schema)
    _write_single_lmdb(tmp_path, "dual-prior-v2", wrong_schema)
    with pytest.raises(ValueError, match="P23 tensors require explicit sample schema"):
        _compact_selected_prior_dataset(
            tmp_path,
            "dual-prior-v2",
            p23_source,
            prior_kind="p23",
            require_blocks=False,
        ).get(0)


def test_runtime_contract_scans_each_immutable_record_once_per_worker(
    tmp_path, monkeypatch
):
    record, _, p23_source = _compact_dual_prior_record()
    _write_single_lmdb(tmp_path, "dual-prior-runtime-cache", record)
    dataset = _compact_selected_prior_dataset(
        tmp_path,
        "dual-prior-runtime-cache",
        p23_source,
        prior_kind="p23",
        require_blocks=True,
    )

    counts = {
        "canonical_edge_graph": 0,
        "edge_graph_fingerprint": 0,
        "row_fingerprint": 0,
        "fingerprint_fields": 0,
        "fingerprint_text_fields": 0,
        "strict_packed_blocks": 0,
    }

    def _counted(module, name, counter):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            counts[counter] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    _counted(lmdb_dataset_module, "canonical_edge_graph", "canonical_edge_graph")
    _counted(lmdb_dataset_module, "edge_graph_fingerprint", "edge_graph_fingerprint")
    _counted(
        lmdb_dataset_module,
        "fingerprint_present_row_aligned_fields",
        "row_fingerprint",
    )
    _counted(lmdb_dataset_module, "fingerprint_fields", "fingerprint_fields")
    _counted(
        lmdb_dataset_module,
        "fingerprint_text_fields",
        "fingerprint_text_fields",
    )
    _counted(
        blockwise_tensor_module,
        "validate_packed_non_soc_blocks",
        "strict_packed_blocks",
    )

    feature_finite_flags = []
    original_feature_validator = lmdb_dataset_module.validate_p2_feature_pair

    def counted_feature_validator(*args, **kwargs):
        feature_finite_flags.append(kwargs.get("check_finite", True))
        return original_feature_validator(*args, **kwargs)

    monkeypatch.setattr(
        lmdb_dataset_module, "validate_p2_feature_pair", counted_feature_validator
    )
    block_expensive_flags = []
    original_block_validator = lmdb_dataset_module.validate_non_soc_p2_block_tensors

    def counted_block_validator(*args, **kwargs):
        block_expensive_flags.append(kwargs.get("expensive_checks", True))
        return original_block_validator(*args, **kwargs)

    monkeypatch.setattr(
        lmdb_dataset_module,
        "validate_non_soc_p2_block_tensors",
        counted_block_validator,
    )

    first = dataset.get(0)
    assert dataset._last_lmdb_pickle_bytes > 0
    assert dataset._last_lmdb_record_identity[1] == 0
    first_counts = dict(counts)
    second = dataset.get(0)

    torch.testing.assert_close(
        first[_keys.NODE_P23_KEY], second[_keys.NODE_P23_KEY]
    )
    assert all(value > 0 for value in first_counts.values())
    assert counts == first_counts
    assert feature_finite_flags == [True, False]
    assert block_expensive_flags == [True, False]
    assert len(dataset._validated_record_contracts) == 1
    assert dataset.__getstate__()["_validated_record_contracts"] == {}


def test_tampered_first_read_is_never_added_to_runtime_contract_cache(
    tmp_path, monkeypatch
):
    record, _, p23_source = _compact_dual_prior_record()
    record[_keys.NODE_P23_KEY] = record[_keys.NODE_P23_KEY].copy()
    record[_keys.NODE_P23_KEY][0, 0] += 0.25
    _write_single_lmdb(tmp_path, "dual-prior-runtime-tampered", record)
    dataset = _compact_selected_prior_dataset(
        tmp_path,
        "dual-prior-runtime-tampered",
        p23_source,
        prior_kind="p23",
        require_blocks=False,
    )

    calls = {"row": 0}
    original = lmdb_dataset_module.fingerprint_present_row_aligned_fields

    def counted_row_fingerprint(*args, **kwargs):
        calls["row"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        lmdb_dataset_module,
        "fingerprint_present_row_aligned_fields",
        counted_row_fingerprint,
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="row_aligned_data_fingerprint mismatch"):
            dataset.get(0)
    assert calls["row"] == 2
    assert dataset._validated_record_contracts == {}


def test_compact_p2_contract_preserves_graph_and_audits_cross_representation(tmp_path):
    record, p2_source = _compact_p2_record()
    _write_single_lmdb(tmp_path, "p2-compact-good", record)
    sample = _compact_dataset(tmp_path, "p2-compact-good", p2_source).get(0)
    torch.testing.assert_close(
        sample[_keys.EDGE_INDEX_KEY], torch.tensor([[1, 0], [0, 1]])
    )

    partial = deepcopy(record)
    del partial[_keys.EDGE_CELL_SHIFT_KEY]
    _write_single_lmdb(tmp_path, "p2-compact-partial", partial)
    with pytest.raises(ValueError, match="both edge_index and edge_cell_shift"):
        _compact_dataset(tmp_path, "p2-compact-partial", p2_source).get(0)

    inconsistent = deepcopy(record)
    inconsistent[_keys.EDGE_P2_BLOCKS_KEY][:] = 0.25
    _refresh_compact_fingerprints(inconsistent)
    _write_single_lmdb(tmp_path, "p2-compact-inconsistent", inconsistent)
    with pytest.raises(ValueError, match="RME/AO representations disagree"):
        _compact_dataset(tmp_path, "p2-compact-inconsistent", p2_source).get(0)


def _prior_model(
    device: str = "cpu",
    *,
    prior_kind: str = "p2",
    add_prior: bool = True,
    use_memory: bool = True,
):
    prior_node_key = f"node_{prior_kind}"
    prior_edge_key = f"edge_{prior_kind}"
    prediction = {
        "method": "block_native",
        "scale_type": "no_scale",
        "block_decoder": "expansion_cg",
        "blockwise_hamiltonian": True,
        "add_prior": add_prior,
    }
    if add_prior:
        prediction.update(
            {
                "prior_node_block_field": f"node_{prior_kind}_blocks",
                "prior_edge_block_field": f"edge_{prior_kind}_blocks",
                "prior_label": prior_kind.upper(),
            }
        )
    return build_model(
        common_options={
            "basis": {"H": "1s", "O": "1s1p"},
            "overlap": False,
            "dtype": "float32",
            "device": device,
        },
        model_options={
            "embedding": {
                "method": "lem_moe_v3_prior",
                "output_route": "h_b0",
                "n_layers": 1,
                "avg_num_neighbors": 2.0,
                "r_max": 4.0,
                "irreps_hidden": "4x0e+4x1o+4x1e+4x2e",
                "env_embed_multiplicity": 4,
                "latent_dim": 8,
                "latent_channels": [8],
                "edge_one_hot_dim": 4,
                "num_experts": 1,
                "num_shared_experts": 1,
                "top_k": 1,
                "universal": True,
                "use_layer_onehot_tp": False,
                "use_out_onehot_tp": False,
                "use_interpolation_out": False,
                "tp_radial_emb": False,
                "mole_linear_mode": "indexed_ref",
                "so2_fusion_mode": "streamed_m_major_ref",
                "rme_fusion_rank": 4,
                "rme_fusion_init": 0.0,
                "prior_kind": prior_kind,
                "prior_node_key": prior_node_key,
                "prior_edge_key": prior_edge_key,
                "use_soft_edge_memory": use_memory,
                "soft_edge_memory_num_slots": 8,
                "soft_edge_memory_num_heads": 2,
                "soft_edge_memory_head_dim": 4,
                "soft_edge_memory_gate_mode": "deepseek",
                "soft_edge_memory_diagnostics_mode": "full",
            },
            "prediction": prediction,
        },
        train_options={},
        no_check=False,
    )


def _prior_data(model):
    device = next(model.parameters()).device
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_type = torch.tensor(
        [model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]],
        dtype=torch.long,
        device=device,
    )
    rme = model.idp.orbpair_irreps.dim
    max_norb = model.idp.full_basis_norb
    prior_kind = model.embedding.prior_kind
    node_prior_key = f"node_{prior_kind}"
    edge_prior_key = f"edge_{prior_kind}"
    node_prior_blocks_key = f"node_{prior_kind}_blocks"
    edge_prior_blocks_key = f"edge_{prior_kind}_blocks"
    data = {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        ),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros(
            2, 3, dtype=torch.float32, device=device
        ),
        _keys.ATOM_TYPE_KEY: torch.tensor(
            [[h], [o]], dtype=torch.long, device=device
        ),
        _keys.EDGE_TYPE_KEY: edge_type,
        node_prior_key: torch.randn(2, rme, device=device),
        edge_prior_key: torch.randn(2, rme, device=device),
    }
    packed = feature_tensors_to_block_tensors(
        data,
        model.idp,
        node_features=data[node_prior_key],
        edge_features=data[edge_prior_key],
        node_pad_shape=(max_norb, max_norb),
        edge_pad_shape=(max_norb, max_norb),
        complete_edges=True,
        strict_complete_edges=True,
    )
    data[node_prior_blocks_key] = packed.node_blocks
    data[edge_prior_blocks_key] = packed.edge_blocks
    data[f"node_{prior_kind}_block_shape"] = packed.node_shapes
    data[f"edge_{prior_kind}_block_shape"] = packed.edge_shapes
    return data


def test_p23_embedding_supports_residual_and_direct_full_h_heads():
    torch.manual_seed(23)
    residual_model = _prior_model(prior_kind="p23", add_prior=True)
    residual_data = _prior_data(residual_model)
    p23_node = residual_data[_keys.NODE_P23_BLOCKS_KEY].clone()
    p23_edge = residual_data[_keys.EDGE_P23_BLOCKS_KEY].clone()
    residual_output = residual_model(residual_data)
    torch.testing.assert_close(
        residual_output["node_full_hamil_blocks"],
        p23_node + residual_output[NODE_PRED_HAMIL_BLOCKS_KEY],
    )
    torch.testing.assert_close(
        residual_output["edge_full_hamil_blocks"],
        p23_edge + residual_output[EDGE_PRED_HAMIL_BLOCKS_KEY],
    )

    direct_model = _prior_model(
        prior_kind="p23", add_prior=False, use_memory=False
    )
    direct_output = direct_model(_prior_data(direct_model))
    assert NODE_PRED_HAMIL_BLOCKS_KEY in direct_output
    assert EDGE_PRED_HAMIL_BLOCKS_KEY in direct_output
    assert "node_full_hamil_blocks" not in direct_output
    assert "edge_full_hamil_blocks" not in direct_output


def test_prior_embedding_model_full_h_loss_backward_checkpoint_and_inference(tmp_path):
    torch.manual_seed(19)
    model = _prior_model()
    data = _prior_data(model)
    p2_node = data[NODE_P2_BLOCKS_KEY].clone()
    p2_edge = data[EDGE_P2_BLOCKS_KEY].clone()
    output = model(data)

    correction_node = output[NODE_PRED_HAMIL_BLOCKS_KEY]
    correction_edge = output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    torch.testing.assert_close(
        output["node_full_hamil_blocks"], p2_node + correction_node
    )
    torch.testing.assert_close(
        output["edge_full_hamil_blocks"], p2_edge + correction_edge
    )
    for key in (
        "soft_edge_memory_attention_entropy",
        "soft_edge_memory_attention_max_probability",
        "soft_edge_memory_gate_mean",
    ):
        assert key in output and torch.isfinite(output[key])

    output[NODE_FULL_HAMIL_TARGET_BLOCKS_KEY] = (
        output["node_full_hamil_blocks"].detach() + 0.01
    )
    output[EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY] = (
        output["edge_full_hamil_blocks"].detach() - 0.01
    )
    output[NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY] = output[
        "node_hamil_block_shape"
    ].clone()
    output[EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY] = output[
        "edge_hamil_block_shape"
    ].clone()
    loss = HamilBlockwiseNexTHamLoss(
        idp=model.idp,
        pred_node_block_key="node_full_hamil_blocks",
        pred_edge_block_key="edge_full_hamil_blocks",
        target_node_block_key=NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
        target_edge_block_key=EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
        target_node_shape_key=NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
        target_edge_shape_key=EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    )(output)
    assert loss.item() > 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()

    checkpoint_path = tmp_path / "p2_prior_smoke.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    restored = _prior_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    restored.eval()

    inference_source = _prior_data(model)

    def _clone_input():
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in inference_source.items()
        }

    with torch.no_grad():
        expected = model(_clone_input())
        actual = restored(_clone_input())
    torch.testing.assert_close(
        actual["node_full_hamil_blocks"], expected["node_full_hamil_blocks"]
    )
    torch.testing.assert_close(
        actual["edge_full_hamil_blocks"], expected["edge_full_hamil_blocks"]
    )

    missing = _prior_data(model)
    del missing[_keys.EDGE_P2_KEY]
    with pytest.raises(KeyError, match="edge P2 field"):
        model(missing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_prior_embedding_cuda_forward_backward_smoke():
    torch.manual_seed(23)
    model = _prior_model("cuda")
    data = _prior_data(model)
    output = model(data)
    loss = (
        output["node_full_hamil_blocks"].square().mean()
        + output["edge_full_hamil_blocks"].square().mean()
    )
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
