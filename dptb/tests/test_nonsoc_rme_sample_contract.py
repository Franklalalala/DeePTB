import numpy as np
import pytest
from types import SimpleNamespace

from dptb.data.interfaces.p2_contract import (
    BASIS_FINGERPRINT_KEY,
    DENSITY_MATRIX_RME_SEMANTICS,
    EDGE_GRAPH_FINGERPRINT_KEY,
    NONSOC_DM_RME_SAMPLE_SCHEMA,
    NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA,
    NONSOC_P23_RESIDUAL_RME_SAMPLE_SCHEMA,
    P2_BLOCK_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_RESIDUAL_RME_SEMANTICS,
    P2_RME_FINGERPRINT_KEY,
    P2_SOURCE_FINGERPRINT_KEY,
    P23_BLOCK_FINGERPRINT_KEY,
    P23_BUNDLE_FINGERPRINT_KEY,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    P23_RESIDUAL_RME_SEMANTICS,
    P23_RME_FINGERPRINT_KEY,
    P23_SOURCE_FINGERPRINT_KEY,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    assert_nonsoc_rme_sample_contract,
    edge_graph_fingerprint,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
)
from dptb.data.dataset.record_pipeline import RecordSchemaValidator


def _base_record(schema: str, semantics: str) -> dict:
    record = {
        SAMPLE_SCHEMA_KEY: schema,
        TARGET_SEMANTICS_KEY: semantics,
        "hamiltonian_target_source": "test_oracle",
        "atomic_numbers": np.asarray([1, 1], dtype=np.int64),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "edge_cell_shift": np.zeros((2, 3), dtype=np.int64),
        "node_features": np.zeros((2, 4), dtype=np.float32),
        "edge_features": np.zeros((2, 4), dtype=np.float32),
        "raw_case_source_fingerprint": "a" * 64,
        BASIS_FINGERPRINT_KEY: "1" * 64,
    }
    if schema == NONSOC_DM_RME_SAMPLE_SCHEMA:
        record["node_h0"] = np.ones((2, 4), dtype=np.float32)
        record["edge_h0"] = np.ones((2, 4), dtype=np.float32)
    elif schema == NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA:
        record["node_p2"] = np.ones((2, 4), dtype=np.float32)
        record["edge_p2"] = np.ones((2, 4), dtype=np.float32)
        record[P2_SOURCE_FINGERPRINT_KEY] = "5" * 64
        record[P2_RME_FINGERPRINT_KEY] = "6" * 64
        record[P2_BLOCK_FINGERPRINT_KEY] = "b" * 64
    else:
        record["node_p23"] = np.ones((2, 4), dtype=np.float32)
        record["edge_p23"] = np.ones((2, 4), dtype=np.float32)
        record[P2_SOURCE_FINGERPRINT_KEY] = "5" * 64
        record[P2_RME_FINGERPRINT_KEY] = "6" * 64
        record[P2_BLOCK_FINGERPRINT_KEY] = "b" * 64
        record[P23_SOURCE_FINGERPRINT_KEY] = "7" * 64
        record[P23_RME_FINGERPRINT_KEY] = "8" * 64
        record[P23_BLOCK_FINGERPRINT_KEY] = "c" * 64
    record[EDGE_GRAPH_FINGERPRINT_KEY] = edge_graph_fingerprint(
        record["atomic_numbers"],
        record["edge_index"],
        record["edge_cell_shift"],
        basis_fingerprint=record[BASIS_FINGERPRINT_KEY],
    )
    if schema in {
        NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA,
        NONSOC_P23_RESIDUAL_RME_SAMPLE_SCHEMA,
    }:
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
    if schema == NONSOC_P23_RESIDUAL_RME_SAMPLE_SCHEMA:
        record[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY] = record[
            P2_BUNDLE_FINGERPRINT_KEY
        ]
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


@pytest.mark.parametrize(
    ("schema", "semantics"),
    (
        (NONSOC_DM_RME_SAMPLE_SCHEMA, DENSITY_MATRIX_RME_SEMANTICS),
        (NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA, P2_RESIDUAL_RME_SEMANTICS),
        (NONSOC_P23_RESIDUAL_RME_SAMPLE_SCHEMA, P23_RESIDUAL_RME_SEMANTICS),
    ),
)
def test_nonsoc_rme_contract_accepts_complete_slim_views(schema: str, semantics: str):
    assert_nonsoc_rme_sample_contract(_base_record(schema, semantics))


def test_nonsoc_rme_contract_rejects_missing_stored_graph():
    record = _base_record(
        NONSOC_DM_RME_SAMPLE_SCHEMA, DENSITY_MATRIX_RME_SEMANTICS
    )
    record.pop("edge_cell_shift")
    with pytest.raises(ValueError, match="edge_cell_shift"):
        assert_nonsoc_rme_sample_contract(record)


def test_nonsoc_rme_contract_rejects_semantic_mismatch_and_nonfinite_rows():
    record = _base_record(
        NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA, P2_RESIDUAL_RME_SEMANTICS
    )
    record[TARGET_SEMANTICS_KEY] = DENSITY_MATRIX_RME_SEMANTICS
    with pytest.raises(ValueError, match="requires hamiltonian_target_semantics"):
        assert_nonsoc_rme_sample_contract(record)

    record[TARGET_SEMANTICS_KEY] = P2_RESIDUAL_RME_SEMANTICS
    record["edge_features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN or infinity"):
        assert_nonsoc_rme_sample_contract(record)


def test_nonsoc_rme_contract_rejects_same_count_edge_row_permutation():
    record = _base_record(
        NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA, P2_RESIDUAL_RME_SEMANTICS
    )
    record["edge_index"] = record["edge_index"][:, ::-1].copy()
    with pytest.raises(ValueError, match="edge_graph_fingerprint mismatch"):
        assert_nonsoc_rme_sample_contract(record)


def test_nonsoc_p23_rme_contract_rejects_wrong_parent_p2_bundle():
    record = _base_record(
        NONSOC_P23_RESIDUAL_RME_SAMPLE_SCHEMA, P23_RESIDUAL_RME_SEMANTICS
    )
    record[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY] = "a" * 64
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
    with pytest.raises(ValueError, match="parent P2 bundle"):
        assert_nonsoc_rme_sample_contract(record)


def test_nonsoc_rme_contract_rejects_persisted_ao_blocks():
    record = _base_record(
        NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA, P2_RESIDUAL_RME_SEMANTICS
    )
    record["node_p2_blocks"] = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="forbids persisted AO/block-native"):
        assert_nonsoc_rme_sample_contract(record)


def test_loader_rejects_p2_p23_schema_cross_wiring():
    record = _base_record(
        NONSOC_P2_RESIDUAL_RME_SAMPLE_SCHEMA, P2_RESIDUAL_RME_SEMANTICS
    )
    dataset = SimpleNamespace(
        require_prior_residual_rme_target=True,
        prior_kind="p23",
    )
    with pytest.raises(ValueError, match="prior_kind='p23'.*requires"):
        RecordSchemaValidator().validate_schema_and_basis(dataset, record)
