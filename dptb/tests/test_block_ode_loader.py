"""Block-ODE LMDB loader contracts: physical-H0 authority, raw full-H/residual materialization,
converter-product provenance gates and the residual-shrink H0-quality guard."""
from __future__ import annotations

import json
import logging
import pickle
import shutil

import lmdb
import numpy as np
import pytest
import torch

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    LMDBDataset,
    assert_absolute_full_h_target_contract,
    assert_residual_target_shrinks,
    build_residual_hamiltonian_target_blocks,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    DEDICATED_PHYSICAL_H0_SOURCE,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    H0_RESIDUAL_SEMANTICS,
    P2_SAMPLE_SCHEMA,
    PHYSICAL_H0_SOURCE_FINGERPRINT_KEY,
    PHYSICAL_H0_SOURCE_KEY,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    RAW_PHYSICAL_H0_SOURCE,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    physical_h0_dataset_fingerprint,
    physical_h0_record_fingerprint,
)
from dptb.tests.block_ode_fixtures import _uureal_mapper

FP64_ATOL = 1e-10


def _raw_h_h0_record() -> dict:
    """A raw H2 record: onsite H=10/12 over H0=9/11, offsite H=3 over H0=2.5 (delta [1, 1] / [.5, .5])."""
    h_blocks = {
        "0_0_0_0_0": np.asarray([[10.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[12.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[3.0]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[3.0]], dtype=np.float32),
    }
    h0_blocks = {
        "0_0_0_0_0": np.asarray([[9.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[11.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[2.5]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[2.5]], dtype=np.float32),
    }
    return {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        "case_id": "h2",
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_0": h0_blocks,
        SAMPLE_SCHEMA_KEY: RAW_HAMILTONIAN_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "raw_hamiltonian",
    }


def _dedicated_h0_record(schema: str) -> dict:
    record = _raw_h_h0_record()
    record.pop("hamiltonian_0")
    blocks = np.zeros((2, 1, 1), dtype=np.float32)
    shapes = np.ones((2, 2), dtype=np.int64)
    record.update(
        {
            SAMPLE_SCHEMA_KEY: schema,
            TARGET_SOURCE_KEY: "dedicated_full_h_blocks",
            PHYSICAL_H0_SOURCE_KEY: DEDICATED_PHYSICAL_H0_SOURCE,
            BASIS_FINGERPRINT_KEY: "1" * 64,
            EDGE_GRAPH_FINGERPRINT_KEY: "2" * 64,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY: "3" * 64,
            ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY: "4" * 64,
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: blocks.copy(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: blocks.copy(),
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: shapes.copy(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: shapes.copy(),
            _keys.NODE_H0_BLOCKS_KEY: blocks.copy(),
            _keys.EDGE_H0_BLOCKS_KEY: blocks.copy(),
            _keys.NODE_H0_BLOCK_SHAPE_KEY: shapes.copy(),
            _keys.EDGE_H0_BLOCK_SHAPE_KEY: shapes.copy(),
        }
    )
    record[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY] = physical_h0_dataset_fingerprint(
        [physical_h0_record_fingerprint(record)]
    )
    return record


def _dataset_from_record(tmp_path, record: dict, *, name: str, **kwargs) -> LMDBDataset:
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    dataset = DatasetBuilder()(
        root=str(tmp_path), r_max=2.0, type="LMDBDataset", prefix=name, separator=".",
        basis={"H": "1s"}, get_Hamiltonian=True, get_H0=True, **kwargs,
    )
    assert isinstance(dataset, LMDBDataset)
    return dataset


def _two_record_dataset(tmp_path, records, *, name: str, **kwargs) -> LMDBDataset:
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            for row_id, record in enumerate(records):
                txn.put(row_id.to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    return DatasetBuilder()(
        root=str(tmp_path), r_max=2.0, type="LMDBDataset", prefix=name, separator=".",
        basis={"H": "1s"}, get_Hamiltonian=True, get_H0=True, **kwargs,
    )


# ---------------------------------------------------------------------------
# Physical-H0 authority contract (direct calls): declared raw or dedicated
# source accepted; every fail-closed mutation of the dedicated path rejected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema", (P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA))
def test_full_h_h0_authority_accepts_declared_raw_or_dedicated(schema: str):
    raw = _raw_h_h0_record()
    raw[SAMPLE_SCHEMA_KEY] = schema
    raw[PHYSICAL_H0_SOURCE_KEY] = RAW_PHYSICAL_H0_SOURCE
    assert_absolute_full_h_target_contract(raw, require_h0=True)
    dedicated = _dedicated_h0_record(schema)
    assert_absolute_full_h_target_contract(
        dedicated, require_h0=True,
        expected_physical_h0_source_fingerprint=dedicated[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY],
    )


@pytest.mark.parametrize("schema", (P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA))
@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("undeclared", "physical_h0_source"),
        ("partial", "bundle is incomplete"),
        ("dual_authority", "ambiguous dual authority"),
        ("unbound", "row_aligned_bundle_fingerprint"),
        ("missing_record_fingerprint", "physical_h0_source_fingerprint"),
        ("missing_external_fingerprint", "externally trusted"),
        ("source_mismatch", "does not match"),
    ),
)
def test_dedicated_full_h_h0_authority_is_fail_closed(schema: str, mutation: str, match: str):
    record = _dedicated_h0_record(schema)
    expected = record[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY]
    if mutation == "undeclared":
        record.pop(PHYSICAL_H0_SOURCE_KEY)
    elif mutation == "partial":
        record.pop(_keys.EDGE_H0_BLOCKS_KEY)
    elif mutation == "dual_authority":
        record["hamiltonian_0"] = {"0_0_0_0_0": np.zeros((1, 1))}
    elif mutation == "unbound":
        record.pop(ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY)
    elif mutation == "missing_record_fingerprint":
        record.pop(PHYSICAL_H0_SOURCE_FINGERPRINT_KEY)
    elif mutation == "missing_external_fingerprint":
        expected = None
    else:
        expected = "f" * 64
    with pytest.raises(ValueError, match=match):
        assert_absolute_full_h_target_contract(
            record, require_h0=True, expected_physical_h0_source_fingerprint=expected
        )


@pytest.mark.parametrize("schema", (P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA))
def test_full_h_h0_authority_is_opt_in_for_legacy_routes(schema: str):
    record = _dedicated_h0_record(schema)
    record.pop(PHYSICAL_H0_SOURCE_KEY)
    assert_absolute_full_h_target_contract(record, require_h0=False)


# ---------------------------------------------------------------------------
# H9: the Full-H route and the residual-from-Full-H route consume the same raw
# records and must agree on physical-H0 authority self-consistency.
# ---------------------------------------------------------------------------
def _raw_record_with_dedicated_h0_source(*, fingerprint=None) -> dict:
    record = _raw_h_h0_record()
    record[PHYSICAL_H0_SOURCE_KEY] = DEDICATED_PHYSICAL_H0_SOURCE
    if fingerprint is not None:
        record[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY] = fingerprint
    return record


def test_h9_residual_route_matches_full_h_route_and_rejects_contradictory_authority(tmp_path):
    """A record contradicting its own schema-fixed raw authority (claiming dedicated physical-H0
    provenance) must fail closed on the residual-from-full-H route exactly as the Full-H route
    does, instead of silently double-subtracting raw hamiltonian_0."""
    dataset = _dataset_from_record(
        tmp_path, _raw_record_with_dedicated_h0_source(), name="h9-residual-repro",
        residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="authority"):
        dataset.get(0)


def test_h9_residual_route_rejects_dedicated_source_with_wrong_fingerprint(tmp_path):
    dataset = _dataset_from_record(
        tmp_path, _raw_record_with_dedicated_h0_source(fingerprint="0" * 64),
        name="h9-residual-wrong-fingerprint",
        residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="authority"):
        dataset.get(0)


@pytest.mark.parametrize("physical_h0_source", [None, RAW_PHYSICAL_H0_SOURCE], ids=["unset", "explicit-raw"])
def test_h9_residual_route_does_not_misfire_on_legitimate_raw_authority(tmp_path, physical_h0_source):
    record = _raw_h_h0_record()
    if physical_h0_source is not None:
        record[PHYSICAL_H0_SOURCE_KEY] = physical_h0_source
    dataset = _dataset_from_record(
        tmp_path, record, name=f"h9-legit-{physical_h0_source or 'unset'}",
        residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Raw full-H schema: materialization, required metadata, prepacked overrides
# ---------------------------------------------------------------------------
def test_generic_raw_full_h_schema_rejects_unversioned_dataset_prior(tmp_path):
    record = _raw_h_h0_record()
    record["hamiltonian_p2"] = {key: np.asarray(value) * 0.5 for key, value in record["hamiltonian"].items()}
    dataset = _dataset_from_record(
        tmp_path, record, name="raw-full-h-unversioned-prior",
        residual_hamiltonian=False, require_full_h_target=True,
        get_prior=True, expected_prior_source_fingerprint="f" * 64,
    )
    with pytest.raises(ValueError, match="versioned P2/dual-prior sample schema"):
        dataset.get(0)


def test_generic_raw_full_h_schema_materializes_dedicated_target_blocks(tmp_path):
    dataset = _dataset_from_record(
        tmp_path, _raw_h_h0_record(), name="block-ode-full-h",
        residual_hamiltonian=False, require_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(sample[_keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(), torch.tensor([10.0, 12.0]))
    torch.testing.assert_close(sample[_keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(), torch.tensor([3.0, 3.0]))
    assert torch.equal(sample[_keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY], torch.ones((2, 2), dtype=torch.long))
    assert torch.equal(sample[_keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY], torch.ones((2, 2), dtype=torch.long))
    assert _keys.NODE_P2_KEY not in sample and _keys.EDGE_P2_KEY not in sample
    torch.testing.assert_close(sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0]))
    torch.testing.assert_close(sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5]))


@pytest.mark.parametrize(
    ("missing_key", "match"),
    (
        (SAMPLE_SCHEMA_KEY, "explicit sample schema"),
        (TARGET_SEMANTICS_KEY, "requires hamiltonian_target_semantics"),
        (TARGET_SOURCE_KEY, "hamiltonian_target_source"),
    ),
)
def test_generic_raw_full_h_metadata_is_fail_closed(tmp_path, missing_key: str, match: str):
    record = _raw_h_h0_record()
    record.pop(missing_key)
    dataset = _dataset_from_record(
        tmp_path, record, name=f"missing-{missing_key.replace('_', '-')}",
        residual_hamiltonian=False, require_full_h_target=True,
    )
    with pytest.raises(ValueError, match=match):
        dataset.get(0)


@pytest.mark.parametrize(
    ("missing_key", "match"),
    (("hamiltonian", "raw.*hamiltonian"), ("hamiltonian_0", "physical-H0 block dictionary")),
)
def test_generic_raw_full_h_contract_requires_both_raw_h_and_physical_h0(tmp_path, missing_key: str, match: str):
    record = _raw_h_h0_record()
    record.pop(missing_key)
    dataset = _dataset_from_record(
        tmp_path, record, name=f"missing-raw-{missing_key.replace('_', '-')}",
        residual_hamiltonian=False, require_full_h_target=True,
    )
    with pytest.raises(ValueError, match=match):
        dataset.get(0)


def test_generic_raw_full_h_contract_rejects_prepacked_target_override(tmp_path):
    record = _raw_h_h0_record()
    record.update(
        {
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros((2, 1, 1), dtype=np.float32),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros((2, 1, 1), dtype=np.float32),
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        }
    )
    dataset = _dataset_from_record(
        tmp_path, record, name="prepacked-target-override", residual_hamiltonian=False, require_full_h_target=True,
    )
    with pytest.raises(ValueError, match="independently prepacked"):
        dataset.get(0)


def _prepacked_h0_fields(*, complete: bool) -> dict:
    fields = {_keys.NODE_H0_BLOCKS_KEY: np.zeros((2, 1, 1), dtype=np.float32)}
    if complete:
        fields.update(
            {
                _keys.EDGE_H0_BLOCKS_KEY: np.zeros((2, 1, 1), dtype=np.float32),
                _keys.NODE_H0_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
                _keys.EDGE_H0_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
            }
        )
    return fields


@pytest.mark.parametrize("complete", (False, True))
@pytest.mark.parametrize(("residual_hamiltonian", "require_full_h_target"), ((False, True), (True, False)))
def test_raw_authority_rejects_prepacked_h0_block_override(tmp_path, complete, residual_hamiltonian, require_full_h_target):
    record = _raw_h_h0_record()
    if residual_hamiltonian:
        record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    record.update(_prepacked_h0_fields(complete=complete))
    dataset = _dataset_from_record(
        tmp_path, record,
        name=f"prepacked-h0-{'complete' if complete else 'partial'}-{'residual' if residual_hamiltonian else 'absolute'}",
        residual_hamiltonian=residual_hamiltonian, require_full_h_target=require_full_h_target,
    )
    with pytest.raises(ValueError, match="prepacked H0 block fields"):
        dataset.get(0)


def test_precomputed_h0_features_do_not_override_raw_h0_blocks(tmp_path):
    record = _raw_h_h0_record()
    record[_keys.NODE_H0_KEY] = np.full((2, 1), 101.0, dtype=np.float32)
    record[_keys.EDGE_H0_KEY] = np.full((2, 1), 103.0, dtype=np.float32)
    dataset = _dataset_from_record(
        tmp_path, record, name="precomputed-h0-features", residual_hamiltonian=False, require_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(sample[_keys.NODE_H0_KEY], torch.full((2, 1), 101.0))
    torch.testing.assert_close(sample[_keys.EDGE_H0_KEY], torch.full((2, 1), 103.0))
    torch.testing.assert_close(sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0]))
    torch.testing.assert_close(sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5]))


def test_residual_loader_materializes_delta_and_h0_block_side_channels(tmp_path):
    """The direct h0_residual route (residual_hamiltonian=True on an already-h0_residual record):
    the legacy product feature stays absolute H, and the authoritative dH/H0 AO block side
    channels materialize independently."""
    record = _raw_h_h0_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    dataset = _dataset_from_record(
        tmp_path, record, name="block-ode-residual",
        residual_hamiltonian=True, require_full_h_target=False, require_residual_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(sample[_keys.NODE_FEATURES_KEY].flatten(), torch.tensor([10.0, 12.0]))
    torch.testing.assert_close(sample[_keys.EDGE_FEATURES_KEY].flatten(), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0]))
    torch.testing.assert_close(sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5]))


@pytest.mark.parametrize(
    ("bad_key", "bad_value", "match"),
    (
        (SAMPLE_SCHEMA_KEY, None, "explicit sample schema"),
        (TARGET_SEMANTICS_KEY, None, "hamiltonian_target_semantics"),
        (TARGET_SEMANTICS_KEY, ABSOLUTE_FULL_H_SEMANTICS, "h0_residual"),
        (TARGET_SOURCE_KEY, None, "hamiltonian_target_source"),
        (TARGET_SOURCE_KEY, "dedicated_full_h_blocks", "raw_hamiltonian"),
    ),
)
def test_residual_loader_metadata_is_fail_closed(tmp_path, bad_key: str, bad_value, match: str):
    record = _raw_h_h0_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    if bad_value is None:
        record.pop(bad_key)
    else:
        record[bad_key] = bad_value
    dataset = _dataset_from_record(
        tmp_path, record, name=f"residual-bad-{bad_key}-{bad_value}",
        residual_hamiltonian=True, require_full_h_target=False, require_residual_h_target=True,
    )
    with pytest.raises(ValueError, match=match):
        dataset.get(0)


# ---------------------------------------------------------------------------
# require_residual_from_full_h_target: materializes D1 = raw H - H0 from the
# SAME absolute_full_h raw records the Full-H route uses; converter/compact
# provenance markers on such a record are a masquerade and must fail closed
# before any subtraction.
# ---------------------------------------------------------------------------
_CONVERTER_MARKERS = {
    "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
    "blockwise_target_mode": "already-delta",
    "soc_uureal_compact": True,
    "soc_uureal_full_rme": 8,
    "soc_uureal_keep": 1,
    "blockwise_source_target_feature_width": 8,
    "blockwise_source_h0_feature_width": 8,
}


def test_loader_materializes_residual_from_absolute_full_h_record(tmp_path):
    dataset = _dataset_from_record(
        tmp_path, _raw_h_h0_record(), name="b-residual-from-full-h",
        residual_hamiltonian=True, require_full_h_target=False, require_residual_h_target=False,
        require_residual_from_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0]))
    torch.testing.assert_close(sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5]))


def test_loader_rejects_h0_residual_semantics_under_new_flag(tmp_path):
    record = _raw_h_h0_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    dataset = _dataset_from_record(
        tmp_path, record, name="b-wrong-semantics", residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="absolute_full_h"):
        dataset.get(0)


def test_loader_requires_physical_h0_dictionary(tmp_path):
    record = _raw_h_h0_record()
    record.pop("hamiltonian_0")
    dataset = _dataset_from_record(
        tmp_path, record, name="b-missing-h0", residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="hamiltonian_0"):
        dataset.get(0)


def test_loader_new_flag_mutually_exclusive_with_residual_h_target(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _dataset_from_record(
            tmp_path, _raw_h_h0_record(), name="b-excl-residual-h",
            residual_hamiltonian=True, require_residual_h_target=True, require_residual_from_full_h_target=True,
        )


def test_loader_new_flag_requires_residual_hamiltonian(tmp_path):
    with pytest.raises(ValueError, match="residual_hamiltonian"):
        _dataset_from_record(
            tmp_path, _raw_h_h0_record(), name="b-needs-residual",
            residual_hamiltonian=False, require_residual_from_full_h_target=True,
        )


def test_loader_new_flag_mutually_exclusive_with_uureal_block_ode(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive|already-delta|stay false"):
        _dataset_from_record(
            tmp_path, _raw_h_h0_record(), name="b-excl-uureal",
            residual_hamiltonian=True, require_uureal_block_ode=True, require_residual_from_full_h_target=True,
        )


def test_frozen_require_residual_h_target_gate_still_rejects_absolute_record(tmp_path):
    """The frozen require_residual_h_target gate (h0_residual demanded) is unchanged by the new mode."""
    dataset = _dataset_from_record(
        tmp_path, _raw_h_h0_record(), name="b-frozen-residual-gate", residual_hamiltonian=True, require_residual_h_target=True,
    )
    with pytest.raises(ValueError, match="h0_residual"):
        dataset.get(0)


@pytest.mark.parametrize("marker,value", sorted(_CONVERTER_MARKERS.items()))
def test_loader_rejects_each_converter_marker_on_a_residual_from_full_h_record(tmp_path, marker, value):
    """An otherwise-clean absolute_full_h record bearing a single converter/compact provenance
    marker fails closed on the residual-from-full-H route, naming that marker."""
    record = _raw_h_h0_record()
    record[marker] = value
    dataset = _dataset_from_record(
        tmp_path, record, name=f"b-single-marker-{marker}",
        residual_hamiltonian=True, require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match=marker):
        dataset.get(0)


# ===========================================================================
# H7: residual_shrink_policy H0-quality gate (error/warn/off) and its ratio.
# ===========================================================================
def _shrink_inputs(h, h0):
    data_dict = {"hamiltonian_0": {"0_0_0_0_0": np.asarray([[h0]], dtype=np.float32)}}
    blocks = {"0_0_0_0_0": np.asarray([[h]], dtype=np.float32)}
    return data_dict, blocks


@pytest.mark.parametrize(
    ("policy", "expect_raise", "expect_warning"),
    [("error", True, None), ("warn", False, True), ("off", False, False)],
)
def test_h7_shrink_policy_controls_raise_and_log_on_non_shrinking_record(policy, expect_raise, expect_warning, caplog):
    """A legit-but-non-shrinking record (H=1, H0=0 -> D=1): 'error' fails closed naming the
    H0-quality gate; 'warn' loads (delta==1) and logs; 'off' loads silently."""
    data_dict, blocks = _shrink_inputs(1.0, 0.0)
    if expect_raise:
        with pytest.raises(RuntimeError, match="H0-quality"):
            build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy=policy)
        return
    with caplog.at_level(logging.WARNING, logger="dptb.data.dataset.lmdb_dataset"):
        delta = build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy=policy)
    assert float(delta["0_0_0_0_0"][0, 0]) == pytest.approx(1.0)
    logged = any("residual_shrink_policy" in rec.getMessage() for rec in caplog.records)
    assert logged is bool(expect_warning)


def test_h7_min_residual_shrink_ratio_is_honored():
    """min_residual_shrink tunes the required ratio: a 1.3x-shrinking record (H=1.3, H0=0.3 ->
    D=1.0) passes at 1.2 but fails at 2.0."""
    data_dict, blocks = _shrink_inputs(1.3, 0.3)
    delta = build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy="error", min_shrink=1.2)
    assert float(delta["0_0_0_0_0"][0, 0]) == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="H0-quality"):
        build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy="error", min_shrink=2.0)


def _raw_non_shrinking_record() -> dict:
    record = _raw_h_h0_record()
    for key in record["hamiltonian"]:
        record["hamiltonian"][key] = np.asarray([[1.0]], dtype=np.float32)
        record["hamiltonian_0"][key] = np.asarray([[0.0]], dtype=np.float32)
    return record


def test_h7_dataset_ctor_arg_reaches_the_shrink_gate(tmp_path):
    error_ds = _dataset_from_record(
        tmp_path, _raw_non_shrinking_record(), name="b-nonshrink-error",
        residual_hamiltonian=True, require_residual_from_full_h_target=True, residual_shrink_policy="error",
    )
    with pytest.raises(RuntimeError, match="H0-quality"):
        error_ds.get(0)
    off_ds = _dataset_from_record(
        tmp_path, _raw_non_shrinking_record(), name="b-nonshrink-off",
        residual_hamiltonian=True, require_residual_from_full_h_target=True, residual_shrink_policy="off",
    )
    sample = off_ds.get(0)
    torch.testing.assert_close(sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0]))


def test_residual_guard_accepts_genuine_full_h():
    """Full-H slot: subtracting a close H0 shrinks the target -> guard passes."""
    rng = np.random.default_rng(0)
    h0 = {"1_1_0_0_0": rng.normal(0.0, 1.0, (5, 5)), "1_2_0_0_0": rng.normal(0.0, 0.5, (5, 5))}
    blocks = {k: v + rng.normal(0.0, 0.02, v.shape) for k, v in h0.items()}
    delta = {k: np.asarray(blocks[k]) - np.asarray(h0[k]) for k in blocks}
    assert_residual_target_shrinks(blocks, delta)


def test_residual_guard_rejects_delta_in_h_slot():
    """Delta-in-H-slot convention (the 'hamiltonian' slot already holds a small residual, H0 is
    full-H scale): subtracting H0 inflates -> guard raises."""
    rng = np.random.default_rng(1)
    blocks = {"1_1_0_0_0": rng.normal(0.0, 0.015, (5, 5))}
    h0 = {"1_1_0_0_0": rng.normal(0.0, 1.0, (5, 5))}
    delta = {k: np.asarray(blocks[k]) - np.asarray(h0[k]) for k in blocks}
    with pytest.raises(RuntimeError, match="double-subtract"):
        assert_residual_target_shrinks(blocks, delta)


def test_residual_guard_keeps_complex_magnitude():
    """Imaginary components must participate in the shrink decision: real-only magnitudes would
    see a 10x shrink and pass, but the complex magnitude shrinks by only ~1.12x."""
    blocks = {"1_1_0_0_0": np.array([[1.0 + 10.0j]])}
    delta = {"1_1_0_0_0": np.array([[0.1 + 9.0j]])}
    with pytest.raises(RuntimeError, match="does not shrink"):
        assert_residual_target_shrinks(blocks, delta)


def test_residual_builder_rejects_prepacked_target_provenance():
    data = {"hamiltonian_0": {"1_1_0_0_0": np.zeros((1, 1))}, "node_delta_hamil_blocks": np.zeros((1, 1, 1))}
    blocks = {"1_1_0_0_0": np.ones((1, 1))}
    with pytest.raises(ValueError, match="already contains prepacked"):
        build_residual_hamiltonian_target_blocks(data, blocks)


def test_residual_builder_validates_every_call_and_block_shape():
    good_blocks = {"1_1_0_0_0": np.ones((2, 2))}
    good_data = {"hamiltonian_0": {"1_1_0_0_0": np.full((2, 2), 0.95)}}
    delta = build_residual_hamiltonian_target_blocks(good_data, good_blocks)
    assert np.allclose(delta["1_1_0_0_0"], 0.05)
    bad_data = {"hamiltonian_0": {"1_1_0_0_0": np.zeros((1, 2))}}
    with pytest.raises(ValueError, match="mismatched Hamiltonian/H0 shapes"):
        build_residual_hamiltonian_target_blocks(bad_data, good_blocks)


# ===========================================================================
# H8: per-record sample_uid stability (packed (shard_id<<32)|row_id identity).
# ===========================================================================
def test_h8_sample_uid_is_stable_per_record_and_distinct_across_records(tmp_path):
    rec0 = _raw_h_h0_record()
    rec1 = _raw_h_h0_record()
    rec1["case_id"] = "h2-second"
    dataset = _two_record_dataset(tmp_path, [rec0, rec1], name="b-uid-two")

    uid0_first = dataset.get(0)[_keys.SAMPLE_UID_KEY]
    uid0_second = dataset.get(0)[_keys.SAMPLE_UID_KEY]
    uid1 = dataset.get(1)[_keys.SAMPLE_UID_KEY]

    assert torch.equal(uid0_first, uid0_second)
    assert not torch.equal(uid0_first, uid1)
    v0, v1 = int(uid0_first.item()), int(uid1.item())
    assert (v0 >> 32) == (v1 >> 32)
    assert (v1 & 0xFFFFFFFF) == (v0 & 0xFFFFFFFF) + 1
    assert v0 >= 0 and v1 >= 0


# ---------------------------------------------------------------------------
# uureal converter chain: the official convert_feature_lmdb_to_blockwise.py
# product loads through LMDBDataset(require_uureal_block_ode=True).
# ---------------------------------------------------------------------------
def _full_soc_source_record(full_width: int):
    """A full-SOC-width feature record on the same H-C geometry ``_uureal_record`` uses."""
    generator = torch.Generator().manual_seed(0)
    n_nodes, n_edges = 2, 2
    return {
        "atomic_numbers": np.asarray([1, 6], dtype="int64"),
        "pos": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"),
        "cell": (np.eye(3) * 8.0).astype("float32"),
        "pbc": np.asarray([False, False, False]),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype="int64"),
        "edge_cell_shift": np.zeros((2, 3), dtype="float32"),
        "node_features": torch.randn(n_nodes, full_width, generator=generator).numpy().astype("float32"),
        "edge_features": torch.randn(n_edges, full_width, generator=generator).numpy().astype("float32"),
        "node_h0": torch.randn(n_nodes, full_width, generator=generator).numpy().astype("float32"),
        "edge_h0": torch.randn(n_edges, full_width, generator=generator).numpy().astype("float32"),
        "hamiltonian_semantics": "delta (H - H0), uu_real",
        "soc_real_channel_order": np.asarray(["uu_re", "uu_im", "ud_re", "ud_im", "du_re", "du_im", "dd_re", "dd_im"]),
        "full_soc_feature_width": full_width,
        "idx": 0,
        "nf": 1,
    }


def test_official_converter_product_passes_loader_gate(tmp_path):
    """converter -> loader chain: a genuine full-SOC source (feature width keep*8) converts and
    the product loads through LMDBDataset(require_uureal_block_ode=True); a tampered product
    whose recorded source width falls below keep still fails closed."""
    from tools.convert_feature_lmdb_to_blockwise import convert_root

    mapper = _uureal_mapper()
    keep = int(mapper.reduced_matrix_element)
    full_width = keep * 8
    record = _full_soc_source_record(full_width)

    source_root = tmp_path / "source"
    (source_root / "data.0000.lmdb").mkdir(parents=True)
    env = lmdb.open(str(source_root / "data.0000.lmdb"), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(record, protocol=4))
    env.sync()
    env.close()

    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            {
                "common_options": {
                    "basis": {"H": "1s", "C": "1s1p"}, "has_soc": True,
                    "nextham_uureal_mask": True, "full_soc_prediction": False,
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "blockwise"
    summary = convert_root(
        input_root=source_root, output_root=output_root, input_json=input_json, target_mode="already-delta",
    )
    assert summary["entries"] == 1

    env = lmdb.open(str(output_root / "data.0000.lmdb"), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        produced = pickle.loads(txn.get((0).to_bytes(4, "big")))
    env.close()

    info_files = {
        "data.0000.lmdb": {
            "r_max": 2.0, "er_max": None, "oer_max": None, "wave_align": False,
            "train_w_homo_lumo_gap": False, "train_w_eps": False, "train_w_charge": False,
            "train_dip": False, "train_polar": False,
        }
    }
    dataset = LMDBDataset(
        root=str(output_root), info_files=info_files, type_mapper=mapper,
        get_Hamiltonian=True, get_H0=True, prefer_precomputed_h0=True,
        residual_hamiltonian=False, require_uureal_block_ode=True,
    )
    assert dataset.get(0) is not None

    tampered_root = tmp_path / "tampered"
    shutil.copytree(output_root, tampered_root)
    bad = dict(produced)
    bad["blockwise_source_target_feature_width"] = keep - 1
    env = lmdb.open(str(tampered_root / "data.0000.lmdb"), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(bad, protocol=4), overwrite=True)
    env.sync()
    env.close()
    tampered = LMDBDataset(
        root=str(tampered_root), info_files=info_files, type_mapper=mapper,
        get_Hamiltonian=True, get_H0=True, prefer_precomputed_h0=True,
        residual_hamiltonian=False, require_uureal_block_ode=True,
    )
    with pytest.raises(ValueError, match="blockwise_source_target_feature_width"):
        tampered.get(0)
