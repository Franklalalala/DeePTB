from __future__ import annotations

import pickle

import lmdb
import numpy as np
import pytest
import torch

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    LMDBDataset,
    assert_absolute_full_h_target_contract,
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


def _raw_h_h0_record() -> dict:
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
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
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


@pytest.mark.parametrize("schema", (P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA))
def test_full_h_h0_authority_accepts_declared_raw_or_dedicated(schema: str):
    raw = _raw_h_h0_record()
    raw[SAMPLE_SCHEMA_KEY] = schema
    raw[PHYSICAL_H0_SOURCE_KEY] = RAW_PHYSICAL_H0_SOURCE
    assert_absolute_full_h_target_contract(raw, require_h0=True)
    dedicated = _dedicated_h0_record(schema)
    assert_absolute_full_h_target_contract(
        dedicated,
        require_h0=True,
        expected_physical_h0_source_fingerprint=dedicated[
            PHYSICAL_H0_SOURCE_FINGERPRINT_KEY
        ],
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
def test_dedicated_full_h_h0_authority_is_fail_closed(
    schema: str, mutation: str, match: str
):
    record = _dedicated_h0_record(schema)
    expected = record[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY]
    if mutation == "undeclared":
        record.pop(PHYSICAL_H0_SOURCE_KEY)
    elif mutation == "partial":
        record.pop(_keys.EDGE_H0_BLOCKS_KEY)
    elif mutation == "dual_authority":
        record["hamiltonian_0"] = {"0_0_0_0_0": np.zeros((1, 1))}
    else:
        if mutation == "unbound":
            record.pop(ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY)
        elif mutation == "missing_record_fingerprint":
            record.pop(PHYSICAL_H0_SOURCE_FINGERPRINT_KEY)
        elif mutation == "missing_external_fingerprint":
            expected = None
        else:
            expected = "f" * 64

    with pytest.raises(ValueError, match=match):
        assert_absolute_full_h_target_contract(
            record,
            require_h0=True,
            expected_physical_h0_source_fingerprint=expected,
        )


@pytest.mark.parametrize("schema", (P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA))
def test_full_h_h0_authority_is_opt_in_for_legacy_routes(schema: str):
    record = _dedicated_h0_record(schema)
    record.pop(PHYSICAL_H0_SOURCE_KEY)
    assert_absolute_full_h_target_contract(record, require_h0=False)


def _dataset_from_record(
    tmp_path,
    record: dict,
    *,
    name: str,
    residual_hamiltonian: bool,
    require_full_h_target: bool,
    dataset_options: dict | None = None,
) -> LMDBDataset:
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()

    options = dict(dataset_options or {})
    dataset = DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix=name,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_H0=True,
        residual_hamiltonian=residual_hamiltonian,
        require_full_h_target=require_full_h_target,
        require_residual_h_target=residual_hamiltonian,
        **options,
    )
    assert isinstance(dataset, LMDBDataset)
    return dataset


def test_generic_raw_full_h_schema_rejects_unversioned_dataset_prior(tmp_path):
    record = _raw_h_h0_record()
    record["hamiltonian_p2"] = {
        key: np.asarray(value) * 0.5
        for key, value in record["hamiltonian"].items()
    }
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name="raw-full-h-unversioned-prior",
        residual_hamiltonian=False,
        require_full_h_target=True,
        dataset_options={
            "get_prior": True,
            "expected_prior_source_fingerprint": "f" * 64,
        },
    )

    with pytest.raises(ValueError, match="versioned P2/dual-prior sample schema"):
        dataset.get(0)


def test_generic_raw_full_h_schema_materializes_dedicated_target_blocks(tmp_path):
    dataset = _dataset_from_record(
        tmp_path,
        _raw_h_h0_record(),
        name="block-ode-full-h",
        residual_hamiltonian=False,
        require_full_h_target=True,
    )

    sample = dataset.get(0)

    torch.testing.assert_close(
        sample[_keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(),
        torch.tensor([10.0, 12.0]),
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY].flatten(),
        torch.tensor([3.0, 3.0]),
    )
    assert torch.equal(
        sample[_keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
        torch.ones((2, 2), dtype=torch.long),
    )
    assert torch.equal(
        sample[_keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
        torch.ones((2, 2), dtype=torch.long),
    )
    assert _keys.NODE_P2_KEY not in sample
    assert _keys.EDGE_P2_KEY not in sample
    torch.testing.assert_close(
        sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5])
    )
    assert torch.equal(
        sample[_keys.NODE_H0_BLOCK_SHAPE_KEY],
        torch.ones((2, 2), dtype=torch.long),
    )
    assert torch.equal(
        sample[_keys.EDGE_H0_BLOCK_SHAPE_KEY],
        torch.ones((2, 2), dtype=torch.long),
    )


@pytest.mark.parametrize(
    ("missing_key", "match"),
    (
        (SAMPLE_SCHEMA_KEY, "explicit sample schema"),
        (TARGET_SEMANTICS_KEY, "requires hamiltonian_target_semantics"),
        (TARGET_SOURCE_KEY, "hamiltonian_target_source"),
    ),
)
def test_generic_raw_full_h_metadata_is_fail_closed(
    tmp_path, missing_key: str, match: str
):
    record = _raw_h_h0_record()
    record.pop(missing_key)
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name=f"missing-{missing_key.replace('_', '-')}",
        residual_hamiltonian=False,
        require_full_h_target=True,
    )

    with pytest.raises(ValueError, match=match):
        dataset.get(0)


@pytest.mark.parametrize(
    ("missing_key", "match"),
    (
        ("hamiltonian", "raw.*hamiltonian"),
        ("hamiltonian_0", "physical-H0 block dictionary"),
    ),
)
def test_generic_raw_full_h_contract_requires_both_raw_h_and_physical_h0(
    tmp_path, missing_key: str, match: str
):
    record = _raw_h_h0_record()
    record.pop(missing_key)
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name=f"missing-raw-{missing_key.replace('_', '-')}",
        residual_hamiltonian=False,
        require_full_h_target=True,
    )

    with pytest.raises(ValueError, match=match):
        dataset.get(0)


def test_generic_raw_full_h_contract_rejects_prepacked_target_override(tmp_path):
    record = _raw_h_h0_record()
    record.update(
        {
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros(
                (2, 1, 1), dtype=np.float32
            ),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.zeros(
                (2, 1, 1), dtype=np.float32
            ),
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones(
                (2, 2), dtype=np.int64
            ),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones(
                (2, 2), dtype=np.int64
            ),
        }
    )
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name="prepacked-target-override",
        residual_hamiltonian=False,
        require_full_h_target=True,
    )

    with pytest.raises(ValueError, match="independently prepacked"):
        dataset.get(0)


def _prepacked_h0_fields(*, complete: bool) -> dict:
    fields = {
        _keys.NODE_H0_BLOCKS_KEY: np.zeros((2, 1, 1), dtype=np.float32),
    }
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
@pytest.mark.parametrize(
    ("residual_hamiltonian", "require_full_h_target"),
    ((False, True), (True, False)),
)
def test_raw_authority_rejects_prepacked_h0_block_override(
    tmp_path,
    complete: bool,
    residual_hamiltonian: bool,
    require_full_h_target: bool,
):
    record = _raw_h_h0_record()
    if residual_hamiltonian:
        record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    record.update(_prepacked_h0_fields(complete=complete))
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name=(
            f"prepacked-h0-{'complete' if complete else 'partial'}-"
            f"{'residual' if residual_hamiltonian else 'absolute'}"
        ),
        residual_hamiltonian=residual_hamiltonian,
        require_full_h_target=require_full_h_target,
    )

    with pytest.raises(ValueError, match="prepacked H0 block fields"):
        dataset.get(0)


def test_precomputed_h0_features_do_not_override_raw_h0_blocks(tmp_path):
    record = _raw_h_h0_record()
    record[_keys.NODE_H0_KEY] = np.full((2, 1), 101.0, dtype=np.float32)
    record[_keys.EDGE_H0_KEY] = np.full((2, 1), 103.0, dtype=np.float32)
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name="precomputed-h0-features",
        residual_hamiltonian=False,
        require_full_h_target=True,
    )

    sample = dataset.get(0)
    torch.testing.assert_close(
        sample[_keys.NODE_H0_KEY], torch.full((2, 1), 101.0)
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_KEY], torch.full((2, 1), 103.0)
    )
    torch.testing.assert_close(
        sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5])
    )


def test_residual_loader_materializes_delta_and_h0_block_side_channels(tmp_path):
    record = _raw_h_h0_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name="block-ode-residual",
        residual_hamiltonian=True,
        require_full_h_target=False,
    )

    sample = dataset.get(0)

    # Keep the legacy product-coordinate main feature semantics: the main
    # feature is absolute H, not dH.  Block-ODE flow must ignore this channel
    # and use the authoritative H0 and dH AO block side channels below.
    torch.testing.assert_close(
        sample[_keys.NODE_FEATURES_KEY].flatten(), torch.tensor([10.0, 12.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_FEATURES_KEY].flatten(), torch.tensor([3.0, 3.0])
    )
    torch.testing.assert_close(
        sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(),
        torch.tensor([1.0, 1.0]),
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(),
        torch.tensor([0.5, 0.5]),
    )
    torch.testing.assert_close(
        sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5])
    )


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
def test_residual_loader_metadata_is_fail_closed(
    tmp_path, bad_key: str, bad_value: str | None, match: str
):
    record = _raw_h_h0_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    if bad_value is None:
        record.pop(bad_key)
    else:
        record[bad_key] = bad_value
    dataset = _dataset_from_record(
        tmp_path,
        record,
        name=f"residual-bad-{bad_key}-{bad_value}",
        residual_hamiltonian=True,
        require_full_h_target=False,
    )

    with pytest.raises(ValueError, match=match):
        dataset.get(0)
