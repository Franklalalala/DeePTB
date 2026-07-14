from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch

import dptb.data._keys as _keys
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P23_BUNDLE_FINGERPRINT_KEY,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
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
from dptb.data.interfaces.p23_table import P23_VNA_ASSEMBLY_SCHEMA
from dptb.data.transforms_upper_triangle import OrbitalMapper
import tools.materialize_nonsoc_dual_prior_cache as dual_cache
from tools.materialize_nonsoc_dual_prior_cache import (
    _FULL_H_TARGET_FIELDS,
    _assert_dual_record_contract,
    _split_contracts,
    _validate_p23_table_audit,
    augment_p2_record_with_p23,
    transform_vna3c_abacus_to_deeptb,
)


def _graph() -> dict[str, torch.Tensor]:
    return {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1, 1], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros(2, 3),
    }


def _minimal_p2_record(mapper: OrbitalMapper) -> dict:
    basis = mapper_basis_fingerprint(mapper)
    graph = _graph()
    record = {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([True, True, True]),
        _keys.EDGE_INDEX_KEY: graph[_keys.EDGE_INDEX_KEY].numpy(),
        _keys.EDGE_CELL_SHIFT_KEY: graph[_keys.EDGE_CELL_SHIFT_KEY].numpy(),
        _keys.NODE_P2_KEY: np.asarray([[1.0], [2.0]], dtype=np.float32),
        _keys.EDGE_P2_KEY: np.asarray([[0.2], [0.2]], dtype=np.float32),
        _keys.NODE_P2_BLOCKS_KEY: np.asarray([[[1.0]], [[2.0]]], dtype=np.float32),
        _keys.EDGE_P2_BLOCKS_KEY: np.asarray([[[0.2]], [[0.2]]], dtype=np.float32),
        _keys.NODE_P2_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.EDGE_P2_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.asarray(
            [[[1.4]], [[2.4]]], dtype=np.float32
        ),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: np.asarray(
            [[[0.5]], [[0.5]]], dtype=np.float32
        ),
        _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: np.ones((2, 2), dtype=np.int64),
        SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "dedicated_full_h_blocks",
        BASIS_FINGERPRINT_KEY: basis,
        P2_BUNDLE_FINGERPRINT_KEY: hashlib.sha256(b"qualified-p2-bundle").hexdigest(),
    }
    record[EDGE_GRAPH_FINGERPRINT_KEY] = edge_graph_fingerprint(
        record[_keys.ATOMIC_NUMBERS_KEY],
        record[_keys.EDGE_INDEX_KEY],
        record[_keys.EDGE_CELL_SHIFT_KEY],
        basis_fingerprint=basis,
    )
    record[FULL_H_TARGET_FINGERPRINT_KEY] = fingerprint_fields(
        record, _FULL_H_TARGET_FIELDS
    )
    return record


class _FakeStore:
    species = {
        "H": {
            "orbital_shells": [0],
            "orbital_norb": 1,
        }
    }


class _FakeAssembler:
    store = _FakeStore()

    def assemble_graph_addition(self, **kwargs):
        assert kwargs["node_pad_shape"] == (1, 1)
        assert kwargs["edge_pad_shape"] == (1, 1)
        node = np.asarray([[[0.5]], [[-0.25]]], dtype=np.float32)
        edge = np.asarray([[[0.1]], [[0.1]]], dtype=np.float32)
        return node, edge, {
            "schema": P23_VNA_ASSEMBLY_SCHEMA,
            "directed_true_third_centre_terms": 4,
            "unique_factor_queries": 3,
        }


def test_abacus_gauge_addition_is_rotated_then_reverse_is_exact():
    species = {
        "C": {"orbital_shells": [1]},
        "Si": {"orbital_shells": [2]},
    }
    symbols = ["C", "Si"]
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    shifts = np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    node_shapes = np.asarray([[3, 3], [5, 5]], dtype=np.int64)
    edge_shapes = np.asarray([[3, 5], [5, 3]], dtype=np.int64)
    node = np.zeros((2, 5, 5), dtype=np.float64)
    raw_p = np.arange(9, dtype=np.float64).reshape(3, 3)
    raw_d = np.arange(25, dtype=np.float64).reshape(5, 5)
    node[0, :3, :3] = 0.5 * (raw_p + raw_p.T)
    node[1] = 0.5 * (raw_d + raw_d.T)
    edge = np.zeros((2, 5, 5), dtype=np.float64)
    raw_edge = np.arange(15, dtype=np.float64).reshape(3, 5) + 0.25
    edge[0, :3, :5] = raw_edge
    edge[1, :5, :3] = raw_edge.T

    node_out, edge_out, stats = transform_vna3c_abacus_to_deeptb(
        node_addition=node,
        edge_addition=edge,
        symbols=symbols,
        edge_index=edge_index,
        edge_cell_shift=shifts,
        node_shapes=node_shapes,
        edge_shapes=edge_shapes,
        species_contract=species,
    )
    u_p = np.asarray(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    u_d = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    expected_edge = (u_p @ raw_edge @ u_d.T).astype(np.float32)
    np.testing.assert_array_equal(edge_out[0, :3, :5], expected_edge)
    np.testing.assert_array_equal(edge_out[1, :5, :3], expected_edge.T)
    np.testing.assert_array_equal(node_out[0, :3, :3], node_out[0, :3, :3].T)
    np.testing.assert_array_equal(node_out[1], node_out[1].T)
    assert stats["assembler_ao_gauge"] == "ABACUS"
    assert stats["stored_ao_gauge"] == "DeePTB"
    assert stats["max_reverse_error_ev"] == 0.0

    broken_edge = edge.copy()
    broken_edge[1, 0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="exact raw reverse transpose"):
        transform_vna3c_abacus_to_deeptb(
            node_addition=node,
            edge_addition=broken_edge,
            symbols=symbols,
            edge_index=edge_index,
            edge_cell_shift=shifts,
            node_shapes=node_shapes,
            edge_shapes=edge_shapes,
            species_contract=species,
        )


def test_stru_geometry_identity_accepts_periodic_image_only(tmp_path):
    cell_angstrom = np.diag([6.0, 7.0, 8.0])
    stored_positions = np.asarray([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]])
    stru_positions = stored_positions.copy()
    stru_positions[1] += cell_angstrom[0]
    structure = SimpleNamespace(
        atoms=[SimpleNamespace(species="H"), SimpleNamespace(species="H")],
        cart_positions=stru_positions / dual_cache.Bohr2Ang,
        cell_bohr=cell_angstrom / dual_cache.Bohr2Ang,
    )
    gate = SimpleNamespace(
        parse_stru=lambda _: SimpleNamespace(structure=structure)
    )
    case = tmp_path / "periodic_image"
    case.mkdir()
    (case / "STRU").write_text("synthetic\n", encoding="utf-8")
    record = {
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1]),
        _keys.POSITIONS_KEY: stored_positions,
        _keys.CELL_KEY: cell_angstrom,
        _keys.PBC_KEY: np.asarray([True, True, True]),
    }
    _, _, _, provenance = dual_cache._structure_geometry_bohr(
        record=record,
        case_path=case,
        gate1=gate,
        table_species={"H": {}},
    )
    assert provenance["max_position_identity_error_angstrom"] < 1.0e-12
    assert provenance["max_raw_position_image_error_angstrom"] == pytest.approx(6.0)
    assert provenance["position_identity_semantics"] == (
        "minimum_image_on_periodic_axes"
    )

    record[_keys.PBC_KEY] = np.asarray([False, True, True])
    with pytest.raises(ValueError, match="compact geometry is not the STRU geometry"):
        dual_cache._structure_geometry_bohr(
            record=record,
            case_path=case,
            gate1=gate,
            table_species={"H": {}},
        )


def test_stru_geometry_identity_accepts_cell_serialization_roundoff(tmp_path):
    cell_angstrom = np.diag([6.0, 7.0, 8.0])
    positions = np.asarray([[0.5, 0.5, 0.5]])
    structure = SimpleNamespace(
        atoms=[SimpleNamespace(species="H")],
        cart_positions=positions / dual_cache.Bohr2Ang,
        cell_bohr=cell_angstrom / dual_cache.Bohr2Ang,
    )
    gate = SimpleNamespace(parse_stru=lambda _: SimpleNamespace(structure=structure))
    case = tmp_path / "cell_roundoff"
    case.mkdir()
    (case / "STRU").write_text("synthetic\n", encoding="utf-8")
    record = {
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1]),
        _keys.POSITIONS_KEY: positions,
        _keys.CELL_KEY: cell_angstrom + np.diag([7.5e-5, 0.0, 0.0]),
        _keys.PBC_KEY: np.asarray([True, True, True]),
    }
    _, _, _, provenance = dual_cache._structure_geometry_bohr(
        record=record,
        case_path=case,
        gate1=gate,
        table_species={"H": {}},
    )
    assert provenance["max_cell_identity_error_angstrom"] == pytest.approx(7.5e-5)

    record[_keys.CELL_KEY] = cell_angstrom + np.diag([2.0e-4, 0.0, 0.0])
    with pytest.raises(ValueError, match="compact geometry is not the STRU geometry"):
        dual_cache._structure_geometry_bohr(
            record=record,
            case_path=case,
            gate1=gate,
            table_species={"H": {}},
        )


def test_resume_identity_allows_only_declared_geometry_tolerance_upgrade():
    old_script = next(iter(dual_cache.LEGACY_GEOMETRY_IDENTITY_UPGRADES))
    actual = {
        "schema": dual_cache.IDENTITY_SCHEMA,
        "materializer_script_sha256": old_script,
        "geometry_tolerance_angstrom": 5.0e-5,
        "unchanged": "bound-inputs",
    }
    actual["identity_sha256"] = dual_cache._json_sha256(actual)
    expected = {
        **actual,
        "materializer_script_sha256": hashlib.sha256(b"new-script").hexdigest(),
        "geometry_tolerance_angstrom": 1.0e-4,
    }
    expected.pop("identity_sha256")
    expected["identity_sha256"] = dual_cache._json_sha256(expected)

    with pytest.raises(ValueError, match="identity changed"):
        dual_cache._validate_identity(actual, expected)
    migration = dual_cache._validate_identity(
        actual, expected, allow_geometry_tolerance_upgrade=True
    )
    assert migration["kind"] == "geometry_serialization_tolerance_upgrade"

    changed = dict(expected)
    changed["unchanged"] = "different-input"
    changed.pop("identity_sha256")
    changed["identity_sha256"] = dual_cache._json_sha256(changed)
    with pytest.raises(ValueError, match="identity changed"):
        dual_cache._validate_identity(
            actual, changed, allow_geometry_tolerance_upgrade=True
        )


def test_single_record_dual_materialization_preserves_p2_and_full_h():
    mapper = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    record = _minimal_p2_record(mapper)
    original_p2_node = record[_keys.NODE_P2_BLOCKS_KEY].copy()
    original_p2_edge = record[_keys.EDGE_P2_BLOCKS_KEY].copy()
    original_target = record[_keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY].copy()
    p23_source = hashlib.sha256(b"p23-table-manifest").hexdigest()

    dual, stats = augment_p2_record_with_p23(
        record=record,
        data=_graph(),
        idp=mapper,
        assembler=_FakeAssembler(),
        symbols=["H", "H"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cell_bohr=np.eye(3) * 16.0,
        p23_source_fingerprint=p23_source,
        geometry_provenance={"compact_length_unit": "angstrom"},
    )

    assert dual[SAMPLE_SCHEMA_KEY] == DUAL_PRIOR_SAMPLE_SCHEMA
    np.testing.assert_array_equal(dual[_keys.NODE_P2_BLOCKS_KEY], original_p2_node)
    np.testing.assert_array_equal(dual[_keys.EDGE_P2_BLOCKS_KEY], original_p2_edge)
    np.testing.assert_array_equal(
        dual[_keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY], original_target
    )
    np.testing.assert_allclose(
        dual[_keys.NODE_P23_BLOCKS_KEY],
        np.asarray([[[1.5]], [[1.75]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        dual[_keys.EDGE_P23_BLOCKS_KEY],
        np.asarray([[[0.3]], [[0.3]]], dtype=np.float32),
    )
    assert dual[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY] == dual[
        P2_BUNDLE_FINGERPRINT_KEY
    ]
    assert dual[P23_BUNDLE_FINGERPRINT_KEY] != dual[P2_BUNDLE_FINGERPRINT_KEY]
    assert stats["ao_gauge_conversion"]["gauge_transform"] == (
        "OrbAbacus2DeepTB.transform"
    )
    assert dual[ROW_ALIGNED_DATA_FINGERPRINT_KEY] == (
        fingerprint_present_row_aligned_fields(dual)
    )
    assert dual[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] == fingerprint_text_fields(
        dual,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ),
    )

    tampered = dict(dual)
    tampered[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY] = "0" * 64
    with pytest.raises(ValueError, match="parent P2 bundle"):
        _assert_dual_record_contract(
            tampered,
            _graph(),
            mapper,
            expected_p23_source=p23_source,
        )


def test_aggregate_physical_cases_must_equal_numeric_shard_path_order(tmp_path):
    dataset_root = tmp_path / "raw"
    dataset_root.mkdir()
    cases = []
    for name in ("case_a", "case_b"):
        case = dataset_root / name
        case.mkdir()
        (case / "STRU").write_text("synthetic\n", encoding="utf-8")
        cases.append(case)
    (dataset_root / "ordered_paths.txt").write_text(
        "".join(f"{case}\n" for case in cases), encoding="utf-8"
    )
    cache_root = tmp_path / "full_h"
    split = cache_root / "train"
    split.mkdir(parents=True)
    for shard_index, case in enumerate(cases):
        lmdb_path = split / f"data.{shard_index:04d}.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), b"synthetic")
        env.close()
        (split / f"data.{shard_index:04d}.paths.txt").write_text(
            f"{case}\n", encoding="utf-8"
        )
    manifest = {
        "splits": {
            "train": {
                "physical_cases": ["case_b", "case_a"],
                "records": 2,
                "logical_cases": 2,
            }
        }
    }
    with pytest.raises(ValueError, match="physical_cases order"):
        _split_contracts(
            p2_cache_root=cache_root,
            p2_manifest=manifest,
            dataset_root=dataset_root,
            selected_splits=["train"],
            selected_cases=[],
        )


def test_p23_table_audit_contract_is_exact_and_hash_bound(tmp_path):
    manifest_sha = hashlib.sha256(b"table-manifest").hexdigest()
    audit = {
        "schema": "deeptb.p23_factorized_vna_table_audit/v1",
        "status": "pass",
        "manifest_sha256": manifest_sha,
        "verified_shards": 950,
        "integrity_eligible": True,
        "experimental_training_eligible": True,
        "production_eligible": True,
        "physical_oracle": {"status": "pass"},
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    assert _validate_p23_table_audit(
        path, expected_table_manifest_sha256=manifest_sha
    )["status"] == "pass"

    audit["physical_oracle"] = {"status": "fail"}
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="physical_oracle.status"):
        _validate_p23_table_audit(
            path, expected_table_manifest_sha256=manifest_sha
        )


def test_strict_output_audit_uses_fresh_dataset_per_prior_kind(
    tmp_path, monkeypatch
):
    built = []

    class FakeDataset:
        def __init__(self):
            self.kind = None
            self.closed = False
            self._validated_record_contracts = {}

        def __len__(self):
            return 1

        def get(self, index):
            assert index == 0
            if self._validated_record_contracts:
                return {"masked_by_prior_cache": self.kind}
            if self.kind == "p23":
                raise ValueError("tampered P23 fingerprint")
            self._validated_record_contracts[("record", 0)] = (None, None)
            return {"ok": self.kind}

    def fake_build_context_dataset(input_json, split_root):
        dataset = FakeDataset()
        built.append(dataset)
        return dataset, None

    def fake_configure_strict_dataset(dataset, *, kind, source):
        dataset.kind = kind
        dataset.source = source

    def fake_close_dataset(dataset):
        dataset.closed = True

    monkeypatch.setattr(
        dual_cache, "_build_context_dataset", fake_build_context_dataset
    )
    monkeypatch.setattr(
        dual_cache, "_configure_strict_dataset", fake_configure_strict_dataset
    )
    monkeypatch.setattr(dual_cache, "_close_dataset", fake_close_dataset)
    monkeypatch.setattr(dual_cache, "_heartbeat", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="tampered P23 fingerprint"):
        dual_cache._strict_audit_output(
            input_json=tmp_path / "input.json",
            full_h_root=tmp_path / "full_h",
            contracts={"train": {"entries": 1}},
            p2_source="2" * 64,
            p23_source="3" * 64,
            work_root=tmp_path,
        )

    assert [dataset.kind for dataset in built] == ["p2", "p23"]
    assert all(dataset.closed for dataset in built)
