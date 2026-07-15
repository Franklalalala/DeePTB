from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch

import dptb.data._keys as _keys
from dptb.data.interfaces.abacus import _abacus_parse, recursive_parse
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
from dptb.data.interfaces.p2_table import P2_COMPONENT_NUMERICAL_GATE_SCHEMA
from dptb.data.interfaces.p23_table import P23_VNA_ASSEMBLY_SCHEMA
from dptb.data.transforms_upper_triangle import OrbitalMapper
import tools.benchmark_dual_prior_loader as benchmark_module
import tools.materialize_nonsoc_dual_prior_cache as dual_cache
from tools.build_nonsoc_dual_prior_ablation_configs import (
    DEFAULT_REFERENCE,
    build_configs,
)
from tools.materialize_nonsoc_dual_prior_cache import (
    _FULL_H_TARGET_FIELDS,
    _assert_dual_record_contract,
    _split_contracts,
    _validate_p23_table_audit,
    augment_p2_record_with_p23,
    transform_vna3c_abacus_to_deeptb,
)
from tools.materialize_nonsoc_p2_cache import (
    RAW_CASE_SOURCE_FINGERPRINT_KEY,
    RY_TO_EV,
    SCHEMA,
    _case_source_fingerprint_map,
    _dataset_source_identity,
    _require_qualified_table_manifest,
    _raw_staging_identity,
    _validate_raw_record_source,
    _validate_raw_staging_identity,
    _write_raw_staging_identity,
    _guard_output,
    _required_block_keys_from_graph,
    complete_sparse_zero_blocks,
    dense_p2_to_deeptb_blocks,
    parse_basis_lines,
    project_non_soc_blocks_hermitian,
)


# ===========================================================================
# Single-prior (P2) cache materializer
# ===========================================================================


def test_abacus_parser_exposes_separate_h0_switches():
    assert "parse_H0" in inspect.signature(recursive_parse).parameters
    assert "get_H0" in inspect.signature(_abacus_parse).parameters


def test_parse_basis_lines_expands_radial_shells():
    assert parse_basis_lines(b"4s2p2d1f\n2s1p\n", expected_atoms=2) == [
        [0, 0, 0, 0, 1, 1, 2, 2, 3],
        [0, 0, 1],
    ]


@pytest.mark.parametrize("basis", ["1q", "1s trailing", ""])
def test_parse_basis_lines_rejects_unsupported_or_missing_basis(basis):
    with pytest.raises(ValueError):
        parse_basis_lines(basis, expected_atoms=1)


def test_dense_p2_cache_converts_units_and_preserves_r_keys():
    r_keys = np.asarray([[0, 0, 0], [1, -2, 3]], dtype=np.int64)
    p2 = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.complex128,
    )
    blocks = dense_p2_to_deeptb_blocks(
        r_keys=r_keys,
        p2=p2,
        basis_lines="1s\n1s\n",
        atom_count=2,
    )

    assert set(blocks) == {
        "0_0_0_0_0",
        "0_1_0_0_0",
        "1_0_0_0_0",
        "1_1_0_0_0",
        "0_0_1_-2_3",
        "0_1_1_-2_3",
        "1_0_1_-2_3",
        "1_1_1_-2_3",
    }
    assert blocks["0_1_0_0_0"].dtype == np.float32
    np.testing.assert_allclose(blocks["0_1_0_0_0"], [[2.0 * RY_TO_EV]])
    np.testing.assert_allclose(blocks["1_0_1_-2_3"], [[7.0 * RY_TO_EV]])


def test_dense_p2_cache_rejects_non_soc_imaginary_content():
    with pytest.raises(ValueError, match="imaginary"):
        dense_p2_to_deeptb_blocks(
            r_keys=np.zeros((1, 3), dtype=np.int64),
            p2=np.asarray([[[1.0 + 1.0e-5j]]]),
            basis_lines="1s",
            atom_count=1,
        )


@pytest.mark.parametrize(
    "r_keys,match",
    [
        (np.asarray([[0.5, 0.0, 0.0]]), "integer lattice"),
        (np.asarray([[0, 0, 0], [0, 0, 0]]), "duplicate"),
    ],
)
def test_dense_p2_cache_rejects_invalid_lattice_keys(r_keys, match):
    with pytest.raises(ValueError, match=match):
        dense_p2_to_deeptb_blocks(
            r_keys=r_keys,
            p2=np.zeros((len(r_keys), 1, 1)),
            basis_lines="1s",
            atom_count=1,
        )


def test_guard_output_rejects_overlapping_roots(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        _guard_output(dataset, dataset / "cache", overwrite=False)
    with pytest.raises(ValueError, match="disjoint"):
        _guard_output(dataset, tmp_path, overwrite=False)


def test_guard_output_resume_requires_existing_disjoint_root(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    work = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="existing work root"):
        _guard_output(
            dataset,
            work,
            overwrite=False,
            resume_raw_staging=True,
        )
    work.mkdir()
    _guard_output(
        dataset,
        work,
        overwrite=False,
        resume_raw_staging=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _guard_output(
            dataset,
            work,
            overwrite=True,
            resume_raw_staging=True,
        )


def _write_case_sources(dataset, case_name="case-a"):
    case = dataset / "cases" / case_name
    output = case / "OUT.ABACUS"
    output.mkdir(parents=True)
    (output / "running_scf.log").write_text("structure-v1\n", encoding="utf-8")
    (output / "data-HR-sparse_SPIN0.csr").write_text("full-h-v1\n", encoding="utf-8")
    (output / "data-HR0_SPIN0.csr").write_text("h0-v1\n", encoding="utf-8")
    (case / "STRU").write_text("stru-v1\n", encoding="utf-8")
    dataset.mkdir(exist_ok=True)
    (dataset / "ordered_paths.txt").write_text(f"{case.resolve()}\n", encoding="utf-8")
    return case


def _identity_inputs(tmp_path):
    input_json = tmp_path / "input.json"
    input_json.write_text('{"data_options": {}}\n', encoding="utf-8")
    gate1_script = tmp_path / "gate1.py"
    gate1_script.write_text("# synthetic gate1\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    case = _write_case_sources(dataset)
    dataset_identity = _dataset_source_identity(
        dataset,
        [case],
        require_stru=True,
    )
    return input_json, gate1_script, dataset_identity, case


def test_raw_staging_identity_resume_is_reusable_and_fails_closed(tmp_path):
    # Collapses the three raw-staging resume-identity guards (reusable match,
    # missing/mismatch rejection, tampered-record fail-closed) into one owner
    # while preserving every scenario.
    input_json, gate1_script, dataset_identity, _case = _identity_inputs(tmp_path)

    def _identity(p2_source_fingerprint="a" * 64):
        return _raw_staging_identity(
            input_json=input_json,
            gate1_script=gate1_script,
            p2_source_fingerprint=p2_source_fingerprint,
            p2_source_kind="radial_table",
            dataset_source_identity=dataset_identity,
        )

    expected = _identity()

    # A matching resume identity is reusable.
    reusable = tmp_path / "work_reusable"
    _write_raw_staging_identity(reusable, expected)
    (reusable / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA, "raw_staging_identity": expected}),
        encoding="utf-8",
    )
    _validate_raw_staging_identity(reusable, expected)

    # A missing identity, then a mismatched identity, both fail closed.
    missing = tmp_path / "work_missing"
    missing.mkdir()
    (missing / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity is missing"):
        _validate_raw_staging_identity(missing, expected)

    _write_raw_staging_identity(missing, expected)
    mismatched = _identity(p2_source_fingerprint="b" * 64)
    (missing / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA, "raw_staging_identity": mismatched}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_raw_staging_identity(missing, expected)

    # A tampered symmetric P2 row without a written identity still fails closed
    # on the missing identity before any data is trusted.
    tampered = tmp_path / "work_tampered"
    raw_split = tampered / "raw_staging" / "train" / "data.0000.lmdb"
    raw_split.parent.mkdir(parents=True)
    env = lmdb.open(str(raw_split), map_size=1 << 20, subdir=True, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            txn.put(
                (0).to_bytes(length=4, byteorder="big"),
                pickle.dumps(
                    {
                        "case_id": "tampered",
                        "hamiltonian_schema": "deeptb.p2_training_sample/v2",
                        "p2_source_fingerprint": expected["p2_source_fingerprint"],
                        "hamiltonian_p2": {
                            "0_0_0_0_0": np.asarray([[123.0]], dtype=np.float32)
                        },
                        "p2_hermitian_projection": {
                            "max_pre_projection_mismatch": 0.0,
                        },
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )
    finally:
        env.close()
    (tampered / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity is missing"):
        _validate_raw_staging_identity(tampered, expected)


@pytest.mark.parametrize(
    "relative_path",
    [
        "OUT.ABACUS/running_scf.log",
        "OUT.ABACUS/data-HR-sparse_SPIN0.csr",
        "OUT.ABACUS/data-HR0_SPIN0.csr",
        "STRU",
    ],
)
def test_dataset_source_identity_binds_each_authoritative_case_file(
    tmp_path,
    relative_path,
):
    dataset = tmp_path / "dataset"
    case = _write_case_sources(dataset)
    before = _dataset_source_identity(dataset, [case], require_stru=True)

    (case / relative_path).write_text("changed source\n", encoding="utf-8")
    after = _dataset_source_identity(dataset, [case], require_stru=True)

    assert before["identity_sha256"] != after["identity_sha256"]
    assert (
        before["selected_cases"][0]["source_fingerprint"]
        != after["selected_cases"][0]["source_fingerprint"]
    )


def test_raw_record_source_rejects_same_basename_other_root_and_source_update(tmp_path):
    dataset_a = tmp_path / "a" / "dataset"
    dataset_b = tmp_path / "b" / "dataset"
    case_a = _write_case_sources(dataset_a, case_name="same")
    case_b = _write_case_sources(dataset_b, case_name="same")
    identity_a = _dataset_source_identity(dataset_a, [case_a], require_stru=True)
    identity_b = _dataset_source_identity(dataset_b, [case_b], require_stru=True)
    fingerprints_a = _case_source_fingerprint_map(identity_a)
    fingerprints_b = _case_source_fingerprint_map(identity_b)
    fingerprint_a = fingerprints_a[str(case_a.resolve())]
    record = {
        "source": str(case_a.resolve()),
        RAW_CASE_SOURCE_FINGERPRINT_KEY: fingerprint_a,
    }

    with pytest.raises(ValueError, match="source path"):
        _validate_raw_record_source(
            record,
            case_b,
            case_source_fingerprints=fingerprints_b,
            label="train[0]",
        )

    record["source"] = str(case_b.resolve())
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        _validate_raw_record_source(
            record,
            case_b,
            case_source_fingerprints=fingerprints_b,
            label="train[0]",
        )


@pytest.mark.parametrize("gate", [None, {"status": "fail"}])
def test_materializer_rejects_table_without_persisted_pass_gate(tmp_path, gate):
    manifest = {"complete": True, "component_numerical_gate": gate}
    with pytest.raises(ValueError, match="persisted component numerical gate"):
        _require_qualified_table_manifest(
            manifest,
            manifest_path=tmp_path / "manifest.json",
        )

    _require_qualified_table_manifest(
        {
            "complete": True,
            "component_numerical_gate": {
                "schema": P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
                "status": "pass",
            },
        },
        manifest_path=tmp_path / "manifest.json",
    )


def test_table_cache_assembles_only_onsite_and_canonical_graph_blocks():
    keys = _required_block_keys_from_graph(
        np.asarray([1, 8]),
        np.asarray([[0, 0, 1, 1], [1, 1, 0, 0]]),
        np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0]],
            dtype=np.float32,
        ),
    )
    assert keys == [
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 1, 1, 0, 0),
        (1, 0, -1, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 0, 0, 0),
    ]


def test_abacus_sparse_zero_completion_is_explicit_and_shape_safe():
    blocks = {
        "0_0_0_0_0": np.ones((1, 1), dtype=np.float32),
        # The reverse is sufficient for the requested 0->1 edge.
        "1_0_0_0_0": np.ones((4, 1), dtype=np.float32),
        "1_1_0_0_0": np.eye(4, dtype=np.float32),
    }
    completed, filled = complete_sparse_zero_blocks(
        blocks,
        required_keys=[
            (0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (0, 1, 1, 0, 0),
        ],
        basis_lines="1s\n1s1p\n",
        atom_count=2,
        label="hamiltonian",
    )
    assert filled == [(0, 1, 1, 0, 0)]
    assert completed["0_1_1_0_0"].shape == (1, 4)
    assert completed["0_1_1_0_0"].dtype == np.float32
    assert not np.any(completed["0_1_1_0_0"])

    broken = dict(blocks)
    broken["0_0_0_0_0"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        complete_sparse_zero_blocks(
            broken,
            required_keys=[(0, 0, 0, 0, 0)],
            basis_lines="1s\n1s1p\n",
            atom_count=2,
            label="hamiltonian",
        )


def test_p2_hermitian_projection_reverse_exact_and_fails_closed():
    # Collapses the three hermitian-projection guards (makes reverse exact,
    # rejects a missing reverse graph block, rejects a large pre-projection
    # mismatch without mutating inputs) into one owner while preserving every
    # scenario.
    blocks = {
        "0_0_0_0_0": np.asarray([[1.0, 2.0], [2.00002, 3.0]], dtype=np.float32),
        "0_1_1_0_0": np.asarray([[4.0, 5.0]], dtype=np.float32),
        "1_0_-1_0_0": np.asarray([[4.00002], [4.99998]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[6.0]], dtype=np.float32),
    }
    projected, stats = project_non_soc_blocks_hermitian(
        blocks,
        required_keys=[
            (0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0),
            (1, 0, -1, 0, 0),
            (1, 1, 0, 0, 0),
        ],
    )
    np.testing.assert_array_equal(
        projected["0_0_0_0_0"], projected["0_0_0_0_0"].T
    )
    np.testing.assert_array_equal(
        projected["0_1_1_0_0"], projected["1_0_-1_0_0"].T
    )
    assert stats["max_pre_projection_mismatch"] > 1.0e-5
    assert stats["max_projection_correction"] > 0.0

    with pytest.raises(ValueError, match="no reverse graph block"):
        project_non_soc_blocks_hermitian(
            {"0_1_1_0_0": np.ones((1, 1), dtype=np.float32)},
            required_keys=[(0, 1, 1, 0, 0)],
        )

    large_mismatch = {
        "0_1_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[1.1]], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="exceeds tolerance") as error:
        project_non_soc_blocks_hermitian(
            large_mismatch,
            required_keys=[(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)],
            mismatch_tolerance=1.0e-3,
        )
    assert '"above_tolerance_count": 1' in str(error.value)
    # Input evidence is not modified on failure.
    np.testing.assert_array_equal(
        large_mismatch["0_1_0_0_0"], np.asarray([[1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        large_mismatch["1_0_0_0_0"], np.asarray([[1.1]], dtype=np.float32)
    )


# ===========================================================================
# Dual-prior (P2 + P23) cache materializer
# ===========================================================================


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


# ===========================================================================
# Dual-prior loader benchmark
# ===========================================================================


class _FakeValidatedDataset:
    def __init__(self, records: int = 3):
        self._records = records
        self._validated_record_contracts = {}
        self._lmdb_env_cache = {}

    def __len__(self):
        return self._records

    def get(self, index: int):
        self._last_lmdb_pickle_bytes = (1 << 20) + int(index)
        self._last_lmdb_record_identity = ("/fake/data.lmdb", int(index))
        key = (("/fake/data.lmdb",), int(index))
        self._validated_record_contracts[key] = (None, None)
        return {"index": index}


def test_benchmarks_four_unique_loader_contracts_and_gate_cache(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=tmp_path / "dual",
        output_dir=config_dir,
        expected_p2_source_fingerprint="2" * 64,
        expected_p23_source_fingerprint="3" * 64,
    )
    builds = []

    def fake_build_dataset(**kwargs):
        builds.append(kwargs)
        return _FakeValidatedDataset()

    monkeypatch.setattr(benchmark_module, "build_dataset", fake_build_dataset)
    report = benchmark_module.benchmark_matrix(
        config_dir=config_dir,
        split="train",
        warm_repeats=2,
        max_records=2,
    )

    assert report["schema"] == benchmark_module.SCHEMA
    assert set(report["arms"]) == {
        "p2_residual",
        "p2_direct",
        "p23_residual",
        "p23_direct",
    }
    # Two fresh datasets per arm: probe/warm and sequential first pass.
    assert len(builds) == 8
    for name, result in report["arms"].items():
        cold = result["cold_first_get"]
        warm = result["warm_same_record"]
        sequential = result["sequential_first_pass"]
        assert cold["first_gate_executed"] is True
        assert cold["validation_cache_entries_before"] == 0
        assert cold["validation_cache_entries_after"] == 1
        assert cold["pickle_bytes"] == 1 << 20
        assert warm["records"] == 2
        assert warm["validation_cache_entries_before"] == 1
        assert warm["validation_cache_entries_after"] == 1
        assert warm["pickle_bytes"] == 2 * (1 << 20)
        assert sequential["records"] == 2
        assert sequential["pickle_bytes"] == 2 * (1 << 20) + 1
        assert sequential["validation_cache_entries_before"] == 0
        assert sequential["validation_cache_entries_after"] == 2
        assert sequential["complete_split"] is False
        assert sequential["records_per_second"] > 0.0
        assert sequential["mib_per_second"] > 0.0
        assert result["require_prior_ao_blocks"] is name.endswith("residual")
