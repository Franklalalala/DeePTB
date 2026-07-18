from __future__ import annotations

import inspect
import json
from copy import deepcopy

import numpy as np
import pytest

import dptb.data.AtomicDataDict as AtomicDataDict
from dptb.data.interfaces.abacus import _abacus_parse, recursive_parse
from dptb.data.interfaces.p2_contract import (
    DEDICATED_PHYSICAL_H0_SOURCE,
    PHYSICAL_H0_SOURCE_KEY,
    P2_SOURCE_FINGERPRINT_KEY,
)
from dptb.utils.argcheck import normalize
from tools.build_nonsoc_p2_ablation_configs import DEFAULT_REFERENCE, build_configs
from tools.materialize_nonsoc_p2_cache import (
    RY_TO_EV,
    SCHEMA,
    _raw_staging_identity,
    _validate_raw_staging_identity,
    _write_raw_staging_identity,
    _guard_output,
    _base_record,
    _required_block_keys_from_graph,
    complete_sparse_zero_blocks,
    dense_p2_to_deeptb_blocks,
    parse_basis_lines,
    project_non_soc_blocks_hermitian,
)


def test_abacus_parser_exposes_separate_h0_switches():
    assert "parse_H0" in inspect.signature(recursive_parse).parameters
    assert "get_H0" in inspect.signature(_abacus_parse).parameters


def test_compact_base_declares_dedicated_physical_h0_authority():
    raw = {
        AtomicDataDict.CELL_KEY: np.eye(3),
        AtomicDataDict.POSITIONS_KEY: np.zeros((1, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: np.asarray([1]),
        AtomicDataDict.PBC_KEY: np.asarray([False, False, False]),
        "case_id": "one",
        "source": "synthetic",
        "p2_cache_source": {},
        P2_SOURCE_FINGERPRINT_KEY: "1" * 64,
    }
    data = {
        AtomicDataDict.EDGE_INDEX_KEY: np.empty((2, 0), dtype=np.int64),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: np.empty((0, 3)),
    }
    compact = _base_record(raw, data, 0, basis_fingerprint="2" * 64)
    assert compact[PHYSICAL_H0_SOURCE_KEY] == DEDICATED_PHYSICAL_H0_SOURCE


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


def _identity_inputs(tmp_path):
    input_json = tmp_path / "input.json"
    input_json.write_text('{"data_options": {}}\n', encoding="utf-8")
    gate1_script = tmp_path / "gate1.py"
    gate1_script.write_text("# synthetic gate1\n", encoding="utf-8")
    return input_json, gate1_script


def test_raw_staging_identity_matching_resume_is_reusable(tmp_path):
    input_json, gate1_script = _identity_inputs(tmp_path)
    work = tmp_path / "work"
    expected = _raw_staging_identity(
        input_json=input_json,
        gate1_script=gate1_script,
        p2_source_fingerprint="a" * 64,
        p2_source_kind="radial_table",
    )
    _write_raw_staging_identity(work, expected)
    (work / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA, "raw_staging_identity": expected}),
        encoding="utf-8",
    )

    _validate_raw_staging_identity(work, expected)


def test_raw_staging_identity_rejects_missing_and_mismatch(tmp_path):
    input_json, gate1_script = _identity_inputs(tmp_path)
    work = tmp_path / "work"
    expected = _raw_staging_identity(
        input_json=input_json,
        gate1_script=gate1_script,
        p2_source_fingerprint="a" * 64,
        p2_source_kind="radial_table",
    )
    work.mkdir()
    (work / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity is missing"):
        _validate_raw_staging_identity(work, expected)

    _write_raw_staging_identity(work, expected)
    mismatched = _raw_staging_identity(
        input_json=input_json,
        gate1_script=gate1_script,
        p2_source_fingerprint="b" * 64,
        p2_source_kind="radial_table",
    )
    (work / "manifest.partial.json").write_text(
        json.dumps({"schema": SCHEMA, "raw_staging_identity": mismatched}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_raw_staging_identity(work, expected)


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


def test_p2_hermitian_projection_makes_reverse_edges_exact():
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


def test_p2_hermitian_projection_rejects_missing_reverse_graph_block():
    with pytest.raises(ValueError, match="no reverse graph block"):
        project_non_soc_blocks_hermitian(
            {"0_1_1_0_0": np.ones((1, 1), dtype=np.float32)},
            required_keys=[(0, 1, 1, 0, 0)],
        )


def test_p2_hermitian_projection_rejects_large_mismatch_before_averaging():
    blocks = {
        "0_1_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[1.1]], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="exceeds tolerance") as error:
        project_non_soc_blocks_hermitian(
            blocks,
            required_keys=[(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)],
            mismatch_tolerance=1.0e-3,
        )
    assert '"above_tolerance_count": 1' in str(error.value)
    # Input evidence is not modified on failure.
    np.testing.assert_array_equal(
        blocks["0_1_0_0_0"], np.asarray([[1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        blocks["1_0_0_0_0"], np.asarray([[1.1]], dtype=np.float32)
    )


def test_four_ablation_configs_lock_hb0_and_comparable_loss(tmp_path):
    paths = build_configs(
        reference=DEFAULT_REFERENCE,
        full_root=tmp_path / "full_h",
        delta_root=tmp_path / "h0_delta",
        output_dir=tmp_path / "configs",
        total_steps=50_000,
        train_count=180,
        expected_p2_source_fingerprint="a" * 64,
    )
    configs = {
        name: normalize(deepcopy(json.loads(path.read_text())))
        for name, path in paths.items()
        if name[0].isdigit() and name != "00_cache_materialization_context"
    }
    assert set(configs) == {
        "01_h0_delta_hb0",
        "02_p2_residual_hb0",
        "03_p2_memory_hb0",
        "04_full_h_direct_hb0",
    }
    for config in configs.values():
        embedding = config["model_options"]["embedding"]
        prediction = config["model_options"]["prediction"]
        train = config["train_options"]
        loss = train["loss_options"]["train"]
        assert embedding["output_route"] == "h_b0"
        assert prediction["method"] == "block_native"
        assert prediction["block_decoder"] == "expansion_cg"
        assert train["batch_size"] == 1
        assert train["lr_scheduler"]["type"] == "wsd"
        assert train["lr_scheduler"]["total_steps"] == 50_000
        assert train["num_epoch"] == 278
        assert train["display_freq"] >= train["validation_freq"] > 0
        assert loss["method"] == "hamil_blockwise_nextham"
        assert loss["optimization"] == "block_l1_rmse"
        assert loss["block_reduction"] == "equal_onsite_hopping"
        assert loss["complex_reduction"] == "modulus"

    h0 = configs["01_h0_delta_hb0"]
    assert h0["model_options"]["embedding"]["method"] == "lem_moe_v3_h0"
    assert h0["data_options"]["train"]["get_H0"] is True
    assert h0["data_options"]["train"]["residual_hamiltonian"] is False

    p2 = configs["02_p2_residual_hb0"]
    memory = configs["03_p2_memory_hb0"]
    assert p2["data_options"]["train"]["get_P2"] is True
    assert p2["model_options"]["embedding"]["use_soft_edge_memory"] is False
    assert memory["model_options"]["embedding"]["use_soft_edge_memory"] is True
    for config in (p2, memory):
        assert config["model_options"]["prediction"]["add_prior"] is True
        assert config["data_options"]["train"]["require_p2_blocks"] is True
        assert config["data_options"]["train"]["require_full_h_target"] is True
        assert config["train_options"]["loss_options"]["train"][
            "pred_node_block_key"
        ] == "node_full_hamil_blocks"
        assert config["train_options"]["loss_options"]["train"][
            "target_node_block_key"
        ] == "node_full_hamil_target_blocks"

    direct = configs["04_full_h_direct_hb0"]
    assert direct["model_options"]["embedding"]["method"] == "lem_moe_v3"
    assert direct["data_options"]["train"]["get_H0"] is False
    assert direct["data_options"]["train"]["get_P2"] is False
    assert direct["data_options"]["train"]["require_full_h_target"] is True
    assert direct["train_options"]["loss_options"]["train"][
        "target_node_block_key"
    ] == "node_full_hamil_target_blocks"

    wrong_direct = deepcopy(direct)
    wrong_direct["train_options"]["loss_options"]["train"][
        "target_node_block_key"
    ] = "node_delta_hamil_blocks"
    with pytest.raises(ValueError, match="explicit absolute Full-H"):
        normalize(wrong_direct)
