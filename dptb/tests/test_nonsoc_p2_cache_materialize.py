from __future__ import annotations

import inspect
import json
from copy import deepcopy

import numpy as np
import pytest

from dptb.data.interfaces.abacus import _abacus_parse, recursive_parse
from dptb.utils.argcheck import normalize
from tools.build_nonsoc_p2_ablation_configs import DEFAULT_REFERENCE, build_configs
from tools.materialize_nonsoc_p2_cache import (
    RY_TO_EV,
    _guard_output,
    dense_p2_to_deeptb_blocks,
    parse_basis_lines,
)


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


def test_four_ablation_configs_lock_hb0_and_comparable_loss(tmp_path):
    paths = build_configs(
        reference=DEFAULT_REFERENCE,
        full_root=tmp_path / "full_h",
        delta_root=tmp_path / "h0_delta",
        output_dir=tmp_path / "configs",
        total_steps=50_000,
        train_count=180,
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
        assert config["train_options"]["loss_options"]["train"][
            "pred_node_block_key"
        ] == "node_full_hamil_blocks"

    direct = configs["04_full_h_direct_hb0"]
    assert direct["model_options"]["embedding"]["method"] == "lem_moe_v3"
    assert direct["data_options"]["train"]["get_H0"] is False
    assert direct["data_options"]["train"]["get_P2"] is False
