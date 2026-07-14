from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.build_nonsoc_dual_prior_ablation_configs import (
    DEFAULT_REFERENCE,
    MUON_CLIP_MODE,
    SCHEMA,
    build_configs,
)


P2_FINGERPRINT = "2" * 64
P23_FINGERPRINT = "3" * 64


def _load_all(written: dict[str, Path]) -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in written.items()
    }


def _remove_memory_axis(embedding: dict) -> None:
    embedding["soft_edge_memory"]["enabled"] = "<memory-axis>"


def _canonical_without_axes(config: dict) -> dict:
    """Erase only the declared prior, head and memory axes."""

    canonical = copy.deepcopy(config)
    embedding = canonical["model_options"]["embedding"]
    prediction = canonical["model_options"]["prediction"]
    embedding["prior_kind"] = "<prior-axis>"
    embedding["prior_node_key"] = "<prior-axis>"
    embedding["prior_edge_key"] = "<prior-axis>"
    _remove_memory_axis(embedding)

    prediction["reconstruction"] = "<head-axis>"
    for key in (
        "prior_node_block_field",
        "prior_edge_block_field",
        "prior_label",
        "full_output_node_field",
        "full_output_edge_field",
    ):
        prediction.pop(key, None)

    for split in ("train", "validation"):
        data = canonical["data_options"][split]
        data["prior_kind"] = "<prior-axis>"
        data["p2_key"] = "<prior-axis>"
        data["expected_p2_source_fingerprint"] = "<prior-axis>"
        data["require_p2_blocks"] = "<head-axis>"
        loss = canonical["train_options"]["loss_options"][split]
        loss["pred_node_block_key"] = "<head-axis>"
        loss["pred_edge_block_key"] = "<head-axis>"
    return canonical


def test_builds_unique_complete_eight_arm_matrix(tmp_path: Path) -> None:
    dual_root = tmp_path / "dual_full_h"
    output = tmp_path / "configs"
    written = build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=dual_root,
        output_dir=output,
        expected_p2_source_fingerprint=P2_FINGERPRINT,
        expected_p23_source_fingerprint=P23_FINGERPRINT,
    )
    configs = _load_all(written)

    assert len(configs) == 8
    assert len(set(configs)) == 8
    axes = set()
    serialized = set()
    for name, config in configs.items():
        embedding = config["model_options"]["embedding"]
        prediction = config["model_options"]["prediction"]
        axes.add(
            (
                embedding["prior_kind"],
                (
                    "residual_add_prior"
                    if prediction["reconstruction"] == "prior_residual"
                    else "direct_full_h"
                ),
                embedding["soft_edge_memory"]["enabled"],
            )
        )
        serialized.add(json.dumps(config, sort_keys=True))
        assert name.startswith(tuple(f"{index:02d}_" for index in range(1, 9)))

    assert axes == {
        (prior, head, memory)
        for prior in ("p2", "p23")
        for head in ("residual_add_prior", "direct_full_h")
        for memory in (False, True)
    }
    assert len(serialized) == 8

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == SCHEMA
    assert len(manifest["experiments"]) == 8
    assert manifest["optimizer_contract"]["muon_clip_mode"] == MUON_CLIP_MODE
    assert manifest["source_binding_contract"] == {
        "allow_unbound_source_fingerprints": False,
        "production_qualified": True,
        "production_preflight_rule": (
            "reject unless production_qualified=true and both expected source "
            "fingerprints are non-empty 64-hex digests"
        ),
    }
    assert manifest["iteration_contract"]["formal_eval_iter"] == 50_000
    assert manifest["iteration_contract"]["scheduler_total_steps"] == 50_000
    assert manifest["iteration_contract"]["scheduled_data_steps"] == 50_040
    assert manifest["iteration_contract"]["validation_epoch_freq"] == 0
    for name, path in written.items():
        entry = manifest["files"][name]
        assert Path(entry["path"]) == path
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_axis_contracts_and_all_other_options_are_invariant(tmp_path: Path) -> None:
    dual_root = (tmp_path / "dual_full_h").resolve()
    written = build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=dual_root,
        output_dir=tmp_path / "configs",
        expected_p2_source_fingerprint=P2_FINGERPRINT,
        expected_p23_source_fingerprint=P23_FINGERPRINT,
    )
    configs = _load_all(written)
    canonical = [_canonical_without_axes(config) for config in configs.values()]
    assert all(item == canonical[0] for item in canonical[1:])

    for config in configs.values():
        common = config["common_options"]
        embedding = config["model_options"]["embedding"]
        prediction = config["model_options"]["prediction"]
        train = config["train_options"]
        prior = embedding["prior_kind"]
        add_prior = prediction["reconstruction"] == "prior_residual"

        assert common["seed"] == 42
        assert common["has_soc"] is False
        assert embedding["output_route"] == "h_b0"
        assert prediction["method"] == "block_native"
        assert prediction["block_decoder"] == "expansion_cg"
        assert prediction["blockwise_hamiltonian"] is True

        expected = {
            "p2": {
                "raw": "hamiltonian_p2",
                "node": "node_p2",
                "edge": "edge_p2",
                "node_blocks": "node_p2_blocks",
                "edge_blocks": "edge_p2_blocks",
                "label": "P2",
                "fingerprint": P2_FINGERPRINT,
            },
            "p23": {
                "raw": "hamiltonian_p23",
                "node": "node_p23",
                "edge": "edge_p23",
                "node_blocks": "node_p23_blocks",
                "edge_blocks": "edge_p23_blocks",
                "label": "P23",
                "fingerprint": P23_FINGERPRINT,
            },
        }[prior]
        assert embedding["prior_node_key"] == expected["node"]
        assert embedding["prior_edge_key"] == expected["edge"]

        for split in ("train", "validation"):
            data = config["data_options"][split]
            assert data["root"] == str(dual_root / split)
            assert data["get_P2"] is True
            assert data["prior_kind"] == prior
            assert data["p2_key"] == expected["raw"]
            assert data["expected_p2_source_fingerprint"] == expected["fingerprint"]
            assert data["allow_unbound_prior_source_fingerprint"] is False
            assert data["require_full_h_target"] is True
            assert data["residual_hamiltonian"] is False
            assert data["require_p2_blocks"] is add_prior

            loss = train["loss_options"][split]
            assert loss["target_node_block_key"] == "node_full_hamil_target_blocks"
            assert loss["target_edge_block_key"] == "edge_full_hamil_target_blocks"
            if add_prior:
                assert loss["pred_node_block_key"] == "node_full_hamil_blocks"
                assert loss["pred_edge_block_key"] == "edge_full_hamil_blocks"
            else:
                assert loss["pred_node_block_key"] == "node_hamil_blocks"
                assert loss["pred_edge_block_key"] == "edge_hamil_blocks"

        if add_prior:
            assert prediction["prior_node_block_field"] == expected["node_blocks"]
            assert prediction["prior_edge_block_field"] == expected["edge_blocks"]
            assert prediction["prior_label"] == expected["label"]
        else:
            assert "prior_node_block_field" not in prediction
            assert "prior_edge_block_field" not in prediction

        optimizer = train["optimizer"]
        assert optimizer == {
            "type": "HybridMuon",
            "lr": 0.01,
            "weight_decay": 0.01,
            "adam_betas": [0.98, 0.999],
            "adam_eps": 1e-20,
            "muon_beta": 0.95,
            "muon_scale": 0.2,
            "muon_clip": True,
            "muon_clip_mode": "auto",
            "muon_clip_rms": 0.2,
        }
        assert train["clip_grad"] == pytest.approx(0.3)
        assert train["batch_size"] == 1
        assert train["save_freq"] == 1000
        assert train["validation_freq"] == 1000
        assert train["validation_epoch_freq"] == 0
        assert train["display_freq"] == 1000
        assert train["lr_scheduler"] == {
            "type": "wsd",
            "total_steps": 50_000,
            "warmup_steps": 2500,
            "decay_ratio": 0.65,
            "min_lr": 1e-5,
            "warmup_lr": 1e-5,
            "decay_type": "cosine",
            "last_epoch": -1,
        }


def test_rejects_invalid_schedule_and_fingerprints(tmp_path: Path) -> None:
    kwargs = {
        "reference": DEFAULT_REFERENCE,
        "dual_full_root": tmp_path / "dual",
        "output_dir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="2500-step warmup"):
        build_configs(**kwargs, total_steps=2500)
    with pytest.raises(ValueError, match="train_count must be positive"):
        build_configs(**kwargs, train_count=0)
    with pytest.raises(ValueError, match="64-character SHA256"):
        build_configs(**kwargs, expected_p23_source_fingerprint="not-a-digest")
    with pytest.raises(ValueError, match="require non-empty 64-character"):
        build_configs(**kwargs)


def test_explicit_unbound_dev_override_is_never_production_qualified(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    written = build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=tmp_path / "dual",
        output_dir=output,
        allow_unbound_source_fingerprints=True,
    )
    assert len(written) == 8
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    binding = manifest["source_binding_contract"]
    assert binding["allow_unbound_source_fingerprints"] is True
    assert binding["production_qualified"] is False
    for prior in ("p2", "p23"):
        assert manifest["prior_contracts"][prior]["expected_source_fingerprint"] == ""
    configs = _load_all(written)
    for config in configs.values():
        for split in ("train", "validation"):
            assert config["data_options"][split][
                "allow_unbound_prior_source_fingerprint"
            ] is True


def test_all_eight_configs_pass_strict_argcheck(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from dptb.utils.argcheck import normalize

    written = build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=tmp_path / "dual",
        output_dir=tmp_path / "out",
        expected_p2_source_fingerprint=P2_FINGERPRINT,
        expected_p23_source_fingerprint=P23_FINGERPRINT,
    )
    for path in written.values():
        config = json.loads(path.read_text(encoding="utf-8"))
        normalized = normalize(copy.deepcopy(config))
        assert normalized["data_options"]["train"]["prior_kind"] in {"p2", "p23"}


def test_argcheck_requires_prior_source_hash_unless_explicit_dev_override(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from dptb.utils.argcheck import normalize

    written = build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=tmp_path / "dual",
        output_dir=tmp_path / "out",
        expected_p2_source_fingerprint=P2_FINGERPRINT,
        expected_p23_source_fingerprint=P23_FINGERPRINT,
    )
    config = json.loads(next(iter(written.values())).read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        config["data_options"][split]["expected_p2_source_fingerprint"] = ""
        config["data_options"][split][
            "allow_unbound_prior_source_fingerprint"
        ] = False

    with pytest.raises(ValueError, match="expected_p2_source_fingerprint is required"):
        normalize(copy.deepcopy(config))

    for split in ("train", "validation"):
        config["data_options"][split][
            "allow_unbound_prior_source_fingerprint"
        ] = True
    normalized = normalize(copy.deepcopy(config))
    assert normalized["data_options"]["train"][
        "allow_unbound_prior_source_fingerprint"
    ] is True
