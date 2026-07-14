#!/usr/bin/env python3
"""Generate the eight non-SOC P2/P23 H-B0 Full-H ablation configs.

Every arm consumes the same dual-prior LMDB view and is supervised against the
same explicit absolute Full-H target.  The experiment matrix changes only:

* physical prior: P2 or P23;
* output head: add the prior AO blocks to a learned correction, or predict Full
  H directly while using the prior only as an embedding feature;
* soft edge memory: disabled or enabled.

The optimizer and 50k-step WSD schedule follow the fresh Hanhai HybridMuon
rerun contract.  ``muon_clip_mode=auto`` is deliberate: the Hanhai production
line used automatic relative-step clipping, while ``muon_clip_rms=0.2`` remains
the requested hard update-RMS cap.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "configs"
    / "0711_expert_prior_ablation"
    / "E_H0node_H0edge_noncfm_baseline.json"
)
SCHEMA = "deeptb.nonsoc_dual_prior_ablation_configs/v1"
MUON_CLIP_MODE = "auto"
MUON_CLIP_MODE_EVIDENCE = (
    "Hanhai production HybridMuon used automatic relative-step clipping; "
    "muon_clip_rms=0.2 is the requested hard update-RMS cap."
)

PRIOR_SPECS: dict[str, dict[str, str]] = {
    "p2": {
        "raw_key": "hamiltonian_p2",
        "node_key": "node_p2",
        "edge_key": "edge_p2",
        "node_block_key": "node_p2_blocks",
        "edge_block_key": "edge_p2_blocks",
        "label": "P2",
    },
    "p23": {
        "raw_key": "hamiltonian_p23",
        "node_key": "node_p23",
        "edge_key": "edge_p23",
        "node_block_key": "node_p23_blocks",
        "edge_block_key": "edge_p23_blocks",
        "label": "P23",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_optional_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if normalized and not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be empty or a 64-character SHA256 digest.")
    return normalized


def _embedding_base(r_max: dict[str, float]) -> dict[str, Any]:
    """Keep the established non-SOC H-B0 backbone fixed across all arms."""

    return {
        "method": "lem_moe_v3",
        "n_layers": 3,
        "top_k": 1,
        "env_embed_multiplicity": 10,
        "avg_num_neighbors": 80,
        "irreps_hidden": (
            "32x0e+32x1o+32x1e+32x2e+32x2o+32x3o+32x3e+"
            "32x4e+32x4o+32x5o+32x6e"
        ),
        "r_max": copy.deepcopy(r_max),
        "universal": True,
        "use_interpolation_out": True,
        "latent_dim": 128,
        "latent_channels": [200, 128],
        "edge_one_hot_dim": 128,
        "use_out_onehot_tp": False,
        "num_experts": 1,
        "num_shared_experts": 1,
        "use_layer_onehot_tp": True,
        "tp_radial_emb": True,
        "equivariant_norm_type": "merged_rms",
        "so2_wigner_apply_mode": "compact_blocks",
        "so2_fusion_mode": "streamed_m_major_ref",
        "mole_linear_mode": "split_loop",
        "so2_m_linear_mode": "standard",
        "mole_full_expert_fast_path": True,
        "onehot_tp_mode": "scalar_fast",
        "output_route": "h_b0",
        "rme_fusion_rank": 4,
        "rme_fusion_init": 0.0,
    }


def _prior_embedding(
    r_max: dict[str, float], *, prior_kind: str, use_memory: bool
) -> dict[str, Any]:
    spec = PRIOR_SPECS[prior_kind]
    embedding = _embedding_base(r_max)
    embedding.update(
        {
            "method": "lem_moe_v3_prior",
            "prior_kind": prior_kind,
            "prior_node_key": spec["node_key"],
            "prior_edge_key": spec["edge_key"],
            "use_prior_init": True,
            "prior_merge_mode": "replace",
            "use_soft_edge_memory": use_memory,
            # Production runs rely on the completed all-record ingest audit.
            "prior_validate_inputs": False,
            "soft_edge_memory_diagnostics_mode": "off",
        }
    )
    if use_memory:
        embedding.update(
            {
                "soft_edge_memory_num_slots": 64,
                "soft_edge_memory_num_heads": 4,
                "soft_edge_memory_head_dim": 16,
                "soft_edge_memory_temperature": 1.0,
                "soft_edge_memory_dropout": 0.0,
                "soft_edge_memory_gate_mode": "deepseek",
                "soft_edge_memory_gate_bias": 0.0,
                "soft_edge_memory_zero_init_output": True,
            }
        )
    return embedding


def _prediction(*, prior_kind: str, head_mode: str) -> dict[str, Any]:
    add_prior = head_mode == "residual_add_prior"
    prediction: dict[str, Any] = {
        "method": "block_native",
        "scale_type": "no_scale",
        "block_decoder": "expansion_cg",
        "blockwise_hamiltonian": True,
        "add_h0": False,
        "add_prior": add_prior,
        "validate_prior_blocks": False,
    }
    if add_prior:
        spec = PRIOR_SPECS[prior_kind]
        prediction.update(
            {
                "prior_node_block_field": spec["node_block_key"],
                "prior_edge_block_field": spec["edge_block_key"],
                "prior_label": spec["label"],
                "full_output_node_field": "node_full_hamil_blocks",
                "full_output_edge_field": "edge_full_hamil_blocks",
            }
        )
    return prediction


def _absolute_full_h_loss(*, head_mode: str) -> dict[str, Any]:
    if head_mode == "residual_add_prior":
        pred_node = "node_full_hamil_blocks"
        pred_edge = "edge_full_hamil_blocks"
    elif head_mode == "direct_full_h":
        pred_node = "node_hamil_blocks"
        pred_edge = "edge_hamil_blocks"
    else:  # pragma: no cover - guarded by the fixed matrix below
        raise ValueError(f"unsupported head_mode: {head_mode}")
    return {
        "method": "hamil_blockwise_nextham",
        "optimization": "block_l1_rmse",
        "block_reduction": "equal_onsite_hopping",
        "complex_reduction": "modulus",
        "target_node_block_key": "node_full_hamil_target_blocks",
        "target_edge_block_key": "edge_full_hamil_target_blocks",
        "target_node_shape_key": "node_full_hamil_target_block_shape",
        "target_edge_shape_key": "edge_full_hamil_target_block_shape",
        "pred_node_block_key": pred_node,
        "pred_edge_block_key": pred_edge,
        "log_feature_compatible": True,
        "feature_log_no_grad": True,
        "distributed_log_reduce": False,
    }


def _train_options(*, total_steps: int, train_count: int, head_mode: str) -> dict[str, Any]:
    loss = _absolute_full_h_loss(head_mode=head_mode)
    return {
        "batch_size": 1,
        "ref_batch_size": 1,
        "val_batch_size": 1,
        "use_ddp": False,
        "num_epoch": int(math.ceil(total_steps / train_count)),
        "optimizer": {
            "type": "HybridMuon",
            "lr": 1.0e-2,
            "weight_decay": 1.0e-2,
            "adam_betas": [0.98, 0.999],
            "adam_eps": 1.0e-20,
            "muon_beta": 0.95,
            "muon_scale": 0.2,
            "muon_clip": True,
            "muon_clip_mode": MUON_CLIP_MODE,
            "muon_clip_rms": 0.2,
        },
        "clip_grad": 0.3,
        "lr_scheduler": {
            "type": "wsd",
            "total_steps": total_steps,
            "warmup_steps": 2500,
            "decay_ratio": 0.65,
            "min_lr": 1.0e-5,
            "warmup_lr": 1.0e-5,
            "decay_type": "cosine",
            "last_epoch": -1,
        },
        "loss_options": {
            "train": copy.deepcopy(loss),
            "validation": copy.deepcopy(loss),
        },
        "save_freq": 1000,
        "validation_freq": 1000,
        "validation_epoch_freq": 0,
        "display_freq": 1000,
        "sliding_win_size": 1000,
        "update_lr_per_iter": True,
        "valid_fast": False,
        "use_tensorboard": True,
        "train_num_workers": 0,
        "val_num_workers": 0,
        "ref_num_workers": 0,
        "data_pin_memory": False,
        "data_persistent_workers": False,
        "log_single_model_compatible_loss": True,
        "log_single_model_compatible_loss_mode": "reduce",
        "monitor_flag": False,
        "allow_tf32": False,
        "float32_matmul_precision": "highest",
        "monitor_cuda_memory": True,
        "precompute_lem_active_edges": True,
        "precompute_lem_cutoff_coeffs": True,
        "flow_options": {"enabled": False},
    }


def _data_split(
    root: Path,
    *,
    prior_kind: str,
    head_mode: str,
    expected_source_fingerprint: str,
    allow_unbound_source_fingerprint: bool,
) -> dict[str, Any]:
    spec = PRIOR_SPECS[prior_kind]
    result: dict[str, Any] = {
        "root": str(root),
        "prefix": "data",
        "separator": ".",
        "type": "LMDBDataset",
        "get_DM": False,
        "get_overlap": False,
        "get_Hamiltonian": True,
        "get_H0": False,
        # Kept for backward-compatible loader wiring; prior_kind selects P2/P23.
        "get_P2": True,
        "prior_kind": prior_kind,
        "h0_key": "hamiltonian_0",
        "prefer_precomputed_h0": True,
        "p2_key": spec["raw_key"],
        "prefer_precomputed_p2": True,
        "require_p2_blocks": head_mode == "residual_add_prior",
        "require_full_h_target": True,
        "audit_p2_representations": False,
        "residual_hamiltonian": False,
        "allow_unbound_prior_source_fingerprint": bool(
            allow_unbound_source_fingerprint
        ),
    }
    if expected_source_fingerprint:
        # The historical option name is retained by the generic P2/P23 loader.
        result["expected_p2_source_fingerprint"] = expected_source_fingerprint
    return result


def _variants() -> list[tuple[str, str, str, bool]]:
    variants: list[tuple[str, str, str, bool]] = []
    index = 1
    for prior_kind in ("p2", "p23"):
        for head_mode, head_slug in (
            ("residual_add_prior", "residual"),
            ("direct_full_h", "direct_full_h"),
        ):
            for use_memory, memory_slug in ((False, "nomemory"), (True, "memory")):
                name = (
                    f"{index:02d}_{prior_kind}_{head_slug}_{memory_slug}_hb0"
                )
                variants.append((name, prior_kind, head_mode, use_memory))
                index += 1
    return variants


def build_configs(
    *,
    reference: Path,
    dual_full_root: Path,
    output_dir: Path,
    total_steps: int = 50_000,
    train_count: int = 180,
    expected_p2_source_fingerprint: str = "",
    expected_p23_source_fingerprint: str = "",
    allow_unbound_source_fingerprints: bool = False,
) -> dict[str, Path]:
    if total_steps <= 2500:
        raise ValueError("total_steps must be greater than the 2500-step warmup.")
    if train_count <= 0:
        raise ValueError("train_count must be positive.")
    fingerprints = {
        "p2": _validate_optional_sha256(
            expected_p2_source_fingerprint,
            field="expected_p2_source_fingerprint",
        ),
        "p23": _validate_optional_sha256(
            expected_p23_source_fingerprint,
            field="expected_p23_source_fingerprint",
        ),
    }
    missing_fingerprints = [kind for kind, value in fingerprints.items() if not value]
    if missing_fingerprints and not allow_unbound_source_fingerprints:
        raise ValueError(
            "Production dual-prior configs require non-empty 64-character "
            "source fingerprints for both P2 and P23; missing "
            f"{missing_fingerprints}. Use allow_unbound_source_fingerprints=True "
            "only for synthetic/dev configs, which are not production-qualified."
        )

    source = _read_json(reference)
    basis = copy.deepcopy(source["common_options"]["basis"])
    r_max = copy.deepcopy(source["model_options"]["embedding"]["r_max"])
    common = {
        "basis": basis,
        "device": "cuda",
        "dtype": "float32",
        "seed": 42,
        "overlap": False,
        "has_soc": False,
    }
    dual_full_root = dual_full_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    experiments: dict[str, dict[str, Any]] = {}
    for name, prior_kind, head_mode, use_memory in _variants():
        config = {
            "common_options": copy.deepcopy(common),
            "model_options": {
                "embedding": _prior_embedding(
                    r_max, prior_kind=prior_kind, use_memory=use_memory
                ),
                "prediction": _prediction(
                    prior_kind=prior_kind, head_mode=head_mode
                ),
            },
            "train_options": _train_options(
                total_steps=total_steps,
                train_count=train_count,
                head_mode=head_mode,
            ),
            "data_options": {
                "train": _data_split(
                    dual_full_root / "train",
                    prior_kind=prior_kind,
                    head_mode=head_mode,
                    expected_source_fingerprint=fingerprints[prior_kind],
                    allow_unbound_source_fingerprint=(
                        allow_unbound_source_fingerprints
                    ),
                ),
                "validation": _data_split(
                    dual_full_root / "validation",
                    prior_kind=prior_kind,
                    head_mode=head_mode,
                    expected_source_fingerprint=fingerprints[prior_kind],
                    allow_unbound_source_fingerprint=(
                        allow_unbound_source_fingerprints
                    ),
                ),
            },
        }
        path = output_dir / f"{name}.json"
        _write_json(path, config)
        written[name] = path
        experiments[name] = {
            "prior_kind": prior_kind,
            "head_mode": head_mode,
            "soft_memory": use_memory,
            "target_semantics": "absolute Full-H",
            "prediction_fields": {
                "node": (
                    "node_full_hamil_blocks"
                    if head_mode == "residual_add_prior"
                    else "node_hamil_blocks"
                ),
                "edge": (
                    "edge_full_hamil_blocks"
                    if head_mode == "residual_add_prior"
                    else "edge_hamil_blocks"
                ),
            },
        }

    manifest = {
        "schema": SCHEMA,
        "reference": str(reference.resolve()),
        "dual_full_root": str(dual_full_root),
        "total_steps": total_steps,
        "train_count": train_count,
        "scheduled_epochs": int(math.ceil(total_steps / train_count)),
        "scheduled_data_steps": int(math.ceil(total_steps / train_count))
        * train_count,
        "iteration_contract": {
            "formal_eval_iter": total_steps,
            "scheduler_total_steps": total_steps,
            "scheduled_epochs": int(math.ceil(total_steps / train_count)),
            "scheduled_data_steps": int(math.ceil(total_steps / train_count))
            * train_count,
            "validation_epoch_freq": 0,
            "note": (
                "Trainer epochs are integer-ceiled, so the latest checkpoint/log "
                "may reach scheduled_data_steps (for raw200: 50040) while formal "
                "comparisons are pinned to scheduler_total_steps/formal_eval_iter."
            ),
        },
        "target_semantics": "explicit absolute Full-H for all eight arms",
        "head_contract": "H-B0 ordinary hidden -> expansion-CG AO blocks",
        "optimizer_contract": {
            "source": "hanhai_fresh_rerun",
            "type": "HybridMuon",
            "muon_clip_mode": MUON_CLIP_MODE,
            "muon_clip_mode_evidence": MUON_CLIP_MODE_EVIDENCE,
        },
        "source_binding_contract": {
            "allow_unbound_source_fingerprints": bool(
                allow_unbound_source_fingerprints
            ),
            "production_qualified": bool(
                not allow_unbound_source_fingerprints
                and all(fingerprints.values())
            ),
            "production_preflight_rule": (
                "reject unless production_qualified=true and both expected "
                "source fingerprints are non-empty 64-hex digests"
            ),
        },
        "prior_contracts": {
            kind: {
                **copy.deepcopy(spec),
                "expected_source_fingerprint": fingerprints[kind],
            }
            for kind, spec in PRIOR_SPECS.items()
        },
        "experiments": experiments,
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in written.items()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--dual-full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--train-count", type=int, default=180)
    parser.add_argument("--expected-p2-source-fingerprint", default="")
    parser.add_argument("--expected-p23-source-fingerprint", default="")
    parser.add_argument(
        "--allow-unbound-source-fingerprints",
        action="store_true",
        help=(
            "Allow missing source hashes only for synthetic/dev generation. "
            "The manifest is marked production_qualified=false."
        ),
    )
    args = parser.parse_args(argv)
    written = build_configs(
        reference=args.reference.resolve(),
        dual_full_root=args.dual_full_root.resolve(),
        output_dir=args.output_dir.resolve(),
        total_steps=args.total_steps,
        train_count=args.train_count,
        expected_p2_source_fingerprint=args.expected_p2_source_fingerprint,
        expected_p23_source_fingerprint=args.expected_p23_source_fingerprint,
        allow_unbound_source_fingerprints=(
            args.allow_unbound_source_fingerprints
        ),
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
