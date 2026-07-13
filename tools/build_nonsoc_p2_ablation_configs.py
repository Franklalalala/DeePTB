#!/usr/bin/env python3
"""Generate the four H-B0 non-SOC raw200 ablation configs.

The checked-in 0711 universal config remains the basis/cutoff source of truth.
This builder changes only the data view and prior-conditioning route while
holding the H-B0 head, optimizer, WSD schedule, batch size, and loss fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "configs"
    / "0711_expert_prior_ablation"
    / "E_H0node_H0edge_noncfm_baseline.json"
)
SCHEMA = "deeptb.nonsoc_p2_ablation_configs/v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _data_split(root: Path, *, get_h0: bool, get_p2: bool) -> dict[str, Any]:
    return {
        "root": str(root),
        "prefix": "data",
        "separator": ".",
        "type": "LMDBDataset",
        "get_DM": False,
        "get_overlap": False,
        "get_Hamiltonian": True,
        "get_H0": get_h0,
        "get_P2": get_p2,
        "h0_key": "hamiltonian_0",
        "prefer_precomputed_h0": True,
        "p2_key": "hamiltonian_p2",
        "prefer_precomputed_p2": True,
        # H0->dH is already materialized in its own target view.  Turning the
        # loader switch on here would subtract H0 a second time.
        "residual_hamiltonian": False,
    }


def _embedding_base(r_max: dict[str, float]) -> dict[str, Any]:
    return {
        "method": "lem_moe_v3",
        "n_layers": 3,
        "top_k": 1,
        "env_embed_multiplicity": 10,
        "avg_num_neighbors": 80,
        # Complete non-SOC AO-pair content through f x f (L <= 6).
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


def _h0_embedding(r_max: dict[str, float]) -> dict[str, Any]:
    embedding = _embedding_base(r_max)
    embedding.update(
        {
            "method": "lem_moe_v3_h0",
            "h0_node_key": "node_h0",
            "h0_edge_key": "edge_h0",
            "use_h0_init": True,
            "use_h0_node_init": True,
            "use_h0_edge_init": True,
            "h0_node_mode": "direct",
            "h0_merge_mode": "replace",
            "h0_fallback_to_hamiltonian": False,
            "fallback_to_hamiltonian": False,
        }
    )
    return embedding


def _p2_embedding(r_max: dict[str, float], *, use_memory: bool) -> dict[str, Any]:
    embedding = _embedding_base(r_max)
    embedding.update(
        {
            "method": "lem_moe_v3_prior",
            "prior_kind": "p2",
            "prior_node_key": "node_p2",
            "prior_edge_key": "edge_p2",
            "use_prior_init": True,
            "prior_merge_mode": "replace",
            "use_soft_edge_memory": use_memory,
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


def _prediction(*, add_prior: bool) -> dict[str, Any]:
    prediction: dict[str, Any] = {
        "method": "block_native",
        "scale_type": "no_scale",
        "block_decoder": "expansion_cg",
        "blockwise_hamiltonian": True,
        "add_h0": False,
        "add_prior": add_prior,
    }
    if add_prior:
        prediction.update(
            {
                "prior_node_block_field": "node_p2_blocks",
                "prior_edge_block_field": "edge_p2_blocks",
                "prior_label": "P2",
                "full_output_node_field": "node_full_hamil_blocks",
                "full_output_edge_field": "edge_full_hamil_blocks",
            }
        )
    return prediction


def _loss(*, full_output: bool) -> dict[str, Any]:
    loss: dict[str, Any] = {
        "method": "hamil_blockwise_nextham",
        "optimization": "block_l1_rmse",
        "block_reduction": "equal_onsite_hopping",
        "complex_reduction": "modulus",
        "target_node_block_key": "node_delta_hamil_blocks",
        "target_edge_block_key": "edge_delta_hamil_blocks",
        "target_node_shape_key": "node_delta_hamil_block_shape",
        "target_edge_shape_key": "edge_delta_hamil_block_shape",
        "log_feature_compatible": True,
        "feature_log_no_grad": True,
        "distributed_log_reduce": False,
    }
    if full_output:
        loss.update(
            {
                "pred_node_block_key": "node_full_hamil_blocks",
                "pred_edge_block_key": "edge_full_hamil_blocks",
            }
        )
    return loss


def _train_options(*, total_steps: int, train_count: int, full_output: bool) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "ref_batch_size": 1,
        "val_batch_size": 1,
        "use_ddp": False,
        "num_epoch": int(math.ceil(total_steps / train_count)),
        "optimizer": {
            "lr": 5.0e-4,
            "type": "AdamW",
            "betas": [0.99, 0.999],
            "weight_decay": 0.0,
        },
        "clip_grad": 1.0,
        "lr_scheduler": {
            "type": "wsd",
            "total_steps": total_steps,
            "warmup_steps": 1000,
            "decay_ratio": 0.65,
            "min_lr": 1.0e-6,
            "warmup_lr": 0.0,
            "decay_type": "cosine",
            "last_epoch": -1,
        },
        "loss_options": {
            "train": _loss(full_output=full_output),
            "validation": _loss(full_output=full_output),
        },
        "save_freq": 5000,
        "validation_freq": 1000,
        "display_freq": 100,
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


def build_configs(
    *,
    reference: Path,
    full_root: Path,
    delta_root: Path,
    output_dir: Path,
    total_steps: int,
    train_count: int,
) -> dict[str, Path]:
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

    variants = {
        "01_h0_delta_hb0": {
            "view": delta_root,
            "get_h0": True,
            "get_p2": False,
            "embedding": _h0_embedding(r_max),
            "add_prior": False,
            "full_output": False,
            "target_semantics": "cached dH=FullH-H0; block loss equals reconstructed Full-H error",
        },
        "02_p2_residual_hb0": {
            "view": full_root,
            "get_h0": False,
            "get_p2": True,
            "embedding": _p2_embedding(r_max, use_memory=False),
            "add_prior": True,
            "full_output": True,
            "target_semantics": "absolute Full-H",
        },
        "03_p2_memory_hb0": {
            "view": full_root,
            "get_h0": False,
            "get_p2": True,
            "embedding": _p2_embedding(r_max, use_memory=True),
            "add_prior": True,
            "full_output": True,
            "target_semantics": "absolute Full-H",
        },
        "04_full_h_direct_hb0": {
            "view": full_root,
            "get_h0": False,
            "get_p2": False,
            "embedding": _embedding_base(r_max),
            "add_prior": False,
            "full_output": False,
            "target_semantics": "absolute Full-H",
        },
    }

    written: dict[str, Path] = {}
    for name, variant in variants.items():
        view = Path(variant["view"])
        config = {
            "common_options": copy.deepcopy(common),
            "model_options": {
                "embedding": variant["embedding"],
                "prediction": _prediction(add_prior=bool(variant["add_prior"])),
            },
            "train_options": _train_options(
                total_steps=total_steps,
                train_count=train_count,
                full_output=bool(variant["full_output"]),
            ),
            "data_options": {
                "train": _data_split(
                    view / "train",
                    get_h0=bool(variant["get_h0"]),
                    get_p2=bool(variant["get_p2"]),
                ),
                "validation": _data_split(
                    view / "validation",
                    get_h0=bool(variant["get_h0"]),
                    get_p2=bool(variant["get_p2"]),
                ),
            },
        }
        path = output_dir / f"{name}.json"
        _write_json(path, config)
        written[name] = path

    context = copy.deepcopy(_read_json(written["03_p2_memory_hb0"]))
    context["data_options"] = {
        "train": _data_split(
            output_dir / "__raw_staging_is_overridden_by_materializer__",
            get_h0=True,
            get_p2=True,
        )
    }
    context_path = output_dir / "00_cache_materialization_context.json"
    _write_json(context_path, context)
    written["00_cache_materialization_context"] = context_path

    manifest = {
        "schema": SCHEMA,
        "reference": str(reference.resolve()),
        "full_root": str(full_root.resolve()),
        "delta_root": str(delta_root.resolve()),
        "total_steps": total_steps,
        "train_count": train_count,
        "head_contract": "H-B0 ordinary hidden -> expansion-CG AO blocks",
        "scheduled_epochs": int(math.ceil(total_steps / train_count)),
        "scheduled_data_steps": int(math.ceil(total_steps / train_count))
        * train_count,
        "experiments": {
            name: {
                "target_semantics": variant["target_semantics"],
                "get_h0": bool(variant["get_h0"]),
                "get_p2": bool(variant["get_p2"]),
                "add_prior": bool(variant["add_prior"]),
                "soft_memory": bool(
                    variant["embedding"].get("use_soft_edge_memory", False)
                ),
            }
            for name, variant in variants.items()
        },
        "files": {name: str(path.resolve()) for name, path in written.items()},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--delta-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--train-count", type=int, default=180)
    args = parser.parse_args(argv)
    if args.total_steps <= 1000:
        raise ValueError("total_steps must be greater than the 1000-step warmup.")
    if args.train_count <= 0:
        raise ValueError("train_count must be positive.")
    written = build_configs(
        reference=args.reference.resolve(),
        full_root=args.full_root.resolve(),
        delta_root=args.delta_root.resolve(),
        output_dir=args.output_dir.resolve(),
        total_steps=args.total_steps,
        train_count=args.train_count,
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
