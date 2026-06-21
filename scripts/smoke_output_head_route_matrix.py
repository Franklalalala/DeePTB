#!/usr/bin/env python3
"""Build smoke for the six output-head routes.

The route matrix is split by final LEM contract:

ordinary hidden:
  h_a0: hidden -> no-CG RME -> E3Hamiltonian
  h_a1: hidden -> ICT RME -> E3Hamiltonian
  h_b0: hidden -> Wigner AO block
  h_b1: hidden -> ICT AO block

AO-pair recontract:
  p_b0: AO-pair irreps -> reference Wigner projector -> AO block
  p_b1: AO-pair irreps -> precomputed ICT/projector bank -> AO block
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dptb.nn.build import build_model


BASIS = {"H": "2s1p", "O": "3s2p1d"}
ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"
AO_WIDTH = 14


ROUTES = {
    "h_a0": {
        "mode": "late_rme_expansion_nocg",
        "head": "LateRMEExpansionNoCGHead",
        "final_contract": "ordinary_hidden",
        "uses_ict": False,
        "prediction": {"method": "e3tb", "scale_type": "no_scale"},
    },
    "h_a1": {
        "mode": "late_rme_cartesian_hybrid",
        "head": "LateRMECartesianHybridHead",
        "final_contract": "ordinary_hidden",
        "uses_ict": True,
        "prediction": {"method": "e3tb", "scale_type": "no_scale"},
    },
    "h_b0": {
        "mode": "late_block_expansion_cg",
        "head": "LateBlockExpansionCGHead",
        "final_contract": "ordinary_hidden",
        "uses_ict": False,
        "prediction": {
            "method": "block_native",
            "block_decoder": "expansion_cg",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "h_b1": {
        "mode": "late_block_cartesian_projector",
        "head": "LateBlockCartesianProjectorHead",
        "final_contract": "ordinary_hidden",
        "uses_ict": True,
        "prediction": {
            "method": "block_native",
            "block_decoder": "cartesian_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "p_b0": {
        "mode": "direct_ao_projector",
        "head": "AOAngularProjectorHead",
        "final_contract": "ao_pair",
        "uses_ict": False,
        "extra_embedding": {"ao_projector_backend": "reference_wigner"},
        "prediction": {
            "method": "block_native",
            "block_decoder": "ao_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "p_b1": {
        "mode": "direct_ao_projector",
        "head": "AOAngularProjectorHead",
        "final_contract": "ao_pair",
        "uses_ict": True,
        "extra_embedding": {
            "ao_projector_backend": "precomputed",
            "ao_projector_bank_path": "assets/spd_ao_projectors.json",
        },
        "prediction": {
            "method": "block_native",
            "block_decoder": "ao_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
}


def embedding_options(spec: dict) -> dict:
    options = {
        "method": "lem_moe_v3",
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": ORDINARY_HIDDEN,
        "env_embed_multiplicity": 4,
        "latent_dim": 8,
        "latent_channels": [8],
        "edge_one_hot_dim": 4,
        "num_experts": 1,
        "num_shared_experts": 1,
        "top_k": 1,
        "universal": True,
        "use_layer_onehot_tp": False,
        "use_out_onehot_tp": True,
        "use_interpolation_out": False,
        "tp_radial_emb": False,
        "mole_linear_mode": "indexed_ref",
        "so2_fusion_mode": "streamed_m_major_ref",
        "rme_head_mode": spec["mode"],
        "rme_fusion_rank": 4,
        "rme_fusion_init": 0.0,
        "rme_cartesian_scope": "missing_only",
    }
    options.update(spec.get("extra_embedding", {}))
    return options


def build_route(name: str, spec: dict):
    model = build_model(
        common_options={
            "basis": BASIS,
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding_options(spec),
            "prediction": spec["prediction"],
        },
        train_options={},
        no_check=True,
    )
    embedding = model.embedding
    final_irreps = embedding.layers[-1].irreps_out
    if spec["final_contract"] == "ordinary_hidden":
        assert final_irreps == o3.Irreps(ORDINARY_HIDDEN)
    else:
        assert final_irreps.dim == AO_WIDTH * AO_WIDTH
    assert type(embedding.out_node).__name__ == spec["head"]
    assert getattr(embedding.out_node, "uses_ict") is spec["uses_ict"]
    if spec["prediction"]["method"] == "block_native":
        assert not hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "ao_block"
    else:
        assert hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "rme"
    return {
        "route": name,
        "mode": embedding.rme_head_mode,
        "head": type(embedding.out_node).__name__,
        "final_contract": spec["final_contract"],
        "final_dim": final_irreps.dim,
        "output_contract": embedding.out_node.output_contract,
        "uses_ict": getattr(embedding.out_node, "uses_ict"),
        "uses_e3hamiltonian": hasattr(model, "hamiltonian"),
    }


def main() -> int:
    bank = ROOT / "assets" / "spd_ao_projectors.json"
    if not bank.exists():
        raise FileNotFoundError(bank)
    results = [build_route(name, spec) for name, spec in ROUTES.items()]
    print(json.dumps(results, indent=2, sort_keys=True))
    print("OUTPUT_HEAD_ROUTE_MATRIX_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
