#!/usr/bin/env python3
"""Build smoke for the six canonical output routes from the route registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dptb.nn.build import build_model
from dptb.nn.embedding.output_routes import OFFICIAL_OUTPUT_ROUTES


BASIS = {"H": "2s1p", "O": "3s2p1d"}
ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"
AO_WIDTH = 14


ROUTES = {
    "h_a0": {
        "prediction": {"method": "e3tb", "scale_type": "no_scale"},
    },
    "h_a1": {
        "extra_embedding": {"rme_cartesian_scope": "all"},
        "prediction": {"method": "e3tb", "scale_type": "no_scale"},
    },
    "h_b0": {
        "prediction": {
            "method": "block_native",
            "block_decoder": "expansion_cg",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "h_b1": {
        "prediction": {
            "method": "block_native",
            "block_decoder": "cartesian_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "p_b0": {
        "extra_embedding": {"ao_projector_backend": "reference_wigner"},
        "prediction": {
            "method": "block_native",
            "block_decoder": "ao_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
    "p_b1_ict": {
        "extra_embedding": {
            "ao_projector_backend": "precomputed",
            "ao_projector_bank_path": "assets/spd_ao_projectors_ict.json",
        },
        "prediction": {
            "method": "block_native",
            "block_decoder": "ao_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
    },
}


def embedding_options(route_name: str, route: dict) -> dict:
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
        "output_route": route_name,
        "rme_fusion_rank": 4,
        "rme_fusion_init": 0.0,
    }
    options.update(route.get("extra_embedding", {}))
    return options


def build_route(name: str, route: dict):
    model = build_model(
        common_options={
            "basis": BASIS,
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding_options(name, route),
            "prediction": route["prediction"],
        },
        train_options={},
        no_check=True,
    )
    embedding = model.embedding
    spec = embedding.output_route_spec
    assert spec.canonical_name == name
    final_irreps = embedding.layers[-1].irreps_out
    if spec.final_irreps_kind == "ordinary_hidden":
        assert final_irreps == o3.Irreps(ORDINARY_HIDDEN)
    elif spec.final_irreps_kind == "ao_pair":
        assert final_irreps.dim == AO_WIDTH * AO_WIDTH
    else:
        assert final_irreps == embedding.idp.orbpair_irreps.sort()[0].simplify()

    assert type(embedding.out_node).__name__ == spec.head_class_name
    assert getattr(embedding.out_node, "uses_ict", False) is spec.uses_ict
    assert getattr(
        embedding.out_node, "uses_precomputed_projector", False
    ) is spec.uses_precomputed_projector
    assert hasattr(model, "hamiltonian") is spec.uses_e3hamiltonian
    assert embedding.out_node.output_contract == spec.output_contract

    if name == "h_a1":
        assert embedding.out_node.coverage_report["product_paths"] > 0
    if name == "h_b1":
        assert embedding.out_node.coverage_report["direct_paths"] > 0
        assert embedding.out_node.coverage_report["product_paths"] == 0
        assert not hasattr(embedding.out_node, "left")
        assert not hasattr(embedding.out_node, "right")
    if name == "p_b1_ict":
        assert embedding.out_node.projector_source == "cartesian_ict"
        assert embedding.out_node.projector_provenance.generator_id == (
            "deeptb.cartesian_stf_3j/v1"
        )

    result = spec.metadata()
    result.update(
        {
            "route": name,
            "final_dim": final_irreps.dim,
            "projector_source": getattr(embedding.out_node, "projector_source", None),
        }
    )
    return result


def main() -> int:
    if tuple(ROUTES) != OFFICIAL_OUTPUT_ROUTES:
        raise RuntimeError(
            f"Smoke routes {tuple(ROUTES)!r} != registry {OFFICIAL_OUTPUT_ROUTES!r}."
        )
    bank = ROOT / "assets" / "spd_ao_projectors_ict.json"
    if not bank.exists():
        raise FileNotFoundError(bank)
    results = [build_route(name, route) for name, route in ROUTES.items()]
    print(json.dumps(results, indent=2, sort_keys=True))
    print("OUTPUT_HEAD_ROUTE_MATRIX_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
