from __future__ import annotations

import copy

import torch

from dptb.data import _keys
from dptb.nn.build import build_model

from test_residual_ao_block_ode import _b_flow, _b_record


def _flow_model(*, mp_cutoff=None, pair_refine_enable=True):
    embedding = {
        "method": "lem_pair",
        "output_route": "h_b0",
        "h0_init_scope": "both",
        "use_spatial_residual_block_input": True,
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": "2x0e+2x1o+2x1e+2x2e",
        "env_embed_multiplicity": 2,
        "latent_dim": 6,
        "latent_channels": [6],
        "edge_one_hot_dim": 3,
        "num_experts": 1,
        "num_shared_experts": 1,
        "top_k": 1,
        "universal": True,
        "use_layer_onehot_tp": False,
        "use_out_onehot_tp": False,
        "use_interpolation_out": False,
        "tp_radial_emb": False,
        "mole_linear_mode": "indexed_ref",
        "so2_fusion_mode": "streamed_m_major_ref",
        "rme_fusion_rank": 3,
        "rme_fusion_init": 0.0,
        "use_flow_time_embedding": True,
        "flow_time_condition_edges": True,
        "flow_time_allow_missing": False,
        "require_full_block_edge_coverage": True,
        "pair_refine_enable": pair_refine_enable,
        "pair_refine_rank": 4,
        "pair_refine_init": 0.1,
    }
    if mp_cutoff is not None:
        embedding["mp_cutoff"] = mp_cutoff
        embedding["mp_avg_num_neighbors"] = 1.0
    return build_model(
        common_options={
            "basis": {"H": "1s", "C": "1s1p"},
            "overlap": False,
            "dtype": "float64",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding,
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "scale_type": "no_scale",
            },
        },
        train_options={},
        no_check=False,
    ).to(dtype=torch.float64).eval()


def test_enabled_pair_refine_preserves_flow_full_coverage_contract():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(20260723)
        model = _flow_model()
        flow = _b_flow(model.idp, dtype=torch.float64)
        raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=31)
        model_data, _, _ = flow.prepare_batch(
            copy.deepcopy(raw),
            copy.deepcopy(raw),
            t=torch.tensor([0.41], dtype=torch.float64),
        )
        certified = []
        original = model.embedding._require_ordered_full_block_edge_coverage

        def capture(edge_index, active_edges, *args):
            certified.append(active_edges.detach().clone())
            return original(edge_index, active_edges, *args)

        model.embedding._require_ordered_full_block_edge_coverage = capture
        with torch.no_grad():
            output = model(model_data)

        expected = torch.arange(output[_keys.EDGE_INDEX_KEY].shape[1])
        assert len(certified) == 1
        assert torch.equal(certified[0].cpu(), expected)
        assert flow._strict_certification_mode == "always"
        assert flow._strict_certification_batches == 1
        assert model.embedding._edge_graph_invariant_checked is True
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            "node_hamil_blocks",
            "edge_hamil_blocks",
        ):
            assert torch.isfinite(output[key]).all()
    finally:
        torch.set_default_dtype(previous)
