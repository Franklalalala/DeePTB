"""UniTB defaults and explicit translation of historical configuration names."""
from copy import deepcopy
import logging

log = logging.getLogger(__name__)

UNITB_DEFAULTS = {
    "n_layers": 3, "n_radial_basis": 128, "r_max": 7.408480947893776,
    "irreps_hidden": "128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e",
    "avg_num_neighbors": 80, "env_embed_multiplicity": 10,
    "latent_channels": [200, 128], "latent_dim": 128, "edge_one_hot_dim": 128,
    "tp_radial_emb": True, "tp_radial_channels": [32], "res_update_ratios": 0.5,
    "equivariant_norm_type": "merged_rms", "onehot_tp_mode": "scalar_fast",
    "universal": True, "use_interpolation_out": False,
    "num_experts": 4, "num_shared_experts": 1, "top_k": 2,
    "mole_expert_parameterization": "pdq_moe", "mole_expert_rank": 64,
    "mole_linear_mode": "cublas_grouped", "mole_full_expert_fast_path": True,
    "so2_fusion_mode": "streamed_m_major_fused_p0",
    "edge_moe_compact_min_edges": 0, "edge_router_prior_activate": True,
    "edge_router_prior_cg": True, "edge_router_logit": "cosine",
    "edge_router_logit_scale": 10.0, "edge_router_select": "logit",
    "edge_router_bias_at_eval": True, "h0_init_scope": "both",
    "use_flow_time_embedding": True,
}
DENSE_DEFAULTS = {
    "num_shared_experts": 0, "top_k": 1,
    "mole_expert_parameterization": "full", "edge_router_prior_activate": False,
    "edge_router_prior_cg": False, "edge_router_logit": "raw",
    "edge_router_select": "sigmoid", "edge_router_bias_at_eval": False,
    "use_flow_time_embedding": False,
    "irreps_hidden": "32x0e+32x1o+32x2e+32x3o+32x4e+32x5o+32x6e",
}
# Renaming configuration does not rename a parameter or change initialization.
ALIASES = {
    "expert_parameterization": "mole_expert_parameterization",
    "expert_rank": "mole_expert_rank",
    "expert_mixing": "so2_expert_mixing_mode",
    "router_input": "edge_router_input",
    "router_gate": "edge_router_gate",
}
# These branches belong to immutable releases or the frozen graph API.
ARCHIVE_DEFAULTS = {
    "sympe": {"enabled": False}, "latent_irrep_dot": {"enabled": False},
    "node_bilinear": {"enabled": False}, "layer_prior_memory": {"enabled": False},
    "prior_operator": {"enabled": False}, "prior_operator_messages": {"enabled": False},
    "only2b": False, "num_focus": 1,
    "node_message_aggregation": "scatter", "edge_aggregation_gated_attention": False,
    "ffn_hidden_factor": 0.0, "ffn_apply_to_last": False,
    "two_stage_pair_enable": False, "use_uureal_residual_block_input": False,
    "use_spatial_residual_block_input": False,
}
# "lem": edge update -> node update per layer. "slem": hidden-state update ->
# edge update -> node update, with node features local to one cutoff sphere.
LAYER_TOPOLOGIES = ("lem", "slem")


def _archived_option_active(key, value):
    default = ARCHIVE_DEFAULTS[key]
    return bool(value.get("enabled", False)) if isinstance(default, dict) else value != default


def check_layer_topology(topology, options):
    """Validate ``layer_topology``; the SLEM layers accept no archived option."""
    if topology not in LAYER_TOPOLOGIES:
        raise ValueError(f"layer_topology must be one of {list(LAYER_TOPOLOGIES)}, got {topology!r}")
    if topology == "slem":
        for key in ARCHIVE_DEFAULTS:
            if key in options and _archived_option_active(key, options[key]):
                raise ValueError(f"{key} belongs to an archived model and is not available "
                                 "with layer_topology='slem'")
    return topology


def unitb_options(options, *, legacy=False):
    """Resolve defaults once; old methods retain their historical defaults."""
    # Runtime options include the shared OrbitalMapper; preserve object identity.
    options = dict(options)
    for new, old in ALIASES.items():
        if new in options:
            value = options.pop(new)
            if old in options and options[old] != value:
                raise ValueError(f"Conflicting UniTB options: {new} and {old}")
            options[old] = value
    if not legacy:
        for key in ARCHIVE_DEFAULTS:
            if key in options and _archived_option_active(key, options[key]):
                raise ValueError(f"{key} belongs to an archived model, not UniTB")
        defaults = deepcopy(UNITB_DEFAULTS)
        if options.get("num_experts", 4) == 1:
            defaults.update(DENSE_DEFAULTS)
        structure = options.get("structure_mole") or {}
        if structure.get("enabled", False):
            # Structure controls own their router and shared-core defaults.
            count = 1 if structure.get("route_scope", "structure") == "constant" else 4
            defaults.update(num_experts=count, top_k=count, num_shared_experts=1,
                            mole_expert_parameterization="pdq_moe",
                            edge_router_prior_activate=False, edge_router_prior_cg=False,
                            edge_router_bias_speed=0.0, use_flow_time_embedding=False)
        defaults.update(options)
        options = defaults
    if options.get("mole_expert_parameterization") == "pdq_moe":
        options["mole_expert_parameterization"] = "shared_core"
    if any(k.startswith(("mole_expert_", "edge_router_")) for k in options):
        log.info("UniTB reads legacy mole_expert_* and edge_router_* options through PDQ-MoE compatibility translation; parameter keys are unchanged.")
    return options
