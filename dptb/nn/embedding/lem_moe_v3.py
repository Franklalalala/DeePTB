"""Frozen graph-router API backed by the shared UniTB layer implementation."""
from .emb import Embedding
from .unitb_backbone import UniTBBackbone
from .unitb_ops import (_normalize_node_message_aggregation, _normalize_edge_attention_key_source, _normalize_onehot_tp_mode, _normalize_stable_standard_compat_mode, _normalize_so2_expert_mixing_mode, _normalize_so2_moe_layers, _normalize_cg_head_impl, _build_so2_post_activation_expert_mixer, _apply_so2_tp_or_post_activation_mixer, _instruction_get, _onehot_weight_shape, _split_last, ScalarOnehotTP, _scalar_onehot_tp_fast, _capture_shift_hidden, _apply_onehot_tp)
from .unitb_layers import (ShiftedSoftPlus, _cosine_cutoff_per_edge, _polynomial_cutoff_per_edge, InitLayer, UpdateNode, UpdateEdge, Layer)


@Embedding.register("lem_moe_v3")
class LemMoEV3(UniTBBackbone):
    """Graph-level routing with the historical positional and state contracts."""
