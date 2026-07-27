Model Options
=============

``0726-light`` keeps one embedding-plus-prediction model family. The strict runtime schema in ``dptb.utils.argcheck`` remains the source of truth. This page intentionally documents shared fields once so aliases do not multiply the same schema.

Supported embedding methods
---------------------------

* ``emoles``
* ``emoles_openequi``
* ``emoles_openequi_norm``
* ``emoles_openequi_norm_v2``
* ``emoles_openequi_eqv3``
* ``emoles_openequi_eqv3_ffn``
* ``emoles_openequi_nodeffn``
* ``lem_moe_openequi``
* ``lem_in_frame``
* ``lem_in_frame_openequi``
* ``lem_moe_v3``
* ``lem_moe_v3_edge``
* ``lem_moe_v3_h0``
* ``lem_pair``
* ``lem_moe_v3_prior``
* ``lem_moe_v3_edge_h0``
* ``lem_non_linear``
* ``lem_non_linear_h0``

.. note::

   The ``*_openequi*`` methods build their tensor products through
   ``dptb.nn.embedding.oeq_tp.OEQTensorProduct``, which accepts six connection
   modes (``uvw``, ``uvu``, ``uvv``, ``uuw``, ``uuu``, ``uvuv``). Upstream
   openequivariance 0.6.8 documents support for ``uvw`` and ``uvu`` only.
   ``self_mix_mode`` is a free-form string, so a value containing ``uuw``
   (together with ``self_mix_flag: true``) reaches an undocumented upstream
   mode. DeePTB does not reject it; validate it against your installed
   openequivariance build first.

Shared embedding options
------------------------

shared_embedding_options:
    | type: ``dict``, optional
    | argument path: ``shared_embedding_options``

    irreps_hidden:
        | type: ``str``
        | argument path: ``shared_embedding_options/irreps_hidden``

    avg_num_neighbors:
        | type: ``int`` | ``float``
        | argument path: ``shared_embedding_options/avg_num_neighbors``

    r_max:
        | type: ``int`` | ``dict`` | ``float``
        | argument path: ``shared_embedding_options/r_max``

    n_layers:
        | type: ``int``
        | argument path: ``shared_embedding_options/n_layers``

    self_mix_mode:
        | type: ``str``, optional, default: ``full``
        | argument path: ``shared_embedding_options/self_mix_mode``

    self_mix_type:
        | type: ``str``, optional, default: ``all``
        | argument path: ``shared_embedding_options/self_mix_type``

    self_mix_flag:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/self_mix_flag``

    optimized_in_frame:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/optimized_in_frame``

    self_mix_iter:
        | type: ``int``, optional, default: ``2``
        | argument path: ``shared_embedding_options/self_mix_iter``

    n_radial_basis:
        | type: ``int``, optional, default: ``128``
        | argument path: ``shared_embedding_options/n_radial_basis``

    top_k:
        | type: ``int``, optional, default: ``4``
        | argument path: ``shared_embedding_options/top_k``

        The number of experts to be used in MoE. Default: 1

    num_experts:
        | type: ``int``, optional, default: ``24``
        | argument path: ``shared_embedding_options/num_experts``

        The number of experts for MoE. Default: 8

    num_shared_experts:
        | type: ``int``, optional, default: ``4``
        | argument path: ``shared_embedding_options/num_shared_experts``

        The number of experts for MoE. Default: 8

    mole_full_expert_fast_path:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/mole_full_expert_fast_path``

        When `top_k >= num_experts`, skip top-k/one-hot/scatter router work and directly use dense normalized expert weights. This is mathematically equivalent to selecting all routed experts. Default: `True`.

    PolynomialCutoff_p:
        | type: ``int``, optional, default: ``6``
        | argument path: ``shared_embedding_options/PolynomialCutoff_p``

        The order of polynomial cutoff function. Default: 6

    cutoff_type:
        | type: ``str``, optional, default: ``polynomial``
        | argument path: ``shared_embedding_options/cutoff_type``

        The type of cutoff function. Default: polynomial

    color_mode:
        | type: ``str``, optional, default: ``tp``
        | argument path: ``shared_embedding_options/color_mode``

        The type of color mode. Default: tp

    onehot_mode:
        | type: ``str``, optional, default: ``FullTP``
        | argument path: ``shared_embedding_options/onehot_mode``

        The type of onehot mode. Default: FullTP

    env_embed_multiplicity:
        | type: ``int``, optional, default: ``64``
        | argument path: ``shared_embedding_options/env_embed_multiplicity``

    tp_radial_emb:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/tp_radial_emb``

        Whether to use tensor product radial embedding.

    tp_radial_channels:
        | type: ``list``, optional, default: ``[32]``
        | argument path: ``shared_embedding_options/tp_radial_channels``

        The number of channels in tensor product radial embedding.

    latent_channels:
        | type: ``list``, optional, default: ``[32]``
        | argument path: ``shared_embedding_options/latent_channels``

        The number of channels in latent embedding.

    latent_dim:
        | type: ``int``, optional, default: ``64``
        | argument path: ``shared_embedding_options/latent_dim``

        The dimension of latent embedding.

    edge_one_hot_dim:
        | type: ``int``, optional, default: ``128``
        | argument path: ``shared_embedding_options/edge_one_hot_dim``

        The dimension of edge_one_hot.

    use_out_onehot_tp:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/use_out_onehot_tp``

        Whether to use out_onehot_tp.

    use_layer_onehot_tp:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/use_layer_onehot_tp``

        Whether to use layer_onehot_tp.

    output_route:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/output_route``

        Canonical output route. Official matrix: `h_a0`, `h_a1`, `h_b0`, `h_b1`, `p_b0`, `p_b1_ict`. Controls: `legacy_rme`, `rme_fusion`, `p_b1_reference`, `debug_block_linear`.

    rme_head_mode:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/rme_head_mode``

        Deprecated output-route alias retained for old configs/checkpoints. Prefer `output_route`.

    rme_fusion_rank:
        | type: ``int``, optional, default: ``16``
        | argument path: ``shared_embedding_options/rme_fusion_rank``

        Low-rank scalar-conditioning width for output heads. Default: 16.

    rme_fusion_init:
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``shared_embedding_options/rme_fusion_init``

        Stddev of dynamic output-head projections. 0.0 disables dynamic residual/path weights at initialization.

    rme_fusion_condition:
        | type: ``str``, optional, default: ``scalar_0e``
        | argument path: ``shared_embedding_options/rme_fusion_condition``

        Condition source for output heads. Currently only `scalar_0e`.

    rme_cartesian_scope:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/rme_cartesian_scope``

        ICT/Cartesian product scope for `late_rme_cartesian_hybrid` and `late_block_cartesian_projector`: `missing_only` or `all`.

    rme_ict_scope:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/rme_ict_scope``

        ICT/Cartesian product scope for `late_rme_cartesian_hybrid` and `late_block_cartesian_projector`: `missing_only` or `all`.

    cg_head_impl:
        | type: ``str``, optional, default: ``legacy``
        | argument path: ``shared_embedding_options/cg_head_impl``

        Reduction path for the h_b0 late-CG output head (`late_block_expansion_cg`). `legacy` (default) accumulates per-path contributions in the fixed Python-loop order used since 0715-refactor -- bit-stable with existing checkpoints/configs. `fused` opts in to a grouped-einsum + scatter reassociation of the same sum for a speedup not reproduced in-tree (external benchmark: ~4-5x on the head, ~-3.2% wall-clock per training iteration), at the cost of floating-point reassociation drift versus `legacy` (see late_block_expansion_cg.py for the certified tolerances and the cancellation-regime caveat). Falls back to `legacy` regardless of this setting under autocast, a non-fp32/64 dtype, or `use_deterministic_algorithms(True)`. Only meaningful when the resolved output_route is h_b0; ignored otherwise.

    ao_projector_channels:
        | type: ``int``, optional, default: ``0``
        | argument path: ``shared_embedding_options/ao_projector_channels``

        Direct AO-pair decoder multiplicity. `0` builds the complete ordered AO-pair representation with dimension max_norb^2; positive values are compressed ablations.

    ao_projector_normalization:
        | type: ``str``, optional, default: ``e3hamiltonian``
        | argument path: ``shared_embedding_options/ao_projector_normalization``

        AO-pair projector normalization. Currently `e3hamiltonian`.

    ao_projector_basis_convention:
        | type: ``str``, optional, default: ``deeptb_real_ao``
        | argument path: ``shared_embedding_options/ao_projector_basis_convention``

        AO-pair projector basis convention. Currently `deeptb_real_ao`.

    ao_projector_backend:
        | type: ``str``, optional, default: ``reference_wigner``
        | argument path: ``shared_embedding_options/ao_projector_backend``

        AO-pair projector source: `reference_wigner` or convention-checked `precomputed` bank.

    ao_projector_bank_path:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/ao_projector_bank_path``

        Path to projector bank JSON when ao_projector_backend=`precomputed`.

    res_update:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/res_update``

        Whether to use residual update.

    res_update_ratios:
        | type: ``float``, optional, default: ``0.5``
        | argument path: ``shared_embedding_options/res_update_ratios``

        The ratios of residual update, should in (0,1).

    norm_bottleneck_ratio:
        | type: ``float``, optional, default: ``0.1``
        | argument path: ``shared_embedding_options/norm_bottleneck_ratio``

        The ratios of norm bottle neck gate.

    res_update_ratios_learnable:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/res_update_ratios_learnable``

        Whether to make the ratios of residual update learnable.

    use_interpolation_out:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/use_interpolation_out``

        Set true to activate SO2 interpolation layer in the final output layer. Default: `False`

    so2_attn_aggressive:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/so2_attn_aggressive``

        Set true to activate SO2 attention radical mode. Default: `False`

    universal:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/universal``

        Set true to activate universal model related features. Currently, this will create a broader onehot embedding for the transfer learning into unseen elements. Other features are on the way. Default: `False`

    in_frame_flag:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/in_frame_flag``

    ln_flag:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/ln_flag``

    use_angle:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/use_angle``

        Whether to use angle.

    norm_eps:
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``shared_embedding_options/norm_eps``

        eps in SeperableLayerNorm.

    equivariant_norm_type:
        | type: ``str``, optional, default: ``none``
        | argument path: ``shared_embedding_options/equivariant_norm_type``

        Equivariant normalization on the flat irreps path. Supported: `none`, `merged_rms`.

    hidden_edge_activation_type:
        | type: ``str``, optional, default: ``gate``
        | argument path: ``shared_embedding_options/hidden_edge_activation_type``

        Activation used for hidden UpdateEdge blocks. Supported: `gate`, `swiglu_s2`.

    hidden_node_activation_type:
        | type: ``str``, optional, default: ``gate``
        | argument path: ``shared_embedding_options/hidden_node_activation_type``

        Activation used for hidden UpdateNode blocks. Supported: `gate`, `swiglu_s2`.

    swiglu_s2_grid_resolution:
        | type: ``list``, optional, default: ``[14, 14]``
        | argument path: ``shared_embedding_options/swiglu_s2_grid_resolution``

        Grid resolution `[lat, long]` for the flat SwiGLU-S2 adapter.

    swiglu_s2_compat_mode:
        | type: ``str``, optional, default: ``modern``
        | argument path: ``shared_embedding_options/swiglu_s2_compat_mode``

        Compatibility mode for hidden `swiglu_s2`. `modern` uses the new flexible layout; `legacy_uniform_only` preserves the old behavior that falls back to Gate when irreps multiplicities are not uniform across degrees.

    ffn_hidden_factor:
        | type: ``float``, optional, default: ``0.0``
        | argument path: ``shared_embedding_options/ffn_hidden_factor``

        Expansion factor for the optional node-wise equivariant FFN. Values `<= 1.0` disable it.

    ffn_apply_to_last:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/ffn_apply_to_last``

        Whether to also attach the node-wise FFN to the final layer. Default: `False`.

    so2_wigner_apply_mode:
        | type: ``str``, optional, default: ``compact_blocks``
        | argument path: ``shared_embedding_options/so2_wigner_apply_mode``

        Wigner rotation application mode for SO2 TP. Supported: `compact_blocks`, `full_dense`. Default uses compact per-l Wigner blocks to reduce peak memory; set `full_dense` to restore the previous dense Wigner path.

    so2_fusion_mode:
        | type: ``str``, optional, default: ``streamed_m_major_cueq``
        | argument path: ``shared_embedding_options/so2_fusion_mode``

        SO2_Linear fusion mode. Supported: `staged`, `streamed_m_major_ref`, `streamed_m_major_cueq`, `streamed_m_major_fused_p0`. The 0425-stable branch defaults to `streamed_m_major_cueq`; `streamed_m_major_fused_p0` is an opt-in trainable prototype that treats Wigner/R as constants and falls back on unsupported shapes.

    mole_linear_mode:
        | type: ``str`` | ``NoneType``, optional, default: ``cueq_indexed_linear``
        | argument path: ``shared_embedding_options/mole_linear_mode``

        MoLELinear backend. Supported: `split_loop`, `indexed_ref`, `cueq_indexed_linear`, `cublas_grouped`. The 0422-cueq-fastest branch defaults to `cueq_indexed_linear`.

    so2_m_linear_mode:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/so2_m_linear_mode``

        SO2 m-linear backend for non-MoE SO2 TP. Supported values are `standard`, `indexed_sandwich_multi`, or null; `cublas_grouped` is accepted only as a legacy alias. Triton experiment modes remain unsupported.

    so2_expert_mixing_mode:
        | type: ``str``, optional, default: ``pre_activation``
        | argument path: ``shared_embedding_options/so2_expert_mixing_mode``

        Expert mixing placement for SO2 MoE TP. `pre_activation` keeps the existing fused-weight path; `post_activation` evaluates raw expert TP outputs, applies equivariant activation, routes from 0e output scalars, and mixes activated outputs.

    so2_expert_route_chunk_size:
        | type: ``int`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/so2_expert_route_chunk_size``

        Maximum original SO2 rows processed per post-activation expert-mixing chunk. Null or non-positive means process all rows in one chunk.

    so2_expert_route_checkpoint:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/so2_expert_route_checkpoint``

        Whether to activation-checkpoint each post-activation expert-route chunk. This recomputes TP/activation/router during backward to reduce saved route activations.

    so2_output_router_hidden_dim:
        | type: ``int``, optional, default: ``32``
        | argument path: ``shared_embedding_options/so2_output_router_hidden_dim``

        Hidden size for the 0e router used by `so2_expert_mixing_mode=post_activation`.

    mole_linear_m0_mode:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/mole_linear_m0_mode``

        Legacy Triton route compatibility key. The 0425-stable branch accepts only `standard` or null; non-standard Triton values belong on the Triton experiment branch.

    onehot_tp_mode:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``shared_embedding_options/onehot_tp_mode``

        Backend for scalar onehot tensor products. The 0422-cueq-fastest branch supports only `scalar_fast`, storing a lightweight scalar-onehot module and applying TP as direct per-irrep scaling/mixing.

    node_message_aggregation:
        | type: ``str``, optional, default: ``scatter``
        | argument path: ``shared_embedding_options/node_message_aggregation``

        Node message aggregation mode. Supported: `scatter` for the legacy sum, `single_head_0e` for DPA4-style envelope-gated scalar attention.

    num_focus:
        | type: ``int``, optional, default: ``1``
        | argument path: ``shared_embedding_options/num_focus``

        Number of post-activation 0e focus gates. Values larger than 1 enable DPA4-style channel focus routing.

    focus_attention_dim:
        | type: ``int``, optional, default: ``32``
        | argument path: ``shared_embedding_options/focus_attention_dim``

        Hidden dimension of the single-head 0e attention query/key projections.

    edge_aggregation_gated_attention:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/edge_aggregation_gated_attention``

        Apply query-dependent sigmoid gating after edge-to-node aggregation, following the SDPA-output gated-attention pattern while preserving equivariant irrep groups. Default: `False`.

    edge_attention_key_source:
        | type: ``str``, optional, default: ``message``
        | argument path: ``shared_embedding_options/edge_attention_key_source``

        Key source for single-head edge attention. Currently supported: `message`, using post-activation edge message 0e scalars as keys. Default: `message`.

    edge_attention_envelope_power:
        | type: ``float``, optional, default: ``1.0``
        | argument path: ``shared_embedding_options/edge_attention_envelope_power``

        Power applied to cutoff coefficients in single-head edge attention numerator. `1.0` preserves the legacy implementation; `2.0` uses cutoff^2. Default: `1.0`.

    edge_attention_use_latent_bias:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/edge_attention_use_latent_bias``

        Whether to add latent-conditioned bias to single-head edge attention logits. Default: `True`, preserving the legacy implementation.

    edge_attention_key_layer_norm:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/edge_attention_key_layer_norm``

        Apply LayerNorm only to message 0e scalars before the single-head edge-attention key projection. Default: `False`.

    edge_attention_query_layer_norm:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/edge_attention_query_layer_norm``

        Apply LayerNorm only to destination node 0e scalars before the single-head edge-attention query projection. Default: `False`.

    edge_attention_qk_layer_norm:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``shared_embedding_options/edge_attention_qk_layer_norm``

        Shortcut that applies LayerNorm to both query and key 0e scalar inputs before the single-head edge-attention projections. Default: `False`.

    edge_message_env_weight:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/edge_message_env_weight``

        Whether to apply the legacy latent-conditioned env value weighting to node-update edge messages before aggregation. Default: `True`, preserving the legacy implementation.

    norm_build_node_condition_branch:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/norm_build_node_condition_branch``

        Whether to build the conditioned branch for node layer norm. Default: `True`

    norm_use_node_onehot:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/norm_use_node_onehot``

        Whether to use node one-hot as conditioning in node layer norm. Default: `True`

    norm_build_edge_condition_branch:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/norm_build_edge_condition_branch``

        Whether to build the conditioned branch for edge layer norm. Default: `True`

    norm_use_edge_onehot:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``shared_embedding_options/norm_use_edge_onehot``

        Whether to use edge one-hot embedding as conditioning in edge layer norm. Default: `True`

Method-specific embedding keys
------------------------------

* ``emoles``: shared options only

* ``emoles_openequi``: shared options only

* ``emoles_openequi_norm``: shared options only

* ``emoles_openequi_norm_v2``: shared options only

* ``emoles_openequi_eqv3``: shared options only

* ``emoles_openequi_eqv3_ffn``: shared options only

* ``emoles_openequi_nodeffn``: shared options only

* ``lem_moe_openequi``: shared options only

* ``lem_in_frame``: shared options only

* ``lem_in_frame_openequi``: shared options only

* ``lem_moe_v3``: shared options only

* ``lem_moe_v3_edge``: ``edge_router_in_features``, ``edge_router_unique_types``, ``edge_moe_compact_dispatch``, ``edge_moe_compact_min_edges``

* ``lem_moe_v3_h0``: ``h0_init_scope``, ``h0_node_key``, ``h0_edge_key``, ``h0_node_mode``, ``fallback_to_hamiltonian``, ``fallback_node_key``, ``fallback_edge_key``, ``allow_target_fallback_in_training``, ``use_uureal_residual_block_input``, ``use_spatial_residual_block_input``, ``h0_merge_mode``, ``h0_self_edge_tol``, ``use_flow_time_embedding``, ``flow_time_condition_edges``, ``flow_time_key``, ``flow_time_keys``, ``flow_time_max_positions``, ``flow_time_allow_missing``, ``flow_time_missing_value``, ``require_full_block_edge_coverage``, ``hb0_hermitian_average``, ``condition_source``, ``log_head_input_rms``, ``two_stage_pair_enable``, ``allow_no_h0_current_state``, ``two_stage_pair_refine_layers``, ``two_stage_pair_tail_gate``, ``two_stage_pair_refine_rank``, ``two_stage_pair_refine_condition``, ``two_stage_pair_refine_radial_dim``, ``two_stage_pair_refine_edge_chunk_size``

* ``lem_pair``: ``h0_init_scope``, ``h0_node_key``, ``h0_edge_key``, ``h0_node_mode``, ``fallback_to_hamiltonian``, ``fallback_node_key``, ``fallback_edge_key``, ``allow_target_fallback_in_training``, ``use_uureal_residual_block_input``, ``use_spatial_residual_block_input``, ``h0_merge_mode``, ``h0_self_edge_tol``, ``use_flow_time_embedding``, ``flow_time_condition_edges``, ``flow_time_key``, ``flow_time_keys``, ``flow_time_max_positions``, ``flow_time_allow_missing``, ``flow_time_missing_value``, ``require_full_block_edge_coverage``, ``hb0_hermitian_average``, ``condition_source``, ``log_head_input_rms``, ``two_stage_pair_enable``, ``allow_no_h0_current_state``, ``two_stage_pair_refine_layers``, ``two_stage_pair_tail_gate``, ``two_stage_pair_refine_rank``, ``two_stage_pair_refine_condition``, ``two_stage_pair_refine_radial_dim``, ``two_stage_pair_refine_edge_chunk_size``, ``mp_cutoff``, ``mp_avg_num_neighbors``, ``res_update_additive``, ``latents_layernorm``, ``pair_refine_enable``, ``pair_refine_rank``, ``pair_refine_condition``, ``pair_refine_internal_weights``, ``pair_refine_init``, ``pair_refine_weight_mode``, ``pair_refine_max_weight_numel``, ``pair_refine_identity_init``

* ``lem_moe_v3_prior``: ``prior_init_scope``, ``prior_kind``, ``prior_node_key``, ``prior_edge_key``, ``prior_node_mode``, ``prior_merge_mode``, ``prior_self_edge_tol``, ``soft_edge_memory``, ``prior_validate_inputs``

* ``lem_moe_v3_edge_h0``: ``h0_init_scope``, ``h0_node_key``, ``h0_edge_key``, ``h0_node_mode``, ``fallback_to_hamiltonian``, ``fallback_node_key``, ``fallback_edge_key``, ``allow_target_fallback_in_training``, ``use_uureal_residual_block_input``, ``use_spatial_residual_block_input``, ``h0_merge_mode``, ``h0_self_edge_tol``, ``use_flow_time_embedding``, ``flow_time_condition_edges``, ``flow_time_key``, ``flow_time_keys``, ``flow_time_max_positions``, ``flow_time_allow_missing``, ``flow_time_missing_value``, ``require_full_block_edge_coverage``, ``hb0_hermitian_average``, ``condition_source``, ``log_head_input_rms``, ``two_stage_pair_enable``, ``allow_no_h0_current_state``, ``two_stage_pair_refine_layers``, ``two_stage_pair_tail_gate``, ``two_stage_pair_refine_rank``, ``two_stage_pair_refine_condition``, ``two_stage_pair_refine_radial_dim``, ``two_stage_pair_refine_edge_chunk_size``, ``edge_router_in_features``, ``edge_router_unique_types``, ``edge_moe_compact_dispatch``, ``edge_moe_compact_min_edges``

* ``lem_non_linear``: shared options only

* ``lem_non_linear_h0``: ``h0_init_scope``, ``h0_node_key``, ``h0_edge_key``, ``h0_node_mode``, ``fallback_to_hamiltonian``, ``fallback_node_key``, ``fallback_edge_key``, ``allow_target_fallback_in_training``, ``use_uureal_residual_block_input``, ``use_spatial_residual_block_input``, ``h0_merge_mode``, ``h0_self_edge_tol``, ``use_flow_time_embedding``, ``flow_time_condition_edges``, ``flow_time_key``, ``flow_time_keys``, ``flow_time_max_positions``, ``flow_time_allow_missing``, ``flow_time_missing_value``, ``require_full_block_edge_coverage``, ``hb0_hermitian_average``, ``condition_source``, ``log_head_input_rms``, ``two_stage_pair_enable``, ``allow_no_h0_current_state``, ``two_stage_pair_refine_layers``, ``two_stage_pair_tail_gate``, ``two_stage_pair_refine_rank``, ``two_stage_pair_refine_condition``, ``two_stage_pair_refine_radial_dim``, ``two_stage_pair_refine_edge_chunk_size``

Embedding extension option reference
------------------------------------

embedding_extension_options:
    | type: ``dict``, optional
    | argument path: ``embedding_extension_options``

    edge_router_in_features:
        | type: ``int`` | ``NoneType``, optional, default: ``None``
        | argument path: ``embedding_extension_options/edge_router_in_features``

        Input dimension for the edge-wise MoE router. Defaults to `edge_one_hot_dim`.

    edge_router_unique_types:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/edge_router_unique_types``

        For edge-wise MoE, route unique active bond types once and map them back to active edges. Default: `True`.

    edge_moe_compact_dispatch:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/edge_moe_compact_dispatch``

        For edge-wise MoE with unique-type routing, enable grouped compact dispatch for large-edge batches. Default: `True`.

    edge_moe_compact_min_edges:
        | type: ``int``, optional, default: ``16384``
        | argument path: ``embedding_extension_options/edge_moe_compact_min_edges``

        Minimum active-edge count before grouped compact dispatch is used. Default: `16384`.

    h0_init_scope:
        | type: ``str``, optional, default: ``both``
        | argument path: ``embedding_extension_options/h0_init_scope``

        H0 initialization scope: both, node, edge, auxiliary, or none. Default: both.

    h0_node_key:
        | type: ``str``, optional, default: ``node_h0``
        | argument path: ``embedding_extension_options/h0_node_key``

        Node-wise H0 key. Defaults to `node_h0`. When absent and fallback is enabled, the plugin checks `node_hamiltonian` first and then the configured fallback feature key.

    h0_edge_key:
        | type: ``str``, optional, default: ``edge_h0``
        | argument path: ``embedding_extension_options/h0_edge_key``

        Edge-wise H0 key. Defaults to `edge_h0`. When absent and fallback is enabled, the plugin checks `edge_hamiltonian` first and then the configured fallback feature key.

    h0_node_mode:
        | type: ``str``, optional, default: ``direct``
        | argument path: ``embedding_extension_options/h0_node_mode``

        How to build node init from H0. Supported: `direct`, `self_edge`. Default: `direct`.

    fallback_to_hamiltonian:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/fallback_to_hamiltonian``

        Whether to fall back to the LMDB Hamiltonian-derived node/edge features when explicit H0 keys are absent. Default: `True`.

    fallback_node_key:
        | type: ``str``, optional, default: ``node_features``
        | argument path: ``embedding_extension_options/fallback_node_key``

        Fallback node key used when explicit H0 is absent. Default: `node_features`.

    fallback_edge_key:
        | type: ``str``, optional, default: ``edge_features``
        | argument path: ``embedding_extension_options/fallback_edge_key``

        Fallback edge key used when explicit H0 is absent. Default: `edge_features`.

    allow_target_fallback_in_training:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/allow_target_fallback_in_training``

        Permit the H0 input fallback to resolve to the target Hamiltonian/features while the module is in training mode. Off by default: that fallback is a label leak during training (it is a deliberate surrogate only at inference).

    use_uureal_residual_block_input:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/use_uureal_residual_block_input``

        Enable the mapper-derived bias-free residual AO-block projector.

    use_spatial_residual_block_input:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/use_spatial_residual_block_input``

        Enable the non-SOC direct-residual (spatial) AO-block projector for residual_ao_block_ode.

    h0_merge_mode:
        | type: ``str``, optional, default: ``replace``
        | argument path: ``embedding_extension_options/h0_merge_mode``

        How to combine H0-projected features with the base init output. Supported: `replace`, `add`. Default: `replace`.

    h0_self_edge_tol:
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``embedding_extension_options/h0_self_edge_tol``

        Tolerance used to detect self-edges in `self_edge` node mode. Default: `1e-8`.

    use_flow_time_embedding:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/use_flow_time_embedding``

        Whether to inject graph-level flow time into scalar channels before message passing. Default: `False`.

    flow_time_condition_edges:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/flow_time_condition_edges``

        Whether to also inject graph-level flow time into active edge scalar channels when flow-time embedding is enabled. Default: `True`.

    flow_time_key:
        | type: ``str``, optional, default: ``flow_time``
        | argument path: ``embedding_extension_options/flow_time_key``

        Graph-level flow time key written by train_options.flow_options. Default: `flow_time`.

    flow_time_keys:
        | type: ``list``, optional, default: ``[]``
        | argument path: ``embedding_extension_options/flow_time_keys``

        Optional list of graph-level time keys to embed and sum, e.g. [`flow_time_t`, `flow_time_r`, `flow_time_h`] for Pixel MeanFlow.

    flow_time_max_positions:
        | type: ``int``, optional, default: ``2000``
        | argument path: ``embedding_extension_options/flow_time_max_positions``

        Scale used by the sinusoidal flow-time embedding. Default: `2000`.

    flow_time_allow_missing:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/flow_time_allow_missing``

        Whether missing flow time may fall back to flow_time_missing_value. Default: `True`; block-ODE requires `False`.

    flow_time_missing_value:
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``embedding_extension_options/flow_time_missing_value``

        Fallback normalized time when flow_time is absent. Default: `0.0`.

    require_full_block_edge_coverage:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/require_full_block_edge_coverage``

        Fail before the H-B0 head unless its actual active rows are the ordered full graph-edge range with finite positive cutoff coefficients. Default: `False`; block-ODE requires `True`.

    hb0_hermitian_average:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/hb0_hermitian_average``

        Transpose-average directed reverse H-B0 edge blocks at the embedding output boundary.

    condition_source:
        | type: ``str``, optional, default: ``edge_0e``
        | argument path: ``embedding_extension_options/condition_source``

        H-B0 edge-head conditioner source: edge_0e or endpoints.

    log_head_input_rms:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/log_head_input_rms``

        Attach detached per-irrep-slice node/edge head-input RMS tensors to model output.

    two_stage_pair_enable:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/two_stage_pair_enable``

    allow_no_h0_current_state:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/allow_no_h0_current_state``

        Allow the absolute Full-H block-ODE route to supply its current state through node/edge features without physical-H0 feature init.

    two_stage_pair_refine_layers:
        | type: ``int``, optional, default: ``2``
        | argument path: ``embedding_extension_options/two_stage_pair_refine_layers``

    two_stage_pair_tail_gate:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/two_stage_pair_tail_gate``

    two_stage_pair_refine_rank:
        | type: ``int``, optional, default: ``16``
        | argument path: ``embedding_extension_options/two_stage_pair_refine_rank``

    two_stage_pair_refine_condition:
        | type: ``str``, optional, default: ``scalar_0e``
        | argument path: ``embedding_extension_options/two_stage_pair_refine_condition``

    two_stage_pair_refine_radial_dim:
        | type: ``int``, optional, default: ``4``
        | argument path: ``embedding_extension_options/two_stage_pair_refine_radial_dim``

    two_stage_pair_refine_edge_chunk_size:
        | type: ``int``, optional, default: ``64``
        | argument path: ``embedding_extension_options/two_stage_pair_refine_edge_chunk_size``

    mp_cutoff:
        | type: ``int`` | ``dict`` | ``float``, optional
        | argument path: ``embedding_extension_options/mp_cutoff``

    mp_avg_num_neighbors:
        | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
        | argument path: ``embedding_extension_options/mp_avg_num_neighbors``

    res_update_additive:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/res_update_additive``

        Use unscaled x + delta residual updates in the pair backbone.

    latents_layernorm:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/latents_layernorm``

        Apply LayerNorm before pair-backbone latent updates.

    pair_refine_enable:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/pair_refine_enable``

    pair_refine_rank:
        | type: ``int``, optional, default: ``16``
        | argument path: ``embedding_extension_options/pair_refine_rank``

    pair_refine_condition:
        | type: ``str``, optional, default: ``scalar_0e``
        | argument path: ``embedding_extension_options/pair_refine_condition``

    pair_refine_internal_weights:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``embedding_extension_options/pair_refine_internal_weights``

    pair_refine_init:
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``embedding_extension_options/pair_refine_init``

    pair_refine_weight_mode:
        | type: ``str``, optional, default: ``full``
        | argument path: ``embedding_extension_options/pair_refine_weight_mode``

        Dynamic TP weights: legacy `full`, instruction-gated `per_path`, or channel-diagonal `qhflow`.

    pair_refine_max_weight_numel:
        | type: ``int`` | ``NoneType``, optional, default: ``None``
        | argument path: ``embedding_extension_options/pair_refine_max_weight_numel``

        Optional constructor guard on the full FCTP weight count.

    pair_refine_identity_init:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/pair_refine_identity_init``

        Zero dynamic and static refinement weights at initialization.

    prior_init_scope:
        | type: ``str``, optional, default: ``both``
        | argument path: ``embedding_extension_options/prior_init_scope``

        Physical-prior initialization scope: both, node, edge, auxiliary, or none.

    prior_kind:
        | type: ``str``, optional, default: ``p2``
        | argument path: ``embedding_extension_options/prior_kind``

        Physical prior family: p2 or p23. This single value derives the node/edge RME fields, AO-block fields and label.

    prior_node_key:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``embedding_extension_options/prior_node_key``

        Deprecated/optional: node-wise selected-prior RME field. Leave empty to derive from prior_kind (node_p2/node_p23); an explicit value must match the derived one.

    prior_edge_key:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``embedding_extension_options/prior_edge_key``

        Deprecated/optional: edge-wise selected-prior RME field. Leave empty to derive from prior_kind (edge_p2/edge_p23); an explicit value must match the derived one.

    prior_node_mode:
        | type: ``str``, optional, default: ``direct``
        | argument path: ``embedding_extension_options/prior_node_mode``

        Supported: direct or self_edge.

    prior_merge_mode:
        | type: ``str``, optional, default: ``replace``
        | argument path: ``embedding_extension_options/prior_merge_mode``

        Supported: replace or add.

    prior_self_edge_tol:
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``embedding_extension_options/prior_self_edge_tol``

    soft_edge_memory:
        | type: ``dict``, optional, default: ``{'enabled': True}``
        | argument path: ``embedding_extension_options/soft_edge_memory``

        Scalar-only equivariant edge-memory configuration.

        enabled:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``embedding_extension_options/soft_edge_memory/enabled``

        num_slots:
            | type: ``int``, optional, default: ``64``
            | argument path: ``embedding_extension_options/soft_edge_memory/num_slots``

        num_heads:
            | type: ``int``, optional, default: ``4``
            | argument path: ``embedding_extension_options/soft_edge_memory/num_heads``

        head_dim:
            | type: ``int``, optional, default: ``16``
            | argument path: ``embedding_extension_options/soft_edge_memory/head_dim``

        temperature:
            | type: ``float``, optional, default: ``1.0``
            | argument path: ``embedding_extension_options/soft_edge_memory/temperature``

        dropout:
            | type: ``float``, optional, default: ``0.0``
            | argument path: ``embedding_extension_options/soft_edge_memory/dropout``

        gate_mode:
            | type: ``str``, optional, default: ``deepseek``
            | argument path: ``embedding_extension_options/soft_edge_memory/gate_mode``

        gate_bias:
            | type: ``float``, optional, default: ``0.0``
            | argument path: ``embedding_extension_options/soft_edge_memory/gate_bias``

        gate_eps:
            | type: ``float``, optional, default: ``1e-06``
            | argument path: ``embedding_extension_options/soft_edge_memory/gate_eps``

        zero_init_output:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``embedding_extension_options/soft_edge_memory/zero_init_output``

        input_norm:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``embedding_extension_options/soft_edge_memory/input_norm``

        diagnostics_mode:
            | type: ``str``, optional, default: ``off``
            | argument path: ``embedding_extension_options/soft_edge_memory/diagnostics_mode``

        diagnostics_sample_size:
            | type: ``int``, optional, default: ``1024``
            | argument path: ``embedding_extension_options/soft_edge_memory/diagnostics_sample_size``

    prior_validate_inputs:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``embedding_extension_options/prior_validate_inputs``

        Debug-only finite checks for P2 tensors during every forward; production LMDBs are validated at ingest.

Prediction methods
------------------

* ``e3tb``
* ``block_native``

e3tb prediction options
~~~~~~~~~~~~~~~~~~~~~~~

e3tb:
    | type: ``dict``
    | argument path: ``e3tb``

    neural network options for prediction model.

    scales_trainable:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/scales_trainable``

        The scale parameter is from the statistics. Whether to train this parameter.

    shifts_trainable:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/shifts_trainable``

        The scale parameter is from the statistics. Whether to train this parameter.

    neurons:
        | type: ``list`` | ``NoneType``, optional, default: ``None``
        | argument path: ``e3tb/neurons``

        neurons in the neural network.

    activation:
        | type: ``str``, optional, default: ``tanh``
        | argument path: ``e3tb/activation``

        activation function.

    scale_type:
        | type: ``str``, optional, default: ``scale_w_back_grad``
        | argument path: ``e3tb/scale_type``

        Which scale method to use. Can be no_scale, scale_wo_back_grad (the scale parameter will not engage the back grad computation graph), scale_w_back_grad (the scale parameter will engage the back grad computation graph)

    if_batch_normalized:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/if_batch_normalized``

        if to turn on batch normalization

    blockwise_hamiltonian:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/blockwise_hamiltonian``

        If true, materialize E3 Hamiltonian feature predictions into AO block tensors for block-wise loss. This is non-SOC AO/block supervision, not a block-native head.

    node_pad_shape:
        | type: ``list`` | ``NoneType``, optional, default: ``None``
        | argument path: ``e3tb/node_pad_shape``

        Padded node AO block shape for blockwise Hamiltonian output.

    edge_pad_shape:
        | type: ``list`` | ``NoneType``, optional, default: ``None``
        | argument path: ``e3tb/edge_pad_shape``

        Padded edge AO block shape for blockwise Hamiltonian output.

    symmetrize_onsite:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``e3tb/symmetrize_onsite``

        Hermitian-complete onsite AO blocks in blockwise output.

    complete_edges:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``e3tb/complete_edges``

        Fill missing edge AO entries from reverse directed edges in blockwise output.

    strict_complete_edges:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/strict_complete_edges``

        Fail if reverse-edge completion leaves unresolved valid AO entries.

    reconstruction:
        | type: ``str``, optional, default: ``direct``
        | argument path: ``e3tb/reconstruction``

        Full-H reconstruction mode: direct, h0_residual, or prior_residual. Replaces the mutually exclusive add_h0/add_prior flags.

    prior_node_block_field:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``e3tb/prior_node_block_field``

        Deprecated/optional: node AO-block field used for physical-prior Full-H reconstruction. Leave empty to derive from the embedding prior_kind (node_p2_blocks/node_p23_blocks); an explicit value must match the derived one.

    prior_edge_block_field:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``e3tb/prior_edge_block_field``

        Deprecated/optional: edge AO-block field used for physical-prior Full-H reconstruction. Leave empty to derive from the embedding prior_kind (edge_p2_blocks/edge_p23_blocks); an explicit value must match the derived one.

    prior_label:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``e3tb/prior_label``

        Deprecated/optional: human label for the physical prior. Leave empty to derive from prior_kind (P2/P23).

    validate_prior_blocks:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``e3tb/validate_prior_blocks``

        Debug-only finite checks for prior AO blocks during every model forward. Production caches are validated at dataset ingest.

    full_output_node_field:
        | type: ``str``, optional, default: ``node_full_hamil_blocks``
        | argument path: ``e3tb/full_output_node_field``

        Output key for reconstructed Full-H node blocks in a residual reconstruction mode.

    full_output_edge_field:
        | type: ``str``, optional, default: ``edge_full_hamil_blocks``
        | argument path: ``e3tb/full_output_edge_field``

        Output key for reconstructed Full-H edge blocks in a residual reconstruction mode.

block_native prediction options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

block_native:
    | type: ``dict``
    | argument path: ``block_native``

    neural network options for prediction model.

    scale_type:
        | type: ``str``, optional, default: ``no_scale``
        | argument path: ``block_native/scale_type``

        Block-native decoder bypasses RME scale/shift and E3Hamiltonian; use no_scale.

    block_decoder:
        | type: ``str``, optional, default: ``linear``
        | argument path: ``block_native/block_decoder``

        Block-native decoder backend: `linear`, `expansion_cg`, `cartesian_projector`, or `ao_projector`.

    blockwise_hamiltonian:
        | type: ``bool``, optional, default: ``True``
        | argument path: ``block_native/blockwise_hamiltonian``

        Whether the downstream consumer expects explicit AO Hamiltonian blocks.

    reconstruction:
        | type: ``str``, optional, default: ``direct``
        | argument path: ``block_native/reconstruction``

        Full-H reconstruction mode: direct, h0_residual, or prior_residual.

    prior_node_block_field:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``block_native/prior_node_block_field``

        Deprecated/optional: leave empty to derive from the embedding prior_kind (node_p2_blocks/node_p23_blocks); an explicit value must match the derived one.

    prior_edge_block_field:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``block_native/prior_edge_block_field``

        Deprecated/optional: leave empty to derive from the embedding prior_kind (edge_p2_blocks/edge_p23_blocks); an explicit value must match the derived one.

    prior_label:
        | type: ``str``, optional, default: (empty string)
        | argument path: ``block_native/prior_label``

        Deprecated/optional: leave empty to derive from prior_kind (P2/P23).

    validate_prior_blocks:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``block_native/validate_prior_blocks``

        Debug-only finite checks for prior AO blocks during every model forward. Production caches are validated at dataset ingest.

    full_output_node_field:
        | type: ``str``, optional, default: ``node_full_hamil_blocks``
        | argument path: ``block_native/full_output_node_field``

        Output key for reconstructed Full-H node blocks in a residual reconstruction mode.

    full_output_edge_field:
        | type: ``str``, optional, default: ``edge_full_hamil_blocks``
        | argument path: ``block_native/full_output_edge_field``

        Output key for reconstructed Full-H edge blocks in a residual reconstruction mode.
