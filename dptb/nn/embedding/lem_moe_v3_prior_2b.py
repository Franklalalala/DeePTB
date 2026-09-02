# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-stage prior embedding: concat(P-mapped, geo) into the first SO2 layer.

Stage 1 (``only2b=true``) and stage 2 (``only2b=false``) both:

- map P RME with the same ``H0InitLayer`` projectors as ``lem_moe_v3_prior``
- **concat** (not replace) those features with geometric InitLayer features
- run SO2 + scatter layers on the concatenated tensor
- predict residual RME whose label is Full-H − P (the dataset, not an add-back)

Stage 2 freezes the 2b skip (concat-init Linear plus P projectors and geo
InitLayer) so ``ŷ = ŷ_2b + ŷ_GNN`` keeps a stable H-space baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from e3nn import o3
from e3nn.o3 import Linear
from torch_scatter import scatter_mean

from dptb.configuration import resolve_init_scope
from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3_h0 import LemMoEV3H0
from .lem_moe_v3_h0_helpers import H0InitLayer, _get_feature_source_with_key


PRIOR_2B_KINDS = {
    "p2": (_keys.NODE_P2_KEY, _keys.EDGE_P2_KEY),
    "p23": (_keys.NODE_P23_KEY, _keys.EDGE_P23_KEY),
    "na_cf": (_keys.NODE_P23_KEY, _keys.EDGE_P2_KEY),
}


def resolve_prior_2b_keys(prior_kind: str) -> Tuple[str, str]:
    kind = str(prior_kind).strip().lower()
    if kind not in PRIOR_2B_KINDS:
        raise ValueError(
            "lem_moe_v3_prior_2b supports prior_kind='p2', 'p23', or 'na_cf'; "
            f"got {prior_kind!r}."
        )
    return PRIOR_2B_KINDS[kind]


def _unwrap_h0_init(init_layer: torch.nn.Module) -> H0InitLayer:
    if isinstance(init_layer, H0InitLayer):
        return init_layer
    prior_init = getattr(init_layer, "prior_init", None)
    if isinstance(prior_init, H0InitLayer):
        return prior_init
    raise TypeError(
        "lem_moe_v3_prior_2b expected H0InitLayer (optionally wrapped); got "
        f"{type(init_layer).__name__}."
    )


@Embedding.register("lem_moe_v3_prior_2b")
class LemMoEV3Prior2b(LemMoEV3H0):
    """Concat-P first SO2 layer + frozen 2b RME skip for two-stage training."""

    def __init__(
        self,
        *,
        only2b: bool = False,
        prior_kind: str = "na_cf",
        prior_init_scope: Optional[str] = None,
        use_prior_init: Optional[bool] = None,
        prior_node_key: str = "",
        prior_edge_key: str = "",
        use_prior_node_init: Optional[bool] = None,
        use_prior_edge_init: Optional[bool] = None,
        prior_node_mode: str = "direct",
        prior_merge_mode: str = "concat",
        prior_self_edge_tol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        merge = str(prior_merge_mode).strip().lower()
        if merge not in {"concat", "add"}:
            raise ValueError(
                "lem_moe_v3_prior_2b requires prior_merge_mode='concat' "
                f"(or 'add'); got {prior_merge_mode!r}. replace is refused."
            )
        if merge == "add":
            raise ValueError(
                "lem_moe_v3_prior_2b uses concat(geo, P-map) as the first "
                "SO2-layer input; prior_merge_mode='add' is not this route. "
                "Use 'concat'."
            )

        node_key, edge_key = resolve_prior_2b_keys(prior_kind)
        if prior_node_key not in (None, "") and str(prior_node_key) != node_key:
            raise ValueError(
                f"prior_kind={prior_kind!r} requires prior_node_key={node_key!r}; "
                f"got {prior_node_key!r}."
            )
        if prior_edge_key not in (None, "") and str(prior_edge_key) != edge_key:
            raise ValueError(
                f"prior_kind={prior_kind!r} requires prior_edge_key={edge_key!r}; "
                f"got {prior_edge_key!r}."
            )

        (
            self.prior_init_scope,
            use_prior_init,
            use_prior_node_init,
            use_prior_edge_init,
        ) = resolve_init_scope(
            prior_init_scope,
            enabled=use_prior_init,
            node=use_prior_node_init,
            edge=use_prior_edge_init,
            option_name="prior_init_scope",
        )
        if not use_prior_init:
            raise ValueError(
                "lem_moe_v3_prior_2b requires prior_init_scope != 'none' "
                "so stage 1 already consumes P RME."
            )

        self._prior2b_layer_kwargs = {
            "tp_radial_emb": kwargs.get("tp_radial_emb", False),
            "tp_radial_channels": kwargs.get("tp_radial_channels", [128, 128]),
            "latent_channels": kwargs.get("latent_channels", [128, 128]),
            "use_layer_onehot_tp": kwargs.get("use_layer_onehot_tp", True),
            "edge_one_hot_dim": kwargs.get("edge_one_hot_dim", 128),
            "res_update": kwargs.get("res_update", True),
            "res_update_ratios": kwargs.get("res_update_ratios", None),
            "res_update_ratios_learnable": kwargs.get(
                "res_update_ratios_learnable", False
            ),
            "equivariant_norm_type": kwargs.get("equivariant_norm_type", "none"),
            "swiglu_s2_grid_resolution": kwargs.get(
                "swiglu_s2_grid_resolution", (14, 14)
            ),
            "swiglu_s2_compat_mode": kwargs.get("swiglu_s2_compat_mode", "modern"),
            "ffn_hidden_factor": kwargs.get("ffn_hidden_factor", 0.0),
            "so2_wigner_apply_mode": kwargs.get(
                "so2_wigner_apply_mode", "compact_blocks"
            ),
            "so2_fusion_mode": kwargs.get(
                "so2_fusion_mode", "streamed_m_major_cueq"
            ),
            "mole_linear_mode": kwargs.get(
                "mole_linear_mode", "cueq_indexed_linear"
            ),
            "so2_expert_mixing_mode": kwargs.get(
                "so2_expert_mixing_mode", "pre_activation"
            ),
            "so2_expert_route_chunk_size": kwargs.get(
                "so2_expert_route_chunk_size", None
            ),
            "so2_expert_route_checkpoint": kwargs.get(
                "so2_expert_route_checkpoint", False
            ),
            "so2_output_router_hidden_dim": kwargs.get(
                "so2_output_router_hidden_dim", 32
            ),
            "onehot_tp_mode": kwargs.get("onehot_tp_mode", None),
            "node_message_aggregation": kwargs.get(
                "node_message_aggregation", "scatter"
            ),
            "num_focus": kwargs.get("num_focus", 1),
            "focus_attention_dim": kwargs.get("focus_attention_dim", 32),
            "edge_aggregation_gated_attention": kwargs.get(
                "edge_aggregation_gated_attention", False
            ),
            "edge_attention_key_source": kwargs.get(
                "edge_attention_key_source", "message"
            ),
            "edge_attention_envelope_power": kwargs.get(
                "edge_attention_envelope_power", 1.0
            ),
            "edge_attention_use_latent_bias": kwargs.get(
                "edge_attention_use_latent_bias", True
            ),
            "edge_attention_key_layer_norm": kwargs.get(
                "edge_attention_key_layer_norm", False
            ),
            "edge_attention_query_layer_norm": kwargs.get(
                "edge_attention_query_layer_norm", False
            ),
            "edge_attention_qk_layer_norm": kwargs.get(
                "edge_attention_qk_layer_norm", False
            ),
            "edge_message_env_weight": kwargs.get("edge_message_env_weight", True),
            "norm_eps": kwargs.get("norm_eps", 1e-8),
            "num_shared_experts": kwargs.get("num_shared_experts", 1),
        }
        n_layers = int(kwargs.get("n_layers", 3))
        self._prior2b_first_is_last = n_layers == 1
        self._prior2b_use_interpolation = bool(
            kwargs.get("use_interpolation_out", True)
        )

        super().__init__(
            h0_init_scope=self.prior_init_scope,
            h0_node_key=node_key,
            h0_edge_key=edge_key,
            h0_node_mode=prior_node_mode,
            fallback_to_hamiltonian=False,
            allow_target_fallback_in_training=False,
            h0_merge_mode="add",
            h0_self_edge_tol=prior_self_edge_tol,
            **kwargs,
        )
        if bool(getattr(self.idp, "has_soc", False)):
            raise NotImplementedError(
                "lem_moe_v3_prior_2b is non-SOC; residual Full-H − P is real RME."
            )

        self.only2b = bool(only2b)
        self.prior_kind = str(prior_kind).strip().lower()
        self.prior_node_key = node_key
        self.prior_edge_key = edge_key
        self.prior_merge_mode = "concat"
        self.h0_init = _unwrap_h0_init(self.init_layer)
        geo_irreps = o3.Irreps(self.h0_init.irreps_out)
        self.concat_irreps = geo_irreps + geo_irreps
        self._rebuild_first_layer_for_concat()
        self.two_b_out_node = Linear(
            self.concat_irreps,
            self.idp.orbpair_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.two_b_out_edge = Linear(
            self.concat_irreps,
            self.idp.orbpair_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        if not self.only2b:
            self._freeze_two_b_skip()

    def _rebuild_first_layer_for_concat(self) -> None:
        old = self.layers[0]
        hidden_act = "gate" if self._prior2b_first_is_last else "gate"
        use_interpolation_tp = bool(
            self._prior2b_first_is_last
            and self._prior2b_use_interpolation
            and getattr(self.output_route_spec, "final_irreps_kind", "") == "orbpair"
        )
        ffn_hidden_factor = float(self._prior2b_layer_kwargs["ffn_hidden_factor"])
        use_node_ffn = ffn_hidden_factor > 1.0 and self._prior2b_first_is_last
        new_layer = self._layer_type()(
            num_types=self.n_atom,
            avg_num_neighbors=old.avg_num_neighbors,
            irreps_in=self.concat_irreps,
            irreps_out=old.irreps_out,
            latent_dim=self.latent_dim,
            use_interpolation_tp=use_interpolation_tp,
            edge_activation_type=hidden_act,
            node_activation_type=hidden_act,
            use_node_ffn=use_node_ffn,
            dtype=self.dtype,
            device=self.device,
            num_experts=self.num_experts,
            **{
                k: v
                for k, v in self._prior2b_layer_kwargs.items()
                if k != "ffn_hidden_factor"
            },
            ffn_hidden_factor=ffn_hidden_factor,
        )
        self.layers[0] = new_layer

    def _freeze_two_b_skip(self) -> None:
        for module in (
            self.two_b_out_node,
            self.two_b_out_edge,
            self.h0_init.node_projector,
            self.h0_init.edge_projector,
            self.h0_init.base_init,
        ):
            for param in module.parameters():
                param.requires_grad = False

    def _project_prior(
        self,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        active_edges: torch.Tensor,
        n_nodes: int,
        n_active: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.h0_init
        node_source, node_key = _get_feature_source_with_key(
            data=data,
            candidate_keys=[self.prior_node_key],
            expected_dim=h0.h0_dim,
            dtype=self.dtype,
            device=self.device,
            label="node prior",
        )
        if node_source is None:
            raise KeyError(
                f"lem_moe_v3_prior_2b requires node field {self.prior_node_key!r}."
            )
        h0._guard_target_fallback(node_key, self.prior_node_key, "node prior")
        node_source = h0._mask_node_source(node_source, atom_type)
        node_source = node_source.index_select(1, h0._h0_sort_index)
        node_p = h0.node_projector(node_source)
        if node_p.shape[0] != n_nodes:
            node_p = h0._align_feature_rows(node_p, n_nodes)

        edge_source, edge_key = _get_feature_source_with_key(
            data=data,
            candidate_keys=[self.prior_edge_key],
            expected_dim=h0.h0_dim,
            dtype=self.dtype,
            device=self.device,
            label="edge prior",
        )
        if edge_source is None:
            raise KeyError(
                f"lem_moe_v3_prior_2b requires edge field {self.prior_edge_key!r}."
            )
        h0._guard_target_fallback(edge_key, self.prior_edge_key, "edge prior")
        edge_source = h0._mask_edge_source(edge_source, bond_type)
        edge_source = edge_source.index_select(1, h0._h0_sort_index)
        edge_p = h0.edge_projector(edge_source[active_edges])
        if edge_p.shape[0] != n_active:
            edge_p = h0._align_feature_rows(edge_p, n_active)
        return node_p, edge_p

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        preserved_split_sizes = data.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if preserved_split_sizes is not None:
            data = data.copy()
            data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)
        if preserved_split_sizes is not None:
            data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = preserved_split_sizes

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        global_feat = scatter_mean(node_one_hot, batch, dim=0)
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)
        topk_indices, topk_values = self.router.last_topk()
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv

        num_nodes_total = node_one_hot.shape[0]
        precomputed_active_edges = data.get(_keys.LEM_ACTIVE_EDGES_KEY, None)
        precomputed_cutoff_coeffs = data.get(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        precomputed_split_sizes = data.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if precomputed_cutoff_coeffs is not None and edge_length.requires_grad:
            raise RuntimeError(
                "Precomputed LEM cutoff coefficients cannot be used when "
                "edge_length requires gradients."
            )

        latents, geo_node, geo_edge, cutoff_coeffs, active_edges = (
            self.h0_init.base_init(
                edge_index,
                atom_type,
                bond_type,
                edge_sh,
                edge_length,
                edge_one_hot,
                precomputed_active_edges,
                precomputed_cutoff_coeffs,
            )
        )
        prior_node, prior_edge = self._project_prior(
            data,
            edge_index,
            atom_type,
            bond_type,
            active_edges,
            geo_node.shape[0],
            geo_edge.shape[0],
        )
        node_features = torch.cat([geo_node, prior_node], dim=-1)
        edge_features = torch.cat([geo_edge, prior_edge], dim=-1)
        if node_features.shape[-1] != self.concat_irreps.dim:
            raise RuntimeError(
                "concat(geo, P) width "
                f"{node_features.shape[-1]} != concat_irreps.dim "
                f"{self.concat_irreps.dim}."
            )

        y2b_node = self.two_b_out_node(node_features)
        y2b_edge = self.two_b_out_edge(edge_features)

        node_batch = batch[: node_features.shape[0]]
        if node_features.shape[0] < num_nodes_total:
            safe_node_one_hot = node_one_hot[: node_features.shape[0]]
        else:
            safe_node_one_hot = node_one_hot
        edge_one_hot = edge_one_hot[active_edges]
        if precomputed_split_sizes is not None:
            mole_globals = MOLEGlobals(
                coefficients=coeffs,
                split_sizes=precomputed_split_sizes,
                topk_indices=topk_indices,
                topk_values=topk_values,
            )
        else:
            edge_batch = batch[edge_index[0][active_edges]]
            num_systems = coeffs.shape[0]
            edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
            mole_globals = MOLEGlobals(
                coefficients=coeffs,
                sizes=edge_sizes,
                graph_index=edge_batch,
                topk_indices=topk_indices,
                topk_values=topk_values,
            )

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None
        for layer in self.layers:
            latents, node_features, edge_features, wigner_D_all = layer(
                latents,
                node_features,
                edge_features,
                safe_node_one_hot,
                edge_index,
                edge_vector,
                atom_type,
                cutoff_coeffs,
                active_edges,
                edge_one_hot,
                wigner_D_all,
                mole_globals,
                node_batch,
            )

        if node_features.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - node_features.shape[0]
            pad = torch.zeros(
                pad_num,
                node_features.shape[1],
                device=node_features.device,
                dtype=node_features.dtype,
            )
            node_features = torch.cat([node_features, pad], dim=0)
        if y2b_node.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - y2b_node.shape[0]
            y2b_node = torch.cat(
                [
                    y2b_node,
                    y2b_node.new_zeros(pad_num, y2b_node.shape[1]),
                ],
                dim=0,
            )

        if getattr(self, "use_block_native_output", False):
            raise NotImplementedError(
                "lem_moe_v3_prior_2b writes e3tb residual RME "
                "(Full-H − P labels). Do not set output_route='h_b0'."
            )

        gnn_node, gnn_edge = self._apply_rme_output_heads(
            node_features, edge_features, node_one_hot, edge_one_hot
        )
        out_node = y2b_node + gnn_node
        out_edge = y2b_edge + gnn_edge

        data[_keys.NODE_FEATURES_KEY] = out_node
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=self.dtype,
            device=self.device,
        )
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
            data[_keys.EDGE_FEATURES_KEY],
            0,
            active_edges,
            out_edge,
        )
        data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
        data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        return data


__all__ = ["LemMoEV3Prior2b", "resolve_prior_2b_keys", "PRIOR_2B_KINDS"]
