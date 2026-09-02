# SPDX-License-Identifier: LGPL-3.0-or-later
"""only2b-style two-stage embedding on prior residual labels (y = Full-H - P).

Both stages consume the prior RME (na_cf: node_p23 / edge_p2) through the same
projector design as ``lem_moe_v3_prior`` (mask -> sorted-irrep permutation ->
``e3nn.Linear``), and *concat* it with the geometric InitLayer features instead
of replacing them:  phi = [h_geo ; Pi(P)].

Two parallel branches share nothing but the geometry:

* ``two_b_*``  - the cheap pairwise branch (Trinity's ``Twoness`` analogue):
  its own geometric InitLayer clone, its own P projectors and a linear RME
  readout  y_2b = W_2b phi_2b.  No message passing.
* the GNN     - ``h0_init`` (InitLayer + P projectors) -> first SO2 layer with
  ``irreps_in = 2 x geo`` -> remaining layers -> RME heads  y_gnn.

``only2b=true``  (stage 1): y = y_2b.  The GNN is not executed and gets no
                            gradient, so stage 1 is a pairwise fit of H - P.
``only2b=false`` (stage 2): y = y_2b* + y_gnn.  The 2b branch is frozen (a fixed
                            H-space baseline); InitLayer, projectors and all
                            layers train.  When a stage-1 checkpoint is loaded
                            the GNN InitLayer/projectors are seeded once from
                            the trained 2b branch (``two_b_seed_gnn``).
"""

from __future__ import annotations

import copy
import inspect
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

from .lem_moe_v3 import LemMoEV3
from .lem_moe_v3_h0 import LemMoEV3H0
from .lem_moe_v3_h0_helpers import H0InitLayer, _get_feature_source_with_key


PRIOR_2B_KINDS = {
    "p2": (_keys.NODE_P2_KEY, _keys.EDGE_P2_KEY),
    "p23": (_keys.NODE_P23_KEY, _keys.EDGE_P23_KEY),
    "na_cf": (_keys.NODE_P23_KEY, _keys.EDGE_P2_KEY),
}

# Layer kwargs the base class forwards verbatim from its own __init__ arguments.
_LAYER_PASSTHROUGH = (
    "tp_radial_emb",
    "tp_radial_channels",
    "use_layer_onehot_tp",
    "edge_one_hot_dim",
    "latent_channels",
    "res_update",
    "res_update_ratios",
    "res_update_ratios_learnable",
    "equivariant_norm_type",
    "swiglu_s2_grid_resolution",
    "swiglu_s2_compat_mode",
    "ffn_hidden_factor",
    "so2_wigner_apply_mode",
    "so2_fusion_mode",
    "mole_linear_mode",
    "so2_expert_route_chunk_size",
    "so2_expert_route_checkpoint",
    "so2_output_router_hidden_dim",
    "focus_attention_dim",
    "num_shared_experts",
)
# Layer kwargs the base class forwards from its normalised attributes.
_LAYER_FROM_SELF = (
    "so2_expert_mixing_mode",
    "onehot_tp_mode",
    "node_message_aggregation",
    "num_focus",
    "edge_aggregation_gated_attention",
    "edge_attention_key_source",
    "edge_attention_envelope_power",
    "edge_attention_use_latent_bias",
    "edge_attention_key_layer_norm",
    "edge_attention_query_layer_norm",
    "edge_attention_qk_layer_norm",
    "edge_message_env_weight",
)


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


def _clone_module(module: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module but share the (large, read-only) OrbitalMapper."""
    idp = getattr(module, "idp", None)
    if idp is not None:
        module.idp = None
    try:
        clone = copy.deepcopy(module)
    finally:
        if idp is not None:
            module.idp = idp
    if idp is not None:
        clone.idp = idp
    return clone


@torch.no_grad()
def _copy_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=True)


@Embedding.register("lem_moe_v3_prior_2b")
class LemMoEV3Prior2b(LemMoEV3H0):
    """Frozen pairwise 2b branch + concat-P GNN for only2b two-stage training."""

    def __init__(
        self,
        *,
        only2b: bool = False,
        two_b_seed_gnn: bool = True,
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
        if merge != "concat":
            raise ValueError(
                "lem_moe_v3_prior_2b feeds concat(geo, P-map) into the first SO2 "
                f"layer; prior_merge_mode must be 'concat', got {prior_merge_mode!r}."
            )
        node_key, edge_key = resolve_prior_2b_keys(prior_kind)
        for given, expect, label in (
            (prior_node_key, node_key, "prior_node_key"),
            (prior_edge_key, edge_key, "prior_edge_key"),
        ):
            if given not in (None, "") and str(given) != expect:
                raise ValueError(
                    f"prior_kind={prior_kind!r} requires {label}={expect!r}; got {given!r}."
                )
        scope, use_init, _, _ = resolve_init_scope(
            prior_init_scope,
            enabled=use_prior_init,
            node=use_prior_node_init,
            edge=use_prior_edge_init,
            option_name="prior_init_scope",
        )
        if not use_init or scope != "both":
            raise ValueError(
                "lem_moe_v3_prior_2b needs prior_init_scope='both': both the 2b "
                "branch and the GNN read node and edge prior RME."
            )
        self.prior_init_scope = scope
        self._prior2b_raw_kwargs: Dict[str, Any] = dict(kwargs)

        super().__init__(
            h0_init_scope=scope,
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
                "lem_moe_v3_prior_2b is non-SOC; residual Full-H - P is real RME."
            )

        self.only2b = bool(only2b)
        self.two_b_seed_gnn = bool(two_b_seed_gnn)
        self.prior_kind = str(prior_kind).strip().lower()
        self.prior_node_key = node_key
        self.prior_edge_key = edge_key
        self.prior_merge_mode = "concat"
        self.h0_init = _unwrap_h0_init(self.init_layer)
        geo_irreps = o3.Irreps(self.h0_init.irreps_out)
        self.concat_irreps = geo_irreps + geo_irreps
        self._rebuild_first_layer_for_concat()

        # Independent pairwise branch (Trinity Twoness analogue).
        self.two_b_init = _clone_module(self.h0_init.base_init)
        self.two_b_node_proj = _clone_module(self.h0_init.node_projector)
        self.two_b_edge_proj = _clone_module(self.h0_init.edge_projector)
        linear_kwargs = dict(shared_weights=True, internal_weights=True, biases=True)
        self.two_b_out_node = Linear(self.concat_irreps, self.idp.orbpair_irreps, **linear_kwargs)
        self.two_b_out_edge = Linear(self.concat_irreps, self.idp.orbpair_irreps, **linear_kwargs)
        self.register_buffer("two_b_gnn_seeded", torch.zeros((), dtype=torch.bool))
        self.register_load_state_dict_post_hook(self._seed_gnn_from_two_b_hook)

        if not self.only2b:
            for module in self._two_b_modules():
                for param in module.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------ build
    def _layer_option(self, name: str) -> Any:
        """Value the base __init__ saw for ``name`` (explicit kwarg or its default)."""
        if name in self._prior2b_raw_kwargs:
            return self._prior2b_raw_kwargs[name]
        return inspect.signature(LemMoEV3.__init__).parameters[name].default

    def _rebuild_first_layer_for_concat(self) -> None:
        """Rebuild layers[0] with doubled input irreps, mirroring the base loop for i == 0."""
        old = self.layers[0]
        n_layers = int(self._layer_option("n_layers"))
        is_last = n_layers == 1
        if is_last:
            edge_act = node_act = "gate"
        else:
            edge_act = self._layer_option("hidden_edge_activation_type")
            node_act = self._layer_option("hidden_node_activation_type")
        ffn_hidden_factor = float(self._layer_option("ffn_hidden_factor"))
        use_node_ffn = ffn_hidden_factor > 1.0 and (
            (not is_last) or bool(self._layer_option("ffn_apply_to_last"))
        )
        use_interpolation_tp = bool(
            is_last
            and self._layer_option("use_interpolation_out")
            and getattr(self.output_route_spec, "final_irreps_kind", "") == "orbpair"
        )
        layer_kwargs = {name: self._layer_option(name) for name in _LAYER_PASSTHROUGH}
        layer_kwargs.update({name: getattr(self, name) for name in _LAYER_FROM_SELF})
        self.layers[0] = self._layer_type()(
            num_types=self.n_atom,
            avg_num_neighbors=old.avg_num_neighbors,
            irreps_in=self.concat_irreps,
            irreps_out=old.irreps_out,
            latent_dim=self.latent_dim,
            edge_activation_type=edge_act,
            node_activation_type=node_act,
            use_node_ffn=use_node_ffn,
            use_interpolation_tp=use_interpolation_tp,
            num_experts=self.num_experts,
            dtype=self.dtype,
            device=self.device,
            **layer_kwargs,
        )

    def _two_b_modules(self) -> Tuple[torch.nn.Module, ...]:
        return (
            self.two_b_init,
            self.two_b_node_proj,
            self.two_b_edge_proj,
            self.two_b_out_node,
            self.two_b_out_edge,
        )

    @staticmethod
    def _seed_gnn_from_two_b_hook(module: "LemMoEV3Prior2b", incompatible_keys: Any) -> None:
        """Stage-2 start: copy the trained 2b InitLayer/projectors into the GNN once."""
        if module.only2b or not module.two_b_seed_gnn or bool(module.two_b_gnn_seeded):
            return
        _copy_params(module.two_b_init, module.h0_init.base_init)
        _copy_params(module.two_b_node_proj, module.h0_init.node_projector)
        _copy_params(module.two_b_edge_proj, module.h0_init.edge_projector)
        module.two_b_gnn_seeded.fill_(True)

    # ---------------------------------------------------------------- forward
    def _project_prior(
        self,
        data: AtomicDataDict.Type,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        active_edges: torch.Tensor,
        n_nodes: int,
        n_active: int,
        node_proj: torch.nn.Module,
        edge_proj: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same mask -> sorted-irrep -> Linear path as H0InitLayer, with explicit projectors."""
        h0 = self.h0_init
        out = []
        for key, expected, label, mask, proj, rows in (
            (self.prior_node_key, h0.h0_dim, "node prior", h0._mask_node_source, node_proj, n_nodes),
            (self.prior_edge_key, h0.h0_dim, "edge prior", h0._mask_edge_source, edge_proj, n_active),
        ):
            source, found_key = _get_feature_source_with_key(
                data=data,
                candidate_keys=[key],
                expected_dim=expected,
                dtype=self.dtype,
                device=self.device,
                label=label,
            )
            if source is None:
                raise KeyError(f"lem_moe_v3_prior_2b requires field {key!r} in the batch.")
            h0._guard_target_fallback(found_key, key, label)
            source = mask(source, atom_type if label == "node prior" else bond_type)
            source = source.index_select(1, h0._h0_sort_index)
            if label == "edge prior":
                source = source[active_edges]
            feat = proj(source)
            if feat.shape[0] != rows:
                feat = h0._align_feature_rows(feat, rows)
            out.append(feat)
        return out[0], out[1]

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
        edge_sh = self.sh(edge_vector[:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]
        num_nodes_total = node_one_hot.shape[0]

        precomputed_active_edges = data.get(_keys.LEM_ACTIVE_EDGES_KEY, None)
        precomputed_cutoff_coeffs = data.get(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        if precomputed_cutoff_coeffs is not None and edge_length.requires_grad:
            raise RuntimeError(
                "Precomputed LEM cutoff coefficients cannot be used when "
                "edge_length requires gradients."
            )
        init_args = (
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
            precomputed_active_edges,
            precomputed_cutoff_coeffs,
        )

        # Router runs in both stages so the monitor keys always exist.
        global_feat = scatter_mean(node_one_hot, batch, dim=0)
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)
        topk_indices, topk_values = self.router.last_topk()
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv

        # --- pairwise 2b branch: y_2b = W_2b [h_geo ; Pi(P)], no message passing
        latents, geo_node, geo_edge, cutoff_coeffs, active_edges = self.two_b_init(*init_args)
        prior_node, prior_edge = self._project_prior(
            data, atom_type, bond_type, active_edges, geo_node.shape[0], geo_edge.shape[0],
            self.two_b_node_proj, self.two_b_edge_proj,
        )
        y2b_node = self.two_b_out_node(torch.cat([geo_node, prior_node], dim=-1))
        y2b_edge = self.two_b_out_edge(torch.cat([geo_edge, prior_edge], dim=-1))
        out_node, out_edge = y2b_node, y2b_edge

        # --- GNN branch (stage 2 only): first SO2 layer eats concat(geo, P-map)
        if not self.only2b:
            latents, geo_node, geo_edge, cutoff_coeffs, gnn_active = self.h0_init.base_init(*init_args)
            if gnn_active.shape != active_edges.shape or not torch.equal(gnn_active, active_edges):
                raise RuntimeError("2b branch and GNN InitLayer disagree on the active edge set.")
            prior_node, prior_edge = self._project_prior(
                data, atom_type, bond_type, active_edges, geo_node.shape[0], geo_edge.shape[0],
                self.h0_init.node_projector, self.h0_init.edge_projector,
            )
            node_features = torch.cat([geo_node, prior_node], dim=-1)
            edge_features = torch.cat([geo_edge, prior_edge], dim=-1)
            if node_features.shape[-1] != self.concat_irreps.dim:
                raise RuntimeError(
                    f"concat(geo, P) width {node_features.shape[-1]} != "
                    f"concat_irreps.dim {self.concat_irreps.dim}."
                )

            node_batch = batch[: node_features.shape[0]]
            safe_node_one_hot = node_one_hot[: node_features.shape[0]]
            active_edge_one_hot = edge_one_hot[active_edges]
            if preserved_split_sizes is not None:
                mole_globals = MOLEGlobals(
                    coefficients=coeffs,
                    split_sizes=preserved_split_sizes,
                    topk_indices=topk_indices,
                    topk_values=topk_values,
                )
            else:
                edge_batch = batch[edge_index[0][active_edges]]
                mole_globals = MOLEGlobals(
                    coefficients=coeffs,
                    sizes=torch.bincount(edge_batch, minlength=coeffs.shape[0]),
                    graph_index=edge_batch,
                    topk_indices=topk_indices,
                    topk_values=topk_values,
                )

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
                    active_edge_one_hot,
                    wigner_D_all,
                    mole_globals,
                    node_batch,
                )
            if node_features.shape[0] < num_nodes_total:
                node_features = torch.cat(
                    [
                        node_features,
                        node_features.new_zeros(
                            num_nodes_total - node_features.shape[0], node_features.shape[1]
                        ),
                    ],
                    dim=0,
                )
            if getattr(self, "use_block_native_output", False):
                raise NotImplementedError(
                    "lem_moe_v3_prior_2b writes e3tb residual RME (Full-H - P labels); "
                    "do not set output_route='h_b0'."
                )
            gnn_node, gnn_edge = self._apply_rme_output_heads(
                node_features, edge_features, node_one_hot, active_edge_one_hot
            )
            if y2b_node.shape[0] < gnn_node.shape[0]:
                y2b_node = torch.cat(
                    [y2b_node, y2b_node.new_zeros(gnn_node.shape[0] - y2b_node.shape[0], y2b_node.shape[1])],
                    dim=0,
                )
            out_node = y2b_node + gnn_node
            out_edge = y2b_edge + gnn_edge

        if out_node.shape[0] < num_nodes_total:
            out_node = torch.cat(
                [out_node, out_node.new_zeros(num_nodes_total - out_node.shape[0], out_node.shape[1])],
                dim=0,
            )

        data[_keys.EDGE_OVERLAP_KEY] = latents
        data[_keys.NODE_FEATURES_KEY] = out_node
        full_edge = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=out_edge.dtype,
            device=out_edge.device,
        )
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(full_edge, 0, active_edges, out_edge)
        data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
        data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        return data


__all__ = ["LemMoEV3Prior2b", "resolve_prior_2b_keys", "PRIOR_2B_KINDS"]
