import os
from typing import Any

import torch

from dptb.configuration import resolve_init_scope
from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLERouterV3

from .lem_moe_v3 import LemMoEV3, _apply_onehot_tp
from .lem_moe_v3_h0_helpers import H0InitLayer, _sorted_irrep_coordinate_index


@Embedding.register("lem_moe_v3_edge")
class LemMoEV3Edge(LemMoEV3):
    """LEM MoE v3 variant with per-active-edge routing coefficients."""

    def __init__(self, **kwargs: Any):
        edge_router_in_features = kwargs.pop("edge_router_in_features", None)
        self.edge_router_unique_types = bool(kwargs.pop("edge_router_unique_types", True))
        self.edge_moe_compact_dispatch = bool(kwargs.pop("edge_moe_compact_dispatch", True))
        self.edge_moe_compact_min_edges = int(kwargs.pop("edge_moe_compact_min_edges", 16384))
        self.edge_router_prior_activate = bool(kwargs.pop("edge_router_prior_activate", False))
        self.edge_router_prior_stats = str(kwargs.pop("edge_router_prior_stats", "") or "")
        edge_one_hot_dim = int(edge_router_in_features or kwargs.get("edge_one_hot_dim", 128))
        self.edge_one_hot_dim = edge_one_hot_dim
        self.edge_router_in_features = edge_one_hot_dim
        top_k = kwargs.get("top_k", 1)
        prev_so2_env = None
        if self.edge_router_prior_activate:
            # softmax over a single gathered logit is the constant 1, so with
            # top_k=1 the routing coefficient carries no gradient at all: the
            # zero-initialised descriptor columns would stay zero forever and the
            # whole mode would be a silent no-op that still pays for staged SO2.
            if int(top_k) < 2:
                raise ValueError(
                    "edge_router_prior_activate needs top_k >= 2; got %r. With "
                    "top_k=1 the gate is constant and the router receives no "
                    "gradient, so the descriptor could never earn any influence."
                    % (top_k,)
                )
            # post_activation mixing reaches the experts through apply_experts,
            # whose reference backend gathers one [out, in] weight per row -- the
            # very cost activation space exists to avoid, and the weight-space
            # guard does not sit on that path.
            mixing = kwargs.get("so2_expert_mixing_mode", "pre_activation")
            if mixing != "pre_activation":
                raise ValueError(
                    "edge_router_prior_activate requires "
                    "so2_expert_mixing_mode='pre_activation'; got %r, which "
                    "dispatches through apply_experts and would materialise one "
                    "weight per edge." % (mixing,)
                )
            # 'staged' is the only SO2 route that reaches every MOLELinear through
            # MOLELinear.forward; the fused routes call _mix_expert_parameters,
            # which activation space forbids (it would build per-edge weights).
            kwargs["so2_fusion_mode"] = "staged"
            # SO2_Linear lets DPTB_SO2_FUSION_MODE override exactly the value
            # "staged", so the kwarg alone is not enough wherever the deployment
            # exports one.  Suppress it across construction, then verify.
            prev_so2_env = os.environ.pop("DPTB_SO2_FUSION_MODE", None)
        try:
            super().__init__(**kwargs)
        finally:
            if prev_so2_env is not None:
                os.environ["DPTB_SO2_FUSION_MODE"] = prev_so2_env
        if self.edge_router_prior_activate:
            stray = [name for name, mod in self.named_modules()
                     if getattr(mod, "so2_fusion_mode", "staged") != "staged"]
            if stray:
                raise RuntimeError(
                    "edge_router_prior_activate needs every SO2 layer on the "
                    "'staged' route so each MOLELinear is reached through "
                    "MOLELinear.forward, but %d are not: %s"
                    % (len(stray), stray[:4])
                )

        self._prior_chunks = []
        self._prior_source_dim = 0
        self.edge_router_prior_dim = 0
        if self.edge_router_prior_activate:
            prior_irreps, sort_index = _sorted_irrep_coordinate_index(self.idp)
            self.register_buffer("_prior_sort_index", sort_index, persistent=False)
            offset = 0
            desc_dim = 0
            for mul, ir in prior_irreps:
                mul, ir_dim = int(mul), int(ir.dim)
                width = mul * ir_dim
                self._prior_chunks.append((offset, offset + width, mul, ir_dim))
                offset += width
                # l=0 is already invariant: keep every channel, signed.
                # l>0 needs a quadratic invariant, and the complete
                # Clebsch-Gordan-free one is the whole Gram matrix of the block,
                # not just its diagonal.  Same (l, parity) only -- a cross-parity
                # contraction is a pseudoscalar and would break inversion
                # equivariance.
                desc_dim += mul if ir_dim == 1 else mul * (mul + 1) // 2
            self._prior_source_dim = offset
            self.edge_router_prior_dim = desc_dim
            self.edge_router_in_features = edge_one_hot_dim + desc_dim
            mean = torch.zeros(desc_dim, dtype=self.dtype, device=self.device)
            std = torch.ones(desc_dim, dtype=self.dtype, device=self.device)
            if self.edge_router_prior_stats:
                blob = torch.load(self.edge_router_prior_stats, map_location="cpu")
                loaded_mean = blob["mean"].reshape(-1)
                loaded_std = blob["std"].reshape(-1)
                if loaded_mean.numel() != desc_dim or loaded_std.numel() != desc_dim:
                    raise ValueError(
                        "edge_router_prior_stats has "
                        f"{loaded_mean.numel()} channels, descriptor has {desc_dim}."
                    )
                mean = loaded_mean.to(dtype=self.dtype, device=self.device)
                std = loaded_std.to(dtype=self.dtype, device=self.device)
            # Frozen on purpose: standardisation is a property of the dataset,
            # not something the routing loss gets to move.  Non-persistent so a
            # resume takes the stats named by the current config instead of
            # silently restoring whatever the checkpoint was written with.
            self.register_buffer("_prior_mean", mean, persistent=False)
            self.register_buffer("_prior_std", std, persistent=False)

        self.router = MOLERouterV3(
            in_features=self.edge_router_in_features,
            num_experts=self.num_experts,
            top_k=top_k,
            aux_loss_free=True,
            bias_update_speed=0.005,
        )
        if self.edge_router_prior_dim:
            # The descriptor contributes exactly zero at step 0, so it can earn
            # influence without injecting a scale shock into the gate, and
            # gradients still flow into these columns.  Note this is NOT a
            # bit-identical baseline: widening the router input also rescales
            # nn.Linear's fan_in initialisation, so the surviving one-hot columns
            # are redrawn and narrower than in the 128-wide model.
            with torch.no_grad():
                self.router.net[0].weight[:, edge_one_hot_dim:].zero_()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
        )

        return self._finish_edge_routed_forward(
            data=data,
            edge_index=edge_index,
            edge_vector=edge_vector,
            node_one_hot=node_one_hot,
            atom_type=atom_type,
            bond_type=bond_type,
            latents=latents,
            node_features=node_features,
            edge_features=edge_features,
            cutoff_coeffs=cutoff_coeffs,
            active_edges=active_edges,
            edge_one_hot=edge_one_hot,
            num_nodes_total=num_nodes_total,
        )

    def _finish_edge_routed_forward(
        self,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        node_one_hot: torch.Tensor,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        latents: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        cutoff_coeffs: torch.Tensor,
        active_edges: torch.Tensor,
        edge_one_hot: torch.Tensor,
        num_nodes_total: int,
    ) -> AtomicDataDict.Type:
        active_edges = active_edges.to(device=edge_vector.device)
        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        active_edge_one_hot = edge_one_hot[active_edges]
        active_bond_type = bond_type.to(device=active_edges.device)[active_edges]
        router_input = active_edge_one_hot
        if self.edge_router_prior_activate:
            descriptor = self._gram_descriptor(
                self._raw_prior_source(data, bond_type, active_edges)
            )
            router_input = torch.cat(
                [active_edge_one_hot, descriptor.to(active_edge_one_hot.dtype)], dim=-1
            )
        mole_globals, monitor_val, expert_load_cv, num_route_tokens = self._make_edge_moe_globals(
            router_input,
            active_bond_type,
        )
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv
        data["edge_moe_num_active_edges"] = torch.as_tensor(
            active_edge_one_hot.shape[0],
            device=active_edge_one_hot.device,
        )
        data["edge_moe_num_route_tokens"] = num_route_tokens

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
                active_edge_one_hot,
                wigner_D_all,
                mole_globals,
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

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            out_node_features = out_node_features + _apply_onehot_tp(
                self.out_node_ele_tp, node_features, node_one_hot, self.onehot_tp_mode
            )
            out_edge_features = out_edge_features + _apply_onehot_tp(
                self.out_edge_ele_tp, edge_features, active_edge_one_hot, self.onehot_tp_mode
            )

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=out_edge_features.dtype,
            device=out_edge_features.device,
        )
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
            data[_keys.EDGE_FEATURES_KEY],
            0,
            active_edges,
            out_edge_features,
        )

        return data

    def _raw_prior_source(
        self,
        data: AtomicDataDict.Type,
        bond_type: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """The frozen edge prior, masked and permuted into sorted-irrep order.

        Read straight from the data dict, with no fallback: H0InitLayer's
        fallback chain can reach the target Hamiltonian, and routing on the
        target is label leakage.  Missing key is a hard error.
        """
        key = getattr(getattr(self, "init_layer", None), "h0_edge_key", None)
        if key is None:
            key = getattr(self, "h0_edge_key", None)
        if key is None or key not in data:
            raise KeyError(
                "edge_router_prior_activate needs the frozen edge prior at "
                f"data[{key!r}]; it exists only on lem_moe_v3_edge_h0 with a "
                "dataset that carries edge_h0."
            )
        source = data[key].to(dtype=self.dtype)
        mask = self.idp.mask_to_erme.to(source.device)[bond_type.flatten()]
        source = source * mask.to(dtype=source.dtype)
        source = source.index_select(1, self._prior_sort_index.to(source.device))
        return source.index_select(0, active_edges)

    def _gram_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation-invariant descriptor: l=0 signed, l>0 the full Gram matrix.

        signed-log is applied per channel purely for conditioning; it is
        strictly monotone, so no ordering information is lost.
        """
        if x.shape[-1] != self._prior_source_dim:
            raise ValueError(
                "edge prior descriptor expects a source of width "
                f"{self._prior_source_dim}, got {x.shape[-1]}."
            )
        parts = []
        for start, stop, mul, ir_dim in self._prior_chunks:
            block = x[:, start:stop].reshape(x.shape[0], mul, ir_dim)
            if ir_dim == 1:
                parts.append(block.reshape(x.shape[0], mul))
                continue
            gram = torch.einsum("nam,nbm->nab", block, block)
            iu = torch.triu_indices(mul, mul, offset=0, device=x.device)
            parts.append(gram[:, iu[0], iu[1]])
        desc = torch.cat(parts, dim=-1)
        desc = torch.sign(desc) * torch.log1p(desc.abs()).clamp(max=20.0)
        return (desc - self._prior_mean) / self._prior_std.clamp_min(1e-6)

    def _make_edge_moe_globals(
        self,
        active_edge_one_hot: torch.Tensor,
        active_bond_type: torch.Tensor,
    ):
        num_active_edges = int(active_edge_one_hot.shape[0])
        if active_edge_one_hot.shape[-1] != self.edge_router_in_features:
            raise ValueError(
                "edge_router_in_features mismatch: router was built with "
                f"{self.edge_router_in_features}, but active edge input has "
                f"{active_edge_one_hot.shape[-1]}."
            )
        if num_active_edges == 0:
            coeffs = active_edge_one_hot.new_zeros((0, self.num_experts))
            zero = active_edge_one_hot.new_zeros(())
            return MOLEGlobals(coefficients=coeffs, sizes=None), zero, zero, zero

        if self.edge_router_prior_activate:
            # One routing decision per edge.  No dedup: the whole point of this
            # mode is that pooling by any function of (bond type, r) would make
            # the coefficients a function of (bond type, r) too, no matter what
            # the descriptor carries.
            coeffs, monitor_val, expert_load_cv = self.router(active_edge_one_hot)
            topk_indices, topk_values = self.router.last_topk()
            if topk_indices is None or topk_values is None:
                raise RuntimeError(
                    "edge_router_prior_activate requires top_k < num_experts so "
                    "the router exposes top-k metadata for activation-space "
                    "dispatch."
                )
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))
            return (
                MOLEGlobals(
                    coefficients=coeffs,
                    sizes=None,
                    topk_indices=topk_indices,
                    topk_values=topk_values,
                    activation_space=True,
                ),
                monitor_val,
                expert_load_cv,
                num_route_tokens,
            )

        if self.edge_router_unique_types:
            unique_bond_type, inverse, counts = torch.unique(
                active_bond_type,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            unique_inputs = active_edge_one_hot.new_zeros(
                unique_bond_type.shape[0],
                active_edge_one_hot.shape[-1],
            )
            unique_inputs.index_add_(0, inverse, active_edge_one_hot)
            unique_inputs = unique_inputs / counts.to(dtype=active_edge_one_hot.dtype).unsqueeze(-1).clamp_min_(1)
            coeffs, monitor_val, expert_load_cv = self.router(
                unique_inputs,
                sizes=counts.to(dtype=active_edge_one_hot.dtype),
            )
            topk_indices, topk_values = self.router.last_topk()
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))
            use_compact_dispatch = (
                self.edge_moe_compact_dispatch
                and num_active_edges >= self.edge_moe_compact_min_edges
            )
            if use_compact_dispatch:
                return (
                    MOLEGlobals(
                        coefficients=coeffs,
                        sizes=None,
                        graph_index=inverse,
                        topk_indices=topk_indices,
                        topk_values=topk_values,
                    ),
                    monitor_val,
                    expert_load_cv,
                    num_route_tokens,
                )
            coeffs = coeffs.index_select(0, inverse)
            if topk_indices is not None and topk_values is not None:
                topk_indices = topk_indices.index_select(0, inverse)
                topk_values = topk_values.index_select(0, inverse)
        else:
            coeffs, monitor_val, expert_load_cv = self.router(active_edge_one_hot)
            topk_indices, topk_values = self.router.last_topk()
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))

        return (
            MOLEGlobals(
                coefficients=coeffs,
                sizes=None,
                topk_indices=topk_indices,
                topk_values=topk_values,
            ),
            monitor_val,
            expert_load_cv,
            num_route_tokens,
        )


@Embedding.register("lem_moe_v3_edge_h0")
class LemMoEV3EdgeH0(LemMoEV3Edge):
    """H0 initialization variant of edge-wise LEM MoE v3."""

    def __init__(
        self,
        h0_init_scope: Any = None,
        use_h0_init: Any = None,
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        use_h0_node_init: Any = None,
        use_h0_edge_init: Any = None,
        h0_node_mode: str = "direct",
        fallback_to_hamiltonian: Any = None,
        h0_fallback_to_hamiltonian: Any = None,
        allow_target_fallback_in_training: bool = False,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        h0_merge_mode: str = "replace",
        h0_self_edge_tol: float = 1e-8,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        (
            self.h0_init_scope,
            self.use_h0_init,
            use_h0_node_init,
            use_h0_edge_init,
        ) = resolve_init_scope(
            h0_init_scope,
            enabled=use_h0_init,
            node=use_h0_node_init,
            edge=use_h0_edge_init,
            option_name="h0_init_scope",
        )
        if fallback_to_hamiltonian is None:
            fallback_to_hamiltonian = (
                True
                if h0_fallback_to_hamiltonian is None
                else bool(h0_fallback_to_hamiltonian)
            )
        elif (
            h0_fallback_to_hamiltonian is not None
            and bool(fallback_to_hamiltonian) != bool(h0_fallback_to_hamiltonian)
        ):
            raise ValueError(
                "fallback_to_hamiltonian conflicts with deprecated "
                "h0_fallback_to_hamiltonian."
            )

        if self.use_h0_init:
            self.init_layer = H0InitLayer(
                base_init=self.init_layer,
                h0_node_key=h0_node_key,
                h0_edge_key=h0_edge_key,
                use_h0_node_init=use_h0_node_init,
                use_h0_edge_init=use_h0_edge_init,
                h0_node_mode=h0_node_mode,
                fallback_to_hamiltonian=fallback_to_hamiltonian,
                fallback_node_key=fallback_node_key,
                fallback_edge_key=fallback_edge_key,
                allow_target_fallback_in_training=allow_target_fallback_in_training,
                merge_mode=h0_merge_mode,
                self_edge_tol=h0_self_edge_tol,
                dtype=self.dtype,
                device=self.device,
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not self.use_h0_init:
            return super().forward(data)

        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            data,
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
        )

        return self._finish_edge_routed_forward(
            data=data,
            edge_index=edge_index,
            edge_vector=edge_vector,
            node_one_hot=node_one_hot,
            atom_type=atom_type,
            bond_type=bond_type,
            latents=latents,
            node_features=node_features,
            edge_features=edge_features,
            cutoff_coeffs=cutoff_coeffs,
            active_edges=active_edges,
            edge_one_hot=edge_one_hot,
            num_nodes_total=num_nodes_total,
        )
