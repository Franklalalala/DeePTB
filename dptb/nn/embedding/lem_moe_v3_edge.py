from typing import Any

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLERouterV3

from .lem_moe_v3 import LemMoEV3
from .lem_moe_v3_h0_helpers import H0InitLayer


@Embedding.register("lem_moe_v3_edge")
class LemMoEV3Edge(LemMoEV3):
    """LEM MoE v3 variant with per-active-edge routing coefficients."""

    def __init__(self, **kwargs: Any):
        edge_router_in_features = kwargs.pop("edge_router_in_features", None)
        self.edge_router_unique_types = bool(kwargs.pop("edge_router_unique_types", True))
        self.edge_moe_compact_dispatch = bool(kwargs.pop("edge_moe_compact_dispatch", True))
        self.edge_moe_compact_min_edges = int(kwargs.pop("edge_moe_compact_min_edges", 16384))
        edge_one_hot_dim = int(edge_router_in_features or kwargs.get("edge_one_hot_dim", 128))
        self.edge_router_in_features = edge_one_hot_dim
        top_k = kwargs.get("top_k", 1)
        super().__init__(**kwargs)
        self.router = MOLERouterV3(
            in_features=edge_one_hot_dim,
            num_experts=self.num_experts,
            top_k=top_k,
            aux_loss_free=True,
            bias_update_speed=0.005,
        )

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
        mole_globals, monitor_val, expert_load_cv, num_route_tokens = self._make_edge_moe_globals(
            active_edge_one_hot,
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
            out_node_features = out_node_features + self.out_node_ele_tp(node_features, node_one_hot)
            out_edge_features = out_edge_features + self.out_edge_ele_tp(edge_features, active_edge_one_hot)

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
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))
            top_k = getattr(self.router, "top_k", None)
            full_expert_routing = top_k is None or top_k >= self.num_experts
            use_compact_dispatch = (
                self.edge_moe_compact_dispatch
                and full_expert_routing
                and num_active_edges >= self.edge_moe_compact_min_edges
            )
            if use_compact_dispatch:
                return (
                    MOLEGlobals(coefficients=coeffs, sizes=None, graph_index=inverse),
                    monitor_val,
                    expert_load_cv,
                    num_route_tokens,
                )
            coeffs = coeffs.index_select(0, inverse)
        else:
            coeffs, monitor_val, expert_load_cv = self.router(active_edge_one_hot)
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))

        return MOLEGlobals(coefficients=coeffs, sizes=None), monitor_val, expert_load_cv, num_route_tokens


@Embedding.register("lem_moe_v3_edge_h0")
class LemMoEV3EdgeH0(LemMoEV3Edge):
    """H0 initialization variant of edge-wise LEM MoE v3."""

    def __init__(
        self,
        use_h0_init: bool = True,
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        h0_node_mode: str = "direct",
        fallback_to_hamiltonian: bool = True,
        h0_fallback_to_hamiltonian: Any = None,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        h0_merge_mode: str = "replace",
        h0_self_edge_tol: float = 1e-8,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_h0_init = use_h0_init
        if h0_fallback_to_hamiltonian is not None:
            fallback_to_hamiltonian = bool(h0_fallback_to_hamiltonian)

        if self.use_h0_init:
            self.init_layer = H0InitLayer(
                base_init=self.init_layer,
                h0_node_key=h0_node_key,
                h0_edge_key=h0_edge_key,
                h0_node_mode=h0_node_mode,
                fallback_to_hamiltonian=fallback_to_hamiltonian,
                fallback_node_key=fallback_node_key,
                fallback_edge_key=fallback_edge_key,
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
