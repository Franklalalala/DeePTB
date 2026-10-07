"""Shared edge-routed message passing and output assembly."""

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.tensor_product_moe_v3 import write_router_regularizers

from .unitb_backbone import UniTBBackbone
from .unitb_ops import _apply_onehot_tp, _capture_shift_hidden


from .unitb_router import UniTBRouter

class EdgeRouting:
    """The production edge route; structure controls use the same call contract."""

    @staticmethod
    def route(embedding, data, bond_type, active_edges, active_edge_one_hot, edge_vector):
        active_bond_type = bond_type.to(device=active_edges.device)[active_edges]
        router_input = embedding._edge_router_input(
            data, bond_type, active_edges, active_edge_one_hot, edge_vector
        )
        return embedding._make_edge_moe_globals(
            router_input, active_bond_type, data=data, active_edges=active_edges
        )


class UniTBEdge(UniTBRouter, UniTBBackbone):
    def __init__(self, **kwargs):
        from .unitb_structure import StructureRouting, options, validate_embedding

        self.structure_mole_options = options(kwargs.pop("structure_mole", None))
        self.structure_mole_enabled = self.structure_mole_options["enabled"]
        strategy = None
        if self.structure_mole_enabled:
            validate_embedding(self.structure_mole_options, kwargs)
            strategy = StructureRouting(self.structure_mole_options)
        self._routing = strategy if strategy is not None else EdgeRouting()
        super().__init__(_structure_strategy=strategy, **kwargs)

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
        mole_globals, monitor_val, expert_load_cv, num_route_tokens = self._routing.route(
            self, data, bond_type, active_edges, active_edge_one_hot, edge_vector
        )
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv
        write_router_regularizers(self.router, data)
        data["edge_moe_num_active_edges"] = torch.as_tensor(
            active_edge_one_hot.shape[0],
            device=active_edge_one_hot.device,
        )
        data["edge_moe_num_route_tokens"] = num_route_tokens

        data[_keys.EDGE_OVERLAP_KEY] = latents
        layer_args = dict(
            data=data,
            latents=latents,
            edge_features=edge_features,
            node_one_hot=node_one_hot,
            safe_node_one_hot=safe_node_one_hot,
            edge_index=edge_index,
            edge_vector=edge_vector,
            atom_type=atom_type,
            cutoff_coeffs=cutoff_coeffs,
            active_edges=active_edges,
            active_edge_one_hot=active_edge_one_hot,
            mole_globals=mole_globals,
            num_nodes_total=num_nodes_total,
        )
        out_node_features, out_edge_features = self._edge_layers_and_heads(
            node_features=node_features, **layer_args
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


    def _edge_layers_and_heads(
        self,
        *,
        data: AtomicDataDict.Type,
        latents: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_one_hot: torch.Tensor,
        safe_node_one_hot: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        atom_type: torch.Tensor,
        cutoff_coeffs: torch.Tensor,
        active_edges: torch.Tensor,
        active_edge_one_hot: torch.Tensor,
        mole_globals,
        num_nodes_total: int,
    ):
        wigner_D_all = None
        for idx, layer in enumerate(self.layers):
            _capture_shift_hidden(self, data, idx, node_features, num_nodes_total, active_edges)
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

        if getattr(self, "capture_shift_features", False):
            data["_shift_node_features"] = node_features
            data["_shift_active_edges"] = active_edges

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            out_node_features = out_node_features + _apply_onehot_tp(
                self.out_node_ele_tp, node_features, node_one_hot, self.onehot_tp_mode
            )
            out_edge_features = out_edge_features + _apply_onehot_tp(
                self.out_edge_ele_tp, edge_features, active_edge_one_hot, self.onehot_tp_mode
            )
        return out_node_features, out_edge_features
