"""Prior initialization and time-conditioning parameter construction."""
from typing import Any
from .flow_time import FlowTimeConditioner

import torch

from dptb.configuration import resolve_init_scope
from dptb.data import _keys

from .prior_common import H0InitLayer


class UniTBPrior:
    @staticmethod
    def _h0_layer_type():
        return H0InitLayer

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
        h0_ao_cg: bool = True,
        **kwargs: Any,
    ):
        use_flow_time_embedding = bool(kwargs.pop("use_flow_time_embedding", False))
        flow_time_condition_edges = bool(kwargs.pop("flow_time_condition_edges", True))
        flow_time_key = str(kwargs.pop("flow_time_key", "flow_time"))
        flow_time_keys = kwargs.pop("flow_time_keys", None)
        flow_time_max_positions = int(kwargs.pop("flow_time_max_positions", 2000))
        flow_time_allow_missing = bool(kwargs.pop("flow_time_allow_missing", True))
        flow_time_missing_value = float(kwargs.pop("flow_time_missing_value", 0.0))
        flow_time_key_weights = kwargs.pop("flow_time_key_weights", None)
        super().__init__(**kwargs)
        self.use_flow_time_embedding = use_flow_time_embedding
        self.flow_time_condition_edges = flow_time_condition_edges
        self._edge_graph_invariant_checked = False
        self.flow_time_conditioner = (
            FlowTimeConditioner(
                scalar_channels=int(getattr(self, "env_embed_multiplicity", 10) or 10),
                flow_time_key=flow_time_key,
                flow_time_keys=flow_time_keys,
                max_positions=flow_time_max_positions,
                allow_missing_time=flow_time_allow_missing,
                missing_time_value=flow_time_missing_value,
                key_weights=flow_time_key_weights,
            )
            if use_flow_time_embedding
            else None
        )
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
            self.init_layer = self._h0_layer_type()(
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
                h0_ao_cg=h0_ao_cg,
                dtype=self.dtype,
                device=self.device,
            )


    def _initialize_prior(self, data, edge_index, atom_type, bond_type, edge_sh,
                          edge_length, edge_one_hot):
        """Initialize features and retain the original time-conditioning order."""
        args = (edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot)
        if self.use_h0_init:
            args = (data,) + args
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(*args)
        if getattr(self, "flow_time_conditioner", None) is not None:
            node_features = self.flow_time_conditioner(node_features, data)
            if self.flow_time_condition_edges:
                batch = data[_keys.BATCH_KEY]
                active_src = edge_index[0][active_edges]
                edge_batch = batch[active_src]
                if not self._edge_graph_invariant_checked:
                    dst_batch = batch[edge_index[1][active_edges]]
                    if not torch.equal(edge_batch, dst_batch):
                        raise ValueError(
                            "flow-time edge conditioning requires intra-graph active edges"
                        )
                    self._edge_graph_invariant_checked = True
                edge_features = self.flow_time_conditioner(
                    edge_features, data, batch=edge_batch
                )

        return latents, node_features, edge_features, cutoff_coeffs, active_edges
