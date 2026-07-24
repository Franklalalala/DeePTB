from __future__ import annotations

from typing import Any

import torch
from torch_scatter import scatter_mean

from dptb.configuration import resolve_init_scope
from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    attach_prediction_block_tensors,
    infer_block_shapes,
)
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3 import LemMoEV3
from .lem_moe_v3_h0_helpers import H0InitLayer
from .flow_time import FlowTimeConditioner
from .late_block_expansion_cg import LateBlockExpansionCGHead


@Embedding.register("lem_moe_v3_h0")
class LemMoEV3H0(LemMoEV3):
    supports_full_block_edge_coverage = True

    @staticmethod
    def _require_integral_metadata(
        value: Any,
        *,
        device: torch.device,
        label: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError(
                f"Block-space ODE requires integral {label}; got dtype={tensor.dtype}."
            )
        return tensor.reshape(-1)

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
        use_uureal_residual_block_input: bool = False,
        use_spatial_residual_block_input: bool = False,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        h0_merge_mode: str = "replace",
        h0_self_edge_tol: float = 1e-8,
        use_flow_time_embedding: bool = False,
        flow_time_condition_edges: bool = True,
        flow_time_key: str = "flow_time",
        flow_time_keys: Any = None,
        flow_time_max_positions: int = 2000,
        flow_time_allow_missing: bool = True,
        flow_time_missing_value: float = 0.0,
        flow_time_key_weights: Any = None,
        require_full_block_edge_coverage: bool = False,
        hb0_hermitian_average: bool = False,
        condition_source: str = "edge_0e",
        log_head_input_rms: bool = False,
        env_embed_multiplicity: int = 32,
        **kwargs: Any,
    ):
        condition_source = LateBlockExpansionCGHead.normalize_condition_source(
            condition_source
        )
        super().__init__(env_embed_multiplicity=env_embed_multiplicity, **kwargs)
        self.hb0_hermitian_average = bool(hb0_hermitian_average)
        if self.hb0_hermitian_average and self.output_route_name != "h_b0":
            raise ValueError(
                "hb0_hermitian_average=true requires output_route='h_b0'."
            )
        self.condition_source = condition_source
        if self.condition_source == "endpoints":
            if self.output_route_name != "h_b0":
                raise ValueError(
                    "condition_source='endpoints' requires output_route='h_b0'."
                )
            self.out_edge.configure_condition_source(
                "endpoints", node_irreps=self.layers[-1].irreps_out
            )
        self.log_head_input_rms = bool(log_head_input_rms)
        if self.log_head_input_rms and self.output_route_name != "h_b0":
            raise ValueError(
                "log_head_input_rms=true requires output_route='h_b0'."
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
        self.require_full_block_edge_coverage = bool(require_full_block_edge_coverage)
        if self.require_full_block_edge_coverage and (
            not self.use_h0_init or self.output_route_name != "h_b0"
        ):
            raise ValueError(
                "require_full_block_edge_coverage=true is supported only by "
                "the LemMoEV3H0 H-B0 route with use_h0_init=true."
            )
        self.use_flow_time_embedding = bool(use_flow_time_embedding)
        self.flow_time_condition_edges = bool(flow_time_condition_edges)
        self._edge_graph_invariant_checked = False
        self.flow_time_conditioner = (
            FlowTimeConditioner(
                scalar_channels=env_embed_multiplicity,
                flow_time_key=flow_time_key,
                flow_time_keys=flow_time_keys,
                max_positions=flow_time_max_positions,
                allow_missing_time=flow_time_allow_missing,
                missing_time_value=flow_time_missing_value,
                key_weights=flow_time_key_weights,
            )
            if self.use_flow_time_embedding
            else None
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
                use_uureal_residual_block_input=use_uureal_residual_block_input,
                use_spatial_residual_block_input=use_spatial_residual_block_input,
                merge_mode=h0_merge_mode,
                self_edge_tol=h0_self_edge_tol,
                dtype=self.dtype,
                device=self.device,
            )

    @staticmethod
    def _require_ordered_full_block_edge_coverage(
        edge_index: torch.Tensor,
        active_edges: torch.Tensor,
        cutoff_coeffs: torch.Tensor,
        batch: torch.Tensor,
        split_sizes: Any,
        num_systems: int,
    ) -> None:
        """Certify the exact rows consumed by the H-B0 head and scatter."""
        n_edges = int(edge_index.shape[1])
        active_edges = LemMoEV3H0._require_integral_metadata(
            active_edges,
            device=edge_index.device,
            label="active-edge indices",
        ).to(dtype=torch.long)
        expected = torch.arange(n_edges, device=edge_index.device, dtype=torch.long)
        if not torch.equal(active_edges, expected):
            raise ValueError(
                "Block-space ODE requires the ordered full H-B0 active-edge range "
                f"0..{n_edges - 1}; got {active_edges[:16].detach().cpu().tolist()} "
                f"(total={int(active_edges.numel())})."
            )

        cutoff_coeffs = torch.as_tensor(
            cutoff_coeffs, device=edge_index.device
        ).reshape(-1)
        valid = torch.isfinite(cutoff_coeffs) & (cutoff_coeffs > 0)
        if cutoff_coeffs.numel() != n_edges or not bool(valid.all().item()):
            invalid = torch.nonzero(~valid, as_tuple=False).flatten()
            raise ValueError(
                "Block-space ODE requires one finite, strictly positive H-B0 "
                f"cutoff coefficient per graph edge; got shape={tuple(cutoff_coeffs.shape)} "
                f"and invalid indices={invalid[:16].detach().cpu().tolist()}."
            )
        edge_batch = batch[edge_index[0]]
        if not torch.equal(edge_batch, batch[edge_index[1]]):
            raise ValueError(
                "Block-space ODE requires every H-B0 edge to stay within one "
                "batch graph; source and destination graph indices disagree."
            )
        if split_sizes is not None:
            split_sizes = LemMoEV3H0._require_integral_metadata(
                split_sizes,
                device=edge_index.device,
                label="active-edge split sizes",
            ).to(dtype=torch.long)
            expected_sizes = torch.bincount(edge_batch, minlength=int(num_systems))
            if split_sizes.numel() != int(num_systems) or not torch.equal(
                split_sizes, expected_sizes
            ):
                raise ValueError(
                    "Block-space ODE active-edge split sizes must exactly match "
                    "the graph ownership of the ordered full edge range; "
                    f"expected={expected_sizes.detach().cpu().tolist()}, "
                    f"got={split_sizes.detach().cpu().tolist()}."
                )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not self.use_h0_init:
            return super().forward(data)

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
        if self.require_full_block_edge_coverage and precomputed_active_edges is not None:
            precomputed_active_edges = self._require_integral_metadata(
                precomputed_active_edges,
                device=edge_index.device,
                label="active-edge indices",
            ).to(dtype=torch.long)
        if precomputed_cutoff_coeffs is not None and edge_length.requires_grad:
            raise RuntimeError(
                "Precomputed LEM cutoff coefficients cannot be used when edge_length requires gradients. "
                "Set train_options.precompute_lem_cutoff_coeffs=false for force/stress/virial training."
            )
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            data,
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
            precomputed_active_edges,
            precomputed_cutoff_coeffs,
        )
        if self.require_full_block_edge_coverage:
            self._require_ordered_full_block_edge_coverage(
                edge_index,
                active_edges,
                cutoff_coeffs,
                batch,
                precomputed_split_sizes,
                int(coeffs.shape[0]),
            )
        if self.flow_time_conditioner is not None:
            node_features = self.flow_time_conditioner(node_features, data)
            if self.flow_time_condition_edges:
                active_src = edge_index[0][active_edges]
                edge_batch = batch[active_src]
                if not self._edge_graph_invariant_checked:
                    # Edge time conditioning indexes graph time by the edge's
                    # source node; that is only correct if edges never cross
                    # graphs. Verify once per module (torch.equal syncs the
                    # GPU, so this must not run every forward).
                    dst_batch = batch[edge_index[1][active_edges]]
                    if not torch.equal(edge_batch, dst_batch):
                        raise ValueError(
                            "flow-time edge conditioning requires intra-graph active edges; "
                            "source and destination nodes disagree on graph index. Check "
                            "radius-graph construction or precomputed active edges."
                        )
                    self._edge_graph_invariant_checked = True
                edge_features = self.flow_time_conditioner(
                    edge_features,
                    data,
                    batch=edge_batch,
                )

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

        if getattr(self, "use_block_native_output", False):
            head_kwargs = {}
            if getattr(self, "pair_refine_enable", False):
                head_kwargs["full_cutoff_coeffs"] = cutoff_coeffs
            head_outputs = self._apply_block_native_output_heads(
                node_features,
                edge_features,
                atom_type,
                edge_index,
                active_edges,
                **head_kwargs,
            )
            if getattr(self, "log_head_input_rms", False):
                out_node_blocks, out_edge_blocks, head_input_rms = head_outputs
                data["head_input_rms"] = head_input_rms
            else:
                out_node_blocks, out_edge_blocks = head_outputs
            data[_keys.NODE_HAMILTONIAN_KEY] = out_node_blocks
            data[_keys.EDGE_HAMILTONIAN_KEY] = torch.zeros(
                edge_index.shape[1],
                self.out_edge.max_norb,
                self.out_edge.max_norb,
                dtype=self.dtype,
                device=self.device,
            )
            data[_keys.EDGE_HAMILTONIAN_KEY] = torch.index_copy(
                data[_keys.EDGE_HAMILTONIAN_KEY],
                0,
                active_edges,
                out_edge_blocks,
            )
            node_shapes, edge_shapes = infer_block_shapes(
                data, self.idp, device=out_node_blocks.device
            )
            attach_prediction_block_tensors(
                data,
                BlockTensorResult(
                    node_blocks=data[_keys.NODE_HAMILTONIAN_KEY],
                    edge_blocks=data[_keys.EDGE_HAMILTONIAN_KEY],
                    node_shapes=node_shapes,
                    edge_shapes=edge_shapes,
                ),
            )
            data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
            data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
            data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
            return data

        out_node_features, out_edge_features = self._apply_rme_output_heads(
            node_features, edge_features, node_one_hot, edge_one_hot
        )

        data[_keys.NODE_FEATURES_KEY] = out_node_features
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
            out_edge_features,
        )
        data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
        data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        return data
