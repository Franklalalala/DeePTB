"""Late pair construction and norm-free readout refinement.

The pair stream is intentionally independent of message-passing cutoffs.  Its
first stage applies a fresh SO(2) edge update to mature node representations,
while retaining the current edge state as the ordered-pair seed.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

from .lem_moe_v3 import (
    UpdateEdge,
    _apply_onehot_tp,
    _apply_so2_tp_or_post_activation_mixer,
)


def _scalar_0e_indices(
    irreps: o3.Irreps,
    *,
    device: Union[str, torch.device],
) -> torch.Tensor:
    indices = []
    for term_slice, (_, ir) in zip(irreps.slices(), irreps):
        if ir.l == 0 and ir.p == 1:
            indices.extend(range(term_slice.start, term_slice.stop))
    if not indices:
        raise ValueError(f"Two-stage pair refinement requires 0e scalars; got {irreps}.")
    return torch.tensor(indices, dtype=torch.long, device=device)


class _LatePairUpdateEdge(UpdateEdge):
    """The edge-producing half of ``UpdateEdge(res_update=False)``.

    The parent computes a new latent after it has already produced the edge
    output.  A readout stream discards that latent, which would leave the latent
    LayerNorm and MLP parameters permanently unused.  This narrow subclass
    preserves the parent's SO(2) edge math exactly and omits only that dead
    post-edge branch.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["res_update"] = False
        super().__init__(*args, **kwargs)
        del self.ln
        del self.latents_mlp_1
        del self.latents_mlp_2

    def forward(
        self,
        latents,
        node_features,
        node_onehot,
        edge_features,
        edge_index,
        edge_vector,
        cutoff_coeffs,
        active_edges,
        edge_one_hot,
        wigner_D_all,
        mole_globals,
    ):
        del node_onehot, cutoff_coeffs
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]
        node_in = (
            self.node_norm(node_features)
            if self.node_norm is not None
            else node_features
        )
        edge_in = (
            self.edge_norm(edge_features)
            if self.edge_norm is not None
            else edge_features
        )
        tp_input = torch.cat(
            (
                node_in[edge_center[active_edges]],
                edge_in,
                node_in[edge_neighbor[active_edges]],
            ),
            dim=-1,
        )
        new_edge_features, wigner_D_all = _apply_so2_tp_or_post_activation_mixer(
            self,
            tp_input,
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )
        new_edge_features = self.lin_post(new_edge_features)
        weights = self.edge_embed_mlps(latents[active_edges])
        edge_features = self._edge_weighter(new_edge_features, weights)
        if self.use_layer_onehot_tp:
            edge_features = edge_features + _apply_onehot_tp(
                self.edge_onehot_tp,
                edge_features,
                edge_one_hot,
                self.onehot_tp_mode,
            )
        return edge_features, latents, wigner_D_all


class NormFreePairRefineLayer(torch.nn.Module):
    """One bare-add two-body CG refinement with per-path invariant gates."""

    def __init__(
        self,
        node_irreps: Union[str, o3.Irreps],
        edge_irreps: Union[str, o3.Irreps],
        *,
        rank: int = 16,
        radial_dim: int = 4,
        edge_chunk_size: int = 64,
        tail_gate: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.rank = int(rank)
        self.radial_dim = int(radial_dim)
        self.edge_chunk_size = int(edge_chunk_size)
        self.tail_gate = bool(tail_gate)
        if self.rank <= 0:
            raise ValueError(f"refine rank must be positive, got {rank}.")
        if self.radial_dim <= 0:
            raise ValueError(f"refine radial_dim must be positive, got {radial_dim}.")
        if self.edge_chunk_size <= 0:
            raise ValueError(
                f"refine edge_chunk_size must be positive, got {edge_chunk_size}."
            )

        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.node_irreps,
            self.node_irreps,
            self.edge_irreps,
            shared_weights=False,
            internal_weights=False,
        ).to(dtype=dtype, device=device)
        if self.tensor_product.weight_numel <= 0:
            raise ValueError(
                "No compatible node-node-to-edge CG paths for "
                f"{self.node_irreps} x {self.node_irreps} -> {self.edge_irreps}."
            )
        self.path_count = len(self.tensor_product.instructions)
        path_weight_counts = [
            math.prod(instruction.path_shape)
            for instruction in self.tensor_product.instructions
        ]
        if sum(path_weight_counts) != self.tensor_product.weight_numel:
            raise RuntimeError(
                "Unexpected e3nn tensor-product weight layout: instruction path "
                "sizes do not cover weight_numel."
            )
        self.register_buffer(
            "_weight_path_index",
            torch.repeat_interleave(
                torch.arange(self.path_count, dtype=torch.long, device=device),
                torch.tensor(path_weight_counts, dtype=torch.long, device=device),
            ),
            persistent=False,
        )
        self.register_buffer(
            "_node_scalar_indices",
            _scalar_0e_indices(self.node_irreps, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_edge_scalar_indices",
            _scalar_0e_indices(self.edge_irreps, device=device),
            persistent=False,
        )

        condition_dim = (
            2 * int(self._node_scalar_indices.numel())
            + int(self._edge_scalar_indices.numel())
            + self.radial_dim
        )
        factory_kwargs = {"dtype": dtype, "device": device}
        self.condition_down = torch.nn.Linear(
            condition_dim, self.rank, bias=True, **factory_kwargs
        )
        self.path_gate_up = torch.nn.Linear(
            self.rank, self.path_count, bias=True, **factory_kwargs
        )
        self.static_weights = torch.nn.Parameter(
            torch.empty(self.tensor_product.weight_numel, **factory_kwargs)
        )
        (
            self.path_tensor_product,
            path_output_index,
            path_gate_index,
        ) = self._build_path_tensor_product(dtype=dtype, device=device)
        self.register_buffer(
            "_path_output_index",
            path_output_index,
            persistent=False,
        )
        self.register_buffer(
            "_path_gate_index",
            path_gate_index,
            persistent=False,
        )
        self.amplitude_gate = (
            torch.nn.Linear(self.rank, 1, bias=True, **factory_kwargs)
            if self.tail_gate
            else None
        )
        self.reset_parameters()

    @property
    def weight_numel(self) -> int:
        return int(self.tensor_product.weight_numel)

    @property
    def dynamic_dof_per_edge(self) -> int:
        return int(self.path_count + (1 if self.tail_gate else 0))

    def _build_path_tensor_product(
        self,
        *,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> tuple[o3.TensorProduct, torch.Tensor, torch.Tensor]:
        """Expose each full-FCTP instruction in a separate output block.

        e3nn stores the final normalization coefficient in
        ``Instruction.path_weight``.  Disabling normalization in the split
        tensor product and supplying its square as the raw instruction weight
        reproduces the original contribution exactly.  This lets one shared
        static weight vector serve every edge; invariant gates are applied to
        the split output coordinates before the original irreps layout is
        recovered with ``scatter_add_``.
        """
        path_output_terms = []
        path_instructions = []
        output_index = []
        gate_index = []
        edge_slices = self.edge_irreps.slices()
        for path, instruction in enumerate(self.tensor_product.instructions):
            mul, irrep = self.edge_irreps[instruction.i_out]
            path_output_terms.append((mul, irrep))
            path_instructions.append(
                (
                    instruction.i_in1,
                    instruction.i_in2,
                    path,
                    instruction.connection_mode,
                    True,
                    float(instruction.path_weight) ** 2,
                )
            )
            edge_slice = edge_slices[instruction.i_out]
            output_index.extend(range(edge_slice.start, edge_slice.stop))
            gate_index.extend([path] * (edge_slice.stop - edge_slice.start))

        path_tensor_product = o3.TensorProduct(
            self.node_irreps,
            self.node_irreps,
            o3.Irreps(path_output_terms),
            path_instructions,
            irrep_normalization="none",
            path_normalization="none",
            shared_weights=True,
            internal_weights=False,
        ).to(dtype=dtype, device=device)
        if path_tensor_product.weight_numel != self.weight_numel:
            raise RuntimeError(
                "Split per-path tensor-product static layout disagrees with "
                f"the full FCTP: {path_tensor_product.weight_numel} != "
                f"{self.weight_numel}."
            )
        # This tensor product is fully determined by the irreps and owns no
        # learned state.  e3nn registers generated constants (output masks and
        # Wigner tensors) as persistent buffers by default; marking them
        # non-persistent keeps the pre-optimization state_dict key set stable.
        for module in path_tensor_product.modules():
            module._non_persistent_buffers_set.update(module._buffers.keys())
        return (
            path_tensor_product,
            torch.tensor(output_index, dtype=torch.long, device=device),
            torch.tensor(gate_index, dtype=torch.long, device=device),
        )

    def reset_parameters(self) -> None:
        self.condition_down.reset_parameters()
        self.path_gate_up.reset_parameters()
        if self.amplitude_gate is not None:
            self.amplitude_gate.reset_parameters()
        torch.nn.init.normal_(
            self.static_weights,
            mean=0.0,
            std=1.0 / math.sqrt(float(self.weight_numel)),
        )

    def _condition(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        active_edge_index: torch.Tensor,
        active_edge_vector: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = active_edge_index
        length_unit = active_edge_vector.norm(dim=-1, keepdim=True)
        length_unit = length_unit / (1.0 + length_unit)
        radial = torch.cat(
            [length_unit.pow(power) for power in range(1, self.radial_dim + 1)],
            dim=-1,
        )
        condition = torch.cat(
            (
                node_features.index_select(0, src).index_select(
                    1, self._node_scalar_indices
                ),
                node_features.index_select(0, dst).index_select(
                    1, self._node_scalar_indices
                ),
                edge_features.index_select(1, self._edge_scalar_indices),
                radial,
            ),
            dim=-1,
        )
        return torch.nn.functional.silu(self.condition_down(condition))

    def _prepare_forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        active_edges = active_edges.to(
            device=edge_index.device, dtype=torch.long
        ).reshape(-1)
        active_edge_index = edge_index.index_select(1, active_edges)
        active_edge_vector = edge_vector.index_select(0, active_edges)
        if edge_features.shape != (active_edges.numel(), self.edge_irreps.dim):
            raise ValueError(
                "refinement edge_features must align with active_edges and "
                f"edge_irreps={self.edge_irreps}; got {tuple(edge_features.shape)}."
            )
        if active_edges.numel() == 0:
            return (
                active_edges,
                active_edge_index,
                edge_features.new_empty((0, self.path_count)),
                edge_features.new_empty((0,), dtype=torch.long),
                edge_features.new_empty((0,), dtype=torch.long),
                None,
            )

        invariant = self._condition(
            node_features,
            edge_features,
            active_edge_index,
            active_edge_vector,
        )
        path_gates = 1.0 + torch.tanh(self.path_gate_up(invariant))
        amplitude = (
            2.0 * torch.sigmoid(self.amplitude_gate(invariant))
            if self.amplitude_gate is not None
            else None
        )
        src, dst = active_edge_index
        return active_edges, active_edge_index, path_gates, src, dst, amplitude

    def _forward_materialized(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Reference implementation retaining the former expanded-weight path."""
        prepared = self._prepare_forward(
            node_features,
            edge_features,
            edge_index,
            edge_vector,
            active_edges,
        )
        active_edges = prepared[0]
        if active_edges.numel() == 0:
            return edge_features
        _, _, path_gates, src, dst, amplitude = prepared
        refinements = []
        for start in range(0, active_edges.numel(), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, active_edges.numel())
            expanded_gates = path_gates[start:stop].index_select(
                1, self._weight_path_index
            )
            weights = expanded_gates * self.static_weights.unsqueeze(0)
            update = self.tensor_product(
                node_features.index_select(0, src[start:stop]),
                node_features.index_select(0, dst[start:stop]),
                weights,
            )
            if amplitude is not None:
                update = update * amplitude[start:stop]
            refinements.append(update)
        refinement = torch.cat(refinements, dim=0)

        # Eq. 14 norm-free tail: exact unit-coefficient residual addition.
        return edge_features + refinement

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self._prepare_forward(
            node_features,
            edge_features,
            edge_index,
            edge_vector,
            active_edges,
        )
        active_edges = prepared[0]
        if active_edges.numel() == 0:
            return edge_features
        _, _, path_gates, src, dst, amplitude = prepared

        # edge_chunk_size now bounds split path outputs only.  No
        # [chunk, weight_numel] tensor is materialized: the shared static
        # vector is consumed once by e3nn, then per-path gates are applied to
        # output coordinates before restoring the original irreps layout.
        refinements = []
        for start in range(0, active_edges.numel(), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, active_edges.numel())
            path_features = self.path_tensor_product(
                node_features.index_select(0, src[start:stop]),
                node_features.index_select(0, dst[start:stop]),
                self.static_weights,
            )
            component_gates = path_gates[start:stop].index_select(
                1, self._path_gate_index
            )
            path_features = path_features * component_gates
            update = edge_features.new_zeros(
                (stop - start, self.edge_irreps.dim)
            )
            update.scatter_add_(
                1,
                self._path_output_index.expand(stop - start, -1),
                path_features,
            )
            if amplitude is not None:
                update = update * amplitude[start:stop]
            refinements.append(update)
        refinement = torch.cat(refinements, dim=0)

        # Eq. 14 norm-free tail: exact unit-coefficient residual addition.
        return edge_features + refinement


class TwoStagePairStream(torch.nn.Module):
    """Construct an all-edge pair stream from mature node and edge states."""

    def __init__(
        self,
        *,
        num_types: int,
        node_irreps: Union[str, o3.Irreps],
        edge_irreps: Union[str, o3.Irreps],
        latent_dim: int,
        norm_eps: float = 1.0e-8,
        latent_channels: Sequence[int] = (128, 128),
        radial_emb: bool = False,
        radial_channels: Sequence[int] = (128, 128),
        use_layer_onehot_tp: bool = True,
        edge_one_hot_dim: int = 128,
        equivariant_norm_type: str = "none",
        activation_type: str = "gate",
        swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
        swiglu_s2_compat_mode: str = "modern",
        so2_wigner_apply_mode: str = "compact_blocks",
        so2_fusion_mode: str = "streamed_m_major_cueq",
        mole_linear_mode: Optional[str] = "cueq_indexed_linear",
        so2_expert_mixing_mode: str = "pre_activation",
        so2_expert_route_chunk_size: Optional[int] = None,
        so2_expert_route_checkpoint: bool = False,
        so2_output_router_hidden_dim: int = 32,
        onehot_tp_mode: Optional[str] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        num_experts: int = 8,
        num_shared_experts: int = 1,
        n_refine_layers: int = 2,
        refine_rank: int = 16,
        refine_condition: str = "scalar_0e",
        refine_radial_dim: int = 4,
        refine_edge_chunk_size: int = 64,
        tail_gate: bool = False,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.n_refine_layers = int(n_refine_layers)
        if self.n_refine_layers < 0:
            raise ValueError(
                f"n_refine_layers must be non-negative, got {n_refine_layers}."
            )
        refine_condition = str(refine_condition).strip().lower()
        if refine_condition != "scalar_0e":
            raise ValueError(
                "two-stage pair refinement supports only "
                f"refine_condition='scalar_0e', got {refine_condition!r}."
            )

        # PairUpdateEdge only changes residual coefficients.  With
        # res_update=False that override is unreachable, so using UpdateEdge
        # preserves the exact Eq. 13 math without importing lem_pair (which
        # would create a lem_moe_v3_h0 <-> lem_pair cycle).
        self.pair_conv = _LatePairUpdateEdge(
            num_types=num_types,
            node_irreps_in=self.node_irreps,
            irreps_in=self.edge_irreps,
            irreps_out=self.edge_irreps,
            latent_dim=int(latent_dim),
            norm_eps=float(norm_eps),
            latent_channels=list(latent_channels),
            radial_emb=bool(radial_emb),
            radial_channels=list(radial_channels),
            res_update=False,
            use_layer_onehot_tp=bool(use_layer_onehot_tp),
            use_interpolation_tp=False,
            edge_one_hot_dim=int(edge_one_hot_dim),
            equivariant_norm_type=equivariant_norm_type,
            activation_type=activation_type,
            swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
            swiglu_s2_compat_mode=swiglu_s2_compat_mode,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            so2_expert_mixing_mode=so2_expert_mixing_mode,
            so2_expert_route_chunk_size=so2_expert_route_chunk_size,
            so2_expert_route_checkpoint=so2_expert_route_checkpoint,
            so2_output_router_hidden_dim=so2_output_router_hidden_dim,
            onehot_tp_mode=onehot_tp_mode,
            dtype=dtype,
            device=device,
            num_experts=int(num_experts),
            num_shared_experts=int(num_shared_experts),
        )
        self.refine_layers = torch.nn.ModuleList(
            [
                NormFreePairRefineLayer(
                    self.node_irreps,
                    self.edge_irreps,
                    rank=refine_rank,
                    radial_dim=refine_radial_dim,
                    edge_chunk_size=refine_edge_chunk_size,
                    tail_gate=tail_gate,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(self.n_refine_layers)
            ]
        )

    def forward(
        self,
        latents: torch.Tensor,
        node_features: torch.Tensor,
        node_onehot: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        cutoff_coeffs: torch.Tensor,
        active_edges: torch.Tensor,
        edge_one_hot: torch.Tensor,
        wigner_D_all,
        mole_globals,
    ) -> torch.Tensor:
        """Return Eq. 13 pair features on every active ordered edge.

        ``edge_features`` is deliberately the seed passed into the fresh pair
        convolution.  It carries row-specific H0/residual/flow-time state and
        must never be replaced by zeros.
        """

        active_edges = active_edges.reshape(-1).to(
            device=edge_index.device, dtype=torch.long
        )
        if edge_features.ndim != 2 or edge_features.shape != (
            active_edges.numel(),
            self.edge_irreps.dim,
        ):
            raise ValueError(
                "edge_features must be the current active-edge state with shape "
                f"[{active_edges.numel()}, {self.edge_irreps.dim}], got "
                f"{tuple(edge_features.shape)}."
            )
        if node_features.ndim != 2 or node_features.shape[-1] != self.node_irreps.dim:
            raise ValueError(
                f"node_features must end in dimension {self.node_irreps.dim}, "
                f"got {tuple(node_features.shape)}."
            )

        pair_features, _, _ = self.pair_conv(
            latents,
            node_features,
            node_onehot,
            edge_features,
            edge_index,
            edge_vector,
            cutoff_coeffs,
            active_edges,
            edge_one_hot,
            wigner_D_all,
            mole_globals,
        )
        for refine_layer in self.refine_layers:
            pair_features = refine_layer(
                node_features,
                pair_features,
                edge_index,
                edge_vector,
                active_edges,
            )
        return pair_features
