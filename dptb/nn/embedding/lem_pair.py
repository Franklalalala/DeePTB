"""Pair-centric LEM embedding with private MP topology and mature pair heads."""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional, Union

import torch
from torch_scatter import scatter

from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3 import InitLayer, Layer, UpdateEdge
from .lem_moe_v3_h0 import LemMoEV3H0
from .pair_so3_refine import PairSO3RefineTP


Cutoff = Union[float, int, Dict[str, float]]


def _get_mp_edge_mask(
    edge_length: torch.Tensor,
    bond_type: torch.Tensor,
    idp: Any,
    mp_cutoff: Cutoff,
) -> torch.Tensor:
    """Return a private message-passing mask in stored edge-row order."""
    edge_length = edge_length.reshape(-1)
    bond_type = bond_type.reshape(-1)
    if isinstance(mp_cutoff, (float, int)):
        return edge_length < float(mp_cutoff)
    if isinstance(mp_cutoff, dict):
        mask = torch.zeros_like(edge_length, dtype=torch.bool)
        for bond, type_index in idp.bond_to_type.items():
            atoms = bond.split("-")
            if len(atoms) != 2:
                continue
            cutoff_i = mp_cutoff.get(atoms[0])
            cutoff_j = mp_cutoff.get(atoms[1])
            if cutoff_i is None or cutoff_j is None:
                continue
            pair_cutoff = 0.5 * (float(cutoff_i) + float(cutoff_j))
            mask |= (bond_type == int(type_index)) & (edge_length < pair_cutoff)
        return mask
    raise TypeError(
        "mp_cutoff must be a scalar or an element-cutoff dictionary; "
        f"got {type(mp_cutoff)!r}."
    )


class PairInitLayer(InitLayer):
    """Legacy init plus MP-only node aggregation for a real cutoff split."""

    _pair_owner_ref = None

    def forward(
        self,
        edge_index,
        atom_type,
        bond_type,
        edge_sh,
        edge_length,
        edge_one_hot,
        active_edges: Optional[torch.Tensor] = None,
        cutoff_coeffs: Optional[torch.Tensor] = None,
    ):
        owner = self._pair_owner_ref() if self._pair_owner_ref is not None else None
        if owner is None or owner.mp_cutoff is None:
            return super().forward(
                edge_index,
                atom_type,
                bond_type,
                edge_sh,
                edge_length,
                edge_one_hot,
                active_edges,
                cutoff_coeffs,
            )

        mp_mask = _get_mp_edge_mask(
            edge_length=edge_length,
            bond_type=bond_type,
            idp=self.idp,
            mp_cutoff=owner.mp_cutoff,
        )
        owner._last_mp_mask = mp_mask.detach()
        owner._pair_run_dual = not bool(mp_mask.all().item())
        if not owner._pair_run_dual:
            return super().forward(
                edge_index,
                atom_type,
                bond_type,
                edge_sh,
                edge_length,
                edge_one_hot,
                active_edges,
                cutoff_coeffs,
            )

        edge_center = edge_index[0]
        edge_invariants = self.bessel(edge_length)
        if cutoff_coeffs is None:
            cutoff_coeffs = self.cutoff_coefficients(edge_length, bond_type)
        else:
            cutoff_coeffs = cutoff_coeffs.to(
                device=edge_length.device, dtype=edge_length.dtype
            ).reshape(-1)
        if active_edges is None:
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        else:
            active_edges = active_edges.to(
                device=edge_length.device, dtype=torch.long
            ).reshape(-1)

        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )
        new_latents = self.two_body_latent(
            torch.cat(
                [edge_one_hot[active_edges], edge_invariants[active_edges]], dim=-1
            )
        )
        latents = torch.index_copy(
            latents,
            0,
            active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents,
        )
        weights_e = self.env_embed_mlp(latents[active_edges])
        edge_features = self._env_weighter(edge_sh[active_edges], weights_e)

        mp_active_mask = mp_mask.index_select(0, active_edges)
        mp_edges = active_edges[mp_active_mask]
        node_features = scatter(
            edge_features[mp_active_mask],
            edge_center[mp_edges],
            dim=0,
            dim_size=atom_type.numel(),
        )
        node_features = node_features * torch.as_tensor(
            owner.mp_avg_num_neighbors,
            dtype=node_features.dtype,
            device=node_features.device,
        ).rsqrt()
        return latents, node_features, edge_features, cutoff_coeffs, active_edges


class PairLayer(Layer):
    """Run legacy layer math on the private MP subset and read all pairs last."""

    _pair_owner_ref = None
    _pair_is_last = False

    def forward(
        self,
        latents,
        node_features,
        edge_features,
        node_onehot,
        edge_index,
        edge_vector,
        atom_type,
        cutoff_coeffs,
        active_edges,
        edge_one_hot,
        wigner_D_all,
        mole_globals,
        node_batch=None,
    ):
        owner = self._pair_owner_ref() if self._pair_owner_ref is not None else None
        if owner is None or not owner._pair_run_dual:
            return super().forward(
                latents,
                node_features,
                edge_features,
                node_onehot,
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

        mp_active_mask = owner._last_mp_mask.index_select(0, active_edges)
        mp_edges = active_edges[mp_active_mask]
        if edge_features.shape[0] == active_edges.numel():
            edge_features = edge_features[mp_active_mask]
        if edge_one_hot.shape[0] == active_edges.numel():
            mp_edge_one_hot = edge_one_hot[mp_active_mask]
        else:
            mp_edge_one_hot = edge_one_hot
        if node_batch is None:
            raise ValueError("lem_pair dual cutoff requires node batch metadata.")
        mp_edge_batch = node_batch.index_select(0, edge_index[0][mp_edges])
        mp_edge_sizes = torch.bincount(
            mp_edge_batch, minlength=int(mole_globals.coefficients.shape[0])
        )
        mp_mole_globals = MOLEGlobals(
            coefficients=mole_globals.coefficients,
            sizes=mp_edge_sizes,
            graph_index=mp_edge_batch,
            topk_indices=mole_globals.topk_indices,
            topk_values=mole_globals.topk_values,
        )

        old_normalization = self.node_update.env_sum_normalizations
        self.node_update.env_sum_normalizations = torch.as_tensor(
            owner.mp_avg_num_neighbors,
            dtype=node_features.dtype,
            device=node_features.device,
        ).rsqrt()
        try:
            latents, node_features, edge_features, wigner_D_all = super().forward(
                latents,
                node_features,
                edge_features,
                node_onehot,
                edge_index,
                edge_vector,
                atom_type,
                cutoff_coeffs,
                mp_edges,
                mp_edge_one_hot,
                wigner_D_all,
                mp_mole_globals,
                node_batch,
            )
        finally:
            self.node_update.env_sum_normalizations = old_normalization

        if self._pair_is_last:
            readout_input = node_features.new_zeros(
                active_edges.numel(), self.irreps_out.dim
            )
            edge_features, _, _ = owner.dual_cutoff_pair_readout(
                latents,
                node_features,
                node_onehot,
                readout_input,
                edge_index,
                edge_vector,
                cutoff_coeffs,
                active_edges,
                edge_one_hot,
                None,
                mole_globals,
            )
            edge_features = (
                edge_features * owner.dual_cutoff_readout_normalization
            )
        return latents, node_features, edge_features, wigner_D_all


@Embedding.register("lem_pair")
class LemPair(LemMoEV3H0):
    """H0 LEM whose backbone topology and pair heads are independently tunable."""

    pair_refine_enable = False

    @staticmethod
    def _init_layer_type():
        return PairInitLayer

    @staticmethod
    def _layer_type():
        return PairLayer

    def __init__(
        self,
        *,
        avg_num_neighbors: Optional[float] = None,
        mp_cutoff: Optional[Cutoff] = None,
        mp_avg_num_neighbors: Optional[float] = None,
        pair_refine_enable: bool = False,
        pair_refine_rank: int = 16,
        pair_refine_condition: str = "scalar_0e",
        pair_refine_internal_weights: bool = True,
        pair_refine_init: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(avg_num_neighbors=avg_num_neighbors, **kwargs)
        self.mp_cutoff = mp_cutoff
        self.mp_avg_num_neighbors = (
            avg_num_neighbors
            if mp_avg_num_neighbors is None
            else float(mp_avg_num_neighbors)
        )
        self._last_mp_mask = None
        self._pair_run_dual = False
        self.dual_cutoff_pair_readout = None
        self.pair_refine_enable = bool(pair_refine_enable)

        if self.mp_cutoff is not None:
            final_irreps = self.layers[-1].irreps_out
            self.dual_cutoff_pair_readout = UpdateEdge(
                num_types=self.n_atom,
                node_irreps_in=final_irreps,
                irreps_in=final_irreps,
                irreps_out=final_irreps,
                latent_dim=self.latent_dim,
                norm_eps=kwargs.get("norm_eps", 1e-8),
                latent_channels=kwargs.get("latent_channels", [128, 128]),
                radial_emb=kwargs.get("tp_radial_emb", False),
                radial_channels=kwargs.get("tp_radial_channels", [128, 128]),
                res_update=False,
                use_layer_onehot_tp=kwargs.get("use_layer_onehot_tp", True),
                use_interpolation_tp=False,
                edge_one_hot_dim=kwargs.get("edge_one_hot_dim", 128),
                equivariant_norm_type=kwargs.get("equivariant_norm_type", "none"),
                activation_type="gate",
                swiglu_s2_grid_resolution=kwargs.get(
                    "swiglu_s2_grid_resolution", (14, 14)
                ),
                swiglu_s2_compat_mode=kwargs.get(
                    "swiglu_s2_compat_mode", "modern"
                ),
                so2_wigner_apply_mode=kwargs.get(
                    "so2_wigner_apply_mode", "compact_blocks"
                ),
                so2_fusion_mode=kwargs.get(
                    "so2_fusion_mode", "streamed_m_major_cueq"
                ),
                mole_linear_mode=kwargs.get(
                    "mole_linear_mode", "cueq_indexed_linear"
                ),
                so2_expert_mixing_mode=kwargs.get(
                    "so2_expert_mixing_mode", "pre_activation"
                ),
                so2_expert_route_chunk_size=kwargs.get(
                    "so2_expert_route_chunk_size", None
                ),
                so2_expert_route_checkpoint=kwargs.get(
                    "so2_expert_route_checkpoint", False
                ),
                so2_output_router_hidden_dim=kwargs.get(
                    "so2_output_router_hidden_dim", 32
                ),
                onehot_tp_mode=kwargs.get("onehot_tp_mode", None),
                dtype=self.dtype,
                device=self.device,
                num_experts=kwargs.get("num_experts", 8),
                num_shared_experts=kwargs.get("num_shared_experts", 1),
            )
            self.register_buffer(
                "dual_cutoff_readout_normalization",
                torch.as_tensor(
                    avg_num_neighbors, dtype=self.dtype, device=self.device
                ).rsqrt(),
            )

        if self.pair_refine_enable:
            if not self.use_h0_init or self.output_route_name != "h_b0":
                raise ValueError(
                    "pair_refine_enable=true requires the LemPair H-B0 route "
                    "with H0 initialization enabled."
                )
            final_irreps = self.layers[-1].irreps_out
            self.pair_refine = PairSO3RefineTP(
                node_irreps=final_irreps,
                edge_irreps=final_irreps,
                rank=pair_refine_rank,
                condition=pair_refine_condition,
                internal_weights=pair_refine_internal_weights,
                dynamic_init=pair_refine_init,
                dtype=self.dtype,
                device=self.device,
            )

        owner_ref = weakref.ref(self)
        base_init = getattr(self.init_layer, "base_init", self.init_layer)
        base_init._pair_owner_ref = owner_ref
        for index, layer in enumerate(self.layers):
            layer._pair_owner_ref = owner_ref
            layer._pair_is_last = index == len(self.layers) - 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Dispatch the legacy skeleton through pair-specific init/layers."""
        self._pair_run_dual = False
        return super().forward(data)

    def _apply_block_native_output_heads(
        self,
        node_features,
        edge_features,
        atom_type,
        edge_index,
        active_edges,
    ):
        if self.pair_refine_enable:
            active_edge_index = edge_index.index_select(
                1, active_edges.to(device=edge_index.device, dtype=torch.long)
            )
            edge_features = self.pair_refine(
                node_features, edge_features, active_edge_index
            )
        return super()._apply_block_native_output_heads(
            node_features, edge_features, atom_type, edge_index, active_edges
        )
