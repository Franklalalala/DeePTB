"""Pair-centric LEM embedding with private MP topology and mature pair heads."""

from __future__ import annotations

import copy
import math
import logging
from typing import Any, Dict, Optional, Union

import torch
from e3nn.o3 import Linear
from e3nn.o3._spherical_harmonics import (
    _spherical_harmonics as _e3nn_spherical_harmonics,
)
from torch_scatter import scatter

from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3 import InitLayer, Layer, UpdateEdge, UpdateNode
from .lem_moe_v3_h0 import LemMoEV3H0
from .pair_so3_refine import PairSO3RefineTP


Cutoff = Union[float, int, Dict[str, float]]
log = logging.getLogger(__name__)


def _positive_finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{label} must be a real scalar, got {type(value)!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive, got {value!r}.")
    return result


def _validated_mp_cutoff(mp_cutoff: Optional[Cutoff], idp: Any) -> Optional[Cutoff]:
    if mp_cutoff is None:
        return None
    if isinstance(mp_cutoff, bool):
        raise TypeError("mp_cutoff must not be boolean.")
    if isinstance(mp_cutoff, (float, int)):
        return _positive_finite_scalar(mp_cutoff, label="mp_cutoff")
    if not isinstance(mp_cutoff, dict):
        raise TypeError(
            "mp_cutoff must be a scalar or an element-cutoff dictionary; "
            f"got {type(mp_cutoff)!r}."
        )
    if not mp_cutoff:
        raise ValueError("mp_cutoff element dictionary must not be empty.")
    normalized: Dict[str, float] = {}
    for symbol, value in mp_cutoff.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise TypeError(
                "mp_cutoff element keys must be non-empty chemical-symbol strings; "
                f"got {symbol!r}."
            )
        normalized[symbol] = _positive_finite_scalar(
            value, label=f"mp_cutoff[{symbol!r}]"
        )
    required_symbols = set(idp.basis)
    missing = sorted(required_symbols.difference(normalized))
    if missing:
        raise ValueError(
            "mp_cutoff element dictionary must cover every basis species; "
            f"missing {missing}."
        )
    return normalized


def _pair_cutoff_value(
    cutoff: Any,
    atom_i: str,
    atom_j: str,
) -> Optional[float]:
    """Return the configured pair cutoff, or ``None`` when it is not provable."""
    if isinstance(cutoff, bool):
        return None
    if isinstance(cutoff, (float, int)):
        value = float(cutoff)
        return value if math.isfinite(value) else None
    if not isinstance(cutoff, dict):
        return None
    if atom_i not in cutoff or atom_j not in cutoff:
        return None
    try:
        value_i = float(cutoff[atom_i])
        value_j = float(cutoff[atom_j])
    except (TypeError, ValueError):
        return None
    pair_cutoff = 0.5 * (value_i + value_j)
    return pair_cutoff if math.isfinite(pair_cutoff) else None


def _canonicalize_mp_cutoff(
    mp_cutoff: Optional[Cutoff],
    r_max: Any,
    idp: Any,
) -> Optional[Cutoff]:
    """Disable a private cutoff only when every represented pair is redundant."""
    normalized = _validated_mp_cutoff(mp_cutoff, idp)
    if normalized is None:
        return None
    for bond in idp.bond_to_type:
        atoms = bond.split("-")
        if len(atoms) != 2:
            return normalized
        mp_pair = _pair_cutoff_value(normalized, atoms[0], atoms[1])
        head_pair = _pair_cutoff_value(r_max, atoms[0], atoms[1])
        if mp_pair is None or head_pair is None or mp_pair < head_pair:
            return normalized
    log.warning(
        "mp_cutoff is not smaller than r_max for every represented element "
        "pair; canonicalizing mp_cutoff to None (legacy architecture)."
    )
    return None


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

    mp_cutoff = None
    mp_avg_num_neighbors = None

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
        if self.mp_cutoff is None:
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
            mp_cutoff=self.mp_cutoff,
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
            self.mp_avg_num_neighbors,
            dtype=node_features.dtype,
            device=node_features.device,
        ).rsqrt()
        return latents, node_features, edge_features, cutoff_coeffs, active_edges


class PairUpdateNode(UpdateNode):
    """Node update with an opt-in unscaled residual."""

    res_update_additive = False

    def _residual_coefficients(self):
        if self.res_update_additive:
            return 1.0, 1.0
        return super()._residual_coefficients()


class PairUpdateEdge(UpdateEdge):
    """Edge/latent update with an opt-in unscaled residual."""

    res_update_additive = False

    def _residual_coefficients(self):
        if self.res_update_additive:
            return 1.0, 1.0
        return super()._residual_coefficients()


class PairLayer(Layer):
    """Run legacy layer math on the private MP subset and read all pairs last."""

    mp_cutoff = None
    _pair_is_last = False

    @staticmethod
    def _edge_update_type():
        return PairUpdateEdge

    @staticmethod
    def _node_update_type():
        return PairUpdateNode

    def configure_dual_cutoff(self, mp_cutoff: Cutoff, idp: Any) -> None:
        """Store an immutable cutoff copy and a type-pair lookup buffer."""
        self.mp_cutoff = copy.deepcopy(mp_cutoff)
        template = self.node_update.env_sum_normalizations
        table = template.new_empty((len(idp.type_names), len(idp.type_names)))
        if isinstance(self.mp_cutoff, (float, int)):
            table.fill_(float(self.mp_cutoff))
        else:
            for atom_i, type_i in idp.chemical_symbol_to_type.items():
                for atom_j, type_j in idp.chemical_symbol_to_type.items():
                    table[type_i, type_j] = 0.5 * (
                        float(self.mp_cutoff[atom_i])
                        + float(self.mp_cutoff[atom_j])
                    )
        self.register_buffer("_mp_cutoff_by_type", table, persistent=False)

    def _mp_active_mask(
        self,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        atom_type: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        active_edges = active_edges.to(
            device=edge_index.device, dtype=torch.long
        ).reshape(-1)
        active_edge_index = edge_index.index_select(1, active_edges)
        src_type = atom_type.index_select(0, active_edge_index[0])
        dst_type = atom_type.index_select(0, active_edge_index[1])
        pair_cutoff = self._mp_cutoff_by_type[
            src_type.to(dtype=torch.long), dst_type.to(dtype=torch.long)
        ].to(device=edge_vector.device, dtype=edge_vector.dtype)
        edge_length = edge_vector.index_select(0, active_edges).norm(dim=-1)
        return edge_length < pair_cutoff

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
        if self.mp_cutoff is None:
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

        if isinstance(edge_features, tuple):
            edge_features, initial_edge_context = edge_features
        else:
            initial_edge_context = edge_features
        mp_active_mask = self._mp_active_mask(
            edge_index, edge_vector, atom_type, active_edges
        )
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

        if self._pair_is_last:
            if initial_edge_context.shape[0] != active_edges.numel():
                raise ValueError(
                    "lem_pair dual cutoff full-edge context and active-edge rows "
                    f"disagree: {initial_edge_context.shape[0]} != "
                    f"{active_edges.numel()}."
                )
            readout_input = self.dual_cutoff_edge_context_projection(
                initial_edge_context
            )
            mp_positions = torch.nonzero(
                mp_active_mask, as_tuple=False
            ).flatten()
            if edge_features.shape[0] != mp_positions.numel():
                raise ValueError(
                    "lem_pair dual cutoff mature MP edge rows and mask disagree: "
                    f"{edge_features.shape[0]} != {mp_positions.numel()}."
                )
            readout_input = torch.index_copy(
                readout_input, 0, mp_positions, edge_features
            )
            edge_features, _, _ = self.dual_cutoff_pair_readout(
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
                edge_features * self.dual_cutoff_readout_normalization
            )
        else:
            edge_features = (edge_features, initial_edge_context)
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
        res_update_additive: bool = False,
        latents_layernorm: bool = True,
        pair_refine_enable: bool = False,
        pair_refine_rank: int = 16,
        pair_refine_condition: str = "scalar_0e",
        pair_refine_internal_weights: bool = True,
        pair_refine_init: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if (
            res_update_additive
            and bool(kwargs.get("res_update", True))
            and bool(kwargs.get("res_update_ratios_learnable", False))
        ):
            raise ValueError(
                "res_update_additive=true bypasses the learned residual-ratio "
                "parameters; set res_update_ratios_learnable=false."
            )
        r_max = kwargs.get("r_max", 5.0)
        super().__init__(avg_num_neighbors=avg_num_neighbors, **kwargs)
        # e3nn stores a ScriptFunction cache on SphericalHarmonics, which
        # cannot be deep-copied or pickled. The underlying Python function is
        # numerically identical and makes whole-model lifecycle operations work.
        self.sh.sph_func = _e3nn_spherical_harmonics
        self.mp_cutoff = _canonicalize_mp_cutoff(mp_cutoff, r_max, self.idp)
        neighbor_count = (
            avg_num_neighbors
            if mp_avg_num_neighbors is None
            else mp_avg_num_neighbors
        )
        self.mp_avg_num_neighbors = _positive_finite_scalar(
            neighbor_count, label="mp_avg_num_neighbors"
        )
        self.res_update_additive = bool(res_update_additive)
        self.latents_layernorm = bool(latents_layernorm)
        self.pair_refine_enable = bool(pair_refine_enable)

        for layer in self.layers:
            layer.res_update_additive = self.res_update_additive
            layer.latents_layernorm = self.latents_layernorm
            layer.node_update.res_update_additive = self.res_update_additive
            layer.edge_update.res_update_additive = self.res_update_additive
            if not self.latents_layernorm:
                layer.edge_update.ln = torch.nn.Identity()

        if self.mp_cutoff is not None:
            base_init = getattr(self.init_layer, "base_init", self.init_layer)
            base_init.mp_cutoff = copy.deepcopy(self.mp_cutoff)
            base_init.mp_avg_num_neighbors = self.mp_avg_num_neighbors
            mp_normalization = torch.as_tensor(
                self.mp_avg_num_neighbors,
                dtype=self.dtype,
                device=self.device,
            ).rsqrt()
            for layer in self.layers:
                layer.configure_dual_cutoff(self.mp_cutoff, self.idp)
                # Dual topology is a construction-time choice, so the MP
                # aggregation normalization is a permanent layer constant.
                layer.node_update.env_sum_normalizations = mp_normalization.clone()
            final_irreps = self.layers[-1].irreps_out
            final_layer = self.layers[-1]
            final_layer.dual_cutoff_edge_context_projection = Linear(
                base_init.irreps_out,
                final_irreps,
                shared_weights=True,
                internal_weights=True,
                biases=False,
            ).to(dtype=self.dtype, device=self.device)
            final_layer.dual_cutoff_pair_readout = PairUpdateEdge(
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
            # The fresh readout has res_update=False and discards its returned
            # latent, so residual-additive and latent-LN switches do not apply.
            final_layer.register_buffer(
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

        for index, layer in enumerate(self.layers):
            layer._pair_is_last = index == len(self.layers) - 1

    @property
    def dual_cutoff_pair_readout(self):
        """Compatibility accessor; parameters are owned by the final pair layer."""
        if self.mp_cutoff is None:
            return None
        return getattr(self.layers[-1], "dual_cutoff_pair_readout", None)

    @property
    def dual_cutoff_edge_context_projection(self):
        """Compatibility accessor; parameters are owned by the final pair layer."""
        if self.mp_cutoff is None:
            return None
        return getattr(
            self.layers[-1], "dual_cutoff_edge_context_projection", None
        )

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
