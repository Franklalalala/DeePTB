"""One authoritative initial, edge-update, and node-update layer implementation."""
from typing import Optional, List, Union, Tuple
import math
import torch
from torch_runstats.scatter import scatter
from e3nn import o3
from e3nn.o3 import TensorProduct
from dptb.nn.e3nn_fast import Linear
from ..radial_basis import BesselBasis
from ..base import ScalarMLPFunction
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from .unitb_activations import (
    build_equivariant_norm,
    build_gate_activation,
)
# Note: Modified SO2_Linear and MOLE classes imported here
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, SO2SlotPostActivationMixer, SO2SharedPostActivationMixer
import math


import logging

log = logging.getLogger(__name__)


from .unitb_ops import _normalize_node_message_aggregation, _normalize_edge_attention_key_source, _normalize_onehot_tp_mode, _normalize_so2_expert_mixing_mode, _build_so2_post_activation_expert_mixer, _apply_so2_tp_or_post_activation_mixer, ScalarOnehotTP, _apply_onehot_tp

@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def _cosine_cutoff_per_edge(
    x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8
) -> torch.Tensor:
    r_decay = r_start_cos_ratio * r_max
    x = torch.minimum(torch.maximum(x, r_decay), r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


def _polynomial_cutoff_per_edge(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    assert p >= 2.0
    x = x / r_max
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * (x < 1.0)


class InitLayer(torch.nn.Module):
    def __init__(
            self,
            # required params
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            avg_num_neighbors: Optional[float] = None,
            irreps_sh: o3.Irreps = None,
            env_embed_multiplicity: int = 32,
            # MLP parameters:
            two_body_latent_channels: list = [128, 128],
            latent_dim: int = 128,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super(InitLayer, self).__init__()
        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        if isinstance(r_max, float) or isinstance(r_max, int):
            max_r_max_value = float(r_max)
            r_max_tensor = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            c_set = set(list(r_max.values()))
            max_r_max_value = max(list(r_max.values()))
            r_max_tensor = torch.tensor(max_r_max_value, device=device, dtype=dtype)
            if len(r_max) == 1 or len(c_set) == 1:
                self.r_max_dict = None
            else:
                self.r_max_dict = {}
                for k, v in r_max.items():
                    self.r_max_dict[k] = torch.tensor(v, device=device, dtype=dtype)
        else:
            raise TypeError("r_max should be either float, int or dict")

        self.idp = idp
        self.register_buffer("r_max", r_max_tensor)
        self._r_max_cpu = r_max_tensor.detach().cpu()
        r_max_by_edge_type = None
        r_max_edge_type_valid = None
        if self.r_max_dict is not None:
            max_edge_type = max(int(v) for v in self.idp.bond_to_type.values())
            edge_type_count = max(max_edge_type + 1, int(num_types) * int(num_types))
            r_max_by_edge_type = torch.zeros(edge_type_count, device=device, dtype=dtype)
            r_max_edge_type_valid = torch.zeros(edge_type_count, device=device, dtype=torch.bool)
            for bond, ty in self.idp.bond_to_type.items():
                iatom, jatom = bond.split("-")
                if iatom not in self.r_max_dict or jatom not in self.r_max_dict:
                    continue
                r_max_by_edge_type[int(ty)] = 0.5 * (self.r_max_dict[iatom] + self.r_max_dict[jatom])
                r_max_edge_type_valid[int(ty)] = True
        self.register_buffer("r_max_by_edge_type", r_max_by_edge_type)
        self.register_buffer("r_max_edge_type_valid", r_max_edge_type_valid)
        self._r_max_by_edge_type_cpu = (
            None if r_max_by_edge_type is None else r_max_by_edge_type.detach().cpu()
        )
        self._r_max_edge_type_valid_cpu = (
            None if r_max_edge_type_valid is None else r_max_edge_type_valid.detach().cpu()
        )
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert all(mul == 1 for mul, _ in irreps_sh)
        assert (
                irreps_sh[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self.two_body_latent = ScalarMLPFunction(
            mlp_input_dimension=(edge_one_hot_dim + n_radial_basis),
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=two_body_latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self._env_weighter = Linear(
            irreps_in=irreps_sh,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
            path_normalization="element",
        )

        self.env_embed_mlp = ScalarMLPFunction(
            mlp_input_dimension=self.two_body_latent.out_features,
            mlp_output_dimension=self._env_weighter.weight_numel,
            mlp_latent_dimensions=[],
            mlp_nonlinearity=None,
            mlp_initialization="uniform",
        )

        self.bessel = BesselBasis(r_max=float(max_r_max_value), num_basis=n_radial_basis, trainable=True)

    def _r_max_for(self, edge_length: torch.Tensor) -> torch.Tensor:
        if edge_length.device.type == "cpu":
            return self._r_max_cpu.to(dtype=edge_length.dtype)
        return self.r_max.to(device=edge_length.device, dtype=edge_length.dtype)

    def _r_max_tables_for(self, edge_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_length.device.type == "cpu":
            return (
                self._r_max_by_edge_type_cpu.to(dtype=edge_length.dtype),
                self._r_max_edge_type_valid_cpu,
            )
        return (
            self.r_max_by_edge_type.to(device=edge_length.device, dtype=edge_length.dtype),
            self.r_max_edge_type_valid.to(device=edge_length.device),
        )

    def cutoff_coefficients(self, edge_length: torch.Tensor, bond_type: torch.Tensor) -> torch.Tensor:
        if self.r_max_dict is None:
            r_max = self._r_max_for(edge_length)
            if self.cutoff_type == "cosine":
                cutoff_coeffs = cosine_cutoff(
                    edge_length,
                    r_max.reshape(-1),
                    r_start_cos_ratio=self.r_start_cos_ratio,
                ).flatten()

            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = polynomial_cutoff(
                    edge_length, r_max.reshape(-1), p=self.polynomial_cutoff_p
                ).flatten()

            else:
                assert False, "Invalid cutoff type"
        else:
            r_max_by_edge_type, r_max_edge_type_valid = self._r_max_tables_for(edge_length)
            bond_type_flat = bond_type.reshape(-1).to(device=edge_length.device, dtype=torch.long)
            edge_length_flat = edge_length.reshape(-1)
            table_size = r_max_by_edge_type.shape[0]
            in_range = (bond_type_flat >= 0) & (bond_type_flat < table_size)
            safe_bond_type = torch.where(in_range, bond_type_flat, torch.zeros_like(bond_type_flat))
            bond_r_max = r_max_by_edge_type.index_select(0, safe_bond_type)
            valid_bond_type = r_max_edge_type_valid.index_select(0, safe_bond_type) & in_range
            safe_bond_r_max = torch.where(
                valid_bond_type,
                bond_r_max.clamp_min(torch.finfo(edge_length.dtype).eps),
                torch.ones_like(bond_r_max),
            )
            safe_edge_length = torch.where(
                valid_bond_type,
                edge_length_flat,
                torch.zeros_like(edge_length_flat),
            )
            if self.cutoff_type == "cosine":
                cutoff_coeffs = _cosine_cutoff_per_edge(
                    safe_edge_length,
                    safe_bond_r_max,
                    r_start_cos_ratio=self.r_start_cos_ratio,
                )
            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = _polynomial_cutoff_per_edge(
                    safe_edge_length,
                    safe_bond_r_max,
                    p=self.polynomial_cutoff_p,
                )
            else:
                assert False, "Invalid cutoff type"
            cutoff_coeffs = cutoff_coeffs * valid_bond_type.to(dtype=cutoff_coeffs.dtype)

        return cutoff_coeffs

    def precompute_cutoff_metadata(
        self,
        edge_length: torch.Tensor,
        bond_type: torch.Tensor,
        compute_cutoff: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            cutoff_coeffs = self.cutoff_coefficients(edge_length, bond_type)
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1).to(dtype=torch.long)
            if compute_cutoff:
                return active_edges, cutoff_coeffs
            return active_edges, None

    def cutoff_config_signature(self):
        table = None
        valid = None
        if self._r_max_by_edge_type_cpu is not None:
            table = tuple(float(v) for v in self._r_max_by_edge_type_cpu.reshape(-1).tolist())
            valid = tuple(bool(v) for v in self._r_max_edge_type_valid_cpu.reshape(-1).tolist())
        return (
            self.cutoff_type,
            float(self.r_start_cos_ratio),
            float(self.polynomial_cutoff_p),
            tuple(float(v) for v in self._r_max_cpu.reshape(-1).tolist()),
            table,
            valid,
        )

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
        edge_center = edge_index[0]

        edge_invariants = self.bessel(edge_length)

        if cutoff_coeffs is None:
            cutoff_coeffs = self.cutoff_coefficients(edge_length, bond_type)
        else:
            cutoff_coeffs = cutoff_coeffs.to(device=edge_length.device, dtype=edge_length.dtype).reshape(-1)

        if active_edges is None:
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        else:
            active_edges = active_edges.to(device=edge_length.device, dtype=torch.long).reshape(-1)

        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )

        new_latents = self.two_body_latent(torch.cat([
            edge_one_hot[active_edges],
            edge_invariants[active_edges],
        ], dim=-1))

        latents = torch.index_copy(
            latents, 0, active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        )

        weights_e = self.env_embed_mlp(latents[active_edges])

        edge_features = self._env_weighter(
            edge_sh[active_edges], weights_e
        )

        node_features = scatter(
            edge_features,
            edge_center[active_edges],
            dim=0,
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        node_features = node_features * norm_const

        return latents, node_features, edge_features, cutoff_coeffs, active_edges


class UpdateNode(torch.nn.Module):
    def __init__(
            self,
            edge_irreps_in: o3.Irreps,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            norm_eps: float = 1e-8,
            radial_emb: bool = False,
            radial_channels: list = [128, 128],
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            avg_num_neighbors: Optional[float] = None,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "streamed_m_major_cueq",
            mole_linear_mode: Optional[str] = "cueq_indexed_linear",
            mole_expert_parameterization: str = "full",
            mole_expert_rank: int = 64,
            so2_expert_mixing_mode: str = "pre_activation",
            so2_expert_route_chunk_size: Optional[int] = None,
            so2_expert_route_checkpoint: bool = False,
            so2_output_router_hidden_dim: int = 32,
            onehot_tp_mode: Optional[str] = None,
            node_message_aggregation: str = "scatter",
            num_focus: int = 1,
            focus_attention_dim: int = 32,
            edge_aggregation_gated_attention: bool = False,
            edge_attention_key_source: str = "message",
            edge_attention_envelope_power: float = 1.0,
            edge_attention_use_latent_bias: bool = True,
            edge_attention_key_layer_norm: bool = False,
            edge_attention_query_layer_norm: bool = False,
            edge_attention_qk_layer_norm: bool = False,
            edge_message_env_weight: bool = True,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
    ):
        super(UpdateNode, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.edge_irreps_in = edge_irreps_in
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        self.onehot_tp_mode = _normalize_onehot_tp_mode(onehot_tp_mode)
        self.so2_expert_mixing_mode = _normalize_so2_expert_mixing_mode(so2_expert_mixing_mode)
        self.node_message_aggregation = _normalize_node_message_aggregation(node_message_aggregation)
        self.edge_aggregation_gated_attention = bool(edge_aggregation_gated_attention)
        self.edge_attention_key_source = _normalize_edge_attention_key_source(edge_attention_key_source)
        self.edge_attention_envelope_power = float(edge_attention_envelope_power)
        self.edge_attention_use_latent_bias = bool(edge_attention_use_latent_bias)
        self.edge_attention_key_layer_norm = bool(edge_attention_key_layer_norm)
        self.edge_attention_query_layer_norm = bool(edge_attention_query_layer_norm)
        self.edge_attention_qk_layer_norm = bool(edge_attention_qk_layer_norm)
        self.edge_message_env_weight = bool(edge_message_env_weight)

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self._env_weighter = E3ElementLinear(
            irreps_in=irreps_out,
            dtype=dtype,
            device=device,
        )

        assert irreps_out[0].ir.l == 0

        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel,
        )

        self.node_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.irreps_in,
            norm_eps,
            dtype,
            device,
        )
        self.edge_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.edge_irreps_in,
            norm_eps,
            dtype,
            device,
        )

        if activation_type != "gate":
            raise ValueError("Only gate activation is supported; grid activations belong to archived models")
        self.activation = build_gate_activation(self.irreps_out)

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in + self.edge_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=0,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            mole_expert_parameterization=mole_expert_parameterization,
            mole_expert_rank=mole_expert_rank,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.post_activation_expert_mixer = None
        if self.so2_expert_mixing_mode == "post_activation":
            scalar_dim = self.activation.irreps_out[0].dim
            self.post_activation_expert_mixer = _build_so2_post_activation_expert_mixer(
                self.tp,
                self.activation,
                scalar_dim,
                so2_output_router_hidden_dim,
                so2_expert_route_chunk_size,
                so2_expert_route_checkpoint,
            )
        elif self.so2_expert_mixing_mode == "post_activation_slot":
            # per-row top-k slots activated separately, mixed with the router's own coefficients
            if so2_expert_route_checkpoint or so2_expert_route_chunk_size:
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_slot' implements neither "
                    "so2_expert_route_checkpoint nor so2_expert_route_chunk_size; unset them."
                )
            self.post_activation_expert_mixer = SO2SlotPostActivationMixer(self.tp, self.activation)
        elif self.so2_expert_mixing_mode == "post_activation_shared":
            # separate shared branch + per-slot routed experts, each activated (DPA3-MoE eq. 4)
            if so2_expert_route_checkpoint or so2_expert_route_chunk_size:
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_shared' implements neither "
                    "so2_expert_route_checkpoint nor so2_expert_route_chunk_size; unset them."
                )
            self.post_activation_expert_mixer = SO2SharedPostActivationMixer(self.tp, self.activation)

        if int(num_focus) != 1 or self.edge_aggregation_gated_attention:
            raise ValueError("Focus gates and gated edge aggregation belong to archived models")

        # Legacy checkpoints persist this identity-route index under the old module path.
        self.focus_gate = torch.nn.Module()
        self.focus_gate.register_buffer(
            "focus_index", torch.zeros(self.irreps_out.dim, dtype=torch.long, device=device)
        )

        if res_update:
            self.linear_res = Linear(
                self.irreps_in,
                self.irreps_out,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )

        if res_update_ratios is None:
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )
        self.use_layer_onehot_tp = use_layer_onehot_tp
        if use_layer_onehot_tp:
            instructions = []
            for i, (mul, ir) in enumerate(self.irreps_out):
                instructions.append((i, 0, i, 'uvu', True))
            self.node_onehot_tp = ScalarOnehotTP.from_e3nn(TensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=f'95x0e',
                irreps_out=self.irreps_out,
                instructions=instructions
            ))
        self.use_identity_res = (self.irreps_in == self.irreps_out) and res_update
        if not self.use_identity_res:
            if res_update:
                self.linear_res = Linear(
                    self.irreps_in,
                    self.irreps_out,
                    shared_weights=True,
                    internal_weights=True,
                    biases=True,
                )

    def _residual_coefficients(self):
        update_coefficients = self._res_update_params.sigmoid()
        coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
        coefficient_new = update_coefficients * coefficient_old
        return coefficient_old, coefficient_new

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                cutoff_coeffs, active_edges, wigner_D_all, mole_globals, node_batch=None):  # Accept globals
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = self.node_norm(new_node_features) if self.node_norm is not None else new_node_features
        edge_in = self.edge_norm(edge_features) if self.edge_norm is not None else edge_features
        tp_input = torch.cat(
            [node_in[edge_center[active_edges]], edge_in],
            dim=-1,
        )
        message, _ = _apply_so2_tp_or_post_activation_mixer(
            self,
            tp_input,
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )
        message = self.lin_post(message)
        scalars = message[:, :self.irreps_out[0].dim]

        if self.edge_message_env_weight:
            weights = self.env_embed_mlps(latents[active_edges])
            weighted_message = self._env_weighter(message, weights)
        else:
            weighted_message = message
        active_edge_center = edge_center[active_edges]
        new_node_features = scatter(
            weighted_message,
            active_edge_center,
            dim=0,
            dim_size=node_features.shape[0],
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        assert len(scalars.shape) == 2

        new_node_features = new_node_features * norm_const

        if self.res_update:
            coefficient_old, coefficient_new = self._residual_coefficients()

            if self.use_identity_res:
                node_features = coefficient_old * node_features + coefficient_new * new_node_features
            else:
                # Different representations require an equivariant residual projection.
                node_features = coefficient_old * self.linear_res(node_features) + coefficient_new * new_node_features

        else:
            node_features = new_node_features

        if self.use_layer_onehot_tp:
            onehot_tune_node_feat = _apply_onehot_tp(
                self.node_onehot_tp, node_features, node_onehot, self.onehot_tp_mode
            )
            node_features = node_features + onehot_tune_node_feat

        return node_features


class UpdateEdge(torch.nn.Module):
    """SO(2) update of an edge state from ``[h_i, m_ij, h_j]``.

    The LEM edge update reads its own edge state as the message ``m_ij`` and
    updates the edge latents. The SLEM edge update reads the hidden state
    ``x_ij`` instead (``message_irreps_in``) and leaves the latents unchanged
    (``update_latents=False``). Without ``h_j`` (``use_neighbor_node=False``)
    this is the SLEM hidden-state update, :class:`UpdateHidden`.
    """

    def __init__(
            self,
            num_types,
            node_irreps_in: o3.Irreps,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            norm_eps: float = 1e-8,
            latent_channels: list = [128, 128],
            radial_emb: bool = False,
            radial_channels: list = [128, 128],
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            edge_one_hot_dim: int = 128,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "streamed_m_major_cueq",
            mole_linear_mode: Optional[str] = "cueq_indexed_linear",
            mole_expert_parameterization: str = "full",
            mole_expert_rank: int = 64,
            so2_expert_mixing_mode: str = "pre_activation",
            so2_expert_route_chunk_size: Optional[int] = None,
            so2_expert_route_checkpoint: bool = False,
            so2_output_router_hidden_dim: int = 32,
            onehot_tp_mode: Optional[str] = None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
            message_irreps_in: Optional[o3.Irreps] = None,
            use_neighbor_node: bool = True,
            update_latents: bool = True,
    ):
        super(UpdateEdge, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.node_irreps_in = node_irreps_in
        # Irreps of the edge message entering the SO(2) map; the residual stream
        # keeps irreps_in.
        self.message_irreps_in = irreps_in if message_irreps_in is None else o3.Irreps(message_irreps_in)
        self.use_neighbor_node = bool(use_neighbor_node)
        self.update_latents = bool(update_latents)
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        self.onehot_tp_mode = _normalize_onehot_tp_mode(onehot_tp_mode)
        self.so2_expert_mixing_mode = _normalize_so2_expert_mixing_mode(so2_expert_mixing_mode)

        self._edge_weighter = E3ElementLinear(
            irreps_in=irreps_out,
            dtype=dtype,
            device=device,
        )

        self.edge_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._edge_weighter.weight_numel,
        )

        if self.update_latents:
            self.ln = torch.nn.LayerNorm(latent_dim)

        self.node_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.node_irreps_in,
            norm_eps,
            dtype,
            device,
        )
        self.edge_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.message_irreps_in,
            norm_eps,
            dtype,
            device,
        )

        if activation_type != "gate":
            raise ValueError("Only gate activation is supported; grid activations belong to archived models")
        self.activation = build_gate_activation(self.irreps_out)

        tp_irreps_in = self.node_irreps_in + self.message_irreps_in
        if self.use_neighbor_node:
            tp_irreps_in = tp_irreps_in + self.node_irreps_in
        self.tp = SO2_Linear(
            irreps_in=tp_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=0,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            mole_expert_parameterization=mole_expert_parameterization,
            mole_expert_rank=mole_expert_rank,
        )

        if self.update_latents:
            self.latents_mlp_1 = ScalarMLPFunction(
                mlp_input_dimension=latent_dim + self.irreps_out[0].dim,
                mlp_output_dimension=latent_dim,
                mlp_latent_dimensions=latent_channels,
                mlp_nonlinearity="silu",
                mlp_initialization="uniform",
            )

            self.latents_mlp_2 = ScalarMLPFunction(
                mlp_input_dimension=latent_dim + edge_one_hot_dim,
                mlp_output_dimension=latent_dim,
                mlp_latent_dimensions=latent_channels,
                mlp_nonlinearity="silu",
                mlp_initialization="uniform",
            )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.post_activation_expert_mixer = None
        if self.so2_expert_mixing_mode == "post_activation":
            scalar_dim = self.activation.irreps_out[0].dim
            self.post_activation_expert_mixer = _build_so2_post_activation_expert_mixer(
                self.tp,
                self.activation,
                scalar_dim,
                so2_output_router_hidden_dim,
                so2_expert_route_chunk_size,
                so2_expert_route_checkpoint,
            )
        elif self.so2_expert_mixing_mode == "post_activation_slot":
            # per-row top-k slots activated separately, mixed with the router's own coefficients
            if so2_expert_route_checkpoint or so2_expert_route_chunk_size:
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_slot' implements neither "
                    "so2_expert_route_checkpoint nor so2_expert_route_chunk_size; unset them."
                )
            self.post_activation_expert_mixer = SO2SlotPostActivationMixer(self.tp, self.activation)
        elif self.so2_expert_mixing_mode == "post_activation_shared":
            # separate shared branch + per-slot routed experts, each activated (DPA3-MoE eq. 4)
            if so2_expert_route_checkpoint or so2_expert_route_chunk_size:
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_shared' implements neither "
                    "so2_expert_route_checkpoint nor so2_expert_route_chunk_size; unset them."
                )
            self.post_activation_expert_mixer = SO2SharedPostActivationMixer(self.tp, self.activation)

        if res_update:
            self.linear_res = Linear(
                self.irreps_in,
                self.irreps_out,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )

        if res_update_ratios is None:
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )

        self.use_layer_onehot_tp = use_layer_onehot_tp
        if use_layer_onehot_tp:
            instructions = []
            for i, (mul, ir) in enumerate(self.irreps_out):
                instructions.append((i, 0, i, 'uvu', True))
            self.edge_onehot_tp = ScalarOnehotTP.from_e3nn(TensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=f'{edge_one_hot_dim}x0e',
                irreps_out=self.irreps_out,
                instructions=instructions
            ))

        self.use_identity_res = (self.irreps_in == self.irreps_out) and res_update
        if not self.use_identity_res:
            if res_update:
                self.linear_res = Linear(
                    self.irreps_in,
                    self.irreps_out,
                    shared_weights=True,
                    internal_weights=True,
                    biases=True,
                )

    def _residual_coefficients(self):
        update_coefficients = self._res_update_params.sigmoid()
        coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
        coefficient_new = update_coefficients * coefficient_old
        return coefficient_old, coefficient_new

    def forward(self, latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs,
                active_edges, edge_one_hot, wigner_D_all, mole_globals, message_features=None):  # Accept globals
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = self.node_norm(new_node_features) if self.node_norm is not None else new_node_features
        message = edge_features if message_features is None else message_features
        edge_in = self.edge_norm(message) if self.edge_norm is not None else message

        edge_latents = latents[active_edges]
        tp_parts = [node_in[edge_center[active_edges]], edge_in]
        if self.use_neighbor_node:
            tp_parts.append(node_in[edge_neighbor[active_edges]])
        tp_input = torch.cat(tp_parts, dim=-1)
        new_edge_features, wigner_D_all = _apply_so2_tp_or_post_activation_mixer(
            self,
            tp_input,
            edge_vector[active_edges],
            mole_globals,
            edge_latents,
            wigner_D_all,
        )
        new_edge_features = self.lin_post(new_edge_features)

        scalars = new_edge_features[:, :self.irreps_out[0].dim]
        assert len(scalars.shape) == 2

        weights = self.edge_embed_mlps(edge_latents)
        new_edge_features = self._edge_weighter(new_edge_features, weights)

        if self.update_latents:
            new_latents = self.latents_mlp_1(torch.cat(
                [
                    self.ln(edge_latents),
                    scalars,
                ], dim=-1))

            new_latents = self.latents_mlp_2(torch.cat(
                [
                    new_latents,
                    edge_one_hot,
                ], dim=-1))

            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

        if self.res_update:
            coefficient_old, coefficient_new = self._residual_coefficients()

            if self.use_identity_res:
                edge_features = coefficient_old * edge_features + coefficient_new * new_edge_features
            else:
                # Different representations require an equivariant residual projection.
                edge_features = coefficient_old * self.linear_res(edge_features) + coefficient_new * new_edge_features

            if self.update_latents:
                latents = torch.index_copy(
                    latents, 0, active_edges,
                    coefficient_new * new_latents + coefficient_old * edge_latents
                )
        else:
            edge_features = new_edge_features
            if self.update_latents:
                latents = torch.index_copy(
                    latents, 0, active_edges,
                    new_latents
                )
        if self.use_layer_onehot_tp:
            onehot_tune_edge_feat = _apply_onehot_tp(
                self.edge_onehot_tp, edge_features, edge_one_hot, self.onehot_tp_mode
            )
            edge_features = edge_features + onehot_tune_edge_feat

        return edge_features, latents, wigner_D_all


class UpdateHidden(UpdateEdge):
    """SLEM hidden-state update: SO(2) on ``[h_i, x_ij]`` -> ``x_ij``.

    It owns the edge-latent update of a SLEM layer. Neither input depends on a
    neighbor's node state, so ``x_ij`` and the latents stay within the cutoff
    sphere of atom ``i``.
    """

    def __init__(self, **kwargs):
        super().__init__(use_neighbor_node=False, **kwargs)


class Layer(torch.nn.Module):
    """LEM interaction layer: edge update, then node update from the new edges."""

    layer_topology = "lem"

    @staticmethod
    def _edge_update_type():
        return UpdateEdge

    @staticmethod
    def _node_update_type():
        return UpdateNode

    def __init__(
            self,
            num_types: int,
            # required params
            avg_num_neighbors: Optional[float] = None,
            irreps_in: o3.Irreps = None,
            irreps_out: o3.Irreps = None,
            tp_radial_emb: bool = False,
            tp_radial_channels: list = [128, 128],
            # MLP parameters:
            norm_eps: float = 1e-8,
            latent_channels: list = [128, 128],
            latent_dim: int = 128,
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            edge_one_hot_dim: int = 128,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            edge_activation_type: str = "gate",
            node_activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            ffn_hidden_factor: float = 0.0,
            use_node_ffn: bool = False,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "streamed_m_major_cueq",
            mole_linear_mode: Optional[str] = "cueq_indexed_linear",
            mole_expert_parameterization: str = "full",
            mole_expert_rank: int = 64,
            so2_expert_mixing_mode: str = "pre_activation",
            so2_expert_route_chunk_size: Optional[int] = None,
            so2_expert_route_checkpoint: bool = False,
            so2_output_router_hidden_dim: int = 32,
            onehot_tp_mode: Optional[str] = None,
            node_message_aggregation: str = "scatter",
            num_focus: int = 1,
            focus_attention_dim: int = 32,
            edge_aggregation_gated_attention: bool = False,
            edge_attention_key_source: str = "message",
            edge_attention_envelope_power: float = 1.0,
            edge_attention_use_latent_bias: bool = True,
            edge_attention_key_layer_norm: bool = False,
            edge_attention_query_layer_norm: bool = False,
            edge_attention_qk_layer_norm: bool = False,
            edge_message_env_weight: bool = True,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
            hidden_irreps_in: Optional[o3.Irreps] = None,
            hidden_irreps_out: Optional[o3.Irreps] = None,
    ):
        super(Layer, self).__init__()

        self.res_update = res_update
        self.avg_num_neighbors = avg_num_neighbors
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.dtype = dtype
        self.device = device
        self.num_types = num_types

        slem = self.layer_topology == "slem"
        hidden_given = (hidden_irreps_in is not None, hidden_irreps_out is not None)
        if hidden_given != (slem, slem):
            raise ValueError("hidden_irreps_in and hidden_irreps_out are required by the SLEM layer "
                             "and not accepted by the LEM layer")

        # Options shared by every edge-shaped SO(2) update of this layer.
        edge_options = dict(
            num_types=num_types,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            use_layer_onehot_tp=use_layer_onehot_tp,
            edge_one_hot_dim=edge_one_hot_dim,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            equivariant_norm_type=equivariant_norm_type,
            activation_type=edge_activation_type,
            swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
            swiglu_s2_compat_mode=swiglu_s2_compat_mode,
            dtype=dtype,
            device=device,
            norm_eps=norm_eps,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            mole_expert_parameterization=mole_expert_parameterization,
            mole_expert_rank=mole_expert_rank,
            so2_expert_mixing_mode=so2_expert_mixing_mode,
            so2_expert_route_chunk_size=so2_expert_route_chunk_size,
            so2_expert_route_checkpoint=so2_expert_route_checkpoint,
            so2_output_router_hidden_dim=so2_output_router_hidden_dim,
            onehot_tp_mode=onehot_tp_mode,
        )
        edge_message = {}
        node_message_irreps = None
        if slem:
            # The hidden state stays inside the stack; the output interpolation
            # block applies only to the final edge and node maps.
            self.hidden_update = UpdateHidden(
                node_irreps_in=self.irreps_in,
                irreps_in=hidden_irreps_in,
                irreps_out=hidden_irreps_out,
                use_interpolation_tp=False,
                **edge_options,
            )
            edge_message = dict(message_irreps_in=hidden_irreps_out, update_latents=False)
            node_message_irreps = hidden_irreps_out

        self.edge_update = self._edge_update_type()(
            node_irreps_in=self.irreps_in,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            use_interpolation_tp=use_interpolation_tp,
            **edge_options,
            **edge_message,
        )

        self.node_update = self._node_update_type()(
            edge_irreps_in=self.edge_update.irreps_out if node_message_irreps is None else node_message_irreps,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            radial_emb=tp_radial_emb,
            use_layer_onehot_tp=use_layer_onehot_tp,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            avg_num_neighbors=avg_num_neighbors,
            equivariant_norm_type=equivariant_norm_type,
            activation_type=node_activation_type,
            swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
            swiglu_s2_compat_mode=swiglu_s2_compat_mode,
            dtype=dtype,
            device=device,
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            mole_expert_parameterization=mole_expert_parameterization,
            mole_expert_rank=mole_expert_rank,
            so2_expert_mixing_mode=so2_expert_mixing_mode,
            so2_expert_route_chunk_size=so2_expert_route_chunk_size,
            so2_expert_route_checkpoint=so2_expert_route_checkpoint,
            so2_output_router_hidden_dim=so2_output_router_hidden_dim,
            onehot_tp_mode=onehot_tp_mode,
            node_message_aggregation=node_message_aggregation,
            num_focus=num_focus,
            focus_attention_dim=focus_attention_dim,
            edge_aggregation_gated_attention=edge_aggregation_gated_attention,
            edge_attention_key_source=edge_attention_key_source,
            edge_attention_envelope_power=edge_attention_envelope_power,
            edge_attention_use_latent_bias=edge_attention_use_latent_bias,
            edge_attention_key_layer_norm=edge_attention_key_layer_norm,
            edge_attention_query_layer_norm=edge_attention_query_layer_norm,
            edge_attention_qk_layer_norm=edge_attention_qk_layer_norm,
            edge_message_env_weight=edge_message_env_weight,
        )

        if use_node_ffn:
            raise ValueError("Grid feed-forward layers belong to archived models")

    def forward(self, latents, node_features, edge_features, node_onehot, edge_index, edge_vector, atom_type,
                cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all, mole_globals, node_batch=None):
        edge_features, latents, wigner_D_all = self.edge_update(latents, node_features, node_onehot, edge_features,
                                                                edge_index, edge_vector, cutoff_coeffs, active_edges,
                                                                edge_one_hot, wigner_D_all, mole_globals)
        node_features = self.node_update(latents, node_features, edge_features, atom_type, node_onehot, edge_index,
                                         edge_vector, cutoff_coeffs, active_edges, wigner_D_all, mole_globals,
                                         node_batch=node_batch)

        return latents, node_features, edge_features, wigner_D_all


class SlemLayer(Layer):
    """SLEM interaction layer: hidden-state update, edge update, node update.

    The hidden update maps ``[h_i, x_ij]`` to ``x_ij`` and updates the edge
    latents; the edge update maps ``[h_i, x_ij, h_j]`` to ``e_ij`` without
    changing the latents; the node update aggregates ``[h_i, x_ij]`` over
    ``j``. All three read the node features entering the layer. Edge features
    feed only the next edge residual and the edge output head, so node features
    depend only on atoms within one cutoff sphere.
    """

    layer_topology = "slem"

    def forward(self, latents, node_features, edge_features, hidden_features, node_onehot, edge_index, edge_vector,
                atom_type, cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all, mole_globals, node_batch=None):
        hidden_features, latents, wigner_D_all = self.hidden_update(
            latents, node_features, node_onehot, hidden_features, edge_index, edge_vector, cutoff_coeffs,
            active_edges, edge_one_hot, wigner_D_all, mole_globals)
        edge_features, _, wigner_D_all = self.edge_update(
            latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs,
            active_edges, edge_one_hot, wigner_D_all, mole_globals, message_features=hidden_features)
        node_features = self.node_update(latents, node_features, hidden_features, atom_type, node_onehot, edge_index,
                                         edge_vector, cutoff_coeffs, active_edges, wigner_D_all, mole_globals,
                                         node_batch=node_batch)
        return latents, node_features, edge_features, hidden_features, wigner_D_all


