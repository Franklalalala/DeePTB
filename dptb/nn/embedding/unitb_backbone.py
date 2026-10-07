"""Shared parameter construction and output contracts for UniTB and legacy graphs."""
from typing import Optional, List, Union, Dict, Tuple
import math
import functools
import torch
from e3nn import o3
from torch_scatter import scatter_mean
from e3nn.o3 import SphericalHarmonics, FullyConnectedTensorProduct
from dptb.nn.so2_parity import normalize_so2_parity, enforce_so2_parity
from dptb.data import AtomicDataDict
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    attach_prediction_block_tensors,
    infer_block_shapes,
)
from dptb.data import _keys
# Note: Modified SO2_Linear and MOLE classes imported here
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLEGlobals, MOLERouterV3, write_router_regularizers
import math
from dptb.data.transforms import OrbitalMapper
from dptb.utils.soc_target import resolve_nextham_uureal_mask
from .ao_projector_bank import build_ao_decoder_irreps
from .output_routes import (
    OutputHeadContext,
    build_output_heads,
    effective_product_scope,
    resolve_output_route,
    select_final_irreps,
)
from .block_native_head import (
    compact_blocks_to_species_layout,
    species_compact_index,
)
from ..type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch


import logging

log = logging.getLogger(__name__)


from .unitb_ops import _normalize_node_message_aggregation, _normalize_edge_attention_key_source, _normalize_onehot_tp_mode, _normalize_stable_standard_compat_mode, _normalize_so2_expert_mixing_mode, _normalize_so2_moe_layers, _normalize_cg_head_impl, ScalarOnehotTP, _capture_shift_hidden, _apply_onehot_tp
from .unitb_layers import InitLayer, Layer

class UniTBBackbone(torch.nn.Module):
    @staticmethod
    def _init_layer_type():
        """Extension point for embeddings that reuse the LEM v3 backbone."""
        return InitLayer

    @staticmethod
    def _layer_type():
        """Extension point for embeddings that reuse the LEM v3 backbone."""
        return Layer

    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            # required params
            n_layers: int = 3,
            n_radial_basis: int = 10,
            r_max: float = 5.0,
            irreps_hidden: o3.Irreps = None,
            avg_num_neighbors: Optional[float] = None,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            # general hyperparameters:
            env_embed_multiplicity: int = 32,
            sh_normalized: bool = True,
            sh_normalization: str = "component",
            # tp parameters:
            tp_radial_emb: bool = False,
            tp_radial_channels: list = [128, 128],
            # MLP parameters:
            latent_channels: list = [128, 128],
            latent_dim: int = 128,
            edge_one_hot_dim: int = 128,
            use_out_onehot_tp: bool = True,
            use_layer_onehot_tp: bool = True,
            output_route: Optional[str] = None,
            rme_head_mode: Optional[str] = None,
            rme_fusion_rank: int = 16,
            rme_fusion_init: float = 0.0,
            rme_fusion_condition: str = "scalar_0e",
            rme_cartesian_scope: Optional[str] = None,
            rme_ict_scope: Optional[str] = None,
            ao_projector_channels: int = 0,
            ao_projector_normalization: str = "e3hamiltonian",
            ao_projector_basis_convention: str = "deeptb_real_ao",
            ao_projector_backend: str = "reference_wigner",
            ao_projector_bank_path: Optional[str] = None,
            cg_head_impl: str = "legacy",
            res_update: bool = True,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            hidden_edge_activation_type: str = "gate",
            hidden_node_activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            ffn_hidden_factor: float = 0.0,
            ffn_apply_to_last: bool = False,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "streamed_m_major_cueq",
            mole_linear_mode: Optional[str] = "cueq_indexed_linear",
            mole_expert_parameterization: str = "full",
            mole_expert_rank: int = 64,
            so2_expert_mixing_mode: str = "pre_activation",
            so2_parity: str = "none",
            so2_moe_layers: Union[str, List[int]] = "all",
            so2_expert_route_chunk_size: Optional[int] = None,
            so2_expert_route_checkpoint: bool = False,
            so2_output_router_hidden_dim: int = 32,
            so2_m_linear_mode: Optional[str] = None,
            mole_linear_m0_mode: Optional[str] = None,
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
            universal: Optional[bool] = False,
            use_interpolation_out: Optional[bool] = True,
            # MOE parameters
            num_experts: int = 8,
            num_shared_experts: int = 1,
            top_k: Optional[int] = 1,
            mole_full_expert_fast_path: bool = True,
            **kwargs,
    ):

        super(UniTBBackbone, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        self.so2_parity = normalize_so2_parity(so2_parity)
        if self.so2_parity == "enforce":
            if use_interpolation_out:
                raise ValueError("so2_parity='enforce' requires use_interpolation_out=false: "
                                 "component-wise interpolation MLPs are not SO(2) equivariant")
            if (hidden_edge_activation_type != "gate" or hidden_node_activation_type != "gate"
                    or ffn_hidden_factor > 1.0):
                raise ValueError("so2_parity='enforce' requires gate activations and no grid FFN; "
                                 "finite S2 grids are not exactly rotation equivariant")
            if irreps_hidden[0].ir != o3.Irrep("0e"):
                raise ValueError("so2_parity='enforce' requires leading 0e invariant scalars")
            if (getattr(self, "edge_router_prior_activate", False)
                    and getattr(self, "edge_router_input", "onehot_prior") == "onehot_prior"
                    and not getattr(self, "edge_router_prior_cg", False)):
                raise ValueError("so2_parity='enforce' requires edge_router_prior_cg=true "
                                 "for a router consuming AO H0 blocks")
        if hidden_edge_activation_type != "gate" or hidden_node_activation_type != "gate":
            raise ValueError("Grid activations belong to archived models; use gate activation")
        if ffn_hidden_factor > 1.0 or ffn_apply_to_last:
            raise ValueError("Grid feed-forward layers belong to archived models")
        lmax = irreps_hidden.lmax
        self.num_experts = num_experts

        # Report the effective model configuration.
        log.info(f'[LemMoEV3] Initialized DeepSeek-V3 Style MoE.')
        log.info(f'  - Num Shared Experts: {num_shared_experts}')
        log.info(f'  - Num Routed Experts: {self.num_experts}')
        log.info(f'  - Top-K Actived Routed Experts: {top_k}')
        log.info(f'  - Full Expert Fast Path: {mole_full_expert_fast_path}')
        log.info(f'  - MoLE Linear Mode: {mole_linear_mode or "env/default"}')
        log.info(f'  - Strategy: Shared Expert + Aux-Loss-Free Balancing (Sigmoid Routing)')
        mean_max_prob_lower_bound = 1.0 / self.num_experts
        # A one-hot distribution reaches the upper bound.
        mean_max_prob_upper_bound = 1.0

        log.info(f"[LemMoEV3] Theoretical mean_max_prob Bounds -> "
                 f"Min (Uniform): {mean_max_prob_lower_bound:.6f} | Max (One-Hot): {mean_max_prob_upper_bound:.6f}")
        effective_top_k = self.num_experts if top_k is None else min(top_k, self.num_experts)
        cv_lower_bound = 0.0
        cv_upper_bound = math.sqrt((self.num_experts - effective_top_k) / effective_top_k) if effective_top_k else 0.0
        log.info(f"[LemMoEV3] Theoretical expert_load_cv Bounds -> "
                 f"Min (Balanced): {cv_lower_bound:.6f} | Max (Collapsed): {cv_upper_bound:.6f}")

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.has_soc = bool(kwargs.get("has_soc", False))
        self.full_soc_prediction = bool(kwargs.get("full_soc_prediction", False))
        self.nextham_uureal_mask = resolve_nextham_uureal_mask(
            nextham_uureal_mask=kwargs.get("nextham_uureal_mask", False),
            full_soc_prediction=self.full_soc_prediction,
        )
        self.onehot_tp_mode = _normalize_onehot_tp_mode(onehot_tp_mode)
        self.output_route_spec = resolve_output_route(
            output_route=output_route,
            legacy_mode=rme_head_mode,
            projector_backend=ao_projector_backend,
            projector_bank_path=ao_projector_bank_path,
        )
        self.output_route_name = self.output_route_spec.canonical_name
        self.rme_head_mode = self.output_route_spec.legacy_mode
        self.use_block_native_output = self.output_route_spec.is_block_native
        self.output_head_contract = self.output_route_spec.output_contract
        self.rme_fusion_rank = int(rme_fusion_rank)
        self.rme_fusion_init = float(rme_fusion_init)
        self.rme_fusion_condition = str(rme_fusion_condition)
        if rme_ict_scope is not None:
            if (
                rme_cartesian_scope is not None
                and str(rme_cartesian_scope).strip().lower()
                != str(rme_ict_scope).strip().lower()
            ):
                raise ValueError(
                    "rme_ict_scope conflicts with rme_cartesian_scope; use only "
                    "rme_cartesian_scope."
                )
            log.warning("rme_ict_scope is deprecated; use rme_cartesian_scope instead.")
            rme_cartesian_scope = rme_ict_scope
        self.rme_cartesian_scope = effective_product_scope(
            self.output_route_spec, rme_cartesian_scope
        )
        self.ao_projector_channels = int(ao_projector_channels)
        self.ao_projector_normalization = str(ao_projector_normalization)
        self.ao_projector_basis_convention = str(ao_projector_basis_convention)
        self.ao_projector_backend = str(ao_projector_backend)
        self.ao_projector_bank_path = ao_projector_bank_path
        self.cg_head_impl = _normalize_cg_head_impl(cg_head_impl)
        self.so2_expert_mixing_mode = _normalize_so2_expert_mixing_mode(so2_expert_mixing_mode)
        self.so2_moe_layers = _normalize_so2_moe_layers(so2_moe_layers, n_layers)
        if len(self.so2_moe_layers) != n_layers:
            if not getattr(self, "edge_router_prior_activate", False):
                raise ValueError("Selective so2_moe_layers requires per-edge prior-activate routing")
            if num_shared_experts < 1:
                raise ValueError("Selective so2_moe_layers requires at least one shared expert")
        if (self.so2_expert_mixing_mode in ("post_activation_slot", "post_activation_shared")
                and not getattr(self, "edge_router_prior_activate", False)):
            # the slot mixers need per-row top-k metadata (prior_activate)
            raise ValueError("so2_expert_mixing_mode=%r needs per-edge routing "
                             "(lem_moe_v3_edge* with edge_router_prior_activate=true)." % (self.so2_expert_mixing_mode,))
        self.node_message_aggregation = _normalize_node_message_aggregation(node_message_aggregation)
        self.num_focus = int(num_focus)
        self.edge_aggregation_gated_attention = bool(edge_aggregation_gated_attention)
        self.edge_attention_key_source = _normalize_edge_attention_key_source(edge_attention_key_source)
        self.edge_attention_envelope_power = float(edge_attention_envelope_power)
        self.edge_attention_use_latent_bias = bool(edge_attention_use_latent_bias)
        self.edge_attention_key_layer_norm = bool(edge_attention_key_layer_norm)
        self.edge_attention_query_layer_norm = bool(edge_attention_query_layer_norm)
        self.edge_attention_qk_layer_norm = bool(edge_attention_qk_layer_norm)
        self.edge_message_env_weight = bool(edge_message_env_weight)
        self.so2_m_linear_mode = _normalize_stable_standard_compat_mode(
            "so2_m_linear_mode", so2_m_linear_mode
        )
        self.mole_linear_m0_mode = _normalize_stable_standard_compat_mode(
            "mole_linear_m0_mode", mole_linear_m0_mode
        )
        log.info(f"  - OneHot TP Mode: {self.onehot_tp_mode}")
        log.info(
            "  - Output Head: route=%s legacy_mode=%s contract=%s rank=%d "
            "init=%g condition=%s cg_head_impl=%s",
            self.output_route_name,
            self.rme_head_mode,
            self.output_head_contract,
            self.rme_fusion_rank,
            self.rme_fusion_init,
            self.rme_fusion_condition,
            self.cg_head_impl,
        )
        log.info(f"  - SO2 Expert Mixing Mode: {self.so2_expert_mixing_mode}")
        log.info(
            "  - DPA4-style Focus/Aggregation: "
            f"num_focus={self.num_focus}, node_message_aggregation={self.node_message_aggregation}, "
            f"focus_attention_dim={focus_attention_dim}, "
            f"edge_aggregation_gated_attention={self.edge_aggregation_gated_attention}, "
            f"edge_attention_key_source={self.edge_attention_key_source}, "
            f"edge_attention_envelope_power={self.edge_attention_envelope_power}, "
            f"edge_attention_use_latent_bias={self.edge_attention_use_latent_bias}, "
            f"edge_attention_key_layer_norm={self.edge_attention_key_layer_norm}, "
            f"edge_attention_query_layer_norm={self.edge_attention_query_layer_norm}, "
            f"edge_attention_qk_layer_norm={self.edge_attention_qk_layer_norm}, "
            f"edge_message_env_weight={self.edge_message_env_weight}"
        )

        if basis is not None:
            self.idp = OrbitalMapper(
                basis,
                method="e3tb",
                device=self.device,
                has_soc=self.has_soc,
                nextham_uureal_mask=self.nextham_uureal_mask,
                full_soc_prediction=self.full_soc_prediction,
            )
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
        self.has_soc = bool(getattr(self.idp, "has_soc", self.has_soc))
        self.nextham_uureal_mask = bool(getattr(self.idp, "nextham_uureal_mask", self.nextham_uureal_mask))

        latent_kwargs = {
            "mlp_latent_dimensions": latent_channels + [latent_dim],
            "mlp_nonlinearity": "silu",
            "mlp_initialization": "uniform"
        },
        self.latent_dim = latent_dim

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()
        self.ao_decoder_irreps = (
            build_ao_decoder_irreps(
                self.idp.full_basis, channels=self.ao_projector_channels
            )
            if self.output_route_spec.final_irreps_kind == "ao_pair"
            else None
        )

        irreps_out = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out += [o3.Irrep(str(irr)) for irr in ir1 * ir2]
        irreps_out = o3.Irreps(irreps_out).sort()[0].simplify()

        assert all(ir in irreps_out for _, ir in
                   orbpair_irreps), "hidden irreps should at least cover all the reqired irreps in the hamiltonian data {}".format(
            orbpair_irreps)

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        # --- MOE Router V3 (DeepSeek Style) ---
        self.router = MOLERouterV3(
            in_features=self.n_atom,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_free=True,  # Auxiliary-loss-free load balancing.
            bias_update_speed=0.005,
            full_expert_fast_path=mole_full_expert_fast_path,
        )

        self.init_layer = self._init_layer_type()(
            idp=self.idp,
            num_types=self.n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max,
            irreps_sh=irreps_sh,
            avg_num_neighbors=avg_num_neighbors,
            env_embed_multiplicity=env_embed_multiplicity,
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p,
            cutoff_type=cutoff_type,
            device=device,
            dtype=dtype,
            edge_one_hot_dim=edge_one_hot_dim,
            norm_eps=norm_eps
        )

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden

            if i == n_layers - 1:
                irreps_out = select_final_irreps(
                    self.output_route_spec,
                    ordinary_hidden=irreps_hidden,
                    orbpair_irreps=orbpair_irreps,
                    ao_pair_irreps=self.ao_decoder_irreps,
                )
                use_interpolation_tp = bool(
                    use_interpolation_out
                    and self.output_route_spec.final_irreps_kind == "orbpair"
                )
            else:
                irreps_out = irreps_hidden
                use_interpolation_tp = False

            if i == n_layers - 1:
                edge_activation_type = "gate"
                node_activation_type = "gate"
            else:
                edge_activation_type = hidden_edge_activation_type
                node_activation_type = hidden_node_activation_type

            routed_layer = i in self.so2_moe_layers

            self.layers.append(self._layer_type()(
                num_types=self.n_atom,
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                use_layer_onehot_tp=use_layer_onehot_tp,
                edge_one_hot_dim=edge_one_hot_dim,
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                equivariant_norm_type=equivariant_norm_type,
                edge_activation_type=edge_activation_type,
                node_activation_type=node_activation_type,
                swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
                swiglu_s2_compat_mode=swiglu_s2_compat_mode,
                ffn_hidden_factor=ffn_hidden_factor,
                so2_wigner_apply_mode=so2_wigner_apply_mode,
                so2_fusion_mode=so2_fusion_mode,
                mole_linear_mode=mole_linear_mode,
                mole_expert_parameterization=mole_expert_parameterization if routed_layer else "full",
                mole_expert_rank=mole_expert_rank,
                so2_expert_mixing_mode=self.so2_expert_mixing_mode if routed_layer else "pre_activation",
                so2_expert_route_chunk_size=so2_expert_route_chunk_size,
                so2_expert_route_checkpoint=so2_expert_route_checkpoint,
                so2_output_router_hidden_dim=so2_output_router_hidden_dim,
                onehot_tp_mode=self.onehot_tp_mode,
                node_message_aggregation=self.node_message_aggregation,
                num_focus=self.num_focus,
                focus_attention_dim=focus_attention_dim,
                edge_aggregation_gated_attention=self.edge_aggregation_gated_attention,
                edge_attention_key_source=self.edge_attention_key_source,
                edge_attention_envelope_power=self.edge_attention_envelope_power,
                edge_attention_use_latent_bias=self.edge_attention_use_latent_bias,
                edge_attention_key_layer_norm=self.edge_attention_key_layer_norm,
                edge_attention_query_layer_norm=self.edge_attention_query_layer_norm,
                edge_attention_qk_layer_norm=self.edge_attention_qk_layer_norm,
                edge_message_env_weight=self.edge_message_env_weight,
                dtype=dtype,
                device=device,
                use_interpolation_tp=use_interpolation_tp,
                num_experts=num_experts if routed_layer else 0,
                num_shared_experts=num_shared_experts,  # Pass down to Layer -> SO2_Linear
            ))

            if use_interpolation_tp:
                print(f'Use interpolation SO2 layer in layer {i}')

        self.use_out_onehot_tp = (
            bool(use_out_onehot_tp)
            and self.output_route_spec.output_contract == "rme"
        )
        if self.use_out_onehot_tp:
            onehot_irreps_in = (
                self.idp.orbpair_irreps
                if self.output_route_spec.onehot_after_head
                else self.layers[-1].irreps_out
            )
            self.out_node_ele_tp = ScalarOnehotTP.from_e3nn(FullyConnectedTensorProduct(
                irreps_in1=onehot_irreps_in,
                irreps_in2='95x0e',
                irreps_out=self.idp.orbpair_irreps,
            ))
            self.out_edge_ele_tp = ScalarOnehotTP.from_e3nn(FullyConnectedTensorProduct(
                irreps_in1=onehot_irreps_in,
                irreps_in2=f'{edge_one_hot_dim}x0e',
                irreps_out=self.idp.orbpair_irreps,
            ))
        max_norb = int(getattr(self.idp, "full_basis_norb", 0))
        if max_norb <= 0:
            max_norb = sum(
                int(v) for v in getattr(self.idp, "basis_to_full_basis", {}).values()
            )
        head_context = OutputHeadContext(
            final_irreps=self.layers[-1].irreps_out,
            orbpair_irreps=self.idp.orbpair_irreps,
            full_basis=tuple(self.idp.full_basis),
            max_norb=max_norb,
            rank=self.rme_fusion_rank,
            init=self.rme_fusion_init,
            condition=self.rme_fusion_condition,
            product_scope=self.rme_cartesian_scope,
            ao_projector_normalization=self.ao_projector_normalization,
            ao_projector_basis_convention=self.ao_projector_basis_convention,
            ao_projector_backend=self.ao_projector_backend,
            ao_projector_bank_path=self.ao_projector_bank_path,
            cg_head_impl=self.cg_head_impl,
            dtype=self.dtype,
            device=self.device,
        )
        self.out_edge, self.out_node = build_output_heads(
            self.output_route_spec, head_context
        )
        if self.so2_parity == "enforce":
            for module in self.modules():
                if isinstance(module, SO2_Linear):
                    enforce_so2_parity(module)
            # H0 subclasses install their initializer after this constructor.
            self.register_forward_pre_hook(self._check_parity_h0_contract)

    @staticmethod
    def _check_parity_h0_contract(module, inputs):
        if not getattr(module.init_layer, "h0_ao_cg", True):
            raise ValueError("so2_parity='enforce' requires h0_ao_cg=true for AO H0 inputs")

    @property
    def out_edge_irreps(self):
        return (
            None
            if self.output_route_spec.output_contract == "ao_block"
            else self.idp.orbpair_irreps
        )

    @property
    def out_node_irreps(self):
        return (
            None
            if self.output_route_spec.output_contract == "ao_block"
            else self.idp.orbpair_irreps
        )

    def _apply_rme_output_heads(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_one_hot: torch.Tensor,
        edge_one_hot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            node_tp_input = (
                out_node_features
                if self.output_route_spec.onehot_after_head
                else node_features
            )
            edge_tp_input = (
                out_edge_features
                if self.output_route_spec.onehot_after_head
                else edge_features
            )
            out_node_features = out_node_features + _apply_onehot_tp(
                self.out_node_ele_tp, node_tp_input, node_one_hot, self.onehot_tp_mode
            )
            out_edge_features = out_edge_features + _apply_onehot_tp(
                self.out_edge_ele_tp, edge_tp_input, edge_one_hot, self.onehot_tp_mode
            )
        return out_node_features, out_edge_features

    def _species_compact_maps(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = getattr(self, "_species_compact_cache", None)
        if cached is not None and cached[0].device == device:
            return cached[0], cached[1]
        index, norb = species_compact_index(self.idp.mask_to_basis.to(device=device))
        self._species_compact_cache = (index, norb)
        return index, norb

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _cached_hb0_reverse_edge_permutation(
        edge_pairs: Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, ...]:
        """Return active-row reverse mates without mutating module state."""
        lookup = {}
        for row, pair in enumerate(edge_pairs):
            if pair in lookup:
                raise ValueError(
                    "hb0_hermitian_average requires unique directed active edges; "
                    f"duplicate edge {pair} occurs at rows {lookup[pair]} and {row}."
                )
            lookup[pair] = row
        reverse = []
        for row, (src, dst) in enumerate(edge_pairs):
            mate = lookup.get((dst, src))
            if mate is None:
                raise ValueError(
                    "hb0_hermitian_average requires every active directed edge to "
                    f"have a reverse partner; row {row} edge ({src}, {dst}) is missing "
                    f"({dst}, {src})."
                )
            reverse.append(mate)
        return tuple(reverse)

    @classmethod
    def _hermitian_average_hb0_edge_blocks(
        cls,
        edge_blocks: torch.Tensor,
        edge_index: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        selected = edge_index.index_select(1, active_edges).detach().cpu().T.tolist()
        edge_pairs = tuple((int(src), int(dst)) for src, dst in selected)
        reverse = torch.tensor(
            cls._cached_hb0_reverse_edge_permutation(edge_pairs),
            dtype=torch.long,
            device=edge_blocks.device,
        )
        rows = torch.arange(reverse.numel(), dtype=torch.long, device=edge_blocks.device)
        canonical = rows[rows <= reverse]
        mates = reverse.index_select(0, canonical)
        averaged = 0.5 * (
            edge_blocks.index_select(0, canonical)
            + edge_blocks.index_select(0, mates).transpose(-1, -2)
        )
        projected = torch.zeros_like(edge_blocks)
        projected = torch.index_copy(projected, 0, canonical, averaged)
        projected = torch.index_copy(
            projected, 0, mates, averaged.transpose(-1, -2)
        )
        return projected

    @staticmethod
    def _head_input_rms_by_irreps(
        features: torch.Tensor,
        irreps: o3.Irreps,
    ) -> torch.Tensor:
        detached = features.detach()
        values = []
        for term_slice in irreps.slices():
            block = detached[..., term_slice]
            if block.numel() == 0:
                values.append(detached.new_zeros(()))
            else:
                values.append(block.square().mean().sqrt())
        return torch.stack(values)

    def _head_input_rms(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_irreps = o3.Irreps(self.out_node.irreps_in)
        edge_irreps = o3.Irreps(self.out_edge.irreps_in)
        return {
            "node": self._head_input_rms_by_irreps(node_features, node_irreps),
            "edge": self._head_input_rms_by_irreps(edge_features, edge_irreps),
            "node_l": torch.tensor(
                [ir.l for _, ir in node_irreps],
                dtype=torch.long,
                device=node_features.device,
            ),
            "edge_l": torch.tensor(
                [ir.l for _, ir in edge_irreps],
                dtype=torch.long,
                device=edge_features.device,
            ),
        }

    def _apply_block_native_output_heads(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        atom_type: torch.Tensor,
        edge_index: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_node_blocks = self.out_node(node_features)
        if getattr(self.out_edge, "condition_source", "edge_0e") == "endpoints":
            active_edges_for_condition = active_edges.to(
                device=edge_index.device, dtype=torch.long
            ).reshape(-1)
            active_edge_index = edge_index.index_select(
                1, active_edges_for_condition
            ).to(device=node_features.device)
            node_0e = node_features.index_select(
                -1, self.out_edge._node_scalar_indices
            )
            extra_condition = torch.cat(
                [
                    node_0e.index_select(0, active_edge_index[0]),
                    node_0e.index_select(0, active_edge_index[1]),
                ],
                dim=-1,
            )
            out_edge_blocks = self.out_edge(
                edge_features, extra_condition=extra_condition
            )
        else:
            out_edge_blocks = self.out_edge(edge_features)

        # Heads emit union full-basis slot canvases; blockwise targets/loss use
        # the species-contiguous layout, so translate at this boundary.
        compact_index, compact_norb = self._species_compact_maps(node_features.device)
        atom_type = atom_type.to(device=node_features.device, dtype=torch.long).flatten()
        out_node_blocks = compact_blocks_to_species_layout(
            out_node_blocks, compact_index[atom_type], compact_norb[atom_type]
        )

        edge_index = edge_index.to(device=node_features.device)
        active_edges = active_edges.to(device=node_features.device, dtype=torch.long).reshape(-1)
        src_type = atom_type[edge_index[0, active_edges]]
        dst_type = atom_type[edge_index[1, active_edges]]
        out_edge_blocks = compact_blocks_to_species_layout(
            out_edge_blocks,
            compact_index[src_type],
            compact_norb[src_type],
            compact_index[dst_type],
            compact_norb[dst_type],
        )
        if getattr(self, "hb0_hermitian_average", False):
            out_edge_blocks = self._hermitian_average_hb0_edge_blocks(
                out_edge_blocks, edge_index, active_edges
            )
        if getattr(self, "log_head_input_rms", False):
            return (
                out_node_blocks,
                out_edge_blocks,
                self._head_input_rms(node_features, edge_features),
            )
        return out_node_blocks, out_edge_blocks

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
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        # --- MOLE Routing Logic ---
        # 1. Global Feature per system: Mean of node one-hot
        global_feat = scatter_mean(node_one_hot, batch, dim=0)  # [Batch, n_atom]

        # 2. Compute Routing Coefficients
        # Coefficients are [batch, experts]; monitor is the mean maximum probability.
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)
        topk_indices, topk_values = self.router.last_topk()

        # Record mean maximum probability independently of the task loss.
        # Larger values indicate more concentrated routing.
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv
        write_router_regularizers(self.router, data)
        # 3. Prepare MOLEGlobals
        num_nodes_total = node_one_hot.shape[0]
        precomputed_active_edges = data.get(_keys.LEM_ACTIVE_EDGES_KEY, None)
        precomputed_cutoff_coeffs = data.get(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        precomputed_split_sizes = data.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if precomputed_cutoff_coeffs is not None and edge_length.requires_grad:
            raise RuntimeError(
                "Precomputed LEM cutoff coefficients cannot be used when edge_length requires gradients. "
                "Set train_options.precompute_lem_cutoff_coeffs=false for force/stress/virial training."
            )
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type,
                                                                                             bond_type, edge_sh,
                                                                                             edge_length, edge_one_hot,
                                                                                             precomputed_active_edges,
                                                                                             precomputed_cutoff_coeffs)

        n_active_nodes = node_features.shape[0]
        node_batch = batch[:n_active_nodes]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

        # Determine sizes for active edges for Weight Merging in MOLELinear
        if precomputed_split_sizes is not None:
            mole_globals = MOLEGlobals(
                coefficients=coeffs,
                split_sizes=precomputed_split_sizes,
                topk_indices=topk_indices,
                topk_values=topk_values,
            )
        else:
            edge_batch = batch[edge_index[0][active_edges]]  # Map edge to graph index
            num_systems = coeffs.shape[0]
            edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
            mole_globals = MOLEGlobals(
                coefficients=coeffs,
                sizes=edge_sizes,
                graph_index=edge_batch,
                topk_indices=topk_indices,
                topk_values=topk_values,
            )
        # --------------------------

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None
        for idx, layer in enumerate(self.layers):
            _capture_shift_hidden(self, data, idx, node_features, num_nodes_total, active_edges)
            latents, node_features, edge_features, wigner_D_all = \
                layer(
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
                    mole_globals,  # Pass globals to layers
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
        if self.use_block_native_output:
            if getattr(self, "log_head_input_rms", False):
                (
                    out_node_blocks,
                    out_edge_blocks,
                    head_input_rms,
                ) = self._apply_block_native_output_heads(
                    node_features, edge_features, atom_type, edge_index, active_edges
                )
                data["head_input_rms"] = head_input_rms
            else:
                out_node_blocks, out_edge_blocks = self._apply_block_native_output_heads(
                    node_features, edge_features, atom_type, edge_index, active_edges
                )
            data[_keys.NODE_HAMILTONIAN_KEY] = out_node_blocks
            data[_keys.EDGE_HAMILTONIAN_KEY] = out_edge_blocks.new_zeros(
                (edge_index.shape[1], self.out_edge.max_norb, self.out_edge.max_norb)
            )
            data[_keys.EDGE_HAMILTONIAN_KEY] = torch.index_copy(
                data[_keys.EDGE_HAMILTONIAN_KEY], 0, active_edges, out_edge_blocks
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

        if getattr(self, "capture_shift_features", False):
            data["_shift_node_features"] = node_features
            data["_shift_active_edges"] = active_edges

        out_node_features, out_edge_features = self._apply_rme_output_heads(
            node_features, edge_features, node_one_hot, edge_one_hot
        )

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype,
                                                    device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges,
                                                         out_edge_features)

        data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
        data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        return data


