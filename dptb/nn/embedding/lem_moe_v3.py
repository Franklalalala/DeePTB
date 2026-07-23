from typing import Optional, List, Union, Dict, Tuple
import math
import functools
import os
import torch
from torch_runstats.scatter import scatter
from torch import fx
from e3nn import o3
from torch_scatter import scatter_add, scatter_max, scatter_mean
from e3nn.o3 import Linear, SphericalHarmonics, FullyConnectedTensorProduct, TensorProduct
from dptb.data import AtomicDataDict
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    attach_prediction_block_tensors,
    infer_block_shapes,
)
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from ..base import ScalarMLPFunction
from dptb.nn.embedding.from_deephe3.deephe3 import tp_path_exists
from dptb.data import _keys
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from .lem_moe_v3_plugins import (
    EqV3StyleNodeFFN,
    FlatSwiGLUS2Merge,
    build_equivariant_norm,
    build_gate_activation,
    can_use_flat_s2_patch,
)
# Note: Modified SO2_Linear and MOLE classes imported here
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLEGlobals, MOLERouterV3, SO2PostActivationExpertMixer
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

from math import ceil

import logging

log = logging.getLogger(__name__)


def _normalize_node_message_aggregation(mode: Optional[str]) -> str:
    mode = mode or "scatter"
    allowed = {"scatter", "single_head_0e"}
    if mode not in allowed:
        raise ValueError(
            "node_message_aggregation must be one of "
            f"{sorted(allowed)}, got {mode!r}."
        )
    return mode


def _normalize_edge_attention_key_source(source: Optional[str]) -> str:
    source = source or "message"
    allowed = {"message"}
    if source not in allowed:
        raise ValueError(
            "edge_attention_key_source must be one of "
            f"{sorted(allowed)}, got {source!r}."
        )
    return source


def _normalize_onehot_tp_mode(mode: Optional[str]) -> str:
    mode = mode or os.environ.get("DPTB_ONEHOT_TP_MODE", "scalar_fast")
    allowed = {"scalar_fast"}
    if mode not in allowed:
        raise ValueError(
            "0422-cueq-fastest only supports onehot_tp_mode='scalar_fast'. "
            f"Use the legacy cueq branch for e3nn/auto A/B, got {mode!r}."
        )
    return mode


def _normalize_stable_standard_compat_mode(name: str, mode: Optional[str]) -> str:
    if mode in (None, "", "standard"):
        return "standard"
    raise ValueError(
        f"0425-stable accepts only {name}=None or 'standard'. "
        f"Use the Triton experiment branch for {name}={mode!r}."
    )


def _normalize_so2_expert_mixing_mode(mode: Optional[str]) -> str:
    mode = mode or "pre_activation"
    allowed = {"pre_activation", "post_activation"}
    if mode not in allowed:
        raise ValueError(f"so2_expert_mixing_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return mode


def _normalize_cg_head_impl(mode: Optional[str]) -> str:
    """P1-3: reduction-path opt-in for the h_b0 late-CG head. ``legacy`` (default)
    reproduces the pre-fusion (0715-refactor) per-path accumulation bit-stably;
    ``fused`` opts in to the grouped-einsum reassociation (see
    ``late_block_expansion_cg.py`` for the full numerics discussion). Validated
    here so a bad value fails fast at embedding construction regardless of which
    output_route is selected -- the h_b0 head itself performs the same check."""
    mode = mode or "legacy"
    allowed = {"legacy", "fused"}
    if mode not in allowed:
        raise ValueError(f"cg_head_impl must be one of {sorted(allowed)}, got {mode!r}.")
    return mode


def _build_so2_post_activation_expert_mixer(
        tp: SO2_Linear,
        activation: torch.nn.Module,
        scalar_dim: int,
        router_hidden_dim: int,
        route_chunk_size: Optional[int],
        route_checkpoint: bool,
) -> SO2PostActivationExpertMixer:
    return SO2PostActivationExpertMixer(
        tp=tp,
        activation=activation,
        router_from_0e=torch.nn.Sequential(
            torch.nn.Linear(scalar_dim, router_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(router_hidden_dim, 1),
        ),
        scalar_dim=scalar_dim,
        route_chunk_size=route_chunk_size,
        checkpoint_routes=route_checkpoint,
    )


def _apply_so2_tp_or_post_activation_mixer(
        module: torch.nn.Module,
        tp_input: torch.Tensor,
        edge_vector: torch.Tensor,
        mole_globals: MOLEGlobals,
        latents: torch.Tensor,
        wigner_D_all,
):
    if module.post_activation_expert_mixer is None:
        features, wigner_D_all = module.tp(
            tp_input,
            edge_vector,
            mole_globals,
            latents,
            wigner_D_all,
        )
        features = module.activation(features)
        return features, wigner_D_all
    return module.post_activation_expert_mixer(
        tp_input,
        edge_vector,
        mole_globals,
        latents,
        wigner_D_all,
    )


def _irreps_slices(irreps: o3.Irreps) -> List[slice]:
    slices = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        slices.append(slice(offset, offset + width))
        offset += width
    return slices


def _focus_feature_index(irreps: o3.Irreps, num_focus: int) -> torch.Tensor:
    focus_indices = []
    for mul, ir in irreps:
        per_channel_focus = torch.arange(mul, dtype=torch.long).remainder(num_focus)
        focus_indices.append(per_channel_focus.repeat_interleave(ir.dim))
    if not focus_indices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(focus_indices, dim=0)


def _equivariant_gate_feature_index(irreps: o3.Irreps) -> Tuple[torch.Tensor, int]:
    gate_indices = []
    offset = 0
    for mul, ir in irreps:
        per_mul_gate = torch.arange(offset, offset + mul, dtype=torch.long)
        gate_indices.append(per_mul_gate.repeat_interleave(ir.dim))
        offset += mul
    if not gate_indices:
        return torch.empty(0, dtype=torch.long), 0
    return torch.cat(gate_indices, dim=0), offset


class PostActivation0eFocusGate(torch.nn.Module):
    """Post-activation scalar router that gates equivariant message channels."""

    def __init__(
            self,
            irreps: o3.Irreps,
            num_focus: int = 1,
            hidden_dim: Optional[int] = None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_focus = int(num_focus)
        if self.num_focus < 1:
            raise ValueError(f"num_focus must be >= 1, got {num_focus!r}")
        if self.irreps[0].ir.l != 0:
            raise ValueError("PostActivation0eFocusGate expects irreps to start with 0e scalars.")
        self.scalar_dim = self.irreps[0].dim
        self.enabled = self.num_focus > 1
        self.register_buffer(
            "focus_index",
            _focus_feature_index(self.irreps, self.num_focus).to(device=device),
        )
        if self.enabled:
            hidden = hidden_dim or max(16, self.scalar_dim)
            self.router = torch.nn.Sequential(
                torch.nn.Linear(self.scalar_dim, hidden, dtype=dtype, device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, self.num_focus, dtype=dtype, device=device),
            )
            torch.nn.init.zeros_(self.router[-1].weight)
            torch.nn.init.zeros_(self.router[-1].bias)
        else:
            self.router = None

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        if not self.enabled or message.numel() == 0:
            return message
        scalars = message[:, :self.scalar_dim]
        focus_scale = 2.0 * torch.sigmoid(self.router(scalars))
        feature_scale = focus_scale.index_select(1, self.focus_index.to(device=message.device))
        return message * feature_scale


class GatedEdgeAggregation(torch.nn.Module):
    """Query-dependent sigmoid gate applied after edge-to-node aggregation."""

    def __init__(
            self,
            irreps: o3.Irreps,
            query_scalar_dim: int,
            sparsity_threshold: float = 1.0e-2,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if query_scalar_dim < 1:
            raise ValueError(f"query_scalar_dim must be >= 1, got {query_scalar_dim!r}")
        feature_gate_index, gate_dim = _equivariant_gate_feature_index(self.irreps)
        if gate_dim < 1:
            raise ValueError("GatedEdgeAggregation requires a non-empty output irreps.")
        self.query_scalar_dim = int(query_scalar_dim)
        self.gate_dim = int(gate_dim)
        self.sparsity_threshold = float(sparsity_threshold)
        self.register_buffer("feature_gate_index", feature_gate_index.to(device=device))
        self.gate = torch.nn.Linear(
            self.query_scalar_dim,
            self.gate_dim,
            dtype=dtype,
            device=device,
        )
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)
        self.last_stats: Optional[Dict[str, float]] = None
        self.last_heatmap: Optional[Dict[str, object]] = None
        self.heatmap_max_nodes = 64

    @staticmethod
    def _tensor_stat(tensor: torch.Tensor, op: str) -> float:
        if tensor.numel() == 0:
            return 0.0
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            return 0.0
        if op == "mean":
            return float(finite.mean().item())
        if op == "std":
            return float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
        if op == "min":
            return float(finite.min().item())
        if op == "max":
            return float(finite.max().item())
        raise ValueError(f"unsupported tensor stat {op!r}")

    @staticmethod
    def _sparsity(tensor: torch.Tensor, threshold: float) -> float:
        if tensor.numel() == 0:
            return 0.0
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            return 0.0
        return float((finite.abs() < threshold).float().mean().item())

    def _top_edge_share_stats(
            self,
            dst_index: Optional[torch.Tensor],
            edge_message: Optional[torch.Tensor],
            dim_size: int,
    ) -> Tuple[float, float, int, int]:
        if dst_index is None or edge_message is None or dst_index.numel() == 0:
            return 0.0, 0.0, 0, 0
        dst = dst_index.detach()
        contribution = edge_message.detach().float().abs().sum(dim=-1)
        active_edges = int(contribution.numel())
        if active_edges == 0:
            return 0.0, 0.0, 0, 0
        per_node_sum = scatter_add(contribution, dst, dim=0, dim_size=dim_size)
        per_node_max = scatter_max(contribution, dst, dim=0, dim_size=dim_size)[0]
        per_node_max = torch.where(
            torch.isfinite(per_node_max),
            per_node_max,
            torch.zeros_like(per_node_max),
        )
        valid = per_node_sum > torch.finfo(per_node_sum.dtype).tiny
        nodes_with_edges = int(valid.sum().item())
        if nodes_with_edges == 0:
            return 0.0, 0.0, active_edges, 0
        share = per_node_max[valid] / per_node_sum[valid].clamp_min(torch.finfo(per_node_sum.dtype).tiny)
        return (
            float(share.mean().item()),
            float(share.max().item()),
            active_edges,
            nodes_with_edges,
        )

    def _build_heatmap(
            self,
            dst_index: Optional[torch.Tensor],
            src_index: Optional[torch.Tensor],
            node_batch: Optional[torch.Tensor],
            edge_message: Optional[torch.Tensor],
            dim_size: int,
    ) -> Optional[Dict[str, object]]:
        if (
                dst_index is None
                or src_index is None
                or edge_message is None
                or dst_index.numel() == 0
                or src_index.numel() != dst_index.numel()
        ):
            return None

        max_nodes = max(1, int(getattr(self, "heatmap_max_nodes", 64)))
        dst = dst_index.detach().long()
        src = src_index.detach().long()
        contribution = edge_message.detach().float().abs().sum(dim=-1)
        valid_edges = (
            (dst >= 0)
            & (src >= 0)
            & (dst < dim_size)
            & (src < dim_size)
            & torch.isfinite(contribution)
            & (contribution > 0)
        )
        if not bool(valid_edges.any().item()):
            return None

        dst = dst[valid_edges]
        src = src[valid_edges]
        contribution = contribution[valid_edges]
        edge_message = edge_message.detach().float()[valid_edges]
        device = contribution.device

        if node_batch is not None and node_batch.numel() >= dim_size:
            batch = node_batch.detach().long().to(device=device)[:dim_size]
            same_graph = batch[dst] == batch[src]
            if not bool(same_graph.any().item()):
                return None
            dst = dst[same_graph]
            src = src[same_graph]
            contribution = contribution[same_graph]
            edge_message = edge_message[same_graph]
            edge_graph = batch[dst]
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
            graph_mass = scatter_add(contribution, edge_graph, dim=0, dim_size=num_graphs)
            selected_graph = int(torch.argmax(graph_mass).item())
            graph_nodes = torch.nonzero(batch == selected_graph, as_tuple=False).flatten()
            graph_edges = edge_graph == selected_graph
            dst = dst[graph_edges]
            src = src[graph_edges]
            contribution = contribution[graph_edges]
            edge_message = edge_message[graph_edges]
        else:
            selected_graph = -1
            graph_nodes = torch.unique(torch.cat([dst, src]), sorted=True)

        if graph_nodes.numel() == 0:
            return None

        visible_nodes = graph_nodes[:max_nodes]
        visible = torch.zeros(dim_size, dtype=torch.bool, device=device)
        visible[visible_nodes] = True
        visible_edges = visible[dst]
        selected_edges = visible_edges & visible[src]
        if not bool(visible_edges.any().item()) or not bool(selected_edges.any().item()):
            return None

        query_pos = torch.full((dim_size,), -1, dtype=torch.long, device=device)
        key_pos = torch.full((dim_size,), -1, dtype=torch.long, device=device)
        query_pos[visible_nodes] = torch.arange(visible_nodes.numel(), device=device)
        key_pos[visible_nodes] = torch.arange(visible_nodes.numel(), device=device)
        qi = query_pos[dst]
        ki = key_pos[src]
        selected = (qi >= 0) & (ki >= 0)
        if not bool(selected.any().item()):
            return None

        flat_size = int(visible_nodes.numel() * visible_nodes.numel())
        flat_index = qi[selected] * int(visible_nodes.numel()) + ki[selected]
        heatmap = scatter_add(
            contribution[selected],
            flat_index,
            dim=0,
            dim_size=flat_size,
        ).reshape(int(visible_nodes.numel()), int(visible_nodes.numel()))

        row_total_all_keys = scatter_add(contribution, dst, dim=0, dim_size=dim_size)[visible_nodes]
        heatmap = heatmap / row_total_all_keys.unsqueeze(-1).clamp_min(
            torch.finfo(heatmap.dtype).tiny
        )
        column_score = heatmap.mean(dim=0)
        top_key_score = float(column_score.max().item()) if column_score.numel() else 0.0
        first_key_score = float(column_score[0].item()) if column_score.numel() else 0.0
        visible_mass = float(heatmap.sum(dim=-1).mean().item()) if heatmap.numel() else 0.0
        local_pos = torch.full((dim_size,), -1, dtype=torch.long, device=device)
        local_pos[graph_nodes] = torch.arange(graph_nodes.numel(), device=device)

        irrep_labels = []
        irrep_mass = []
        query_edge_message = edge_message[visible_edges]
        for idx, ((mul, ir), sl) in enumerate(zip(self.irreps, _irreps_slices(self.irreps))):
            irrep_labels.append(f"{idx}:{mul}x{ir}")
            if query_edge_message.numel() == 0:
                irrep_mass.append(0.0)
            else:
                irrep_mass.append(float(query_edge_message[:, sl].abs().sum().item()))
        irrep_mass_tensor = torch.tensor(irrep_mass, dtype=torch.float32)
        irrep_total = float(irrep_mass_tensor.sum().item())
        if irrep_total > 0.0:
            irrep_share = (irrep_mass_tensor / irrep_total).tolist()
        else:
            irrep_share = [0.0 for _ in irrep_mass]
        return {
            "matrix": heatmap.detach().cpu(),
            "query_nodes": visible_nodes.detach().cpu(),
            "key_nodes": visible_nodes.detach().cpu(),
            "query_node_local": local_pos[visible_nodes].detach().cpu(),
            "key_node_local": local_pos[visible_nodes].detach().cpu(),
            "sample_index": selected_graph,
            "top_key_score": top_key_score,
            "first_key_score": first_key_score,
            "visible_mass": visible_mass,
            "irrep_labels": irrep_labels,
            "irrep_share": irrep_share,
        }

    def _record_stats(
            self,
            gate_values: torch.Tensor,
            pre_gate: torch.Tensor,
            post_gate: torch.Tensor,
            dst_index: Optional[torch.Tensor],
            src_index: Optional[torch.Tensor],
            node_batch: Optional[torch.Tensor],
            edge_message: Optional[torch.Tensor],
            dim_size: int,
    ) -> None:
        with torch.no_grad():
            gate_det = gate_values.detach().float()
            pre_det = pre_gate.detach().float()
            post_det = post_gate.detach().float()
            top_mean, top_max, active_edges, nodes_with_edges = self._top_edge_share_stats(
                dst_index,
                edge_message,
                dim_size,
            )
            self.last_stats = {
                "gate_mean": self._tensor_stat(gate_det, "mean"),
                "gate_std": self._tensor_stat(gate_det, "std"),
                "gate_min": self._tensor_stat(gate_det, "min"),
                "gate_max": self._tensor_stat(gate_det, "max"),
                "gate_sparsity_lt_0_1": self._sparsity(gate_det, 0.1),
                "gate_sparsity_lt_1e_2": self._sparsity(gate_det, 1.0e-2),
                "pre_sparsity_lt_1e_2": self._sparsity(pre_det, self.sparsity_threshold),
                "post_sparsity_lt_1e_2": self._sparsity(post_det, self.sparsity_threshold),
                "pre_activation_max": self._tensor_stat(pre_det.abs(), "max"),
                "post_activation_max": self._tensor_stat(post_det.abs(), "max"),
                "top_edge_share_mean": top_mean,
                "top_edge_share_max": top_max,
                "active_edges": active_edges,
                "nodes_with_edges": nodes_with_edges,
            }
            self.last_heatmap = self._build_heatmap(
                dst_index,
                src_index,
                node_batch,
                edge_message,
                dim_size,
            )

    def forward(
            self,
            aggregated: torch.Tensor,
            query_scalars: torch.Tensor,
            dst_index: Optional[torch.Tensor] = None,
            src_index: Optional[torch.Tensor] = None,
            node_batch: Optional[torch.Tensor] = None,
            edge_message: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        if query_scalars.shape[0] != aggregated.shape[0]:
            raise ValueError(
                "query_scalars and aggregated node messages must have the same leading dimension, "
                f"got {query_scalars.shape[0]} and {aggregated.shape[0]}."
            )
        gate_values = torch.sigmoid(self.gate(query_scalars))
        feature_gate = gate_values.index_select(
            1,
            self.feature_gate_index.to(device=aggregated.device),
        )
        gated = aggregated * feature_gate
        self._record_stats(
            gate_values,
            aggregated,
            gated,
            dst_index,
            src_index,
            node_batch,
            edge_message,
            int(dim_size or aggregated.shape[0]),
        )
        return gated


class SingleHead0eEnvelopeAttention(torch.nn.Module):
    """Single-head destination attention from 0e scalars with cutoff envelope gating."""

    def __init__(
            self,
            node_scalar_dim: int,
            message_scalar_dim: int,
            latent_dim: int,
            attn_dim: int = 32,
            envelope_power: float = 1.0,
            use_latent_bias: bool = True,
            key_source: str = "message",
            key_layer_norm: bool = False,
            query_layer_norm: bool = False,
            qk_layer_norm: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        if attn_dim < 1:
            raise ValueError(f"attn_dim must be >= 1, got {attn_dim!r}")
        if envelope_power <= 0.0:
            raise ValueError(f"envelope_power must be > 0, got {envelope_power!r}")
        self.envelope_power = float(envelope_power)
        self.use_latent_bias = bool(use_latent_bias)
        self.key_source = _normalize_edge_attention_key_source(key_source)
        self.query_layer_norm = bool(query_layer_norm) or bool(qk_layer_norm)
        self.key_layer_norm = bool(key_layer_norm) or bool(qk_layer_norm)
        self.qk_layer_norm = bool(qk_layer_norm)
        self.query_norm = (
            torch.nn.LayerNorm(node_scalar_dim, dtype=dtype, device=device)
            if self.query_layer_norm
            else torch.nn.Identity()
        )
        self.query = torch.nn.Linear(node_scalar_dim, attn_dim, bias=False, dtype=dtype, device=device)
        self.key_norm = (
            torch.nn.LayerNorm(message_scalar_dim, dtype=dtype, device=device)
            if self.key_layer_norm
            else torch.nn.Identity()
        )
        self.key = torch.nn.Linear(message_scalar_dim, attn_dim, bias=False, dtype=dtype, device=device)
        if self.use_latent_bias:
            self.latent_bias = torch.nn.Sequential(
                torch.nn.LayerNorm(latent_dim, dtype=dtype, device=device),
                torch.nn.Linear(latent_dim, attn_dim, dtype=dtype, device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(attn_dim, 1, dtype=dtype, device=device),
            )
        else:
            self.latent_bias = None
        self.logit_scale = attn_dim ** -0.5

    def attention_weights(
            self,
            dst_index: torch.Tensor,
            node_scalars: torch.Tensor,
            message_scalars: torch.Tensor,
            latents: torch.Tensor,
            cutoff_coeffs: torch.Tensor,
            dim_size: int,
    ) -> torch.Tensor:
        if dst_index.numel() == 0:
            return message_scalars.new_zeros((0,))

        q = self.query(self.query_norm(node_scalars.index_select(0, dst_index)))
        k = self.key(self.key_norm(message_scalars))
        logits = (q * k).sum(dim=-1) * self.logit_scale
        if self.latent_bias is not None:
            logits = logits + self.latent_bias(latents).squeeze(-1)

        # detached max: softmax is shift-invariant, so the stabilizer carries no
        # gradient anyway, and torch_scatter's custom kernel must not see
        # forward-mode dual tensors (jvp backend).
        max_per_node = scatter_max(logits.detach(), dst_index, dim=0, dim_size=dim_size)[0]
        max_per_node = torch.where(
            torch.isfinite(max_per_node),
            max_per_node,
            torch.zeros_like(max_per_node),
        )
        shifted = logits - max_per_node.index_select(0, dst_index)
        numer = torch.exp(shifted)
        numer = numer * cutoff_coeffs.clamp_min(0.0).pow(self.envelope_power)

        denom = scatter_add(numer, dst_index, dim=0, dim_size=dim_size)
        denom = denom.clamp_min(torch.finfo(numer.dtype).tiny)
        return numer / denom.index_select(0, dst_index)

    def forward(
            self,
            message: torch.Tensor,
            dst_index: torch.Tensor,
            node_scalars: torch.Tensor,
            message_scalars: torch.Tensor,
            latents: torch.Tensor,
            cutoff_coeffs: torch.Tensor,
            dim_size: int,
    ) -> torch.Tensor:
        weights = self.attention_weights(
            dst_index=dst_index,
            node_scalars=node_scalars,
            message_scalars=message_scalars,
            latents=latents,
            cutoff_coeffs=cutoff_coeffs,
            dim_size=dim_size,
        )
        return scatter_add(message * weights.unsqueeze(-1), dst_index, dim=0, dim_size=dim_size)


def _instruction_get(instruction, name: str, index: int):
    if hasattr(instruction, name):
        return getattr(instruction, name)
    return instruction[index]


def _onehot_weight_shape(connection_mode: str, mul1: int, mul2: int, mul_out: int) -> Tuple[int, ...]:
    if connection_mode == "uvu":
        return (mul1, mul2)
    if connection_mode == "uvw":
        return (mul1, mul2, mul_out)
    raise ValueError(f"unsupported scalar onehot TP connection mode {connection_mode!r}")


class ScalarOnehotTP(torch.nn.Module):
    """Fast scalar-onehot tensor product without storing an e3nn TP module."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *,
        weight_shapes: Optional[List[Tuple[int, ...]]] = None,
        path_weights: Optional[List[float]] = None,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self._in1_slices = _irreps_slices(self.irreps_in1)
        self._in2_slices = _irreps_slices(self.irreps_in2)
        self._out_slices = _irreps_slices(self.irreps_out)

        paths = []
        scales = []
        offset = 0
        for idx, instruction in enumerate(instructions):
            has_weight = _instruction_get(instruction, "has_weight", 4)
            if not has_weight:
                raise ValueError("ScalarOnehotTP requires weighted instructions")

            i_in1 = _instruction_get(instruction, "i_in1", 0)
            i_in2 = _instruction_get(instruction, "i_in2", 1)
            i_out = _instruction_get(instruction, "i_out", 2)
            connection_mode = _instruction_get(instruction, "connection_mode", 3)

            mul1, ir1 = self.irreps_in1[i_in1]
            mul2, ir2 = self.irreps_in2[i_in2]
            mul_out, ir_out = self.irreps_out[i_out]
            if ir2.l != 0 or ir2.p != 1 or ir1 != ir_out:
                raise ValueError("ScalarOnehotTP only supports scalar second input preserving irreps")

            shape = (
                tuple(weight_shapes[idx])
                if weight_shapes is not None
                else _onehot_weight_shape(connection_mode, mul1, mul2, mul_out)
            )
            numel = math.prod(shape)
            path_weight = (
                float(path_weights[idx])
                if path_weights is not None
                else float(_instruction_get(instruction, "path_weight", 5))
            )
            scale = path_weight / math.sqrt(ir_out.dim)
            paths.append((i_in1, i_in2, i_out, connection_mode, offset, shape, mul1, mul2, mul_out, ir1.dim))
            scales.append(scale)
            offset += numel

        self._paths = tuple(paths)
        self._uvu_same_input = (
            len(self._paths) > 1
            and all(path[3] == "uvu" for path in self._paths)
            and len({path[1] for path in self._paths}) == 1
            and len({path[7] for path in self._paths}) == 1
        )
        self._output_paths_are_unique = len({path[2] for path in self._paths}) == len(self._paths)
        if self._uvu_same_input:
            uvu_gain_scales = []
            for idx, path in enumerate(self._paths):
                uvu_gain_scales.extend([scales[idx]] * path[6])
            self._uvu_total_mul = sum(path[6] for path in self._paths)
            self._uvu_mul2 = self._paths[0][7]
        else:
            uvu_gain_scales = []
            self._uvu_total_mul = 0
            self._uvu_mul2 = 0
        init_dtype = weight.dtype if weight is not None else torch.get_default_dtype()
        init_device = weight.device if weight is not None else None
        self.register_buffer(
            "_path_scales",
            torch.tensor(scales, dtype=init_dtype, device=init_device),
            persistent=False,
        )
        self.register_buffer(
            "_uvu_gain_scales",
            torch.tensor(uvu_gain_scales, dtype=init_dtype, device=init_device),
            persistent=False,
        )
        self.weight = torch.nn.Parameter(torch.empty(offset, dtype=init_dtype, device=init_device))
        if weight is None:
            torch.nn.init.normal_(self.weight, std=1.0 / math.sqrt(max(offset, 1)))
        else:
            if weight.numel() != offset:
                raise ValueError(f"ScalarOnehotTP weight has {weight.numel()} values, expected {offset}")
            with torch.no_grad():
                self.weight.copy_(weight.reshape(-1).to(dtype=self.weight.dtype))

    @classmethod
    def from_e3nn(cls, tp: torch.nn.Module) -> "ScalarOnehotTP":
        weight_views = list(tp.weight_views())
        path_weights = [float(_instruction_get(instruction, "path_weight", 5)) for instruction in tp.instructions]
        return cls(
            tp.irreps_in1,
            tp.irreps_in2,
            tp.irreps_out,
            tp.instructions,
            weight_shapes=[tuple(view.shape) for view in weight_views],
            path_weights=path_weights,
            weight=tp.weight.detach(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[:-1] != y.shape[:-1]:
            raise ValueError(f"onehot TP input shapes must share leading dims, got {x.shape} and {y.shape}")
        if x.shape[-1] != self.irreps_in1.dim or y.shape[-1] != self.irreps_in2.dim:
            raise ValueError(
                f"onehot TP input dims mismatch: got x={x.shape[-1]}, y={y.shape[-1]}, "
                f"expected x={self.irreps_in1.dim}, y={self.irreps_in2.dim}"
            )

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        out = x_flat.new_zeros((x_flat.shape[0], self.irreps_out.dim))
        path_scales = self._path_scales.to(dtype=x_flat.dtype, device=x_flat.device)

        if self._uvu_same_input:
            in2_idx = self._paths[0][1]
            y_block = y_flat[:, self._in2_slices[in2_idx]]
            packed_weight = self.weight.reshape(self._uvu_total_mul, self._uvu_mul2)
            packed_weight = packed_weight * self._uvu_gain_scales.to(
                dtype=x_flat.dtype, device=x_flat.device
            ).unsqueeze(-1)
            gains = torch.matmul(y_block, packed_weight.transpose(0, 1))
            gain_offset = 0
            for path in self._paths:
                i_in1, _, i_out, _, _, _, mul1, _, _, ir_dim = path
                x_block = x_flat[:, self._in1_slices[i_in1]].reshape(x_flat.shape[0], mul1, ir_dim)
                gain_block = gains.narrow(1, gain_offset, mul1).unsqueeze(-1)
                mixed = (x_block * gain_block).reshape(
                    x_flat.shape[0], mul1 * ir_dim
                )
                if self._output_paths_are_unique:
                    out[:, self._out_slices[i_out]] = mixed
                else:
                    out[:, self._out_slices[i_out]] += mixed
                gain_offset += mul1
            return out.reshape(*leading_shape, self.irreps_out.dim)

        for idx, path in enumerate(self._paths):
            i_in1, i_in2, i_out, connection_mode, offset, shape, mul1, mul2, mul_out, ir_dim = path
            weight = self.weight.narrow(0, offset, math.prod(shape)).reshape(shape)
            x_block = x_flat[:, self._in1_slices[i_in1]].reshape(x_flat.shape[0], mul1, ir_dim)
            y_block = y_flat[:, self._in2_slices[i_in2]].reshape(y_flat.shape[0], mul2)

            if connection_mode == "uvu":
                mixed = x_block * torch.einsum("nv,uv->nu", y_block, weight).unsqueeze(-1)
            elif connection_mode == "uvw":
                mixed = torch.einsum("nui,nv,uvw->nwi", x_block, y_block, weight)
            else:
                raise ValueError(f"unsupported scalar onehot TP connection mode {connection_mode!r}")

            out[:, self._out_slices[i_out]] += (
                mixed * path_scales[idx]
            ).reshape(x_flat.shape[0], mul_out * ir_dim)

        return out.reshape(*leading_shape, self.irreps_out.dim)


def _scalar_onehot_tp_fast(tp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact fast path for scalar-onehot tensor products.

    The LEM onehot tensor products all use scalar second inputs. For these paths
    e3nn's TP reduces to per-irrep channel scaling/mixing, so we can avoid the
    generic tensor product dispatcher while reusing the original e3nn weights.
    """
    irreps_in1 = tp.irreps_in1
    irreps_in2 = tp.irreps_in2
    irreps_out = tp.irreps_out
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(f"onehot TP input shapes must share leading dims, got {x.shape} and {y.shape}")
    if x.shape[-1] != irreps_in1.dim or y.shape[-1] != irreps_in2.dim:
        raise ValueError(
            f"onehot TP input dims mismatch: got x={x.shape[-1]}, y={y.shape[-1]}, "
            f"expected x={irreps_in1.dim}, y={irreps_in2.dim}"
        )

    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    out = x_flat.new_zeros((x_flat.shape[0], irreps_out.dim))

    in1_slices = _irreps_slices(irreps_in1)
    in2_slices = _irreps_slices(irreps_in2)
    out_slices = _irreps_slices(irreps_out)
    weight_views = iter(tp.weight_views())

    for instruction in tp.instructions:
        has_weight = _instruction_get(instruction, "has_weight", 4)
        if not has_weight:
            raise ValueError("scalar_fast onehot TP requires weighted instructions")

        i_in1 = _instruction_get(instruction, "i_in1", 0)
        i_in2 = _instruction_get(instruction, "i_in2", 1)
        i_out = _instruction_get(instruction, "i_out", 2)
        connection_mode = _instruction_get(instruction, "connection_mode", 3)
        path_weight = _instruction_get(instruction, "path_weight", 5)

        mul1, ir1 = irreps_in1[i_in1]
        mul2, ir2 = irreps_in2[i_in2]
        mul_out, ir_out = irreps_out[i_out]
        if ir2.l != 0 or ir2.p != 1 or ir1 != ir_out:
            raise ValueError("scalar_fast onehot TP only supports scalar second input preserving irreps")

        weight = next(weight_views)
        x_block = x_flat[:, in1_slices[i_in1]].reshape(x_flat.shape[0], mul1, ir1.dim)
        y_block = y_flat[:, in2_slices[i_in2]].reshape(y_flat.shape[0], mul2)
        path_weight = x_flat.new_tensor(path_weight)

        if connection_mode == "uvu":
            if mul_out != mul1:
                raise ValueError("uvu scalar_fast path requires output multiplicity to match input1")
            mixed = x_block * torch.einsum("nv,uv->nu", y_block, weight).unsqueeze(-1)
        elif connection_mode == "uvw":
            mixed = torch.einsum("nui,nv,uvw->nwi", x_block, y_block, weight)
        else:
            raise ValueError(f"unsupported scalar_fast onehot TP connection mode {connection_mode!r}")

        # e3nn's real CG convention for l x 0 -> l contributes 1/sqrt(2l+1).
        cg_scalar_norm = x_flat.new_tensor(ir_out.dim).rsqrt()
        out[:, out_slices[i_out]] += (
            mixed * path_weight * cg_scalar_norm
        ).reshape(x_flat.shape[0], mul_out * ir_out.dim)

    return out.reshape(*leading_shape, irreps_out.dim)


def _apply_onehot_tp(tp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    mode = _normalize_onehot_tp_mode(mode)
    if isinstance(tp, ScalarOnehotTP):
        return tp(x, y)
    return _scalar_onehot_tp_fast(tp, x, y)


@Embedding.register("lem_moe_v3")
class LemMoEV3(torch.nn.Module):
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
            so2_expert_mixing_mode: str = "pre_activation",
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

        super(LemMoEV3, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax
        self.num_experts = num_experts

        # 使用 log.info 打印参数
        log.info(f'[LemMoEV3] Initialized DeepSeek-V3 Style MoE.')
        log.info(f'  - Num Shared Experts: {num_shared_experts}')
        log.info(f'  - Num Routed Experts: {self.num_experts}')
        log.info(f'  - Top-K Actived Routed Experts: {top_k}')
        log.info(f'  - Full Expert Fast Path: {mole_full_expert_fast_path}')
        log.info(f'  - MoLE Linear Mode: {mole_linear_mode or "env/default"}')
        log.info(f'  - Strategy: Shared Expert + Aux-Loss-Free Balancing (Sigmoid Routing)')
        if ffn_hidden_factor > 1.0:
            log.info(
                f"  - EqV3 SO3 Grid FFN: enabled (hidden_factor={ffn_hidden_factor}, "
                f"apply_to_last={ffn_apply_to_last})"
            )
        else:
            log.info("  - EqV3 SO3 Grid FFN: disabled")
        mean_max_prob_lower_bound = 1.0 / self.num_experts
        # 上界：One-Hot 分布，max为 1.0，平方为 1.0
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
            aux_loss_free=True,  # 开启 DeepSeek 负载均衡
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

            use_node_ffn = ffn_hidden_factor > 1.0 and ((i < n_layers - 1) or ffn_apply_to_last)

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
                use_node_ffn=use_node_ffn,
                so2_wigner_apply_mode=so2_wigner_apply_mode,
                so2_fusion_mode=so2_fusion_mode,
                mole_linear_mode=mole_linear_mode,
                so2_expert_mixing_mode=self.so2_expert_mixing_mode,
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
                num_experts=num_experts,
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

    def _apply_block_native_output_heads(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        atom_type: torch.Tensor,
        edge_index: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_node_blocks = self.out_node(node_features)
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
        # 返回: coeffs [Batch, Num_Experts], monitor_val (mean max prob)
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)
        topk_indices, topk_values = self.router.last_topk()

        # 不再记录 z_loss，改为记录监控指标 mean_max_prob
        # 这个值越接近 1.0 表示路由越自信，接近 0.5 (TopK=1时) 表示犹豫
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv
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

        if activation_type not in {"gate", "swiglu_s2"}:
            raise ValueError(f"Unsupported activation_type={activation_type!r}")
        use_s2 = activation_type == "swiglu_s2" and can_use_flat_s2_patch(
            self.irreps_out,
            mode=swiglu_s2_compat_mode,
        )
        if use_s2:
            self.activation = FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=swiglu_s2_grid_resolution,
            )
            tp_out_irreps = self.activation.tp_main_irreps
            extra_m0_outsize = self.activation.extra_m0_outsize
        else:
            self.activation = build_gate_activation(self.irreps_out)
            tp_out_irreps = self.activation.irreps_in
            extra_m0_outsize = 0

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in + self.edge_irreps_in,
            irreps_out=tp_out_irreps,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
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

        self.focus_gate = PostActivation0eFocusGate(
            self.irreps_out,
            num_focus=num_focus,
            dtype=dtype,
            device=device,
        )
        if self.node_message_aggregation == "single_head_0e":
            self.node_attention = SingleHead0eEnvelopeAttention(
                node_scalar_dim=self.irreps_in[0].dim,
                message_scalar_dim=self.irreps_out[0].dim,
                latent_dim=latent_dim,
                attn_dim=focus_attention_dim,
                envelope_power=self.edge_attention_envelope_power,
                use_latent_bias=self.edge_attention_use_latent_bias,
                key_source=self.edge_attention_key_source,
                key_layer_norm=self.edge_attention_key_layer_norm,
                query_layer_norm=self.edge_attention_query_layer_norm,
                qk_layer_norm=self.edge_attention_qk_layer_norm,
                dtype=dtype,
                device=device,
            )
        else:
            self.node_attention = None
        if self.edge_aggregation_gated_attention:
            self.edge_aggregation_gate = GatedEdgeAggregation(
                self.irreps_out,
                query_scalar_dim=self.irreps_in[0].dim,
                dtype=dtype,
                device=device,
            )
        else:
            self.edge_aggregation_gate = None

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
        message = self.focus_gate(message)
        scalars = message[:, :self.irreps_out[0].dim]

        if self.edge_message_env_weight:
            weights = self.env_embed_mlps(latents[active_edges])
            weighted_message = self._env_weighter(message, weights)
        else:
            weighted_message = message
        active_edge_center = edge_center[active_edges]
        active_edge_neighbor = edge_neighbor[active_edges]
        if self.node_attention is None:
            new_node_features = scatter(
                weighted_message,
                active_edge_center,
                dim=0,
                dim_size=node_features.shape[0],
            )
        else:
            node_scalars = node_in[:, :self.irreps_in[0].dim]
            new_node_features = self.node_attention(
                weighted_message,
                active_edge_center,
                node_scalars,
                scalars,
                latents[active_edges],
                cutoff_coeffs[active_edges],
                dim_size=node_features.shape[0],
            )
        if self.edge_aggregation_gate is not None:
            if new_node_features.shape[0] != node_features.shape[0]:
                padded_node_features = node_features.new_zeros(
                    node_features.shape[0],
                    new_node_features.shape[1],
                )
                padded_node_features[:new_node_features.shape[0]] = new_node_features
                new_node_features = padded_node_features
            node_scalars = node_in[:, :self.irreps_in[0].dim]
            new_node_features = self.edge_aggregation_gate(
                new_node_features,
                node_scalars,
                dst_index=active_edge_center,
                src_index=active_edge_neighbor,
                node_batch=node_batch,
                edge_message=weighted_message,
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
                # 维度不同，必须经过 linear_res
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
            so2_expert_mixing_mode: str = "pre_activation",
            so2_expert_route_chunk_size: Optional[int] = None,
            so2_expert_route_checkpoint: bool = False,
            so2_output_router_hidden_dim: int = 32,
            onehot_tp_mode: Optional[str] = None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
    ):
        super(UpdateEdge, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.node_irreps_in = node_irreps_in
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
            self.irreps_in,
            norm_eps,
            dtype,
            device,
        )

        if activation_type not in {"gate", "swiglu_s2"}:
            raise ValueError(f"Unsupported activation_type={activation_type!r}")
        use_s2 = activation_type == "swiglu_s2" and can_use_flat_s2_patch(
            self.irreps_out,
            mode=swiglu_s2_compat_mode,
        )
        if use_s2:
            self.activation = FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=swiglu_s2_grid_resolution,
            )
            tp_out_irreps = self.activation.tp_main_irreps
            extra_m0_outsize = self.activation.extra_m0_outsize
        else:
            self.activation = build_gate_activation(self.irreps_out)
            tp_out_irreps = self.activation.irreps_in
            extra_m0_outsize = 0

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.irreps_in + self.node_irreps_in,
            irreps_out=tp_out_irreps,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
        )

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
                active_edges, edge_one_hot, wigner_D_all, mole_globals):  # Accept globals
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = self.node_norm(new_node_features) if self.node_norm is not None else new_node_features
        edge_in = self.edge_norm(edge_features) if self.edge_norm is not None else edge_features

        tp_input = torch.cat(
            [
                node_in[edge_center[active_edges]],
                edge_in,
                node_in[edge_neighbor[active_edges]],
            ],
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

        scalars = new_edge_features[:, :self.irreps_out[0].dim]
        assert len(scalars.shape) == 2

        weights = self.edge_embed_mlps(latents[active_edges])
        new_edge_features = self._edge_weighter(new_edge_features, weights)

        # update latent

        new_latents = self.latents_mlp_1(torch.cat(
            [
                self.ln(latents[active_edges]),
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
                # 维度不同，必须经过 linear_res
                edge_features = coefficient_old * self.linear_res(edge_features) + coefficient_new * new_edge_features

            latents = torch.index_copy(
                latents, 0, active_edges,
                coefficient_new * new_latents + coefficient_old * latents[active_edges]
            )
        else:
            edge_features = new_edge_features
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


class Layer(torch.nn.Module):
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
        super(Layer, self).__init__()

        self.res_update = res_update
        self.avg_num_neighbors = avg_num_neighbors
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.dtype = dtype
        self.device = device
        self.num_types = num_types

        self.edge_update = self._edge_update_type()(
            node_irreps_in=self.irreps_in,
            num_types=num_types,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
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
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
            mole_linear_mode=mole_linear_mode,
            so2_expert_mixing_mode=so2_expert_mixing_mode,
            so2_expert_route_chunk_size=so2_expert_route_chunk_size,
            so2_expert_route_checkpoint=so2_expert_route_checkpoint,
            so2_output_router_hidden_dim=so2_output_router_hidden_dim,
            onehot_tp_mode=onehot_tp_mode,
        )

        self.node_update = self._node_update_type()(
            edge_irreps_in=self.edge_update.irreps_out,
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

        self.node_ffn = None
        if use_node_ffn:
            self.node_ffn = EqV3StyleNodeFFN(
                self.irreps_out,
                hidden_factor=ffn_hidden_factor,
                norm_type=equivariant_norm_type,
                norm_eps=norm_eps,
                grid_resolution=swiglu_s2_grid_resolution,
                dtype=dtype,
                device=device,
            )

    def forward(self, latents, node_features, edge_features, node_onehot, edge_index, edge_vector, atom_type,
                cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all, mole_globals, node_batch=None):
        edge_features, latents, wigner_D_all = self.edge_update(latents, node_features, node_onehot, edge_features,
                                                                edge_index, edge_vector, cutoff_coeffs, active_edges,
                                                                edge_one_hot, wigner_D_all, mole_globals)
        node_features = self.node_update(latents, node_features, edge_features, atom_type, node_onehot, edge_index,
                                         edge_vector, cutoff_coeffs, active_edges, wigner_D_all, mole_globals,
                                         node_batch=node_batch)
        if self.node_ffn is not None:
            node_features = self.node_ffn(node_features)

        return latents, node_features, edge_features, wigner_D_all
