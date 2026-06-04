from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from e3nn import o3
from torch_runstats.scatter import scatter

from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3 import LemMoEV3, _apply_onehot_tp
from .lem_moe_v3_h0 import LemMoEV3H0

log = logging.getLogger(__name__)


def _reset_copied_expert(module: torch.nn.Module) -> None:
    for child in module.modules():
        if child is module:
            continue
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _num_graphs_from_globals(mole_globals: MOLEGlobals | None) -> int:
    if mole_globals is None:
        return 1
    coefficients = getattr(mole_globals, "coefficients", None)
    if coefficients is not None:
        return int(coefficients.shape[0])
    split_sizes = getattr(mole_globals, "split_sizes", None)
    if split_sizes is not None:
        return len(split_sizes)
    sizes = getattr(mole_globals, "_sizes_tensor", None)
    if sizes is not None:
        return int(sizes.numel())
    return 1


def _single_expert_globals(
        mole_globals: MOLEGlobals | None,
        *,
        n_rows: int,
        device: torch.device,
        dtype: torch.dtype,
) -> MOLEGlobals:
    coefficients = torch.ones(
        (_num_graphs_from_globals(mole_globals), 1),
        device=device,
        dtype=dtype,
    )
    if mole_globals is None:
        return MOLEGlobals(coefficients=coefficients, split_sizes=(int(n_rows),))

    return MOLEGlobals(
        coefficients=coefficients,
        sizes=getattr(mole_globals, "sizes", None),
        split_sizes=getattr(mole_globals, "split_sizes", None),
        graph_index=getattr(mole_globals, "graph_index", None),
    )


class NonLinearExpertSO2Stack(torch.nn.Module):
    """Run each expert TP independently before post-activation output mixing."""

    def __init__(self, base_tp: torch.nn.Module, num_experts: int = 2):
        super().__init__()
        self.num_experts = int(num_experts)
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts!r}")

        experts = [base_tp]
        for _ in range(1, self.num_experts):
            expert = copy.deepcopy(base_tp)
            _reset_copied_expert(expert)
            experts.append(expert)
        self.experts = torch.nn.ModuleList(experts)

    def forward(
            self,
            x: torch.Tensor,
            R: torch.Tensor,
            mole_globals: MOLEGlobals | None,
            latents: torch.Tensor | None = None,
            wigner_D_all=None,
    ):
        single_globals = _single_expert_globals(
            mole_globals,
            n_rows=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []
        current_wigner = wigner_D_all
        for expert in self.experts:
            out, current_wigner = expert(
                x,
                R,
                single_globals,
                latents,
                current_wigner,
            )
            outputs.append(out)
        return torch.stack(outputs, dim=1), current_wigner


class PostActivationExpertMixer(torch.nn.Module):
    """Mix expert outputs after equivariant activation using only 0e scalars."""

    def __init__(
            self,
            irreps: o3.Irreps,
            num_experts: int = 2,
            hidden_dim: int | None = None,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_experts = int(num_experts)
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts!r}")
        if self.irreps[0].ir.l != 0:
            raise ValueError("PostActivationExpertMixer expects irreps to start with 0e scalars.")
        self.scalar_dim = self.irreps[0].dim
        hidden = hidden_dim or max(16, self.scalar_dim)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(self.scalar_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.router[-1].weight)
        torch.nn.init.zeros_(self.router[-1].bias)
        self.last_weights = None

    def forward(self, expert_outputs: torch.Tensor):
        if expert_outputs.ndim != 3:
            raise ValueError(
                "expert_outputs must have shape [n_edges, num_experts, irreps_dim], "
                f"got {tuple(expert_outputs.shape)}"
            )
        if expert_outputs.shape[1] != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} expert outputs, got {expert_outputs.shape[1]}"
            )
        scalars = expert_outputs[:, :, :self.scalar_dim]
        logits = self.router(scalars.reshape(-1, self.scalar_dim)).reshape(
            expert_outputs.shape[0],
            self.num_experts,
        )
        weights = torch.softmax(logits, dim=1)
        self.last_weights = weights.detach()
        mixed = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return mixed, weights


def _activate_expert_outputs(activation: torch.nn.Module, expert_outputs: torch.Tensor) -> torch.Tensor:
    activated = [
        activation(expert_outputs[:, expert_idx, :])
        for expert_idx in range(expert_outputs.shape[1])
    ]
    return torch.stack(activated, dim=1)


class NonLinearExpertUpdateNode(torch.nn.Module):
    def __init__(self, base_update: torch.nn.Module, num_experts: int = 2):
        super().__init__()
        self.base = base_update
        self.expert_tp = NonLinearExpertSO2Stack(self.base.tp, num_experts=num_experts)
        delattr(self.base, "tp")
        self.expert_mixer = PostActivationExpertMixer(
            self.base.activation.irreps_out,
            num_experts=num_experts,
        )
        self.irreps_in = self.base.irreps_in
        self.irreps_out = self.base.irreps_out

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                cutoff_coeffs, active_edges, wigner_D_all, mole_globals):
        base = self.base
        edge_center = edge_index[0]

        new_node_features = node_features
        node_in = base.node_norm(new_node_features) if base.node_norm is not None else new_node_features
        edge_in = base.edge_norm(edge_features) if base.edge_norm is not None else edge_features
        expert_messages, wigner_D_all = self.expert_tp(
            torch.cat(
                [node_in[edge_center[active_edges]], edge_in],
                dim=-1,
            ),
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )
        expert_messages = _activate_expert_outputs(base.activation, expert_messages)
        message, _ = self.expert_mixer(expert_messages)
        message = base.lin_post(message)
        if hasattr(base, "focus_gate"):
            message = base.focus_gate(message)
        scalars = message[:, :base.irreps_out[0].dim]

        weights = base.env_embed_mlps(latents[active_edges])
        weighted_message = base._env_weighter(message, weights)
        active_edge_center = edge_center[active_edges]
        if getattr(base, "node_attention", None) is None:
            new_node_features = scatter(
                weighted_message,
                active_edge_center,
                dim=0,
            )
        else:
            node_scalars = node_in[:, :base.irreps_in[0].dim]
            new_node_features = base.node_attention(
                weighted_message,
                active_edge_center,
                node_scalars,
                scalars,
                latents[active_edges],
                cutoff_coeffs[active_edges],
                dim_size=node_features.shape[0],
            )

        if base.env_sum_normalizations.ndim < 1:
            norm_const = base.env_sum_normalizations
        else:
            norm_const = base.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        assert len(scalars.shape) == 2

        new_node_features = new_node_features * norm_const

        if base.res_update:
            update_coefficients = base._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old

            if base.use_identity_res:
                node_features = coefficient_old * node_features + coefficient_new * new_node_features
            else:
                node_features = coefficient_old * base.linear_res(node_features) + coefficient_new * new_node_features
        else:
            node_features = new_node_features

        if base.use_layer_onehot_tp:
            onehot_tune_node_feat = _apply_onehot_tp(
                base.node_onehot_tp, node_features, node_onehot, base.onehot_tp_mode
            )
            node_features = node_features + onehot_tune_node_feat

        return node_features


class NonLinearExpertUpdateEdge(torch.nn.Module):
    def __init__(self, base_update: torch.nn.Module, num_experts: int = 2):
        super().__init__()
        self.base = base_update
        self.expert_tp = NonLinearExpertSO2Stack(self.base.tp, num_experts=num_experts)
        delattr(self.base, "tp")
        self.expert_mixer = PostActivationExpertMixer(
            self.base.activation.irreps_out,
            num_experts=num_experts,
        )
        self.irreps_in = self.base.irreps_in
        self.irreps_out = self.base.irreps_out

    def forward(self, latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs,
                active_edges, edge_one_hot, wigner_D_all, mole_globals):
        base = self.base
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = base.node_norm(new_node_features) if base.node_norm is not None else new_node_features
        edge_in = base.edge_norm(edge_features) if base.edge_norm is not None else edge_features

        expert_edge_features, wigner_D_all = self.expert_tp(
            torch.cat(
                [
                    node_in[edge_center[active_edges]],
                    edge_in,
                    node_in[edge_neighbor[active_edges]],
                ],
                dim=-1,
            ),
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )

        expert_edge_features = _activate_expert_outputs(base.activation, expert_edge_features)
        new_edge_features, _ = self.expert_mixer(expert_edge_features)
        new_edge_features = base.lin_post(new_edge_features)

        scalars = new_edge_features[:, :base.irreps_out[0].dim]
        assert len(scalars.shape) == 2

        weights = base.edge_embed_mlps(latents[active_edges])
        new_edge_features = base._edge_weighter(new_edge_features, weights)

        new_latents = base.latents_mlp_1(torch.cat(
            [
                base.ln(latents[active_edges]),
                scalars,
            ], dim=-1))

        new_latents = base.latents_mlp_2(torch.cat(
            [
                new_latents,
                edge_one_hot,
            ], dim=-1))

        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

        if base.res_update:
            update_coefficients = base._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old

            if base.use_identity_res:
                edge_features = coefficient_old * edge_features + coefficient_new * new_edge_features
            else:
                edge_features = coefficient_old * base.linear_res(edge_features) + coefficient_new * new_edge_features

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
        if base.use_layer_onehot_tp:
            onehot_tune_edge_feat = _apply_onehot_tp(
                base.edge_onehot_tp, edge_features, edge_one_hot, base.onehot_tp_mode
            )
            edge_features = edge_features + onehot_tune_edge_feat

        return edge_features, latents, wigner_D_all


def _install_non_linear_experts(model: torch.nn.Module, num_experts: int) -> None:
    for layer in model.layers:
        layer.edge_update = NonLinearExpertUpdateEdge(layer.edge_update, num_experts=num_experts)
        layer.node_update = NonLinearExpertUpdateNode(layer.node_update, num_experts=num_experts)


def _prepare_parent_kwargs(
        *,
        num_experts: int,
        num_shared_experts: int,
        top_k,
        mole_linear_mode,
        so2_fusion_mode,
        kwargs: dict[str, Any],
) -> dict[str, Any]:
    if num_experts != 2:
        log.warning("lem_non_linear is intended for two full experts in this experiment; got num_experts=%s", num_experts)
    if num_shared_experts not in (0, None):
        log.warning("lem_non_linear ignores shared experts; got num_shared_experts=%s", num_shared_experts)
    if top_k not in (None, num_experts):
        log.warning("lem_non_linear fully activates every expert and ignores top_k=%s", top_k)
    if mole_linear_mode not in (None, "split_loop"):
        log.warning("lem_non_linear torch path forces mole_linear_mode='split_loop', got %s", mole_linear_mode)
    if so2_fusion_mode not in (None, "staged"):
        log.warning("lem_non_linear torch path forces so2_fusion_mode='staged', got %s", so2_fusion_mode)

    parent_kwargs = dict(kwargs)
    parent_kwargs.update(
        num_experts=1,
        num_shared_experts=0,
        top_k=None,
        mole_full_expert_fast_path=True,
        mole_linear_mode="split_loop",
        so2_fusion_mode="staged",
    )
    return parent_kwargs


@Embedding.register("lem_non_linear")
class LemNonLinear(LemMoEV3):
    def __init__(
            self,
            num_experts: int = 2,
            num_shared_experts: int = 0,
            top_k=None,
            mole_linear_mode=None,
            so2_fusion_mode=None,
            mole_full_expert_fast_path: bool = True,
            **kwargs: Any,
    ):
        self.non_linear_num_experts = int(num_experts)
        parent_kwargs = _prepare_parent_kwargs(
            num_experts=self.non_linear_num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            mole_linear_mode=mole_linear_mode,
            so2_fusion_mode=so2_fusion_mode,
            kwargs=kwargs,
        )
        super().__init__(**parent_kwargs)
        self.num_experts = self.non_linear_num_experts
        _install_non_linear_experts(self, self.non_linear_num_experts)
        log.info(
            "[LemNonLinear] Enabled post-activation expert mixing: num_experts=%s, backend=torch_split_loop",
            self.non_linear_num_experts,
        )


@Embedding.register("lem_non_linear_h0")
class LemNonLinearH0(LemMoEV3H0):
    def __init__(
            self,
            num_experts: int = 2,
            num_shared_experts: int = 0,
            top_k=None,
            mole_linear_mode=None,
            so2_fusion_mode=None,
            mole_full_expert_fast_path: bool = True,
            **kwargs: Any,
    ):
        self.non_linear_num_experts = int(num_experts)
        parent_kwargs = _prepare_parent_kwargs(
            num_experts=self.non_linear_num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            mole_linear_mode=mole_linear_mode,
            so2_fusion_mode=so2_fusion_mode,
            kwargs=kwargs,
        )
        super().__init__(**parent_kwargs)
        self.num_experts = self.non_linear_num_experts
        _install_non_linear_experts(self, self.non_linear_num_experts)
        log.info(
            "[LemNonLinearH0] Enabled post-activation expert mixing: num_experts=%s, backend=torch_split_loop",
            self.non_linear_num_experts,
        )
