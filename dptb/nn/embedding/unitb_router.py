"""Edge-router construction, invariant descriptors, and dispatch metadata."""
import os
from typing import Any

import torch

from dptb.data import AtomicDataDict
from dptb.nn.route_drop import (validate_route_drop, sample_structure_routes,
                                apply_structure_routes, record_route_drop)
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLERouterV3

from .prior_common import _sorted_irrep_coordinate_index


# SO2 routes that apply activation-space (per-edge) dispatch without building a weight
# per edge: 'staged' and the grouped streaming route reach every MOLELinear through
# MOLELinear.forward; fused_p0 segments its grouped GEMM by expert id
# (dptb.nn.so2_activation_routes).  The other SO2CUDA fused routes consume
# fc._mix_expert_parameters() and would build one [out_features, in_features]
# weight per edge.
_PRIOR_ACTIVATE_ROUTES = ("staged", "streamed_m_major_cueq", "streamed_m_major_fused_p0")

class UniTBRouter:
    @staticmethod
    def _sample_routes(data, probability):
        return sample_structure_routes(data, probability)

    def __init__(self, **kwargs: Any):
        structure_strategy = kwargs.pop("_structure_strategy", None)
        for name in ("sympe", "latent_irrep_dot", "node_bilinear"):
            config = kwargs.pop(name, None)
            if config and config.get("enabled", False):
                raise ValueError(f"{name} belongs to an archived model")
        self.edge_router_top1_mode = kwargs.pop("edge_router_top1_mode", "legacy")
        if self.edge_router_top1_mode not in ("legacy", "switch"):
            raise ValueError("edge_router_top1_mode must be legacy or switch")
        edge_router_in_features = kwargs.pop("edge_router_in_features", None)
        self.edge_router_unique_types = bool(kwargs.pop("edge_router_unique_types", True))
        self.edge_moe_compact_dispatch = bool(kwargs.pop("edge_moe_compact_dispatch", True))
        self.edge_moe_compact_min_edges = int(kwargs.pop("edge_moe_compact_min_edges", 16384))
        self.edge_router_prior_activate = bool(kwargs.pop("edge_router_prior_activate", False))
        self.edge_router_prior_stats = str(kwargs.pop("edge_router_prior_stats", "") or "")
        self.edge_router_prior_cg = bool(kwargs.pop("edge_router_prior_cg", False))
        self.edge_router_temperature = float(kwargs.pop("edge_router_temperature", 1.0))
        if self.edge_router_top1_mode == "switch" and self.edge_router_temperature != 1.0:
            raise ValueError("edge_router_temperature applies to the top-k gate; the Switch top-1 router has none")
        # edge-router options; the defaults reproduce the production router, see MOLERouterV3
        self.edge_router_options = dict(
            logit_kind=str(kwargs.pop("edge_router_logit", "raw")),
            logit_scale=float(kwargs.pop("edge_router_logit_scale", 10.0)),
            select=str(kwargs.pop("edge_router_select", "sigmoid")),
            bias_at_eval=bool(kwargs.pop("edge_router_bias_at_eval", False)),
            bias_schedule=str(kwargs.pop("edge_router_bias_schedule", "const")),
            select_noise=float(kwargs.pop("edge_router_select_noise", 0.0)),
            bias_freeze_after_step=int(kwargs.pop("edge_router_bias_freeze_after_step", 0)),
            gate=str(kwargs.pop("edge_router_gate", "renorm")),
            type_support=int(kwargs.pop("edge_router_type_support", 0)),
            type_support_seed=int(kwargs.pop("edge_router_type_support_seed", 0)),
        )
        self.edge_router_bias_speed = float(kwargs.pop("edge_router_bias_speed", 0.005))
        # what the per-edge (prior_activate) router sees: learned bond-type embedding plus the prior Gram descriptor
        # (production), plus a radial basis of the edge length instead (control), or the embedding alone
        self.edge_router_input = str(kwargs.pop("edge_router_input", "onehot_prior"))
        self.edge_router_rbf = int(kwargs.pop("edge_router_rbf", 16))
        self.edge_router_rbf_rmax = float(kwargs.pop("edge_router_rbf_rmax", 10.0))
        if self.edge_router_input not in ("onehot_prior", "onehot_r", "onehot"):
            raise ValueError("edge_router_input must be onehot_prior, onehot_r or onehot; got %r" % self.edge_router_input)
        if self.edge_router_options["type_support"] > 0 and not self.edge_router_prior_activate:
            raise ValueError("edge_router_type_support needs edge_router_prior_activate=true (per-edge routing)")
        if self.edge_router_input != "onehot_prior" and not self.edge_router_prior_activate:
            raise ValueError("edge_router_input=%r needs edge_router_prior_activate=true (per-edge routing)"
                             % self.edge_router_input)
        if self.edge_router_input == "onehot_r" and (self.edge_router_rbf < 1 or self.edge_router_rbf_rmax <= 0.0):
            raise ValueError("edge_router_input=onehot_r needs edge_router_rbf >= 1 and edge_router_rbf_rmax > 0")
        edge_one_hot_dim = int(edge_router_in_features or kwargs.get("edge_one_hot_dim", 128))
        self.edge_one_hot_dim = edge_one_hot_dim
        self.edge_router_in_features = edge_one_hot_dim
        if not self.edge_router_prior_activate:
            kwargs.setdefault("so2_fusion_mode", "streamed_m_major_fused_p0")
        top_k = kwargs.get("top_k", 1)
        self.edge_router_route_drop_p = kwargs.pop("edge_router_route_drop_p", 0.0)
        self.edge_router_route_drop_scale = kwargs.pop("edge_router_route_drop_scale", "inverted")
        validate_route_drop(self.edge_router_route_drop_p, self.edge_router_route_drop_scale)
        self.edge_router_route_drop_p = float(self.edge_router_route_drop_p)
        if self.edge_router_route_drop_p > 0.0:
            if (not self.edge_router_prior_activate or self.edge_router_top1_mode != "legacy"
                    or top_k is None or top_k < 2 or kwargs.get("num_shared_experts", 1) < 1
                    or kwargs.get("so2_expert_mixing_mode", "pre_activation") != "pre_activation"):
                raise ValueError("edge_router_route_drop_p > 0 requires edge_router_prior_activate=true, "
                                 "pre_activation, top_k >= 2, shared experts and no switch/top1 routing")
        if self.edge_router_top1_mode == "switch" and (top_k != 1 or not self.edge_router_prior_activate):
            raise ValueError("Switch top-1 branch requires top_k=1 and prior_activate=true")
        if self.edge_router_top1_mode == "switch" and kwargs.get("num_shared_experts", 1) != 0:
            raise ValueError("Switch top-1 branch requires num_shared_experts=0")
        prev_so2_env = None
        if self.edge_router_prior_activate:
            # softmax over a single gathered logit is the constant 1, so with
            # top_k=1 the routing coefficient carries no gradient at all: the
            # zero-initialised descriptor columns would stay zero forever and the
            # whole mode would be a silent no-op that still pays for staged SO2.
            if int(top_k) < 2 and self.edge_router_top1_mode != "switch":
                raise ValueError(
                    "edge_router_prior_activate needs top_k >= 2; got %r. With "
                    "top_k=1 the gate is constant and the router receives no "
                    "gradient, so the descriptor could never earn any influence."
                    % (top_k,)
                )
            # post_activation mixing reaches the experts through apply_experts,
            # whose reference backend gathers one [out, in] weight per row -- the
            # very cost activation space exists to avoid, and the weight-space
            # guard does not sit on that path.
            # post_activation_slot runs every top-k slot through the same
            # activation-space route (SO2SlotPostActivationMixer), so it is allowed.
            mixing = kwargs.get("so2_expert_mixing_mode", "pre_activation")
            if mixing == "post_activation_slot" and self.edge_router_top1_mode == "switch":
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_slot' folds the shared expert into every "
                    "slot, which needs top-k coefficients summing to one; the Switch top-1 route "
                    "keeps its retained probability instead."
                )
            if mixing == "post_activation_slot" and self.edge_router_options["gate"] != "renorm":
                raise ValueError(
                    "so2_expert_mixing_mode='post_activation_slot' folds the shared expert into every slot, "
                    "which needs coefficients summing to one (edge_router_gate='renorm'); use "
                    "'post_activation_shared' with edge_router_gate='full_softmax'."
                )
            if mixing not in ("pre_activation", "post_activation_slot", "post_activation_shared"):
                raise ValueError(
                    "edge_router_prior_activate requires so2_expert_mixing_mode='pre_activation', "
                    "'post_activation_slot' or 'post_activation_shared'; "
                    "got %r, which dispatches through apply_experts and would "
                    "materialise one weight per edge." % (mixing,)
                )
            # Activation space needs every expert applied to activations, never a
            # weight mixed per route token.  'staged' and the grouped streaming
            # route reach the m-linears through SO2_m_Linear.forward ->
            # self.fc(x_m, mole_globals); fused_p0 segments its grouped GEMM by
            # expert id and sums the top-k slot outputs before the scatter.  The
            # other SO2CUDA fused routes take the weights via
            # _mix_expert_parameters, which activation space forbids.  fused_p0 is
            # the default, as for the other edge-MoE modes; off CUDA float32 it
            # runs the grouped streaming route.
            requested_route = kwargs.get("so2_fusion_mode")
            if requested_route is None:
                kwargs["so2_fusion_mode"] = "streamed_m_major_fused_p0"
            elif requested_route not in _PRIOR_ACTIVATE_ROUTES:
                # Silently rewriting this used to let a config claim a fused route
                # and train on a different one with nothing in the log.
                raise ValueError(
                    "edge_router_prior_activate is incompatible with "
                    "so2_fusion_mode=%r: that route consumes "
                    "fc._mix_expert_parameters(), which builds one "
                    "[out_features, in_features] weight per route token -- with "
                    "per-edge routing that is one weight per edge. Use one of %s "
                    "(omit the key for the faster default, %r)."
                    % (requested_route, list(_PRIOR_ACTIVATE_ROUTES), "streamed_m_major_fused_p0")
                )
            # The m-linear cuBLAS fusion inside the grouped route is the one part
            # of it that DOES call _mix_expert_parameters, so it has to stay off.
            if os.environ.get("DPTB_SO2_FUSE_M_CUBLAS", "0") not in ("", "0", "false", "False", "FALSE"):
                raise RuntimeError(
                    "edge_router_prior_activate is incompatible with "
                    "DPTB_SO2_FUSE_M_CUBLAS=1: that path fuses the m-linears by "
                    "calling fc._mix_expert_parameters(), which would materialise "
                    "one weight per edge."
                )
            # SO2_Linear lets DPTB_SO2_FUSION_MODE override exactly the value
            # "staged", so the kwarg alone is not enough wherever the deployment
            # exports one.  Suppress it across construction, then verify.
            prev_so2_env = os.environ.pop("DPTB_SO2_FUSION_MODE", None)
        try:
            super().__init__(**kwargs)
        finally:
            if prev_so2_env is not None:
                os.environ["DPTB_SO2_FUSION_MODE"] = prev_so2_env
        if self.edge_router_prior_activate:
            stray = [name for name, mod in self.named_modules()
                     if getattr(mod, "so2_fusion_mode", "staged") not in _PRIOR_ACTIVATE_ROUTES]
            if stray:
                raise RuntimeError(
                    "edge_router_prior_activate needs every SO2 layer on a route "
                    "that reaches MOLELinear.forward (one of %s) so the per-edge "
                    "coefficients are applied to activations, but %d are not: %s"
                    % (list(_PRIOR_ACTIVATE_ROUTES), len(stray), stray[:4])
                )

        self._prior_chunks = []
        self._prior_source_dim = 0
        self.edge_router_prior_dim = 0
        if self.edge_router_prior_activate or structure_strategy is not None:
            prior_irreps, sort_index = _sorted_irrep_coordinate_index(self.idp)
            self.register_buffer("_prior_sort_index", sort_index, persistent=False)
            if self.edge_router_prior_cg:
                from .prior_common import _build_uureal_cg_change_of_basis
                # Independent of legacy h0_ao_cg: this flag fixes routing only.
                # No parameters, RNG draws or state_dict keys are added.
                self.register_buffer(
                    "_prior_cg_change_of_basis",
                    _build_uureal_cg_change_of_basis(self.idp, dtype=self.dtype, device=self.device),
                    persistent=False,
                )
            offset = 0
            desc_dim = 0
            for mul, ir in prior_irreps:
                mul, ir_dim = int(mul), int(ir.dim)
                width = mul * ir_dim
                self._prior_chunks.append((offset, offset + width, mul, ir_dim))
                offset += width
                # l=0 is already invariant: keep every channel, signed.
                # l>0 needs a quadratic invariant, and the complete
                # Clebsch-Gordan-free one is the whole Gram matrix of the block,
                # not just its diagonal.  Same (l, parity) only -- a cross-parity
                # contraction is a pseudoscalar and would break inversion
                # equivariance.
                desc_dim += mul if ir_dim == 1 else mul * (mul + 1) // 2
            self._prior_source_dim = offset
            self.edge_router_prior_dim = desc_dim
            self.edge_router_in_features = edge_one_hot_dim + desc_dim
            if self.edge_router_input == "onehot_r":
                self.edge_router_in_features = edge_one_hot_dim + self.edge_router_rbf
                centers = torch.linspace(0.0, self.edge_router_rbf_rmax, self.edge_router_rbf,
                                         dtype=self.dtype, device=self.device)
                self.register_buffer("_router_rbf_centers", centers, persistent=False)
                self._router_rbf_width = self.edge_router_rbf_rmax / max(self.edge_router_rbf - 1, 1)
            elif self.edge_router_input == "onehot":
                self.edge_router_in_features = edge_one_hot_dim
            mean = torch.zeros(desc_dim, dtype=self.dtype, device=self.device)
            std = torch.ones(desc_dim, dtype=self.dtype, device=self.device)
            if self.edge_router_prior_stats:
                blob = torch.load(self.edge_router_prior_stats, map_location="cpu")
                loaded_mean = blob["mean"].reshape(-1)
                loaded_std = blob["std"].reshape(-1)
                if loaded_mean.numel() != desc_dim or loaded_std.numel() != desc_dim:
                    raise ValueError(
                        "edge_router_prior_stats has "
                        f"{loaded_mean.numel()} channels, descriptor has {desc_dim}."
                    )
                mean = loaded_mean.to(dtype=self.dtype, device=self.device)
                std = loaded_std.to(dtype=self.dtype, device=self.device)
            # Frozen on purpose: standardisation is a property of the dataset,
            # not something the routing loss gets to move.  Non-persistent so a
            # resume takes the stats named by the current config instead of
            # silently restoring whatever the checkpoint was written with.
            self.register_buffer("_prior_mean", mean, persistent=False)
            self.register_buffer("_prior_std", std, persistent=False)

        if structure_strategy is not None:
            structure_strategy.configure(self)
            return

        router_type, router_kwargs = MOLERouterV3, dict(mixing_temperature=self.edge_router_temperature,
                                                         **self.edge_router_options)
        if self.edge_router_top1_mode == "switch":
            from dptb.nn.top1_prior import Top1PriorRouter
            if self.edge_router_options != dict(logit_kind="raw", logit_scale=10.0, select="sigmoid",
                                                bias_at_eval=False, bias_schedule="const", select_noise=0.0,
                                                bias_freeze_after_step=0, gate="renorm", type_support=0,
                                                type_support_seed=0):
                raise ValueError("edge_router_logit/select/bias_*/select_noise apply to the top-k router only, "
                                 "not to edge_router_top1_mode=switch")
            router_type, router_kwargs = Top1PriorRouter, {}
        self.router = router_type(
            in_features=self.edge_router_in_features,
            num_experts=self.num_experts,
            top_k=top_k,
            aux_loss_free=self.edge_router_top1_mode != "switch",
            bias_update_speed=0.0 if self.edge_router_top1_mode == "switch" else self.edge_router_bias_speed,
            **router_kwargs,
        )
        if self.edge_router_prior_dim and self.edge_router_input == "onehot_r":
            # radial columns start at zero, like the descriptor columns of the production router
            with torch.no_grad():
                self.router.net[0].weight[:, edge_one_hot_dim:].zero_()
        elif self.edge_router_prior_dim and self.edge_router_input == "onehot_prior":
            # The descriptor contributes exactly zero at step 0, so it can earn
            # influence without injecting a scale shock into the gate, and
            # gradients still flow into these columns.  Note this is NOT a
            # bit-identical baseline: widening the router input also rescales
            # nn.Linear's fan_in initialisation, so the surviving one-hot columns
            # are redrawn and narrower than in the 128-wide model.
            with torch.no_grad():
                self.router.net[0].weight[:, edge_one_hot_dim:].zero_()

        from .unitb_router_inputs import router_input_strategy
        self._router_input_strategy = router_input_strategy(
            self.edge_router_input, per_edge=self.edge_router_prior_activate
        )

    def _edge_router_input(self, data, bond_type, active_edges, active_edge_one_hot, edge_vector):
        return self._router_input_strategy(self, data, bond_type, active_edges,
                                           active_edge_one_hot, edge_vector)

    def _raw_prior_source(
        self,
        data: AtomicDataDict.Type,
        bond_type: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """The frozen prior, masked, optionally CG-coupled, then sorted.

        Read straight from the data dict, with no fallback: H0InitLayer's
        fallback chain can reach the target Hamiltonian, and routing on the
        target is label leakage.  Missing key is a hard error.
        """
        key = getattr(getattr(self, "init_layer", None), "h0_edge_key", None)
        if key is None:
            key = getattr(self, "h0_edge_key", None)
        if key is None or key not in data:
            raise KeyError(
                "edge_router_prior_activate needs the frozen edge prior at "
                f"data[{key!r}]; it exists only on lem_moe_v3_edge_h0 with a "
                "dataset that carries edge_h0."
            )
        source = data[key].to(dtype=self.dtype)
        mask = self.idp.mask_to_erme.to(source.device)[bond_type.flatten()]
        source = source * mask.to(dtype=source.dtype)
        if self.edge_router_prior_cg:
            from .prior_common import _h0_is_coupled_rme
            if not _h0_is_coupled_rme(data):
                source = torch.einsum(
                    "kc,nc->nk", self._prior_cg_change_of_basis.to(source), source
                )
        source = source.index_select(1, self._prior_sort_index.to(source.device))
        return source.index_select(0, active_edges)


    def _gram_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation-invariant descriptor: l=0 signed, l>0 the full Gram matrix.

        signed-log is applied per channel purely for conditioning; it is
        strictly monotone, so no ordering information is lost.
        """
        if x.shape[-1] != self._prior_source_dim:
            raise ValueError(
                "edge prior descriptor expects a source of width "
                f"{self._prior_source_dim}, got {x.shape[-1]}."
            )
        parts = []
        for start, stop, mul, ir_dim in self._prior_chunks:
            block = x[:, start:stop].reshape(x.shape[0], mul, ir_dim)
            if ir_dim == 1:
                parts.append(block.reshape(x.shape[0], mul))
                continue
            gram = torch.einsum("nam,nbm->nab", block, block)
            iu = torch.triu_indices(mul, mul, offset=0, device=x.device)
            parts.append(gram[:, iu[0], iu[1]])
        desc = torch.cat(parts, dim=-1)
        desc = torch.sign(desc) * torch.log1p(desc.abs()).clamp(max=20.0)
        return (desc - self._prior_mean) / self._prior_std.clamp_min(1e-6)


    def _make_edge_moe_globals(self, active_edge_one_hot, active_bond_type, *, data=None, active_edges=None):
        # Exact legacy fast path: no extra RNG draws, tensor operations or state.
        if getattr(self, "edge_router_route_drop_p", 0.0) == 0.0 or not self.training:
            return UniTBRouter._make_undropped_edge_moe_globals(self, active_edge_one_hot, active_bond_type)
        if data is None or active_edges is None:
            raise ValueError("structure route dropout needs batch data and active_edges")
        keep, edge_keep = self._sample_routes(data, self.edge_router_route_drop_p)
        active_keep = edge_keep.index_select(0, active_edges)
        route, monitor, cv, count = self._make_undropped_edge_moe_globals(
            active_edge_one_hot, active_bond_type, regularizer_weights=active_keep)
        route = apply_structure_routes(route, active_keep, self.edge_router_route_drop_p,
                                       self.edge_router_route_drop_scale, self.router.top_k)
        # Dispatch and the public last_topk view agree; no masks enter state_dict.
        self.router._last_topk_indices = route.topk_indices
        self.router._last_topk_values = route.topk_values
        self.last_route_drop_mask = ~keep
        self.last_route_drop_stats = record_route_drop(
            data, keep, edge_keep, active_keep, step=self.router.opt_step)
        return route, monitor, cv, count


    def _make_undropped_edge_moe_globals(
        self,
        active_edge_one_hot: torch.Tensor,
        active_bond_type: torch.Tensor,
        regularizer_weights=None,
    ):
        num_active_edges = int(active_edge_one_hot.shape[0])
        if active_edge_one_hot.shape[-1] != self.edge_router_in_features:
            raise ValueError(
                "edge_router_in_features mismatch: router was built with "
                f"{self.edge_router_in_features}, but active edge input has "
                f"{active_edge_one_hot.shape[-1]}."
            )
        full_soft = (num_active_edges == 0 and self.edge_router_prior_activate and
                     self.edge_router_top1_mode != "switch" and
                     (self.router.top_k is None or self.router.top_k >= self.num_experts))
        if num_active_edges == 0 and self.edge_router_prior_activate and full_soft:
            # Use the router even for an empty full-soft batch: this clears stale
            # routes and provides the same [N, E] metadata and logging contract.
            coeffs, monitor, cv = self.router(active_edge_one_hot, bond_type=active_bond_type)
            indices, values = self.router.last_topk()
            return MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=indices,
                               topk_values=values, activation_space=True,
                               coefficients_sum_to_one=True), monitor, cv, coeffs.new_zeros(())
        if num_active_edges == 0:
            coeffs = active_edge_one_hot.new_zeros((0, self.num_experts))
            zero = active_edge_one_hot.new_zeros(())
            if regularizer_weights is not None:
                self.router.last_router_z_loss = zero
            return MOLEGlobals(coefficients=coeffs, sizes=None), zero, zero, zero

        if self.edge_router_top1_mode == "switch":
            route, monitor, cv = self.router(active_edge_one_hot)
            return route, monitor, cv, active_edge_one_hot.new_tensor(float(num_active_edges))

        if self.edge_router_prior_activate:
            # One routing decision per edge.  No dedup: the whole point of this
            # mode is that pooling by any function of (bond type, r) would make
            # the coefficients a function of (bond type, r) too, no matter what
            # the descriptor carries.
            if self.edge_router_options["type_support"] > 0:
                coeffs, monitor_val, expert_load_cv = self.router(
                    active_edge_one_hot, bond_type=active_bond_type, regularizer_weights=regularizer_weights)
            else:
                coeffs, monitor_val, expert_load_cv = self.router(
                    active_edge_one_hot, regularizer_weights=regularizer_weights)
            topk_indices, topk_values = self.router.last_topk()
            if topk_indices is None or topk_values is None:
                raise RuntimeError(
                    "edge_router_prior_activate requires router dispatch metadata "
                    "for every selected expert."
                )
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))
            return (
                MOLEGlobals(
                    coefficients=coeffs,
                    sizes=None,
                    topk_indices=topk_indices,
                    topk_values=topk_values,
                    activation_space=True,
                    # MOLERouterV3 with gate='renorm' normalises the selected top-k
                    # logits with a softmax, so these sum to 1 exactly and the shared
                    # expert can be folded into the routed weights; gate='full_softmax'
                    # keeps the router's probability mass (< 1), so it may not.
                    coefficients_sum_to_one=bool(getattr(self.router, "coefficients_sum_to_one", True)),
                ),
                monitor_val,
                expert_load_cv,
                num_route_tokens,
            )

        if self.edge_router_unique_types:
            unique_bond_type, inverse, counts = torch.unique(
                active_bond_type,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            unique_inputs = active_edge_one_hot.new_zeros(
                unique_bond_type.shape[0],
                active_edge_one_hot.shape[-1],
            )
            unique_inputs.index_add_(0, inverse, active_edge_one_hot)
            unique_inputs = unique_inputs / counts.to(dtype=active_edge_one_hot.dtype).unsqueeze(-1).clamp_min_(1)
            coeffs, monitor_val, expert_load_cv = self.router(
                unique_inputs,
                sizes=counts.to(dtype=active_edge_one_hot.dtype),
            )
            topk_indices, topk_values = self.router.last_topk()
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))
            use_compact_dispatch = (
                self.edge_moe_compact_dispatch
                and num_active_edges >= self.edge_moe_compact_min_edges
            )
            if use_compact_dispatch:
                return (
                    MOLEGlobals(
                        coefficients=coeffs,
                        sizes=None,
                        graph_index=inverse,
                        topk_indices=topk_indices,
                        topk_values=topk_values,
                    ),
                    monitor_val,
                    expert_load_cv,
                    num_route_tokens,
                )
            coeffs = coeffs.index_select(0, inverse)
            if topk_indices is not None and topk_values is not None:
                topk_indices = topk_indices.index_select(0, inverse)
                topk_values = topk_values.index_select(0, inverse)
        else:
            coeffs, monitor_val, expert_load_cv = self.router(active_edge_one_hot)
            topk_indices, topk_values = self.router.last_topk()
            num_route_tokens = coeffs.new_tensor(float(coeffs.shape[0]))

        # One coefficient row per active edge: row i routes edge i.  Without the
        # graph_index every MOLELinear backend treats all rows as one route and
        # applies row 0's mix to every edge.
        return (
            MOLEGlobals(
                coefficients=coeffs,
                sizes=None,
                graph_index=torch.arange(num_active_edges, device=coeffs.device),
                topk_indices=topk_indices,
                topk_values=topk_values,
            ),
            monitor_val,
            expert_load_cv,
            num_route_tokens,
        )


