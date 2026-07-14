from __future__ import annotations

"""Pixel MeanFlow objective for DeePTB Hamiltonian flow training.

Split out of :mod:`dptb.nnops.flow`.  ``HamiltonianPixelMeanFlow`` subclasses
``HamiltonianCFM`` (imported from that module, whose ``log`` it also reuses so
warnings stay under the "dptb.nnops.flow" logger) and inherits its
prior/masking machinery; only the pixel-meanflow overrides live here.
"""

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from dptb.configuration import canonicalize_flow_options
from dptb.data import AtomicDataDict
from dptb.nnops.flow import HamiltonianCFM, log
from dptb.nnops.flow_context import PixelMFContext
from dptb.nnops.layout import project_uureal_to_like


class HamiltonianPixelMeanFlow(HamiltonianCFM):
    """Pixel MeanFlow objective for residual Hamiltonian endpoint predictors.

    The model still predicts the clean endpoint residual ``x``.  The pMF average
    velocity is induced by ``u=(z_t-x_theta)/t`` for
    ``z_t=(1-t)x+t eps`` and trained through
    ``u+(t-r) stopgrad(d u/dt)`` against the path velocity ``eps-x``.
    """

    model_in_loss = True

    def __init__(
        self,
        options: Optional[Dict[str, Any]],
        *,
        idp: Any = None,
        dtype: Any = torch.float32,
        device: Any = torch.device("cpu"),
    ) -> None:
        options = canonicalize_flow_options(options)
        super().__init__(options, idp=idp, dtype=dtype, device=device)
        mf = dict(options.get("meanflow", {}) or {})
        profile = str(mf.get("profile", "conservative")).lower()
        if profile not in {"conservative", "aggressive"}:
            raise ValueError("pixel meanflow profile must be 'conservative' or 'aggressive'.")
        aggressive = profile == "aggressive"

        self.meanflow_profile = profile
        self.meanflow_time_sampling = str(mf.get("time_sampling", "logit_normal")).lower()
        self.meanflow_p_mean = float(mf.get("p_mean", -0.4))
        self.meanflow_p_std = float(mf.get("p_std", 1.0))
        self.meanflow_data_proportion = float(mf.get("data_proportion", 0.50))
        self.meanflow_tr_uniform_prob = float(mf.get("tr_uniform_prob", 0.10))
        self.meanflow_min_t = float(mf.get("min_t", 0.05))
        self.meanflow_fd_eps = float(mf.get("fd_eps", 1.0e-3))
        self.meanflow_du_dt_backend = str(
            mf.get("du_dt_backend", "finite_difference")
        ).lower().replace("-", "_")
        if self.meanflow_du_dt_backend in {"fd", "finite_diff"}:
            self.meanflow_du_dt_backend = "finite_difference"
        if self.meanflow_du_dt_backend not in {"finite_difference", "jvp"}:
            raise ValueError(
                "pixel_meanflow.du_dt_backend must be 'finite_difference' or 'jvp', "
                f"got {self.meanflow_du_dt_backend!r}."
            )
        # jvp failures (forward-mode-unsupported ops, DDP wrappers, custom
        # kernels) fall back to finite_difference for the rest of the run
        # unless the user makes them fatal.
        self.meanflow_jvp_fallback = bool(mf.get("jvp_fallback", True))
        # Memory-efficient jvp: compute the primal (training signal) in a normal
        # grad forward and the detached du/dt tangent in a separate no_grad
        # forward-mode pass, instead of one fused dual forward. The fused pass
        # stores every activation as a primal+tangent dual (~2.2x peak); the
        # split pass keeps only the primal reverse graph (~1x, like
        # finite_difference) because forward-mode tangents free layer-by-layer
        # under no_grad. Costs one extra model call. Default on: production is
        # memory-bound (bs96 must fit the card).
        self.meanflow_jvp_memory_efficient = bool(
            mf.get("jvp_memory_efficient", True)
        )
        # Safety switches for the jvp path (review findings 1 & 2).
        # A None forward tangent for an active component is almost never a valid
        # du/dt: it means a detach / no_grad island / forward-AD-unsupported op
        # swallowed the dual. Zeroing it would silently bias the MeanFlow
        # objective while the canary still reports jvp live, so by default we
        # raise (the loss_with_model try/except then degrades to
        # finite_difference if jvp_fallback=true). Synthetic constant-output
        # test models legitimately have no tangent -> opt out there.
        self.meanflow_jvp_require_tangents = bool(mf.get("jvp_require_tangents", True))
        # In split mode the training primal and the du/dt tangent come from two
        # separate forwards; if they disagree (nondeterministic routing, stateful
        # cache) dx/dt is evaluated at the wrong point. Cheap allclose on the
        # endpoint guards it; loose tol tolerates GPU-atomic scatter noise. A
        # mismatch raises -> finite_difference fallback rather than silent-wrong.
        self.meanflow_jvp_split_check_primal = bool(
            mf.get("jvp_split_check_primal", True)
        )
        self.meanflow_jvp_split_check_rtol = float(mf.get("jvp_split_check_rtol", 5.0e-4))
        self.meanflow_jvp_split_check_atol = float(mf.get("jvp_split_check_atol", 5.0e-5))
        self._meanflow_jvp_disabled = False
        self.meanflow_norm_eps = float(mf.get("norm_eps", 0.01))
        self.meanflow_norm_p = float(mf.get("norm_p", 1.0 if aggressive else 0.0))
        self.meanflow_aux_endpoint_weight = float(mf.get("aux_endpoint_weight", 0.05))
        self.meanflow_aux_boundary_v_weight = float(
            mf.get("aux_boundary_v_weight", 0.10 if aggressive else 0.0)
        )
        self.meanflow_objective = str(
            mf.get("objective", "finite_difference")
        ).lower().replace("-", "_")
        if self.meanflow_objective in {"fd", "finite_diff", "jvp"}:
            self.meanflow_objective = "finite_difference"
        if self.meanflow_objective in {"kaist", "semigroup_meanflow", "semigroup_mf"}:
            self.meanflow_objective = "semigroup"
        if self.meanflow_objective not in {"finite_difference", "semigroup", "hybrid"}:
            raise ValueError(
                "pixel_meanflow.meanflow.objective must be 'finite_difference', "
                "'semigroup', or 'hybrid', "
                f"got {self.meanflow_objective!r}."
            )
        self.meanflow_semigroup_weight = float(
            mf.get(
                "semigroup_weight",
                1.0 if self.meanflow_objective in {"semigroup", "hybrid"} else 0.0,
            )
        )
        self.meanflow_semigroup_endpoint_weight = float(
            mf.get(
                "semigroup_endpoint_weight",
                1.0 if self.meanflow_objective == "semigroup" else self.meanflow_aux_endpoint_weight,
            )
        )
        self.meanflow_jvp_tangent = str(mf.get("jvp_tangent", "boundary")).lower()
        if self.meanflow_jvp_tangent not in {"path", "boundary"}:
            raise ValueError("pixel_meanflow.jvp_tangent must be 'path' or 'boundary'.")
        self.meanflow_sample_final_forward = bool(mf.get("sample_final_forward", True))

        self.flow_time_r_key = str(options.get("flow_time_r_key", "flow_time_r"))
        self.flow_time_t_key = str(options.get("flow_time_t_key", "flow_time_t"))
        self.flow_time_h_key = str(options.get("flow_time_h_key", "flow_time_h"))
        # pMF computes its optimization loss inside loss_with_model, but legacy
        # train/validation loss keys are a cross-route endpoint contract. Keep
        # compatible endpoint logging forced on; user flags may not opt out.
        self.log_train_compatible_loss = True
        self.log_validation_compatible_loss = True
        self.compatible_loss_to_legacy_keys = True

        # A completely disabled validation path returns literal zero from
        # Trainer.validation(), which is indistinguishable from a perfect model
        # in the logs. Keep at least the pMF random-time objective on.
        if self.enabled and not any(
            (
                self.log_validation_random_t_loss,
                self.log_validation_t0_loss,
                self.log_validation_flow_euler_loss,
                self.log_validation_compatible_loss,
            )
        ):
            log.warning(
                "Pixel MeanFlow validation has all validation metrics disabled; "
                "enabling log_validation_random_t_loss to avoid zero-valued validation logs."
            )
            self.log_validation_random_t_loss = True

        # With a sinusoidal time embedding, the finite-difference time step must
        # stay small relative to the fastest embedding frequency, or du/dt
        # measures embedding oscillation instead of the path derivative.
        approx_phase_step = float(mf.get("time_embedding_max_positions", 2000.0)) * self.meanflow_fd_eps
        if self.enabled and approx_phase_step > 2.0:
            log.warning(
                "Pixel MeanFlow finite difference uses fd_eps=%.3g with sinusoidal "
                "max_positions~%.3g (phase step ~%.3g rad). This can dominate du/dt; "
                "consider fd_eps<=5e-4 or a smaller flow_time_max_positions ablation.",
                self.meanflow_fd_eps,
                float(mf.get("time_embedding_max_positions", 2000.0)),
                approx_phase_step,
            )

        if self.enabled:
            log.info(
                "Pixel MeanFlow enabled: profile=%s objective=%s sampling=%s min_t=%.3g "
                "data_prop=%.3g du_dt=%s jvp_tangent=%s norm_p=%.3g "
                "aux_x=%.3g aux_v=%.3g semigroup_w=%.3g semigroup_x=%.3g",
                self.meanflow_profile,
                self.meanflow_objective,
                self.meanflow_time_sampling,
                self.meanflow_min_t,
                self.meanflow_data_proportion,
                self.meanflow_du_dt_backend,
                self.meanflow_jvp_tangent,
                self.meanflow_norm_p,
                self.meanflow_aux_endpoint_weight,
                self.meanflow_aux_boundary_v_weight,
                self.meanflow_semigroup_weight,
                self.meanflow_semigroup_endpoint_weight,
            )

    def _sample_time_base(
        self,
        num_graphs: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.meanflow_time_sampling == "uniform":
            return torch.rand(num_graphs, device=device, dtype=dtype)
        if self.meanflow_time_sampling == "logit_normal":
            raw = torch.randn(num_graphs, device=device, dtype=dtype)
            return torch.sigmoid(raw * self.meanflow_p_std + self.meanflow_p_mean)
        raise ValueError(f"Unsupported pixel meanflow time sampling {self.meanflow_time_sampling!r}.")

    def _sample_rt(
        self,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self._sample_time_base(num_graphs, device=device, dtype=dtype)
        r = self._sample_time_base(num_graphs, device=device, dtype=dtype)
        if self.meanflow_tr_uniform_prob > 0.0:
            use_uniform = torch.rand(num_graphs, device=device) < self.meanflow_tr_uniform_prob
            t = torch.where(use_uniform, torch.rand(num_graphs, device=device, dtype=dtype), t)
            r = torch.where(use_uniform, torch.rand(num_graphs, device=device, dtype=dtype), r)
        fm_mask = torch.rand(num_graphs, device=device) < self.meanflow_data_proportion
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        t = t.clamp(min=self.meanflow_min_t, max=1.0)
        r = torch.minimum(r.clamp(min=0.0, max=1.0), t)
        r = torch.where(fm_mask, t, r)
        return r, t, fm_mask

    def _write_times(
        self,
        data: AtomicDataDict.Type,
        r: torch.Tensor,
        t: torch.Tensor,
        *,
        detach: bool = True,
    ) -> None:
        # detach=False is required by the jvp du/dt backend: forward-mode
        # tangents on t must reach the model's time conditioning, and
        # .detach() strips the dual part.
        tt = t.detach() if detach else t
        rr = r.detach() if detach else r
        hh = tt - rr
        data[self.flow_time_key] = tt
        data[self.flow_time_t_key] = tt
        data[self.flow_time_r_key] = rr
        data[self.flow_time_h_key] = hh
        data["t"] = tt
        data["r"] = rr
        data["meanflow_h"] = hh

    @staticmethod
    def _view_time(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return t.reshape((-1,) + (1,) * (like.ndim - 1)).clamp_min(1.0e-8)

    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[AtomicDataDict.Type, AtomicDataDict.Type, PixelMFContext]:
        if not self.enabled:
            raise RuntimeError("HamiltonianPixelMeanFlow.prepare_batch called while disabled")

        data = data.copy()
        ref_data = ref_data.copy()
        node_target = ref_data.get(self.node_target_key, None)
        edge_target = ref_data.get(self.edge_target_key, None)
        if node_target is None and edge_target is None:
            raise KeyError(
                "Pixel MeanFlow requires node and/or edge Hamiltonian targets in ref_data; "
                f"looked for `{self.node_target_key}` and `{self.edge_target_key}`."
            )

        like = node_target if node_target is not None else edge_target
        device = like.device
        dtype = like.dtype if torch.is_floating_point(like) else self.dtype
        num_graphs = self._num_graphs(data)
        if r is None or t is None:
            r, t, fm_mask = self._sample_rt(num_graphs=num_graphs, device=device, dtype=dtype)
        else:
            r = self._normalize_t(r, num_graphs=num_graphs, device=device, dtype=dtype)
            t = self._normalize_t(t, num_graphs=num_graphs, device=device, dtype=dtype)
            t, r = torch.maximum(t, r), torch.minimum(t, r)
            t = t.clamp(min=self.meanflow_min_t, max=1.0)
            r = torch.minimum(r.clamp(min=0.0, max=1.0), t)
            fm_mask = torch.isclose(r, t)

        node_t, edge_t = self._expand_graph_times(
            data,
            t,
            node_count=None if node_target is None else node_target.shape[0],
            edge_count=None if edge_target is None else edge_target.shape[0],
        )
        node_r, edge_r = self._expand_graph_times(
            data,
            r,
            node_count=None if node_target is None else node_target.shape[0],
            edge_count=None if edge_target is None else edge_target.shape[0],
        )

        node_base = edge_base = node_clean = edge_clean = None
        node_state = edge_state = node_prior = edge_prior = None
        # One shared Haar-DM candidate for node+edge; None for every other prior.
        haar_candidate_idx = self._resolve_haar_candidate_idx(
            data, node_like=node_target, edge_like=edge_target
        )
        if node_target is not None:
            node_target = node_target.to(device=device, dtype=dtype)
            node_base = self._base_like(data, node_target, self.node_h0_key, "node")
            node_clean = node_target - node_base if self.mode == "residual" else node_target
            node_prior = self._prior_like(
                node_clean,
                self.node_sigma,
                data=data,
                label="node",
                base=node_base,
                candidate_idx=haar_candidate_idx,
            )
            node_t_view = node_t.reshape((-1,) + (1,) * (node_clean.ndim - 1))
            node_state = (1.0 - node_t_view) * node_clean + node_t_view * node_prior
            current = node_base + node_state if self.mode == "residual" else node_state
            if self.detach_interpolated_h0:
                current = current.detach()
            data[self.node_h0_key] = current
            if self.overwrite_feature_keys:
                data[self.node_target_key] = current
        if edge_target is not None:
            edge_target = edge_target.to(device=device, dtype=dtype)
            edge_base = self._base_like(data, edge_target, self.edge_h0_key, "edge")
            edge_clean = edge_target - edge_base if self.mode == "residual" else edge_target
            edge_prior = self._prior_like(
                edge_clean,
                self.edge_sigma,
                data=data,
                label="edge",
                base=edge_base,
                candidate_idx=haar_candidate_idx,
            )
            edge_t_view = edge_t.reshape((-1,) + (1,) * (edge_clean.ndim - 1))
            edge_state = (1.0 - edge_t_view) * edge_clean + edge_t_view * edge_prior
            current = edge_base + edge_state if self.mode == "residual" else edge_state
            if self.detach_interpolated_h0:
                current = current.detach()
            data[self.edge_h0_key] = current
            if self.overwrite_feature_keys:
                data[self.edge_target_key] = current

        self._write_times(data, r, t)
        self._write_times(ref_data, r, t)
        return data, ref_data, PixelMFContext(
            r=r,
            t=t,
            fm_mask=fm_mask,
            node_r=node_r,
            node_t=node_t,
            edge_r=edge_r,
            edge_t=edge_t,
            node_base=node_base,
            edge_base=edge_base,
            node_clean=node_clean,
            edge_clean=edge_clean,
            node_state=node_state,
            edge_state=edge_state,
            node_prior=node_prior,
            edge_prior=edge_prior,
        )

    def _predict_clean(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        ctx: PixelMFContext,
        node_state: Optional[torch.Tensor],
        edge_state: Optional[torch.Tensor],
        *,
        r: torch.Tensor,
        t: torch.Tensor,
        detach_times: bool = True,
    ) -> Tuple[AtomicDataDict.Type, Optional[torch.Tensor], Optional[torch.Tensor]]:
        model_data = data.copy()
        if node_state is not None:
            node_current = ctx.node_base + node_state if self.mode == "residual" else node_state
            model_data[self.node_h0_key] = node_current
            if self.overwrite_feature_keys:
                model_data[self.node_target_key] = node_current
        if edge_state is not None:
            edge_current = ctx.edge_base + edge_state if self.mode == "residual" else edge_state
            model_data[self.edge_h0_key] = edge_current
            if self.overwrite_feature_keys:
                model_data[self.edge_target_key] = edge_current
        self._write_times(model_data, r, t, detach=detach_times)
        pred = model(model_data)
        node_x = None
        if ctx.node_clean is not None and self.node_target_key in pred:
            node_pred = pred[self.node_target_key]
            node_pred, _raw_mask = project_uureal_to_like(self.idp, node_pred, ctx.node_clean)
            node_x = node_pred - ctx.node_base if self.mode == "residual" else node_pred
        edge_x = None
        if ctx.edge_clean is not None and self.edge_target_key in pred:
            edge_pred = pred[self.edge_target_key]
            edge_pred, _raw_mask = project_uureal_to_like(self.idp, edge_pred, ctx.edge_clean)
            edge_x = edge_pred - ctx.edge_base if self.mode == "residual" else edge_pred
        return pred, node_x, edge_x

    @staticmethod
    def _adaptive_metric_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str,
        *,
        norm_p: float = 0.0,
        norm_eps: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_f = mask.to(device=diff.device, dtype=diff.dtype)
        if mask_f.shape != diff.shape:
            mask_f = mask_f.expand_as(diff)
        reduce_dims = tuple(range(1, diff.ndim))
        count = mask_f.sum(dim=reduce_dims).clamp_min(1.0)
        sq = (diff.square() * mask_f).sum(dim=reduce_dims) / count
        ab = (diff.abs() * mask_f).sum(dim=reduce_dims) / count
        if loss_type == "l1_rmse":
            per_item = 0.5 * (ab + torch.sqrt(sq + 1e-12))
        else:
            per_item = sq
        if norm_p != 0.0:
            per_item = per_item / (per_item.detach() + norm_eps).pow(norm_p)
        return per_item.mean(), sq.mean(), ab.mean()

    def _reverse_meanflow_step(
        self,
        state_z: torch.Tensor,
        pred_x: torch.Tensor,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
    ) -> torch.Tensor:
        h_view = (start_time - end_time).reshape((-1,) + (1,) * (state_z.ndim - 1))
        u = (state_z - pred_x) / self._view_time(start_time, state_z)
        return state_z - h_view * u

    def _component_semigroup_loss(
        self,
        *,
        diff_prefix: str,
        pred_x: torch.Tensor,
        clean: torch.Tensor,
        state_z: torch.Tensor,
        state_two_step: torch.Tensor,
        comp_r: torch.Tensor,
        comp_t: torch.Tensor,
        mask: torch.Tensor,
        weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state_one_step = self._reverse_meanflow_step(state_z, pred_x, comp_t, comp_r)
        semigroup_loss, semigroup_mse, semigroup_mae = self._adaptive_metric_stats(
            state_one_step - state_two_step.detach(),
            mask,
            self.loss_type,
            norm_p=self.meanflow_norm_p,
            norm_eps=self.meanflow_norm_eps,
        )
        endpoint_loss, endpoint_mse, endpoint_mae = self._adaptive_metric_stats(
            pred_x - clean,
            mask,
            self.loss_type,
        )
        total = weight * (
            self.meanflow_semigroup_weight * semigroup_loss
            + self.meanflow_semigroup_endpoint_weight * endpoint_loss
        )
        state = {
            f"{diff_prefix}_semigroup_loss": semigroup_loss.detach(),
            f"{diff_prefix}_semigroup_mse": semigroup_mse.detach(),
            f"{diff_prefix}_semigroup_mae": semigroup_mae.detach(),
            f"{diff_prefix}_endpoint_loss": endpoint_loss.detach(),
            f"{diff_prefix}_endpoint_mse": endpoint_mse.detach(),
            f"{diff_prefix}_endpoint_mae": endpoint_mae.detach(),
        }
        return total, state

    def _semigroup_loss_with_model(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        ctx: PixelMFContext,
        *,
        prefix: str,
        node_x: Optional[torch.Tensor] = None,
        edge_x: Optional[torch.Tensor] = None,
        primary_aliases: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        explicit_model_calls = 0
        needs_main = (
            (ctx.node_clean is not None and node_x is None)
            or (ctx.edge_clean is not None and edge_x is None)
        )
        if needs_main:
            _, fetched_node_x, fetched_edge_x = self._predict_clean(
                model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.r, t=ctx.t
            )
            explicit_model_calls += 1
            if node_x is None:
                node_x = fetched_node_x
            if edge_x is None:
                edge_x = fetched_edge_x

        split_t = 0.5 * (ctx.r + ctx.t)
        split_t = torch.minimum(
            torch.maximum(split_t, ctx.t.new_full(ctx.t.shape, self.meanflow_min_t)),
            ctx.t,
        )
        node_split_t, edge_split_t = self._expand_graph_times(
            data,
            split_t,
            node_count=None if ctx.node_state is None else ctx.node_state.shape[0],
            edge_count=None if ctx.edge_state is None else ctx.edge_state.shape[0],
        )

        with torch.no_grad():
            _, node_x_to_split, edge_x_to_split = self._predict_clean(
                model,
                data,
                ctx,
                ctx.node_state,
                ctx.edge_state,
                r=split_t,
                t=ctx.t,
            )
            explicit_model_calls += 1

            node_state_split = edge_state_split = None
            if ctx.node_state is not None and node_x_to_split is not None:
                node_state_split = self._reverse_meanflow_step(
                    ctx.node_state,
                    node_x_to_split,
                    ctx.node_t,
                    node_split_t,
                )
            if ctx.edge_state is not None and edge_x_to_split is not None:
                edge_state_split = self._reverse_meanflow_step(
                    ctx.edge_state,
                    edge_x_to_split,
                    ctx.edge_t,
                    edge_split_t,
                )

            _, node_x_to_r, edge_x_to_r = self._predict_clean(
                model,
                data,
                ctx,
                node_state_split,
                edge_state_split,
                r=ctx.r,
                t=split_t,
            )
            explicit_model_calls += 1

            node_state_two_step = edge_state_two_step = None
            if node_state_split is not None and node_x_to_r is not None:
                node_state_two_step = self._reverse_meanflow_step(
                    node_state_split,
                    node_x_to_r,
                    node_split_t,
                    ctx.node_r,
                )
            if edge_state_split is not None and edge_x_to_r is not None:
                edge_state_two_step = self._reverse_meanflow_step(
                    edge_state_split,
                    edge_x_to_r,
                    edge_split_t,
                    ctx.edge_r,
                )

        total = None
        state: Dict[str, torch.Tensor] = {
            f"{prefix}_flow_objective_semigroup": ctx.t.new_tensor(1.0),
            f"{prefix}_flow_semigroup_split_t": split_t.detach().mean(),
            f"{prefix}_flow_semigroup_weight": ctx.t.new_tensor(
                float(self.meanflow_semigroup_weight)
            ),
            f"{prefix}_flow_semigroup_endpoint_weight": ctx.t.new_tensor(
                float(self.meanflow_semigroup_endpoint_weight)
            ),
            f"{prefix}_flow_semigroup_explicit_model_calls": ctx.t.new_tensor(
                float(explicit_model_calls)
            ),
        }
        if ctx.node_clean is not None and node_x is not None and node_state_two_step is not None:
            node_mask = self._node_mask(data, node_x)
            comp_total, comp_state = self._component_semigroup_loss(
                diff_prefix=f"{prefix}_flow_onsite",
                pred_x=node_x,
                clean=ctx.node_clean,
                state_z=ctx.node_state,
                state_two_step=node_state_two_step,
                comp_r=ctx.node_r,
                comp_t=ctx.node_t,
                mask=node_mask,
                weight=self.node_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
            if prefix == "train" and self.log_train_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        node_x - ctx.node_clean,
                        node_mask,
                        "onsite",
                        metric_space=self._target_metric_space(
                            self.node_target_key,
                            node_x,
                        ),
                    )
                )
            if primary_aliases:
                state[f"{prefix}_flow_onsite_loss"] = comp_state[
                    f"{prefix}_flow_onsite_semigroup_loss"
                ]
                if prefix == "train":
                    state["train_onsite_loss"] = comp_state[
                        f"{prefix}_flow_onsite_endpoint_loss"
                    ]
        if ctx.edge_clean is not None and edge_x is not None and edge_state_two_step is not None:
            edge_mask = self._edge_mask(data, edge_x)
            comp_total, comp_state = self._component_semigroup_loss(
                diff_prefix=f"{prefix}_flow_hopping",
                pred_x=edge_x,
                clean=ctx.edge_clean,
                state_z=ctx.edge_state,
                state_two_step=edge_state_two_step,
                comp_r=ctx.edge_r,
                comp_t=ctx.edge_t,
                mask=edge_mask,
                weight=self.edge_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
            if prefix == "train" and self.log_train_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        edge_x - ctx.edge_clean,
                        edge_mask,
                        "hopping",
                        metric_space=self._target_metric_space(
                            self.edge_target_key,
                            edge_x,
                        ),
                    )
                )
            if primary_aliases:
                state[f"{prefix}_flow_hopping_loss"] = comp_state[
                    f"{prefix}_flow_hopping_semigroup_loss"
                ]
                if prefix == "train":
                    state["train_hopping_loss"] = comp_state[
                        f"{prefix}_flow_hopping_endpoint_loss"
                    ]
        if total is None:
            raise KeyError("Pixel MeanFlow semigroup objective could not compute a loss.")
        return total, state, explicit_model_calls

    def _component_meanflow_loss(
        self,
        *,
        diff_prefix: str,
        pred_x: torch.Tensor,
        boundary_x: Optional[torch.Tensor],
        clean: torch.Tensor,
        prior: torch.Tensor,
        state_z: torch.Tensor,
        comp_r: torch.Tensor,
        comp_t: torch.Tensor,
        mask: torch.Tensor,
        weight: float,
        pred_x_eps: Optional[torch.Tensor] = None,
        pred_x_dot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t_view = self._view_time(comp_t, state_z)
        h_view = (comp_t - comp_r).reshape((-1,) + (1,) * (state_z.ndim - 1))
        target_v = prior - clean
        u = (state_z - pred_x) / t_view
        if self.meanflow_jvp_tangent == "boundary" and boundary_x is not None:
            tangent = (state_z - boundary_x) / t_view
        else:
            tangent = target_v
        if pred_x_dot is not None:
            # Exact forward-mode derivative along (dz/dt, dr/dt, dt/dt) =
            # (tangent, 0, 1) with u = (z - x)/t:
            #   du/dt = (tangent - dx/dt)/t - u/t.
            u_detached = (state_z - pred_x.detach()) / t_view
            du_dt = (
                (tangent.detach() - pred_x_dot) / t_view - u_detached / t_view
            ).detach()
        elif pred_x_eps is not None:
            signed_dt = torch.where(
                comp_t <= 1.0 - self.meanflow_fd_eps,
                comp_t.new_full(comp_t.shape, self.meanflow_fd_eps),
                comp_t.new_full(comp_t.shape, -self.meanflow_fd_eps),
            )
            t_eps = (comp_t + signed_dt).clamp(min=self.meanflow_min_t, max=1.0)
            signed_dt = t_eps - comp_t
            dt_view = self._view_time(signed_dt, state_z)
            u_eps = (state_z + dt_view * tangent.detach() - pred_x_eps) / self._view_time(
                t_eps, state_z
            )
            du_dt = ((u_eps - u.detach()) / dt_view).detach()
        else:
            raise ValueError(
                "pixel meanflow component loss needs either pred_x_eps "
                "(finite_difference) or pred_x_dot (jvp)."
            )
        compound_v = u + h_view * du_dt

        velocity_loss, velocity_mse, velocity_mae = self._adaptive_metric_stats(
            compound_v - target_v,
            mask,
            self.loss_type,
            norm_p=self.meanflow_norm_p,
            norm_eps=self.meanflow_norm_eps,
        )
        endpoint_loss, endpoint_mse, endpoint_mae = self._adaptive_metric_stats(
            pred_x - clean,
            mask,
            self.loss_type,
        )
        boundary_loss = endpoint_loss.new_zeros(())
        boundary_mse = endpoint_loss.new_zeros(())
        boundary_mae = endpoint_loss.new_zeros(())
        if boundary_x is not None:
            boundary_v = (state_z - boundary_x) / t_view
            boundary_loss, boundary_mse, boundary_mae = self._adaptive_metric_stats(
                boundary_v - target_v,
                mask,
                self.loss_type,
                norm_p=self.meanflow_norm_p,
                norm_eps=self.meanflow_norm_eps,
            )
        total = weight * (
            velocity_loss
            + self.meanflow_aux_endpoint_weight * endpoint_loss
            + self.meanflow_aux_boundary_v_weight * boundary_loss
        )
        state = {
            f"{diff_prefix}_velocity_loss": velocity_loss.detach(),
            f"{diff_prefix}_velocity_mse": velocity_mse.detach(),
            f"{diff_prefix}_velocity_mae": velocity_mae.detach(),
            f"{diff_prefix}_endpoint_loss": endpoint_loss.detach(),
            f"{diff_prefix}_endpoint_mse": endpoint_mse.detach(),
            f"{diff_prefix}_endpoint_mae": endpoint_mae.detach(),
            f"{diff_prefix}_boundary_v_loss": boundary_loss.detach(),
            f"{diff_prefix}_boundary_v_mse": boundary_mse.detach(),
            f"{diff_prefix}_boundary_v_mae": boundary_mae.detach(),
        }
        return total, state

    def _component_tangent(
        self,
        state: torch.Tensor,
        boundary_x: Optional[torch.Tensor],
        comp_t: torch.Tensor,
        prior: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        if self.meanflow_jvp_tangent == "boundary" and boundary_x is not None:
            return (state - boundary_x) / self._view_time(comp_t, state)
        return prior - clean

    def _meanflow_use_jvp(self) -> bool:
        return self.meanflow_du_dt_backend == "jvp" and not self._meanflow_jvp_disabled

    def _disable_meanflow_jvp(self, exc: Exception) -> None:
        self._meanflow_jvp_disabled = True
        log.warning(
            "Pixel MeanFlow jvp du/dt backend failed (%s: %s); falling back to "
            "finite_difference for the rest of this run. Set "
            "pixel_meanflow.jvp_fallback=false to make this fatal.",
            type(exc).__name__,
            exc,
        )

    def _jvp_du_dt(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        ctx: PixelMFContext,
        node_tangent: Optional[torch.Tensor],
        edge_tangent: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """One forward-mode call returning x-prediction and dx/dt together.

        This is the paper's ``jvp(u_fn, (z, r, t), (v, 0, 1))`` (pMF Alg. 1) up
        to the u = (z - x)/t re-parameterization, which
        _component_meanflow_loss applies analytically.

        Implemented with native ``torch.autograd.forward_ad`` dual tensors
        rather than ``torch.func.jvp``. functorch wraps every tensor in
        storageless interpreter wrappers, so a custom-Function ``jvp``
        staticmethod cannot call a CUDA kernel that reads ``data_ptr()`` (the
        production SO2/cublas grouped-GEMM kernels). Native dual tensors carry
        real storage, so the kernels run; forward-mode tangents also propagate
        layer-by-layer and can be freed as the pass advances, keeping the
        memory overhead well below functorch's. The primal output keeps its
        reverse-mode graph (forward-over-reverse composition), so it replaces
        both the main grad forward and the fd_eps forward of the
        finite_difference backend.
        """
        import torch.autograd.forward_ad as fwAD

        has_node = ctx.node_state is not None
        has_edge = ctx.edge_state is not None

        def _run_dual(node_state, edge_state):
            node_dual = (
                fwAD.make_dual(node_state, node_tangent.detach())
                if has_node
                else None
            )
            edge_dual = (
                fwAD.make_dual(edge_state, edge_tangent.detach())
                if has_edge
                else None
            )
            # (dz/dt, dr/dt, dt/dt) = (tangent, 0, 1): only t carries a unit
            # tangent; r stays primal.
            t_dual = fwAD.make_dual(ctx.t, torch.ones_like(ctx.t))
            return self._predict_clean(
                model, data, ctx, node_dual, edge_dual,
                r=ctx.r, t=t_dual, detach_times=False,
            )

        def _require_tangent(dot, primal, label: str):
            if dot is not None:
                return dot.detach()
            if self.meanflow_jvp_require_tangents:
                raise RuntimeError(
                    "pixel meanflow jvp backend produced no forward tangent for "
                    f"`{label}`; a module/custom autograd.Function likely dropped "
                    "the dual tensor. Falling back to finite_difference is safer "
                    "than treating du/dt as zero (set "
                    "meanflow.jvp_require_tangents=false only for synthetic "
                    "constant-output tests)."
                )
            return torch.zeros_like(primal)

        def _unpack(node_x_dual, edge_x_dual):
            n_x = n_dot = e_x = e_dot = None
            if has_node:
                if node_x_dual is None:
                    raise RuntimeError(
                        "pixel meanflow jvp backend requires the model to emit "
                        f"`{self.node_target_key}` for the node component."
                    )
                n_x, n_dot = fwAD.unpack_dual(node_x_dual)
                n_dot = _require_tangent(n_dot, n_x, self.node_target_key)
            if has_edge:
                if edge_x_dual is None:
                    raise RuntimeError(
                        "pixel meanflow jvp backend requires the model to emit "
                        f"`{self.edge_target_key}` for the edge component."
                    )
                e_x, e_dot = fwAD.unpack_dual(edge_x_dual)
                e_dot = _require_tangent(e_dot, e_x, self.edge_target_key)
            return n_x, n_dot, e_x, e_dot

        def _check_split_primal(label, grad_primal, dual_primal):
            if not self.meanflow_jvp_split_check_primal:
                return
            if grad_primal is None or dual_primal is None:
                return
            if not torch.allclose(
                dual_primal.detach(), grad_primal.detach(),
                rtol=self.meanflow_jvp_split_check_rtol,
                atol=self.meanflow_jvp_split_check_atol,
            ):
                diff = (dual_primal.detach() - grad_primal.detach()).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                raise RuntimeError(
                    f"pixel meanflow split jvp primal mismatch for `{label}` "
                    f"(max_abs={max_abs:.3g}): the no_grad dual forward did not "
                    "match the grad-tracked primal forward, so dx/dt would be "
                    "evaluated at the wrong point (nondeterministic routing?)."
                )

        if self.meanflow_jvp_memory_efficient:
            # Split pass: primal (with reverse graph, the training signal) in a
            # normal forward, then the detached du/dt tangent in a no_grad
            # forward-mode pass whose activations free layer-by-layer. Peak
            # memory stays ~1x (like finite_difference) instead of ~2.2x.
            _, node_x, edge_x = self._predict_clean(
                model, data, ctx, ctx.node_state, ctx.edge_state,
                r=ctx.r, t=ctx.t, detach_times=True,
            )
            with torch.no_grad(), fwAD.dual_level():
                _, node_xd, edge_xd = _run_dual(ctx.node_state, ctx.edge_state)
                node_xd_primal, node_x_dot, edge_xd_primal, edge_x_dot = _unpack(
                    node_xd, edge_xd
                )
            # The dual forward's primal must equal the grad-tracked primal, else
            # dx/dt is evaluated at a different point than the training signal.
            _check_split_primal(self.node_target_key, node_x, node_xd_primal)
            _check_split_primal(self.edge_target_key, edge_x, edge_xd_primal)
            return node_x, edge_x, node_x_dot, edge_x_dot

        # Fused pass: one grad-tracking dual forward yields primal + tangent
        # together (one fewer model call, but every stored activation is a
        # primal+tangent dual -> ~2.2x peak memory).
        with fwAD.dual_level():
            _, node_x_dual, edge_x_dual = _run_dual(ctx.node_state, ctx.edge_state)
            node_x, node_x_dot, edge_x, edge_x_dot = _unpack(node_x_dual, edge_x_dual)
        # unpack_dual's primal keeps the reverse-mode grad_fn built inside the
        # dual level, so node_x/edge_x remain valid training signals here.
        return node_x, edge_x, node_x_dot, edge_x_dot

    def loss_with_model(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        prefix: str = "train",
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        data, ref_data, ctx = self.prepare_batch(data, ref_data, r=r, t=t)
        if self.meanflow_objective == "semigroup":
            total, state, explicit_model_calls = self._semigroup_loss_with_model(
                model, data, ctx, prefix=prefix, primary_aliases=True
            )
            common_state: Dict[str, torch.Tensor] = {
                f"{prefix}_flow_r": ctx.r.detach().mean(),
                f"{prefix}_flow_t": ctx.t.detach().mean(),
                f"{prefix}_flow_h": (ctx.t - ctx.r).detach().mean(),
                f"{prefix}_flow_fm_frac": ctx.fm_mask.detach().float().mean(),
                f"{prefix}_flow_weight": ctx.t.new_tensor(1.0),
                f"{prefix}_flow_objective_finite_difference": ctx.t.new_tensor(0.0),
                f"{prefix}_flow_du_dt_backend_jvp": ctx.t.new_tensor(0.0),
                f"{prefix}_flow_explicit_model_calls": ctx.t.new_tensor(
                    float(explicit_model_calls)
                ),
            }
            common_state.update(state)
            common_state[f"{prefix}_flow_loss"] = total.detach()
            self.last_state = common_state
            return total, common_state

        use_jvp = self._meanflow_use_jvp()
        explicit_model_calls = 0
        node_x = edge_x = None
        if not use_jvp:
            # finite_difference keeps its historical call order:
            # main grad forward -> boundary -> fd_eps forward.
            _, node_x, edge_x = self._predict_clean(
                model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.r, t=ctx.t
            )
            explicit_model_calls += 1
        need_boundary = (
            self.meanflow_jvp_tangent == "boundary"
            or self.meanflow_aux_boundary_v_weight > 0.0
        )
        node_x_boundary = edge_x_boundary = None
        if need_boundary:
            boundary_context = (
                nullcontext()
                if self.meanflow_aux_boundary_v_weight > 0.0
                else torch.no_grad()
            )
            with boundary_context:
                _, node_x_boundary, edge_x_boundary = self._predict_clean(
                    model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.t, t=ctx.t
                )
            explicit_model_calls += 1

        node_tangent = edge_tangent = None
        if ctx.node_state is not None:
            node_tangent = self._component_tangent(
                ctx.node_state, node_x_boundary, ctx.node_t, ctx.node_prior, ctx.node_clean
            )
        if ctx.edge_state is not None:
            edge_tangent = self._component_tangent(
                ctx.edge_state, edge_x_boundary, ctx.edge_t, ctx.edge_prior, ctx.edge_clean
            )

        node_x_dot = edge_x_dot = None
        if use_jvp:
            try:
                node_x, edge_x, node_x_dot, edge_x_dot = self._jvp_du_dt(
                    model, data, ctx, node_tangent, edge_tangent
                )
                # split (memory-efficient) does primal + tangent forwards;
                # fused does one combined dual forward.
                explicit_model_calls += 2 if self.meanflow_jvp_memory_efficient else 1
            except Exception as exc:
                if not self.meanflow_jvp_fallback:
                    raise
                self._disable_meanflow_jvp(exc)
                use_jvp = False
                _, node_x, edge_x = self._predict_clean(
                    model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.r, t=ctx.t
                )
                explicit_model_calls += 1

        node_x_eps = edge_x_eps = None
        if not use_jvp:
            node_state_eps = edge_state_eps = None
            if ctx.node_state is not None:
                node_dt = torch.where(
                    ctx.node_t <= 1.0 - self.meanflow_fd_eps,
                    ctx.node_t.new_full(ctx.node_t.shape, self.meanflow_fd_eps),
                    ctx.node_t.new_full(ctx.node_t.shape, -self.meanflow_fd_eps),
                )
                node_dt = (ctx.node_t + node_dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.node_t
                node_state_eps = ctx.node_state + self._view_time(node_dt, ctx.node_state) * node_tangent.detach()
            if ctx.edge_state is not None:
                edge_dt = torch.where(
                    ctx.edge_t <= 1.0 - self.meanflow_fd_eps,
                    ctx.edge_t.new_full(ctx.edge_t.shape, self.meanflow_fd_eps),
                    ctx.edge_t.new_full(ctx.edge_t.shape, -self.meanflow_fd_eps),
                )
                edge_dt = (ctx.edge_t + edge_dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.edge_t
                edge_state_eps = ctx.edge_state + self._view_time(edge_dt, ctx.edge_state) * edge_tangent.detach()
            graph_dt = torch.where(
                ctx.t <= 1.0 - self.meanflow_fd_eps,
                ctx.t.new_full(ctx.t.shape, self.meanflow_fd_eps),
                ctx.t.new_full(ctx.t.shape, -self.meanflow_fd_eps),
            )
            t_eps = (ctx.t + graph_dt).clamp(min=self.meanflow_min_t, max=1.0)
            with torch.no_grad():
                _, node_x_eps, edge_x_eps = self._predict_clean(
                    model,
                    data,
                    ctx,
                    node_state_eps if node_state_eps is not None else ctx.node_state,
                    edge_state_eps if edge_state_eps is not None else ctx.edge_state,
                    r=ctx.r,
                    t=t_eps,
                )
            explicit_model_calls += 1

        total = None
        state: Dict[str, torch.Tensor] = {
            f"{prefix}_flow_r": ctx.r.detach().mean(),
            f"{prefix}_flow_t": ctx.t.detach().mean(),
            f"{prefix}_flow_h": (ctx.t - ctx.r).detach().mean(),
            f"{prefix}_flow_fm_frac": ctx.fm_mask.detach().float().mean(),
            f"{prefix}_flow_weight": ctx.t.new_tensor(1.0),
            f"{prefix}_flow_objective_finite_difference": ctx.t.new_tensor(1.0),
            f"{prefix}_flow_objective_semigroup": ctx.t.new_tensor(
                1.0 if self.meanflow_objective == "hybrid" else 0.0
            ),
            # canary scalars: catch silent jvp fallbacks and count the explicit
            # model calls per step (boundary + main/jvp [+ fd_eps]).
            f"{prefix}_flow_du_dt_backend_jvp": ctx.t.new_tensor(
                1.0 if use_jvp else 0.0
            ),
            f"{prefix}_flow_explicit_model_calls": ctx.t.new_tensor(
                float(explicit_model_calls)
            ),
        }
        if ctx.node_clean is not None and node_x is not None:
            node_mask = self._node_mask(data, node_x)
            comp_total, comp_state = self._component_meanflow_loss(
                diff_prefix=f"{prefix}_flow_onsite",
                pred_x=node_x,
                boundary_x=node_x_boundary,
                clean=ctx.node_clean,
                prior=ctx.node_prior,
                state_z=ctx.node_state,
                comp_r=ctx.node_r,
                comp_t=ctx.node_t,
                pred_x_eps=node_x_eps,
                pred_x_dot=node_x_dot,
                mask=node_mask,
                weight=self.node_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
            if prefix == "train" and self.log_train_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        node_x - ctx.node_clean,
                        node_mask,
                        "onsite",
                        metric_space=self._target_metric_space(
                            self.node_target_key,
                            node_x,
                        ),
                    )
                )
            # Legacy aliases so pMF logs line up with CFM/supervised curves:
            # *_flow_onsite_loss mirrors the velocity objective; the train_*
            # keys carry the endpoint error, which is the cross-route
            # comparable quantity (see plan §4.3).
            state[f"{prefix}_flow_onsite_loss"] = comp_state[f"{prefix}_flow_onsite_velocity_loss"]
            if prefix == "train":
                state["train_onsite_loss"] = comp_state[f"{prefix}_flow_onsite_endpoint_loss"]
        if ctx.edge_clean is not None and edge_x is not None:
            edge_mask = self._edge_mask(data, edge_x)
            comp_total, comp_state = self._component_meanflow_loss(
                diff_prefix=f"{prefix}_flow_hopping",
                pred_x=edge_x,
                boundary_x=edge_x_boundary,
                clean=ctx.edge_clean,
                prior=ctx.edge_prior,
                state_z=ctx.edge_state,
                comp_r=ctx.edge_r,
                comp_t=ctx.edge_t,
                pred_x_eps=edge_x_eps,
                pred_x_dot=edge_x_dot,
                mask=edge_mask,
                weight=self.edge_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
            if prefix == "train" and self.log_train_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        edge_x - ctx.edge_clean,
                        edge_mask,
                        "hopping",
                        metric_space=self._target_metric_space(
                            self.edge_target_key,
                            edge_x,
                        ),
                    )
                )
            state[f"{prefix}_flow_hopping_loss"] = comp_state[f"{prefix}_flow_hopping_velocity_loss"]
            if prefix == "train":
                state["train_hopping_loss"] = comp_state[f"{prefix}_flow_hopping_endpoint_loss"]
        if total is None:
            raise KeyError("Pixel MeanFlow could not compute a loss from configured node/edge targets.")
        if (
            self.meanflow_objective == "hybrid"
            and (
                self.meanflow_semigroup_weight != 0.0
                or self.meanflow_semigroup_endpoint_weight != 0.0
            )
        ):
            semigroup_total, semigroup_state, semigroup_calls = self._semigroup_loss_with_model(
                model,
                data,
                ctx,
                prefix=prefix,
                node_x=node_x,
                edge_x=edge_x,
                primary_aliases=False,
            )
            total = total + semigroup_total
            state.update(semigroup_state)
            state[f"{prefix}_flow_explicit_model_calls"] = (
                state[f"{prefix}_flow_explicit_model_calls"]
                + ctx.t.new_tensor(float(semigroup_calls))
            )
        if self.router_z_loss_coef > 0.0:
            # The main prediction is intentionally not retained; keep router regularization
            # out of pMF unless a future model-level integration returns it explicitly.
            log.debug("z_loss_coef is ignored by HamiltonianPixelMeanFlow.loss_with_model")
        state[f"{prefix}_flow_loss"] = total.detach()
        self.last_state = state
        return total, state

    def loss(self, pred_data: AtomicDataDict.Type, ref_data: AtomicDataDict.Type, ctx: PixelMFContext):
        raise RuntimeError("HamiltonianPixelMeanFlow requires loss_with_model(model, data, ref_data).")

    def sample(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        *,
        num_steps: int,
    ) -> AtomicDataDict.Type:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        state = data.copy()
        node_base = self._sampling_base(state, self.node_h0_key, self.node_target_key, "node")
        edge_base = self._sampling_base(state, self.edge_h0_key, self.edge_target_key, "edge")
        if node_base is None and edge_base is None:
            raise KeyError("Pixel MeanFlow sampling requires node and/or edge Hamiltonian start features.")
        # One shared Haar-DM candidate for node+edge; None for every other prior.
        haar_candidate_idx = self._resolve_haar_candidate_idx(
            state, node_like=node_base, edge_like=edge_base
        )
        node_z = None if node_base is None else self._prior_like(
            torch.zeros_like(node_base),
            self.node_sigma,
            data=state,
            label="node",
            base=node_base,
            reference_scale=False,
            candidate_idx=haar_candidate_idx,
        )
        edge_z = None if edge_base is None else self._prior_like(
            torch.zeros_like(edge_base),
            self.edge_sigma,
            data=state,
            label="edge",
            base=edge_base,
            reference_scale=False,
            candidate_idx=haar_candidate_idx,
        )
        like = node_z if node_z is not None else edge_z
        num_graphs = self._num_graphs(state)
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=like.device, dtype=like.dtype)
        ctx = PixelMFContext(
            r=grid.new_full((num_graphs,), 0.0),
            t=grid.new_full((num_graphs,), 1.0),
            fm_mask=torch.zeros(num_graphs, device=like.device, dtype=torch.bool),
            node_r=None,
            node_t=None,
            edge_r=None,
            edge_t=None,
            node_base=node_base,
            edge_base=edge_base,
            node_clean=None if node_z is None else torch.zeros_like(node_z),
            edge_clean=None if edge_z is None else torch.zeros_like(edge_z),
            node_state=node_z,
            edge_state=edge_z,
            node_prior=node_z,
            edge_prior=edge_z,
        )
        for step in range(num_steps):
            t = torch.full((num_graphs,), float(grid[step].item()), device=like.device, dtype=like.dtype)
            t = t.clamp_min(self.meanflow_min_t)
            r = torch.full((num_graphs,), float(grid[step + 1].item()), device=like.device, dtype=like.dtype)
            ctx.r, ctx.t = r, t
            ctx.node_t, ctx.edge_t = self._expand_graph_times(
                state,
                t,
                node_count=None if node_z is None else node_z.shape[0],
                edge_count=None if edge_z is None else edge_z.shape[0],
            )
            ctx.node_r, ctx.edge_r = self._expand_graph_times(
                state,
                r,
                node_count=None if node_z is None else node_z.shape[0],
                edge_count=None if edge_z is None else edge_z.shape[0],
            )
            _, node_x, edge_x = self._predict_clean(model, state, ctx, node_z, edge_z, r=r, t=t)
            if node_z is not None:
                node_h = (ctx.node_t - ctx.node_r).reshape((-1,) + (1,) * (node_z.ndim - 1))
                node_z = node_z - node_h * (
                    (node_z - node_x) / self._view_time(ctx.node_t, node_z)
                )
                ctx.node_state = node_z
            if edge_z is not None:
                edge_h = (ctx.edge_t - ctx.edge_r).reshape((-1,) + (1,) * (edge_z.ndim - 1))
                edge_z = edge_z - edge_h * (
                    (edge_z - edge_x) / self._view_time(ctx.edge_t, edge_z)
                )
                ctx.edge_state = edge_z
        zero = torch.zeros(num_graphs, device=like.device, dtype=like.dtype)
        if self.meanflow_sample_final_forward:
            # One extra endpoint-conditioned forward so `out` carries the
            # model's full output surface -- block-native heads'
            # node/edge Hamiltonian blocks, router monitors, etc. --
            # mirroring HamiltonianCFM.sample, whose state is always a
            # prediction dict. Without this, pMF samples contain no model
            # outputs at all and block-consuming losses (e.g. the blockwise
            # compatible validation) KeyError. Disable via
            # pixel_meanflow.sample_final_forward=false to save one forward
            # when only the integrated features are needed.
            final_t = zero.clamp_min(self.meanflow_min_t)
            ctx.r, ctx.t = zero, final_t
            ctx.node_t, ctx.edge_t = self._expand_graph_times(
                state,
                final_t,
                node_count=None if node_z is None else node_z.shape[0],
                edge_count=None if edge_z is None else edge_z.shape[0],
            )
            ctx.node_r, ctx.edge_r = self._expand_graph_times(
                state,
                zero,
                node_count=None if node_z is None else node_z.shape[0],
                edge_count=None if edge_z is None else edge_z.shape[0],
            )
            pred, _node_x, _edge_x = self._predict_clean(
                model, state, ctx, node_z, edge_z, r=zero, t=final_t
            )
            out = pred.copy()
        else:
            out = state.copy()
        if node_z is not None:
            out[self.node_h0_key] = node_base + node_z if self.mode == "residual" else node_z
            out[self.node_target_key] = out[self.node_h0_key]
        if edge_z is not None:
            out[self.edge_h0_key] = edge_base + edge_z if self.mode == "residual" else edge_z
            out[self.edge_target_key] = out[self.edge_h0_key]
        self._write_times(out, zero, zero)
        return out
