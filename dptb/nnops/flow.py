from __future__ import annotations

"""Conditional Flow Matching utilities for Hamiltonian training.

This module is intentionally lightweight and trainer-side.  It does not require a
new DeePTB model class: at every training step it replaces the H0 node/edge
fields by an interpolated Hamiltonian state H_t and trains the existing model to
predict the clean converged Hamiltonian.  This mirrors the residual-CFM training
used by QHFlow/QHFlow2, but is adapted to DeePTB/NextHAM-style physical H0
features.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import torch

from dptb.data import AtomicDataDict, _keys

log = logging.getLogger(__name__)


def _to_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return torch.float32


@dataclass
class CFMContext:
    t: torch.Tensor
    node_t: Optional[torch.Tensor]
    edge_t: Optional[torch.Tensor]
    node_base: Optional[torch.Tensor]
    edge_base: Optional[torch.Tensor]
    node_target: Optional[torch.Tensor]
    edge_target: Optional[torch.Tensor]
    node_current: Optional[torch.Tensor]
    edge_current: Optional[torch.Tensor]
    node_prior: Optional[torch.Tensor]
    edge_prior: Optional[torch.Tensor]


class HamiltonianCFM:
    """Trainer-side residual conditional flow matching helper.

    DeePTB's NextHAM-like branch already supports physical initial Hamiltonian
    features through ``node_h0`` and ``edge_h0``.  CFM is implemented by
    replacing these fields with an interpolated state

        H_t = H_base + ((1 - t) * eps + t * (H_ref - H_base)),

    then asking the original model to predict ``H_ref``.  The loss is the
    endpoint parameterization of the CFM velocity loss,

        ||(H_pred - H_t)/(1 - t) - (H_ref - H_t)/(1 - t)||^2,

    i.e. optionally weighted endpoint error ``||H_pred - H_ref||^2/(1-t)^2``.
    """

    def __init__(
        self,
        options: Optional[Dict[str, Any]],
        *,
        idp: Any = None,
        dtype: Any = torch.float32,
        device: Any = torch.device("cpu"),
    ) -> None:
        options = dict(options or {})
        self.enabled = bool(options.get("enabled", False))
        self.options = options
        self.idp = idp
        self.dtype = _to_torch_dtype(dtype)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # Keys.  The defaults match DeePTB's NextHAM/H0 branch.
        self.node_h0_key = str(options.get("node_h0_key", _keys.NODE_H0_KEY))
        self.edge_h0_key = str(options.get("edge_h0_key", _keys.EDGE_H0_KEY))
        self.node_target_key = str(options.get("node_target_key", _keys.NODE_FEATURES_KEY))
        self.edge_target_key = str(options.get("edge_target_key", _keys.EDGE_FEATURES_KEY))
        self.flow_time_key = str(options.get("flow_time_key", "flow_time"))

        # Residual CFM is the recommended mode for DeePTB: base = DFT/NextHAM H0.
        self.mode = str(options.get("mode", "residual")).lower()
        if self.mode not in {"residual", "full"}:
            raise ValueError(f"Unsupported flow_options.mode={self.mode!r}; use 'residual' or 'full'.")

        # In DeePTB, zero prior means the inference start state is exactly physical H0.
        # Gaussian is available for QHFlow-style noisy residual priors.
        self.prior = str(options.get("prior", "zero")).lower()
        if self.prior not in {"zero", "gaussian", "residual_gaussian"}:
            raise ValueError(
                f"Unsupported flow_options.prior={self.prior!r}; "
                "use 'zero', 'gaussian', or 'residual_gaussian'."
            )

        self.node_sigma = float(options.get("node_sigma", 1.0))
        self.edge_sigma = float(options.get("edge_sigma", 1.0))
        self.residual_sigma_floor = float(options.get("residual_sigma_floor", 1.0e-6))

        # Time sampling.  QHFlow uses U(0,1); we expose a t0 mass so the network
        # explicitly sees the physical-H0 one-step inference point.
        self.time_sampling = str(options.get("time_sampling", "uniform")).lower()
        self.t_min = float(options.get("t_min", 0.0))
        self.t_max = float(options.get("t_max", 0.999))
        self.t0_probability = float(options.get("t0_probability", 0.0))
        self.t_eps = float(options.get("t_eps", 1.0e-3))
        self.endpoint_weight_power = float(options.get("endpoint_weight_power", 0.0))
        self.endpoint_weight_cap = float(options.get("endpoint_weight_cap", 100.0))
        self.omit_time_scaling = bool(options.get("omit_time_scaling", True))
        self.validation_ode_steps = tuple(
            sorted({int(v) for v in options.get("validation_ode_steps", [1, 3]) if int(v) > 0})
        )

        # Loss and regularization.
        self.loss_type = str(options.get("loss_type", "mse")).lower()
        if self.loss_type not in {"mse", "l1_rmse"}:
            raise ValueError("flow_options.loss_type must be 'mse' or 'l1_rmse'.")
        self.node_weight = float(options.get("node_weight", 1.0))
        self.edge_weight = float(options.get("edge_weight", 1.0))
        self.router_z_loss_coef = float(options.get("z_loss_coef", 0.0))

        # Safety switches.
        self.overwrite_feature_keys = bool(options.get("overwrite_feature_keys", True))
        self.detach_interpolated_h0 = bool(options.get("detach_interpolated_h0", True))
        self.warn_missing_h0 = bool(options.get("warn_missing_h0", True))
        self.strict_h0 = bool(options.get("strict_h0", True))
        self.component_reduction = str(options.get("component_reduction", "global_elements")).lower()
        if self.component_reduction not in {"global_elements", "equal_components"}:
            raise ValueError(
                "flow_options.component_reduction must be 'global_elements' or 'equal_components'."
            )

        self.last_state: Dict[str, torch.Tensor] = {}
        if self.enabled:
            log.info(
                "Hamiltonian CFM enabled: mode=%s prior=%s t=[%.3g, %.3g] t0_prob=%.3g loss=%s",
                self.mode,
                self.prior,
                self.t_min,
                self.t_max,
                self.t0_probability,
                self.loss_type,
            )

    # ------------------------------------------------------------------
    # Sampling / interpolation
    # ------------------------------------------------------------------
    def _sample_t(
        self,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lo = max(0.0, min(self.t_min, 1.0))
        hi = max(lo, min(self.t_max, 1.0 - self.t_eps))
        if self.time_sampling == "uniform":
            t = lo + (hi - lo) * torch.rand(num_graphs, device=device, dtype=dtype)
        elif self.time_sampling == "logit_normal":
            mean = float(self.options.get("time_logit_mean", -0.4))
            std = float(self.options.get("time_logit_std", 1.0))
            raw = torch.randn(num_graphs, device=device, dtype=dtype) * std + mean
            t = torch.sigmoid(raw)
            t = lo + (hi - lo) * t
        else:
            raise ValueError(f"Unsupported flow_options.time_sampling={self.time_sampling!r}")
        if self.t0_probability > 0.0:
            use_t0 = torch.rand(num_graphs, device=device) < self.t0_probability
            t = torch.where(use_t0, torch.zeros_like(t), t)
        return t.clamp(min=lo, max=hi)

    @staticmethod
    def _num_graphs(data: AtomicDataDict.Type) -> int:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    @staticmethod
    def _normalize_t(
        t: torch.Tensor,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
        if t.numel() == 1:
            return t.expand(num_graphs)
        if t.numel() != num_graphs:
            raise ValueError(f"Expected one flow time per graph ({num_graphs}), got {t.numel()}.")
        return t

    @staticmethod
    def _expand_graph_times(
        data: AtomicDataDict.Type,
        t: torch.Tensor,
        *,
        node_count: Optional[int],
        edge_count: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None:
            if node_count is None:
                batch = torch.zeros(0, device=t.device, dtype=torch.long)
            else:
                batch = torch.zeros(node_count, device=t.device, dtype=torch.long)
        else:
            batch = batch.to(device=t.device, dtype=torch.long).reshape(-1)
        node_t = None if node_count is None else t.index_select(0, batch[:node_count])

        edge_t = None
        if edge_count is not None:
            edge_index = data.get(_keys.EDGE_INDEX_KEY, None)
            if edge_index is None:
                if t.numel() != 1:
                    raise KeyError("Per-graph edge flow time requires `edge_index`.")
                edge_t = t.expand(edge_count)
            else:
                edge_center = edge_index[0].to(device=t.device, dtype=torch.long).reshape(-1)
                edge_graph = batch.index_select(0, edge_center[:edge_count])
                edge_t = t.index_select(0, edge_graph)
        return node_t, edge_t

    def _base_like(self, data: AtomicDataDict.Type, target: torch.Tensor, h0_key: str, label: str) -> torch.Tensor:
        if self.mode == "full":
            return torch.zeros_like(target)

        base = data.get(h0_key, None)
        if base is None:
            if self.strict_h0:
                raise KeyError(
                    f"CFM residual mode requires `{h0_key}` for the {label} base; "
                    "disable strict_h0 only for an explicit zero-base experiment."
                )
            if self.warn_missing_h0:
                log.warning(
                    "CFM residual mode did not find `%s`; falling back to zeros for %s base. "
                    "For NextHAM-style training, make sure the dataset emits node_h0/edge_h0.",
                    h0_key,
                    label,
                )
            base = torch.zeros_like(target)
        else:
            base = base.to(device=target.device, dtype=target.dtype)
            if base.shape != target.shape:
                if self.strict_h0:
                    raise ValueError(
                        f"CFM {label} base `{h0_key}` shape {tuple(base.shape)} "
                        f"!= target shape {tuple(target.shape)}."
                    )
                if self.warn_missing_h0:
                    log.warning(
                        "CFM %s base `%s` shape %s != target shape %s; using zeros.",
                        label,
                        h0_key,
                        tuple(base.shape),
                        tuple(target.shape),
                    )
                base = torch.zeros_like(target)
        return base

    def _prior_like(self, residual: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.prior == "zero":
            return torch.zeros_like(residual)
        if self.prior == "gaussian":
            return torch.randn_like(residual) * sigma
        # residual_gaussian: match global residual scale, useful as a rough TE/GOE proxy.
        scale = residual.detach().std().clamp_min(self.residual_sigma_floor)
        return torch.randn_like(residual) * scale * sigma

    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[AtomicDataDict.Type, AtomicDataDict.Type, CFMContext]:
        """Return a model-input dict with interpolated H_t written to H0 keys."""
        if not self.enabled:
            raise RuntimeError("HamiltonianCFM.prepare_batch called while disabled")

        data = data.copy()
        ref_data = ref_data.copy()

        node_target = ref_data.get(self.node_target_key, None)
        edge_target = ref_data.get(self.edge_target_key, None)
        if node_target is None and edge_target is None:
            raise KeyError(
                "CFM requires node and/or edge Hamiltonian targets in ref_data; "
                f"looked for `{self.node_target_key}` and `{self.edge_target_key}`."
            )

        like = node_target if node_target is not None else edge_target
        device = like.device
        dtype = like.dtype if torch.is_floating_point(like) else self.dtype
        num_graphs = self._num_graphs(data)
        if t is None:
            t = self._sample_t(num_graphs=num_graphs, device=device, dtype=dtype)
        else:
            t = self._normalize_t(t, num_graphs=num_graphs, device=device, dtype=dtype)
        node_t, edge_t = self._expand_graph_times(
            data,
            t,
            node_count=None if node_target is None else node_target.shape[0],
            edge_count=None if edge_target is None else edge_target.shape[0],
        )

        node_base = edge_base = node_current = edge_current = None
        node_prior = edge_prior = None

        if node_target is not None:
            node_target = node_target.to(device=device, dtype=dtype)
            node_base = self._base_like(data, node_target, self.node_h0_key, "node")
            node_res = node_target - node_base
            node_prior = self._prior_like(node_res, self.node_sigma)
            node_t_view = node_t.reshape((-1,) + (1,) * (node_target.ndim - 1))
            node_current = node_base + (1.0 - node_t_view) * node_prior + node_t_view * node_res
            if self.detach_interpolated_h0:
                node_current = node_current.detach()
            data[self.node_h0_key] = node_current
            if self.overwrite_feature_keys:
                data[self.node_target_key] = node_current

        if edge_target is not None:
            edge_target = edge_target.to(device=device, dtype=dtype)
            edge_base = self._base_like(data, edge_target, self.edge_h0_key, "edge")
            edge_res = edge_target - edge_base
            edge_prior = self._prior_like(edge_res, self.edge_sigma)
            edge_t_view = edge_t.reshape((-1,) + (1,) * (edge_target.ndim - 1))
            edge_current = edge_base + (1.0 - edge_t_view) * edge_prior + edge_t_view * edge_res
            if self.detach_interpolated_h0:
                edge_current = edge_current.detach()
            data[self.edge_h0_key] = edge_current
            if self.overwrite_feature_keys:
                data[self.edge_target_key] = edge_current

        data[self.flow_time_key] = t.detach()
        ref_data[self.flow_time_key] = t.detach()

        return data, ref_data, CFMContext(
            t=t,
            node_t=node_t,
            edge_t=edge_t,
            node_base=node_base,
            edge_base=edge_base,
            node_target=node_target,
            edge_target=edge_target,
            node_current=node_current,
            edge_current=edge_current,
            node_prior=node_prior,
            edge_prior=edge_prior,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _node_mask(self, data: AtomicDataDict.Type, pred: torch.Tensor) -> torch.Tensor:
        if self.idp is None or AtomicDataDict.ATOM_TYPE_KEY not in data:
            return torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        mask = self.idp.mask_to_nrme.to(device=atom_types.device)[atom_types]
        if "expert_node_mask" in data:
            mask = mask & data["expert_node_mask"].to(device=mask.device).unsqueeze(-1)
        return mask.to(device=pred.device)

    def _edge_mask(self, data: AtomicDataDict.Type, pred: torch.Tensor) -> torch.Tensor:
        if self.idp is None or AtomicDataDict.EDGE_TYPE_KEY not in data:
            return torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        edge_types = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
        mask = self.idp.mask_to_erme.to(device=edge_types.device)[edge_types]
        if "expert_edge_mask" in data:
            mask = mask & data["expert_edge_mask"].to(device=mask.device).unsqueeze(-1)
        return mask.to(device=pred.device)

    @staticmethod
    def _metric_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_f = mask.to(device=diff.device, dtype=diff.dtype)
        count = mask_f.sum().clamp_min(1.0)
        if weights is None:
            weights_f = torch.ones_like(diff)
        else:
            weights_f = weights.to(device=diff.device, dtype=diff.dtype)
            weights_f = weights_f.reshape((-1,) + (1,) * (diff.ndim - 1))
            weights_f = weights_f.expand_as(diff)
        if loss_type == "mse":
            numerator = (diff.square() * mask_f * weights_f).sum()
            return numerator / count, numerator, count
        abs_sum = (diff.abs() * mask_f * weights_f).sum()
        sq_sum = (diff.square() * mask_f * weights_f).sum()
        l1 = abs_sum / count
        rmse = torch.sqrt(sq_sum / count + 1e-12)
        metric = 0.5 * (l1 + rmse)
        return metric, metric * count, count

    def _time_weight(self, t: torch.Tensor) -> torch.Tensor:
        if self.omit_time_scaling or self.endpoint_weight_power == 0.0:
            return torch.ones_like(t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        w = denom.pow(-self.endpoint_weight_power)
        if self.endpoint_weight_cap > 0:
            w = w.clamp_max(self.endpoint_weight_cap)
        return w

    def loss(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            raise RuntimeError("HamiltonianCFM.loss called while disabled")

        t_weight = self._time_weight(ctx.t).to(device=ctx.t.device, dtype=ctx.t.dtype)
        total = None
        total_numerator = None
        total_count = None
        state: Dict[str, torch.Tensor] = {
            "train_flow_t": ctx.t.detach().mean(),
            "train_flow_weight": t_weight.detach().mean(),
        }

        node_loss = None
        if ctx.node_target is not None and self.node_target_key in pred_data:
            pred = pred_data[self.node_target_key]
            target = ref_data[self.node_target_key].to(device=pred.device, dtype=pred.dtype)
            mask = self._node_mask(pred_data, pred)
            node_weights = self._time_weight(ctx.node_t)
            node_loss, node_numerator, node_count = self._metric_stats(
                pred - target, mask, self.loss_type, node_weights
            )
            total = self.node_weight * node_loss if total is None else total + self.node_weight * node_loss
            total_numerator = self.node_weight * node_numerator
            total_count = self.node_weight * node_count
            state["train_flow_onsite_loss"] = node_loss.detach()
            state["train_onsite_loss"] = node_loss.detach()

        edge_loss = None
        if ctx.edge_target is not None and self.edge_target_key in pred_data:
            pred = pred_data[self.edge_target_key]
            target = ref_data[self.edge_target_key].to(device=pred.device, dtype=pred.dtype)
            mask = self._edge_mask(pred_data, pred)
            edge_weights = self._time_weight(ctx.edge_t)
            edge_loss, edge_numerator, edge_count = self._metric_stats(
                pred - target, mask, self.loss_type, edge_weights
            )
            total = self.edge_weight * edge_loss if total is None else total + self.edge_weight * edge_loss
            if total_numerator is None:
                total_numerator = self.edge_weight * edge_numerator
                total_count = self.edge_weight * edge_count
            else:
                total_numerator = total_numerator + self.edge_weight * edge_numerator
                total_count = total_count + self.edge_weight * edge_count
            state["train_flow_hopping_loss"] = edge_loss.detach()
            state["train_hopping_loss"] = edge_loss.detach()

        if total is None:
            raise KeyError(
                "CFM could not compute a loss because model outputs do not contain "
                f"`{self.node_target_key}` or `{self.edge_target_key}`."
            )
        if self.component_reduction == "global_elements":
            total = total_numerator / total_count.clamp_min(1.0)

        if self.router_z_loss_coef > 0.0 and "mean_max_prob" in pred_data:
            z_loss = pred_data["mean_max_prob"]
            if torch.is_tensor(z_loss):
                total = total + self.router_z_loss_coef * z_loss
                state["mean_max_prob"] = z_loss.detach()

        state["train_flow_loss"] = total.detach()
        self.last_state = state
        return total, state

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sampling_base(
        self,
        data: AtomicDataDict.Type,
        h0_key: str,
        feature_key: str,
        label: str,
    ) -> Optional[torch.Tensor]:
        base = data.get(h0_key, None)
        if base is None and self.mode == "full":
            feature = data.get(feature_key, None)
            return None if feature is None else torch.zeros_like(feature)
        if base is None:
            if self.strict_h0:
                raise KeyError(f"Flow sampling requires `{h0_key}` for the {label} start state.")
            feature = data.get(feature_key, None)
            return None if feature is None else torch.zeros_like(feature)
        return base

    def sample(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        *,
        num_steps: int,
    ) -> AtomicDataDict.Type:
        """Euler-integrate the endpoint-parameterized flow from the configured prior."""
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        state = data.copy()
        node_current = self._sampling_base(state, self.node_h0_key, self.node_target_key, "node")
        edge_current = self._sampling_base(state, self.edge_h0_key, self.edge_target_key, "edge")
        if node_current is None and edge_current is None:
            raise KeyError("Flow sampling requires node and/or edge Hamiltonian start features.")

        if self.prior != "zero":
            if node_current is not None:
                node_current = node_current + self._prior_like(node_current, self.node_sigma)
            if edge_current is not None:
                edge_current = edge_current + self._prior_like(edge_current, self.edge_sigma)

        like = node_current if node_current is not None else edge_current
        num_graphs = self._num_graphs(state)
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            cur_t = float(step) * dt
            graph_t = torch.full((num_graphs,), cur_t, device=like.device, dtype=like.dtype)
            if node_current is not None:
                state[self.node_h0_key] = node_current
                if self.overwrite_feature_keys:
                    state[self.node_target_key] = node_current
            if edge_current is not None:
                state[self.edge_h0_key] = edge_current
                if self.overwrite_feature_keys:
                    state[self.edge_target_key] = edge_current
            state[self.flow_time_key] = graph_t
            prediction = model(state)
            denom = max(1.0 - cur_t, self.t_eps)
            if node_current is not None:
                endpoint = prediction[self.node_target_key]
                node_current = node_current + dt * (endpoint - node_current) / denom
            if edge_current is not None:
                endpoint = prediction[self.edge_target_key]
                edge_current = edge_current + dt * (endpoint - edge_current) / denom
            state = prediction.copy()

        if node_current is not None:
            state[self.node_h0_key] = node_current
            state[self.node_target_key] = node_current
        if edge_current is not None:
            state[self.edge_h0_key] = edge_current
            state[self.edge_target_key] = edge_current
        state[self.flow_time_key] = torch.ones(num_graphs, device=like.device, dtype=like.dtype)
        return state
