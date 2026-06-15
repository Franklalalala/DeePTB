from __future__ import annotations

"""PyTorch Riemannian MeanFlow utilities for DeePTB Hamiltonian CFM."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import torch

from dptb.data import AtomicDataDict
from dptb.nnops.flow import (
    HamiltonianCFM,
    build_hamiltonian_flow as _BASE_BUILD_HAMILTONIAN_FLOW,
)

log = logging.getLogger(__name__)

_RMF_ALIASES = {
    "rmf",
    "riemannian_meanflow",
    "riemannian_mean_flow",
    "riemannian_flow",
    "riemannian_manifold_flow",
}


def _opts(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(options or {})


def _time_view(t: torch.Tensor, like: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return t.reshape((-1,) + (1,) * (like.ndim - 1)).clamp_min(eps)


class ManifoldOps:
    """Minimal manifold API used by Hamiltonian RMF."""

    name = "base"

    def project(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def geodesic_interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.expmap(x0, _time_view(t, x0) * self.logmap(x0, x1))

    def tangent_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.logmap(x0, x1)


class EuclideanManifold(ManifoldOps):
    """Flat residual Hamiltonian manifold fallback; no optional deps."""

    name = "euclidean"

    def project(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return v

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def geodesic_interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x0 + _time_view(t, x0) * (x1 - x0)

    def tangent_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return x1 - x0


def build_manifold(name: str = "euclidean", options: Optional[Dict[str, Any]] = None) -> ManifoldOps:
    del options
    key = str(name or "euclidean").lower()
    if key in {"euclidean", "flat", "r", "rn", "r^n", "residual"}:
        return EuclideanManifold()
    raise NotImplementedError(
        f"RMF manifold {name!r} is not implemented in DeePTB yet; use 'euclidean'."
    )


@dataclass
class RMFContext:
    r: torch.Tensor
    t: torch.Tensor
    fm_mask: torch.Tensor
    node_r: Optional[torch.Tensor]
    node_t: Optional[torch.Tensor]
    edge_r: Optional[torch.Tensor]
    edge_t: Optional[torch.Tensor]
    node_base: Optional[torch.Tensor]
    edge_base: Optional[torch.Tensor]
    node_clean: Optional[torch.Tensor]
    edge_clean: Optional[torch.Tensor]
    node_state: Optional[torch.Tensor]
    edge_state: Optional[torch.Tensor]
    node_prior: Optional[torch.Tensor]
    edge_prior: Optional[torch.Tensor]
    node_target_velocity: Optional[torch.Tensor]
    edge_target_velocity: Optional[torch.Tensor]


class RMFTimeSampler:
    """MeanFlow-compatible two-time sampler."""

    def __init__(self, options: Optional[Dict[str, Any]] = None, fallback: Optional[Dict[str, Any]] = None):
        opts = dict(fallback or {})
        opts.update(dict(options or {}))
        self.kind = str(opts.get("type", opts.get("time_sampling", "logit_normal"))).lower()
        self.p_mean = float(opts.get("p_mean", opts.get("time_logit_mean", -0.4)))
        self.p_std = float(opts.get("p_std", opts.get("time_logit_std", 1.0)))
        self.same_time_probability = float(opts.get("same_time_probability", opts.get("data_proportion", 0.5)))
        self.tr_uniform_prob = float(opts.get("tr_uniform_prob", 0.1))
        self.min_t = float(opts.get("min_t", 0.05))

    def _sample_base(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.kind == "uniform":
            return torch.rand(n, device=device, dtype=dtype)
        if self.kind == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device, dtype=dtype) * self.p_std + self.p_mean)
        raise ValueError(f"Unsupported RMF time sampler {self.kind!r}")

    def sample(self, *, num_graphs: int, device: torch.device, dtype: torch.dtype):
        t = self._sample_base(num_graphs, device=device, dtype=dtype)
        r = self._sample_base(num_graphs, device=device, dtype=dtype)
        if self.tr_uniform_prob > 0.0:
            mask = torch.rand(num_graphs, device=device) < self.tr_uniform_prob
            t = torch.where(mask, torch.rand(num_graphs, device=device, dtype=dtype), t)
            r = torch.where(mask, torch.rand(num_graphs, device=device, dtype=dtype), r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        t = t.clamp(min=self.min_t, max=1.0)
        r = torch.minimum(r.clamp(min=0.0, max=1.0), t)
        fm_mask = torch.rand(num_graphs, device=device) < self.same_time_probability
        return torch.where(fm_mask, t, r), t, fm_mask


class HamiltonianRiemannianMeanFlow(HamiltonianCFM):
    """Opt-in PyTorch RMF for DeePTB residual Hamiltonian endpoint models.

    DeePTB keeps its denoising convention: clean residual is at t=0 and the prior
    residual is at t=1.  The model still sees H0 + z_t and predicts the clean
    endpoint/full Hamiltonian expected by the existing loss stack.
    """

    model_in_loss = True

    def __init__(self, options: Optional[Dict[str, Any]], *, idp=None, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__(options, idp=idp, dtype=dtype, device=device)
        options = _opts(options)
        rmf = _opts(options.get("rmf_options"))
        meanflow = _opts(options.get("meanflow"))
        sampler = _opts(options.get("time_sampler"))
        sampler.update(_opts(rmf.get("time_sampler")))
        self.manifold = build_manifold(options.get("manifold", rmf.get("manifold", "euclidean")), rmf.get("manifold_options"))
        objective = str(rmf.get("objective", "eulerian_meanflow")).lower()
        if objective not in {"eulerian", "eulerian_meanflow", "meanflow"}:
            raise NotImplementedError("Hamiltonian RMF currently supports the finite-difference Eulerian objective.")
        prediction = str(rmf.get("prediction", "x1")).lower()
        if prediction not in {"x1", "endpoint"}:
            raise ValueError("Hamiltonian RMF currently supports x1/endpoint prediction only.")
        self.time_sampler = RMFTimeSampler(sampler, fallback=meanflow)
        self.meanflow_min_t = float(sampler.get("min_t", meanflow.get("min_t", 0.05)))
        self.meanflow_fd_eps = float(meanflow.get("fd_eps", rmf.get("fd_eps", 1.0e-3)))
        backend = str(meanflow.get("du_dt_backend", meanflow.get("jvp_backend", rmf.get("du_dt_backend", "finite_difference")))).lower()
        if backend != "finite_difference":
            raise NotImplementedError("Hamiltonian RMF supports finite_difference du/dt only.")
        self.meanflow_norm_p = float(meanflow.get("norm_p", rmf.get("norm_p", 0.0)))
        self.meanflow_norm_eps = float(meanflow.get("norm_eps", rmf.get("norm_eps", 0.01)))
        self.meanflow_aux_endpoint_weight = float(meanflow.get("aux_endpoint_weight", rmf.get("aux_endpoint_weight", 0.05)))
        self.meanflow_aux_boundary_v_weight = float(meanflow.get("aux_boundary_v_weight", rmf.get("aux_boundary_v_weight", 0.0)))
        self.meanflow_jvp_tangent = str(meanflow.get("jvp_tangent", rmf.get("jvp_tangent", "boundary"))).lower()
        if self.meanflow_jvp_tangent not in {"path", "boundary"}:
            raise ValueError("RMF jvp_tangent must be 'path' or 'boundary'.")
        self.flow_time_r_key = str(options.get("flow_time_r_key", "flow_time_r"))
        self.flow_time_t_key = str(options.get("flow_time_t_key", "flow_time_t"))
        self.flow_time_h_key = str(options.get("flow_time_h_key", "flow_time_h"))
        if self.enabled:
            log.info("Hamiltonian RMF enabled: manifold=%s sampler=%s", self.manifold.name, self.time_sampler.kind)

    def _write_times(self, data: AtomicDataDict.Type, r: torch.Tensor, t: torch.Tensor) -> None:
        data[self.flow_time_key] = t.detach()
        data[self.flow_time_t_key] = t.detach()
        data[self.flow_time_r_key] = r.detach()
        data[self.flow_time_h_key] = (t - r).detach()
        data["t"] = t.detach()
        data["r"] = r.detach()
        data["meanflow_h"] = (t - r).detach()

    def _sample_rt(self, *, num_graphs: int, device: torch.device, dtype: torch.dtype):
        return self.time_sampler.sample(num_graphs=num_graphs, device=device, dtype=dtype)

    def _path_state(self, clean: torch.Tensor, prior: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.manifold.geodesic_interpolate(clean, prior, t)

    def _noise_velocity(self, state: torch.Tensor, endpoint: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.manifold.project(state, -self.manifold.logmap(state, endpoint) / _time_view(t, state))

    def _prepare_component(self, data, target, t, h0_key: str, label: str):
        base = self._base_like(data, target, h0_key, label)
        clean = target - base if self.mode == "residual" else target
        prior = self._prior_like(clean, self.node_sigma if label == "node" else self.edge_sigma)
        state = self._path_state(clean, prior, t)
        current = base + state if self.mode == "residual" else state
        if self.detach_interpolated_h0:
            current = current.detach()
        return base, clean, prior, state, current, self._noise_velocity(state, clean, t)

    def prepare_batch(self, data: AtomicDataDict.Type, ref_data: AtomicDataDict.Type, *, r=None, t=None):
        if not self.enabled:
            raise RuntimeError("HamiltonianRiemannianMeanFlow.prepare_batch called while disabled")
        data, ref_data = data.copy(), ref_data.copy()
        node_target = ref_data.get(self.node_target_key)
        edge_target = ref_data.get(self.edge_target_key)
        if node_target is None and edge_target is None:
            raise KeyError("RMF requires node and/or edge Hamiltonian targets in ref_data.")
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
            data, t,
            node_count=None if node_target is None else node_target.shape[0],
            edge_count=None if edge_target is None else edge_target.shape[0],
        )
        node_r, edge_r = self._expand_graph_times(
            data, r,
            node_count=None if node_target is None else node_target.shape[0],
            edge_count=None if edge_target is None else edge_target.shape[0],
        )
        node_base = edge_base = node_clean = edge_clean = None
        node_state = edge_state = node_prior = edge_prior = None
        node_tv = edge_tv = None
        if node_target is not None:
            node_base, node_clean, node_prior, node_state, current, node_tv = self._prepare_component(data, node_target, node_t, self.node_h0_key, "node")
            data[self.node_h0_key] = current
            if self.overwrite_feature_keys:
                data[self.node_target_key] = current
        if edge_target is not None:
            edge_base, edge_clean, edge_prior, edge_state, current, edge_tv = self._prepare_component(data, edge_target, edge_t, self.edge_h0_key, "edge")
            data[self.edge_h0_key] = current
            if self.overwrite_feature_keys:
                data[self.edge_target_key] = current
        self._write_times(data, r, t)
        self._write_times(ref_data, r, t)
        return data, ref_data, RMFContext(r, t, fm_mask, node_r, node_t, edge_r, edge_t, node_base, edge_base, node_clean, edge_clean, node_state, edge_state, node_prior, edge_prior, node_tv, edge_tv)

    def _predict_clean(self, model, data, ctx: RMFContext, node_state, edge_state, *, r, t):
        model_data = data.copy()
        if node_state is not None:
            current = ctx.node_base + node_state if self.mode == "residual" else node_state
            model_data[self.node_h0_key] = current
            if self.overwrite_feature_keys:
                model_data[self.node_target_key] = current
        if edge_state is not None:
            current = ctx.edge_base + edge_state if self.mode == "residual" else edge_state
            model_data[self.edge_h0_key] = current
            if self.overwrite_feature_keys:
                model_data[self.edge_target_key] = current
        self._write_times(model_data, r, t)
        pred = model(model_data)
        node_x = edge_x = None
        if node_state is not None and self.node_target_key in pred:
            out = pred[self.node_target_key].to(node_state.device, node_state.dtype)
            node_x = out - ctx.node_base if self.mode == "residual" else out
        if edge_state is not None and self.edge_target_key in pred:
            out = pred[self.edge_target_key].to(edge_state.device, edge_state.dtype)
            edge_x = out - ctx.edge_base if self.mode == "residual" else out
        return pred, node_x, edge_x

    def _metric_loss(self, diff: torch.Tensor, mask: torch.Tensor, adaptive: bool = False):
        mask = mask.to(diff.device, diff.dtype)
        count = mask.sum().clamp_min(1.0)
        mse = (diff.square() * mask).sum() / count
        mae = (diff.abs() * mask).sum() / count
        loss = mse if self.loss_type == "mse" else 0.5 * (mae + torch.sqrt(mse + 1.0e-12))
        if adaptive and self.meanflow_norm_p > 0.0:
            loss = (mse.detach() + self.meanflow_norm_eps).pow(-self.meanflow_norm_p) * loss
        return loss, mse, mae

    def _component_loss(self, *, prefix, pred_x, boundary_x, clean, state, r, t, pred_x_eps, mask, weight, path_velocity):
        target_v = self._noise_velocity(state, clean, t)
        u = self._noise_velocity(state, pred_x, t)
        signed_dt = torch.where(t <= 1.0 - self.meanflow_fd_eps, t.new_full(t.shape, self.meanflow_fd_eps), t.new_full(t.shape, -self.meanflow_fd_eps))
        t_eps = (t + signed_dt).clamp(min=self.meanflow_min_t, max=1.0)
        signed_dt = t_eps - t
        tangent = self._noise_velocity(state, boundary_x, t) if self.meanflow_jvp_tangent == "boundary" and boundary_x is not None else path_velocity
        state_eps = self.manifold.expmap(state, _time_view(signed_dt, state) * tangent.detach())
        u_eps = self._noise_velocity(state_eps, pred_x_eps, t_eps)
        du_dt = ((u_eps - u.detach()) / _time_view(signed_dt, state)).detach()
        h = (t - r).reshape((-1,) + (1,) * (state.ndim - 1))
        velocity_loss, velocity_mse, velocity_mae = self._metric_loss(u + h * du_dt - target_v, mask, adaptive=True)
        endpoint_loss, endpoint_mse, endpoint_mae = self._metric_loss(pred_x - clean, mask)
        zero = endpoint_loss.new_zeros(())
        boundary_loss = zero
        if boundary_x is not None and self.meanflow_aux_boundary_v_weight > 0.0:
            boundary_loss, _, _ = self._metric_loss(self._noise_velocity(state, boundary_x, t) - target_v, mask, adaptive=True)
        total = weight * (velocity_loss + self.meanflow_aux_endpoint_weight * endpoint_loss + self.meanflow_aux_boundary_v_weight * boundary_loss)
        metrics = {
            f"{prefix}_loss": endpoint_loss.detach(),
            f"{prefix}_velocity_loss": velocity_loss.detach(),
            f"{prefix}_velocity_mse": velocity_mse.detach(),
            f"{prefix}_velocity_mae": velocity_mae.detach(),
            f"{prefix}_endpoint_loss": endpoint_loss.detach(),
            f"{prefix}_endpoint_mse": endpoint_mse.detach(),
            f"{prefix}_endpoint_mae": endpoint_mae.detach(),
            f"{prefix}_boundary_v_loss": boundary_loss.detach(),
        }
        return total, metrics

    def loss_with_model(self, model, data: AtomicDataDict.Type, ref_data: AtomicDataDict.Type, *, prefix="train", r=None, t=None):
        data, ref_data, ctx = self.prepare_batch(data, ref_data, r=r, t=t)
        main_pred, node_x, edge_x = self._predict_clean(model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.r, t=ctx.t)
        need_boundary = self.meanflow_jvp_tangent == "boundary" or self.meanflow_aux_boundary_v_weight > 0.0
        node_boundary = edge_boundary = None
        if need_boundary:
            _, node_boundary, edge_boundary = self._predict_clean(model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.t, t=ctx.t)
        node_eps = edge_eps = None
        if ctx.node_state is not None:
            tangent = self._noise_velocity(ctx.node_state, node_boundary, ctx.node_t) if need_boundary and node_boundary is not None else ctx.node_target_velocity
            dt = torch.where(ctx.node_t <= 1.0 - self.meanflow_fd_eps, ctx.node_t.new_full(ctx.node_t.shape, self.meanflow_fd_eps), ctx.node_t.new_full(ctx.node_t.shape, -self.meanflow_fd_eps))
            dt = (ctx.node_t + dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.node_t
            node_eps = self.manifold.expmap(ctx.node_state, _time_view(dt, ctx.node_state) * tangent.detach())
        if ctx.edge_state is not None:
            tangent = self._noise_velocity(ctx.edge_state, edge_boundary, ctx.edge_t) if need_boundary and edge_boundary is not None else ctx.edge_target_velocity
            dt = torch.where(ctx.edge_t <= 1.0 - self.meanflow_fd_eps, ctx.edge_t.new_full(ctx.edge_t.shape, self.meanflow_fd_eps), ctx.edge_t.new_full(ctx.edge_t.shape, -self.meanflow_fd_eps))
            dt = (ctx.edge_t + dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.edge_t
            edge_eps = self.manifold.expmap(ctx.edge_state, _time_view(dt, ctx.edge_state) * tangent.detach())
        graph_dt = torch.where(ctx.t <= 1.0 - self.meanflow_fd_eps, ctx.t.new_full(ctx.t.shape, self.meanflow_fd_eps), ctx.t.new_full(ctx.t.shape, -self.meanflow_fd_eps))
        t_eps = (ctx.t + graph_dt).clamp(min=self.meanflow_min_t, max=1.0)
        with torch.no_grad():
            _, node_x_eps, edge_x_eps = self._predict_clean(model, data, ctx, node_eps if node_eps is not None else ctx.node_state, edge_eps if edge_eps is not None else ctx.edge_state, r=ctx.r, t=t_eps)
        total = None
        state = {
            f"{prefix}_flow_r": ctx.r.detach().mean(),
            f"{prefix}_flow_t": ctx.t.detach().mean(),
            f"{prefix}_flow_h": (ctx.t - ctx.r).detach().mean(),
            f"{prefix}_flow_fm_frac": ctx.fm_mask.detach().float().mean(),
        }
        if ctx.node_clean is not None and node_x is not None:
            comp, metrics = self._component_loss(prefix=f"{prefix}_flow_onsite", pred_x=node_x, boundary_x=node_boundary, clean=ctx.node_clean, state=ctx.node_state, r=ctx.node_r, t=ctx.node_t, pred_x_eps=node_x_eps, mask=self._node_mask(data, node_x), weight=self.node_weight, path_velocity=ctx.node_target_velocity)
            total = comp if total is None else total + comp
            state.update(metrics)
            if prefix == "train" and self.compatible_loss_to_legacy_keys:
                state["train_onsite_loss"] = metrics[f"{prefix}_flow_onsite_endpoint_loss"]
        if ctx.edge_clean is not None and edge_x is not None:
            comp, metrics = self._component_loss(prefix=f"{prefix}_flow_hopping", pred_x=edge_x, boundary_x=edge_boundary, clean=ctx.edge_clean, state=ctx.edge_state, r=ctx.edge_r, t=ctx.edge_t, pred_x_eps=edge_x_eps, mask=self._edge_mask(data, edge_x), weight=self.edge_weight, path_velocity=ctx.edge_target_velocity)
            total = comp if total is None else total + comp
            state.update(metrics)
            if prefix == "train" and self.compatible_loss_to_legacy_keys:
                state["train_hopping_loss"] = metrics[f"{prefix}_flow_hopping_endpoint_loss"]
        if total is None:
            raise KeyError("RMF could not compute node or edge loss.")
        if torch.is_tensor(main_pred.get("mean_max_prob")):
            state["mean_max_prob"] = main_pred["mean_max_prob"].detach()
            if self.router_z_loss_coef > 0.0:
                total = total + self.router_z_loss_coef * main_pred["mean_max_prob"]
        if torch.is_tensor(main_pred.get("expert_load_cv")):
            state["expert_load_cv"] = main_pred["expert_load_cv"].detach()
        state[f"{prefix}_flow_loss"] = total.detach()
        self.last_state = state
        return total, state

    def loss(self, pred_data, ref_data, ctx):  # noqa: ARG002
        raise RuntimeError("HamiltonianRiemannianMeanFlow requires loss_with_model(model, data, ref_data).")

    def sample(self, model, data: AtomicDataDict.Type, *, num_steps: int):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        state = data.copy()
        node_base = self._sampling_base(state, self.node_h0_key, self.node_target_key, "node")
        edge_base = self._sampling_base(state, self.edge_h0_key, self.edge_target_key, "edge")
        if node_base is None and edge_base is None:
            raise KeyError("RMF sampling requires node and/or edge H0 features.")
        node_z = None if node_base is None else self._prior_like(torch.zeros_like(node_base), self.node_sigma)
        edge_z = None if edge_base is None else self._prior_like(torch.zeros_like(edge_base), self.edge_sigma)
        like = node_z if node_z is not None else edge_z
        num_graphs = self._num_graphs(state)
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=like.device, dtype=like.dtype)
        ctx = RMFContext(grid.new_zeros(num_graphs), grid.new_ones(num_graphs), torch.zeros(num_graphs, device=like.device, dtype=torch.bool), None, None, None, None, node_base, edge_base, None, None, node_z, edge_z, node_z, edge_z, None, None)
        for i in range(num_steps):
            t = torch.full((num_graphs,), float(grid[i].item()), device=like.device, dtype=like.dtype).clamp_min(self.meanflow_min_t)
            r = torch.full((num_graphs,), float(grid[i + 1].item()), device=like.device, dtype=like.dtype)
            ctx.r, ctx.t = r, t
            ctx.node_t, ctx.edge_t = self._expand_graph_times(state, t, node_count=None if node_z is None else node_z.shape[0], edge_count=None if edge_z is None else edge_z.shape[0])
            ctx.node_r, ctx.edge_r = self._expand_graph_times(state, r, node_count=None if node_z is None else node_z.shape[0], edge_count=None if edge_z is None else edge_z.shape[0])
            _, node_x, edge_x = self._predict_clean(model, state, ctx, node_z, edge_z, r=r, t=t)
            if node_z is not None:
                h = (ctx.node_t - ctx.node_r).reshape((-1,) + (1,) * (node_z.ndim - 1))
                node_z = self.manifold.expmap(node_z, h * self.manifold.logmap(node_z, node_x) / _time_view(ctx.node_t, node_z))
                ctx.node_state = node_z
            if edge_z is not None:
                h = (ctx.edge_t - ctx.edge_r).reshape((-1,) + (1,) * (edge_z.ndim - 1))
                edge_z = self.manifold.expmap(edge_z, h * self.manifold.logmap(edge_z, edge_x) / _time_view(ctx.edge_t, edge_z))
                ctx.edge_state = edge_z
        out = state.copy()
        if node_z is not None:
            out[self.node_h0_key] = node_base + node_z if self.mode == "residual" else node_z
            out[self.node_target_key] = out[self.node_h0_key]
        if edge_z is not None:
            out[self.edge_h0_key] = edge_base + edge_z if self.mode == "residual" else edge_z
            out[self.edge_target_key] = out[self.edge_h0_key]
        zero = torch.zeros(num_graphs, device=like.device, dtype=like.dtype)
        self._write_times(out, zero, zero)
        return out


def build_hamiltonian_flow(options: Optional[Dict[str, Any]], *, idp=None, dtype=torch.float32, device=torch.device("cpu")) -> HamiltonianCFM:
    options = dict(options or {})
    flow_type = str(options.get("type", options.get("objective", "cfm"))).lower()
    objective = str(options.get("objective", flow_type)).lower()
    if flow_type in _RMF_ALIASES or objective in _RMF_ALIASES:
        return HamiltonianRiemannianMeanFlow(options, idp=idp, dtype=dtype, device=device)
    return _BASE_BUILD_HAMILTONIAN_FLOW(options, idp=idp, dtype=dtype, device=device)
