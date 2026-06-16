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


@dataclass
class PixelMFContext:
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
        self.prior = str(options.get("prior", "zero")).lower().replace("-", "_")
        self._te_prior_names = {"te", "structured_te", "block_te", "te_like"}
        allowed_priors = {"zero", "gaussian", "residual_gaussian", *self._te_prior_names}
        if self.prior not in allowed_priors:
            raise ValueError(
                f"Unsupported flow_options.prior={self.prior!r}; "
                "use 'zero', 'gaussian', 'residual_gaussian', or 'te'."
            )

        self.node_sigma = float(options.get("node_sigma", 1.0))
        self.edge_sigma = float(options.get("edge_sigma", 1.0))
        self.residual_sigma_floor = float(options.get("residual_sigma_floor", 1.0e-6))
        self.te_prior_sigma = float(options.get("te_prior_sigma", 1.0))
        self.te_prior_mode = str(options.get("te_prior_mode", "irrep")).lower().replace("-", "_")
        if self.te_prior_mode == "type":
            self.te_prior_mode = "typewise"
        if self.te_prior_mode not in {"irrep", "block", "typewise"}:
            raise ValueError("flow_options.te_prior_mode must be 'irrep', 'block', or 'typewise'.")
        self.te_prior_per_graph = bool(options.get("te_prior_per_graph", True))

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
        self.apply_to_reference = bool(options.get("apply_to_reference", False))
        self.log_compatible_loss = bool(options.get("log_compatible_loss", True))
        self.log_train_compatible_loss = bool(
            options.get("log_train_compatible_loss", self.log_compatible_loss)
        )
        self.log_validation_compatible_loss = bool(
            options.get("log_validation_compatible_loss", self.log_compatible_loss)
        )
        self.compatible_loss_to_legacy_keys = bool(
            options.get("compatible_loss_to_legacy_keys", True)
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

    @staticmethod
    def _align_bool_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=like.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)
        if mask.shape[0] < like.shape[0]:
            pad = torch.ones(
                like.shape[0] - mask.shape[0],
                mask.shape[1],
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > like.shape[0]:
            mask = mask[: like.shape[0]]

        if mask.shape[-1] == 1:
            return mask.expand_as(like)
        if mask.shape[-1] < like.shape[-1]:
            pad = torch.ones(
                mask.shape[0],
                like.shape[-1] - mask.shape[-1],
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > like.shape[-1]:
            mask = mask[:, : like.shape[-1]]
        return mask.expand_as(like)

    def _prior_mask(
        self,
        data: Optional[AtomicDataDict.Type],
        like: torch.Tensor,
        label: Optional[str],
    ) -> torch.Tensor:
        mask = torch.ones_like(like, dtype=torch.bool, device=like.device)
        if data is None or self.idp is None or like.ndim < 2:
            return mask

        if label == "node":
            type_key = AtomicDataDict.ATOM_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_nrme", None)
            expert_key = "expert_node_mask"
        elif label == "edge":
            type_key = AtomicDataDict.EDGE_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_erme", None)
            expert_key = "expert_edge_mask"
        else:
            return mask

        types = data.get(type_key, None)
        if types is not None and mask_table is not None:
            row_types = types.to(device=like.device, dtype=torch.long).reshape(-1)
            if row_types.numel() > 0:
                row_types = row_types[: like.shape[0]]
                table = mask_table.to(device=like.device, dtype=torch.bool)
                row_types = row_types.clamp(min=0, max=max(table.shape[0] - 1, 0))
                type_mask = table.index_select(0, row_types)
                mask = mask & self._align_bool_mask(type_mask, like)

        expert_mask = data.get(expert_key, None)
        if expert_mask is not None:
            mask = mask & self._align_bool_mask(expert_mask, like)
        return mask

    def _row_graph_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> torch.Tensor:
        if count <= 0 or data is None:
            return torch.zeros(count, device=device, dtype=torch.long)

        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return torch.zeros(count, device=device, dtype=torch.long)
        batch = batch.to(device=device, dtype=torch.long).reshape(-1)

        if label == "node":
            if batch.numel() < count:
                batch = torch.cat([batch, batch.new_zeros(count - batch.numel())], dim=0)
            return batch[:count]

        if label == "edge":
            edge_index = data.get(_keys.EDGE_INDEX_KEY, None)
            if edge_index is None or edge_index.numel() == 0:
                return torch.zeros(count, device=device, dtype=torch.long)
            centers = edge_index[0].to(device=device, dtype=torch.long).reshape(-1)
            if centers.numel() < count:
                centers = torch.cat([centers, centers.new_zeros(count - centers.numel())], dim=0)
            centers = centers[:count].clamp(min=0, max=max(batch.numel() - 1, 0))
            return batch.index_select(0, centers)

        return torch.zeros(count, device=device, dtype=torch.long)

    def _row_type_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if count <= 0 or data is None:
            return None
        if label == "node":
            key = AtomicDataDict.ATOM_TYPE_KEY
        elif label == "edge":
            key = AtomicDataDict.EDGE_TYPE_KEY
        else:
            return None
        values = data.get(key, None)
        if values is None:
            return None
        values = values.to(device=device, dtype=torch.long).reshape(-1)
        if values.numel() < count:
            values = torch.cat([values, values.new_zeros(count - values.numel())], dim=0)
        return values[:count]

    def _te_irrep_slices(self, feature_dim: int) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        if self.idp is None:
            return None
        irreps = getattr(self.idp, "orbpair_irreps", None)
        if irreps is None:
            return None
        try:
            irreps = irreps.sort()[0].simplify()
        except Exception:
            try:
                irreps = irreps.simplify()
            except Exception:
                pass

        slices = []
        offset = 0
        try:
            for mul, ir in irreps:
                degree = int(getattr(ir, "l", 0))
                width = int(getattr(ir, "dim", 2 * degree + 1))
                for _ in range(int(mul)):
                    slices.append((offset, offset + width, degree))
                    offset += width
        except Exception:
            return None
        if offset != int(feature_dim):
            return None
        return tuple(slices)

    def _te_radius(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.te_prior_per_graph and graph_index is not None and graph_index.numel() == row_count:
            num_graphs = int(graph_index.max().detach().item()) + 1 if row_count > 0 else 1
            radius = torch.randn(num_graphs, 1, device=device, dtype=dtype).index_select(0, graph_index)
        else:
            radius = torch.randn(row_count, 1, device=device, dtype=dtype)
        return radius * active_dim.sqrt()

    def _block_structured_prior_like(
        self,
        like: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
    ) -> torch.Tensor:
        if like.ndim < 2:
            return torch.randn_like(like)
        mask_f = mask.to(device=like.device, dtype=like.dtype)
        raw = torch.randn_like(like) * mask_f
        norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
        direction = raw / norm
        active_dim = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
        radius = self._te_radius(
            like.shape[0], active_dim, graph_index, device=like.device, dtype=like.dtype
        )
        return direction * radius * mask_f

    def _apply_typewise_residual_scale(
        self,
        noise: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        slices: Tuple[Tuple[int, int, int], ...],
        *,
        reference_scale: bool,
    ) -> torch.Tensor:
        if self.te_prior_mode != "typewise" or not reference_scale:
            return noise
        type_index = self._row_type_index(data, noise.shape[0], label, noise.device)
        if type_index is None or noise.ndim < 2:
            return noise

        out = noise.clone()
        ref = reference.detach().to(device=noise.device, dtype=noise.dtype)
        mask_f = mask.to(device=noise.device, dtype=noise.dtype)
        for type_value in torch.unique(type_index).detach().cpu().tolist():
            rows = type_index == int(type_value)
            if not bool(rows.any().detach().cpu().item()):
                continue
            for start, end, _degree in slices:
                seg_mask = mask_f[rows, start:end]
                count = seg_mask.sum()
                if not bool((count > 0).detach().cpu().item()):
                    continue
                rms = ((ref[rows, start:end].square() * seg_mask).sum() / count).sqrt()
                if not bool(torch.isfinite(rms).detach().cpu().item()):
                    continue
                out[rows, start:end] = out[rows, start:end] * rms.clamp_min(
                    self.residual_sigma_floor
                )
        return out

    def _te_prior_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        reference_scale: bool = True,
    ) -> torch.Tensor:
        if like.numel() == 0:
            return torch.zeros_like(like)

        mask = self._prior_mask(data, like, label)
        slices = None if self.te_prior_mode == "block" else self._te_irrep_slices(like.shape[-1])
        if like.ndim < 2 or slices is None:
            slices = ((0, like.shape[-1], -1),) if like.ndim >= 2 else ((0, like.numel(), -1),)
            noise = self._block_structured_prior_like(like, mask, data, label)
        else:
            noise = torch.zeros_like(like)
            graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
            for start, end, _degree in slices:
                seg_mask = mask[:, start:end].to(device=like.device, dtype=like.dtype)
                raw = torch.randn(like.shape[0], end - start, device=like.device, dtype=like.dtype)
                raw = raw * seg_mask
                norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
                direction = raw / norm
                active_dim = seg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                radius = self._te_radius(
                    like.shape[0], active_dim, graph_index, device=like.device, dtype=like.dtype
                )
                noise[:, start:end] = direction * radius * seg_mask

        noise = self._apply_typewise_residual_scale(
            noise,
            like,
            mask,
            data,
            label,
            slices,
            reference_scale=reference_scale,
        )
        return noise * (float(sigma) * self.te_prior_sigma)

    def _prior_like(
        self,
        residual: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        reference_scale: bool = True,
    ) -> torch.Tensor:
        if self.prior == "zero":
            return torch.zeros_like(residual)
        if self.prior == "gaussian":
            return torch.randn_like(residual) * sigma
        if self.prior in self._te_prior_names:
            return self._te_prior_like(
                residual,
                sigma,
                data=data,
                label=label,
                reference_scale=reference_scale,
            )
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
            node_prior = self._prior_like(node_res, self.node_sigma, data=data, label="node")
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
            edge_prior = self._prior_like(edge_res, self.edge_sigma, data=data, label="edge")
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
            node_loss_detached = node_loss.detach()
            state["train_flow_onsite_loss"] = node_loss_detached
            state["train_onsite_loss"] = node_loss_detached

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
            edge_loss_detached = edge_loss.detach()
            state["train_flow_hopping_loss"] = edge_loss_detached
            state["train_hopping_loss"] = edge_loss_detached

        if total is None:
            raise KeyError(
                "CFM could not compute a loss because model outputs do not contain "
                f"`{self.node_target_key}` or `{self.edge_target_key}`."
            )
        if self.component_reduction == "global_elements":
            total = total_numerator / total_count.clamp_min(1.0)

        mean_max_prob = pred_data.get("mean_max_prob", None)
        if torch.is_tensor(mean_max_prob):
            state["mean_max_prob"] = mean_max_prob.detach()
            if self.router_z_loss_coef > 0.0:
                total = total + self.router_z_loss_coef * mean_max_prob
        expert_load_cv = pred_data.get("expert_load_cv", None)
        if torch.is_tensor(expert_load_cv):
            state["expert_load_cv"] = expert_load_cv.detach()

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
                node_current = node_current + self._prior_like(
                    node_current,
                    self.node_sigma,
                    data=state,
                    label="node",
                    reference_scale=False,
                )
            if edge_current is not None:
                edge_current = edge_current + self._prior_like(
                    edge_current,
                    self.edge_sigma,
                    data=state,
                    label="edge",
                    reference_scale=False,
                )

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
        super().__init__(options, idp=idp, dtype=dtype, device=device)
        options = dict(options or {})
        mf = dict(options.get("meanflow", options.get("pixel_meanflow", {})) or {})
        profile = str(mf.get("profile", options.get("meanflow_profile", "conservative"))).lower()
        if bool(mf.get("aggressive", options.get("meanflow_aggressive", False))):
            profile = "aggressive"
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
            mf.get("du_dt_backend", mf.get("jvp_backend", "finite_difference"))
        ).lower()
        if self.meanflow_du_dt_backend != "finite_difference":
            raise NotImplementedError(
                "DeePTB pixel MeanFlow currently supports finite_difference du/dt only."
            )
        self.meanflow_norm_eps = float(mf.get("norm_eps", 0.01))
        self.meanflow_norm_p = float(mf.get("norm_p", 1.0 if aggressive else 0.0))
        self.meanflow_aux_endpoint_weight = float(mf.get("aux_endpoint_weight", 0.05))
        self.meanflow_aux_boundary_v_weight = float(
            mf.get("aux_boundary_v_weight", 0.10 if aggressive else 0.0)
        )
        self.meanflow_jvp_tangent = str(mf.get("jvp_tangent", "boundary")).lower()
        if self.meanflow_jvp_tangent not in {"path", "boundary"}:
            raise ValueError("pixel_meanflow.jvp_tangent must be 'path' or 'boundary'.")

        self.flow_time_r_key = str(options.get("flow_time_r_key", "flow_time_r"))
        self.flow_time_t_key = str(options.get("flow_time_t_key", "flow_time_t"))
        self.flow_time_h_key = str(options.get("flow_time_h_key", "flow_time_h"))
        self.log_train_compatible_loss = bool(mf.get("log_train_compatible_loss", False))
        self.log_validation_compatible_loss = bool(
            mf.get("log_validation_compatible_loss", self.log_validation_compatible_loss)
        )

        if self.enabled:
            log.info(
                "Pixel MeanFlow enabled: profile=%s sampling=%s min_t=%.3g "
                "data_prop=%.3g du_dt=%s jvp_tangent=%s norm_p=%.3g aux_x=%.3g aux_v=%.3g",
                self.meanflow_profile,
                self.meanflow_time_sampling,
                self.meanflow_min_t,
                self.meanflow_data_proportion,
                self.meanflow_du_dt_backend,
                self.meanflow_jvp_tangent,
                self.meanflow_norm_p,
                self.meanflow_aux_endpoint_weight,
                self.meanflow_aux_boundary_v_weight,
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

    def _write_times(self, data: AtomicDataDict.Type, r: torch.Tensor, t: torch.Tensor) -> None:
        tt = t.detach()
        rr = r.detach()
        hh = (t - r).detach()
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
        if node_target is not None:
            node_target = node_target.to(device=device, dtype=dtype)
            node_base = self._base_like(data, node_target, self.node_h0_key, "node")
            node_clean = node_target - node_base if self.mode == "residual" else node_target
            node_prior = self._prior_like(node_clean, self.node_sigma, data=data, label="node")
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
            edge_prior = self._prior_like(edge_clean, self.edge_sigma, data=data, label="edge")
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
        self._write_times(model_data, r, t)
        pred = model(model_data)
        node_x = None
        if ctx.node_clean is not None and self.node_target_key in pred:
            node_pred = pred[self.node_target_key]
            node_x = node_pred - ctx.node_base if self.mode == "residual" else node_pred
        edge_x = None
        if ctx.edge_clean is not None and self.edge_target_key in pred:
            edge_pred = pred[self.edge_target_key]
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
        pred_x_eps: torch.Tensor,
        mask: torch.Tensor,
        weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t_view = self._view_time(comp_t, state_z)
        h_view = (comp_t - comp_r).reshape((-1,) + (1,) * (state_z.ndim - 1))
        target_v = prior - clean
        u = (state_z - pred_x) / t_view
        if self.meanflow_jvp_tangent == "boundary" and boundary_x is not None:
            tangent = (state_z - boundary_x) / t_view
        else:
            tangent = target_v
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
        _, node_x, edge_x = self._predict_clean(
            model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.r, t=ctx.t
        )
        need_boundary = (
            self.meanflow_jvp_tangent == "boundary"
            or self.meanflow_aux_boundary_v_weight > 0.0
        )
        node_x_boundary = edge_x_boundary = None
        if need_boundary:
            _, node_x_boundary, edge_x_boundary = self._predict_clean(
                model, data, ctx, ctx.node_state, ctx.edge_state, r=ctx.t, t=ctx.t
            )

        node_state_eps = edge_state_eps = None
        if ctx.node_state is not None:
            if self.meanflow_jvp_tangent == "boundary" and node_x_boundary is not None:
                tangent = (ctx.node_state - node_x_boundary) / self._view_time(ctx.node_t, ctx.node_state)
            else:
                tangent = ctx.node_prior - ctx.node_clean
            node_dt = torch.where(
                ctx.node_t <= 1.0 - self.meanflow_fd_eps,
                ctx.node_t.new_full(ctx.node_t.shape, self.meanflow_fd_eps),
                ctx.node_t.new_full(ctx.node_t.shape, -self.meanflow_fd_eps),
            )
            node_dt = (ctx.node_t + node_dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.node_t
            node_state_eps = ctx.node_state + self._view_time(node_dt, ctx.node_state) * tangent.detach()
        if ctx.edge_state is not None:
            if self.meanflow_jvp_tangent == "boundary" and edge_x_boundary is not None:
                tangent = (ctx.edge_state - edge_x_boundary) / self._view_time(ctx.edge_t, ctx.edge_state)
            else:
                tangent = ctx.edge_prior - ctx.edge_clean
            edge_dt = torch.where(
                ctx.edge_t <= 1.0 - self.meanflow_fd_eps,
                ctx.edge_t.new_full(ctx.edge_t.shape, self.meanflow_fd_eps),
                ctx.edge_t.new_full(ctx.edge_t.shape, -self.meanflow_fd_eps),
            )
            edge_dt = (ctx.edge_t + edge_dt).clamp(min=self.meanflow_min_t, max=1.0) - ctx.edge_t
            edge_state_eps = ctx.edge_state + self._view_time(edge_dt, ctx.edge_state) * tangent.detach()
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

        total = None
        state: Dict[str, torch.Tensor] = {
            f"{prefix}_flow_r": ctx.r.detach().mean(),
            f"{prefix}_flow_t": ctx.t.detach().mean(),
            f"{prefix}_flow_h": (ctx.t - ctx.r).detach().mean(),
            f"{prefix}_flow_fm_frac": ctx.fm_mask.detach().float().mean(),
        }
        if ctx.node_clean is not None and node_x is not None:
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
                mask=self._node_mask(data, node_x),
                weight=self.node_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
        if ctx.edge_clean is not None and edge_x is not None:
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
                mask=self._edge_mask(data, edge_x),
                weight=self.edge_weight,
            )
            total = comp_total if total is None else total + comp_total
            state.update(comp_state)
        if total is None:
            raise KeyError("Pixel MeanFlow could not compute a loss from configured node/edge targets.")
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
        node_z = None if node_base is None else self._prior_like(
            torch.zeros_like(node_base),
            self.node_sigma,
            data=state,
            label="node",
            reference_scale=False,
        )
        edge_z = None if edge_base is None else self._prior_like(
            torch.zeros_like(edge_base),
            self.edge_sigma,
            data=state,
            label="edge",
            reference_scale=False,
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


def build_hamiltonian_flow(
    options: Optional[Dict[str, Any]],
    *,
    idp: Any = None,
    dtype: Any = torch.float32,
    device: Any = torch.device("cpu"),
) -> HamiltonianCFM:
    options = dict(options or {})
    objective = str(options.get("objective", options.get("type", "cfm"))).lower()
    if objective in {"pixel_meanflow", "pixel_mean_flow", "pmf", "meanflow", "mean_flow"}:
        return HamiltonianPixelMeanFlow(options, idp=idp, dtype=dtype, device=device)
    return HamiltonianCFM(options, idp=idp, dtype=dtype, device=device)
