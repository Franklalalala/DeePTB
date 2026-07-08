from __future__ import annotations

"""Conditional Flow Matching utilities for Hamiltonian training.

This module is intentionally lightweight and trainer-side.  It does not require a
new DeePTB model class: at every training step it replaces the H0 node/edge
fields by an interpolated Hamiltonian state H_t and trains the existing model to
predict the clean converged Hamiltonian.  This mirrors the residual-CFM training
used by QHFlow/QHFlow2, but is adapted to DeePTB/NextHAM-style physical H0
features.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nnops import prior_physical
from dptb.nnops.flow_context import CFMContext, PixelMFContext, _to_torch_dtype
from dptb.nnops.flow_priors import (
    BasisOnsiteFamily,
    DFTBSKFamily,
    ExternalFamily,
    HaarDMFamily,
    OverlapHuckelFamily,
    PriorContext,
    build_prior_families,
    BASIS_ONSITE_NAMES,
    DFTBSK_NAMES,
    EXTERNAL_NAMES,
    HAAR_DM_NAMES,
    OVERLAP_HUCKEL_NAMES,
)
from dptb.nnops.layout import normalize_idp_mask_layout, project_uureal_to_like

log = logging.getLogger(__name__)


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
        # The TE aliases are feature-space approximations over masked
        # node_h0/edge_h0 rows. They do not materialize dense Tensor-Expansion
        # or Clebsch-Gordan products and should not be interpreted as strict
        # dense TE/CG priors.
        self.prior = str(options.get("prior", "zero")).lower().replace("-", "_")
        self._te_prior_names = {"te", "structured_te", "block_te", "te_like"}
        self._basis_prior_names = BASIS_ONSITE_NAMES
        self._overlap_huckel_prior_names = OVERLAP_HUCKEL_NAMES
        self._haar_dm_prior_names = HAAR_DM_NAMES
        self._external_prior_names = EXTERNAL_NAMES
        self._dftbsk_prior_names = DFTBSK_NAMES
        allowed_priors = {
            "zero",
            "gaussian",
            "residual_gaussian",
            *self._te_prior_names,
            *self._basis_prior_names,
            *self._overlap_huckel_prior_names,
            *self._haar_dm_prior_names,
            *self._external_prior_names,
            *self._dftbsk_prior_names,
        }
        if self.prior not in allowed_priors:
            raise ValueError(
                f"Unsupported flow_options.prior={self.prior!r}; "
                "use 'zero', 'gaussian', 'residual_gaussian', 'te', "
                "'basis_onsite', 'overlap_huckel', 'haar_dm', 'dftbsk', 'external', "
                "'dftb', 'xtb', or 'physical'."
            )

        self.node_sigma = float(options.get("node_sigma", 1.0))
        self.edge_sigma = float(options.get("edge_sigma", 1.0))
        self.residual_sigma_floor = float(options.get("residual_sigma_floor", 1.0e-6))
        self.te_prior_sigma = float(options.get("te_prior_sigma", 1.0))
        default_te_prior_mode = "block" if self.prior == "block_te" else "irrep"
        self.te_prior_mode = str(options.get("te_prior_mode", default_te_prior_mode)).lower().replace("-", "_")
        if self.te_prior_mode == "type":
            self.te_prior_mode = "typewise"
        if self.te_prior_mode not in {"irrep", "block", "typewise"}:
            raise ValueError("flow_options.te_prior_mode must be 'irrep', 'block', or 'typewise'.")
        self.te_prior_per_graph = bool(options.get("te_prior_per_graph", True))

        # Physical prior families each own their option keys (parsed in
        # from_options).  Every family is built once so the external family can
        # fall back to the on-the-fly DFTB-SK guess without re-parsing options.
        self._families = build_prior_families(options, idp)
        self.prior_family = None
        for _cls in (
            BasisOnsiteFamily,
            OverlapHuckelFamily,
            HaarDMFamily,
            ExternalFamily,
            DFTBSKFamily,
        ):
            if self.prior in _cls.NAMES:
                self.prior_family = self._families[_cls]
                break
        self._prior_ctx = PriorContext(self)

        # physical_prior_fallback is cross-cutting (it governs which family the
        # 'physical' prior degrades to), so it stays on the trainer.
        self.physical_prior_fallback = str(
            options.get("physical_prior_fallback", "basis_onsite")
        ).lower().replace("-", "_")
        if self.physical_prior_fallback not in {"basis_onsite", "zero", "error"}:
            raise ValueError(
                "flow_options.physical_prior_fallback must be 'basis_onsite', 'zero', or 'error'."
            )
        self.last_haar_candidate_index = -1
        self._basis_onsite_table_cache: Dict[
            Tuple[str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}
        self._basis_onsite_type_mean_cache: Dict[
            Tuple[str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}
        self.physical_prior_jitter_sigma = float(
            options.get("physical_prior_jitter_sigma", options.get("prior_jitter_sigma", 0.0))
        )
        self.physical_prior_jitter_reference_scale = bool(
            options.get("physical_prior_jitter_reference_scale", True)
        )
        self.physical_prior_jitter_edge_decay = float(
            options.get("physical_prior_jitter_edge_decay", 0.0)
        )

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
        self.log_validation_random_t_loss = bool(
            options.get("log_validation_random_t_loss", True)
        )
        self.log_validation_t0_loss = bool(
            options.get("log_validation_t0_loss", True)
        )
        self.log_validation_flow_euler_loss = bool(
            options.get("log_validation_flow_euler_loss", True)
        )
        self.log_train_compatible_loss = bool(
            options.get("log_train_compatible_loss", self.log_compatible_loss)
        )
        self.log_validation_compatible_loss = bool(
            options.get("log_validation_compatible_loss", self.log_compatible_loss)
        )
        self.compatible_loss_to_legacy_keys = bool(
            options.get("compatible_loss_to_legacy_keys", True)
        )
        if self.enabled:
            self.log_train_compatible_loss = True
            self.log_validation_compatible_loss = True
            self.compatible_loss_to_legacy_keys = True
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
        self._te_irrep_slices_cache: Dict[int, Optional[Tuple[Tuple[int, int, int], ...]]] = {}
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
    # Family-owned option accessors
    # ------------------------------------------------------------------
    # These options now live on the prior-family instances (single source of
    # truth).  The shared basis-onsite/Huckel/Haar helpers below and existing
    # callers still read them off the trainer, so expose read-only views.
    @property
    def basis_onsite_scale(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_scale

    @property
    def basis_onsite_missing_value(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_missing_value

    @property
    def basis_onsite_edge_value(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_edge_value

    @property
    def huckel_edge_energy_fallback(self) -> float:
        return self._families[OverlapHuckelFamily].huckel_edge_energy_fallback

    @property
    def huckel_strict_basis(self) -> bool:
        return self._families[OverlapHuckelFamily].huckel_strict_basis

    @property
    def haar_node_key(self) -> str:
        return self._families[HaarDMFamily].haar_node_key

    @property
    def haar_edge_key(self) -> str:
        return self._families[HaarDMFamily].haar_edge_key

    @property
    def haar_candidate_index(self) -> int:
        return self._families[HaarDMFamily].haar_candidate_index

    @property
    def haar_dm_strict(self) -> bool:
        return self._families[HaarDMFamily].haar_dm_strict

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
    def _align_bool_mask(mask: torch.Tensor, like: torch.Tensor, *, pad_value: bool = False) -> torch.Tensor:
        mask = mask.to(device=like.device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.reshape(1, 1)
        elif mask.ndim == 1:
            mask = mask.reshape(-1, 1)
        elif mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)

        fill = bool(pad_value)
        if mask.shape[0] < like.shape[0]:
            pad = torch.full(
                (like.shape[0] - mask.shape[0], mask.shape[1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > like.shape[0]:
            mask = mask[: like.shape[0]]

        if mask.shape[-1] == 1:
            while mask.ndim < like.ndim:
                mask = mask.unsqueeze(-1)
            return mask.expand_as(like)
        if mask.shape[-1] < like.shape[-1]:
            pad = torch.full(
                (mask.shape[0], like.shape[-1] - mask.shape[-1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > like.shape[-1]:
            mask = mask[:, : like.shape[-1]]
        while mask.ndim < like.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(like)

    def _project_raw_feature_table(self, table: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if table.ndim < 2 or table.shape[-1] == int(feature_dim) or self.idp is None:
            return table
        like = table.new_empty((table.shape[0], int(feature_dim)))
        table, _raw_mask = project_uureal_to_like(self.idp, table, like)
        return table

    def _coerce_prior_source(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
    ) -> torch.Tensor:
        if not torch.is_tensor(source):
            source = torch.as_tensor(source, device=like.device)
        if torch.is_complex(source):
            log.warning("External %s prior `%s` is complex; only the real part is used.", label, key)
            source = source.real
        source = source.to(device=like.device, dtype=like.dtype)
        if source.ndim == 1 and like.ndim >= 2:
            source = source.unsqueeze(0)
        source, _raw_mask = project_uureal_to_like(self.idp, source, like)
        if source.shape == like.shape:
            return source
        if source.numel() == like.numel():
            return source.reshape_as(like)
        raise ValueError(
            f"External {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
            f"does not match target shape {tuple(like.shape)}."
        )

    def _basis_onsite_energy(self, symbol: str, orbital: str) -> float:
        return prior_physical.basis_onsite_energy(
            symbol, orbital, missing=self.basis_onsite_missing_value
        )

    @staticmethod
    def _orbital_l(orbital: str) -> int:
        return prior_physical.orbital_l(orbital)

    def _basis_onsite_table(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        cache_key = (str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._basis_onsite_table_cache:
            return self._basis_onsite_table_cache[cache_key]

        idp = self.idp
        if idp is None:
            self._basis_onsite_table_cache[cache_key] = None
            return None
        table = prior_physical.basis_onsite_table(
            idp,
            device=like.device,
            dtype=like.dtype,
            scale=self.basis_onsite_scale,
            missing=self.basis_onsite_missing_value,
        )
        if table is None:
            self._basis_onsite_table_cache[cache_key] = None
            return None

        table, _raw_mask = project_uureal_to_like(self.idp, table, like)
        if table.ndim < 2 or table.shape[-1] != like.shape[-1]:
            self._basis_onsite_table_cache[cache_key] = None
            return None
        self._basis_onsite_table_cache[cache_key] = table
        return table

    def _basis_onsite_type_mean(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        cache_key = (str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._basis_onsite_type_mean_cache:
            return self._basis_onsite_type_mean_cache[cache_key]

        table = self._basis_onsite_table(like)
        if table is None:
            self._basis_onsite_type_mean_cache[cache_key] = None
            return None
        mean = prior_physical.basis_onsite_type_mean(
            table, fallback=self.huckel_edge_energy_fallback
        )
        self._basis_onsite_type_mean_cache[cache_key] = mean
        return mean

    def _huckel_edge_energy_like(
        self,
        like: torch.Tensor,
        *,
        data: Optional[AtomicDataDict.Type],
    ) -> torch.Tensor:
        fallback = torch.full(
            (like.shape[0],),
            float(self.huckel_edge_energy_fallback),
            device=like.device,
            dtype=like.dtype,
        )
        type_mean = self._basis_onsite_type_mean(like)
        if type_mean is None:
            if self.huckel_strict_basis:
                raise ValueError(
                    "flow_options.prior='overlap_huckel' requires an OrbitalMapper-like idp "
                    "with basis_to_full_basis/orbpair_maps so edge energies can use basis onsite levels."
                )
            return fallback
        type_mean = type_mean.to(device=like.device, dtype=like.dtype)

        if data is None or _keys.EDGE_INDEX_KEY not in data or AtomicDataDict.ATOM_TYPE_KEY not in data:
            if self.huckel_strict_basis:
                raise KeyError(
                    "flow_options.prior='overlap_huckel' requires edge_index and atom_types "
                    "to map overlap rows to basis-derived endpoint energies."
                )
            valid = type_mean.abs() > 0
            if bool(valid.any().item()):
                fallback.fill_(float(type_mean[valid].mean().detach().item()))
            return fallback

        edge_index = data[_keys.EDGE_INDEX_KEY].to(device=like.device, dtype=torch.long)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].to(device=like.device, dtype=torch.long).reshape(-1)
        if edge_index.ndim != 2 or edge_index.shape[0] < 2 or atom_types.numel() == 0:
            if self.huckel_strict_basis:
                raise ValueError(
                    "flow_options.prior='overlap_huckel' expects edge_index with shape [2, n_edge] "
                    "and non-empty atom_types."
                )
            return fallback

        count = int(like.shape[0])
        energy = prior_physical.huckel_edge_energy(
            type_mean,
            edge_index,
            atom_types,
            count,
            fallback=float(self.huckel_edge_energy_fallback),
            strict=self.huckel_strict_basis,
        )
        return energy.reshape((count,) + (1,) * (like.ndim - 1))

    def _absolute_to_flow_prior(
        self,
        absolute_prior: torch.Tensor,
        like: torch.Tensor,
        *,
        base: Optional[torch.Tensor],
    ) -> torch.Tensor:
        absolute_prior = absolute_prior.to(device=like.device, dtype=like.dtype)
        if self.mode != "residual":
            return absolute_prior
        if base is None:
            base = torch.zeros_like(like)
        return absolute_prior - base.to(device=like.device, dtype=like.dtype)

    def _haar_candidate_axis(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
    ) -> Optional[Tuple[int, int]]:
        """Resolve ``(candidate_axis, candidate_count)`` for a Haar-DM source.

        Returns ``None`` when ``source`` already matches ``like`` (no candidate
        axis). Raises ValueError for any rank/shape that is neither the target
        shape nor a single-candidate-axis expansion of it. Only shapes are read,
        so ``source`` need not be on ``like``'s device/dtype.
        """
        if source.shape == like.shape:
            return None
        if source.ndim != like.ndim + 1:
            raise ValueError(
                f"Haar-DM {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
                f"must match target shape {tuple(like.shape)} or include one candidate axis."
            )
        if source.shape[0] == like.shape[0] and source.shape[-1] == like.shape[-1]:
            candidate_axis, candidate_count = 1, int(source.shape[1])
        elif source.shape[1] == like.shape[0] and source.shape[-1] == like.shape[-1]:
            candidate_axis, candidate_count = 0, int(source.shape[0])
        else:
            raise ValueError(
                f"Haar-DM {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
                f"is incompatible with target shape {tuple(like.shape)}."
            )
        if candidate_count <= 0:
            raise ValueError(f"Haar-DM prior `{key}` has zero candidates.")
        return candidate_axis, candidate_count

    def _haar_candidate_count(
        self,
        source: Optional[torch.Tensor],
        like: Optional[torch.Tensor],
        *,
        key: str,
        label: Optional[str],
    ) -> Optional[int]:
        """Candidate-axis size for ``source`` against ``like`` (``None`` if absent)."""
        if source is None or like is None:
            return None
        if not torch.is_tensor(source):
            source = torch.as_tensor(source)
        axis_info = self._haar_candidate_axis(source, like, key=key, label=label)
        return None if axis_info is None else axis_info[1]

    def _resolve_haar_candidate_idx(
        self,
        data: Optional[AtomicDataDict.Type],
        *,
        node_like: Optional[torch.Tensor],
        edge_like: Optional[torch.Tensor],
    ) -> Optional[int]:
        """Draw one Haar-DM candidate index shared by the node and edge priors.

        The precomputed node/edge candidates are per-candidate coherent -- index
        ``i`` of ``haar_node_key`` and index ``i`` of ``haar_edge_key`` encode the
        same Haar density matrix ``D_haar`` (so ``DSD=spin*D`` and ``Tr(DS)=nelec``
        hold per candidate). Node and edge must therefore select the *same*
        candidate; drawing independently in the node and edge calls would break
        the density-matrix semantics whenever the candidate axis K>1. The index is
        threaded explicitly to both ``_prior_like`` calls rather than stashed on an
        instance attribute so concurrent node/edge preparation cannot desync.

        Returns ``None`` when the active prior is not Haar-DM or when neither
        source carries a candidate axis (2D sources are selected as-is
        downstream). ``haar_candidate_index >= 0`` is honored as a fixed override.
        """
        if self.prior not in self._haar_dm_prior_names or data is None:
            return None
        node_count = self._haar_candidate_count(
            data.get(self.haar_node_key, None),
            node_like,
            key=self.haar_node_key,
            label="node",
        )
        edge_count = self._haar_candidate_count(
            data.get(self.haar_edge_key, None),
            edge_like,
            key=self.haar_edge_key,
            label="edge",
        )
        if (
            node_count is not None
            and edge_count is not None
            and node_count != edge_count
        ):
            raise ValueError(
                f"Haar-DM node prior `{self.haar_node_key}` has {node_count} candidates "
                f"but edge prior `{self.haar_edge_key}` has {edge_count}; the node and "
                "edge candidate axes must match so the same Haar density matrix is "
                "selected for both."
            )
        count = node_count if node_count is not None else edge_count
        if count is None:
            return None
        if self.haar_candidate_index >= 0:
            if self.haar_candidate_index >= count:
                raise ValueError(
                    f"flow_options.haar_candidate_index={self.haar_candidate_index} "
                    f"is outside the Haar-DM candidate count {count}."
                )
            return int(self.haar_candidate_index)
        device = (
            node_like.device
            if node_like is not None
            else edge_like.device
            if edge_like is not None
            else torch.device("cpu")
        )
        return int(torch.randint(count, (), device=device).item())

    def _select_haar_candidate(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
        candidate_idx: Optional[int] = None,
    ) -> torch.Tensor:
        source = torch.as_tensor(source, device=like.device, dtype=like.dtype)
        self.last_haar_candidate_index = -1
        axis_info = self._haar_candidate_axis(source, like, key=key, label=label)
        if axis_info is None:
            return source
        candidate_axis, candidate_count = axis_info
        if candidate_idx is not None:
            # Batch-resolved shared index (see _resolve_haar_candidate_idx).
            idx = int(candidate_idx)
            if idx < 0 or idx >= candidate_count:
                raise ValueError(
                    f"Haar-DM {label or 'state'} prior `{key}` candidate index {idx} "
                    f"is outside candidate count {candidate_count}."
                )
        elif self.haar_candidate_index >= 0:
            if self.haar_candidate_index >= candidate_count:
                raise ValueError(
                    f"flow_options.haar_candidate_index={self.haar_candidate_index} "
                    f"is outside `{key}` candidate count {candidate_count}."
                )
            idx = self.haar_candidate_index
        else:
            idx = int(torch.randint(candidate_count, (), device=like.device).item())
        self.last_haar_candidate_index = idx
        return source[:, idx, ...] if candidate_axis == 1 else source[idx, ...]

    def _physical_prior_jitter_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
    ) -> torch.Tensor:
        if self.physical_prior_jitter_sigma <= 0.0:
            return torch.zeros_like(like)
        mask = self._prior_mask(data, like, label).to(dtype=like.dtype)
        noise = torch.randn_like(like) * mask
        if self.physical_prior_jitter_reference_scale and like.ndim >= 2:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            row_rms = ((like.detach().square() * mask).sum(dim=-1, keepdim=True) / denom).sqrt()
            noise = noise * row_rms.clamp_min(float(self.residual_sigma_floor))
        if (
            label == "edge"
            and self.physical_prior_jitter_edge_decay > 0.0
            and data is not None
            and _keys.EDGE_LENGTH_KEY in data
        ):
            edge_lengths = data[_keys.EDGE_LENGTH_KEY].to(device=like.device, dtype=like.dtype)
            edge_lengths = edge_lengths.reshape(-1)
            if int(edge_lengths.shape[0]) == int(like.shape[0]):
                view_shape = (edge_lengths.shape[0],) + (1,) * (like.ndim - 1)
                noise = noise * torch.exp(
                    -edge_lengths.reshape(view_shape) / self.physical_prior_jitter_edge_decay
                )
        return noise * (float(sigma) * self.physical_prior_jitter_sigma)

    def _physical_prior_like(
        self,
        residual: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        base: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # basis_onsite / overlap_huckel / external / dftbsk all resolve through a
        # family instance; only the cross-family fallback chain and the
        # fail-closed errors stay here.
        ctx = self._prior_ctx
        absolute_prior = self.prior_family.absolute_prior_like(
            residual, data=data, label=label, ctx=ctx
        )

        # The external family may additionally try an on-the-fly DFTB-SK guess.
        if absolute_prior is None and self.prior in self._external_prior_names:
            absolute_prior = self._families[DFTBSKFamily].absolute_prior_like(
                residual, data=data, label=label, ctx=ctx
            )

        # Silent basis_onsite fallback is now reserved for prior='physical'
        # (governed by physical_prior_fallback); dftb/xtb/sk/nnsk fail closed.
        if (
            absolute_prior is None
            and self.prior == "physical"
            and self.physical_prior_fallback == "basis_onsite"
        ):
            absolute_prior = self._families[BasisOnsiteFamily].absolute_prior_like(
                residual, data=data, label=label, ctx=ctx
            )

        if absolute_prior is None:
            absolute_prior = self._physical_prior_absent(residual, data=data, label=label)

        prior = self._absolute_to_flow_prior(absolute_prior, residual, base=base)
        return prior + self._physical_prior_jitter_like(
            residual,
            sigma,
            data=data,
            label=label,
        )

    def _physical_prior_absent(
        self,
        residual: torch.Tensor,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
    ) -> torch.Tensor:
        """No family produced an absolute prior: fail closed or degrade to zeros."""
        if self.prior in self._basis_prior_names:
            raise ValueError(
                "flow_options.prior='basis_onsite' requires an OrbitalMapper-like idp "
                "with basis_to_full_basis/orbpair_maps and node atom_types."
            )
        if self.prior in self._dftbsk_prior_names:
            raise ValueError(
                "flow_options.prior='dftbsk' requires a successful on-the-fly "
                "DFTB-SK initialization; set prior_skdata/dftb_skdata to a "
                "Slater-Koster directory or formatted .pth."
            )
        if self.prior in self._overlap_huckel_prior_names:
            raise ValueError(
                "flow_options.prior='overlap_huckel' requires basis onsite levels "
                "and edge overlap features; set huckel_edge_overlap_key or precompute overlap."
            )
        # External family (external/dftb/xtb/physical/sk/nnsk/dftb_xtb).
        external = self._families[ExternalFamily]
        if self.physical_prior_fallback == "zero" or not external.external_prior_strict:
            return torch.zeros_like(residual)
        keys = ", ".join(external.candidate_keys(label)[:8])
        raise KeyError(
            f"flow_options.prior={self.prior!r} did not find an external "
            f"{label or 'state'} prior. Tried keys: {keys}."
        )

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
            table = mask_table.to(device=like.device, dtype=torch.bool)
            if table.ndim == 0:
                table = table.reshape(1, 1)
            elif table.ndim == 1:
                table = table.reshape(-1, 1)
            elif table.ndim > 2:
                table = table.reshape(table.shape[0], -1)
            table = self._project_raw_feature_table(table, like.shape[-1])

            type_mask = torch.zeros(
                like.shape[0],
                table.shape[1],
                device=like.device,
                dtype=torch.bool,
            )
            if table.shape[0] > 0:
                row_types = types.to(device=like.device, dtype=torch.long).reshape(-1)
                take = min(int(row_types.numel()), int(like.shape[0]))
                if take > 0:
                    raw_types = row_types[:take]
                    valid = (raw_types >= 0) & (raw_types < table.shape[0])
                    valid_rows = torch.arange(take, device=like.device, dtype=torch.long)[valid]
                    if valid_rows.numel() > 0:
                        type_mask[valid_rows] = table.index_select(0, raw_types[valid])
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
        feature_dim = int(feature_dim)
        cache = getattr(self, "_te_irrep_slices_cache", None)
        if isinstance(cache, dict) and feature_dim in cache:
            return cache[feature_dim]

        def _remember(value):
            if isinstance(cache, dict):
                cache[feature_dim] = value
            return value

        if self.idp is None:
            return _remember(None)
        irreps = getattr(self.idp, "orbpair_irreps", None)
        if irreps is None:
            get_irreps = getattr(self.idp, "get_irreps", None)
            if not callable(get_irreps):
                return _remember(None)
            try:
                irreps = get_irreps()
            except Exception:
                return _remember(None)
            if irreps is None:
                return _remember(None)
        # Feature rows follow OrbitalMapper/orbpair_maps order. Sorting irreps
        # changes contiguous feature spans and breaks mask/typewise raw-slice priors.
        raw_irreps = irreps

        slices = []
        offset = 0
        try:
            for mul, ir in raw_irreps:
                degree = int(getattr(ir, "l", 0))
                width = int(getattr(ir, "dim", 2 * degree + 1))
                for _ in range(int(mul)):
                    slices.append((offset, offset + width, degree))
                    offset += width
        except Exception:
            return _remember(None)
        if offset != feature_dim:
            raw_mask = getattr(self.idp, "mask_uureal", None)
            if raw_mask is None:
                return _remember(None)
            raw_mask = raw_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            if raw_mask.numel() != offset:
                return _remember(None)
            if int(raw_mask.sum().item()) != feature_dim:
                return _remember(None)
            projected_slices = []
            compressed_offset = 0
            for start, end, degree in slices:
                kept = int(raw_mask[start:end].sum().item())
                if kept <= 0:
                    continue
                projected_slices.append((compressed_offset, compressed_offset + kept, degree))
                compressed_offset += kept
            if compressed_offset != feature_dim:
                return _remember(None)
            return _remember(tuple(projected_slices))
        return _remember(tuple(slices))

    def _te_radius(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        if self.te_prior_per_graph and graph_index is not None and graph_index.numel() == row_count:
            if num_graphs is None:
                num_graphs = int(graph_index.max().detach().item()) + 1 if row_count > 0 else 1
            radius = torch.randn(num_graphs, 1, device=device, dtype=dtype).index_select(0, graph_index)
        else:
            radius = torch.randn(row_count, 1, device=device, dtype=dtype)
        return radius * active_dim.sqrt()

    def _te_radius_for_prior(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        try:
            return self._te_radius(
                row_count,
                active_dim,
                graph_index,
                device=device,
                dtype=dtype,
                num_graphs=num_graphs,
            )
        except TypeError as exc:
            if "num_graphs" not in str(exc):
                raise
            return self._te_radius(
                row_count,
                active_dim,
                graph_index,
                device=device,
                dtype=dtype,
            )

    def _block_structured_prior_like(
        self,
        like: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        *,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        if like.ndim < 2:
            return torch.randn_like(like)
        mask_f = mask.to(device=like.device, dtype=like.dtype)
        raw = torch.randn_like(like) * mask_f
        norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
        direction = raw / norm
        active_dim = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
        radius = self._te_radius_for_prior(
            like.shape[0],
            active_dim,
            graph_index,
            device=like.device,
            dtype=like.dtype,
            num_graphs=num_graphs,
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
        _types, inverse = torch.unique(type_index, sorted=True, return_inverse=True)
        num_types = int(_types.numel())
        if num_types == 0:
            return out

        for start, end, _degree in slices:
            seg_mask = mask_f[:, start:end]
            row_count = seg_mask.sum(dim=-1)
            row_square_sum = (ref[:, start:end].square() * seg_mask).sum(dim=-1)

            type_count = torch.zeros(
                num_types, device=noise.device, dtype=noise.dtype
            ).scatter_add_(0, inverse, row_count)
            type_square_sum = torch.zeros_like(type_count).scatter_add_(
                0, inverse, row_square_sum
            )

            rms = torch.sqrt(type_square_sum / type_count.clamp_min(1.0))
            valid = (type_count > 0) & torch.isfinite(rms)
            scale_by_type = torch.where(
                valid,
                rms.clamp_min(self.residual_sigma_floor),
                torch.ones_like(rms),
            )
            out[:, start:end] = (
                out[:, start:end] * scale_by_type.index_select(0, inverse).unsqueeze(-1)
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
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        if like.numel() == 0:
            return torch.zeros_like(like)

        mask = self._prior_mask(data, like, label)
        if self.te_prior_mode == "block":
            slices = None
        else:
            if like.ndim < 2:
                raise ValueError(
                    f"flow_options.te_prior_mode={self.te_prior_mode!r} requires "
                    "DeePTB RME feature rows with rank >= 2; use te_prior_mode='block' "
                    "for unstructured tensors."
                )
            slices = self._te_irrep_slices(like.shape[-1])
            if slices is None:
                raise ValueError(
                    f"flow_options.te_prior_mode={self.te_prior_mode!r} requires "
                    "idp.orbpair_irreps raw feature spans to match "
                    f"{label or 'unknown'} feature_dim={like.shape[-1]}; "
                    "use te_prior_mode='block' for whole-row structured noise."
                )

        if like.ndim < 2 or slices is None:
            slices = ((0, like.shape[-1], -1),) if like.ndim >= 2 else ((0, like.numel(), -1),)
            noise = self._block_structured_prior_like(
                like,
                mask,
                data,
                label,
                num_graphs=num_graphs,
            )
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
                radius = self._te_radius_for_prior(
                    like.shape[0],
                    active_dim,
                    graph_index,
                    device=like.device,
                    dtype=like.dtype,
                    num_graphs=num_graphs,
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
        base: Optional[torch.Tensor] = None,
        reference_scale: bool = True,
        num_graphs: Optional[int] = None,
        candidate_idx: Optional[int] = None,
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
                num_graphs=num_graphs,
            )
        if self.prior in self._haar_dm_prior_names:
            absolute_prior = self.prior_family.absolute_prior_like(
                residual,
                data=data,
                label=label,
                ctx=self._prior_ctx,
                candidate_idx=candidate_idx,
            )
            if absolute_prior is None:
                return torch.zeros_like(residual)
            return self._absolute_to_flow_prior(absolute_prior, residual, base=base)
        if self.prior_family is not None:
            return self._physical_prior_like(
                residual,
                sigma,
                data=data,
                label=label,
                base=base,
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
        # One shared Haar-DM candidate for node+edge; None for every other prior.
        haar_candidate_idx = self._resolve_haar_candidate_idx(
            data, node_like=node_target, edge_like=edge_target
        )

        if node_target is not None:
            node_target = node_target.to(device=device, dtype=dtype)
            node_base = self._base_like(data, node_target, self.node_h0_key, "node")
            node_res = node_target - node_base
            node_prior = self._prior_like(
                node_res,
                self.node_sigma,
                data=data,
                label="node",
                base=node_base,
                num_graphs=num_graphs,
                candidate_idx=haar_candidate_idx,
            )
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
            edge_prior = self._prior_like(
                edge_res,
                self.edge_sigma,
                data=data,
                label="edge",
                base=edge_base,
                num_graphs=num_graphs,
                candidate_idx=haar_candidate_idx,
            )
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
        return normalize_idp_mask_layout(self.idp, mask, pred, label="node idp mask")

    def _edge_mask(self, data: AtomicDataDict.Type, pred: torch.Tensor) -> torch.Tensor:
        if self.idp is None or AtomicDataDict.EDGE_TYPE_KEY not in data:
            return torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        edge_types = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
        mask = self.idp.mask_to_erme.to(device=edge_types.device)[edge_types]
        if "expert_edge_mask" in data:
            mask = mask & data["expert_edge_mask"].to(device=mask.device).unsqueeze(-1)
        return normalize_idp_mask_layout(self.idp, mask, pred, label="edge idp mask")

    def _project_loss_layout(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, _raw_mask = project_uureal_to_like(self.idp, pred, target)
        if pred.shape != target.shape:
            raise ValueError(
                "prediction layout does not match target layout; "
                "check nextham_uureal_mask/mask_uureal propagation."
            )
        mask = normalize_idp_mask_layout(self.idp, mask, pred, label="loss idp mask")
        return pred, mask

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

    @staticmethod
    def _compatible_clean_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        component: str,
    ) -> Dict[str, torch.Tensor]:
        """Collect non-CFM HamilLossAbs reductions from an already aligned diff."""
        with torch.no_grad():
            diff = diff.detach()
            mask_f = mask.to(device=diff.device, dtype=diff.dtype)
            abs_sum = (diff.abs() * mask_f).sum()
            square_sum = (diff.square() * mask_f).sum()
            count = mask_f.sum()
        return {
            f"{component}_l1_sum": abs_sum.detach(),
            f"{component}_mse_sum": square_sum.detach(),
            f"{component}_count": count.detach(),
        }

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
            pred, mask = self._project_loss_layout(pred, mask, target)
            node_diff = pred - target
            node_weights = self._time_weight(ctx.node_t)
            node_loss, node_numerator, node_count = self._metric_stats(
                node_diff, mask, self.loss_type, node_weights
            )
            if self.log_train_compatible_loss or self.log_validation_compatible_loss:
                state.setdefault("_compatible_clean_stats", {}).update(
                    self._compatible_clean_stats(node_diff, mask, "onsite")
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
            pred, mask = self._project_loss_layout(pred, mask, target)
            edge_diff = pred - target
            edge_weights = self._time_weight(ctx.edge_t)
            edge_loss, edge_numerator, edge_count = self._metric_stats(
                edge_diff, mask, self.loss_type, edge_weights
            )
            if self.log_train_compatible_loss or self.log_validation_compatible_loss:
                state.setdefault("_compatible_clean_stats", {}).update(
                    self._compatible_clean_stats(edge_diff, mask, "hopping")
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
            # One shared Haar-DM candidate for node+edge; None for every other prior.
            haar_candidate_idx = self._resolve_haar_candidate_idx(
                state, node_like=node_current, edge_like=edge_current
            )
            if node_current is not None:
                node_current = node_current + self._prior_like(
                    node_current,
                    self.node_sigma,
                    data=state,
                    label="node",
                    base=node_current,
                    reference_scale=False,
                    candidate_idx=haar_candidate_idx,
                )
            if edge_current is not None:
                edge_current = edge_current + self._prior_like(
                    edge_current,
                    self.edge_sigma,
                    data=state,
                    label="edge",
                    base=edge_current,
                    reference_scale=False,
                    candidate_idx=haar_candidate_idx,
                )

        like = node_current if node_current is not None else edge_current
        num_graphs = self._num_graphs(state)
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            cur_t = float(step) * dt
            graph_t = torch.full((num_graphs,), cur_t, device=like.device, dtype=like.dtype)
            carry_keys = (
                "expert_edge_mask",
                "expert_node_mask",
                "expert_idx",
            )
            carried = {key: state[key] for key in carry_keys if key in state}
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
                endpoint, _raw_mask = project_uureal_to_like(self.idp, endpoint, node_current)
                node_current = node_current + dt * (endpoint - node_current) / denom
            if edge_current is not None:
                endpoint = prediction[self.edge_target_key]
                endpoint, _raw_mask = project_uureal_to_like(self.idp, endpoint, edge_current)
                edge_current = edge_current + dt * (endpoint - edge_current) / denom
            state = prediction.copy()
            state.update(carried)

        if node_current is not None:
            state[self.node_h0_key] = node_current
            state[self.node_target_key] = node_current
        if edge_current is not None:
            state[self.edge_h0_key] = edge_current
            state[self.edge_target_key] = edge_current
        state[self.flow_time_key] = torch.ones(num_graphs, device=like.device, dtype=like.dtype)
        return state


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


def configure_jvp_friendly_backends(flow_options: Optional[Dict[str, Any]]) -> bool:
    """Best-effort process-level prep for the pixel-meanflow jvp backend.

    TorchScript-compiled e3nn modules (SphericalHarmonics first) reject the
    storageless dual tensors that torch.func.jvp propagates, so e3nn's
    jit_mode must be 'eager' before any model is built. No-op unless the
    resolved flow objective is meanflow-family with du_dt_backend=jvp.
    Returns True if e3nn was switched.
    """
    options = dict(flow_options or {})
    if not options.get("enabled", False):
        return False
    objective = str(options.get("objective", options.get("type", "cfm"))).lower()
    if objective not in {"pixel_meanflow", "pixel_mean_flow", "pmf", "meanflow", "mean_flow"}:
        return False
    mf = dict(options.get("meanflow", options.get("pixel_meanflow", {})) or {})
    backend = (
        str(mf.get("du_dt_backend", mf.get("jvp_backend", "finite_difference")))
        .lower()
        .replace("-", "_")
    )
    if backend != "jvp":
        return False
    try:
        import e3nn

        e3nn.set_optimization_defaults(jit_mode="eager")
    except Exception as exc:
        log.warning(
            "pixel_meanflow.du_dt_backend=jvp requested but e3nn could not be "
            "switched to eager jit_mode (%s); the jvp call will likely fall "
            "back to finite_difference at the first scripted module.",
            exc,
        )
        return False
    log.info(
        "pixel_meanflow.du_dt_backend=jvp: switched e3nn jit_mode to 'eager' so "
        "TorchScript-compiled e3nn modules accept forward-mode dual tensors."
    )
    return True


def resolve_flow_log_fields(flow: Optional[HamiltonianCFM]) -> Tuple[list, bool]:
    """Scalar log fields implied by the *effective* flow flags.

    Entry points used to re-derive the field list from raw top-level
    flow_options keys, which drifts from the resolved flow object: pixel
    meanflow reads ``meanflow.*`` overrides, never computes the raw-batch
    train_compatible loss, and emits ``validation_flow_one_step_loss`` instead
    of CFM's ``validation_flow_t0_loss``/``validation_flow_euler_N_loss``. A
    registered field the run never updates is seeded as 0.0 and the terminal
    Logger then prints a constant zero that reads like a perfect model.

    Returns ``(flow_log_fields, register_legacy_validation)``. The second
    element says whether the legacy ``validation_onsite_loss`` /
    ``validation_hopping_loss`` keys will actually be produced: for flow runs
    that is the endpoint-compatible euler-1 pass; with flow disabled the plain
    validation criterion fills them.
    """
    if flow is None or not getattr(flow, "enabled", False):
        return [], True

    model_in_loss = bool(getattr(flow, "model_in_loss", False))
    ode_steps = [int(n) for n in getattr(flow, "validation_ode_steps", ()) or ()]
    fields = [
        "train_flow_loss",
        "train_flow_onsite_loss",
        "train_flow_hopping_loss",
        "train_flow_t",
        "train_flow_weight",
    ]
    if model_in_loss:
        fields.extend(
            [
                "train_flow_r",
                "train_flow_h",
                "train_flow_objective_finite_difference",
                "train_flow_objective_semigroup",
                # canary scalars: a silent jvp->finite_difference fallback is
                # invisible in production without these in the terminal/TB log.
                "train_flow_du_dt_backend_jvp",
                "train_flow_explicit_model_calls",
            ]
        )
        if getattr(flow, "meanflow_objective", "finite_difference") in {"semigroup", "hybrid"}:
            fields.extend(
                [
                    "train_flow_semigroup_split_t",
                    "train_flow_semigroup_weight",
                    "train_flow_semigroup_endpoint_weight",
                    "train_flow_semigroup_explicit_model_calls",
                    "train_flow_onsite_semigroup_loss",
                    "train_flow_hopping_semigroup_loss",
                ]
            )
    if getattr(flow, "log_validation_random_t_loss", True):
        fields.append("validation_flow_random_t_loss")
    if getattr(flow, "log_validation_t0_loss", True):
        fields.append(
            "validation_flow_one_step_loss" if model_in_loss else "validation_flow_t0_loss"
        )
    if not model_in_loss and getattr(flow, "log_validation_flow_euler_loss", True):
        for num_steps in ode_steps:
            fields.append(f"validation_flow_euler_{num_steps}_loss")
    if getattr(flow, "log_train_compatible_loss", False) and not model_in_loss:
        fields.extend(
            [
                "train_compatible_loss",
                "train_compatible_onsite_loss",
                "train_compatible_hopping_loss",
            ]
        )
    log_validation_compatible = True
    if log_validation_compatible:
        for num_steps in ode_steps:
            fields.extend(
                [
                    f"validation_compatible_euler_{num_steps}_loss",
                    f"validation_compatible_euler_{num_steps}_onsite_loss",
                    f"validation_compatible_euler_{num_steps}_hopping_loss",
                ]
            )
    register_legacy = bool(
        log_validation_compatible
        and getattr(flow, "compatible_loss_to_legacy_keys", True)
        and 1 in ode_steps
    )
    return fields, register_legacy


# ---------------------------------------------------------------------------
# Backward-compatible re-export.  ``HamiltonianPixelMeanFlow`` now lives in
# ``flow_meanflow``; importing it here (after ``HamiltonianCFM`` is defined,
# which is what breaks the circular import) keeps existing
# ``from dptb.nnops.flow import HamiltonianPixelMeanFlow`` call sites and the
# ``build_hamiltonian_flow`` factory working. ``CFMContext``/``PixelMFContext``
# /``_to_torch_dtype`` are already re-exported via the top-level flow_context
# import.
# When ``flow_meanflow`` is imported first, it triggers this module's execution
# at its line-17 ``from dptb.nnops.flow import ...``; ``flow_meanflow`` is then
# only partially initialised (its class not yet defined), so this import would
# fail.  In that order the class is instead assigned back onto this module by
# ``flow_meanflow`` once it finishes defining it, so tolerate the partial-init
# ImportError here.
# ---------------------------------------------------------------------------
try:
    from dptb.nnops.flow_meanflow import HamiltonianPixelMeanFlow
except ImportError:
    pass
