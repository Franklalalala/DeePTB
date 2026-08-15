import math
from fnmatch import fnmatchcase
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer


NumberOrList = Union[float, Sequence[float]]
NamedParamMap = Dict[int, str]
ClipStatTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _is_named_parameter(item) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and torch.is_tensor(item[1])
    )


def _normalize_named_parameters(params) -> Tuple[object, NamedParamMap]:
    """Strip optimizer parameter names while retaining them for Muon routing."""
    param_names: NamedParamMap = {}

    def normalize_one(item):
        if _is_named_parameter(item):
            name, param = item
            param_names[id(param)] = name
            return param
        return item

    def normalize_sequence(seq):
        if torch.is_tensor(seq):
            return seq
        return [normalize_one(item) for item in list(seq)]

    if torch.is_tensor(params):
        return params, param_names
    if isinstance(params, dict):
        group = dict(params)
        group["params"] = normalize_sequence(group["params"])
        return [group], param_names

    items = list(params)
    if all(isinstance(item, dict) for item in items):
        groups = []
        for item in items:
            group = dict(item)
            group["params"] = normalize_sequence(group["params"])
            groups.append(group)
        return groups, param_names
    return [normalize_one(item) for item in items], param_names


def _as_pattern_tuple(value: Optional[Sequence[str]], default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        value = default
    if isinstance(value, str):
        return (value,)
    return tuple(str(pattern) for pattern in value if str(pattern))


def _as_float_list(value: NumberOrList, n: int, name: str) -> List[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have length {n}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value) for _ in range(n)]


class WarmupStableDecayLR(torch.optim.lr_scheduler.LRScheduler):
    """DPA4 warmup-stable-decay learning-rate schedule.

    The paper tables use ``decay_ratio`` as the training-progress fraction at
    which the final cosine decay starts. For example, 0.65 keeps the stable LR
    until 65 percent of total steps, then decays to ``min_lr``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 5000,
        decay_ratio: float = 0.65,
        min_lr: NumberOrList = 1.0e-6,
        warmup_lr: NumberOrList = 0.0,
        decay_steps: Optional[int] = None,
        decay_type: str = "cosine",
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.decay_ratio = float(decay_ratio)
        self.decay_steps = None if decay_steps is None else int(decay_steps)
        self.decay_type = str(decay_type).lower()

        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0 for WSD scheduler")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0 for WSD scheduler")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps for WSD scheduler")
        if not (0.0 < self.decay_ratio <= 1.0):
            raise ValueError("decay_ratio must be in (0, 1] for WSD scheduler")
        if self.decay_steps is not None and self.decay_steps <= 0:
            raise ValueError("decay_steps must be > 0 when provided")
        if self.decay_type != "cosine":
            raise ValueError("DPA4 WSD currently supports decay_type='cosine'")

        if self.decay_steps is None:
            self.decay_start_step = int(round(self.total_steps * self.decay_ratio))
            self.decay_steps = self.total_steps - self.decay_start_step
        else:
            self.decay_start_step = self.total_steps - self.decay_steps
        if self.decay_start_step < self.warmup_steps:
            raise ValueError(
                "WSD decay phase starts before warmup ends: "
                f"decay_start_step={self.decay_start_step}, warmup_steps={self.warmup_steps}"
            )
        if self.decay_steps <= 0:
            raise ValueError("WSD decay phase must contain at least one step")

        n_groups = len(optimizer.param_groups)
        self.min_lrs = _as_float_list(min_lr, n_groups, "min_lr")
        self.warmup_lrs = _as_float_list(warmup_lr, n_groups, "warmup_lr")
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_for_step(self, step: int, base_lr: float, warmup_lr: float, min_lr: float) -> float:
        t = max(0, int(step))
        if t < self.warmup_steps:
            if self.warmup_steps == 0:
                return float(base_lr)
            return float(warmup_lr + (base_lr - warmup_lr) * (t / self.warmup_steps))
        if t < self.decay_start_step:
            return float(base_lr)
        if t >= self.total_steps:
            return float(min_lr)

        tau = (t - self.decay_start_step) / self.decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * tau))
        return float(min_lr + (base_lr - min_lr) * cosine)

    def get_lr_at_step(self, step: int) -> List[float]:
        return [
            self._lr_for_step(step, base_lr, warmup_lr, min_lr)
            for base_lr, warmup_lr, min_lr in zip(self.base_lrs, self.warmup_lrs, self.min_lrs)
        ]

    def get_lr(self) -> List[float]:
        return self.get_lr_at_step(self.last_epoch)


class WarmupThenReduceLROnPlateau:
    """Linear warmup followed by metric-driven ReduceLROnPlateau.

    ``warmup_steps`` counts scheduler ``step`` calls. This makes the schedule
    work for both epoch-level stepping and ``update_lr_per_iter=True`` runs.
    """

    requires_metric = True

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 5000,
        warmup_lr: NumberOrList = 0.0,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1.0e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: NumberOrList = 0.0,
        eps: float = 1.0e-8,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0 for warmup_rop scheduler")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_lrs = _as_float_list(warmup_lr, len(optimizer.param_groups), "warmup_lr")
        self.last_epoch = int(last_epoch)
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

        if self.last_epoch < 0:
            self.last_epoch = 0
        if self.last_epoch <= self.warmup_steps:
            self._set_lrs(self.get_warmup_lrs_at_step(self.last_epoch))

    def _set_lrs(self, lrs: Sequence[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)
        self._last_lr = [float(lr) for lr in lrs]

    def _warmup_lr_for_step(self, step: int, base_lr: float, warmup_lr: float) -> float:
        if self.warmup_steps == 0:
            return float(base_lr)
        t = min(max(0, int(step)), self.warmup_steps)
        return float(warmup_lr + (base_lr - warmup_lr) * (t / self.warmup_steps))

    def get_warmup_lrs_at_step(self, step: int) -> List[float]:
        return [
            self._warmup_lr_for_step(step, base_lr, warmup_lr)
            for base_lr, warmup_lr in zip(self.base_lrs, self.warmup_lrs)
        ]

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    def can_step_without_metric(self) -> bool:
        return self.last_epoch < self.warmup_steps

    def step(self, metrics=None):
        if self.last_epoch < self.warmup_steps:
            self.last_epoch += 1
            self._set_lrs(self.get_warmup_lrs_at_step(self.last_epoch))
            return self.get_last_lr()

        if metrics is None:
            raise ValueError("warmup_rop requires a metric after warmup finishes")

        self.last_epoch += 1
        self.plateau.step(metrics)
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
        return self.get_last_lr()

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "warmup_lrs": list(self.warmup_lrs),
            "base_lrs": list(self.base_lrs),
            "last_epoch": self.last_epoch,
            "_last_lr": list(self._last_lr),
            "plateau_state_dict": self.plateau.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.warmup_lrs = [float(v) for v in state_dict["warmup_lrs"]]
        self.base_lrs = [float(v) for v in state_dict["base_lrs"]]
        self.last_epoch = int(state_dict["last_epoch"])
        self._last_lr = [float(v) for v in state_dict.get("_last_lr", self._last_lr)]
        self.plateau.load_state_dict(state_dict["plateau_state_dict"])


class HybridMuon(Optimizer):
    """Hybrid Muon/AdamW optimizer with generic flattened-weight routing."""

    _FAST_COEFF = (3.4445, -4.7750, 2.0315)
    _POLISH_COEFF = (2.0, -1.5, 0.5)
    _STAT_BLOCKS = 0
    _STAT_CLIP_EVENTS = 1
    _STAT_CLIP_MIN_SCALE = 2
    _STAT_UPDATE_RMS_MAX = 3
    _STAT_UPDATE_RMS_SUM = 4
    _STAT_UPDATE_RMS_COUNT = 5
    _STAT_STEP_RATIO_MAX = 6
    _DEFAULT_1D_INCLUDE_PATTERNS = ("*weight*", "*tensor_product*")
    _DEFAULT_1D_EXCLUDE_PATTERNS = (
        "*bias*",
        "*norm*",
        "*scale*",
        "*shift*",
        "*offset*",
        "*res_update*",
        "*bessel*",
        "*cutoff*",
        "*temperature*",
        "*freq*",
        "*router*",
    )

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-3,
        muon_beta: float = 0.95,
        muon_scale: float = 0.18,
        adam_betas: Sequence[float] = (0.9, 0.999),
        adam_eps: float = 1.0e-20,
        matrix_min_dim: int = 2,
        magma_lite: bool = True,
        magma_temperature: float = 2.0,
        magma_ema_beta: float = 0.9,
        magma_min_scale: float = 0.1,
        muon_1d_route_mode: str = "auto",
        muon_1d_include_name_patterns: Optional[Sequence[str]] = None,
        muon_1d_exclude_name_patterns: Optional[Sequence[str]] = None,
        muon_1d_min_numel: int = 16,
        muon_1d_max_aspect_ratio: float = 64.0,
        muon_1d_allow_degenerate_matrix: bool = False,
        muon_force_name_patterns: Optional[Sequence[str]] = None,
        muon_clip: bool = True,
        muon_clip_mode: str = "auto",
        muon_clip_rms: float = 0.6,
        muon_clip_auto_beta: float = 0.98,
        muon_clip_auto_mult: float = 3.0,
        muon_clip_auto_std_mult: float = 2.0,
        muon_clip_min_ratio: float = 0.01,
        muon_clip_max_ratio: float = 0.25,
        muon_clip_param_rms_floor: float = 1.0e-3,
        muon_clip_warmup_steps: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not (0.0 <= muon_beta < 1.0):
            raise ValueError(f"Invalid muon_beta: {muon_beta}")
        if muon_scale <= 0.0:
            raise ValueError(f"Invalid muon_scale: {muon_scale}")
        if len(adam_betas) != 2:
            raise ValueError("adam_betas must contain two values")
        beta1, beta2 = float(adam_betas[0]), float(adam_betas[1])
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid adam_betas: {adam_betas}")
        if adam_eps <= 0.0:
            raise ValueError(f"Invalid adam_eps: {adam_eps}")
        if matrix_min_dim < 1:
            raise ValueError(f"Invalid matrix_min_dim: {matrix_min_dim}")
        if magma_temperature <= 0.0:
            raise ValueError(f"Invalid magma_temperature: {magma_temperature}")
        if not (0.0 <= magma_ema_beta < 1.0):
            raise ValueError(f"Invalid magma_ema_beta: {magma_ema_beta}")
        if not (0.0 <= magma_min_scale <= 1.0):
            raise ValueError(f"Invalid magma_min_scale: {magma_min_scale}")
        route_mode = str(muon_1d_route_mode).lower()
        if route_mode not in {"auto", "force", "off"}:
            raise ValueError("muon_1d_route_mode must be auto, force, or off")
        if muon_1d_min_numel < 1:
            raise ValueError(f"Invalid muon_1d_min_numel: {muon_1d_min_numel}")
        if muon_1d_max_aspect_ratio < 1.0:
            raise ValueError(f"Invalid muon_1d_max_aspect_ratio: {muon_1d_max_aspect_ratio}")
        clip_mode = str(muon_clip_mode).lower()
        if clip_mode == "rms":
            clip_mode = "fixed"
        if clip_mode == "off":
            muon_clip = False
            clip_mode = "fixed"
        if clip_mode not in {"auto", "fixed"}:
            raise ValueError("muon_clip_mode must be auto, fixed/rms, or off")
        if muon_clip_rms <= 0.0:
            raise ValueError(f"Invalid muon_clip_rms: {muon_clip_rms}")
        if not (0.0 <= muon_clip_auto_beta < 1.0):
            raise ValueError(f"Invalid muon_clip_auto_beta: {muon_clip_auto_beta}")
        if muon_clip_auto_mult < 1.0:
            raise ValueError(f"Invalid muon_clip_auto_mult: {muon_clip_auto_mult}")
        if muon_clip_auto_std_mult < 0.0:
            raise ValueError(f"Invalid muon_clip_auto_std_mult: {muon_clip_auto_std_mult}")
        if not (0.0 <= muon_clip_min_ratio <= muon_clip_max_ratio):
            raise ValueError("muon_clip_min_ratio must be between zero and muon_clip_max_ratio")
        if muon_clip_max_ratio <= 0.0:
            raise ValueError("muon_clip_max_ratio must be positive")
        if muon_clip_param_rms_floor <= 0.0:
            raise ValueError(f"Invalid muon_clip_param_rms_floor: {muon_clip_param_rms_floor}")
        if muon_clip_warmup_steps < 0:
            raise ValueError(f"Invalid muon_clip_warmup_steps: {muon_clip_warmup_steps}")

        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            muon_beta=float(muon_beta),
            muon_scale=float(muon_scale),
            adam_betas=(beta1, beta2),
            adam_eps=float(adam_eps),
            matrix_min_dim=int(matrix_min_dim),
            magma_lite=bool(magma_lite),
            magma_temperature=float(magma_temperature),
            magma_ema_beta=float(magma_ema_beta),
            magma_min_scale=float(magma_min_scale),
            muon_1d_route_mode=route_mode,
            muon_1d_include_name_patterns=_as_pattern_tuple(
                muon_1d_include_name_patterns,
                self._DEFAULT_1D_INCLUDE_PATTERNS,
            ),
            muon_1d_exclude_name_patterns=_as_pattern_tuple(
                muon_1d_exclude_name_patterns,
                self._DEFAULT_1D_EXCLUDE_PATTERNS,
            ),
            muon_1d_min_numel=int(muon_1d_min_numel),
            muon_1d_max_aspect_ratio=float(muon_1d_max_aspect_ratio),
            muon_1d_allow_degenerate_matrix=bool(muon_1d_allow_degenerate_matrix),
            muon_force_name_patterns=_as_pattern_tuple(muon_force_name_patterns, ()),
            muon_clip=bool(muon_clip),
            muon_clip_mode=clip_mode,
            muon_clip_rms=float(muon_clip_rms),
            muon_clip_auto_beta=float(muon_clip_auto_beta),
            muon_clip_auto_mult=float(muon_clip_auto_mult),
            muon_clip_auto_std_mult=float(muon_clip_auto_std_mult),
            muon_clip_min_ratio=float(muon_clip_min_ratio),
            muon_clip_max_ratio=float(muon_clip_max_ratio),
            muon_clip_param_rms_floor=float(muon_clip_param_rms_floor),
            muon_clip_warmup_steps=int(muon_clip_warmup_steps),
        )
        params, self._param_names = _normalize_named_parameters(params)
        super().__init__(params, defaults)
        self._last_step_stat_tensors: List[ClipStatTensors] = []
        self._pending_diagnostics_tensor: Optional[torch.Tensor] = None
        self._last_step_diagnostics_cache: Optional[Dict[str, float]] = None
        self._route_summary_cache: Optional[Dict[str, Union[int, float]]] = None

    def add_param_group(self, param_group) -> None:
        super().add_param_group(param_group)
        self._route_summary_cache = None

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        for state in self.state.values():
            step = state.get("step")
            if torch.is_tensor(step):
                state["step"] = int(step.item())
        self._route_summary_cache = None
        return result

    def _ensure_group_defaults(self, group) -> None:
        """Fill new options when loading checkpoints written by older versions."""
        for key, value in self.defaults.items():
            group.setdefault(key, value)

    @staticmethod
    def _effective_shape(param: torch.Tensor) -> List[int]:
        return [int(dim) for dim in param.shape if int(dim) != 1]

    @classmethod
    def _uses_native_muon_shape(cls, param: torch.Tensor, matrix_min_dim: int) -> bool:
        shape = cls._effective_shape(param)
        return len(shape) >= 2 and min(shape[-2:]) >= matrix_min_dim

    @staticmethod
    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        lowered = name.lower()
        return bool(lowered) and any(
            pattern and (pattern.lower() in lowered or fnmatchcase(lowered, pattern.lower()))
            for pattern in patterns
        )

    @staticmethod
    def _factor_1d_as_matrix(
        numel: int,
        matrix_min_dim: int,
        max_aspect_ratio: float,
        allow_degenerate: bool,
    ) -> Optional[List[int]]:
        if numel < matrix_min_dim * matrix_min_dim:
            return None
        for rows in range(int(math.sqrt(numel)), matrix_min_dim - 1, -1):
            if numel % rows != 0:
                continue
            cols = numel // rows
            if cols >= matrix_min_dim and cols / rows <= max_aspect_ratio:
                return [rows, cols]
        return [1, numel] if allow_degenerate else None

    def _flat_1d_matrix_shape(self, param: torch.Tensor, group) -> Optional[List[int]]:
        self._ensure_group_defaults(group)
        if group["muon_1d_route_mode"] == "off":
            return None
        shape = self._effective_shape(param)
        if len(shape) != 1 or shape[0] < group["muon_1d_min_numel"]:
            return None

        name = self._param_names.get(id(param), "")
        forced = self._matches_any(name, group["muon_force_name_patterns"])
        if not forced:
            if not name or self._matches_any(name, group["muon_1d_exclude_name_patterns"]):
                return None
            if (
                group["muon_1d_route_mode"] == "auto"
                and not self._matches_any(name, group["muon_1d_include_name_patterns"])
            ):
                return None

        return self._factor_1d_as_matrix(
            shape[0],
            group["matrix_min_dim"],
            group["muon_1d_max_aspect_ratio"],
            group["muon_1d_allow_degenerate_matrix"],
        )

    def _uses_muon(self, param: torch.Tensor, group) -> bool:
        return self._uses_native_muon_shape(param, group["matrix_min_dim"]) or (
            self._flat_1d_matrix_shape(param, group) is not None
        )

    def _effective_shape_for_param(self, param: torch.Tensor, group) -> List[int]:
        return self._flat_1d_matrix_shape(param, group) or self._effective_shape(param)

    def _compute_route_summary(self) -> Dict[str, Union[int, float]]:
        counts = {"muon": 0, "adam": 0}
        numels = {"muon": 0, "adam": 0, "flat_muon": 0, "total": 0}
        one_d_muon = 0
        for group in self.param_groups:
            self._ensure_group_defaults(group)
            for param in group["params"]:
                route = "muon" if self._uses_muon(param, group) else "adam"
                n = int(param.numel())
                counts[route] += 1
                numels[route] += n
                numels["total"] += n
                if route == "muon" and self._flat_1d_matrix_shape(param, group) is not None:
                    one_d_muon += 1
                    numels["flat_muon"] += n

        total_numel = max(numels["total"], 1)
        return {
            "params_total": counts["muon"] + counts["adam"],
            "params_muon": counts["muon"],
            "params_adam": counts["adam"],
            "params_1d_muon": one_d_muon,
            "numel_total": numels["total"],
            "numel_muon": numels["muon"],
            "numel_adam": numels["adam"],
            "numel_flat_muon": numels["flat_muon"],
            "numel_muon_ratio": numels["muon"] / total_numel,
            "numel_flat_muon_ratio": numels["flat_muon"] / total_numel,
        }

    def route_summary(self) -> Dict[str, Union[int, float]]:
        if self._route_summary_cache is None:
            self._route_summary_cache = self._compute_route_summary()
        return dict(self._route_summary_cache)

    @property
    def route_counts(self) -> Dict[str, int]:
        summary = self.route_summary()
        return {"muon": int(summary["params_muon"]), "adam": int(summary["params_adam"])}

    def _reshape_to_matrix_batch_for_param(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        group,
    ) -> torch.Tensor:
        shape = self._effective_shape_for_param(param, group)
        tensor = tensor.reshape(shape)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    def _orthogonalize(self, update: torch.Tensor, param: torch.Tensor, group) -> torch.Tensor:
        original_shape = update.shape
        x = self._reshape_to_matrix_batch_for_param(update.float(), param, group)
        transposed = x.shape[-2] > x.shape[-1]
        if transposed:
            x = x.transpose(-2, -1)

        # Additive epsilon, matching the reference Muon implementation
        # (KellerJordan/Muon: ``X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)``).
        #
        # A ``clamp_min`` floor instead of an additive epsilon lets a degenerate
        # block be amplified by up to 1/floor. That is unbounded in practice: in
        # float32 ``x * x`` underflows to zero for |x| < ~1.1e-19, so a block whose
        # entries are tiny but non-zero has ``norm() == 0`` exactly. With a 1e-30
        # floor such a block is scaled by 1e30, ``gram = x @ x.mT`` overflows to
        # inf, and the quintic ``a*x + b*(gram@x) + c*(gram@gram@x)`` evaluates
        # inf - inf -> NaN. The NaN is then written into that block's parameters.
        #
        # This is reachable whenever a batched (n_blocks, m, n) update contains a
        # block with a vanishing-but-non-zero gradient -- e.g. a sparsely routed
        # MoE expert (top_k << num_experts) that received almost no tokens. An
        # exactly-zero gradient is safe either way (0 / eps == 0); only the
        # underflow window is affected. The additive epsilon caps the
        # amplification at 1e7, so degenerate blocks simply receive a
        # near-zero update.
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1.0e-7)
        for a, b, c in [self._FAST_COEFF] * 8 + [self._POLISH_COEFF] * 2:
            gram = x @ x.transpose(-2, -1)
            x = a * x + b * (gram @ x) + c * ((gram @ gram) @ x)

        if transposed:
            x = x.transpose(-2, -1)
        return x.reshape(original_shape).to(dtype=update.dtype)

    def _reset_step_stats(self) -> None:
        if self._last_step_diagnostics_cache is not None:
            self._pending_diagnostics_tensor = None
        self._last_step_stat_tensors = []
        self._last_step_diagnostics_cache = None

    def get_diagnostics(self) -> Dict[str, float]:
        if self._last_step_diagnostics_cache is None:
            diagnostics = self._materialize_step_diagnostics()
            diagnostics.update(
                {
                    f"hybrid_muon_route_{key}": float(value)
                    for key, value in self.route_summary().items()
                }
            )
            self._last_step_diagnostics_cache = diagnostics
        return dict(self._last_step_diagnostics_cache)

    def _materialize_step_diagnostics(self) -> Dict[str, float]:
        if self._pending_diagnostics_tensor is None:
            return {
                "muon_blocks": 0.0,
                "muon_clip_events": 0.0,
                "muon_clip_min_scale": 1.0,
                "muon_update_rms_max": 0.0,
                "muon_update_rms_mean": 0.0,
                "muon_step_ratio_max": 0.0,
            }

        materialized = self._pending_diagnostics_tensor.tolist()
        count = materialized[self._STAT_UPDATE_RMS_COUNT]
        mean_rms = materialized[self._STAT_UPDATE_RMS_SUM] / count if count else 0.0
        return {
            "muon_blocks": float(materialized[self._STAT_BLOCKS]),
            "muon_clip_events": float(materialized[self._STAT_CLIP_EVENTS]),
            "muon_clip_min_scale": float(materialized[self._STAT_CLIP_MIN_SCALE]),
            "muon_update_rms_max": float(materialized[self._STAT_UPDATE_RMS_MAX]),
            "muon_update_rms_mean": float(mean_rms),
            "muon_step_ratio_max": float(materialized[self._STAT_STEP_RATIO_MAX]),
        }

    def _finalize_step_stats(self) -> None:
        if not self._last_step_stat_tensors:
            return

        rms = torch.cat([values[0] for values in self._last_step_stat_tensors])
        clip_scale = torch.cat([values[1] for values in self._last_step_stat_tensors])
        step_ratio = torch.cat([values[2] for values in self._last_step_stat_tensors])
        count = rms.new_tensor(float(rms.numel()))
        current = torch.stack(
            [
                count,
                (clip_scale < 0.999999).sum().to(rms.dtype),
                clip_scale.min(),
                rms.max(),
                rms.sum(),
                count,
                step_ratio.max(),
            ]
        )
        if self._pending_diagnostics_tensor is None:
            self._pending_diagnostics_tensor = current
        else:
            pending = self._pending_diagnostics_tensor
            self._pending_diagnostics_tensor = torch.stack(
                [
                    pending[self._STAT_BLOCKS] + current[self._STAT_BLOCKS],
                    pending[self._STAT_CLIP_EVENTS] + current[self._STAT_CLIP_EVENTS],
                    torch.minimum(pending[self._STAT_CLIP_MIN_SCALE], current[self._STAT_CLIP_MIN_SCALE]),
                    torch.maximum(pending[self._STAT_UPDATE_RMS_MAX], current[self._STAT_UPDATE_RMS_MAX]),
                    pending[self._STAT_UPDATE_RMS_SUM] + current[self._STAT_UPDATE_RMS_SUM],
                    pending[self._STAT_UPDATE_RMS_COUNT] + current[self._STAT_UPDATE_RMS_COUNT],
                    torch.maximum(pending[self._STAT_STEP_RATIO_MAX], current[self._STAT_STEP_RATIO_MAX]),
                ]
            )
        self._last_step_stat_tensors = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._reset_step_stats()
        for group in self.param_groups:
            self._ensure_group_defaults(group)
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("HybridMuon does not support sparse gradients")
                if self._uses_muon(param, group):
                    self._muon_step(param, grad, group, lr, weight_decay)
                else:
                    self._adamw_step(param, grad, group, lr, weight_decay)

        self._finalize_step_stats()
        return loss

    def _muon_step(self, param, grad, group, lr, weight_decay):
        state = self.state[param]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        momentum = state["momentum_buffer"]
        beta = group["muon_beta"]
        momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
        update = momentum.mul(beta).add(grad, alpha=1.0 - beta)
        ortho_update = self._orthogonalize(update, param, group)
        if group["magma_lite"]:
            ortho_update = ortho_update * self._magma_lite_scale(param, grad, update, group, state)
        effective_shape = self._effective_shape_for_param(param, group)
        rows, cols = int(effective_shape[-2]), int(effective_shape[-1])
        scaled_update = ortho_update * (group["muon_scale"] * math.sqrt(max(rows, cols)))
        scaled_update = self._clip_muon_update(param, scaled_update, group, state, lr)
        param.add_(scaled_update, alpha=-lr)

    def _magma_lite_scale(self, param, grad, update, group, state):
        grad_blocks = self._reshape_to_matrix_batch_for_param(grad.float(), param, group)
        update_blocks = self._reshape_to_matrix_batch_for_param(update.float(), param, group)
        numerator = (grad_blocks * update_blocks).sum(dim=(-2, -1))
        denom = (
            grad_blocks.norm(dim=(-2, -1))
            * update_blocks.norm(dim=(-2, -1))
        ).clamp_min(1.0e-30)
        chi = (numerator / denom).clamp(min=-1.0, max=1.0)

        tau = group["magma_temperature"]
        lo = torch.sigmoid(chi.new_tensor(-1.0 / tau))
        hi = torch.sigmoid(chi.new_tensor(1.0 / tau))
        score = ((torch.sigmoid(chi / tau) - lo) / (hi - lo)).clamp(min=0.0, max=1.0)

        if "magma_ema" not in state or state["magma_ema"].shape != score.shape:
            state["magma_ema"] = torch.zeros_like(score)
        ema = state["magma_ema"]
        ema.mul_(group["magma_ema_beta"]).add_(score, alpha=1.0 - group["magma_ema_beta"])

        scale = group["magma_min_scale"] + (1.0 - group["magma_min_scale"]) * ema
        effective_shape = self._effective_shape_for_param(param, group)
        if len(effective_shape) == 2:
            scale_view = scale.reshape(1, 1)
        else:
            scale_view = scale.reshape(*effective_shape[:-2], 1, 1)
        return scale_view.expand(effective_shape).reshape(param.shape).to(dtype=param.dtype)

    def _clip_muon_update(self, param, update, group, state, lr):
        blocks = self._reshape_to_matrix_batch_for_param(update.float(), param, group)
        rms = blocks.pow(2).mean(dim=(-2, -1)).sqrt()
        clip_scale = torch.ones_like(rms)
        step_ratio = torch.zeros_like(rms)

        if group["muon_clip"]:
            hard_scale = group["muon_clip_rms"] / rms.clamp_min(1.0e-30)
            clip_scale = torch.minimum(clip_scale, hard_scale.clamp(max=1.0))
            if group["muon_clip_mode"] == "auto":
                param_blocks = self._reshape_to_matrix_batch_for_param(param.float(), param, group)
                param_rms = param_blocks.pow(2).mean(dim=(-2, -1)).sqrt().clamp_min(
                    group["muon_clip_param_rms_floor"]
                )
                step_ratio = (lr * rms / param_rms).detach()
                clip_step = int(state.get("muon_clip_step", 0)) + 1
                state["muon_clip_step"] = clip_step
                beta = group["muon_clip_auto_beta"]
                if "muon_step_ratio_ema" not in state:
                    state["muon_step_ratio_ema"] = step_ratio.clone()
                    state["muon_step_ratio_sq_ema"] = step_ratio.square().clone()
                else:
                    state["muon_step_ratio_ema"].mul_(beta).add_(step_ratio, alpha=1.0 - beta)
                    state["muon_step_ratio_sq_ema"].mul_(beta).addcmul_(
                        step_ratio,
                        step_ratio,
                        value=1.0 - beta,
                    )
                if clip_step > group["muon_clip_warmup_steps"]:
                    ema = state["muon_step_ratio_ema"]
                    std = (state["muon_step_ratio_sq_ema"] - ema.square()).clamp_min(0.0).sqrt()
                    limit = (
                        ema * group["muon_clip_auto_mult"]
                        + std * group["muon_clip_auto_std_mult"]
                    ).clamp(group["muon_clip_min_ratio"], group["muon_clip_max_ratio"])
                    clip_scale = torch.minimum(
                        clip_scale,
                        (limit / step_ratio.clamp_min(1.0e-30)).clamp(max=1.0),
                    )

        state["muon_clip_scale"] = clip_scale.detach()
        self._accumulate_clip_stats(rms, clip_scale, step_ratio)
        effective_shape = self._effective_shape_for_param(param, group)
        scale_view = clip_scale.reshape(
            *((1, 1) if len(effective_shape) == 2 else (*effective_shape[:-2], 1, 1))
        )
        return (update.reshape(effective_shape) * scale_view).reshape(param.shape)

    def _accumulate_clip_stats(self, rms, clip_scale, step_ratio) -> None:
        self._last_step_stat_tensors.append(
            (
                rms.detach().reshape(-1),
                clip_scale.detach().reshape(-1),
                step_ratio.detach().reshape(-1),
            )
        )

    def _adamw_step(self, param, grad, group, lr, weight_decay):
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        beta1, beta2 = group["adam_betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1
        step = state["step"]

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["adam_eps"])
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)
