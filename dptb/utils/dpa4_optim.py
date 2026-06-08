import math
from typing import Iterable, List, Optional, Sequence, Union

import torch
from torch.optim import Optimizer


NumberOrList = Union[float, Sequence[float]]


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


class HybridMuon(Optimizer):
    """DPA4-style hybrid Muon/AdamW optimizer.

    Matrix-shaped tensors use a Muon update with slice-mode trailing matrices.
    Scalar and vector tensors use AdamW-style adaptive updates. This implements
    the portable training path; fused grouped Gram kernels and Magma-lite
    damping can be layered on top later without changing the public config.
    """

    _FAST_COEFF = (3.4445, -4.7750, 2.0315)
    _POLISH_COEFF = (2.0, -1.5, 0.5)

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
        )
        super().__init__(params, defaults)
        self.route_counts = self._count_routes()

    @staticmethod
    def _effective_shape(param: torch.Tensor) -> List[int]:
        return [int(dim) for dim in param.shape if int(dim) != 1]

    @classmethod
    def _uses_muon(cls, param: torch.Tensor, matrix_min_dim: int) -> bool:
        shape = cls._effective_shape(param)
        return len(shape) >= 2 and min(shape[-2:]) >= matrix_min_dim

    def _count_routes(self):
        counts = {"muon": 0, "adam": 0}
        for group in self.param_groups:
            matrix_min_dim = group["matrix_min_dim"]
            for param in group["params"]:
                if self._uses_muon(param, matrix_min_dim):
                    counts["muon"] += 1
                else:
                    counts["adam"] += 1
        return counts

    @classmethod
    def _effective_view(cls, tensor: torch.Tensor) -> torch.Tensor:
        shape = cls._effective_shape(tensor)
        return tensor.reshape(shape)

    @classmethod
    def _reshape_to_matrix_batch(cls, tensor: torch.Tensor) -> torch.Tensor:
        tensor = cls._effective_view(tensor)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    @classmethod
    def _orthogonalize(cls, update: torch.Tensor) -> torch.Tensor:
        original_shape = update.shape
        x = cls._reshape_to_matrix_batch(update.float())
        transposed = x.shape[-2] > x.shape[-1]
        if transposed:
            x = x.transpose(-2, -1)

        denom = x.norm(dim=(-2, -1), keepdim=True).clamp_min(1.0e-30)
        x = x / denom
        coeffs = [cls._FAST_COEFF] * 8 + [cls._POLISH_COEFF] * 2
        for a, b, c in coeffs:
            gram = x @ x.transpose(-2, -1)
            gram2 = gram @ gram
            x = a * x + b * (gram @ x) + c * (gram2 @ x)

        if transposed:
            x = x.transpose(-2, -1)
        return x.reshape(original_shape).to(dtype=update.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            matrix_min_dim = group["matrix_min_dim"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("HybridMuon does not support sparse gradients")

                if self._uses_muon(param, matrix_min_dim):
                    self._muon_step(param, grad, group, lr, weight_decay)
                else:
                    self._adamw_step(param, grad, group, lr)

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
        ortho_update = self._orthogonalize(update)
        if group["magma_lite"]:
            ortho_update = ortho_update * self._magma_lite_scale(param, grad, update, group, state)
        effective_shape = self._effective_shape(update)
        rows, cols = int(effective_shape[-2]), int(effective_shape[-1])
        scale = lr * group["muon_scale"] * math.sqrt(max(rows, cols))
        param.add_(ortho_update, alpha=-scale)

    def _magma_lite_scale(self, param, grad, update, group, state):
        grad_blocks = self._reshape_to_matrix_batch(grad.float())
        update_blocks = self._reshape_to_matrix_batch(update.float())
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
        effective_shape = self._effective_shape(param)
        if len(effective_shape) == 2:
            scale_view = scale.reshape(1, 1)
        else:
            scale_view = scale.reshape(*effective_shape[:-2], 1, 1)
        return scale_view.expand(effective_shape).reshape(param.shape).to(dtype=param.dtype)

    def _adamw_step(self, param, grad, group, lr):
        state = self.state[param]
        if len(state) == 0:
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        beta1, beta2 = group["adam_betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"].add_(1.0)
        step = int(state["step"].item())

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["adam_eps"])
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)
