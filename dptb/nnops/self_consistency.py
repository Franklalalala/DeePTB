"""WS4 phase C: training-period self-consistency loss.

  L_sc = || H_pred - stopgrad(R(H_pred)) ||^2_masked

following Zhang et al., "Self-Consistency Training for Hamiltonian
Prediction", ICML 2024 -- gradients flow only through ``H_pred``; ``R``
(the ABACUS ``restart_dh`` one-shot/full-SCF repair, see
``dptb.postprocess.hrebuild``) is called through an external endpoint and
never differentiated through.

Design (`train_options.self_consistency`, see ``dptb/utils/argcheck.py`` ::
``self_consistency_options``): every ``every_n_steps`` steps, a
``sample_frac`` slice of the batch is *submitted* for repair on a
background thread pool (so a slow ~10-20s ABACUS call never stalls the
training step); results are *consumed* ``staleness_steps`` steps later via
:class:`SelfConsistencyScheduler`, i.e. the target is allowed to be
slightly stale rather than blocking. This module is deliberately
trainer-agnostic (it doesn't import ``dptb.nnops.trainer``/``multi_trainer``
or reach into their loss-assembly code) -- the plan document defers the
exact hook point to whoever wires it in ("先 grep trainer.py/multi_trainer.py
里 loss 的组装点再决定挂法"), since that is a live, actively-developed file
this module should not need to touch to be independently testable. Wiring
it in means, in the training step: call ``maybe_submit`` after the forward
pass, call ``maybe_consume`` before the loss reduction, and add
``weight * compute_self_consistency_loss(...)`` for every consumed sample
to the total loss.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


def compute_self_consistency_loss(
    h_pred: torch.Tensor,
    h_repaired: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``L_sc = mean((H_pred - stopgrad(R(H_pred)))^2)``, optionally masked
    to active blocks/edges. ``h_repaired`` is detached defensively even if
    the caller forgot to (gradients must only flow through ``h_pred``)."""
    if h_pred.shape != h_repaired.shape:
        raise ValueError(f"shape mismatch: h_pred={h_pred.shape} vs h_repaired={h_repaired.shape}")
    h_repaired = h_repaired.detach()
    diff = h_pred - h_repaired
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        return (diff.abs() ** 2).sum() / denom
    return (diff.abs() ** 2).mean()


@dataclass
class _PendingRequest:
    submitted_step: int
    sample_id: Any
    future: Future
    h_pred_snapshot: torch.Tensor  # detached copy, kept for the loss target-shape check at consume time


@dataclass
class SelfConsistencySchedulerConfig:
    every_n_steps: int = 100
    sample_frac: float = 0.1
    staleness_steps: int = 1
    warmup_epochs: int = 0
    max_workers: int = 2


class SelfConsistencyScheduler:
    """Owns the submit-now/consume-later bookkeeping. ``repair_fn`` is
    ``(sample_id, h_pred_detached) -> h_repaired_tensor_or_None`` (``None``
    signals a refused/failed repair -- e.g. the gap-threshold guard, or an
    endpoint error -- and is silently dropped rather than raised, so a
    flaky endpoint degrades to "no self-consistency loss this round"
    instead of crashing training)."""

    def __init__(self, repair_fn: Callable[[Any, torch.Tensor], Optional[torch.Tensor]],
                 config: SelfConsistencySchedulerConfig, rng: Optional[torch.Generator] = None):
        self.repair_fn = repair_fn
        self.config = config
        self.rng = rng
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._pending: Dict[int, List[_PendingRequest]] = {}
        self._lock = threading.Lock()

    def _select_indices(self, n_samples: int) -> List[int]:
        k = max(1, int(round(n_samples * self.config.sample_frac)))
        k = min(k, n_samples)
        if self.rng is not None:
            perm = torch.randperm(n_samples, generator=self.rng)
        else:
            perm = torch.randperm(n_samples)
        return perm[:k].tolist()

    def maybe_submit(self, step: int, epoch: int, samples: List[Tuple[Any, torch.Tensor]]) -> bool:
        """``samples``: list of ``(sample_id, h_pred)`` for the current
        batch. Submits a random ``sample_frac`` subset for background
        repair if ``step`` is on the ``every_n_steps`` cadence and past
        ``warmup_epochs``. Returns whether anything was submitted."""
        if epoch < self.config.warmup_epochs:
            return False
        if self.config.every_n_steps <= 0 or step % self.config.every_n_steps != 0:
            return False
        if not samples:
            return False

        indices = self._select_indices(len(samples))
        due_step = step + self.config.staleness_steps
        pending = []
        for idx in indices:
            sample_id, h_pred = samples[idx]
            snapshot = h_pred.detach().clone()
            future = self._pool.submit(self.repair_fn, sample_id, snapshot)
            pending.append(_PendingRequest(submitted_step=step, sample_id=sample_id, future=future, h_pred_snapshot=snapshot))

        with self._lock:
            self._pending.setdefault(due_step, []).extend(pending)
        return True

    def maybe_consume(self, step: int, current_samples: Dict[Any, torch.Tensor],
                       timeout: Optional[float] = 0.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns ``[(h_pred_now, h_repaired), ...]`` for every request
        due at ``step`` whose future is already done (``timeout=0.0``: skip
        anything not yet finished rather than blocking training -- allowed
        staleness means we simply drop it and try again next cadence, we
        do not accumulate ever-growing backlog). ``current_samples`` maps
        ``sample_id -> h_pred`` from *this* step's forward pass, so the
        loss is taken against the live (graph-attached) prediction even
        though the repair target was computed from an older snapshot."""
        with self._lock:
            due = self._pending.pop(step, [])

        out = []
        for req in due:
            try:
                if not req.future.done():
                    if timeout and timeout > 0:
                        h_repaired = req.future.result(timeout=timeout)
                    else:
                        continue
                else:
                    h_repaired = req.future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[self-consistency] repair failed for sample {req.sample_id!r}: {exc}")
                continue
            if h_repaired is None:
                continue
            h_pred_now = current_samples.get(req.sample_id)
            if h_pred_now is None:
                # sample dropped out of the batch by the time the result matured
                continue
            out.append((h_pred_now, h_repaired))
        return out

    def shutdown(self):
        self._pool.shutdown(wait=True)
