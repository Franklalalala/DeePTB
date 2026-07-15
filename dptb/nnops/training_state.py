"""Restart/checkpoint state machine primitives for DeePTB trainers.

This module centralises the *resume-correctness* metadata that both the single
GPU :class:`~dptb.nnops.trainer.Trainer` and the multi-GPU
:class:`~dptb.nnops.multi_trainer.MultiTrainer` persist into a checkpoint so a
job can be restarted without

* skipping the rest of an in-progress epoch,
* replaying (double-applying) already committed batches,
* losing the per-epoch learning-rate transition, or
* diverging the RNG trajectory from an uninterrupted run.

The :class:`TrainingState` dataclass is the canonical container.  It is written
into every new-shape checkpoint under the ``"training_state"`` key and is backed
by ``trainer.iter``/``trainer.ep`` at save time.  A *legacy reader*
(:func:`read_resume_metadata`) normalises both the new shape and the historical
flat shape (top-level ``epoch``/``iteration``/``stats`` with no
``training_state``) so old in-flight production checkpoints keep loading.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Bump when the *serialized* checkpoint layout changes in a way the legacy
# reader must branch on.  v2 introduces the ``training_state``/``plugin_state``
# blocks and the ``checkpoint_kind`` discriminator.  Absent/`< 2` means the
# historical flat shape.
# v3 fixes the global-step semantics: ``iteration``/``global_step`` store the
# LAST COMPLETED optimizer step for BOTH checkpoint kinds (epoch checkpoints
# previously stored the post-increment counter, so restart's uniform ``+1``
# skipped one global-step value). Restart resumes at ``global_step + 1`` for
# every kind; checkpoints written with schema < 3 keep their historical
# resume numbering (parity — no silent renumbering of old runs).
CHECKPOINT_SCHEMA_VERSION = 3

# Discriminates a mid-epoch (iteration) checkpoint, whose epoch is *not*
# committed and must be re-entered, from an epoch-boundary checkpoint whose
# epoch is committed and whose per-epoch LR step is still pending at save time.
CHECKPOINT_KIND_ITERATION = "iteration"
CHECKPOINT_KIND_EPOCH = "epoch"


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot the global RNG state for python/numpy/torch/cuda.

    All returned tensors live on CPU so the snapshot pickles cleanly through
    ``torch.save`` and reloads on a host with a different device layout.
    """
    import torch

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # pragma: no cover - numpy always present in DeePTB
        pass

    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to capture CUDA RNG state: %s", exc)
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore an RNG snapshot produced by :func:`capture_rng_state`.

    Missing sub-states are skipped rather than raising so a checkpoint captured
    without CUDA (or without numpy) still restores what it can.
    """
    if not state:
        return
    import torch

    py_state = state.get("python")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to restore python RNG state: %s", exc)

    np_state = state.get("numpy")
    if np_state is not None:
        try:
            import numpy as np

            np.random.set_state(np_state)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to restore numpy RNG state: %s", exc)

    torch_state = state.get("torch")
    if torch_state is not None:
        try:
            # torch requires a CPU ByteTensor; be tolerant of dtype/device.
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state.to("cpu", torch.uint8))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to restore torch RNG state: %s", exc)

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            if torch.cuda.device_count() == len(cuda_state):
                torch.cuda.set_rng_state_all(list(cuda_state))
            else:
                log.warning(
                    "Skipping CUDA RNG restore: checkpoint has %d device states "
                    "but current host exposes %d devices.",
                    len(cuda_state),
                    torch.cuda.device_count(),
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to restore CUDA RNG state: %s", exc)


@dataclass
class TrainingState:
    """Canonical, serialisable resume state backed by ``trainer.iter``/``ep``.

    ``global_step`` mirrors ``trainer.iter`` (the global optimizer-step counter)
    and ``epoch`` mirrors ``trainer.ep``.  ``batch_in_epoch`` is the number of
    *committed* (optimizer-stepped, baked-into-``model_state_dict``) batches in
    the current epoch at save time; on an iteration resume the epoch loop
    fast-forwards exactly this many batches so each remaining batch runs once.
    """

    global_step: int = 1
    epoch: int = 1
    batch_in_epoch: int = 0
    checkpoint_kind: str = CHECKPOINT_KIND_ITERATION
    # True on an epoch checkpoint: the per-epoch LR scheduler step for ``epoch``
    # has *not* run yet (Saver.epoch fires before run()'s lr step), so a resume
    # must replay exactly one epoch-end LR step.
    epoch_scheduler_step_pending: bool = False
    rng_state: Optional[Dict[str, Any]] = None
    sampler_token: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    plugin_state: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "checkpoint_kind": self.checkpoint_kind,
            "epoch_scheduler_step_pending": bool(self.epoch_scheduler_step_pending),
            "rng_state": self.rng_state,
            "sampler_token": self.sampler_token,
            "metrics": dict(self.metrics),
            "plugin_state": dict(self.plugin_state),
            "schema_version": int(self.schema_version),
        }

    # ``to_dict`` alias for call sites that read it as plain data.
    to_dict = state_dict

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingState":
        if not data:
            return cls()
        return cls(
            global_step=int(data.get("global_step", 1)),
            epoch=int(data.get("epoch", 1)),
            batch_in_epoch=int(data.get("batch_in_epoch", 0)),
            checkpoint_kind=data.get("checkpoint_kind", CHECKPOINT_KIND_ITERATION),
            epoch_scheduler_step_pending=bool(
                data.get("epoch_scheduler_step_pending", False)
            ),
            rng_state=data.get("rng_state"),
            sampler_token=data.get("sampler_token"),
            metrics=dict(data.get("metrics", {}) or {}),
            plugin_state=dict(data.get("plugin_state", {}) or {}),
            schema_version=int(data.get("schema_version", CHECKPOINT_SCHEMA_VERSION)),
        )

    def load_state_dict(self, data: Optional[Dict[str, Any]]) -> "TrainingState":
        other = self.from_dict(data)
        self.__dict__.update(other.__dict__)
        return self


def read_resume_metadata(ckpt: Dict[str, Any]) -> TrainingState:
    """Legacy-tolerant reader turning any checkpoint dict into a TrainingState.

    New-shape checkpoints carry an explicit ``training_state`` block.  Old flat
    checkpoints only have top-level ``epoch``/``iteration`` and no kind marker;
    those are interpreted as *epoch-committed* checkpoints (the historical
    ``ep = epoch + 1`` behavior) so old jobs resume exactly as they did before
    this state machine existed.
    """
    ts_blob = ckpt.get("training_state")
    if ts_blob:
        state = TrainingState.from_dict(ts_blob)
        # Trust the flat mirrors if the blob somehow lacked them.
        if "global_step" not in ts_blob and "iteration" in ckpt:
            state.global_step = int(ckpt["iteration"])
        if "epoch" not in ts_blob and "epoch" in ckpt:
            state.epoch = int(ckpt["epoch"])
        return state

    # ---- legacy flat shape -------------------------------------------------
    epoch = int(ckpt.get("epoch", 1))
    iteration = int(ckpt.get("iteration", 1))
    # Prefer any explicit top-level kind (defensive); otherwise old checkpoints
    # are treated as epoch-committed to preserve historical resume behavior.
    kind = ckpt.get("checkpoint_kind", CHECKPOINT_KIND_EPOCH)
    return TrainingState(
        global_step=iteration,
        epoch=epoch,
        batch_in_epoch=0,
        checkpoint_kind=kind,
        # Unknown for legacy checkpoints; do not replay a speculative LR step.
        epoch_scheduler_step_pending=False,
        rng_state=ckpt.get("rng_state"),
        plugin_state=ckpt.get("plugin_state", {}) or {},
        schema_version=int(ckpt.get("checkpoint_schema_version", 0)),
    )


def resolve_rank_rng_state(ckpt: Dict[str, Any], resume: "TrainingState") -> Optional[Dict[str, Any]]:
    """Pick the RNG snapshot THIS process should restore (P0-3).

    Checkpoints carry per-rank snapshots under ``rank_states`` (keyed by global
    rank as a string); the training-state blob's single ``rng_state`` was
    captured on the assembling (main) rank. Restoring that one blob on every
    rank would correlate the ranks' dropout/noise streams, so prefer the
    own-rank entry and fall back to the legacy blob only when absent.
    """
    rank = 0
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:  # pragma: no cover - defensive
        rank = 0
    rank_states = ckpt.get("rank_states") or {}
    own = rank_states.get(str(rank))
    if isinstance(own, dict) and own.get("rng_state") is not None:
        return own["rng_state"]
    return resume.rng_state
