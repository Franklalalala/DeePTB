"""Runtime dynamic-batch OOM-fallback controller for :class:`MultiTrainer`.

When dynamic cost batching is enabled, a single oversized batch can exhaust CUDA
memory. ``MultiTrainer`` supports a *fallback* that, on a caught CUDA OOM,
shrinks the sampler's runtime ``max_cost`` and (for the single-process /
non-consensus paths) *skips* the offending iteration instead of crashing. This
module owns that runtime machinery: OOM detection, gradient/cache clearing,
runtime ``max_cost`` shrink + cache invalidation, the expert-DP consensus gate,
skip accounting, and the OOM-boundary display flush.

Coupling / state ownership
--------------------------
The controller is a *behaviour object*: it holds a back-reference to its owning
trainer (``self._t``) and reads/writes fallback state **on the trainer**
(``dynamic_batch_oom_fallback``, ``dynamic_batch_oom_skipped_iters``,
``train_loader``, ``iter``, ...). State is deliberately left on the trainer so
that code and tests which manipulate ``trainer.dynamic_batch_oom_*`` directly
keep working, and so the controller can be lazily attached to trainers built via
``object.__new__`` (no ``__init__``) — see ``MultiTrainer._dynamic_batch_controller``.

Distributed-safety invariant
-----------------------------
This is a **no-sync** fallback. It must never add or reorder a collective. The
skip path is *gated off* whenever expert-DP consensus would be required
(``requires_expert_dp_consensus``) precisely because a per-rank skip cannot
guarantee every rank makes the same collective sequence. The only
collective-adjacent call here is ``trainer._flush_display_window`` inside
``flush_display_after_oom_skip_if_needed`` — invoked in the exact position and
under the exact ``distributed_expert`` / display-cadence conditions as before the
extraction. ``test_dynamic_cost_batch.py`` pins that no ``dist.all_reduce`` is
issued on the OOM path.

The bodies here are a verbatim move of the former ``MultiTrainer`` methods, with
``self.`` rewritten to ``self._t.`` for trainer state and to ``self.`` only for
sibling controller calls.
"""

import logging
import math
from typing import Any, Dict

import torch

from dptb.nnops.metric_pack import DynamicBatchStat

log = logging.getLogger(__name__)


class DynamicBatchController:
    """Owns runtime dynamic-batch OOM fallback (shrink / skip / consensus)."""

    def __init__(self, trainer):
        self._t = trainer

    # -- OOM detection ---------------------------------------------------
    @staticmethod
    def is_cuda_oom(exc: BaseException) -> bool:
        oom_cls = getattr(torch, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        return "out of memory" in str(exc).lower()

    # -- configuration ---------------------------------------------------
    def disable_oom_fallback(self, message: str, *args):
        log.warning(message, *args)
        self._t.dynamic_batch_oom_fallback = False
        if isinstance(getattr(self._t, "dynamic_batch_options", None), dict):
            self._t.dynamic_batch_options["oom_fallback"] = False

    def configure_oom_fallback(self):
        self._t.dynamic_batch_oom_fallback = bool(self._t.dynamic_batch_options.get("oom_fallback", False))
        self._t.dynamic_batch_oom_shrink_factor = float(self._t.dynamic_batch_options.get("oom_shrink_factor", 0.8))
        self._t.dynamic_batch_oom_skipped_iters = 0
        self._t.dynamic_batch_oom_skipped_since_display = 0

        if not (self._t.dynamic_batch_enabled and self._t.dynamic_batch_oom_fallback):
            return
        if not (0.0 < self._t.dynamic_batch_oom_shrink_factor < 1.0):
            self.disable_oom_fallback(
                "dynamic_batch OOM skip fallback is disabled because oom_shrink_factor=%s is not in (0, 1).",
                self._t.dynamic_batch_oom_shrink_factor,
            )
            return
        if self._t.distributed_rank0_prepare_batch:
            self.disable_oom_fallback(
                "dynamic_batch OOM skip fallback is disabled for distributed_rank0_prepare_batch in this version; "
                "dynamic cost batching remains enabled."
            )
            return
        if self.requires_expert_dp_consensus():
            self.disable_oom_fallback(
                "dynamic_batch OOM skip fallback is disabled for expert_data_parallel_size=%s because "
                "a no-sync fallback cannot guarantee expert-DP rank consensus; dynamic cost batching remains enabled.",
                self._t.expert_data_parallel_size,
            )

    def requires_expert_dp_consensus(self) -> bool:
        return self._t.distributed_expert and int(getattr(self._t, "expert_data_parallel_size", 1)) > 1

    # -- OOM cleanup -----------------------------------------------------
    def clear_after_oom(self):
        for opt in getattr(self._t, "optimizers", []):
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        if self._t._is_cuda_device():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -- runtime max_cost ------------------------------------------------
    def runtime_sampler(self):
        sampler = getattr(getattr(self._t, "train_loader", None), "batch_sampler", None)
        if sampler is None or not hasattr(sampler, "max_cost"):
            return None
        return sampler

    def set_runtime_max_cost(self, sampler, max_cost: int):
        sampler.max_cost = int(max_cost)
        opts = getattr(self._t.train_loader, "dynamic_batch_options", None)
        if isinstance(opts, dict):
            opts["max_cost"] = int(max_cost)
            if "max_edge" in opts:
                opts["max_edge"] = int(max_cost)

    def invalidate_runtime_cache(self):
        invalidate_cache = getattr(self._t.train_loader, "invalidate_dynamic_batch_cache", None)
        if not callable(invalidate_cache):
            return
        try:
            invalidate_cache(clear_costs=False)
        except TypeError:
            invalidate_cache()
        except Exception:
            log.exception("dynamic_batch OOM fallback failed to invalidate cached batches after shrink.")

    def shrink_after_oom(self, batch):
        sampler = self.runtime_sampler()
        if sampler is None:
            return None, None
        old_max = int(sampler.max_cost)
        if old_max <= 1:
            self.disable_oom_fallback(
                "[DYNAMIC_BATCH_OOM_SHRINK] disabled fallback because runtime max_cost=%s cannot shrink further",
                old_max,
            )
            return old_max, old_max
        new_max = max(1, int(math.floor(old_max * self._t.dynamic_batch_oom_shrink_factor)))
        if new_max >= old_max:
            self.disable_oom_fallback(
                "[DYNAMIC_BATCH_OOM_SHRINK] disabled fallback because shrink would not progress: "
                "old_max_cost=%s new_max_cost=%s shrink_factor=%s",
                old_max,
                new_max,
                self._t.dynamic_batch_oom_shrink_factor,
            )
            return old_max, new_max
        self.set_runtime_max_cost(sampler, new_max)
        self.invalidate_runtime_cache()
        log_values = self._t._dynamic_batch_oom_log_values(batch)
        batch_max_item_cost = log_values["batch_max_item_cost"]
        if batch_max_item_cost is not None:
            try:
                if float(batch_max_item_cost) > float(new_max):
                    log.warning(
                        "[DYNAMIC_BATCH_OOM_SHRINK] current batch contains item cost %s above new_max_cost=%s; "
                        "future iterators may still emit oversized singletons unless drop_oversized is enabled.",
                        batch_max_item_cost,
                        new_max,
                    )
            except Exception:
                pass
        log.warning(
            "[DYNAMIC_BATCH_OOM_SHRINK] old_max_cost=%s new_max_cost=%s batch_cost=%s "
            "batch_max_item_cost=%s applies=next_loader_iterator",
            old_max,
            new_max,
            log_values["batch_cost"],
            batch_max_item_cost,
        )
        return old_max, new_max

    # -- skip decision + accounting -------------------------------------
    def can_skip_after_oom(self, ref_batch=None, optimizer_step_started: bool = False) -> bool:
        return (
            self._t.dynamic_batch_enabled
            and self._t.dynamic_batch_oom_fallback
            and not self._t.distributed_rank0_prepare_batch
            and not self.requires_expert_dp_consensus()
            and ref_batch is None
            and not optimizer_step_started
        )

    def record_oom_skip(self):
        self._t.dynamic_batch_oom_skipped_iters = int(getattr(self._t, "dynamic_batch_oom_skipped_iters", 0)) + 1
        self._t.dynamic_batch_oom_skipped_since_display = (
            int(getattr(self._t, "dynamic_batch_oom_skipped_since_display", 0)) + 1
        )
        pack = getattr(self._t, "_display_window_dynamic_batch_pack_local", None)
        oom_slot = DynamicBatchStat.index("oom_skipped_count")
        if torch.is_tensor(pack) and pack.numel() > oom_slot:
            pack[oom_slot] += 1.0

    def flush_display_after_oom_skip_if_needed(self, where: str):
        if not self._t.distributed_expert:
            return
        if not self._t._should_flush_display_window_now(self._t.iter):
            return
        if not hasattr(self._t, "_display_window_pack_local"):
            return
        skipped_since_display = getattr(self._t, "dynamic_batch_oom_skipped_since_display", 0)
        state = self._t._flush_display_window(time_idx=self._t.iter)
        if state is not None:
            log.warning(
                "dynamic_batch OOM fallback joined display flush at iter=%s where=%s skipped_since_display=%s",
                self._t.iter,
                where,
                skipped_since_display,
            )
            self._t.call_plugins(queue_name='iteration', time=self._t.iter, **state)

    def handle_oom_skip(self, batch, *, local_oom: bool, where: str):
        self.clear_after_oom()
        self._t._reset_cuda_memory_peak()
        old_max_cost, new_max_cost = self.shrink_after_oom(batch)
        self.record_oom_skip()
        log_values = self._t._dynamic_batch_oom_log_values(batch)
        log.warning(
            "[DYNAMIC_BATCH_OOM_SKIP] iter=%s rank=%s where=%s local_oom=%s "
            "batch_cost=%s batch_max_item_cost=%s num_graphs=%s old_max_cost=%s new_max_cost=%s "
            "total_skipped=%s skipped_since_display=%s",
            self._t.iter,
            getattr(self._t, "rank", 0),
            where,
            bool(local_oom),
            log_values["batch_cost"],
            log_values["batch_max_item_cost"],
            log_values["num_graphs"],
            old_max_cost,
            new_max_cost,
            self._t.dynamic_batch_oom_skipped_iters,
            self._t.dynamic_batch_oom_skipped_since_display,
        )
        self.flush_display_after_oom_skip_if_needed(where)
        if self._t.distributed_expert:
            self._t.iter += 1
        return None

    def maybe_skip_after_oom(self, batch, *, local_oom: bool, where: str, ref_batch=None) -> bool:
        if not self.can_skip_after_oom(ref_batch=ref_batch):
            return False
        if not local_oom:
            return False
        self.handle_oom_skip(batch, local_oom=local_oom, where=where)
        return True
