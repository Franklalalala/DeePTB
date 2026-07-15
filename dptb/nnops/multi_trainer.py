import contextlib
import time
import logging
import copy
import heapq
import math
import os
from typing import Union, Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.profiler import profile as torch_profile, ProfilerActivity

from dptb.configuration import migrate_legacy_checkpoint_train_options
from dptb.utils.tools import (
    get_lr_scheduler,
    get_optimizer,
    lr_scheduler_can_step_without_metric,
    lr_scheduler_requires_metric,
)
from dptb.utils.cuda_cache_memory import cuda_cache_memory_context
from dptb.data import AtomicDataset, AtomicData, DataLoader
from dptb.data import _keys
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nnops.trainer import Trainer
from dptb.nnops.ddp_utils import merge_restart_train_options
from dptb.nnops.metric_pack import (
    MetricPack,
    DynamicBatchStat,
    ExpertDisplayMetric,
)
from dptb.nnops.metric_reducer import MetricReducer
from dptb.nnops.dynamic_batch_controller import DynamicBatchController
from dptb.nnops.objective import Objective, FlowObjective
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    read_resume_metadata,
    resolve_rank_rng_state,
    restore_rng_state,
)
from dptb.nnops.expert_parallel_layout import (
    rank_to_expert_parallel,
    resolve_expert_parallel_layout,
)
from dptb.nn.build import build_model
from dptb.nn.activation_recompute import configure_activation_recompute

log = logging.getLogger(__name__)


def _resolve_local_expert_dp_batch_size(
    batch_size: int,
    *,
    expert_data_parallel_size: int,
    semantics: str = "global",
    option_name: str = "batch_size",
) -> int:
    batch_size = int(batch_size)
    expert_data_parallel_size = int(expert_data_parallel_size)
    semantics = str(semantics).lower()
    if batch_size <= 0:
        raise ValueError(f"{option_name} must be positive, got {batch_size}")
    if expert_data_parallel_size <= 1:
        return batch_size
    if semantics in ("local", "per_rank", "per-replica", "replica"):
        return batch_size
    if semantics not in ("global", "same_expert_global", "per_expert_global"):
        raise ValueError(
            f"{option_name} expert DP batch size semantics must be 'global' or 'local', "
            f"got {semantics!r}"
        )
    if batch_size % expert_data_parallel_size != 0:
        raise ValueError(
            f"{option_name}={batch_size} is interpreted as same-expert global batch "
            f"and must be divisible by expert_data_parallel_size={expert_data_parallel_size}. "
            "Set the corresponding expert DP batch size semantics to 'local' to opt into per-rank semantics."
        )
    return batch_size // expert_data_parallel_size


def _state_dict_has_overlap_head(state_dict) -> bool:
    """Whether a (wrapper) state dict contains overlap-head parameters.

    Used by restart to reconcile a checkpoint whose config claims
    ``overlap=False`` while its weights carry the overlap head (written by an
    entrypoint version that mutated the flag after model construction).
    """
    try:
        keys = state_dict.keys()
    except AttributeError:  # pragma: no cover - defensive
        return False
    return any(
        ("overlaponsite_param" in k) or ("edge_prediction_s." in k)
        for k in keys
    )


def _base_train_options_for_multitrainer(train_options: dict) -> dict:
    base_options = copy.deepcopy(train_options)
    dynamic_batch = base_options.get("dynamic_batch", None)
    if isinstance(dynamic_batch, dict) and dynamic_batch.get("enabled", False):
        dynamic_batch = copy.deepcopy(dynamic_batch)
        dynamic_batch["enabled"] = False
        base_options["dynamic_batch"] = dynamic_batch
    return base_options


class _StageTagger:
    def __init__(
        self,
        trainer,
        enabled: bool,
        freq: int,
        cuda_mem: bool,
        cuda_sync: bool,
        oom_dump: bool,
        reset_peak: bool = True,
    ):
        self.trainer = trainer
        self.enabled = bool(enabled)
        self.freq = max(int(freq), 1)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)
        self.oom_dump = bool(oom_dump)
        self.reset_peak = bool(reset_peak)

    def _device(self) -> torch.device:
        return self.trainer.device if isinstance(self.trainer.device, torch.device) else torch.device(self.trainer.device)

    def _is_cuda(self) -> bool:
        dev = self._device()
        return torch.cuda.is_available() and dev.type == "cuda"

    def _cuda_mem(self):
        if not self._is_cuda():
            return None
        dev = self._device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        free, total = torch.cuda.mem_get_info(dev)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None:
            return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return (
            f" | cuda_alloc={alloc/mb:.1f}MB"
            f" cuda_reserved={reserved/mb:.1f}MB"
            f" cuda_peak={peak/mb:.1f}MB"
            f" free={free/mb:.1f}MB total={total/mb:.1f}MB"
        )

    def dump_cuda_mem_summary(self, where: str):
        if not self._is_cuda():
            return
        dev = self._device()
        mb = 1024 ** 2
        alloc = torch.cuda.memory_allocated(dev) / mb
        reserved = torch.cuda.memory_reserved(dev) / mb
        peak = torch.cuda.max_memory_allocated(dev) / mb
        free, total = torch.cuda.mem_get_info(dev)
        log.error(
            f"[OOM-DUMP] where={where} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB "
            f"free={free/mb:.1f}MB total={total/mb:.1f}MB"
        )
        if self.oom_dump:
            try:
                log.error("[OOM-DUMP] memory_summary:\n%s", torch.cuda.memory_summary(dev, abbreviated=False))
            except Exception:
                pass

    @contextlib.contextmanager
    def tag(self, name: str, *, it: Optional[int] = None, expert: Optional[int] = None, extra: str = ""):
        if not self.enabled:
            yield
            return
        if it is not None and (it % self.freq != 0):
            yield
            return

        prefix = f"[TAG][it={it}]"
        if expert is not None:
            prefix += f"[expert={expert}]"
        prefix += f"[{name}]"

        nvtx_pushed = False
        if self._is_cuda():
            try:
                torch.cuda.nvtx.range_push(f"{prefix}{(' ' + extra) if extra else ''}")
                nvtx_pushed = True
            except Exception:
                nvtx_pushed = False

        dev = self._device()
        if self.cuda_mem and self.reset_peak and self._is_cuda():
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        t0 = time.perf_counter()

        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.dump_cuda_mem_summary(where=f"{name} it={it} expert={expert}")
            raise
        finally:
            if self.cuda_sync and self._is_cuda():
                torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem() if self.cuda_mem else None
            log.info(f"{prefix} dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")

            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


class MultiTrainer(Trainer):
    object_keys = ["lr_schedulers", "optimizers"]
    # Preserve the established stitched-endpoint scheduler semantics for the
    # multi-expert trainer; the single-trainer regression does not apply here.
    scheduler_metric_prefers_objective = False

    _P_LOSS_OPT_SUM = 0
    _P_ONSITE_WEIGHTED_SUM = 1
    _P_HOPPING_WEIGHTED_SUM = 2
    _P_ACTIVE_NODES_SUM = 3
    _P_ACTIVE_EDGES_SUM = 4
    _P_ONSITE_L1_SUM = 5
    _P_ONSITE_MSE_SUM = 6
    _P_ONSITE_CNT_SUM = 7
    _P_HOPPING_L1_SUM = 8
    _P_HOPPING_MSE_SUM = 9
    _P_HOPPING_CNT_SUM = 10
    _P_Z_SUM = 11
    _P_Z_CNT = 12
    _P_CV_SUM = 13
    _P_CV_CNT = 14
    _P_GRAD_NORM_SUM = 15
    _P_STEP_COUNT = 16
    _PACK_LEN = 17
    _DB_NUM_GRAPHS_SUM = 0
    _DB_COST_SUM = 1
    _DB_NUM_NODES_SUM = 2
    _DB_NUM_EDGES_SUM = 3
    _DB_MAX_ITEM_COST_SUM = 4
    _DB_STEP_COUNT = 5
    _DB_OOM_SKIPPED_COUNT = 6
    _DB_PACK_LEN = 7
    _DYNAMIC_BATCH_STATE_ATTRS = (
        ("__dptb_batch_cost__", "batch_cost"),
        ("__dptb_batch_num_graphs__", "batch_num_graphs"),
        ("__dptb_batch_num_nodes__", "batch_num_nodes"),
        ("__dptb_batch_num_edges__", "batch_num_edges"),
        ("__dptb_batch_max_item_cost__", "batch_max_item_cost"),
    )

    def __init__(
        self,
        distance_ranges: list,
        train_options: dict,
        common_options: dict,
        model: torch.nn.Module,
        train_datasets: AtomicDataset,
        reference_datasets: Union[AtomicDataset, None] = None,
        validation_datasets: Union[AtomicDataset, None] = None,
        distributed_expert: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        trainer_common_options = copy.deepcopy(common_options)
        if distributed_expert:
            trainer_common_options["device"] = "cpu"
        trainer_train_options = _base_train_options_for_multitrainer(train_options)

        super().__init__(
            train_options=trainer_train_options,
            common_options=trainer_common_options,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )
        self.common_options = common_options
        self.train_options = train_options
        if self.use_reference:
            self.reference_datasets = getattr(self, "reference_datesets", None)
        else:
            self.reference_datasets = None

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)

        self.distributed_expert = bool(distributed_expert)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.distributed_expert:
            layout = resolve_expert_parallel_layout(
                num_experts=self.num_experts,
                world_size=self.world_size,
                train_options=self.train_options,
            )
            rank_info = rank_to_expert_parallel(
                rank=self.rank,
                num_experts=layout.num_experts,
                expert_data_parallel_size=layout.expert_data_parallel_size,
            )
            self.expert_data_parallel_size = layout.expert_data_parallel_size
            self.local_expert_idx = rank_info.local_expert_idx
            self.expert_dp_rank = rank_info.expert_dp_rank
            self.expert_group_ranks = rank_info.expert_group_ranks
            self.expert_group_src_rank = self.expert_group_ranks[0]
            self.train_options["expert_data_parallel_size"] = self.expert_data_parallel_size
        else:
            self.expert_data_parallel_size = 1
            self.local_expert_idx = None
            self.expert_dp_rank = 0
            self.expert_group_ranks = []
            self.expert_group_src_rank = 0
        self.is_main_process = (not self.distributed_expert) or (self.rank == 0)
        if self.distributed_expert:
            self.device = common_options["device"]
            self._move_aux_modules_to_device(self._device_obj())

        self.parallel_multi = bool(self.train_options.get("parallel_multi", False))
        if self.distributed_expert:
            if self.parallel_multi:
                log.warning("distributed_expert=True: parallel_multi will be disabled.")
            self.parallel_multi = False
            self.train_options["parallel_multi"] = False

        self.debug_tags = bool(self.train_options.get("debug_tags", False))
        self.debug_tag_freq = int(self.train_options.get("debug_tag_freq", 1))
        self.debug_tag_cuda_mem = bool(self.train_options.get("debug_tag_cuda_mem", True))
        self.debug_tag_cuda_sync = bool(self.train_options.get("debug_tag_cuda_sync", False))
        self.debug_oom_dump = bool(self.train_options.get("debug_oom_dump", True))
        self.monitor_cuda_memory = bool(self.train_options.get("monitor_cuda_memory", True))
        debug_tag_reset_peak_opt = self.train_options.get("debug_tag_reset_peak", None)
        self.debug_tag_reset_peak = (
            not self.monitor_cuda_memory
            if debug_tag_reset_peak_opt is None
            else bool(debug_tag_reset_peak_opt)
        )
        if self.debug_tags and self.debug_tag_cuda_mem and self.monitor_cuda_memory and self.debug_tag_reset_peak:
            log.warning(
                "monitor_cuda_memory=True with debug_tag_reset_peak=True will reset CUDA peak counters inside "
                "debug tags and make regular cuda_peak_* fields stage-local instead of window-local."
            )

        self.debug_profile = bool(self.train_options.get("debug_profile", False))
        self.debug_profile_start_iter = int(self.train_options.get("debug_profile_start_iter", 5))
        self.debug_profile_end_iter = int(
            self.train_options.get("debug_profile_end_iter", self.debug_profile_start_iter)
        )
        self.debug_profile_dir = self.train_options.get("debug_profile_dir", None)

        self.display_sync_freq = max(int(self.train_options.get("display_freq", 1)), 1)
        self.expert_dp_batch_size_semantics = str(
            self.train_options.get("expert_dp_batch_size_semantics", "global")
        ).lower()
        eval_batch_size_semantics = self.train_options.get("expert_dp_eval_batch_size_semantics", "local") or "local"
        self.expert_dp_train_batch_size_semantics = str(
            self.train_options.get("expert_dp_train_batch_size_semantics")
            or self.expert_dp_batch_size_semantics
        ).lower()
        self.expert_dp_ref_batch_size_semantics = str(
            self.train_options.get("expert_dp_ref_batch_size_semantics")
            or eval_batch_size_semantics
        ).lower()
        self.expert_dp_val_batch_size_semantics = str(
            self.train_options.get("expert_dp_val_batch_size_semantics")
            or eval_batch_size_semantics
        ).lower()
        self.sync_expert_dp_buffers = bool(self.train_options.get("sync_expert_dp_buffers", True))
        self.expert_dp_grad_sync_mode = str(
            self.train_options.get("expert_dp_grad_sync_mode", "coalesced")
        ).lower()
        self.expert_dp_use_ddp = bool(self.train_options.get("expert_dp_use_ddp", False))
        self.expert_dp_backend = str(self.train_options.get("expert_dp_backend", "manual")).lower()
        if self.expert_dp_use_ddp:
            self.expert_dp_backend = "ddp"
            self.train_options["expert_dp_backend"] = "ddp"
        self.expert_dp_ddp_static_graph = bool(
            self.train_options.get("expert_dp_ddp_static_graph", False)
        )
        self.expert_dp_ddp_gradient_as_bucket_view = bool(
            self.train_options.get("expert_dp_ddp_gradient_as_bucket_view", False)
        )
        self.expert_dp_ddp_find_unused_parameters = bool(
            self.train_options.get("expert_dp_ddp_find_unused_parameters", True)
        )
        self.expert_dp_ddp_broadcast_buffers = bool(
            self.train_options.get("expert_dp_ddp_broadcast_buffers", False)
        )
        self.expert_dp_ddp_bucket_cap_mb = self.train_options.get("expert_dp_ddp_bucket_cap_mb", None)
        if self.expert_dp_ddp_bucket_cap_mb is not None:
            self.expert_dp_ddp_bucket_cap_mb = float(self.expert_dp_ddp_bucket_cap_mb)
        if self.expert_dp_backend not in ("manual", "ddp"):
            raise ValueError(
                "expert_dp_backend must be 'manual' or 'ddp', "
                f"got {self.expert_dp_backend!r}"
            )
        self.expert_dp_grad_check_mode = str(
            self.train_options.get("expert_dp_grad_check_mode", "auto")
        ).lower()
        self.expert_dp_grad_bucket_mb = float(self.train_options.get("expert_dp_grad_bucket_mb", 64))
        self.expert_dp_buffer_sync_mode = str(
            self.train_options.get("expert_dp_buffer_sync_mode", "coalesced")
        ).lower()
        self.expert_dp_buffer_bucket_mb = float(self.train_options.get("expert_dp_buffer_bucket_mb", 64))
        self._expert_dp_coalesced_warned = False
        self.distributed_rank0_prepare_batch = bool(
            self.train_options.get("distributed_rank0_prepare_batch", False)
        )
        if self.distributed_expert and self.expert_data_parallel_size > 1 and self.distributed_rank0_prepare_batch:
            log.warning(
                "expert_data_parallel_size > 1 requires independent data shards per replica; "
                "force disable distributed_rank0_prepare_batch."
            )
            self.distributed_rank0_prepare_batch = False
            self.train_options["distributed_rank0_prepare_batch"] = False
        self.precompute_lem_active_edges = bool(
            self.train_options.get("precompute_lem_active_edges", True)
        )
        self.precompute_lem_cutoff_coeffs = bool(
            self.train_options.get("precompute_lem_cutoff_coeffs", True)
        )
        if self.precompute_lem_cutoff_coeffs:
            self._validate_lem_cutoff_precompute_options()
            log.warning(
                "precompute_lem_cutoff_coeffs=True moves cutoff coefficients out of model forward. "
                "Use only for fixed-geometry Hamiltonian training where geometry gradients are not needed."
            )
        self._lem_cutoff_init_layer = None
        self._lem_cutoff_precompute_checked = False
        self._lem_cutoff_precompute_warned = False

        # dataloader options
        self.train_num_workers = int(self.train_options.get("train_num_workers", self.train_options.get("num_workers", 0)))
        self.ref_num_workers = int(self.train_options.get("ref_num_workers", self.train_num_workers))
        self.val_num_workers = int(self.train_options.get("val_num_workers", self.train_num_workers))

        dev_obj = self._device_obj()
        self.data_pin_memory = bool(self.train_options.get("data_pin_memory", dev_obj.type == "cuda"))

        self.data_persistent_workers = bool(self.train_options.get("data_persistent_workers", self.train_num_workers > 0))
        self.data_prefetch_factor = int(self.train_options.get("data_prefetch_factor", 2))
        self.expert_dp_train_sampler_drop_last = bool(
            self.train_options.get("expert_dp_train_sampler_drop_last", False)
        )
        self.expert_dp_ref_sampler_drop_last = bool(
            self.train_options.get("expert_dp_ref_sampler_drop_last", False)
        )
        self.expert_dp_val_sampler_drop_last = bool(
            self.train_options.get("expert_dp_val_sampler_drop_last", False)
        )
        dynamic_batch_cfg = self.train_options.get("dynamic_batch", None)
        self.dynamic_batch_options = dynamic_batch_cfg if isinstance(dynamic_batch_cfg, dict) else {}
        self.dynamic_batch_enabled = bool(self.dynamic_batch_options.get("enabled", False))
        self._configure_dynamic_batch_oom_fallback()

        self._tagger = _StageTagger(
            trainer=self,
            enabled=self.debug_tags,
            freq=self.debug_tag_freq,
            cuda_mem=self.debug_tag_cuda_mem,
            cuda_sync=self.debug_tag_cuda_sync,
            oom_dump=self.debug_oom_dump,
            reset_peak=self.debug_tag_reset_peak,
        )
        self.expert_dp_process_group = self._create_expert_dp_process_group()

        self.endpoint_loss_mode = str(
            self.train_options.get("endpoint_loss_mode", "reduce")
        ).lower()
        if self.endpoint_loss_mode not in {"reduce", "full_forward"}:
            raise ValueError(
                "train_options.endpoint_loss_mode must be 'reduce' or "
                f"'full_forward', got {self.endpoint_loss_mode!r}."
            )

        if self.distributed_expert and self.endpoint_loss_mode == "full_forward":
            log.warning(
                "distributed_expert=True does not support full stitched forward across GPUs. "
                "Fallback endpoint_loss_mode from 'full_forward' to 'reduce'."
            )
            self.endpoint_loss_mode = "reduce"

        # ---------------- per-expert optimizer / scheduler overrides ----------------
        self.expert_lrs = self._parse_expert_lrs(self.train_options.get("expert_lrs", None))
        self.expert_optimizer_overrides = self._parse_expert_config_overrides(
            self.train_options.get("expert_optimizer_overrides", None),
            field_name="train_options.expert_optimizer_overrides",
        )
        self.expert_lr_scheduler_overrides = self._parse_expert_config_overrides(
            self.train_options.get("expert_lr_scheduler_overrides", None),
            field_name="train_options.expert_lr_scheduler_overrides",
        )
        # ----------------------------------------------------------------------------

        log.info(
            f"[MultiTrainer][rank={self.rank}] num_experts={self.num_experts}, "
            f"distributed_expert={self.distributed_expert}, parallel_multi={self.parallel_multi}, "
            f"expert_data_parallel_size={self.expert_data_parallel_size}, "
            f"local_expert_idx={self.local_expert_idx}, expert_dp_rank={self.expert_dp_rank}, "
            f"display_sync_freq={self.display_sync_freq}, "
            f"expert_dp_use_ddp={self.expert_dp_use_ddp}, "
            f"expert_dp_backend={self.expert_dp_backend}, "
            f"expert_dp_grad_sync_mode={self.expert_dp_grad_sync_mode}, "
            f"expert_dp_grad_check_mode={self.expert_dp_grad_check_mode}, "
            f"expert_dp_grad_bucket_mb={self.expert_dp_grad_bucket_mb}, "
            f"expert_dp_batch_size_semantics={self.expert_dp_batch_size_semantics}, "
            f"expert_dp_train_batch_size_semantics={self.expert_dp_train_batch_size_semantics}, "
            f"expert_dp_ref_batch_size_semantics={self.expert_dp_ref_batch_size_semantics}, "
            f"expert_dp_val_batch_size_semantics={self.expert_dp_val_batch_size_semantics}, "
            f"expert_dp_train_sampler_drop_last={self.expert_dp_train_sampler_drop_last}, "
            f"expert_dp_ref_sampler_drop_last={self.expert_dp_ref_sampler_drop_last}, "
            f"expert_dp_val_sampler_drop_last={self.expert_dp_val_sampler_drop_last}, "
            f"expert_dp_ddp_static_graph={self.expert_dp_ddp_static_graph}, "
            f"expert_dp_ddp_gradient_as_bucket_view={self.expert_dp_ddp_gradient_as_bucket_view}, "
            f"expert_dp_ddp_find_unused_parameters={self.expert_dp_ddp_find_unused_parameters}, "
            f"expert_dp_ddp_broadcast_buffers={self.expert_dp_ddp_broadcast_buffers}, "
            f"expert_dp_ddp_bucket_cap_mb={self.expert_dp_ddp_bucket_cap_mb}, "
            f"expert_dp_buffer_sync_mode={self.expert_dp_buffer_sync_mode}, "
            f"expert_dp_buffer_bucket_mb={self.expert_dp_buffer_bucket_mb}, "
            f"distributed_rank0_prepare_batch={self.distributed_rank0_prepare_batch}, "
            f"train_num_workers={self.train_num_workers}, ref_num_workers={self.ref_num_workers}, val_num_workers={self.val_num_workers}, "
            f"pin_memory={self.data_pin_memory}, persistent_workers={self.data_persistent_workers}, prefetch_factor={self.data_prefetch_factor}, "
            f"dynamic_batch_enabled={self.dynamic_batch_enabled}, dynamic_batch_oom_fallback={self.dynamic_batch_oom_fallback}, "
            f"endpoint_loss_mode={self.endpoint_loss_mode}, "
            f"expert_lrs={'(default optimizer.lr)' if self.expert_lrs is None else self.expert_lrs}, "
            f"expert_optimizer_overrides={self._summarize_expert_override_list(self.expert_optimizer_overrides)}, "
            f"expert_lr_scheduler_overrides={self._summarize_expert_override_list(self.expert_lr_scheduler_overrides)}."
        )

        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

        self.activation_recompute_state = configure_activation_recompute(
            self.model,
            self.train_options.get("activation_recompute", None),
        )

        if self.distributed_expert:
            self._materialize_local_expert_only()
            self._sync_local_expert_parameters()
            self._maybe_wrap_local_expert_ddp()

        def _make_opt_for_expert(i: int):
            opt_cfg = self._build_optimizer_cfg_for_expert(i)
            return get_optimizer(model_param=self._expert_optimizer_parameters(i), **opt_cfg)

        def _make_scheduler_for_expert(i: int, opt):
            sch_cfg = self._build_lr_scheduler_cfg_for_expert(i)
            return get_lr_scheduler(optimizer=opt, **sch_cfg)

        if self.distributed_expert:
            self.optimizers = [None] * self.num_experts
            self.lr_schedulers = [None] * self.num_experts
            idx = self.local_expert_idx

            opt = _make_opt_for_expert(idx)
            sch = _make_scheduler_for_expert(idx, opt)
            self.optimizers[idx] = opt
            self.lr_schedulers[idx] = sch
        else:
            self.optimizers = []
            self.lr_schedulers = []
            for i in range(self.num_experts):
                opt = _make_opt_for_expert(i)
                sch = _make_scheduler_for_expert(i, opt)
                self.optimizers.append(opt)
                self.lr_schedulers.append(sch)

        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "lr_scheduler"):
            del self.lr_scheduler

        self._maybe_rebuild_loaders_in_multi_trainer()
        self._warn_non_expert_trainables()
        self._t_last_iter_end: Optional[float] = None
        self._reset_display_window_buffers()

    # ---------------- per-expert optimizer / scheduler parsing & checks ----------------
    def _parse_expert_lrs(self, expert_lrs) -> Optional[List[float]]:
        """
        train_options.expert_lrs:
          - None / []: disabled
          - list/tuple of float with length == num_experts: enabled
        """
        if expert_lrs is None:
            return None
        if isinstance(expert_lrs, (list, tuple)):
            if len(expert_lrs) == 0:
                return None
            if len(expert_lrs) != self.num_experts:
                raise ValueError(
                    f"train_options.expert_lrs length must match num_experts={self.num_experts}, "
                    f"got len(expert_lrs)={len(expert_lrs)}"
                )
            lrs = [float(x) for x in expert_lrs]
            bad = [i for i, lr in enumerate(lrs) if not (lr > 0.0)]
            if bad:
                raise ValueError(f"train_options.expert_lrs must be all > 0.0, bad indices: {bad}, values={lrs}")
            return lrs
        raise TypeError(f"train_options.expert_lrs must be a list/tuple of float (or empty/None), got {type(expert_lrs)}")

    def _parse_expert_config_overrides(self, overrides, field_name: str) -> Optional[List[Dict[str, Any]]]:
        if overrides is None:
            return None
        if isinstance(overrides, (list, tuple)):
            overrides = list(overrides)
            if len(overrides) == 0:
                return None
            if len(overrides) != self.num_experts:
                if len(overrides) == 1 and self.num_experts > 1:
                    overrides = overrides * self.num_experts
                elif self.num_experts == 1 and all(item == overrides[0] for item in overrides[1:]):
                    log.warning(
                        "%s has %d identical entries for num_experts=1; using the first entry.",
                        field_name,
                        len(overrides),
                    )
                    overrides = overrides[:1]
                else:
                    raise ValueError(
                        f"{field_name} length must match num_experts={self.num_experts} "
                        f"(or be a single shared override), got len({field_name})={len(overrides)}"
                    )
            parsed = []
            for idx, item in enumerate(overrides):
                if item is None:
                    parsed.append({})
                elif isinstance(item, dict):
                    parsed.append(copy.deepcopy(item))
                else:
                    raise TypeError(
                        f"{field_name}[{idx}] must be dict or null/None, got {type(item)}"
                    )
            return parsed
        raise TypeError(f"{field_name} must be a list/tuple of dict (or empty/None), got {type(overrides)}")

    @staticmethod
    def _summarize_expert_override_list(overrides: Optional[List[Dict[str, Any]]]) -> str:
        if overrides is None:
            return "(shared base config)"
        active = [idx for idx, item in enumerate(overrides) if item]
        if not active:
            return "(shared base config)"
        return f"active_experts={active}"

    def _build_optimizer_cfg_for_expert(self, expert_idx: int) -> Dict[str, Any]:
        opt_cfg = copy.deepcopy(self.train_options["optimizer"])
        opt_override = None
        if self.expert_optimizer_overrides is not None:
            opt_override = self.expert_optimizer_overrides[expert_idx]
            opt_cfg.update(opt_override)
        if self.expert_lrs is not None:
            lr_overridden_in_opt = isinstance(opt_override, dict) and ("lr" in opt_override)
            if not lr_overridden_in_opt:
                opt_cfg["lr"] = float(self.expert_lrs[expert_idx])
        return opt_cfg

    def _build_lr_scheduler_cfg_for_expert(self, expert_idx: int) -> Dict[str, Any]:
        sch_cfg = copy.deepcopy(self.train_options["lr_scheduler"])
        if self.expert_lr_scheduler_overrides is not None:
            sch_cfg.update(self.expert_lr_scheduler_overrides[expert_idx])
        return sch_cfg
    # -----------------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # dataloader rebuild in MultiTrainer only
    # ---------------------------------------------------------------------

    def _dynamic_batch_cfg_for_train_loader(self):
        if not self.dynamic_batch_enabled:
            return None
        cfg = copy.deepcopy(self.dynamic_batch_options)
        if self.distributed_expert and (not self.distributed_rank0_prepare_batch) and self.expert_data_parallel_size > 1:
            cfg["rank"] = self.expert_dp_rank
            cfg["world_size"] = self.expert_data_parallel_size
        elif (
            bool(cfg.get("use_global_dist", False))
            and not self.distributed_expert
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            cfg["rank"] = torch.distributed.get_rank()
            cfg["world_size"] = torch.distributed.get_world_size()
        else:
            if (
                not self.distributed_expert
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and not bool(cfg.get("use_global_dist", False))
                and not getattr(self, "_warned_dynamic_batch_global_dist_unsharded", False)
            ):
                log.warning(
                    "dynamic_batch is using rank=0/world_size=1 while torch.distributed "
                    "is initialized. This is intentional for expert-shared batches; "
                    "ordinary DDP should set dynamic_batch.use_global_dist=true."
                )
                self._warned_dynamic_batch_global_dist_unsharded = True
            cfg["rank"] = 0
            cfg["world_size"] = 1
        return cfg

    def _make_loader_compat(self, dataset, batch_size, shuffle, num_workers, sampler=None, dynamic_batch=None):
        kwargs = {"num_workers": int(num_workers)}
        if dynamic_batch is not None:
            sampler = None
        if sampler is not None:
            kwargs["sampler"] = sampler
            shuffle = False
        if kwargs["num_workers"] > 0:
            kwargs["pin_memory"] = self.data_pin_memory
            kwargs["persistent_workers"] = self.data_persistent_workers
            kwargs["prefetch_factor"] = self.data_prefetch_factor
        else:
            kwargs["pin_memory"] = self.data_pin_memory

        trial_kwargs = [
            kwargs,
            {k: v for k, v in kwargs.items() if k != "prefetch_factor"},
            {k: v for k, v in kwargs.items() if k not in ("prefetch_factor", "persistent_workers")},
            {k: v for k, v in kwargs.items() if k != "pin_memory"},
            {},
        ]
        if sampler is not None:
            trial_kwargs = [kw for kw in trial_kwargs if "sampler" in kw]

        last_err = None
        for kw in trial_kwargs:
            try:
                return DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    dynamic_batch=dynamic_batch,
                    **kw,
                )
            except TypeError as e:
                last_err = e
                continue
        if sampler is not None:
            raise RuntimeError(
                "expert data parallel loaders require DataLoader sampler support; "
                "refusing to fall back to an unsharded loader."
            ) from last_err
        raise last_err

    def _make_expert_dp_sampler(self, dataset, *, shuffle: bool, drop_last: bool = False):
        if not self.distributed_expert or self.expert_data_parallel_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.expert_data_parallel_size,
            rank=self.expert_dp_rank,
            shuffle=shuffle,
            drop_last=bool(drop_last),
        )

    def _expert_dp_batch_size_semantics_for(self, option_name: str) -> str:
        if option_name == "batch_size":
            return self.expert_dp_train_batch_size_semantics
        if option_name == "ref_batch_size":
            return self.expert_dp_ref_batch_size_semantics
        if option_name == "val_batch_size":
            return self.expert_dp_val_batch_size_semantics
        return self.expert_dp_batch_size_semantics

    def _local_expert_dp_batch_size(self, option_name: str) -> int:
        return _resolve_local_expert_dp_batch_size(
            self.train_options[option_name],
            expert_data_parallel_size=self.expert_data_parallel_size,
            semantics=self._expert_dp_batch_size_semantics_for(option_name),
            option_name=option_name,
        )

    def _maybe_rebuild_loaders_in_multi_trainer(self):
        worker_keys = {
            "train_num_workers", "ref_num_workers", "val_num_workers",
            "data_pin_memory", "data_persistent_workers", "data_prefetch_factor"
        }
        need_rebuild = (
            self.distributed_expert or
            self.distributed_rank0_prepare_batch or
            self.dynamic_batch_enabled or
            any(k in self.train_options for k in worker_keys)
        )

        if not need_rebuild:
            return

        train_workers = self.train_num_workers
        ref_workers = self.ref_num_workers
        val_workers = self.val_num_workers

        if self.distributed_expert and self.distributed_rank0_prepare_batch and self.rank != 0:
            train_workers = 0
            ref_workers = 0
            val_workers = 0

        train_dynamic_batch = self._dynamic_batch_cfg_for_train_loader()
        train_sampler = None if train_dynamic_batch is not None else self._make_expert_dp_sampler(
            self.train_datasets,
            shuffle=True,
            drop_last=self.expert_dp_train_sampler_drop_last,
        )
        train_batch_size = self._local_expert_dp_batch_size("batch_size")
        self.train_loader = self._make_loader_compat(
            dataset=self.train_datasets,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_workers,
            sampler=train_sampler,
            dynamic_batch=train_dynamic_batch,
        )

        if self.use_reference:
            ref_sampler = self._make_expert_dp_sampler(
                self.reference_datasets,
                shuffle=True,
                drop_last=self.expert_dp_ref_sampler_drop_last,
            )
            ref_batch_size = self._local_expert_dp_batch_size("ref_batch_size")
            self.reference_loader = self._make_loader_compat(
                dataset=self.reference_datasets,
                batch_size=ref_batch_size,
                shuffle=True,
                num_workers=ref_workers,
                sampler=ref_sampler,
            )

        if self.use_validation:
            val_sampler = self._make_expert_dp_sampler(
                self.validation_datasets,
                shuffle=False,
                drop_last=self.expert_dp_val_sampler_drop_last,
            )
            val_batch_size = self._local_expert_dp_batch_size("val_batch_size")
            self.validation_loader = self._make_loader_compat(
                dataset=self.validation_datasets,
                batch_size=val_batch_size,
                shuffle=not self.distributed_expert,
                num_workers=val_workers,
                sampler=val_sampler,
            )

        log.info(
            f"[MultiTrainer][rank={self.rank}] rebuilt loaders in MultiTrainer: "
            f"train_workers={train_workers}, ref_workers={ref_workers}, val_workers={val_workers}, "
            f"global_batch_size={self.train_options['batch_size']}, local_batch_size={train_batch_size}, "
            f"ref_batch_size={self.train_options.get('ref_batch_size')}, local_ref_batch_size={self._local_expert_dp_batch_size('ref_batch_size') if self.use_reference else None}, "
            f"val_batch_size={self.train_options.get('val_batch_size')}, local_val_batch_size={self._local_expert_dp_batch_size('val_batch_size') if self.use_validation else None}, "
            f"expert_dp_train_batch_size_semantics={self.expert_dp_train_batch_size_semantics}, "
            f"expert_dp_ref_batch_size_semantics={self.expert_dp_ref_batch_size_semantics}, "
            f"expert_dp_val_batch_size_semantics={self.expert_dp_val_batch_size_semantics}, "
            f"dynamic_batch={getattr(self.train_loader, 'dynamic_batch_options', None)}"
        )

    def _set_expert_dp_sampler_epoch(self, epoch: int):
        for loader_name in ("train_loader", "reference_loader", "validation_loader"):
            loader = getattr(self, loader_name, None)
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(epoch))
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(int(epoch))

    # ---------------------------------------------------------------------
    # dist helpers
    # ---------------------------------------------------------------------

    def _dist_ready(self):
        return self.distributed_expert and dist.is_available() and dist.is_initialized()

    def _device_obj(self):
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda_device(self):
        return self._device_obj().type == "cuda" and torch.cuda.is_available()

    def _cuda_memory_monitor_enabled(self) -> bool:
        return bool(getattr(self, "monitor_cuda_memory", False)) and self._is_cuda_device()

    def _reset_cuda_memory_peak(self):
        if not self._cuda_memory_monitor_enabled():
            return
        try:
            torch.cuda.reset_peak_memory_stats(self._device_obj())
        except Exception:
            pass

    def _cuda_memory_tensor(self) -> Optional[torch.Tensor]:
        if not self._cuda_memory_monitor_enabled():
            return None

        dev = self._device_obj()
        mb = 1024 ** 2
        try:
            free, total = torch.cuda.mem_get_info(dev)
            peak_reserved = (
                torch.cuda.max_memory_reserved(dev)
                if hasattr(torch.cuda, "max_memory_reserved")
                else torch.cuda.memory_reserved(dev)
            )
            values = [
                torch.cuda.memory_allocated(dev) / mb,
                torch.cuda.memory_reserved(dev) / mb,
                torch.cuda.max_memory_allocated(dev) / mb,
                peak_reserved / mb,
                free / mb,
                total / mb,
            ]
        except Exception:
            return None

        return torch.tensor(values, dtype=self.dtype, device=dev)

    def _gather_cuda_memory_metrics(self) -> List[torch.Tensor]:
        local_metric = self._cuda_memory_tensor()
        if local_metric is None:
            return []
        if not self._dist_ready():
            return [local_metric]

        gathered = [torch.zeros_like(local_metric) for _ in range(self.world_size)]
        self._all_gather_(gathered, local_metric, name="dist/all_gather(cuda_memory_metrics)")
        return gathered

    def _add_cuda_memory_state(self, state: Dict[str, Any], gathered: List[torch.Tensor]) -> None:
        if not gathered:
            return

        names = [
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            "cuda_peak_allocated_mb",
            "cuda_peak_reserved_mb",
            "cuda_free_mb",
            "cuda_total_mb",
        ]
        rows = []
        for metric in gathered:
            metric = metric.detach().to("cpu")
            rows.append([float(metric[i].item()) for i in range(len(names))])

        for idx, name in enumerate(names):
            state[name] = max(row[idx] for row in rows)

        expert_rows: Dict[int, List[List[float]]] = {}
        for rank_idx, row in enumerate(rows):
            expert_idx = self._rank_to_expert_idx(rank_idx)
            expert_rows.setdefault(expert_idx, []).append(row)
            if self.expert_data_parallel_size > 1:
                for idx, name in enumerate(names):
                    state[f"rank_{rank_idx}_{name}"] = row[idx]

        for expert_idx, grouped_rows in expert_rows.items():
            for idx, name in enumerate(names):
                state[f"expert_{expert_idx}_{name}"] = max(row[idx] for row in grouped_rows)

    def _use_cuda_stream_parallel(self):
        return (not self.distributed_expert) and self.parallel_multi and self.num_experts > 1 and self._is_cuda_device()

    def _create_expert_dp_process_group(self):
        if (
            not self._dist_ready()
            or self.expert_data_parallel_size <= 1
        ):
            return None

        local_group = None
        for expert_idx in range(self.num_experts):
            ranks = list(range(
                expert_idx * self.expert_data_parallel_size,
                (expert_idx + 1) * self.expert_data_parallel_size,
            ))
            group = dist.new_group(ranks=ranks)
            if expert_idx == self.local_expert_idx:
                local_group = group

        log.info(
            f"[MultiTrainer][rank={self.rank}] expert_data_parallel group "
            f"for expert {self.local_expert_idx}: ranks={self.expert_group_ranks}"
        )
        return local_group

    def _rank_to_expert_idx(self, rank: int) -> int:
        if not self.distributed_expert:
            return int(rank)
        return int(rank) // int(self.expert_data_parallel_size)

    @staticmethod
    def _unwrap_expert_module(expert):
        if isinstance(expert, DistributedDataParallel):
            return expert.module
        return expert

    def _expert_module(self, expert_idx: int):
        return self._unwrap_expert_module(self.model.experts[expert_idx])

    def _expert_parameters(self, expert_idx: int):
        return self._expert_module(expert_idx).parameters()

    def _expert_optimizer_parameters(self, expert_idx: int):
        return self._expert_module(expert_idx).named_parameters()

    def _maybe_wrap_local_expert_ddp(self):
        if (
            not self.distributed_expert
            or self.expert_data_parallel_size <= 1
            or self.expert_dp_backend != "ddp"
            or self.expert_dp_process_group is None
        ):
            return

        local_idx = self.local_expert_idx
        expert = self.model.experts[local_idx]
        if isinstance(expert, DistributedDataParallel):
            return

        device = self._device_obj()
        if device.type == "cuda":
            device_ids = [device.index if device.index is not None else torch.cuda.current_device()]
            output_device = device_ids[0]
        else:
            device_ids = None
            output_device = None

        self.model.experts[local_idx] = DistributedDataParallel(
            expert,
            device_ids=device_ids,
            output_device=output_device,
            process_group=self.expert_dp_process_group,
            broadcast_buffers=self.expert_dp_ddp_broadcast_buffers,
            find_unused_parameters=self.expert_dp_ddp_find_unused_parameters,
            gradient_as_bucket_view=self.expert_dp_ddp_gradient_as_bucket_view,
            static_graph=self.expert_dp_ddp_static_graph,
            bucket_cap_mb=self.expert_dp_ddp_bucket_cap_mb,
        )
        log.info(
            "[MultiTrainer][rank=%s] wrapped local expert %s with DDP backend "
            "(static_graph=%s, gradient_as_bucket_view=%s, find_unused_parameters=%s, "
            "broadcast_buffers=%s, bucket_cap_mb=%s).",
            self.rank,
            local_idx,
            self.expert_dp_ddp_static_graph,
            self.expert_dp_ddp_gradient_as_bucket_view,
            self.expert_dp_ddp_find_unused_parameters,
            self.expert_dp_ddp_broadcast_buffers,
            self.expert_dp_ddp_bucket_cap_mb,
        )

    def _sync_local_expert_parameters(self):
        if (
            not self._dist_ready()
            or self.expert_data_parallel_size <= 1
            or self.expert_dp_process_group is None
        ):
            return

        expert = self._expert_module(self.local_expert_idx)
        for tensor in list(expert.parameters()) + list(expert.buffers()):
            dist.broadcast(
                tensor.data,
                src=self.expert_group_src_rank,
                group=self.expert_dp_process_group,
            )

    def _sync_local_expert_grads(self, expert_idx: int):
        if self.expert_dp_backend == "ddp":
            return
        if (
            not self._dist_ready()
            or self.expert_data_parallel_size <= 1
            or self.expert_dp_process_group is None
        ):
            return

        params = list(self._expert_parameters(expert_idx))
        if not params:
            return

        bucket_cap_bytes = max(int(self.expert_dp_grad_bucket_mb * 1024 * 1024), 1)
        pending_reductions = []

        def enqueue_flat_bucket(bucket):
            if not bucket:
                return
            if len(bucket) == 1:
                grad = bucket[0]
                work = dist.all_reduce(
                    grad,
                    op=dist.ReduceOp.SUM,
                    group=self.expert_dp_process_group,
                    async_op=True,
                )
                pending_reductions.append((work, bucket, None))
                return

            flat = torch.cat([grad.contiguous().view(-1) for grad in bucket])
            work = dist.all_reduce(
                flat,
                op=dist.ReduceOp.SUM,
                group=self.expert_dp_process_group,
                async_op=True,
            )
            pending_reductions.append((work, bucket, flat))

        def enqueue_coalesced_bucket(bucket):
            if not bucket:
                return
            if (
                self.expert_dp_grad_sync_mode == "coalesced"
                and len(bucket) > 1
                and hasattr(dist, "all_reduce_coalesced")
            ):
                try:
                    work = dist.all_reduce_coalesced(
                        bucket,
                        op=dist.ReduceOp.SUM,
                        group=self.expert_dp_process_group,
                        async_op=True,
                    )
                    pending_reductions.append((work, bucket, None))
                    return
                except (TypeError, RuntimeError) as exc:
                    if not self._expert_dp_coalesced_warned:
                        log.warning(
                            "expert data-parallel coalesced grad sync failed once; "
                            "falling back to flat bucket all_reduce. error=%s",
                            exc,
                        )
                        self._expert_dp_coalesced_warned = True
            enqueue_flat_bucket(bucket)

        def finish_pending_reductions():
            scale = float(self.expert_data_parallel_size)
            for work, bucket, flat in pending_reductions:
                if work is not None:
                    work.wait()
                if flat is not None:
                    flat.div_(scale)
                    offset = 0
                    for grad in bucket:
                        numel = grad.numel()
                        grad.copy_(flat[offset:offset + numel].view_as(grad))
                        offset += numel
                else:
                    for grad in bucket:
                        grad.div_(scale)
            pending_reductions.clear()

        def reduce_param_grads(reduce_params):
            grad_buckets = {}
            for param in reduce_params:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "expert data-parallel grad sync does not support sparse gradients; "
                        "sparse/missing gradients can produce mismatched collectives."
                    )

                grad_bytes = grad.numel() * grad.element_size()
                if grad_bytes >= bucket_cap_bytes:
                    bucket_key = (grad.device, grad.dtype)
                    enqueue_coalesced_bucket(grad_buckets.pop(bucket_key, []))
                    enqueue_coalesced_bucket([grad])
                    continue

                bucket_key = (grad.device, grad.dtype)
                bucket, bucket_bytes = grad_buckets.get(bucket_key, ([], 0))
                if bucket and bucket_bytes + grad_bytes > bucket_cap_bytes:
                    enqueue_coalesced_bucket(bucket)
                    bucket, bucket_bytes = [], 0
                bucket.append(grad)
                grad_buckets[bucket_key] = (bucket, bucket_bytes + grad_bytes)

            for bucket, _bucket_bytes in grad_buckets.values():
                enqueue_coalesced_bucket(bucket)
            finish_pending_reductions()

        if self.expert_dp_grad_check_mode in ("assume_dense", "none", "off", "false"):
            if any(param.grad is not None and param.grad.is_sparse for param in params):
                raise RuntimeError(
                    "expert data-parallel assume_dense grad sync does not support sparse gradients."
                )
            reduce_param_grads(params)
            return

        if self.expert_dp_grad_check_mode not in ("auto", "safe"):
            raise ValueError(
                "expert_dp_grad_check_mode must be 'auto'/'safe' or 'assume_dense'/'none', "
                f"got {self.expert_dp_grad_check_mode!r}"
            )

        grad_status = torch.tensor(
            [
                sum(1 for param in params if param.grad is None),
                sum(1 for param in params if param.grad is not None and param.grad.is_sparse),
            ],
            dtype=torch.int32,
            device=params[0].device,
        )
        dist.all_reduce(
            grad_status,
            op=dist.ReduceOp.SUM,
            group=self.expert_dp_process_group,
        )
        if int(grad_status[1].item()) != 0:
            raise RuntimeError(
                "expert data-parallel grad sync does not support sparse gradients; "
                "set sparse=False or disable expert_data_parallel_size > 1 for this model."
            )
        missing_grads = grad_status[0]
        if int(missing_grads.item()) == 0:
            reduce_param_grads(params)
            return

        grad_flags = torch.tensor(
            [1 if param.grad is not None else 0 for param in params],
            dtype=torch.int32,
            device=params[0].device,
        )
        dist.all_reduce(
            grad_flags,
            op=dist.ReduceOp.SUM,
            group=self.expert_dp_process_group,
        )

        params_with_grads = []
        for param, grad_count in zip(params, grad_flags.tolist()):
            if int(grad_count) == 0:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
            params_with_grads.append(param)
        reduce_param_grads(params_with_grads)

    def _sync_local_expert_buffers(self, expert_idx: int):
        if (
            not self._dist_ready()
            or self.expert_data_parallel_size <= 1
            or self.expert_dp_process_group is None
            or not self.sync_expert_dp_buffers
        ):
            return
        if self.expert_dp_backend == "ddp" and self.expert_dp_ddp_broadcast_buffers:
            return

        bucket_cap_bytes = max(int(self.expert_dp_buffer_bucket_mb * 1024 * 1024), 1)
        pending_reductions = []

        def enqueue_float_buffer_bucket(bucket):
            if not bucket:
                return
            if (
                self.expert_dp_buffer_sync_mode == "coalesced"
                and len(bucket) > 1
                and hasattr(dist, "all_reduce_coalesced")
            ):
                try:
                    work = dist.all_reduce_coalesced(
                        bucket,
                        op=dist.ReduceOp.SUM,
                        group=self.expert_dp_process_group,
                        async_op=True,
                    )
                    pending_reductions.append((work, bucket))
                    return
                except (TypeError, RuntimeError) as exc:
                    if not self._expert_dp_coalesced_warned:
                        log.warning(
                            "expert data-parallel coalesced buffer sync failed once; "
                            "falling back to individual all_reduce. error=%s",
                            exc,
                        )
                        self._expert_dp_coalesced_warned = True
            for buf in bucket:
                work = dist.all_reduce(
                    buf,
                    op=dist.ReduceOp.SUM,
                    group=self.expert_dp_process_group,
                    async_op=True,
                )
                pending_reductions.append((work, [buf]))

        float_buckets = {}
        for buf in self._expert_module(expert_idx).buffers():
            if buf.is_floating_point():
                buf_data = buf.data
                buf_bytes = buf_data.numel() * buf_data.element_size()
                if buf_bytes >= bucket_cap_bytes:
                    bucket_key = (buf_data.device, buf_data.dtype)
                    enqueue_float_buffer_bucket(float_buckets.pop(bucket_key, []))
                    enqueue_float_buffer_bucket([buf_data])
                    continue

                bucket_key = (buf_data.device, buf_data.dtype)
                bucket, bucket_bytes = float_buckets.get(bucket_key, ([], 0))
                if bucket and bucket_bytes + buf_bytes > bucket_cap_bytes:
                    enqueue_float_buffer_bucket(bucket)
                    bucket, bucket_bytes = [], 0
                bucket.append(buf_data)
                float_buckets[bucket_key] = (bucket, bucket_bytes + buf_bytes)
            else:
                dist.broadcast(
                    buf.data,
                    src=self.expert_group_src_rank,
                    group=self.expert_dp_process_group,
                )
        for bucket, _bucket_bytes in float_buckets.values():
            enqueue_float_buffer_bucket(bucket)
        scale = float(self.expert_data_parallel_size)
        for work, bucket in pending_reductions:
            if work is not None:
                work.wait()
            for buf in bucket:
                buf.div_(scale)

    def _mean_expert_dp_scalar(self, value):
        out = self._as_scalar_tensor(value, default=0.0).clone()
        if (
            self._dist_ready()
            and self.expert_data_parallel_size > 1
            and self.expert_dp_process_group is not None
        ):
            dist.all_reduce(
                out,
                op=dist.ReduceOp.SUM,
                group=self.expert_dp_process_group,
            )
            out.div_(float(self.expert_data_parallel_size))
        return out

    def _all_reduce_(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM, name: str = "dist/all_reduce"):
        if self._dist_ready():
            with self._tagger.tag(name, it=self.iter, extra=f"numel={tensor.numel()}"):
                dist.all_reduce(tensor, op=op)
        return tensor

    def _all_gather_(self, output_list: List[torch.Tensor], tensor: torch.Tensor, name: str = "dist/all_gather"):
        if self._dist_ready():
            with self._tagger.tag(name, it=self.iter, extra=f"numel={tensor.numel()} world={self.world_size}"):
                dist.all_gather(output_list, tensor)
        else:
            output_list[0].copy_(tensor)
        return output_list

    def _recursive_set_device_attr(self, module: nn.Module, device: torch.device):
        for m in module.modules():
            if hasattr(m, "device"):
                try:
                    setattr(m, "device", device)
                except Exception:
                    pass

    def _move_aux_modules_to_device(self, device: torch.device):
        for attr in ("train_lossfunc", "validation_lossfunc", "reference_lossfunc"):
            module = getattr(self, attr, None)
            if module is None:
                continue
            if isinstance(module, nn.Module):
                module.to(device)
                self._recursive_set_device_attr(module, device)
            elif hasattr(module, "device"):
                try:
                    setattr(module, "device", device)
                except Exception:
                    pass

    def _materialize_local_expert_only(self):
        local_dev = self._device_obj()
        cpu_dev = torch.device("cpu")

        for i, expert in enumerate(self.model.experts):
            target = local_dev if i == self.local_expert_idx else cpu_dev
            expert.to(target)
            self._recursive_set_device_attr(expert, target)

        if hasattr(self.model, "device"):
            self.model.device = local_dev

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(
            f"[MultiTrainer][rank={self.rank}] local_expert_idx={self.local_expert_idx} on {local_dev}, "
            f"all other experts moved to CPU."
        )

    # ---------------------------------------------------------------------
    # profiler
    # ---------------------------------------------------------------------

    def _should_profile_iter(self, it: int) -> bool:
        return self.debug_profile and (self.debug_profile_start_iter <= it <= self.debug_profile_end_iter)

    @contextlib.contextmanager
    def _maybe_profile_iteration(self, it: int):
        if not self._should_profile_iter(it) or torch_profile is None:
            yield
            return

        acts = [ProfilerActivity.CPU]
        if self._is_cuda_device():
            acts.append(ProfilerActivity.CUDA)

        profile_dir = self.debug_profile_dir or os.path.join(os.getcwd(), "profile_traces")
        os.makedirs(profile_dir, exist_ok=True)

        with torch_profile(
            activities=acts,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            yield

        trace_path = os.path.join(profile_dir, f"rank{self.rank}_iter{it}.json")
        try:
            prof.export_chrome_trace(trace_path)
            log.info(f"[PROFILE][rank={self.rank}][it={it}] trace exported to {trace_path}")
        except Exception as e:
            log.warning(f"[PROFILE][rank={self.rank}][it={it}] export trace failed: {e}")

        try:
            sort_key = "self_cuda_time_total" if self._is_cuda_device() else "self_cpu_time_total"
            log.info("\n%s", prof.key_averages().table(sort_by=sort_key, row_limit=40))
        except Exception as e:
            log.warning(f"[PROFILE][rank={self.rank}][it={it}] print table failed: {e}")

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------

    def _warn_non_expert_trainables(self):
        expert_param_ids = {id(p) for expert in self.model.experts for p in expert.parameters()}
        outside = [
            name for name, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in expert_param_ids
        ]
        if outside:
            preview = outside[:10]
            suffix = "" if len(outside) <= 10 else f" ... (+{len(outside) - 10} more)"
            log.warning(
                "Found trainable params outside `model.experts`. "
                "Isolated optimizers will NOT update them: %s%s",
                preview, suffix
            )

    # ---------------------------------------------------------------------
    # batch prep
    # ---------------------------------------------------------------------

    def _prepare_expert_masks(self, batch_dict, range_dis, expert_idx):
        d_min, d_max = range_dis
        dist_edge = batch_dict['edge_lengths']

        if expert_idx == self.num_experts - 1:
            expert_edge_mask = (dist_edge >= d_min)
        else:
            expert_edge_mask = (dist_edge >= d_min) & (dist_edge < d_max)

        num_nodes = batch_dict["node_features"].shape[0]
        expert_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self._device_obj())
        if d_min > 0:
            expert_node_mask.fill_(False)

        return expert_edge_mask, expert_node_mask

    @staticmethod
    def _truthy_config_value(value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "none", "no", "off"}
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return bool(value)

    @classmethod
    def _contains_geometry_gradient_option(cls, value, *, path=()):
        geometry_terms = ("force", "forces", "stress", "virial")
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text == "muon_force_name_patterns":
                    continue
                child_path = path + (key_text,)
                if any(term in key_text for term in geometry_terms) and cls._truthy_config_value(child):
                    return True
                if cls._contains_geometry_gradient_option(child, path=child_path):
                    return True
            return False
        if isinstance(value, (list, tuple, set)):
            return any(cls._contains_geometry_gradient_option(item, path=path) for item in value)
        if "loss_options" in path and isinstance(value, str):
            text = value.lower()
            return any(term in text for term in geometry_terms) and cls._truthy_config_value(value)
        return False

    def _validate_lem_cutoff_precompute_options(self):
        if self._contains_geometry_gradient_option(self.train_options):
            raise ValueError(
                "precompute_lem_cutoff_coeffs=True is incompatible with force/stress/virial "
                "or other geometry-gradient losses."
            )

    def _assert_no_geometry_grad_for_cutoff_precompute(self, batch):
        if not self.precompute_lem_cutoff_coeffs:
            return
        for key in (_keys.POSITIONS_KEY, _keys.CELL_KEY):
            value = batch[key] if key in batch else None
            if torch.is_tensor(value) and value.requires_grad:
                raise RuntimeError(
                    "precompute_lem_cutoff_coeffs=True requires fixed geometry, but "
                    f"batch[{key!r}] has requires_grad=True."
                )

    def _iter_lem_cutoff_init_layers(self):
        model = getattr(self.model, "module", self.model)
        experts = getattr(model, "experts", None)
        candidates = list(experts) if experts is not None else [model]

        for candidate in candidates:
            expert = getattr(candidate, "module", candidate)
            embedding = getattr(expert, "embedding", None)
            init_layer = getattr(embedding, "init_layer", None)
            base_init = getattr(init_layer, "base_init", init_layer)
            if hasattr(base_init, "precompute_cutoff_metadata"):
                yield base_init

    def _get_lem_cutoff_init_layer(self):
        if not (self.precompute_lem_active_edges or self.precompute_lem_cutoff_coeffs):
            return None
        if self._lem_cutoff_precompute_checked:
            return self._lem_cutoff_init_layer

        layers = list(self._iter_lem_cutoff_init_layers())
        if not layers:
            if not self._lem_cutoff_precompute_warned:
                log.warning("LEM cutoff precompute requested, but no compatible InitLayer was found.")
                self._lem_cutoff_precompute_warned = True
            self._lem_cutoff_precompute_checked = True
            return None

        signatures = [
            layer.cutoff_config_signature()
            for layer in layers
            if hasattr(layer, "cutoff_config_signature")
        ]
        if signatures and any(sig != signatures[0] for sig in signatures[1:]):
            log.warning(
                "LEM cutoff precompute disabled because experts have different cutoff configurations."
            )
            self._lem_cutoff_precompute_checked = True
            return None

        self._lem_cutoff_init_layer = layers[0]
        self._lem_cutoff_precompute_checked = True
        log.info(
            "LEM cutoff metadata precompute enabled: active_edges=%s, cutoff_coeffs=%s",
            self.precompute_lem_active_edges,
            self.precompute_lem_cutoff_coeffs,
        )
        return self._lem_cutoff_init_layer

    @staticmethod
    def _lem_active_edge_split_sizes(batch, active_edges):
        batch_slices = getattr(batch, "__slices__", None)
        if batch_slices is None and isinstance(batch, dict):
            batch_slices = batch.get("__slices__", {})
        slices = None if batch_slices is None else batch_slices.get(_keys.EDGE_INDEX_KEY)
        if slices is None:
            return None
        if torch.is_tensor(active_edges):
            active_edges = active_edges.detach().reshape(-1)
            if active_edges.device.type != "cpu":
                return None
            active_edges = active_edges.to(dtype=torch.long)
            active_edge_ids = [int(v) for v in active_edges.tolist()]
        else:
            active_edge_ids = [int(v) for v in active_edges]

        active_edge_ids.sort()
        split_sizes = []
        cursor = 0
        n_active = len(active_edge_ids)
        for start, end in zip(slices[:-1], slices[1:]):
            start = int(start)
            end = int(end)
            while cursor < n_active and active_edge_ids[cursor] < start:
                cursor += 1
            graph_start = cursor
            while cursor < n_active and active_edge_ids[cursor] < end:
                cursor += 1
            split_sizes.append(cursor - graph_start)
        return tuple(split_sizes)

    @staticmethod
    def _clear_lem_precompute_metadata(batch):
        for key in (
            _keys.LEM_ACTIVE_EDGES_KEY,
            _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY,
            _keys.LEM_CUTOFF_COEFFS_KEY,
        ):
            try:
                if key in batch:
                    if hasattr(batch, "pop"):
                        batch.pop(key, None)
                    else:
                        del batch[key]
            except Exception:
                pass
        return batch

    @staticmethod
    def _attach_lem_cpu_split_sizes(batch_dict, cpu_batch):
        if isinstance(cpu_batch, dict):
            split_sizes = cpu_batch.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        else:
            try:
                split_sizes = cpu_batch[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY]
            except Exception:
                split_sizes = getattr(cpu_batch, _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if split_sizes is None:
            return batch_dict
        if torch.is_tensor(split_sizes):
            if split_sizes.device.type != "cpu":
                return batch_dict
            split_sizes = split_sizes.detach().reshape(-1).to(dtype=torch.long)
        else:
            split_sizes = torch.tensor(
                [int(v) for v in split_sizes],
                dtype=torch.long,
                device="cpu",
            )
        batch_dict[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = split_sizes
        return batch_dict

    def _precompute_lem_cutoff_metadata(self, batch):
        batch = self._clear_lem_precompute_metadata(batch)
        init_layer = self._get_lem_cutoff_init_layer()
        if init_layer is None:
            return batch

        with self._tagger.tag("prepare_batch/precompute_lem_cutoff", it=self.iter):
            if _keys.EDGE_TYPE_KEY not in batch:
                if not self._lem_cutoff_precompute_warned:
                    log.warning("LEM cutoff precompute skipped because the CPU batch has no edge_type key.")
                    self._lem_cutoff_precompute_warned = True
                return batch
            self._assert_no_geometry_grad_for_cutoff_precompute(batch)
            if (
                _keys.EDGE_VECTORS_KEY not in batch
                and _keys.CELL_KEY in batch
                and _keys.EDGE_CELL_SHIFT_KEY not in batch
            ):
                if not self._lem_cutoff_precompute_warned:
                    log.warning(
                        "LEM cutoff precompute skipped because batch has cell but no edge_cell_shift; "
                        "falling back to model forward cutoff computation."
                    )
                    self._lem_cutoff_precompute_warned = True
                return batch
            cutoff_data = {
                key: batch[key]
                for key in (
                    _keys.EDGE_VECTORS_KEY,
                    _keys.EDGE_LENGTH_KEY,
                    _keys.POSITIONS_KEY,
                    _keys.EDGE_INDEX_KEY,
                    _keys.CELL_KEY,
                    _keys.EDGE_CELL_SHIFT_KEY,
                    _keys.BATCH_KEY,
                )
                if key in batch
            }
            cutoff_data = with_edge_vectors(cutoff_data, with_lengths=True)
            if _keys.EDGE_VECTORS_KEY not in batch:
                batch[_keys.EDGE_VECTORS_KEY] = cutoff_data[_keys.EDGE_VECTORS_KEY]
            if _keys.EDGE_LENGTH_KEY not in batch:
                batch[_keys.EDGE_LENGTH_KEY] = cutoff_data[_keys.EDGE_LENGTH_KEY]
            active_edges, cutoff_coeffs = init_layer.precompute_cutoff_metadata(
                cutoff_data[_keys.EDGE_LENGTH_KEY],
                batch[_keys.EDGE_TYPE_KEY],
                compute_cutoff=self.precompute_lem_cutoff_coeffs,
            )
            if self.precompute_lem_active_edges:
                batch[_keys.LEM_ACTIVE_EDGES_KEY] = active_edges
            split_sizes = self._lem_active_edge_split_sizes(batch, active_edges)
            if split_sizes is not None:
                batch[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = split_sizes
            if cutoff_coeffs is not None:
                batch[_keys.LEM_CUTOFF_COEFFS_KEY] = cutoff_coeffs
        return batch

    def _prepare_batch_bundle(self, batch, with_lengths=True):
        batch = self._precompute_lem_cutoff_metadata(batch)
        with self._tagger.tag("prepare_batch/to_device", it=self.iter):
            batch_dev = batch.to(
                self.device,
                non_blocking=bool(self.data_pin_memory and self._is_cuda_device()),
            )

        batch_info = {
            "__slices__": batch_dev.__slices__,
            "__cumsum__": batch_dev.__cumsum__,
            "__cat_dims__": batch_dev.__cat_dims__,
            "__num_nodes_list__": batch_dev.__num_nodes_list__,
            "__data_class__": batch_dev.__data_class__,
        }

        with self._tagger.tag("prepare_batch/to_dict", it=self.iter):
            batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

        if with_lengths:
            with self._tagger.tag("prepare_batch/with_edge_vectors", it=self.iter):
                batch_dict = with_edge_vectors(batch_dict, with_lengths=True)

        batch_dict = self._attach_lem_cpu_split_sizes(batch_dict, batch)
        return batch_dict, batch_info

    # -------------------- packed GPU tensor broadcast --------------------

    @staticmethod
    def _dtype_to_code(dtype: torch.dtype) -> str:
        mp = {
            torch.float32: "float32",
            torch.float64: "float64",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.int64: "int64",
            torch.int32: "int32",
            torch.int16: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        if dtype not in mp:
            raise TypeError(f"Unsupported dtype for packed broadcast: {dtype}")
        return mp[dtype]

    @staticmethod
    def _code_to_dtype(code: str) -> torch.dtype:
        mp = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        return mp[code]

    def _extract_batch_info_from_cpu_batch(self, batch):
        return {
            "__slices__": batch.__slices__,
            "__cumsum__": batch.__cumsum__,
            "__cat_dims__": batch.__cat_dims__,
            "__num_nodes_list__": batch.__num_nodes_list__,
            "__data_class__": batch.__data_class__,
        }

    def _split_tensor_and_object_items(self, d: Dict[str, Any]):
        tensor_items = []
        object_items = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                tensor_items.append((k, v))
            else:
                object_items[k] = v
        return tensor_items, object_items

    def _pack_tensor_groups(self, tensor_items: List[Tuple[str, torch.Tensor]]):
        groups = {}
        meta = []
        for k, t in tensor_items:
            t = t.contiguous()
            code = self._dtype_to_code(t.dtype)
            if code not in groups:
                groups[code] = []
            start = sum(x.numel() for x in groups[code])
            groups[code].append(t.reshape(-1))
            meta.append((k, code, tuple(t.shape), start, t.numel()))
        flat_groups = {}
        group_numel = {}
        for code, ts in groups.items():
            total = sum(x.numel() for x in ts)
            group_numel[code] = total
            if total == 0:
                flat_groups[code] = torch.empty((0,), dtype=self._code_to_dtype(code), device=self.device)
            else:
                flat_groups[code] = torch.cat(ts, dim=0)
        return meta, group_numel, flat_groups

    def _rebuild_tensor_groups_from_broadcast(self, schema, flat_groups):
        out = {}
        for k, code, shape, start, numel in schema["tensor_meta"]:
            dtype = self._code_to_dtype(code)
            if numel == 0:
                out[k] = torch.empty(shape, dtype=dtype, device=self.device)
            else:
                flat = flat_groups[code].narrow(0, int(start), int(numel))
                out[k] = flat.view(shape)
        out.update(schema["object_items"])
        return out

    def _broadcast_prepared_gpu_dict_packed(self, rank0_dict: Optional[Dict[str, Any]], tag_name: str):
        if not self._dist_ready():
            return rank0_dict

        schema_holder = [None]
        rank0_flat_groups = None

        if self.rank == 0:
            tensor_items, object_items = self._split_tensor_and_object_items(rank0_dict)
            tensor_meta, group_numel, rank0_flat_groups = self._pack_tensor_groups(tensor_items)
            schema_holder[0] = {
                "tensor_meta": tensor_meta,
                "group_numel": group_numel,
                "object_items": object_items,
            }

        with self._tagger.tag(f"{tag_name}/broadcast_schema", it=self.iter):
            dist.broadcast_object_list(schema_holder, src=0)

        schema = schema_holder[0]
        recv_flat_groups = {}
        for code, total_numel in schema["group_numel"].items():
            dtype = self._code_to_dtype(code)
            if self.rank == 0:
                flat = rank0_flat_groups[code]
            else:
                flat = torch.empty((int(total_numel),), dtype=dtype, device=self.device)

            if int(total_numel) > 0:
                with self._tagger.tag(
                    f"{tag_name}/broadcast_group",
                    it=self.iter,
                    extra=f"dtype={code} numel={int(total_numel)}"
                ):
                    dist.broadcast(flat, src=0)
            recv_flat_groups[code] = flat

        return self._rebuild_tensor_groups_from_broadcast(schema, recv_flat_groups)

    def _broadcast_prepared_bundle_rank0(
        self,
        batch,
        ref_batch=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self._dist_ready() or not self.distributed_rank0_prepare_batch:
            batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)
            ref_batch_dict = None
            ref_batch_info = None
            if ref_batch is not None:
                ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)
            return batch_dict, batch_info, ref_batch_dict, ref_batch_info

        batch_info_holder = [None]
        ref_batch_info_holder = [None]

        rank0_batch_dict = None
        rank0_ref_batch_dict = None

        if self.rank == 0:
            batch = self._precompute_lem_cutoff_metadata(batch)
            with self._tagger.tag("shared_batch/rank0_extract_batch_info", it=self.iter):
                batch_info_holder[0] = self._extract_batch_info_from_cpu_batch(batch)

            with self._tagger.tag("shared_batch/rank0_to_device", it=self.iter):
                batch_dev = batch.to(
                    self.device,
                    non_blocking=bool(self.data_pin_memory and self._is_cuda_device()),
                )

            with self._tagger.tag("shared_batch/rank0_to_dict", it=self.iter):
                rank0_batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

            with self._tagger.tag("shared_batch/rank0_with_edge_vectors", it=self.iter):
                rank0_batch_dict = with_edge_vectors(rank0_batch_dict, with_lengths=True)
                rank0_batch_dict = self._attach_lem_cpu_split_sizes(rank0_batch_dict, batch)

            if ref_batch is not None:
                ref_batch = self._precompute_lem_cutoff_metadata(ref_batch)
                with self._tagger.tag("shared_batch/rank0_ref_extract_batch_info", it=self.iter):
                    ref_batch_info_holder[0] = self._extract_batch_info_from_cpu_batch(ref_batch)

                with self._tagger.tag("shared_batch/rank0_ref_to_device", it=self.iter):
                    ref_batch_dev = ref_batch.to(
                        self.device,
                        non_blocking=bool(self.data_pin_memory and self._is_cuda_device()),
                    )

                with self._tagger.tag("shared_batch/rank0_ref_to_dict", it=self.iter):
                    rank0_ref_batch_dict = AtomicData.to_AtomicDataDict(ref_batch_dev)

                with self._tagger.tag("shared_batch/rank0_ref_with_edge_vectors", it=self.iter):
                    rank0_ref_batch_dict = with_edge_vectors(rank0_ref_batch_dict, with_lengths=True)
                    rank0_ref_batch_dict = self._attach_lem_cpu_split_sizes(rank0_ref_batch_dict, ref_batch)

        with self._tagger.tag("shared_batch/broadcast_batch_info", it=self.iter):
            dist.broadcast_object_list(batch_info_holder, src=0)
            dist.broadcast_object_list(ref_batch_info_holder, src=0)

        batch_dict = self._broadcast_prepared_gpu_dict_packed(rank0_batch_dict, "shared_batch/main")
        batch_info = batch_info_holder[0]

        if ref_batch_info_holder[0] is not None:
            ref_batch_dict = self._broadcast_prepared_gpu_dict_packed(rank0_ref_batch_dict, "shared_batch/ref")
            ref_batch_info = ref_batch_info_holder[0]
        else:
            ref_batch_dict = None
            ref_batch_info = None

        return batch_dict, batch_info, ref_batch_dict, ref_batch_info

    # ---------------------------------------------------------------------
    # loss metric helpers
    # ---------------------------------------------------------------------

    def _resolve_loss_module(self, loss_obj):
        curr = loss_obj
        visited = set()
        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            found_inner = None
            for attr in ("lossfunc", "loss_fn", "criterion", "method", "loss"):
                inner = getattr(curr, attr, None)
                if inner is None:
                    continue
                if isinstance(inner, nn.Module):
                    found_inner = inner
                    break
            if found_inner is None:
                break
            curr = found_inner
        return curr

    def _as_scalar_tensor(self, value, default=0.0, allow_none=False):
        if value is None:
            if allow_none:
                return None
            return torch.zeros((), dtype=self.dtype, device=self.device) + float(default)

        if torch.is_tensor(value):
            out = value.detach()
            if out.ndim != 0:
                out = out.mean()
            if out.device != self._device_obj():
                out = out.to(self.device)
            return out.to(dtype=self.dtype)

        return torch.tensor(float(value), dtype=self.dtype, device=self.device)

    def _to_float_scalar(self, value, default=0.0):
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            v = value.detach()
            if v.ndim != 0:
                v = v.mean()
            return float(v.item())
        return float(value)

    def _to_int_scalar(self, value, default=0):
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            v = value.detach()
            if v.ndim != 0:
                v = v.mean()
            return int(v.item())
        return int(value)

    def _snapshot_loss_metrics(self, loss_obj) -> Dict[str, Any]:
        loss_module = self._resolve_loss_module(loss_obj)

        out = {
            "onsite": self._as_scalar_tensor(getattr(loss_module, "last_onsite_loss", 0.0), default=0.0),
            "hopping": self._as_scalar_tensor(getattr(loss_module, "last_hopping_loss", 0.0), default=0.0),
            "z_loss": self._as_scalar_tensor(getattr(loss_module, "last_z_loss", None), allow_none=True),
            "expert_load_cv": self._as_scalar_tensor(getattr(loss_module, "expert_load_cv", None), allow_none=True),
        }

        for k in (
            "last_onsite_l1_sum", "last_onsite_mse_sum", "last_onsite_count",
            "last_hopping_l1_sum", "last_hopping_mse_sum", "last_hopping_count",
        ):
            v = getattr(loss_module, k, None)
            out[k] = self._as_scalar_tensor(v, default=0.0) if v is not None else None

        return out

    def _flow_state_with_prefix(self, state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        if prefix == "train":
            return dict(state)
        out = {}
        for key, value in state.items():
            if key.startswith("train_"):
                out[f"{prefix}_{key[len('train_'):]}"] = value
            else:
                out[key] = value
        return out

    def _flow_state_scalar(self, state: Dict[str, Any], *names, default=0.0, allow_none=False):
        for name in names:
            if name in state and state[name] is not None:
                return self._as_scalar_tensor(state[name], default=default, allow_none=allow_none)
        return self._as_scalar_tensor(None, default=default, allow_none=allow_none)

    def _payload_metrics_from_flow_state(
        self,
        state: Dict[str, Any],
        *,
        prefix: str,
    ) -> Dict[str, Any]:
        compatible_stats = state.get("_compatible_clean_stats", {})
        if not isinstance(compatible_stats, dict):
            compatible_stats = {}

        return {
            "onsite": self._flow_state_scalar(
                state,
                f"{prefix}_compatible_onsite_loss",
                f"{prefix}_onsite_loss",
                default=0.0,
            ),
            "hopping": self._flow_state_scalar(
                state,
                f"{prefix}_compatible_hopping_loss",
                f"{prefix}_hopping_loss",
                default=0.0,
            ),
            "z_loss": self._flow_state_scalar(
                state,
                "mean_max_prob",
                f"{prefix}_mean_max_prob",
                default=0.0,
                allow_none=True,
            ),
            "expert_load_cv": self._flow_state_scalar(
                state,
                "expert_load_cv",
                f"{prefix}_expert_load_cv",
                default=0.0,
                allow_none=True,
            ),
            "last_onsite_l1_sum": self._as_scalar_tensor(
                compatible_stats.get("onsite_l1_sum", None),
                allow_none=True,
            ),
            "last_onsite_mse_sum": self._as_scalar_tensor(
                compatible_stats.get("onsite_mse_sum", None),
                allow_none=True,
            ),
            "last_onsite_count": self._as_scalar_tensor(
                compatible_stats.get("onsite_count", None),
                allow_none=True,
            ),
            "last_hopping_l1_sum": self._as_scalar_tensor(
                compatible_stats.get("hopping_l1_sum", None),
                allow_none=True,
            ),
            "last_hopping_mse_sum": self._as_scalar_tensor(
                compatible_stats.get("hopping_mse_sum", None),
                allow_none=True,
            ),
            "last_hopping_count": self._as_scalar_tensor(
                compatible_stats.get("hopping_count", None),
                allow_none=True,
            ),
        }

    def _pack_component_state(
        self,
        pack: torch.Tensor,
        *,
        prefix: str,
        criterion=None,
    ) -> Dict[str, torch.Tensor]:
        if criterion is None:
            criterion = self.train_lossfunc
        return MetricReducer.component_state_from_pack(
            pack,
            loss_module=self._resolve_loss_module(criterion),
            supports_triplet=Trainer._supports_endpoint_triplet(criterion),
            dtype=self.dtype,
            device=self.device,
            prefix=prefix,
            global_step=getattr(self, "iter", None),
        )

    # ---------------------------------------------------------------------
    # core expert fwd/loss
    # ---------------------------------------------------------------------

    @property
    def _objective(self) -> Objective:
        obj = self.__dict__.get("_objective_obj")
        if obj is None:
            obj = Objective(self)
            self.__dict__["_objective_obj"] = obj
        return obj

    @property
    def _flow_objective(self) -> FlowObjective:
        obj = self.__dict__.get("_flow_objective_obj")
        if obj is None:
            obj = FlowObjective(self)
            self.__dict__["_flow_objective_obj"] = obj
        return obj

    def _run_one_expert_loss(
        self,
        batch_dict,
        batch_info,
        criterion,
        expert_idx,
        range_dis,
        capture_metrics=False,
        flow_prefix="train",
        use_flow=None,
    ):
        with self._tagger.tag("expert/prepare_masks", it=self.iter, expert=expert_idx):
            expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)

        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask
        batch_copy["expert_idx"] = int(expert_idx)

        active_nodes = expert_node_mask.sum().detach()
        active_edges = expert_edge_mask.sum().detach()

        configured_flow = bool(
            getattr(getattr(self, "flow_cfm", None), "enabled", False)
        )
        flow_enabled = configured_flow if use_flow is None else (
            configured_flow and bool(use_flow)
        )
        if flow_enabled:
            return self._flow_objective.run(
                batch_copy=batch_copy,
                batch_info=batch_info,
                criterion=criterion,
                expert_idx=expert_idx,
                expert_edge_mask=expert_edge_mask,
                expert_node_mask=expert_node_mask,
                active_nodes=active_nodes,
                active_edges=active_edges,
                flow_prefix=flow_prefix,
            )

        return self._objective.run(
            batch_copy=batch_copy,
            batch_info=batch_info,
            criterion=criterion,
            expert_idx=expert_idx,
            active_nodes=active_nodes,
            active_edges=active_edges,
            capture_metrics=capture_metrics,
        )

    def _build_train_payload(
        self, batch_dict, batch_info, expert_idx, range_dis,
        ref_batch_dict=None, ref_batch_info=None, criterion=None, flow_prefix="train"
    ):
        return self._objective.build_train_payload(
            batch_dict, batch_info, expert_idx, range_dis,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info,
            criterion=criterion,
            flow_prefix=flow_prefix,
        )

    # ---------------------------------------------------------------------
    # stitched loss helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _maybe_call_or_value(x, default: float = 1.0) -> float:
        return MetricReducer.maybe_call_or_value(x, default)

    def _compute_stitched_loss_by_reduce(self, payloads: List[Dict[str, Any]], criterion=None) -> Optional[torch.Tensor]:
        if criterion is None:
            criterion = self.train_lossfunc

        if self.endpoint_loss_mode != "reduce":
            return None

        return MetricReducer.stitched_loss_reduce(
            payloads,
            loss_module=self._resolve_loss_module(criterion),
            dtype=self.dtype,
            device=self.device,
            global_step=self.iter,
        )

    def _compute_compatible_state_from_pack(
        self,
        pack: torch.Tensor,
        criterion=None,
        *,
        prefix: str = "train",
        global_step=None,
    ):
        if criterion is None:
            criterion = self.train_lossfunc
        if global_step is None:
            global_step = getattr(self, "iter", None)
        return MetricReducer.compatible_state_from_pack(
            pack,
            loss_module=self._resolve_loss_module(criterion),
            supports_triplet=Trainer._supports_endpoint_triplet(criterion),
            dtype=self.dtype,
            device=self.device,
            prefix=prefix,
            global_step=global_step,
        )

    def _compute_compatible_loss_from_pack(self, pack: torch.Tensor, criterion=None, *, global_step=None):
        state = self._compute_compatible_state_from_pack(
            pack,
            criterion=criterion,
            prefix="train",
            global_step=global_step,
        )
        if state is None:
            return None
        return state["train_loss"]

    def _compute_local_compatible_loss_from_payload(self, payload: Dict[str, Any], criterion=None) -> torch.Tensor:
        out = self._compute_stitched_loss_by_reduce([payload], criterion=criterion)
        if out is None:
            out = payload["loss_detached"].detach()
        return out.detach()

    # ---------------------------------------------------------------------
    # display window buffers
    # ---------------------------------------------------------------------

    def _reset_display_window_buffers(self):
        dev = self._device_obj()
        self._display_window_pack_local = torch.zeros((MetricPack.LENGTH,), dtype=self.dtype, device=dev)
        self._display_window_dynamic_batch_pack_local = torch.zeros((DynamicBatchStat.LENGTH,), dtype=self.dtype, device=dev)
        self._display_window_expert_onsite_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_hopping_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_active_nodes_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_active_edges_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_last_lr_local = 0.0
        self.dynamic_batch_oom_skipped_since_display = 0
        self._reset_cuda_memory_peak()

    def _has_pending_display_window(self) -> bool:
        local_pack = MetricPack.from_tensor(self._display_window_pack_local)
        if float(local_pack.step_count.item()) > 0.0:
            return True
        local_dynamic = DynamicBatchStat.from_tensor(self._display_window_dynamic_batch_pack_local)
        return (
            self.distributed_expert
            and float(local_dynamic.oom_skipped_count.item()) > 0.0
        )

    def _make_step_pack(self, payload: Dict[str, Any]) -> torch.Tensor:
        # Build the typed pack with named fields; ``to_tensor`` reproduces the
        # exact legacy ``_P_*`` slot layout on the wire (byte-identical).
        pack = MetricPack(
            loss_opt_sum=self._as_scalar_tensor(payload.get("loss_detached", 0.0)),
            onsite_weighted_sum=self._as_scalar_tensor(payload.get("onsite_weighted_sum", 0.0)),
            hopping_weighted_sum=self._as_scalar_tensor(payload.get("hopping_weighted_sum", 0.0)),
            active_nodes_sum=self._as_scalar_tensor(payload.get("active_nodes", 0.0)),
            active_edges_sum=self._as_scalar_tensor(payload.get("active_edges", 0.0)),
            onsite_l1_sum=self._as_scalar_tensor(payload.get("onsite_l1_sum", 0.0)),
            onsite_mse_sum=self._as_scalar_tensor(payload.get("onsite_mse_sum", 0.0)),
            onsite_cnt_sum=self._as_scalar_tensor(payload.get("onsite_cnt", 0.0)),
            hopping_l1_sum=self._as_scalar_tensor(payload.get("hopping_l1_sum", 0.0)),
            hopping_mse_sum=self._as_scalar_tensor(payload.get("hopping_mse_sum", 0.0)),
            hopping_cnt_sum=self._as_scalar_tensor(payload.get("hopping_cnt", 0.0)),
            grad_norm_sum=self._as_scalar_tensor(payload.get("grad_norm", 0.0)),
            step_count=1.0,
        )

        z_vals = [self._as_scalar_tensor(z, default=0.0) for z in payload.get("z_values", []) if z is not None]
        if z_vals:
            pack.z_sum = torch.stack(z_vals).sum()
            pack.z_cnt = float(len(z_vals))

        cv_vals = [self._as_scalar_tensor(v, default=0.0) for v in payload.get("load_cv_values", []) if v is not None]
        if cv_vals:
            pack.cv_sum = torch.stack(cv_vals).sum()
            pack.cv_cnt = float(len(cv_vals))

        return pack.to_tensor(dtype=self.dtype, device=self.device)

    def _make_dynamic_batch_pack(self, dynamic_batch_state: Optional[Dict[str, Any]]) -> torch.Tensor:
        if not dynamic_batch_state:
            return DynamicBatchStat().to_tensor(dtype=self.dtype, device=self.device)
        return DynamicBatchStat(
            num_graphs_sum=self._as_scalar_tensor(dynamic_batch_state.get("batch_num_graphs", 0.0)),
            cost_sum=self._as_scalar_tensor(dynamic_batch_state.get("batch_cost", 0.0)),
            num_nodes_sum=self._as_scalar_tensor(dynamic_batch_state.get("batch_num_nodes", 0.0)),
            num_edges_sum=self._as_scalar_tensor(dynamic_batch_state.get("batch_num_edges", 0.0)),
            max_item_cost_sum=self._as_scalar_tensor(dynamic_batch_state.get("batch_max_item_cost", 0.0)),
            step_count=1.0,
        ).to_tensor(dtype=self.dtype, device=self.device)

    def _update_display_window_local(
        self,
        payload: Dict[str, Any],
        current_local_lr: float,
        dynamic_batch_state: Optional[Dict[str, Any]] = None,
    ):
        self._display_window_pack_local += self._make_step_pack(payload)
        self._display_window_dynamic_batch_pack_local += self._make_dynamic_batch_pack(dynamic_batch_state)
        self._display_window_expert_onsite_sum_local += self._as_scalar_tensor(payload["expert_onsite"], default=0.0)
        self._display_window_expert_hopping_sum_local += self._as_scalar_tensor(payload["expert_hopping"], default=0.0)
        self._display_window_expert_active_nodes_sum_local += self._as_scalar_tensor(payload["active_nodes"], default=0.0)
        self._display_window_expert_active_edges_sum_local += self._as_scalar_tensor(payload["active_edges"], default=0.0)
        self._display_window_last_lr_local = float(current_local_lr)

    def _should_flush_display_window_now(self, it: int) -> bool:
        return (it == 1) or (it % self.display_sync_freq == 0)

    def _cheap_iteration_state(
        self,
        loss_detached=None,
        current_local_lr: Optional[float] = None,
        dynamic_batch_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Per-step plugin state for non-flush ticks (P0-5). No collectives.

        Carries only fields that are already available on this rank for the
        step just committed, so the per-step plugin tick never adds a
        collective. Display metrics that require gathered/reduced values
        (``train_loss``, ``total_grad_norm``, ``expert_*`` aggregates) are
        deliberately absent: ``Monitor._get_value`` returns ``None`` for a
        missing key and the monitor skips the update, so gathered stats only
        refresh on the display-flush cadence. ``field='iteration'`` is
        included because ``Validationer`` keys its iteration-cadence
        validation off it.
        """
        state: Dict[str, Any] = {"field": "iteration", "window_steps": 0}
        if current_local_lr is not None:
            state["lr"] = float(current_local_lr)
        if loss_detached is not None:
            state["loss_detached"] = self._to_float_scalar(loss_detached)
        if dynamic_batch_state:
            state.update(dynamic_batch_state)
        return state

    def _gather_display_window_expert_metrics(self) -> List[torch.Tensor]:
        local_pack = MetricPack.from_tensor(self._display_window_pack_local)
        local_steps = max(float(local_pack.step_count.item()), 1.0)

        # ExpertDisplayMetric.to_tensor reproduces the bare 6-slot gather layout
        # exactly (slot 2 == grad_norm is written but never read downstream).
        local_metric = ExpertDisplayMetric(
            expert_onsite=self._display_window_expert_onsite_sum_local / local_steps,
            expert_hopping=self._display_window_expert_hopping_sum_local / local_steps,
            grad_norm=local_pack.grad_norm_sum / local_steps,
            lr=torch.tensor(float(self._display_window_last_lr_local), dtype=self.dtype, device=self.device),
            active_nodes=self._display_window_expert_active_nodes_sum_local / local_steps,
            active_edges=self._display_window_expert_active_edges_sum_local / local_steps,
        ).to_tensor(dtype=self.dtype, device=self.device)

        if not self._dist_ready():
            return [local_metric]

        gathered = [torch.zeros_like(local_metric) for _ in range(self.world_size)]
        self._all_gather_(gathered, local_metric, name="dist/all_gather(display_window_expert_metrics)")
        return gathered

    def _flush_display_window(self, time_idx: int) -> Optional[Dict[str, Any]]:
        if not self._has_pending_display_window():
            return None

        with self._tagger.tag("display/window_reduce", it=time_idx, extra=f"freq={self.display_sync_freq}"):
            cuda_memory_metrics = self._gather_cuda_memory_metrics()
            reduced_pack = self._display_window_pack_local.clone()
            reduced_dynamic_pack = self._display_window_dynamic_batch_pack_local.clone()
            self._all_reduce_(reduced_pack, name="dist/all_reduce(display_window_metrics_packed)")
            self._all_reduce_(reduced_dynamic_pack, name="dist/all_reduce(display_window_dynamic_batch_packed)")
            gathered = self._gather_display_window_expert_metrics()

        reduced_mp = MetricPack.from_tensor(reduced_pack)
        reduced_db = DynamicBatchStat.from_tensor(reduced_dynamic_pack)

        full_loss_steps = self.num_experts if self._dist_ready() else 1
        raw_total_steps = float(reduced_mp.step_count.item()) / full_loss_steps
        dynamic_batch_oom_skips = int(round(float(reduced_db.oom_skipped_count.item())))
        if raw_total_steps <= 0.0:
            if dynamic_batch_oom_skips > 0:
                log.warning(
                    "dynamic_batch OOM fallback flushed display boundary with skipped_iters=%s and no successful steps",
                    dynamic_batch_oom_skips,
                )
            self._reset_display_window_buffers()
            return None
        total_steps = max(raw_total_steps, 1.0)

        state = MetricReducer.display_state_from_packs(
            reduced_pack,
            reduced_dynamic_pack,
            gathered,
            total_steps=total_steps,
            num_experts=self.num_experts,
            rank_to_expert_idx=self._rank_to_expert_idx,
            train_loss_module=self._resolve_loss_module(self.train_lossfunc),
            supports_triplet=Trainer._supports_endpoint_triplet(self.train_lossfunc),
            dtype=self.dtype,
            device=self.device,
            time_idx=time_idx,
        )

        self._add_optimizer_diagnostics_to_state(state)
        self._add_cuda_memory_state(state, cuda_memory_metrics)

        self._reset_display_window_buffers()
        return state

    # ---------------------------------------------------------------------
    # scheduler
    # ---------------------------------------------------------------------

    def _add_optimizer_diagnostics_to_state(self, state):
        total_clip_events = 0.0
        total_muon_blocks = 0.0
        any_diagnostics = False
        for expert_idx, optimizer in enumerate(getattr(self, "optimizers", [])):
            if optimizer is None or not hasattr(optimizer, "get_diagnostics"):
                continue
            try:
                diagnostics = optimizer.get_diagnostics()
            except Exception as exc:
                log.debug("optimizer diagnostics collection failed for expert %s: %s", expert_idx, exc)
                continue
            any_diagnostics = True
            state.update({f"expert_{expert_idx}_{key}": value for key, value in diagnostics.items()})
            total_clip_events += float(diagnostics.get("muon_clip_events", 0.0))
            total_muon_blocks += float(diagnostics.get("muon_blocks", 0.0))
        if any_diagnostics:
            state["muon_clip_events"] = total_clip_events
            if total_muon_blocks:
                state["muon_clip_event_ratio"] = total_clip_events / total_muon_blocks

    def _get_epoch_scheduler_metric(self):
        validation_stat = self.stats.get("validation_loss", {})
        if isinstance(validation_stat, dict) and ("epoch_mean" in validation_stat):
            metric = validation_stat["epoch_mean"]
        else:
            train_stat = self.stats.get("train_loss", {})
            metric = train_stat.get("epoch_mean", None) if isinstance(train_stat, dict) else None

        if torch.is_tensor(metric):
            metric = metric.detach()
            if metric.ndim != 0:
                metric = metric.mean()
            return float(metric.item())
        return metric

    def _local_scheduler_requires_metric(self) -> bool:
        if not self.update_lr_per_iter:
            return False
        if self.distributed_expert:
            sch = self.lr_schedulers[self.local_expert_idx]
            return sch is not None and lr_scheduler_requires_metric(sch)
        return any(
            lr_scheduler_requires_metric(sch)
            for sch in self.lr_schedulers
            if sch is not None
        )

    def _local_scheduler_step(self, metric_tensor: Optional[torch.Tensor] = None):
        if not self.update_lr_per_iter:
            return
        needs_metric = self._local_scheduler_requires_metric()
        if needs_metric and metric_tensor is None:
            raise RuntimeError("iteration LR scheduler requires a metric but none was provided")

        if self.distributed_expert and needs_metric:
            metric_tensor = self._mean_expert_dp_scalar(metric_tensor)

        def _metric_float():
            if torch.is_tensor(metric_tensor):
                m = metric_tensor.detach()
                if m.ndim != 0:
                    m = m.mean()
                return float(m.item())
            return float(metric_tensor)

        if self.distributed_expert:
            sch = self.lr_schedulers[self.local_expert_idx]
            if sch is None:
                return

            if needs_metric:
                metric_float = _metric_float()
                with self._tagger.tag("scheduler/local_step", it=self.iter, expert=self.local_expert_idx, extra=f"metric={metric_float:.6g}"):
                    if self.iter > 1:
                        sch.step(metric_float)
                    elif lr_scheduler_can_step_without_metric(sch):
                        sch.step()
            else:
                with self._tagger.tag("scheduler/local_step", it=self.iter, expert=self.local_expert_idx, extra="metric=not_required"):
                    sch.step()
        else:
            metric_float = _metric_float() if needs_metric else None
            extra = f"metric={metric_float:.6g}" if metric_float is not None else "metric=not_required"
            with self._tagger.tag("scheduler/local_step(all)", it=self.iter, extra=extra):
                for sch in self.lr_schedulers:
                    if sch is None:
                        continue
                    if lr_scheduler_requires_metric(sch):
                        if self.iter > 1:
                            sch.step(metric_float)
                        elif lr_scheduler_can_step_without_metric(sch):
                            sch.step()
                    else:
                        sch.step()

    def _step_epoch_schedulers(self):
        metric = self._get_epoch_scheduler_metric()
        if self.distributed_expert and metric is not None:
            metric = self._mean_expert_dp_scalar(metric)
        metric_float = None if metric is None else self._to_float_scalar(metric)

        def _step_one_scheduler(sch, expert_idx=None):
            if sch is None:
                return

            extra = f"metric={metric_float:.6g}" if metric_float is not None else "metric=None"
            if lr_scheduler_requires_metric(sch):
                if metric_float is None:
                    log.warning("Skip epoch LR scheduler step: no epoch metric is available.")
                    return
                with self._tagger.tag("scheduler/epoch_step", it=self.iter, expert=expert_idx, extra=extra):
                    sch.step(metric_float)
            else:
                with self._tagger.tag("scheduler/epoch_step", it=self.iter, expert=expert_idx, extra=extra):
                    sch.step()

        if self.distributed_expert:
            _step_one_scheduler(self.lr_schedulers[self.local_expert_idx], expert_idx=self.local_expert_idx)
        else:
            for expert_idx, sch in enumerate(self.lr_schedulers):
                _step_one_scheduler(sch, expert_idx=expert_idx)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for i in range(self.ep, epochs + 1):
            self.epoch()
            self.call_plugins(queue_name='epoch', time=i)

            if not self.update_lr_per_iter:
                self._step_epoch_schedulers()

            self.update()
            self.ep += 1

    # ---------------------------------------------------------------------
    # distributed expert iteration
    # ---------------------------------------------------------------------

    def _iteration_distributed_expert_prepared(
        self,
        batch_dict,
        batch_info,
        ref_batch_dict=None,
        ref_batch_info=None,
        original_batch=None,
    ):
        dynamic_batch_state = self._dynamic_batch_state_from_batch(original_batch)
        with self._tagger.tag("iteration/entry", it=self.iter):
            self.model.train()

        local_idx = self.local_expert_idx
        local_opt = self.optimizers[local_idx]

        local_oom_exc = None
        try:
            with self._tagger.tag("iteration/zero_grad(local)", it=self.iter, expert=local_idx):
                local_opt.zero_grad(set_to_none=True)

            with self._tagger.tag("expert/build_payload(fwd+loss)", it=self.iter, expert=local_idx):
                payload = self._build_train_payload(
                    batch_dict=batch_dict,
                    batch_info=batch_info,
                    expert_idx=local_idx,
                    range_dis=self.distance_ranges[local_idx],
                    ref_batch_dict=ref_batch_dict,
                    ref_batch_info=ref_batch_info,
                    criterion=self.train_lossfunc,
                )

            loss_local = payload["loss"]

            with self._tagger.tag("expert/backward", it=self.iter, expert=local_idx):
                loss_local.backward()
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            local_oom_exc = exc

        if local_oom_exc is not None:
            if self._maybe_skip_dynamic_batch_after_oom(
                original_batch,
                local_oom=True,
                where="distributed_fwd_bwd",
                ref_batch=ref_batch_dict,
            ):
                return None
            raise local_oom_exc

        with self._tagger.tag("expert/sync_dp_grads", it=self.iter, expert=local_idx):
            self._sync_local_expert_grads(local_idx)

        with self._tagger.tag("expert/clip_grad_norm", it=self.iter, expert=local_idx):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._expert_parameters(local_idx),
                max_norm=self.clip_grad_norm
            )

        with self._tagger.tag("expert/optimizer_step", it=self.iter, expert=local_idx):
            local_opt.step()

        with self._tagger.tag("expert/sync_dp_buffers", it=self.iter, expert=local_idx):
            self._sync_local_expert_buffers(local_idx)

        payload["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(
            float(grad_norm), device=self.device, dtype=self.dtype
        )
        payload["loss_detached"] = loss_local.detach()
        del payload["loss"]

        local_sched_metric = None
        if self._local_scheduler_requires_metric():
            with self._tagger.tag("iteration/compute_local_train_loss_compatible", it=self.iter):
                local_sched_metric = self._compute_local_compatible_loss_from_payload(payload, self.train_lossfunc)

        self._local_scheduler_step(local_sched_metric)

        current_local_lr = float(local_opt.param_groups[0]['lr'])
        self._update_display_window_local(
            payload,
            current_local_lr,
            dynamic_batch_state=dynamic_batch_state,
        )

        # The optimizer step for this batch is committed: advance the per-epoch
        # cursor BEFORE plugins so a checkpoint fired here persists the correct
        # mid-epoch position (OOM-skip paths return earlier and do not count).
        self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1

        # P0-5: the plugin dispatcher ticks on EVERY committed optimizer step,
        # decoupling cadence plugins (Saver save_freq, Validationer
        # validation_freq) from the display window. Previously call_plugins ran
        # only when the display window flushed, so those schedules were
        # quantized to (and drifted by) display_freq. Non-flush ticks carry a
        # cheap, locally-available state assembled WITHOUT any collective; the
        # expensive gathered/display metrics keep their display_sync_freq
        # cadence and ride the full flush state. Whether the flush fires (and
        # whether it yields a state — the reduced pack is rank-uniform) is
        # agreed on all ranks, so every rank issues the same single
        # call_plugins per committed step and no collective is added, removed,
        # or reordered relative to the flush path.
        state = None
        if self._should_flush_display_window_now(self.iter):
            state = self._flush_display_window(time_idx=self.iter)
        if state is None:
            state = self._cheap_iteration_state(
                loss_detached=payload["loss_detached"],
                current_local_lr=current_local_lr,
                dynamic_batch_state=dynamic_batch_state,
            )
        with self._tagger.tag("iteration/call_plugins", it=self.iter):
            self.call_plugins(queue_name='iteration', time=self.iter, **state)

        with self._tagger.tag("iteration/exit", it=self.iter):
            self.iter += 1

        return loss_local.detach()

    def _iteration_distributed_expert(self, batch, ref_batch=None):
        local_oom_exc = None
        batch_dict = None
        batch_info = None
        ref_batch_dict = None
        ref_batch_info = None
        try:
            with self._tagger.tag("iteration/prepare_batch", it=self.iter):
                batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

            if ref_batch is not None:
                with self._tagger.tag("iteration/prepare_ref_batch", it=self.iter):
                    ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            local_oom_exc = exc

        if local_oom_exc is not None:
            if self._maybe_skip_dynamic_batch_after_oom(
                batch,
                local_oom=True,
                where="distributed_prepare_batch",
                ref_batch=ref_batch,
            ):
                return None
            raise local_oom_exc

        return self._iteration_distributed_expert_prepared(
            batch_dict=batch_dict,
            batch_info=batch_info,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info,
            original_batch=batch,
        )

    def _iteration_distributed_expert_shared(self, batch, ref_batch=None):
        with self._tagger.tag("iteration/prepare_batch(shared_rank0_only)", it=self.iter):
            batch_dict, batch_info, ref_batch_dict, ref_batch_info = self._broadcast_prepared_bundle_rank0(
                batch=batch,
                ref_batch=ref_batch
            )

        return self._iteration_distributed_expert_prepared(
            batch_dict=batch_dict,
            batch_info=batch_info,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info,
            original_batch=batch,
        )

    # ---------------------------------------------------------------------
    # dynamic batch OOM fallback helpers
    # ---------------------------------------------------------------------

    @property
    def _dynamic_batch_controller(self) -> DynamicBatchController:
        # Lazily attached so trainers built via ``object.__new__`` (unit tests)
        # get a working controller without ``__init__`` running.
        dbc = self.__dict__.get("_dynamic_batch_controller_obj")
        if dbc is None:
            dbc = DynamicBatchController(self)
            self.__dict__["_dynamic_batch_controller_obj"] = dbc
        return dbc

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return DynamicBatchController.is_cuda_oom(exc)

    def _disable_dynamic_batch_oom_fallback(self, message: str, *args):
        return self._dynamic_batch_controller.disable_oom_fallback(message, *args)

    def _configure_dynamic_batch_oom_fallback(self):
        return self._dynamic_batch_controller.configure_oom_fallback()

    def _dynamic_batch_oom_requires_expert_dp_consensus(self) -> bool:
        return self._dynamic_batch_controller.requires_expert_dp_consensus()

    @classmethod
    def _dynamic_batch_state_from_batch(cls, batch) -> Dict[str, Any]:
        if batch is None:
            return {}
        state: Dict[str, Any] = {}
        for attr, key in cls._DYNAMIC_BATCH_STATE_ATTRS:
            if hasattr(batch, attr):
                state[key] = getattr(batch, attr)
        return state

    @classmethod
    def _dynamic_batch_oom_log_values(cls, batch) -> Dict[str, Any]:
        state = cls._dynamic_batch_state_from_batch(batch)
        return {
            "batch_cost": state.get("batch_cost"),
            "batch_max_item_cost": state.get("batch_max_item_cost"),
            "num_graphs": state.get("batch_num_graphs", getattr(batch, "num_graphs", None)),
        }

    def _clear_after_oom(self):
        return self._dynamic_batch_controller.clear_after_oom()

    def _runtime_dynamic_batch_sampler(self):
        return self._dynamic_batch_controller.runtime_sampler()

    def _set_runtime_dynamic_batch_max_cost(self, sampler, max_cost: int):
        return self._dynamic_batch_controller.set_runtime_max_cost(sampler, max_cost)

    def _invalidate_runtime_dynamic_batch_cache(self):
        return self._dynamic_batch_controller.invalidate_runtime_cache()

    def _shrink_dynamic_batch_after_oom(self, batch):
        return self._dynamic_batch_controller.shrink_after_oom(batch)

    def _can_skip_dynamic_batch_after_oom(self, ref_batch=None, optimizer_step_started: bool = False) -> bool:
        return self._dynamic_batch_controller.can_skip_after_oom(
            ref_batch=ref_batch, optimizer_step_started=optimizer_step_started
        )

    def _record_dynamic_batch_oom_skip(self):
        return self._dynamic_batch_controller.record_oom_skip()

    def _flush_display_window_after_oom_skip_if_needed(self, where: str):
        return self._dynamic_batch_controller.flush_display_after_oom_skip_if_needed(where)

    def _handle_dynamic_batch_oom_skip(self, batch, *, local_oom: bool, where: str):
        return self._dynamic_batch_controller.handle_oom_skip(batch, local_oom=local_oom, where=where)

    def _maybe_skip_dynamic_batch_after_oom(self, batch, *, local_oom: bool, where: str, ref_batch=None) -> bool:
        return self._dynamic_batch_controller.maybe_skip_after_oom(
            batch, local_oom=local_oom, where=where, ref_batch=ref_batch
        )

    # ---------------------------------------------------------------------
    # public iteration
    # ---------------------------------------------------------------------

    def iteration(self, batch, ref_batch=None):
        t_now = time.perf_counter()
        if self._t_last_iter_end is not None and self.debug_tags and (self.iter % self.debug_tag_freq == 0):
            log.info(f"[TAG][it={self.iter}][data_wait(outside_iteration)] dt={(t_now - self._t_last_iter_end):.4f}s")

        optimizer_step_started = False
        try:
            with self._maybe_profile_iteration(self.iter):
                if self.distributed_expert:
                    if self.distributed_rank0_prepare_batch:
                        return self._iteration_distributed_expert_shared(batch, ref_batch=ref_batch)
                    return self._iteration_distributed_expert(batch, ref_batch=ref_batch)

                # single-process fallback
                self._reset_cuda_memory_peak()
                dynamic_batch_state = self._dynamic_batch_state_from_batch(batch)
                with self._tagger.tag("iteration/entry", it=self.iter):
                    self.model.train()

                with self._tagger.tag("iteration/prepare_batch", it=self.iter):
                    batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

                ref_batch_dict = None
                ref_batch_info = None
                if ref_batch is not None:
                    with self._tagger.tag("iteration/prepare_ref_batch", it=self.iter):
                        ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)

                total_loss_opt = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
                expert_grad_norms = []

                global_onsite_sum = 0.0
                global_hopping_sum = 0.0
                total_active_nodes = 0
                total_active_edges = 0
                expert_onsite_dict = {}
                expert_hopping_dict = {}
                z_metric_values = []
                expert_load_cv_values = []

                reduce_payloads: List[Dict[str, Any]] = []

                def collect_payload(expert_idx, payload):
                    nonlocal total_loss_opt
                    nonlocal global_onsite_sum, global_hopping_sum
                    nonlocal total_active_nodes, total_active_edges

                    total_loss_opt = total_loss_opt + payload["loss_detached"]
                    expert_grad_norms.append(self._to_float_scalar(payload["grad_norm"]))

                    expert_onsite = self._to_float_scalar(payload["expert_onsite"])
                    expert_hopping = self._to_float_scalar(payload["expert_hopping"])
                    expert_onsite_dict[f"expert_{expert_idx}_onsite"] = expert_onsite
                    expert_hopping_dict[f"expert_{expert_idx}_hopping"] = expert_hopping

                    global_onsite_sum += self._to_float_scalar(payload["onsite_weighted_sum"])
                    global_hopping_sum += self._to_float_scalar(payload["hopping_weighted_sum"])
                    total_active_nodes += self._to_int_scalar(payload["active_nodes"])
                    total_active_edges += self._to_int_scalar(payload["active_edges"])

                    for z in payload.get("z_values", []):
                        if z is not None:
                            z_metric_values.append(self._to_float_scalar(z))
                    for cv in payload.get("load_cv_values", []):
                        if cv is not None:
                            expert_load_cv_values.append(self._to_float_scalar(cv))

                    reduce_payloads.append(payload)

                with self._tagger.tag("iteration/zero_grad(all)", it=self.iter):
                    for opt in self.optimizers:
                        opt.zero_grad(set_to_none=True)

                payload_list = []
                for expert_idx, range_dis in enumerate(self.distance_ranges):
                    with self._tagger.tag("expert/build_payload(fwd+loss)", it=self.iter, expert=expert_idx):
                        payload = self._build_train_payload(
                            batch_dict=batch_dict,
                            batch_info=batch_info,
                            expert_idx=expert_idx,
                            range_dis=range_dis,
                            ref_batch_dict=ref_batch_dict,
                            ref_batch_info=ref_batch_info,
                        )

                    loss_expert = payload["loss"]

                    with self._tagger.tag("expert/backward", it=self.iter, expert=expert_idx):
                        loss_expert.backward()

                    payload["loss_detached"] = loss_expert.detach()
                    del payload["loss"]
                    payload_list.append(payload)

                for expert_idx, payload in enumerate(payload_list):
                    with self._tagger.tag("expert/clip_grad_norm", it=self.iter, expert=expert_idx):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._expert_parameters(expert_idx),
                            max_norm=self.clip_grad_norm
                        )

                    with self._tagger.tag("expert/optimizer_step", it=self.iter, expert=expert_idx):
                        optimizer_step_started = True
                        self.optimizers[expert_idx].step()

                    payload["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(
                        float(grad_norm), device=self.device, dtype=self.dtype
                    )

                with self._tagger.tag("iteration/collect_payloads", it=self.iter):
                    for expert_idx, payload in enumerate(payload_list):
                        collect_payload(expert_idx, payload)

                global_onsite = global_onsite_sum / max(total_active_nodes, 1)
                global_hopping = global_hopping_sum / max(total_active_edges, 1)

                local_pack = torch.zeros(MetricPack.LENGTH, device=self.device, dtype=self.dtype)
                for payload in reduce_payloads:
                    local_pack = local_pack + self._make_step_pack(payload)

                with self._tagger.tag("iteration/compute_train_loss_compatible(reduce)", it=self.iter):
                    compatible_train_state = self._compute_compatible_state_from_pack(
                        local_pack,
                        self.train_lossfunc,
                        prefix="train",
                    )

                final_train_loss = (
                    compatible_train_state["train_loss"]
                    if compatible_train_state is not None
                    else total_loss_opt
                )
                self._local_scheduler_step(final_train_loss)
                # ---------------- NEW: make lr consistent: use mean lr across experts ----------------
                avg_lr = sum(float(opt.param_groups[0]['lr']) for opt in self.optimizers) / max(len(self.optimizers), 1)
                # ----------------------------------------------------------------------------------

                state = {
                    'field': 'iteration',
                    'window_steps': 1,
                    "train_loss": final_train_loss,
                    "train_loss_opt": total_loss_opt,
                    "lr": avg_lr,
                    "total_grad_norm": sum(expert_grad_norms) / max(len(expert_grad_norms), 1),
                }
                if compatible_train_state is not None:
                    state.update({
                        "train_onsite_loss": self._to_float_scalar(
                            compatible_train_state["train_onsite_loss"]
                        ),
                        "train_hopping_loss": self._to_float_scalar(
                            compatible_train_state["train_hopping_loss"]
                        ),
                    })

                for i in range(self.num_experts):
                    if Trainer._supports_endpoint_triplet(self.train_lossfunc):
                        state[f"expert_{i}_onsite"] = expert_onsite_dict.get(f"expert_{i}_onsite", 0.0)
                        state[f"expert_{i}_hopping"] = expert_hopping_dict.get(f"expert_{i}_hopping", 0.0)
                    state[f"expert_{i}_lr"] = float(self.optimizers[i].param_groups[0]['lr'])

                if expert_load_cv_values:
                    state["expert_load_cv"] = sum(expert_load_cv_values) / len(expert_load_cv_values)
                if z_metric_values:
                    state["mean_max_prob"] = sum(z_metric_values) / len(z_metric_values)
                state.update(dynamic_batch_state)

                if self._optimizer_diagnostics_due():
                    self._add_optimizer_diagnostics_to_state(state)
                self._add_cuda_memory_state(state, self._gather_cuda_memory_metrics())

                # All expert optimizer steps for this batch are committed:
                # advance the per-epoch cursor BEFORE plugins so a checkpoint
                # fired here persists the correct mid-epoch position.
                self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1

                with self._tagger.tag("iteration/call_plugins", it=self.iter):
                    self.call_plugins(queue_name='iteration', time=self.iter, **state)

                with self._tagger.tag("iteration/exit", it=self.iter):
                    self.iter += 1

                return total_loss_opt

        except RuntimeError as e:
            if self._is_cuda_oom(e):
                self._tagger.dump_cuda_mem_summary(where="iteration() top-level")
                can_skip = (
                    not self.distributed_expert
                    and self._can_skip_dynamic_batch_after_oom(
                        ref_batch=ref_batch,
                        optimizer_step_started=optimizer_step_started,
                    )
                )
                if can_skip:
                    log.warning(
                        "dynamic_batch caught CUDA OOM at iter=%s; skip current batch.",
                        self.iter,
                    )
                    self._handle_dynamic_batch_oom_skip(batch, local_oom=True, where="single_process")
                    return None
                else:
                    self._clear_after_oom()
            raise
        finally:
            self._t_last_iter_end = time.perf_counter()

    # ---------------------------------------------------------------------
    # epoch override
    # ---------------------------------------------------------------------

    def epoch(self) -> None:
        # Reset the committed-batch cursor at each epoch start (mirrors
        # Trainer.epoch) so iteration checkpoints persist a per-epoch position.
        self._batch_in_epoch = 0
        self._set_expert_dp_sampler_epoch(self.ep)

        if self.distributed_expert and self.distributed_rank0_prepare_batch:
            train_iter = iter(self.train_loader) if self.rank == 0 else None
            ref_iter = iter(self.reference_loader) if (self.use_reference and self.rank == 0) else None

            n_step = len(self.train_loader)
            for _ in range(n_step):
                if self.rank == 0:
                    batch = next(train_iter)
                    if self.use_reference:
                        try:
                            ref_batch = next(ref_iter)
                        except StopIteration:
                            ref_iter = iter(self.reference_loader)
                            ref_batch = next(ref_iter)
                    else:
                        ref_batch = None
                else:
                    batch = None
                    ref_batch = None

                self.iteration(batch, ref_batch)

            if self._has_pending_display_window():
                flush_time = max(self.iter - 1, 1)
                state = self._flush_display_window(time_idx=flush_time)
                if state is not None:
                    self.call_plugins(queue_name='iteration', time=flush_time, **state)
            return

        if self.use_reference:
            ref_iter = iter(self.reference_loader)
            for ibatch in self.train_loader:
                try:
                    ref_batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(self.reference_loader)
                    ref_batch = next(ref_iter)
                self.iteration(ibatch, ref_batch)
        else:
            for ibatch in self.train_loader:
                self.iteration(ibatch)

        if self.distributed_expert and self._has_pending_display_window():
            flush_time = max(self.iter - 1, 1)
            state = self._flush_display_window(time_idx=flush_time)
            if state is not None:
                self.call_plugins(queue_name='iteration', time=flush_time, **state)

    # ---------------------------------------------------------------------
    # validation
    # ---------------------------------------------------------------------

    def _run_full_batch_loss(self, batch_dict, batch_info, criterion):
        batch_copy = batch_dict.copy()
        batch_for_loss = batch_copy.copy()

        with cuda_cache_memory_context(iteration=self.iter, stage="validation/full_forward"):
            pred_batch = self.model(batch_copy)
        pred_batch["global_step"] = int(self.iter)
        pred_batch.update(batch_info)
        batch_for_loss.update(batch_info)

        optimization_loss = criterion(pred_batch, batch_for_loss)
        endpoint_state = Trainer._endpoint_loss_state(
            criterion,
            optimization_loss,
            prefix="validation",
        )
        if Trainer._supports_endpoint_triplet(criterion):
            Trainer._require_endpoint_triplet(
                endpoint_state,
                prefix="validation",
                route="MultiTrainer full-forward validation",
            )
        return endpoint_state["validation_loss"]

    def _build_validation_euler_payload(
        self,
        batch_dict,
        batch_info,
        criterion,
        expert_idx,
        range_dis,
        *,
        num_steps: int,
    ):
        with self._tagger.tag("validation/prepare_euler_masks", it=self.iter, expert=expert_idx):
            expert_edge_mask, expert_node_mask = self._prepare_expert_masks(
                batch_dict, range_dis, expert_idx
            )

        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask
        batch_copy["expert_idx"] = int(expert_idx)

        active_nodes = expert_node_mask.sum().detach()
        active_edges = expert_edge_mask.sum().detach()

        with self._tagger.tag(
            "validation/flow_sample_euler",
            it=self.iter,
            expert=expert_idx,
            extra=f"steps={int(num_steps)}",
        ):
            sampled = self.flow_cfm.sample(
                self.model,
                batch_copy,
                num_steps=int(num_steps),
            )

        sampled["global_step"] = int(self.iter)
        sampled["expert_edge_mask"] = expert_edge_mask
        sampled["expert_node_mask"] = expert_node_mask
        sampled["expert_idx"] = int(expert_idx)
        sampled.update(batch_info)

        batch_for_loss = batch_copy.copy()
        batch_for_loss.update(batch_info)

        with self._tagger.tag(
            "validation/euler_compatible_loss",
            it=self.iter,
            expert=expert_idx,
            extra=f"steps={int(num_steps)}",
        ):
            loss = criterion(sampled, batch_for_loss)
        metrics = self._snapshot_loss_metrics(criterion)

        onsite_weighted_sum = metrics["onsite"] * active_nodes.to(dtype=self.dtype)
        hopping_weighted_sum = metrics["hopping"] * active_edges.to(dtype=self.dtype)

        return {
            "loss": loss,
            "loss_detached": loss.detach() if torch.is_tensor(loss) else loss,
            "expert_onsite": metrics["onsite"].detach(),
            "expert_hopping": metrics["hopping"].detach(),
            "onsite_weighted_sum": onsite_weighted_sum.detach(),
            "hopping_weighted_sum": hopping_weighted_sum.detach(),
            "active_nodes": active_nodes.detach(),
            "active_edges": active_edges.detach(),
            "onsite_l1_sum": metrics["last_onsite_l1_sum"].detach()
            if torch.is_tensor(metrics["last_onsite_l1_sum"])
            else None,
            "onsite_mse_sum": metrics["last_onsite_mse_sum"].detach()
            if torch.is_tensor(metrics["last_onsite_mse_sum"])
            else None,
            "onsite_cnt": metrics["last_onsite_count"].detach()
            if torch.is_tensor(metrics["last_onsite_count"])
            else None,
            "hopping_l1_sum": metrics["last_hopping_l1_sum"].detach()
            if torch.is_tensor(metrics["last_hopping_l1_sum"])
            else None,
            "hopping_mse_sum": metrics["last_hopping_mse_sum"].detach()
            if torch.is_tensor(metrics["last_hopping_mse_sum"])
            else None,
            "hopping_cnt": metrics["last_hopping_count"].detach()
            if torch.is_tensor(metrics["last_hopping_count"])
            else None,
            "z_values": [metrics["z_loss"].detach()]
            if torch.is_tensor(metrics["z_loss"])
            else ([] if metrics["z_loss"] is None else [metrics["z_loss"]]),
            "load_cv_values": [metrics["expert_load_cv"].detach()]
            if torch.is_tensor(metrics["expert_load_cv"])
            else ([] if metrics["expert_load_cv"] is None else [metrics["expert_load_cv"]]),
        }

    def _validation_euler_state_from_pack(
        self,
        pack: torch.Tensor,
        criterion,
        *,
        num_steps: int,
    ) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        compatible_prefix = f"validation_compatible_euler_{int(num_steps)}"
        compatible_state = self._compute_compatible_state_from_pack(
            pack,
            criterion=criterion,
            prefix=compatible_prefix,
            global_step=getattr(self, "iter", None),
        )
        if compatible_state is not None:
            state.update(compatible_state)

        if int(num_steps) == 1:
            legacy_state = self._compute_compatible_state_from_pack(
                pack,
                criterion=criterion,
                prefix="validation",
                global_step=getattr(self, "iter", None),
            )
            if legacy_state is not None:
                state.update(legacy_state)

        return state

    def validation(self, fast=True):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            validation_metric_sums: Dict[str, Any] = {}
            num_batches = 0
            self.model.eval()

            for batch in self.validation_loader:
                with self._tagger.tag("validation/prepare_batch", it=self.iter):
                    batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

                flow_euler_validation = bool(getattr(getattr(self, "flow_cfm", None), "enabled", False))

                if self.distributed_expert:
                    local_idx = self.local_expert_idx
                    if flow_euler_validation:
                        loss_i = None
                        for num_steps in self.flow_cfm.validation_ode_steps:
                            payload = self._build_validation_euler_payload(
                                batch_dict=batch_dict,
                                batch_info=batch_info,
                                criterion=self.validation_lossfunc,
                                expert_idx=local_idx,
                                range_dis=self.distance_ranges[local_idx],
                                num_steps=int(num_steps),
                            )
                            with self._tagger.tag(
                                "validation/reduce_euler_metrics_dist",
                                it=self.iter,
                                extra=f"steps={int(num_steps)}",
                            ):
                                reduced_pack = self._make_step_pack(payload)
                                self._all_reduce_(
                                    reduced_pack,
                                    name=f"dist/all_reduce(validation_euler_{int(num_steps)}_metrics_packed)",
                                )
                            state = self._validation_euler_state_from_pack(
                                reduced_pack,
                                self.validation_lossfunc,
                                num_steps=int(num_steps),
                            )
                            self._accumulate_metric_state(validation_metric_sums, state)
                            if loss_i is None:
                                loss_i = state.get(
                                    "validation_loss",
                                    state.get(
                                        f"validation_compatible_euler_{int(num_steps)}_loss",
                                        None,
                                    ),
                                )
                        if loss_i is None:
                            loss_i = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
                    else:
                        payload = self._build_train_payload(
                            batch_dict=batch_dict,
                            batch_info=batch_info,
                            expert_idx=local_idx,
                            range_dis=self.distance_ranges[local_idx],
                            ref_batch_dict=None,
                            ref_batch_info=None,
                            criterion=self.validation_lossfunc,
                            flow_prefix="validation",
                        )

                        payload["loss_detached"] = payload["loss"].detach()

                        with self._tagger.tag("validation/reduce_packed_metrics_dist", it=self.iter):
                            reduced_pack = self._make_step_pack(payload)
                            self._all_reduce_(reduced_pack, name="dist/all_reduce(validation_metrics_packed)")

                        with self._tagger.tag("validation/compute_reduce_loss_dist_packed", it=self.iter):
                            loss_i = self._compute_compatible_loss_from_pack(reduced_pack, self.validation_lossfunc)
                        if loss_i is None:
                            loss_i = MetricPack.from_tensor(reduced_pack).loss_opt_sum.detach() / max(self.world_size, 1)

                        self._accumulate_metric_state(
                            validation_metric_sums,
                            self._pack_component_state(
                                reduced_pack,
                                prefix="validation",
                                criterion=self.validation_lossfunc,
                            ),
                        )

                else:
                    if flow_euler_validation:
                        loss_i = None
                        for num_steps in self.flow_cfm.validation_ode_steps:
                            local_pack = torch.zeros(MetricPack.LENGTH, device=self.device, dtype=self.dtype)
                            for expert_idx, range_dis in enumerate(self.distance_ranges):
                                payload = self._build_validation_euler_payload(
                                    batch_dict=batch_dict,
                                    batch_info=batch_info,
                                    criterion=self.validation_lossfunc,
                                    expert_idx=expert_idx,
                                    range_dis=range_dis,
                                    num_steps=int(num_steps),
                                )
                                local_pack = local_pack + self._make_step_pack(payload)
                            state = self._validation_euler_state_from_pack(
                                local_pack,
                                self.validation_lossfunc,
                                num_steps=int(num_steps),
                            )
                            self._accumulate_metric_state(validation_metric_sums, state)
                            if loss_i is None:
                                loss_i = state.get(
                                    "validation_loss",
                                    state.get(
                                        f"validation_compatible_euler_{int(num_steps)}_loss",
                                        None,
                                    ),
                                )
                        if loss_i is None:
                            loss_i = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
                    elif self.endpoint_loss_mode == "reduce":
                        payloads = []
                        local_pack = torch.zeros(MetricPack.LENGTH, device=self.device, dtype=self.dtype)
                        for expert_idx, range_dis in enumerate(self.distance_ranges):
                            res = self._run_one_expert_loss(
                                batch_dict=batch_dict,
                                batch_info=batch_info,
                                criterion=self.validation_lossfunc,
                                expert_idx=expert_idx,
                                range_dis=range_dis,
                                capture_metrics=True,
                                flow_prefix="validation",
                            )
                            res["loss_detached"] = res["loss"].detach()
                            local_pack = local_pack + self._make_step_pack(res)
                            payloads.append({
                                "onsite_l1_sum": res.get("last_onsite_l1_sum", None),
                                "onsite_mse_sum": res.get("last_onsite_mse_sum", None),
                                "onsite_cnt": res.get("last_onsite_count", None),
                                "hopping_l1_sum": res.get("last_hopping_l1_sum", None),
                                "hopping_mse_sum": res.get("last_hopping_mse_sum", None),
                                "hopping_cnt": res.get("last_hopping_count", None),
                                "z_values": [res["z_loss"]] if res.get("z_loss", None) is not None else [],
                            })

                        with self._tagger.tag("validation/compute_reduce_loss", it=self.iter):
                            loss_i = self._compute_stitched_loss_by_reduce(payloads, self.validation_lossfunc)

                        if loss_i is None:
                            if bool(getattr(getattr(self, "flow_cfm", None), "enabled", False)):
                                local_mp = MetricPack.from_tensor(local_pack)
                                loss_i = local_mp.loss_opt_sum.detach() / local_mp.step_count.clamp_min(1.0)
                                self._accumulate_metric_state(
                                    validation_metric_sums,
                                    self._pack_component_state(
                                        local_pack,
                                        prefix="validation",
                                        criterion=self.validation_lossfunc,
                                    ),
                                )
                            else:
                                with self._tagger.tag("validation/fallback_full_forward", it=self.iter):
                                    loss_i = self._run_full_batch_loss(batch_dict, batch_info, self.validation_lossfunc)
                                    if Trainer._supports_endpoint_triplet(self.validation_lossfunc):
                                        fallback_metrics = self._snapshot_loss_metrics(self.validation_lossfunc)
                                        self._accumulate_metric_state(
                                            validation_metric_sums,
                                            {
                                                "validation_onsite_loss": fallback_metrics["onsite"],
                                                "validation_hopping_loss": fallback_metrics["hopping"],
                                            },
                                        )
                        else:
                            self._accumulate_metric_state(
                                validation_metric_sums,
                                self._pack_component_state(
                                    local_pack,
                                    prefix="validation",
                                    criterion=self.validation_lossfunc,
                                ),
                            )
                    else:
                        with self._tagger.tag("validation/full_forward_stitched", it=self.iter):
                            loss_i = self._run_full_batch_loss(batch_dict, batch_info, self.validation_lossfunc)
                        if Trainer._supports_endpoint_triplet(self.validation_lossfunc):
                            full_metrics = self._snapshot_loss_metrics(self.validation_lossfunc)
                            self._accumulate_metric_state(
                                validation_metric_sums,
                                {
                                    "validation_onsite_loss": full_metrics["onsite"],
                                    "validation_hopping_loss": full_metrics["hopping"],
                                },
                            )

                total_loss = total_loss + loss_i.detach()
                num_batches += 1

                if fast:
                    break

        if not fast:
            total_loss = total_loss / len(self.validation_loader)
        divisor = max(num_batches, 1)
        self._last_flow_validation_state = {
            key: value / divisor for key, value in validation_metric_sums.items()
        }

        return total_loss

    # ---------------------------------------------------------------------
    # restart
    # ---------------------------------------------------------------------

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options={}, common_options={}, reference_datasets=None,
                validation_datasets=None, distributed_expert=False, rank=0, world_size=1):
        map_loc = "cpu" if distributed_expert else (
            common_options["device"] if len(common_options) > 0 and "device" in common_options else "cpu"
        )
        ckpt = torch.load(checkpoint, map_location=map_loc, weights_only=False)

        ckpt_train_options = migrate_legacy_checkpoint_train_options(
            ckpt["config"].get("train_options", {})
        )
        merged_train_options = merge_restart_train_options(
            train_options,
            ckpt_train_options,
            logger=log,
        )

        merged_common_options = copy.deepcopy(ckpt["config"]["common_options"])
        merged_common_options.update(common_options or {})

        # Fail-safe for checkpoints written while an entrypoint hack mutated
        # common_options["overlap"]=False AFTER the model was built with the
        # pristine flag: the persisted config then contradicts the persisted
        # weights and a strict ensemble rebuild fails on unexpected
        # overlap-head keys. The weights are the ground truth — infer the flag.
        state_for_probe = ckpt.get("model_state_dict") or {}
        if not merged_common_options.get("overlap", False) and _state_dict_has_overlap_head(state_for_probe):
            log.warning(
                "Checkpoint config says overlap=False but the model state "
                "contains overlap-head parameters; rebuilding with "
                "overlap=True (config was written by a version that mutated "
                "the flag after model construction)."
            )
            merged_common_options["overlap"] = True

        build_common_options = copy.deepcopy(merged_common_options)
        if distributed_expert:
            build_common_options["device"] = "cpu"

        model = build_model(
            checkpoint=checkpoint,
            model_options=ckpt["config"]["model_options"],
            common_options=build_common_options,
            train_options=ckpt_train_options
        )

        distance_ranges = ckpt_train_options.get(
            "distance_ranges",
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
        )

        trainer = cls(
            distance_ranges=distance_ranges,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
            train_options=merged_train_options,
            common_options=merged_common_options,
            distributed_expert=distributed_expert,
            rank=rank,
            world_size=world_size,
        )

        trainer.iter = int(ckpt["iteration"]) + 1
        trainer.stats = ckpt["stats"]

        # ---- resume state machine (legacy-tolerant) -----------------------
        resume = read_resume_metadata(ckpt)
        trainer._restored_plugin_state = dict(resume.plugin_state or {})

        # Load optimizer/scheduler states first (before any epoch-end LR replay).
        if distributed_expert:
            idx = trainer.local_expert_idx
            opt_states = ckpt.get("optimizers_state_dict", None)
            sch_states = ckpt.get("lr_schedulers_state_dict", None)
            if opt_states is not None and trainer.optimizers[idx] is not None:
                trainer.optimizers[idx].load_state_dict(opt_states[idx])
            if sch_states is not None and trainer.lr_schedulers[idx] is not None:
                trainer.lr_schedulers[idx].load_state_dict(sch_states[idx])
        else:
            for key in cls.object_keys:
                items = getattr(trainer, key, None)
                if items is not None:
                    saved_states = ckpt[key + "_state_dict"]
                    # Strict length check: a silent zip truncation would leave
                    # some experts on freshly-initialized optimizer/scheduler
                    # state while others resume — a hard-to-detect divergence.
                    if len(saved_states) != len(items):
                        raise RuntimeError(
                            f"Checkpoint {key}_state_dict holds "
                            f"{len(saved_states)} entries but the trainer has "
                            f"{len(items)} experts; refusing a partial restore."
                        )
                    for obj, state in zip(items, saved_states):
                        if obj is not None:
                            obj.load_state_dict(state)

        if resume.checkpoint_kind == CHECKPOINT_KIND_ITERATION:
            # Mid-epoch checkpoint: the checkpointed models/optimizers already
            # contain the committed prefix's updates, and MultiTrainer has no
            # exact per-batch fast-forward yet — re-running the epoch would
            # apply those optimizer steps a second time. Fail closed unless the
            # user explicitly opts into the inexact re-run.
            if os.environ.get("DPTB_ALLOW_INEXACT_RESUME", "").strip() not in ("1", "true", "True"):
                raise RuntimeError(
                    "Cannot resume a MultiTrainer run from a mid-epoch "
                    "(iteration) checkpoint: no exact batch fast-forward is "
                    "implemented for the expert-parallel loaders, and re-running "
                    "the epoch re-applies the committed prefix's optimizer "
                    "updates. Resume from the last epoch checkpoint instead, or "
                    "set DPTB_ALLOW_INEXACT_RESUME=1 to accept the inexact "
                    "re-run."
                )
            log.warning(
                "DPTB_ALLOW_INEXACT_RESUME=1: re-running epoch %s from its "
                "start; optimizer updates for the committed prefix WILL be "
                "applied a second time.",
                int(resume.epoch),
            )
            trainer.ep = int(resume.epoch)
            # The restored stats hold the committed prefix's partial per-epoch
            # accumulators; the full re-run would double-count them into
            # epoch_mean (plateau LR + best gate). Reset before training.
            trainer._reset_epoch_metric_accumulators()
        else:
            # Epoch-committed checkpoint: advance and replay the one pending
            # per-epoch LR step (Saver.epoch persists the scheduler before run()
            # steps it) to avoid an off-by-one LR transition on resume (BUG 1).
            trainer.ep = int(resume.epoch) + 1
            # Prefer this rank's own RNG snapshot over the main-rank blob (P0-3)
            # so the ranks' dropout/noise streams stay decorrelated on resume.
            own_rng = resolve_rank_rng_state(ckpt, resume)
            if own_rng is not None:
                restore_rng_state(own_rng)
            if resume.epoch_scheduler_step_pending and not trainer.update_lr_per_iter:
                try:
                    trainer._step_epoch_schedulers()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to replay epoch-end LR step on restart: %s", exc
                    )

        return trainer


