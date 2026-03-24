import contextlib
import time
import logging
import gc
import math
from typing import Union, Optional, Dict, Any, List

import torch
import torch.nn as nn

from dptb.utils.tools import get_lr_scheduler, get_optimizer
from dptb.data import AtomicDataset, AtomicData, AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model

log = logging.getLogger(__name__)


# =============================================================================
# TAGGER
# =============================================================================
class _StageTagger:
    def __init__(self, trainer, enabled: bool, freq: int, cuda_mem: bool, cuda_sync: bool, oom_dump: bool):
        self.trainer = trainer
        self.enabled = bool(enabled)
        self.freq = max(int(freq), 1)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)
        self.oom_dump = bool(oom_dump)

    def _device(self) -> torch.device:
        return self.trainer.device if isinstance(self.trainer.device, torch.device) else torch.device(
            self.trainer.device)

    def _is_cuda(self) -> bool:
        dev = self._device()
        return torch.cuda.is_available() and dev.type == "cuda"

    def _cuda_mem(self):
        if not self._is_cuda(): return None
        dev = self._device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        free, total = torch.cuda.mem_get_info(dev)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None: return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return f" | cuda_alloc={alloc / mb:.1f}MB cuda_reserved={reserved / mb:.1f}MB cuda_peak={peak / mb:.1f}MB free={free / mb:.1f}MB total={total / mb:.1f}MB"

    def dump_cuda_mem_summary(self, where: str):
        if not self._is_cuda(): return
        dev = self._device()
        mb = 1024 ** 2
        alloc = torch.cuda.memory_allocated(dev) / mb
        reserved = torch.cuda.memory_reserved(dev) / mb
        peak = torch.cuda.max_memory_allocated(dev) / mb
        free, total = torch.cuda.mem_get_info(dev)
        log.error(
            f"[OOM-DUMP] where={where} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB free={free / mb:.1f}MB total={total / mb:.1f}MB")
        if self.oom_dump:
            try:
                log.error("[OOM-DUMP] memory_summary:\n%s", torch.cuda.memory_summary(dev, abbreviated=False))
            except Exception:
                pass

    @contextlib.contextmanager
    def tag(self, name: str, *, it: Optional[int] = None, expert: Optional[int] = None, extra: str = ""):
        if not self.enabled or (it is not None and it % self.freq != 0):
            yield
            return

        prefix = f"[TAG][it={it}][expert={expert}][{name}]" if expert is not None else f"[TAG][it={it}][{name}]"

        nvtx_pushed = False
        if self._is_cuda():
            try:
                torch.cuda.nvtx.range_push(f"{prefix}{(' ' + extra) if extra else ''}")
                nvtx_pushed = True
            except Exception:
                pass

        dev = self._device()
        if self.cuda_mem and self._is_cuda():
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        t0 = time.perf_counter()
        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): self.dump_cuda_mem_summary(where=f"{name} it={it} expert={expert}")
            raise
        finally:
            if self.cuda_sync and self._is_cuda(): torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem() if self.cuda_mem else None
            log.info(f"{prefix} dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


# =============================================================================
# MultiTrainer
# =============================================================================
class MultiTrainer(Trainer):
    object_keys = ["lr_schedulers", "optimizers"]

    def __init__(
            self, distance_ranges: list, train_options: dict, common_options: dict,
            model: torch.nn.Module, train_datasets: AtomicDataset,
            reference_datasets: Union[AtomicDataset, None] = None,
            validation_datasets: Union[AtomicDataset, None] = None,
    ) -> None:
        super().__init__(
            train_options=train_options, common_options=common_options, model=model,
            train_datasets=train_datasets, reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.parallel_multi = bool(self.train_options.get("parallel_multi", False))

        self.debug_tags = bool(self.train_options.get("debug_tags", False))
        self.debug_tag_freq = int(self.train_options.get("debug_tag_freq", 1))
        self.debug_tag_cuda_mem = bool(self.train_options.get("debug_tag_cuda_mem", True))
        self.debug_tag_cuda_sync = bool(self.train_options.get("debug_tag_cuda_sync", False))
        self.debug_oom_dump = bool(self.train_options.get("debug_oom_dump", True))

        self._tagger = _StageTagger(self, self.debug_tags, self.debug_tag_freq, self.debug_tag_cuda_mem,
                                    self.debug_tag_cuda_sync, self.debug_oom_dump)

        self.log_single_model_compatible_loss = bool(self.train_options.get("log_single_model_compatible_loss", True))
        self.log_single_model_compatible_loss_mode = str(
            self.train_options.get("log_single_model_compatible_loss_mode", "reduce")).lower()
        self.shared_scheduler_metric = str(self.train_options.get("shared_scheduler_metric", "train_loss_opt")).lower()

        # 核心显存控制参数
        self.serial_offload_experts = bool(self.train_options.get("serial_offload_experts", False))
        self.serial_empty_cache_per_expert = bool(self.train_options.get("serial_empty_cache_per_expert", True))
        self.serial_gc_collect_per_expert = bool(self.train_options.get("serial_gc_collect_per_expert", False))
        self.serial_zero_grad_immediately = bool(self.train_options.get("serial_zero_grad_immediately", True))

        self._warned_validation_force_reduce = False
        self._t_last_iter_end: Optional[float] = None

        log.info(f"[MultiTrainer] Initialization Complete.")
        log.info(f" -> parallel_multi = {self.parallel_multi}")
        log.info(f" -> serial_offload_experts = {self.serial_offload_experts}")
        log.info(f" -> serial_empty_cache_per_expert = {self.serial_empty_cache_per_expert}")

        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

        self.optimizers = []
        self.lr_schedulers = []
        for i in range(self.num_experts):
            opt = get_optimizer(model_param=self.model.experts[i].parameters(), **self.train_options["optimizer"])
            sch = get_lr_scheduler(optimizer=opt, **self.train_options["lr_scheduler"])
            self.optimizers.append(opt)
            self.lr_schedulers.append(sch)

        if hasattr(self, "optimizer"): del self.optimizer
        if hasattr(self, "lr_scheduler"): del self.lr_scheduler

        if self.serial_offload_experts and self._is_cuda_device():
            self._offload_all_experts_to_cpu(move_optimizer_state=True)

    def _device_obj(self):
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda_device(self):
        return self._device_obj().type == "cuda" and torch.cuda.is_available()

    def _use_cuda_stream_parallel(self):
        return self.parallel_multi and self.num_experts > 1 and self._is_cuda_device()

    def _move_state_value_to_device(self, obj, device: torch.device):
        if torch.is_tensor(obj): return obj.to(device)
        if isinstance(obj, dict): return {k: self._move_state_value_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list): return [self._move_state_value_to_device(v, device) for v in obj]
        if isinstance(obj, tuple): return tuple(self._move_state_value_to_device(v, device) for v in obj)
        return obj

    def _optimizer_to(self, optimizer, device: torch.device):
        for state in optimizer.state.values():
            for k, v in list(state.items()): state[k] = self._move_state_value_to_device(v, device)

    def _offload_all_experts_to_cpu(self, move_optimizer_state: bool = True):
        if not self._is_cuda_device(): return
        cpu = torch.device("cpu")
        for i in range(self.num_experts):
            self.model.experts[i].to(cpu)
            if move_optimizer_state: self._optimizer_to(self.optimizers[i], cpu)

    def _activate_expert_for_train(self, expert_idx: int):
        if not (self.serial_offload_experts and self._is_cuda_device()): return
        dev = self._device_obj()
        self.model.experts[expert_idx].to(dev)
        self._optimizer_to(self.optimizers[expert_idx], dev)

    def _cleanup_after_train_expert(self, expert_idx: int):
        opt = self.optimizers[expert_idx]
        if self.serial_zero_grad_immediately: opt.zero_grad(set_to_none=True)

        if self.serial_offload_experts and self._is_cuda_device():
            cpu = torch.device("cpu")
            self.model.experts[expert_idx].to(cpu)
            self._optimizer_to(opt, cpu)

        # 核心：防碎片化 OOM
        if self.serial_empty_cache_per_expert and self._is_cuda_device():
            torch.cuda.empty_cache()
        if self.serial_gc_collect_per_expert:
            gc.collect()

    def _activate_expert_for_eval(self, expert_idx: int):
        if not (self.serial_offload_experts and self._is_cuda_device()): return
        self.model.experts[expert_idx].to(self._device_obj())

    def _cleanup_after_eval_expert(self, expert_idx: int):
        if self.serial_offload_experts and self._is_cuda_device():
            self.model.experts[expert_idx].to(torch.device("cpu"))
        if self.serial_empty_cache_per_expert and self._is_cuda_device():
            torch.cuda.empty_cache()
        if self.serial_gc_collect_per_expert:
            gc.collect()

    def _prepare_expert_masks(self, batch_dict, range_dis, expert_idx):
        d_min, d_max = range_dis
        dist = batch_dict['edge_lengths']
        expert_edge_mask = (dist >= d_min) if expert_idx == self.num_experts - 1 else (dist >= d_min) & (dist < d_max)

        num_nodes = batch_dict.get(getattr(AtomicDataDict, "ATOM_TYPE_KEY", "atom_types"),
                                   batch_dict.get(getattr(AtomicDataDict, "POSITIONS_KEY", "pos"), dist)).shape[0]
        expert_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
        if d_min > 0: expert_node_mask.fill_(False)
        return expert_edge_mask, expert_node_mask

    def _expert_has_work(self, expert_edge_mask, expert_node_mask) -> bool:
        return bool(expert_edge_mask.any().item()) or bool(expert_node_mask.any().item())

    def _prepare_batch_bundle(self, batch, with_lengths=True):
        batch_dev = batch.to(self.device)
        batch_info = {
            "__slices__": batch_dev.__slices__, "__cumsum__": batch_dev.__cumsum__,
            "__cat_dims__": batch_dev.__cat_dims__, "__num_nodes_list__": batch_dev.__num_nodes_list__,
            "__data_class__": batch_dev.__data_class__,
        }
        batch_dict = AtomicData.to_AtomicDataDict(batch_dev)
        del batch_dev
        if with_lengths: batch_dict = with_edge_vectors(batch_dict, with_lengths=True)
        return batch_dict, batch_info

    def _resolve_loss_module(self, loss_obj):
        curr = loss_obj
        visited = set()
        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            found_inner = None
            for attr in ("lossfunc", "loss_fn", "criterion", "method", "loss"):
                inner = getattr(curr, attr, None)
                if inner is None: continue
                if isinstance(inner, nn.Module):
                    found_inner = inner
                    break
            if found_inner is None: break
            curr = found_inner
        return curr

    def _as_scalar_tensor(self, value, default=0.0, allow_none=False):
        if value is None: return None if allow_none else torch.zeros((), dtype=self.dtype, device=self.device) + float(
            default)
        if torch.is_tensor(value):
            out = value.detach()
            return out.mean().to(self.device) if out.ndim != 0 else out.to(self.device)
        return torch.tensor(float(value), dtype=self.dtype, device=self.device)

    def _to_float_scalar(self, value, default=0.0):
        if value is None: return float(default)
        if torch.is_tensor(value):
            v = value.detach()
            return float(v.mean().item()) if v.ndim != 0 else float(v.item())
        return float(value)

    def _to_int_scalar(self, value, default=0):
        if value is None: return int(default)
        if torch.is_tensor(value):
            v = value.detach()
            return int(v.mean().item()) if v.ndim != 0 else int(v.item())
        return int(value)

    def _snapshot_loss_metrics(self, loss_obj) -> Dict[str, Any]:
        loss_module = self._resolve_loss_module(loss_obj)
        out = {
            "onsite": self._as_scalar_tensor(getattr(loss_module, "last_onsite_loss", 0.0), default=0.0),
            "hopping": self._as_scalar_tensor(getattr(loss_module, "last_hopping_loss", 0.0), default=0.0),
            "z_loss": self._as_scalar_tensor(getattr(loss_module, "last_z_loss", None), allow_none=True),
            "expert_load_cv": self._as_scalar_tensor(getattr(loss_module, "expert_load_cv", None), allow_none=True),
        }
        for k in ("last_onsite_l1_sum", "last_onsite_mse_sum", "last_onsite_count", "last_hopping_l1_sum",
                  "last_hopping_mse_sum", "last_hopping_count"):
            v = getattr(loss_module, k, None)
            out[k] = self._as_scalar_tensor(v, default=0.0) if v is not None else None
        return out

    def _run_one_expert_loss_prepared(self, batch_dict, batch_info, criterion, expert_idx, expert_edge_mask,
                                      expert_node_mask, capture_metrics=False):
        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask
        batch_copy["expert_idx"] = int(expert_idx)

        pred_batch = self.model(batch_copy)
        pred_batch["global_step"] = int(self.iter)
        pred_batch.update(batch_info)

        batch_for_loss = batch_copy.copy()
        batch_for_loss.update(batch_info)

        loss = criterion(pred_batch, batch_for_loss)
        out = {
            "loss": loss,
            "active_nodes": expert_node_mask.sum().detach(),
            "active_edges": expert_edge_mask.sum().detach(),
        }
        if capture_metrics: out.update(self._snapshot_loss_metrics(criterion))

        del pred_batch, batch_for_loss, batch_copy
        return out

    def _make_empty_payload(self):
        return {
            "loss_value": 0.0, "grad_norm": 0.0, "expert_onsite": 0.0, "expert_hopping": 0.0,
            "onsite_weighted_sum": 0.0, "hopping_weighted_sum": 0.0, "active_nodes": 0, "active_edges": 0,
            "onsite_l1_sum": None, "onsite_mse_sum": None, "onsite_cnt": None,
            "hopping_l1_sum": None, "hopping_mse_sum": None, "hopping_cnt": None,
            "z_values": [], "load_cv_values": [],
        }

    def _merge_result_into_payload(self, payload: Dict[str, Any], res: Dict[str, Any]):
        payload["loss_value"] += self._to_float_scalar(res["loss"])
        active_nodes, active_edges = self._to_int_scalar(res["active_nodes"]), self._to_int_scalar(res["active_edges"])
        onsite_val, hopping_val = self._to_float_scalar(res.get("onsite", 0.0)), self._to_float_scalar(
            res.get("hopping", 0.0))

        payload["active_nodes"] += active_nodes
        payload["active_edges"] += active_edges
        payload["onsite_weighted_sum"] += onsite_val * active_nodes
        payload["hopping_weighted_sum"] += hopping_val * active_edges

        for k in ["onsite_l1_sum", "onsite_mse_sum", "hopping_l1_sum", "hopping_mse_sum"]:
            payload[k] = self._to_float_scalar(res.get(f"last_{k}", None)) if res.get(f"last_{k}",
                                                                                      None) is not None else payload[k]
        for k in ["onsite_cnt", "hopping_cnt"]:
            payload[k] = self._to_int_scalar(res.get(f"last_{k.split('_')[0]}_count", None)) if res.get(
                f"last_{k.split('_')[0]}_count", None) is not None else payload[k]

        if res.get("z_loss", None) is not None: payload["z_values"].append(self._to_float_scalar(res["z_loss"]))
        if res.get("expert_load_cv", None) is not None: payload["load_cv_values"].append(
            self._to_float_scalar(res["expert_load_cv"]))

    def _finalize_payload(self, payload: Dict[str, Any]):
        payload["expert_onsite"] = payload["onsite_weighted_sum"] / max(payload["active_nodes"], 1)
        payload["expert_hopping"] = payload["hopping_weighted_sum"] / max(payload["active_edges"], 1)
        return payload

    def _train_one_expert_serial_low_mem(self, batch_dict, batch_info, expert_idx, range_dis, ref_batch_dict=None,
                                         ref_batch_info=None):
        payload = self._make_empty_payload()
        opt = self.optimizers[expert_idx]

        with self._tagger.tag("expert/precompute_masks(main)", it=self.iter, expert=expert_idx):
            main_edge_mask, main_node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)
            main_has_work = self._expert_has_work(main_edge_mask, main_node_mask)

        ref_has_work, ref_edge_mask, ref_node_mask = False, None, None
        if ref_batch_dict is not None:
            with self._tagger.tag("expert/precompute_masks(ref)", it=self.iter, expert=expert_idx):
                ref_edge_mask, ref_node_mask = self._prepare_expert_masks(ref_batch_dict, range_dis, expert_idx)
                ref_has_work = self._expert_has_work(ref_edge_mask, ref_node_mask)

        if not main_has_work and not ref_has_work:
            return self._finalize_payload(payload)

        self._activate_expert_for_train(expert_idx)

        try:
            with self._tagger.tag("expert/zero_grad(begin)", it=self.iter, expert=expert_idx):
                opt.zero_grad(set_to_none=True)

            if main_has_work:
                res = self._run_one_expert_loss_prepared(batch_dict, batch_info, self.train_lossfunc, expert_idx,
                                                         main_edge_mask, main_node_mask, capture_metrics=True)
                loss = res["loss"]
                self._merge_result_into_payload(payload, res)

                with self._tagger.tag("expert/backward(main)", it=self.iter, expert=expert_idx):
                    loss.backward()
                del loss, res

            if ref_has_work:
                ref_res = self._run_one_expert_loss_prepared(ref_batch_dict, ref_batch_info, self.train_lossfunc,
                                                             expert_idx, ref_edge_mask, ref_node_mask,
                                                             capture_metrics=True)
                ref_loss = ref_res["loss"]
                self._merge_result_into_payload(payload, ref_res)

                with self._tagger.tag("expert/backward(ref)", it=self.iter, expert=expert_idx):
                    ref_loss.backward()
                del ref_loss, ref_res

            with self._tagger.tag("expert/clip_grad_norm", it=self.iter, expert=expert_idx):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.experts[expert_idx].parameters(),
                                                           max_norm=self.clip_grad_norm)
            payload["grad_norm"] = self._to_float_scalar(grad_norm)

            with self._tagger.tag("expert/optimizer_step", it=self.iter, expert=expert_idx):
                opt.step()
        finally:
            self._cleanup_after_train_expert(expert_idx)

        return self._finalize_payload(payload)

    # ----------------保留的并行接口 (未来使用)----------------
    def _build_train_payload(self, batch_dict, batch_info, expert_idx, range_dis, ref_batch_dict=None,
                             ref_batch_info=None):
        pass  # 保留空接口，你的老代码结构都在这里，如果重构并行可以沿用

    def _launch_train_payloads_parallel(self, batch_dict, batch_info, ref_batch_dict=None, ref_batch_info=None):
        payloads = [None] * self.num_experts
        device = self._device_obj()
        base_stream = torch.cuda.current_stream(device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(self.num_experts)]

        for s in streams: s.wait_stream(base_stream)

        for expert_idx, range_dis in enumerate(self.distance_ranges):
            with torch.cuda.stream(streams[expert_idx]):
                # 如果未来开启 parallel_multi=True，可以直接调用新的 _train_one_expert_serial_low_mem
                # 因为它内部封装好了 fwd->bwd->清理 逻辑，完美兼容 streams
                payloads[expert_idx] = self._train_one_expert_serial_low_mem(
                    batch_dict, batch_info, expert_idx, range_dis, ref_batch_dict, ref_batch_info
                )

        current = torch.cuda.current_stream(device=device)
        for s in streams: current.wait_stream(s)
        return payloads

    # --------------------------------------------------------

    def _compute_stitched_loss_by_reduce(self, payloads: List[Dict[str, Any]]) -> Optional[float]:
        if (not self.log_single_model_compatible_loss) or (
                self.log_single_model_compatible_loss_mode != "reduce"): return None
        onsite_l1_sum, onsite_mse_sum, onsite_cnt = None, None, None
        hopping_l1_sum, hopping_mse_sum, hopping_cnt = None, None, None
        z_vals = []

        for p in payloads:
            if p is None: continue
            if p.get("onsite_l1_sum") is not None:
                v = self._to_float_scalar(p["onsite_l1_sum"])
                onsite_l1_sum = v if onsite_l1_sum is None else onsite_l1_sum + v
            if p.get("onsite_mse_sum") is not None:
                v = self._to_float_scalar(p["onsite_mse_sum"])
                onsite_mse_sum = v if onsite_mse_sum is None else onsite_mse_sum + v
            if p.get("onsite_cnt") is not None:
                v = self._to_int_scalar(p["onsite_cnt"])
                onsite_cnt = v if onsite_cnt is None else onsite_cnt + v
            if p.get("hopping_l1_sum") is not None:
                v = self._to_float_scalar(p["hopping_l1_sum"])
                hopping_l1_sum = v if hopping_l1_sum is None else hopping_l1_sum + v
            if p.get("hopping_mse_sum") is not None:
                v = self._to_float_scalar(p["hopping_mse_sum"])
                hopping_mse_sum = v if hopping_mse_sum is None else hopping_mse_sum + v
            if p.get("hopping_cnt") is not None:
                v = self._to_int_scalar(p["hopping_cnt"])
                hopping_cnt = v if hopping_cnt is None else hopping_cnt + v
            for z in p.get("z_values", []):
                if z is not None: z_vals.append(self._to_float_scalar(z))

        if onsite_cnt is None and hopping_cnt is None: return None

        def _safe_mean(sum_v, cnt_v):
            return 0.0 if sum_v is None or cnt_v is None else float(sum_v) / max(float(cnt_v), 1.0)

        onsite_loss = 0.5 * (
                    _safe_mean(onsite_l1_sum, onsite_cnt) + math.sqrt(max(_safe_mean(onsite_mse_sum, onsite_cnt), 0.0)))
        hopping_loss = 0.5 * (_safe_mean(hopping_l1_sum, hopping_cnt) + math.sqrt(
            max(_safe_mean(hopping_mse_sum, hopping_cnt), 0.0)))

        loss_module = self._resolve_loss_module(self.train_lossfunc)
        if getattr(loss_module, "onsite_boost", False):
            total = float(getattr(loss_module, "_current_onsite_weight", lambda: 1.0)()) * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        if float(getattr(loss_module, "z_loss_coef", 0.0)) > 0.0 and len(z_vals) > 0: total += float(
            getattr(loss_module, "z_loss_coef", 0.0)) * z_vals[0]
        return float(total)

    def _shared_scheduler_step_after_barrier(self, metric_value):
        if not self.update_lr_per_iter: return
        metric_float = self._to_float_scalar(metric_value)
        for expert_idx, sch in enumerate(self.lr_schedulers):
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.iter > 1: sch.step(metric_float)
            else:
                sch.step()

    def iteration(self, batch, ref_batch=None):
        self.model.train()
        batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)
        ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch,
                                                                    with_lengths=True) if ref_batch is not None else (
            None, None)

        total_loss_opt_value = 0.0
        expert_grad_norms = []
        global_onsite_sum, global_hopping_sum = 0.0, 0.0
        total_active_nodes, total_active_edges = 0, 0
        expert_onsite_dict, expert_hopping_dict = {}, {}
        z_metric_values, expert_load_cv_values = [], []
        reduce_payloads: List[Dict[str, Any]] = []

        def collect_payload(expert_idx, payload):
            nonlocal total_loss_opt_value, global_onsite_sum, global_hopping_sum, total_active_nodes, total_active_edges
            total_loss_opt_value += self._to_float_scalar(payload.get("loss_value", payload.get("loss_detached", 0.0)))
            expert_grad_norms.append(self._to_float_scalar(payload["grad_norm"]))

            expert_onsite_dict[f"expert_{expert_idx}_onsite"] = self._to_float_scalar(payload["expert_onsite"])
            expert_hopping_dict[f"expert_{expert_idx}_hopping"] = self._to_float_scalar(payload["expert_hopping"])

            global_onsite_sum += self._to_float_scalar(payload["onsite_weighted_sum"])
            global_hopping_sum += self._to_float_scalar(payload["hopping_weighted_sum"])
            total_active_nodes += self._to_int_scalar(payload["active_nodes"])
            total_active_edges += self._to_int_scalar(payload["active_edges"])

            for z in payload.get("z_values", []):
                if z is not None: z_metric_values.append(self._to_float_scalar(z))
            for cv in payload.get("load_cv_values", []):
                if cv is not None: expert_load_cv_values.append(self._to_float_scalar(cv))
            reduce_payloads.append(payload)

        # ------------------ 分支逻辑 ------------------
        if self._use_cuda_stream_parallel():
            payload_list = self._launch_train_payloads_parallel(batch_dict, batch_info, ref_batch_dict, ref_batch_info)
            for expert_idx, payload in enumerate(payload_list): collect_payload(expert_idx, payload)
        else:
            with self._tagger.tag("iteration/train_experts(serial_low_mem)", it=self.iter):
                for expert_idx, range_dis in enumerate(self.distance_ranges):
                    payload = self._train_one_expert_serial_low_mem(batch_dict, batch_info, expert_idx, range_dis,
                                                                    ref_batch_dict, ref_batch_info)
                    collect_payload(expert_idx, payload)

        global_onsite = global_onsite_sum / max(total_active_nodes, 1)
        global_hopping = global_hopping_sum / max(total_active_edges, 1)

        comparable_train_loss_value = self._compute_stitched_loss_by_reduce(reduce_payloads)
        final_train_loss_value = comparable_train_loss_value if comparable_train_loss_value is not None else total_loss_opt_value

        sched_metric = final_train_loss_value if self.shared_scheduler_metric == "train_loss" else total_loss_opt_value
        self._shared_scheduler_step_after_barrier(sched_metric)

        state = {
            'field': 'iteration', "train_loss": final_train_loss_value, "train_loss_opt": total_loss_opt_value,
            "lr": self.optimizers[0].param_groups[0]['lr'],
            "total_grad_norm": sum(expert_grad_norms) / max(len(expert_grad_norms), 1),
            "train_onsite_loss": global_onsite, "train_hopping_loss": global_hopping,
        }
        for i in range(self.num_experts):
            state[f"expert_{i}_onsite"] = expert_onsite_dict.get(f"expert_{i}_onsite", 0.0)
            state[f"expert_{i}_hopping"] = expert_hopping_dict.get(f"expert_{i}_hopping", 0.0)
        if expert_load_cv_values: state["expert_load_cv"] = sum(expert_load_cv_values) / len(expert_load_cv_values)
        if z_metric_values: state["mean_max_prob"] = sum(z_metric_values) / len(z_metric_values)

        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return torch.scalar_tensor(total_loss_opt_value, dtype=self.dtype, device=self.device)

    def validation(self, fast=True):
        with torch.no_grad():
            total_loss_value = 0.0
            num_batches = 0
            self.model.eval()

            for batch in self.validation_loader:
                batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)
                payloads = []

                for expert_idx, range_dis in enumerate(self.distance_ranges):
                    payload = self._make_empty_payload()
                    edge_mask, node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)
                    if not self._expert_has_work(edge_mask, node_mask):
                        payloads.append(self._finalize_payload(payload))
                        continue

                    self._activate_expert_for_eval(expert_idx)
                    try:
                        res = self._run_one_expert_loss_prepared(batch_dict, batch_info, self.validation_lossfunc,
                                                                 expert_idx, edge_mask, node_mask, capture_metrics=True)
                        self._merge_result_into_payload(payload, res)
                        del res
                    finally:
                        self._cleanup_after_eval_expert(expert_idx)
                    payloads.append(self._finalize_payload(payload))

                old = self.train_lossfunc
                self.train_lossfunc = self.validation_lossfunc
                loss_i = self._compute_stitched_loss_by_reduce(payloads)
                self.train_lossfunc = old

                if loss_i is None: loss_i = sum(self._to_float_scalar(p["loss_value"]) for p in payloads)

                total_loss_value += float(loss_i)
                num_batches += 1
                if fast: break

        if (not fast) and num_batches > 0: total_loss_value = total_loss_value / num_batches
        return torch.scalar_tensor(total_loss_value, dtype=self.dtype, device=self.device)

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options={}, common_options={}, reference_datasets=None,
                validation_datasets=None):
        ckpt = torch.load(checkpoint, map_location=common_options["device"], weights_only=False)
        model = build_model(
            checkpoint=checkpoint, model_options=ckpt["config"]["model_options"],
            common_options=ckpt["config"]["common_options"],
            train_options=ckpt["config"].get("train_options", train_options)
        )
        if len(train_options) == 0: train_options = ckpt["config"]["train_options"]
        if len(common_options) == 0: common_options = ckpt["config"]["common_options"]

        distance_ranges = train_options.get("distance_ranges", [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]])

        trainer = cls(
            distance_ranges=distance_ranges, model=model, train_datasets=train_datasets,
            reference_datasets=reference_datasets, validation_datasets=validation_datasets,
            train_options=train_options, common_options=common_options
        )

        trainer.ep = ckpt["epoch"] + 1
        trainer.iter = ckpt["iteration"] + 1
        trainer.stats = ckpt["stats"]

        for unit in list(trainer.plugin_queues.keys()):
            for plugin in trainer.plugin_queues[unit]: plugin = (getattr(trainer, unit) + plugin[0], plugin[1],
                                                                 plugin[2])

        for key in cls.object_keys:
            items = getattr(trainer, key, None)
            if items is not None:
                saved_states = ckpt[key + "_state_dict"]
                for obj, state in zip(items, saved_states): obj.load_state_dict(state)

        if trainer.serial_offload_experts and trainer._is_cuda_device():
            trainer._offload_all_experts_to_cpu(move_optimizer_state=True)

        return trainer