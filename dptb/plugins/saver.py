import gc
import json
import shutil
from dptb.plugins.base_plugin import Plugin
import logging
import os
import torch
import torch.distributed as dist
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

# codex checkpoint pressure controls, 2026-05-25.
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_ENV_VALUES


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _delta(end, start):
    if end is None or start is None:
        return None
    return end - start


class Saver(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Saver, self).__init__(interval)
        self.best_loss = 1e7
        self.best_quene = []
        self.latest_quene = []
        self._last_iteration_checkpoint_name = None
        self._last_iteration_checkpoint_iter = None
        self._profile_save_name = None

    def register(self, trainer, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.trainer = trainer

        if self.trainer.model.name == "nnsk":
            push_option = self.trainer.model.model_options["nnsk"].get("push", False)
            if push_option:
                if abs(push_option['rs_thr']) + abs(push_option['w_thr']) != 0.0 and abs(push_option['ovp_thr']) != 0.0:
                    log.error("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")
                    raise ValueError("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")

                if abs(push_option['rs_thr']) + abs(push_option['w_thr']) != 0.0:
                    push = 'rs_w'
                elif abs(push_option['ovp_thr']) != 0.0:
                    push = 'overlap'
                else:
                    push = False
            else:
                push = False
        else:
            push = False
        self.push = push

    def _safe_link_or_copy(self, src_abs, dst):
        try:
            os.symlink(src_abs, dst)
            return
        except Exception as e:
            log.warning(f"Failed to create symlink {dst} -> {src_abs}, fallback to copy. Reason: {e}")
        shutil.copy2(src_abs, dst)

    def _is_dist_expert(self):
        return bool(getattr(self.trainer, "distributed_expert", False)) and dist.is_available() and dist.is_initialized()

    def _is_main(self):
        return bool(getattr(self.trainer, "is_main_process", True))

    def _rank(self):
        return int(getattr(self.trainer, "rank", 0))

    def _barrier_on_current_device(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        if torch.cuda.is_available():
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()])
                return
            except TypeError:
                pass
        dist.barrier()

    def _trainer_cuda_device(self):
        if not torch.cuda.is_available():
            return None
        try:
            raw_device = getattr(self.trainer, "device", None)
            if raw_device is None:
                raw_device = torch.cuda.current_device()
            device = torch.device(raw_device)
        except Exception:
            device = torch.device(torch.cuda.current_device())
        if device.type != "cuda":
            return None
        return device

    def _profile_enabled(self):
        return _env_flag("DPTB_SAVE_PROFILE", False) or _env_flag("DPTB_SAVER_PROFILE", False)

    def _read_proc_status_kb(self):
        out = {}
        try:
            with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                        parts = line.split()
                        if len(parts) >= 2:
                            out[parts[0].rstrip(":")] = int(parts[1])
        except Exception:
            pass
        return out

    def _read_proc_io_bytes(self):
        out = {}
        try:
            with open("/proc/self/io", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    key, value = line.split(":", 1)
                    if key in {"read_bytes", "write_bytes", "cancelled_write_bytes"}:
                        out[key] = int(value.strip())
        except Exception:
            pass
        return out

    def _read_mem_available_kb(self):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return None

    def _profile_snapshot(self):
        status = self._read_proc_status_kb()
        io_bytes = self._read_proc_io_bytes()
        device = self._trainer_cuda_device()
        cuda_allocated_mb, cuda_reserved_mb = (None, None)
        if device is not None:
            cuda_allocated_mb, cuda_reserved_mb = self._cuda_memory_mb(device)
        return {
            "time": time.time(),
            "rank": self._rank(),
            "save_name": self._profile_save_name,
            "vmrss_kb": status.get("VmRSS"),
            "vmhwm_kb": status.get("VmHWM"),
            "vmsize_kb": status.get("VmSize"),
            "mem_available_kb": self._read_mem_available_kb(),
            "proc_read_bytes": io_bytes.get("read_bytes"),
            "proc_write_bytes": io_bytes.get("write_bytes"),
            "cuda_allocated_mb": cuda_allocated_mb,
            "cuda_reserved_mb": cuda_reserved_mb,
        }

    def _profile_write(self, row):
        try:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            path = os.path.join(self.checkpoint_path, f"save_profile_rank{self._rank()}.jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as e:
            log.warning("[SaverProfile][rank=%s] failed to write profile row: %s", self._rank(), e)

    def _profile_record(self, event, **extra):
        if not self._profile_enabled():
            return
        row = self._profile_snapshot()
        row["event"] = event
        row.update(extra)
        self._profile_write(row)

    @contextmanager
    def _profile_stage(self, stage, **extra):
        if not self._profile_enabled():
            yield
            return
        start = self._profile_snapshot()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            end = self._profile_snapshot()
            row = dict(end)
            row["event"] = "stage"
            row["stage"] = stage
            row["elapsed_sec"] = time.perf_counter() - t0
            row["vmrss_delta_kb"] = _delta(end.get("vmrss_kb"), start.get("vmrss_kb"))
            row["vmhwm_delta_kb"] = _delta(end.get("vmhwm_kb"), start.get("vmhwm_kb"))
            row["proc_write_bytes_delta"] = _delta(end.get("proc_write_bytes"), start.get("proc_write_bytes"))
            row["proc_read_bytes_delta"] = _delta(end.get("proc_read_bytes"), start.get("proc_read_bytes"))
            row["cuda_reserved_delta_mb"] = _delta(end.get("cuda_reserved_mb"), start.get("cuda_reserved_mb"))
            row["cuda_allocated_delta_mb"] = _delta(end.get("cuda_allocated_mb"), start.get("cuda_allocated_mb"))
            row.update(extra)
            self._profile_write(row)

    @staticmethod
    def _cuda_memory_mb(device):
        mb = 1024 ** 2
        try:
            return (
                torch.cuda.memory_allocated(device) / mb,
                torch.cuda.memory_reserved(device) / mb,
            )
        except Exception:
            return None, None

    def _clear_cuda_cache_after_iteration_save(self, name):
        if not _env_flag("DPTB_SAVER_CLEAR_CUDA_CACHE_AFTER_ITER_SAVE", False):
            return
        device = self._trainer_cuda_device()
        if device is None:
            return

        self._barrier_on_current_device()
        allocated_before, reserved_before = self._cuda_memory_mb(device)

        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            log.warning(
                "[CUDA Cache Clear][rank=%s] after save_freq checkpoint %s failed: %s",
                getattr(self.trainer, "rank", 0),
                name,
                e,
            )
            self._barrier_on_current_device()
            return
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

        allocated_after, reserved_after = self._cuda_memory_mb(device)
        log.info(
            "[CUDA Cache Clear][rank=%s] after save_freq checkpoint %s: "
            "allocated %.1f -> %.1f MB, reserved %.1f -> %.1f MB",
            getattr(self.trainer, "rank", 0),
            name,
            allocated_before if allocated_before is not None else float("nan"),
            allocated_after if allocated_after is not None else float("nan"),
            reserved_before if reserved_before is not None else float("nan"),
            reserved_after if reserved_after is not None else float("nan"),
        )
        self._barrier_on_current_device()

    def _post_save_cooldown(self, name):
        sleep_sec = max(0.0, _env_float("DPTB_SAVER_POST_SAVE_SLEEP_SEC", 0.0))
        if sleep_sec <= 0.0:
            return

        device = self._trainer_cuda_device()
        if device is not None:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass

        with self._profile_stage("post_save_cooldown_pre_barrier"):
            self._barrier_on_current_device()

        log.info(
            "[SaverCooldown][rank=%s] checkpoint %s sleep %.1fs before training resumes",
            self._rank(),
            name,
            sleep_sec,
        )
        self._profile_record("post_save_sleep_start", sleep_sec=sleep_sec)
        time.sleep(sleep_sec)
        self._profile_record("post_save_sleep_end", sleep_sec=sleep_sec)

        if device is not None:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        self._barrier_on_current_device()

    def _to_cpu_obj(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_cpu_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cpu_obj(v) for v in obj)
        return obj

    def _tensor_bytes(self, obj):
        if torch.is_tensor(obj):
            return int(obj.numel() * obj.element_size())
        if isinstance(obj, dict):
            return sum(self._tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._tensor_bytes(v) for v in obj)
        return 0

    def _is_canonical_expert_dp_rank(self):
        if int(getattr(self.trainer, "expert_data_parallel_size", 1)) <= 1:
            return True
        return int(getattr(self.trainer, "expert_dp_rank", 0)) == 0

    def _state_fingerprint(self, obj):
        stats = torch.zeros(6, dtype=torch.float64)

        def visit(value):
            if torch.is_tensor(value):
                data = value.detach()
                if data.numel() == 0:
                    stats[0] += 1.0
                    return
                flat = data.to(dtype=torch.float64).reshape(-1)
                stats[0] += 1.0
                stats[1] += float(flat.numel())
                stats[2] += flat.sum().cpu()
                stats[3] += flat.abs().sum().cpu()
                stats[4] += (flat * flat).sum().cpu()
                stats[5] = torch.maximum(stats[5], flat.abs().max().cpu())
                return
            if isinstance(value, dict):
                for key in sorted(value.keys(), key=str):
                    visit(value[key])
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return
            if isinstance(value, (int, float, bool)):
                stats[0] += 1.0
                stats[1] += 1.0
                val = float(value)
                stats[2] += val
                stats[3] += abs(val)
                stats[4] += val * val
                stats[5] = torch.maximum(stats[5], torch.tensor(abs(val), dtype=torch.float64))

        visit(obj)
        return stats

    def _validate_expert_dp_replicas(self, expert_obj, opt_obj):
        if (
            int(getattr(self.trainer, "expert_data_parallel_size", 1)) <= 1
            or not _env_flag("DPTB_SAVER_VALIDATE_EXPERT_DP_REPLICAS", False)
        ):
            return
        group = getattr(self.trainer, "expert_dp_process_group", None)
        if group is None or not (dist.is_available() and dist.is_initialized()):
            return

        with self._profile_stage("expert_dp_replica_fingerprint"):
            fp = self._state_fingerprint({"model": expert_obj, "optimizer": opt_obj})
            if torch.cuda.is_available():
                fp = fp.to(torch.cuda.current_device())
            gathered = [torch.zeros_like(fp) for _ in range(int(getattr(self.trainer, "expert_data_parallel_size", 1)))]
            dist.all_gather(gathered, fp, group=group)

        base = gathered[0].detach().cpu()
        atol = _env_float("DPTB_SAVER_EXPERT_DP_CHECK_ATOL", 1e-6)
        rtol = _env_float("DPTB_SAVER_EXPERT_DP_CHECK_RTOL", 1e-5)
        for dp_rank, other in enumerate(gathered[1:], start=1):
            other_cpu = other.detach().cpu()
            if not torch.allclose(base, other_cpu, atol=atol, rtol=rtol):
                msg = (
                    "expert-DP replica checkpoint fingerprints differ for expert "
                    f"{getattr(self.trainer, 'local_expert_idx', None)} between dp_rank=0 and "
                    f"dp_rank={dp_rank}: base={base.tolist()} other={other_cpu.tolist()}"
                )
                if _env_flag("DPTB_SAVER_STRICT_EXPERT_DP_REPLICA_CHECK", True):
                    raise RuntimeError(msg)
                log.warning(msg)

    def _gather_dist_states(self):
        local_idx = self.trainer.local_expert_idx
        dp_size = int(getattr(self.trainer, "expert_data_parallel_size", 1))
        num_experts = int(getattr(self.trainer, "num_experts", max(1, self.trainer.world_size // max(dp_size, 1))))
        canonical_only = _env_flag("DPTB_SAVER_CANONICAL_EXPERT_DP_ONLY", True) and dp_size > 1
        is_canonical = self._is_canonical_expert_dp_rank()

        local_opt = self.trainer.optimizers[local_idx]
        local_sch = self.trainer.lr_schedulers[local_idx]

        local_expert_state = None
        local_opt_state = None
        local_sch_state = None
        local_expert_tensor_bytes = 0
        local_opt_tensor_bytes = 0

        if (not canonical_only) or is_canonical:
            with self._profile_stage("local_expert_state_dict", expert_idx=local_idx, canonical=is_canonical):
                raw_expert_state = self.trainer._unwrap_expert_module(self.trainer.model.experts[local_idx]).state_dict()
            with self._profile_stage("local_optimizer_state_dict", expert_idx=local_idx, canonical=is_canonical):
                raw_opt_state = local_opt.state_dict() if local_opt is not None else None
                raw_sch_state = local_sch.state_dict() if local_sch is not None else None

            self._validate_expert_dp_replicas(raw_expert_state, raw_opt_state)

            with self._profile_stage("local_expert_state_to_cpu", expert_idx=local_idx, canonical=is_canonical):
                local_expert_state = self._to_cpu_obj(raw_expert_state)
            with self._profile_stage("local_optimizer_state_to_cpu", expert_idx=local_idx, canonical=is_canonical):
                local_opt_state = self._to_cpu_obj(raw_opt_state) if raw_opt_state is not None else None
                local_sch_state = self._to_cpu_obj(raw_sch_state) if raw_sch_state is not None else None
            local_expert_tensor_bytes = self._tensor_bytes(local_expert_state)
            local_opt_tensor_bytes = self._tensor_bytes(local_opt_state)
        elif _env_flag("DPTB_SAVER_VALIDATE_EXPERT_DP_REPLICAS", False):
            with self._profile_stage("local_state_dict_for_fingerprint", expert_idx=local_idx, canonical=is_canonical):
                raw_expert_state = self.trainer._unwrap_expert_module(self.trainer.model.experts[local_idx]).state_dict()
                raw_opt_state = local_opt.state_dict() if local_opt is not None else None
            self._validate_expert_dp_replicas(raw_expert_state, raw_opt_state)

        self._profile_record(
            "local_state_prepared",
            expert_idx=local_idx,
            expert_dp_rank=int(getattr(self.trainer, "expert_dp_rank", 0)),
            canonical=is_canonical,
            canonical_only=canonical_only,
            local_expert_tensor_bytes=local_expert_tensor_bytes,
            local_optimizer_tensor_bytes=local_opt_tensor_bytes,
        )

        if _env_flag("DPTB_SAVER_GATHER_TO_MAIN_ONLY", True):
            expert_states = [None for _ in range(self.trainer.world_size)] if self._is_main() else None
            opt_states = [None for _ in range(self.trainer.world_size)] if self._is_main() else None
            sch_states = [None for _ in range(self.trainer.world_size)] if self._is_main() else None

            with self._profile_stage("gather_expert_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.gather_object(local_expert_state, object_gather_list=expert_states, dst=0)
            with self._profile_stage("gather_optimizer_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.gather_object(local_opt_state, object_gather_list=opt_states, dst=0)
            with self._profile_stage("gather_scheduler_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.gather_object(local_sch_state, object_gather_list=sch_states, dst=0)
            if not self._is_main():
                return None, None, None
        else:
            expert_states = [None for _ in range(self.trainer.world_size)]
            opt_states = [None for _ in range(self.trainer.world_size)]
            sch_states = [None for _ in range(self.trainer.world_size)]

            with self._profile_stage("all_gather_expert_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.all_gather_object(expert_states, local_expert_state)
            with self._profile_stage("all_gather_optimizer_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.all_gather_object(opt_states, local_opt_state)
            with self._profile_stage("all_gather_scheduler_states", canonical=is_canonical, canonical_only=canonical_only):
                dist.all_gather_object(sch_states, local_sch_state)

        if dp_size <= 1:
            return expert_states, opt_states, sch_states

        expert_states_by_expert = [None for _ in range(num_experts)]
        opt_states_by_expert = [None for _ in range(num_experts)]
        sch_states_by_expert = [None for _ in range(num_experts)]
        for expert_idx in range(num_experts):
            src_rank = expert_idx * dp_size
            expert_states_by_expert[expert_idx] = expert_states[src_rank]
            opt_states_by_expert[expert_idx] = opt_states[src_rank]
            sch_states_by_expert[expert_idx] = sch_states[src_rank]
            if expert_states_by_expert[expert_idx] is None:
                raise RuntimeError(
                    "missing canonical expert checkpoint state for expert "
                    f"{expert_idx} from rank {src_rank}; check expert_data_parallel_size/world_size layout"
                )

        self._profile_record(
            "main_gather_summary",
            num_experts=num_experts,
            expert_data_parallel_size=dp_size,
            gathered_non_null_experts=sum(x is not None for x in expert_states),
            gathered_expert_tensor_bytes=sum(self._tensor_bytes(x) for x in expert_states if x is not None),
            gathered_optimizer_tensor_bytes=sum(self._tensor_bytes(x) for x in opt_states if x is not None),
        )

        return expert_states_by_expert, opt_states_by_expert, sch_states_by_expert

    def _assemble_full_model_state(self, expert_states):
        with self._profile_stage("base_model_state_dict"):
            raw_base_state = self.trainer.model.state_dict()
        with self._profile_stage("base_model_state_to_cpu"):
            base_state = self._to_cpu_obj(raw_base_state)
        full_state = {}

        for k, v in base_state.items():
            if not k.startswith("experts."):
                full_state[k] = v

        for i, expert_state in enumerate(expert_states):
            for k, v in expert_state.items():
                full_state[f"experts.{i}.{k}"] = v

        return full_state

    def iteration(self, **kwargs):
        if self.push == 'rs_w':
            suffix = ".iter_rs" + "%.3f" % self.trainer.model.hopping_options["rs"] + "_w" + "%.3f" % \
                     self.trainer.model.hopping_options["w"]
            max_ckpt = self.trainer.train_options["max_ckpt"]
        elif self.push == 'overlap':
            suffix = ".iter_ovp" + "%.3f" % self.trainer.model.ovp_factor
            max_ckpt = self.trainer.train_options["max_ckpt"]
        else:
            suffix = ".iter{}".format(self.trainer.iter)
            max_ckpt = self.trainer.train_options["max_ckpt"]

        name = self.trainer.model.name + suffix
        self.latest_quene.append(name)

        delete_name = None
        if len(self.latest_quene) > max_ckpt:
            delete_name = self.latest_quene.pop(0)

        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
        )

        if self._is_main():
            if delete_name is not None:
                delete_path = os.path.join(self.checkpoint_path, delete_name + ".pth")
                try:
                    os.remove(delete_path)
                except Exception:
                    log.info(f"Failed to delete the checkpoint file {delete_path}.")

            if not self.push:
                latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
                if os.path.lexists(latest_symlink):
                    os.unlink(latest_symlink)
                latest_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                latest_ckpt_abs_path = os.path.abspath(latest_ckpt)
                if not os.path.exists(latest_ckpt_abs_path):
                    raise FileNotFoundError(f"Source file {latest_ckpt_abs_path} does not exist.")
                self._safe_link_or_copy(latest_ckpt_abs_path, latest_symlink)

        self._last_iteration_checkpoint_name = name
        self._last_iteration_checkpoint_iter = self.trainer.iter

        self._clear_cuda_cache_after_iteration_save(name)

    def epoch(self, **kwargs):
        updated_loss = self.trainer.stats.get('validation_loss')
        if updated_loss is not None:
            updated_loss = updated_loss.get('epoch_mean', 1e6)
        else:
            updated_loss = self.trainer.stats.get("train_loss").get("epoch_mean", 1e6)

        max_ckpt = self.trainer.train_options["max_ckpt"]

        if updated_loss < self.best_loss:
            suffix = ".ep{}".format(self.trainer.ep)
            name = self.trainer.model.name + suffix
            self.best_quene.append(name)

            delete_name = None
            if len(self.best_quene) > max_ckpt:
                delete_name = self.best_quene.pop(0)

            reused_iter_checkpoint = False
            if (
                _env_flag("DPTB_SAVER_REUSE_ITER_CKPT_FOR_EPOCH_BEST", True)
                and self._last_iteration_checkpoint_name is not None
                and self._last_iteration_checkpoint_iter == self.trainer.iter
            ):
                iter_ckpt = os.path.join(self.checkpoint_path, self._last_iteration_checkpoint_name + ".pth")
                epoch_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                reuse_decision = [False]
                if self._is_main() and os.path.exists(iter_ckpt):
                    if os.path.lexists(epoch_ckpt):
                        os.unlink(epoch_ckpt)
                    self._safe_link_or_copy(os.path.abspath(iter_ckpt), epoch_ckpt)
                    log.info(
                        "checkpoint %s reused from same-iteration checkpoint %s",
                        name,
                        self._last_iteration_checkpoint_name,
                    )
                    reuse_decision[0] = True
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast_object_list(reuse_decision, src=0)
                reused_iter_checkpoint = bool(reuse_decision[0])
                self._barrier_on_current_device()

            if not reused_iter_checkpoint:
                self._save(
                    name=name,
                    model=self.trainer.model,
                    model_options=self.trainer.model.model_options,
                    common_options=self.trainer.common_options,
                    train_options=self.trainer.train_options,
                )

            self.best_loss = updated_loss

            if self._is_main():
                if delete_name is not None:
                    delete_path = os.path.join(self.checkpoint_path, delete_name + ".pth")
                    if os.path.exists(delete_path):
                        os.remove(delete_path)

                best_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".best.pth")
                if os.path.lexists(best_symlink):
                    os.unlink(best_symlink)
                best_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                best_ckpt_abs_path = os.path.abspath(best_ckpt)
                if not os.path.exists(best_ckpt_abs_path):
                    raise FileNotFoundError(f"Source file {best_ckpt_abs_path} does not exist.")
                self._safe_link_or_copy(best_ckpt_abs_path, best_symlink)

                if _env_flag("DPTB_SAVER_EPOCH_UPDATES_LATEST", True):
                    latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
                    if os.path.lexists(latest_symlink):
                        os.unlink(latest_symlink)
                    self._safe_link_or_copy(best_ckpt_abs_path, latest_symlink)

    def _save(self, name, model, model_options, common_options, train_options):
        previous_profile_save_name = self._profile_save_name
        self._profile_save_name = name
        self._profile_record("save_start")
        obj = {}
        obj.update({"config": {"model_options": model_options, "common_options": common_options,
                               "train_options": train_options}})

        if self._is_dist_expert():
            with self._profile_stage("gather_dist_states"):
                expert_states, opt_states, sch_states = self._gather_dist_states()

            if self._is_main():
                with self._profile_stage("assemble_full_model_state"):
                    full_model_state = self._assemble_full_model_state(expert_states)

                obj.update({
                    "model_state_dict": full_model_state,
                    "task": self.trainer.task,
                    "epoch": self.trainer.ep,
                    "iteration": self.trainer.iter,
                    "stats": self.trainer.stats,
                    "optimizers_state_dict": opt_states,
                    "lr_schedulers_state_dict": sch_states,
                })

                f_path = os.path.join(self.checkpoint_path, name + ".pth")
                self._profile_record(
                    "save_object_summary",
                    model_tensor_bytes=self._tensor_bytes(full_model_state),
                    optimizer_tensor_bytes=self._tensor_bytes(opt_states),
                    scheduler_tensor_bytes=self._tensor_bytes(sch_states),
                )
                with self._profile_stage("torch_save"):
                    torch.save(obj, f=f_path)
                self._profile_record(
                    "save_file_summary",
                    checkpoint_file_bytes=os.path.getsize(f_path) if os.path.exists(f_path) else None,
                )
                log.info(msg="checkpoint saved as {}".format(name))

            with self._profile_stage("post_save_barrier"):
                dist.barrier()
            self._profile_record("save_end")
            self._post_save_cooldown(name)
            self._profile_save_name = previous_profile_save_name
            return

        # 单卡 / 非分布式
        if hasattr(self.trainer, "optimizers") and isinstance(self.trainer.optimizers, list):
            with self._profile_stage("local_optimizer_state_dict"):
                optim_state = {"optimizers_state_dict": [opt.state_dict() for opt in self.trainer.optimizers]}
                sched_state = {"lr_schedulers_state_dict": [sch.state_dict() for sch in self.trainer.lr_schedulers]}
        else:
            with self._profile_stage("local_optimizer_state_dict"):
                optim_state = {"optimizer_state_dict": self.trainer.optimizer.state_dict()}
                sched_state = {"lr_scheduler_state_dict": self.trainer.lr_scheduler.state_dict()}

        with self._profile_stage("local_model_state_dict"):
            model_state = model.state_dict()

        obj.update({
            "model_state_dict": model_state,
            "task": self.trainer.task,
            "epoch": self.trainer.ep,
            "iteration": self.trainer.iter,
            "stats": self.trainer.stats
        })

        obj.update(optim_state)
        obj.update(sched_state)

        f_path = os.path.join(self.checkpoint_path, name + ".pth")
        self._profile_record(
            "save_object_summary",
            model_tensor_bytes=self._tensor_bytes(model_state),
            optimizer_tensor_bytes=self._tensor_bytes(optim_state),
            scheduler_tensor_bytes=self._tensor_bytes(sched_state),
        )
        with self._profile_stage("torch_save"):
            torch.save(obj, f=f_path)
        self._profile_record(
            "save_file_summary",
            checkpoint_file_bytes=os.path.getsize(f_path) if os.path.exists(f_path) else None,
        )
        log.info(msg="checkpoint saved as {}".format(name))
        self._profile_record("save_end")
        self._post_save_cooldown(name)
        self._profile_save_name = previous_profile_save_name
