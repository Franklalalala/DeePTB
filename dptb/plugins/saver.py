import copy
import gc
import json
import shutil
from collections import OrderedDict
from dptb.plugins.base_plugin import Plugin
from dptb.plugins.checkpoint_state import StatefulPlugin
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    CHECKPOINT_SCHEMA_VERSION,
    TrainingState,
    capture_rng_state,
)
import logging
import os
import torch
import torch.distributed as dist
import time
import traceback
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


class Saver(Plugin, StatefulPlugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Saver, self).__init__(interval)
        self.best_loss = 1e7
        self.best_quene = []
        self.latest_quene = []
        self.epoch_quene = []
        self._last_iteration_checkpoint_name = None
        self._last_iteration_checkpoint_iter = None
        self._latest_checkpoint_name = None
        self._best_checkpoint_name = None
        self._latest_resumable_checkpoint_name = None
        self._pending_manifest_entries = {}
        self._profile_save_name = None
        # Set True once load_state_dict restores persisted best/retention state
        # (BUG 2): prevents register() from re-seeding best_loss from disk and
        # clobbering the restored value.
        self._state_restored = False

    # ------------------------------------------------------------------
    # StatefulPlugin protocol: persist best_loss + retention queues so a
    # resumed run does not treat its first epoch as a new best and overwrite
    # .best.pth with a worse checkpoint (BUG 2).
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "best_loss": float(self.best_loss),
            "best_quene": list(self.best_quene),
            "latest_quene": list(self.latest_quene),
            "epoch_quene": list(self.epoch_quene),
            "last_iteration_checkpoint_name": self._last_iteration_checkpoint_name,
            "last_iteration_checkpoint_iter": self._last_iteration_checkpoint_iter,
        }

    def load_state_dict(self, state):
        if not state:
            return
        if "best_loss" in state and state["best_loss"] is not None:
            self.best_loss = float(state["best_loss"])
        if isinstance(state.get("best_quene"), list):
            self.best_quene = list(state["best_quene"])
        if isinstance(state.get("latest_quene"), list):
            self.latest_quene = list(state["latest_quene"])
        if isinstance(state.get("epoch_quene"), list):
            self.epoch_quene = list(state["epoch_quene"])
        self._last_iteration_checkpoint_name = state.get(
            "last_iteration_checkpoint_name", self._last_iteration_checkpoint_name
        )
        self._last_iteration_checkpoint_iter = state.get(
            "last_iteration_checkpoint_iter", self._last_iteration_checkpoint_iter
        )
        self._state_restored = True

    def register(self, trainer, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.trainer = trainer
        # If no persisted state was restored (legacy checkpoint or fresh run
        # that already trained), seed best_loss from the restored stats or an
        # existing best.pth so a resumed worse epoch cannot clobber .best.pth.
        if not self._state_restored:
            self._seed_best_loss_from_disk_or_stats()

        self._configure_push()

    def _seed_best_loss_from_disk_or_stats(self):
        """Best-effort recovery of best_loss when no persisted state exists.

        Prefer the loss stored inside an existing ``.best.pth`` (the true best);
        fall back to the restored ``trainer.stats`` epoch means.  A conservative
        seed here stops a resumed *worse* epoch from overwriting ``.best.pth``
        for legacy checkpoints that predate persisted Saver state (BUG 2).
        """
        # 0) Cheapest: the durable manifest records best_loss directly.
        manifest = self._read_manifest()
        if manifest is not None and manifest.get("best_loss") is not None:
            try:
                self.best_loss = float(manifest["best_loss"])
                if isinstance(manifest.get("best_quene"), list):
                    self.best_quene = list(manifest["best_quene"])
                if isinstance(manifest.get("latest_quene"), list):
                    self.latest_quene = list(manifest["latest_quene"])
                if isinstance(manifest.get("epoch_quene"), list):
                    self.epoch_quene = list(manifest["epoch_quene"])
                def _checkpoint_name(target):
                    if isinstance(target, str) and target.endswith(".pth"):
                        return target[:-4]
                    return target

                self._latest_checkpoint_name = _checkpoint_name(
                    manifest.get("latest_target")
                )
                self._best_checkpoint_name = _checkpoint_name(
                    manifest.get("best_target")
                )
                self._latest_resumable_checkpoint_name = _checkpoint_name(
                    manifest.get("latest_resumable_target")
                )
                return
            except (TypeError, ValueError):
                pass

        model_name = getattr(getattr(self.trainer, "model", None), "name", None)
        # 1) Try the on-disk best checkpoint's recorded stats.
        if model_name is not None:
            best_path = os.path.join(self.checkpoint_path, model_name + ".best.pth")
            try:
                if os.path.lexists(best_path):
                    prev = torch.load(best_path, map_location="cpu", weights_only=False)
                    seed = self._loss_from_stats(prev.get("stats", {}))
                    if seed is not None:
                        self.best_loss = seed
                        return
            except Exception as exc:
                log.info("Could not seed best_loss from %s: %s", best_path, exc)
        # 2) Fall back to the restored in-memory stats.
        seed = self._loss_from_stats(getattr(self.trainer, "stats", {}))
        if seed is not None:
            self.best_loss = seed

    @staticmethod
    def _loss_from_stats(stats):
        if not isinstance(stats, dict):
            return None
        for key in ("validation_loss", "train_loss"):
            entry = stats.get(key)
            if isinstance(entry, dict) and entry.get("epoch_mean") is not None:
                try:
                    return float(entry["epoch_mean"])
                except (TypeError, ValueError):
                    continue
        return None

    def _configure_push(self):
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

    def _write_checkpoint_atomic(self, obj, f_path):
        """Serialize to ``<f_path>.tmp`` then ``os.replace`` onto the final name.

        An interrupted/failed ``torch.save`` therefore never leaves a truncated
        ``.pth`` behind: readers only ever observe the fully written previous
        file or the fully written new one (BUG 2, atomicity).
        """
        tmp_path = f"{f_path}.tmp{os.getpid()}"
        try:
            with self._profile_stage("torch_save"):
                torch.save(obj, f=tmp_path)
            os.replace(tmp_path, f_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _atomic_publish(self, src_abs, link_path):
        """Publish ``link_path`` -> ``src_abs`` atomically (symlink, else copy).

        Writes a uniquely named temp then ``os.replace`` swaps it into place so a
        concurrent reader never sees a missing or half-written latest/best link.
        """
        tmp_path = f"{link_path}.tmp{os.getpid()}"
        if os.path.lexists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        published = False
        try:
            os.symlink(src_abs, tmp_path)
            published = True
        except Exception as e:
            log.warning(
                "Failed to create symlink %s -> %s, fallback to copy. Reason: %s",
                tmp_path, src_abs, e,
            )
            shutil.copy2(src_abs, tmp_path)
            published = True
        if published:
            os.replace(tmp_path, link_path)

    MANIFEST_NAME = "checkpoint_manifest.json"

    def _manifest_path(self):
        return os.path.join(self.checkpoint_path, self.MANIFEST_NAME)

    def _write_manifest(self, strict=False, published_checkpoint_name=None):
        """Atomically record a durable pointer to best/latest + retention state.

        Written on the main process after each publish so an operator (or a
        restart) can find the current best/latest and the rotation queues
        without loading any ``.pth``. Diagnostic callers retain best-effort
        behavior; transactional publish callers pass ``strict=True`` so an I/O
        failure participates in the all-rank publish outcome.
        """
        if not self._is_main():
            return
        model_name = getattr(getattr(self.trainer, "model", None), "name", None)
        previous = self._read_manifest() or {}
        entries_by_name = {
            entry.get("filename"): dict(entry)
            for entry in previous.get("entries", [])
            if isinstance(entry, dict) and entry.get("filename")
        }
        entries_by_name.update(
            {
                entry["filename"]: dict(entry)
                for entry in self._pending_manifest_entries.values()
            }
        )
        published_entry = self._pending_manifest_entries.get(
            published_checkpoint_name
        )
        last_completed_step = (
            int(published_entry["last_completed_step"])
            if published_entry is not None
            else max(int(getattr(self.trainer, "iter", 1)) - 1, 0)
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "global_step": last_completed_step,
            "step_semantics": "last_completed",
            "epoch": int(getattr(self.trainer, "ep", 1)),
            "best_loss": float(self.best_loss),
            "best": (model_name + ".best.pth") if model_name else None,
            "latest": (model_name + ".latest.pth") if model_name else None,
            "latest_resumable": (
                model_name + ".latest_resumable.pth" if model_name else None
            ),
            "best_target": (
                self._best_checkpoint_name + ".pth"
                if self._best_checkpoint_name else None
            ),
            "latest_target": (
                self._latest_checkpoint_name + ".pth"
                if self._latest_checkpoint_name else None
            ),
            "latest_resumable_target": (
                self._latest_resumable_checkpoint_name + ".pth"
                if self._latest_resumable_checkpoint_name else None
            ),
            "best_quene": list(self.best_quene),
            "latest_quene": list(self.latest_quene),
            "epoch_quene": list(self.epoch_quene),
            "entries": sorted(
                entries_by_name.values(),
                key=lambda entry: (
                    int(entry.get("last_completed_step", -1)),
                    str(entry.get("filename", "")),
                ),
            ),
        }
        path = self._manifest_path()
        tmp_path = f"{path}.tmp{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception as exc:
            log.warning("Failed to write checkpoint manifest %s: %s", path, exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if strict:
                raise

    def _read_manifest(self):
        try:
            path = self._manifest_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            log.info("Could not read checkpoint manifest: %s", exc)
        return None

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

    def _to_cpu_obj(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            keep_meta = getattr(obj, "_metadata", None)
            if isinstance(obj, OrderedDict) or keep_meta is not None:
                out = OrderedDict((k, self._to_cpu_obj(v)) for k, v in obj.items())
                if keep_meta is not None:
                    out._metadata = copy.deepcopy(keep_meta)
                return out
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

    def _prepare_local_dist_states(self):
        """Prepare this rank's checkpoint payload without world collectives.

        Failures here are exchanged as small status objects before any large
        object gather, preventing one rank from entering the gather while a
        peer has already failed during ``state_dict`` or CPU materialization.
        """
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

        replica_fingerprint = None
        if (
            _env_flag("DPTB_SAVER_VALIDATE_EXPERT_DP_REPLICAS", False)
            and int(getattr(self.trainer, "expert_data_parallel_size", 1)) > 1
        ):
            with self._profile_stage(
                "expert_dp_replica_fingerprint_local",
                expert_idx=local_idx,
                canonical=is_canonical,
            ):
                replica_fingerprint = self._state_fingerprint(
                    {"model": raw_expert_state, "optimizer": raw_opt_state}
                )

        self._profile_record(
            "local_state_prepared",
            expert_idx=local_idx,
            expert_dp_rank=int(getattr(self.trainer, "expert_dp_rank", 0)),
            canonical=is_canonical,
            canonical_only=canonical_only,
            local_expert_tensor_bytes=local_expert_tensor_bytes,
            local_optimizer_tensor_bytes=local_opt_tensor_bytes,
        )

        return {
            "local_expert_state": local_expert_state,
            "local_opt_state": local_opt_state,
            "local_sch_state": local_sch_state,
            "local_idx": local_idx,
            "dp_size": dp_size,
            "num_experts": num_experts,
            "canonical_only": canonical_only,
            "is_canonical": is_canonical,
            "replica_fingerprint": replica_fingerprint,
        }

    def _validate_prepared_expert_dp_replicas(self, prepare_statuses):
        """Compare fingerprints carried by the agreed small status exchange."""
        dp_size = int(getattr(self.trainer, "expert_data_parallel_size", 1))
        if (
            dp_size <= 1
            or not _env_flag("DPTB_SAVER_VALIDATE_EXPERT_DP_REPLICAS", False)
        ):
            return
        atol = _env_float("DPTB_SAVER_EXPERT_DP_CHECK_ATOL", 1e-6)
        rtol = _env_float("DPTB_SAVER_EXPERT_DP_CHECK_RTOL", 1e-5)
        by_expert = {}
        for status in prepare_statuses:
            fingerprint = status.get("replica_fingerprint")
            if fingerprint is None:
                continue
            by_expert.setdefault(int(status["local_expert_idx"]), []).append(status)

        for expert_idx, replicas in sorted(by_expert.items()):
            replicas.sort(key=lambda status: int(status["expert_dp_rank"]))
            base_status = replicas[0]
            base = torch.tensor(base_status["replica_fingerprint"], dtype=torch.float64)
            for other_status in replicas[1:]:
                other = torch.tensor(
                    other_status["replica_fingerprint"], dtype=torch.float64
                )
                if torch.allclose(base, other, atol=atol, rtol=rtol):
                    continue
                msg = (
                    "expert-DP replica checkpoint fingerprints differ for expert "
                    f"{expert_idx} between dp_rank={base_status['expert_dp_rank']} and "
                    f"dp_rank={other_status['expert_dp_rank']}: "
                    f"base={base.tolist()} other={other.tolist()}"
                )
                if _env_flag("DPTB_SAVER_STRICT_EXPERT_DP_REPLICA_CHECK", True):
                    raise RuntimeError(msg)
                log.warning(msg)

    def _gather_dist_states(self, prepared):
        local_expert_state = prepared["local_expert_state"]
        local_opt_state = prepared["local_opt_state"]
        local_sch_state = prepared["local_sch_state"]
        dp_size = prepared["dp_size"]
        num_experts = prepared["num_experts"]
        canonical_only = prepared["canonical_only"]
        is_canonical = prepared["is_canonical"]

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
        full_state = OrderedDict()
        metadata = {}
        base_meta = getattr(base_state, "_metadata", None) or {}
        for meta_key, meta_val in base_meta.items():
            if meta_key == "" or not str(meta_key).startswith("experts."):
                metadata[meta_key] = copy.deepcopy(meta_val)

        for k, v in base_state.items():
            if not k.startswith("experts."):
                full_state[k] = v

        for i, expert_state in enumerate(expert_states):
            prefix = f"experts.{i}."
            expert_cpu = self._to_cpu_obj(expert_state) if expert_state is not None else {}
            expert_meta = getattr(expert_cpu, "_metadata", None) or {}
            for meta_key, meta_val in expert_meta.items():
                copied = copy.deepcopy(meta_val)
                if meta_key == "":
                    metadata[prefix[:-1]] = copied
                else:
                    metadata[prefix + meta_key] = copied
            for k, v in expert_cpu.items():
                full_state[f"{prefix}{k}"] = v

        full_state._metadata = metadata
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
            kind=CHECKPOINT_KIND_ITERATION,
        )

        def _publish_iteration():
            if not self.push:
                latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
                latest_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                latest_ckpt_abs_path = os.path.abspath(latest_ckpt)
                if not os.path.exists(latest_ckpt_abs_path):
                    raise FileNotFoundError(f"Source file {latest_ckpt_abs_path} does not exist.")
                self._atomic_publish(latest_ckpt_abs_path, latest_symlink)
                self._latest_checkpoint_name = name
            self._write_manifest(
                strict=True, published_checkpoint_name=name
            )
            self._gc_checkpoint_best_effort(delete_name)

        # Publish is an agreed phase: a main-rank failure raises on ALL ranks
        # (P0-4) instead of leaving them to wedge at the next collective.
        self._finalize_publish(_publish_iteration)

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
        configured_max_epoch = self.trainer.train_options.get("max_epoch_ckpt")
        max_epoch_ckpt = (
            max_ckpt if configured_max_epoch is None else int(configured_max_epoch)
        )
        is_new_best = updated_loss < self.best_loss
        suffix = ".ep{}".format(self.trainer.ep)
        name = self.trainer.model.name + suffix

        best_delete_name = None
        if is_new_best:
            # Commit the new best BEFORE saving so the harvested Saver state in
            # this very checkpoint persists the updated best_loss (otherwise a
            # resume would restore the stale, higher best and could treat a
            # worse later epoch as best) (BUG 2).
            self.best_loss = updated_loss
            self.best_quene.append(name)
            if len(self.best_quene) > max_ckpt:
                best_delete_name = self.best_quene.pop(0)

        # Every completed epoch gets a committed checkpoint, independently of
        # whether its metric improved the best gate.
        self.epoch_quene.append(name)
        epoch_delete_name = None
        if len(self.epoch_quene) > max_epoch_ckpt:
            epoch_delete_name = self.epoch_quene.pop(0)

        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
            kind=CHECKPOINT_KIND_EPOCH,
        )

        def _publish_epoch_committed():
            epoch_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
            epoch_ckpt_abs_path = os.path.abspath(epoch_ckpt)
            if not os.path.exists(epoch_ckpt_abs_path):
                raise FileNotFoundError(
                    f"Source file {epoch_ckpt_abs_path} does not exist."
                )

            if is_new_best:
                best_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".best.pth")
                self._atomic_publish(epoch_ckpt_abs_path, best_symlink)
                self._best_checkpoint_name = name

                if _env_flag("DPTB_SAVER_EPOCH_UPDATES_LATEST", True):
                    latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
                    self._atomic_publish(epoch_ckpt_abs_path, latest_symlink)
                    self._latest_checkpoint_name = name

            latest_resumable = os.path.join(
                self.checkpoint_path,
                self.trainer.model.name + ".latest_resumable.pth",
            )
            self._atomic_publish(epoch_ckpt_abs_path, latest_resumable)
            self._latest_resumable_checkpoint_name = name

            self._write_manifest(
                strict=True, published_checkpoint_name=name
            )
            self._gc_checkpoint_best_effort(best_delete_name)
            self._gc_checkpoint_best_effort(epoch_delete_name)

        # Agreed publish phase (P0-4): all ranks share the outcome.
        self._finalize_publish(_publish_epoch_committed)

    def _gc_checkpoint_best_effort(self, checkpoint_name):
        """Delete a rotated checkpoint only after pointer+manifest commit."""
        if checkpoint_name is None:
            return
        protected = set(self.best_quene + self.latest_quene + self.epoch_quene)
        protected.update(
            name for name in (
                self._latest_checkpoint_name,
                self._best_checkpoint_name,
                self._latest_resumable_checkpoint_name,
            )
            if name is not None
        )
        if checkpoint_name in protected:
            return
        delete_path = os.path.join(self.checkpoint_path, checkpoint_name + ".pth")
        try:
            if os.path.exists(delete_path):
                os.remove(delete_path)
        except Exception as exc:
            log.warning("Failed to GC checkpoint %s: %s", delete_path, exc)

    def _safe_harvest_plugin_states(self):
        harvester = getattr(self.trainer, "harvest_plugin_states", None)
        if not callable(harvester):
            return {}
        try:
            return harvester()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to harvest plugin states for checkpoint: %s", exc)
            return {}

    def _build_training_state_blob(self, kind):
        """Assemble the resume state machine block (BUG 1/2/3 metadata).

        ``epoch_scheduler_step_pending`` is only true for an epoch checkpoint
        under per-epoch LR scheduling: Saver.epoch fires before run()'s epoch-end
        LR step, so the resumed run must replay exactly one step.
        """
        trainer = self.trainer
        epoch_pending = (
            kind == CHECKPOINT_KIND_EPOCH
            and not bool(getattr(trainer, "update_lr_per_iter", False))
        )
        # Schema >=3 canonical semantics: global_step == LAST COMPLETED optimizer
        # step for BOTH kinds; restart resumes at global_step + 1. Saver.iteration
        # fires before trainer.iter increments (iter == the step just committed),
        # but Saver.epoch fires after the whole loop, when iter has already been
        # advanced past the last step — store iter-1 there or an epoch resume
        # skips one global-step value (LR/warmup/cadence all keyed off it).
        current_iter = int(getattr(trainer, "iter", 1))
        last_completed = current_iter - 1 if kind == CHECKPOINT_KIND_EPOCH else current_iter
        ts = TrainingState(
            global_step=last_completed,
            epoch=int(getattr(trainer, "ep", 1)),
            batch_in_epoch=int(getattr(trainer, "_batch_in_epoch", 0)),
            checkpoint_kind=kind,
            epoch_scheduler_step_pending=epoch_pending,
            rng_state=capture_rng_state(),
            plugin_state=self._safe_harvest_plugin_states(),
            metrics={"best_loss": float(self.best_loss)},
        )
        return ts.state_dict()

    def _resume_capability(self, kind):
        """Classify this checkpoint without over-promising replay exactness."""
        if kind == CHECKPOINT_KIND_EPOCH:
            return "exact"
        checker = getattr(self.trainer, "_loaders_support_exact_fast_forward", None)
        if not callable(checker):
            checker = getattr(self.trainer, "_loader_supports_exact_fast_forward", None)
        if callable(checker):
            try:
                if checker():
                    return "exact"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Could not determine iteration resume capability: %s", exc)
        return "inexact_opt_in"

    def _assemble_checkpoint_obj(self, name, kind, model_options, common_options,
                                 train_options, model_state, opt_blocks,
                                 rank_states=None):
        """ONE checkpoint-dict assembly shared by single and dist-expert paths.

        Carries both the historical flat keys (``epoch``/``iteration``/``stats``
        + optimizer/scheduler) *and* the new ``training_state``/``plugin_state``
        blocks, so legacy readers and the new state machine both resolve. This is
        the single source of truth the CheckpointCoordinator refactor centralises.
        """
        obj = {
            "config": {
                "model_options": model_options,
                "common_options": common_options,
                "train_options": train_options,
            },
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": kind,
            "model_state_dict": model_state,
            "task": self.trainer.task,
            "epoch": self.trainer.ep,
            "stats": self.trainer.stats,
        }
        training_state = self._build_training_state_blob(kind)
        obj["training_state"] = training_state
        # Single source of truth: the flat legacy key mirrors training_state's
        # canonical last-completed-step, so old and new readers agree.
        obj["iteration"] = int(training_state.get("global_step", self.trainer.iter))
        resume_capability = self._resume_capability(kind)
        committed_epoch = (
            int(training_state.get("epoch", self.trainer.ep))
            if kind == CHECKPOINT_KIND_EPOCH
            else max(int(training_state.get("epoch", self.trainer.ep)) - 1, 0)
        )
        world_size = int(getattr(self.trainer, "world_size", 1))
        obj["resume_capability"] = resume_capability
        obj["committed_epoch"] = committed_epoch
        obj["world_size"] = world_size
        self._pending_manifest_entries[name] = {
            "filename": name + ".pth",
            "checkpoint_kind": kind,
            "resume_capability": resume_capability,
            "last_completed_step": int(training_state["global_step"]),
            "committed_epoch": committed_epoch,
            "world_size": world_size,
        }
        # Per-rank resume state (P0-3): restore prefers the own-rank entry over
        # the main-rank blob inside training_state.
        obj["rank_states"] = dict(rank_states or {})
        # Top-level mirror keeps the legacy reader trivially able to find plugin
        # state even if it never learns the training_state schema.
        obj["plugin_state"] = training_state.get("plugin_state", {})
        for block in opt_blocks:
            obj.update(block)
        return obj

    def _collect_rank_states(self):
        """Gather per-rank resume state (P0-3: rank-local RNG).

        The training-state blob is assembled on the main rank only, so its
        ``rng_state`` is rank0's — restoring it identically on every rank would
        correlate the ranks' dropout/noise streams. Collect each rank's own
        snapshot with a collective (runs on ALL ranks in the distributed path)
        and store them keyed by global rank; restore picks the own entry.
        """
        local = {"rng_state": capture_rng_state()}
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            return {str(i): s for i, s in enumerate(gathered)}
        return {"0": local}

    def _finalize_publish(self, publish_fn):
        """Run the main-rank retention/publish/manifest block as an agreed phase.

        The checkpoint bytes are already committed by ``_save``; pointer
        publication happens only on the main rank. If that fails, the other
        ranks must not sail on into the next training collective (they would
        wedge once the main rank died) — broadcast the outcome, raise uniformly
        on every rank, then barrier so all ranks leave the checkpoint phase
        together (P0-4).
        """
        publish_ok = [True]
        publish_err = None
        if self._is_main():
            try:
                publish_fn()
            except Exception as exc:
                publish_ok[0] = False
                publish_err = exc
                log.error("checkpoint publish failed on main rank: %s", exc)
        dist_ready = dist.is_available() and dist.is_initialized()
        if dist_ready:
            dist.broadcast_object_list(publish_ok, src=0)
        if not publish_ok[0]:
            if publish_err is not None:
                raise publish_err
            raise RuntimeError(
                "checkpoint publish failed on the main rank; aborting on all ranks"
            )
        if dist_ready:
            self._barrier_on_current_device()

    def _save(self, name, model, model_options, common_options, train_options,
              kind=CHECKPOINT_KIND_ITERATION):
        previous_profile_save_name = self._profile_save_name
        self._profile_save_name = name
        self._profile_record("save_start")

        if self._is_dist_expert():
            prepared = None
            prepare_exc = None
            try:
                with self._profile_stage("prepare_local_dist_states"):
                    prepared = self._prepare_local_dist_states()
                prepare_status = {
                    "ok": True,
                    "rank": self._rank(),
                    "error_type": None,
                    "error": None,
                    "traceback": None,
                    "local_expert_idx": prepared["local_idx"],
                    "expert_dp_rank": int(getattr(self.trainer, "expert_dp_rank", 0)),
                    "replica_fingerprint": (
                        prepared["replica_fingerprint"].tolist()
                        if prepared["replica_fingerprint"] is not None
                        else None
                    ),
                }
            except Exception as exc:
                prepare_exc = exc
                prepare_status = {
                    "ok": False,
                    "rank": self._rank(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "local_expert_idx": getattr(self.trainer, "local_expert_idx", None),
                    "expert_dp_rank": int(getattr(self.trainer, "expert_dp_rank", 0)),
                    "replica_fingerprint": None,
                }

            prepare_statuses = [None] * dist.get_world_size()
            dist.all_gather_object(prepare_statuses, prepare_status)
            failed_prepare = next(
                (status for status in prepare_statuses if not status.get("ok", False)),
                None,
            )
            if failed_prepare is not None:
                self._profile_save_name = previous_profile_save_name
                message = (
                    "checkpoint local prepare failed on rank "
                    f"{failed_prepare.get('rank')}: "
                    f"{failed_prepare.get('error_type')}: {failed_prepare.get('error')}"
                )
                if prepare_exc is not None and self._rank() == failed_prepare.get("rank"):
                    raise RuntimeError(message) from prepare_exc
                raise RuntimeError(message)

            # Fingerprints ride the same small status exchange, so replica
            # mismatch detection is uniform and adds no pre-gather collective.
            self._validate_prepared_expert_dp_replicas(prepare_statuses)
            with self._profile_stage("gather_dist_states"):
                expert_states, opt_states, sch_states = self._gather_dist_states(prepared)
            # Collective: every rank contributes its own RNG snapshot (P0-3).
            rank_states = self._collect_rank_states()

            # Rank0 writes; all ranks agree on success BEFORE the barrier so a
            # rank0 failure raises uniformly instead of leaving the other ranks
            # blocked forever on dist.barrier() (BUG 2, distributed hang).
            save_ok = [True]
            save_err = None
            if self._is_main():
                try:
                    with self._profile_stage("assemble_full_model_state"):
                        full_model_state = self._assemble_full_model_state(expert_states)
                    obj = self._assemble_checkpoint_obj(
                        name, kind, model_options, common_options, train_options,
                        full_model_state,
                        [{"optimizers_state_dict": opt_states,
                          "lr_schedulers_state_dict": sch_states}],
                        rank_states=rank_states,
                    )
                    f_path = os.path.join(self.checkpoint_path, name + ".pth")
                    self._profile_record(
                        "save_object_summary",
                        model_tensor_bytes=self._tensor_bytes(full_model_state),
                        optimizer_tensor_bytes=self._tensor_bytes(opt_states),
                        scheduler_tensor_bytes=self._tensor_bytes(sch_states),
                    )
                    self._write_checkpoint_atomic(obj, f_path)
                    self._profile_record(
                        "save_file_summary",
                        checkpoint_file_bytes=os.path.getsize(f_path) if os.path.exists(f_path) else None,
                    )
                    log.info(msg="checkpoint saved as {}".format(name))
                except Exception as exc:
                    save_ok[0] = False
                    save_err = exc
                    log.error("rank0 checkpoint save failed for %s: %s", name, exc)

            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(save_ok, src=0)
            if not save_ok[0]:
                # Uniform failure on every rank, raised before the barrier.
                self._profile_save_name = previous_profile_save_name
                if save_err is not None:
                    raise save_err
                raise RuntimeError(
                    f"rank0 failed to save checkpoint {name}; aborting on all ranks"
                )

            with self._profile_stage("post_save_barrier"):
                dist.barrier()
            self._profile_record("save_end")
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

        obj = self._assemble_checkpoint_obj(
            name, kind, model_options, common_options, train_options,
            model_state, [optim_state, sched_state],
            rank_states=self._collect_rank_states(),
        )

        f_path = os.path.join(self.checkpoint_path, name + ".pth")
        self._profile_record(
            "save_object_summary",
            model_tensor_bytes=self._tensor_bytes(model_state),
            optimizer_tensor_bytes=self._tensor_bytes(optim_state),
            scheduler_tensor_bytes=self._tensor_bytes(sched_state),
        )
        self._write_checkpoint_atomic(obj, f_path)
        self._profile_record(
            "save_file_summary",
            checkpoint_file_bytes=os.path.getsize(f_path) if os.path.exists(f_path) else None,
        )
        log.info(msg="checkpoint saved as {}".format(name))
        self._profile_record("save_end")
        self._profile_save_name = previous_profile_save_name
