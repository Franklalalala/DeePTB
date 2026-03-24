import os
import json
import time
import heapq
import logging
import sys
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from dptb.plugins.monitor import (
    TrainLossMonitor, LearningRateMonitor, Validationer, TensorBoardMonitor,
    DeepDoctorMonitor, SO2ModuleMonitor, PreTPBlockMonitor,
    TrainOnsiteLossMonitor, TrainHoppingLossMonitor, TrainZLossMonitor, ExpertLoadCVMonitor,
    ScalarFieldMonitor
)
from dptb.plugins.train_logger import Logger
from dptb.plugins.saver import Saver
from dptb.utils.argcheck import normalize, collect_cutoffs, chk_avg_per_iter
from dptb.utils.tools import j_loader, setup_seed, j_must_have
from dptb.utils.loggers import set_log_handles

from dptb.entrypoints.train import deep_dict_difference, print_model_params_detailed
from dptb.nnops.multi_trainer import MultiTrainer

__all__ = ["multi_train"]

log = logging.getLogger(__name__)


# --------------------------- TAGGER (entrypoint) ---------------------------
class _EntryTagger:
    def __init__(self, enabled: bool, cuda_mem: bool, cuda_sync: bool):
        self.enabled = bool(enabled)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)

    def _cuda_mem(self, device: torch.device):
        if not (torch.cuda.is_available() and device.type == "cuda"): return None
        alloc, reserved, peak, free, total = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(
            device), torch.cuda.max_memory_allocated(device), *torch.cuda.mem_get_info(device)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None: return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return f" | cuda_alloc={alloc / mb:.1f}MB cuda_reserved={reserved / mb:.1f}MB cuda_peak={peak / mb:.1f}MB free={free / mb:.1f}MB total={total / mb:.1f}MB"

    @contextlib.contextmanager
    def tag(self, name: str, device: Optional[torch.device] = None, extra: str = ""):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        dev = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if self.cuda_mem and dev.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass
        try:
            yield
        finally:
            if self.cuda_sync and dev.type == "cuda": torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem(dev) if (self.cuda_mem and dev.type == "cuda") else None
            log.info(f"[TAG][ENTRY][{name}] dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")


# --------------------------- param printing helpers ---------------------------
def _format_params_lazy(num: int) -> str:
    if num >= 1_000_000_000: return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000: return f"{num / 1_000_000:.2f}M"
    if num >= 1_000: return f"{num / 1_000:.2f}K"
    return str(num)


def _count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def print_multi_model_params_detailed(model: nn.Module, logger=None, max_depth: int = 5):
    log_func = logger.info if logger else print
    if not hasattr(model, "experts") or not isinstance(model.experts, nn.ModuleList) or len(model.experts) == 0:
        print_model_params_detailed(model, logger=logger, max_depth=max_depth)
        return

    expert_stats = [_count_params(expert) for expert in model.experts]
    single_stat = expert_stats[0]
    experts_sum = {"total": sum(x["total"] for x in expert_stats),
                   "trainable": sum(x["trainable"] for x in expert_stats)}
    wrapper_stat = _count_params(model)

    log_func("=" * 80)
    log_func("MULTI-EXPERT PARAMETER SUMMARY")
    log_func("=" * 80)
    log_func(f"Number of Experts:       {len(expert_stats)}")
    log_func(
        f"Single Expert (expert_0): total={_format_params_lazy(single_stat['total'])}, trainable={_format_params_lazy(single_stat['trainable'])}")
    log_func(
        f"All Experts Sum:         total={_format_params_lazy(experts_sum['total'])}, trainable={_format_params_lazy(experts_sum['trainable'])}")
    log_func(
        f"Wrapper model.params:    total={_format_params_lazy(wrapper_stat['total'])}, trainable={_format_params_lazy(wrapper_stat['trainable'])}")
    log_func("=" * 80)
    print_model_params_detailed(model.experts[0], logger=logger, max_depth=max_depth)


# --------------------------- main ---------------------------
def multi_train(INPUT: str, init_model: Optional[str], restart: Optional[str], output: str, log_level: int,
                log_path: Optional[str], **kwargs):
    run_opt: Dict[str, Any] = {"init_model": init_model, "restart": restart, "log_path": log_path,
                               "log_level": log_level}
    if all((run_opt["init_model"], restart)): raise RuntimeError(
        "--init-model and --restart should not be set at the same time")

    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        log_path = log_path or os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        run_opt.update(
            {"output": str(Path(output).absolute()), "checkpoint_path": str(Path(checkpoint_path).absolute()),
             "log_path": str(Path(log_path).absolute())})

    set_log_handles(log_level, Path(log_path) if log_path else None)

    if sys.platform.startswith('win'):
        for handler in logging.root.handlers + logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.close()
                handler.stream = open(handler.baseFilename, handler.mode, encoding='utf-8')

    jdata = normalize(j_loader(INPUT))
    dbg = jdata.get("train_options", {})
    entry_tagger = _EntryTagger(enabled=bool(dbg.get("debug_tags", True)),
                                cuda_mem=bool(dbg.get("debug_tag_cuda_mem", True)),
                                cuda_sync=bool(dbg.get("debug_tag_cuda_sync", False)))

    with entry_tagger.tag("set_default_dtype"):
        torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    with entry_tagger.tag("merge_config_from_ckpt_or_restart"):
        if restart or init_model:
            f = torch.load(restart if restart else init_model, map_location="cpu", weights_only=False)
            if jdata.get("model_options", None) is None: jdata["model_options"] = f["config"]["model_options"]
            if restart:
                jdata["train_options"] = f["config"]["train_options"]
            elif jdata.get("train_options", None) is None:
                jdata["train_options"] = f["config"]["train_options"]
            del f
        else:
            j_must_have(jdata, "model_options")
            j_must_have(jdata, "train_options")

    cutoff_options = collect_cutoffs(jdata)
    with entry_tagger.tag("setup_seed"):
        setup_seed(seed=jdata["common_options"]["seed"])

    with entry_tagger.tag("build_dataset"):
        train_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["train"], **jdata["common_options"])
        validation_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["validation"],
                                            **jdata["common_options"]) if jdata["data_options"].get(
            "validation") else None
        reference_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["reference"],
                                           **jdata["common_options"]) if jdata["data_options"].get(
            "reference") else None

    jdata["common_options"]["overlap"] = False
    distance_ranges = jdata["train_options"].get("distance_ranges", [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]])

    # ================= 核心：默认显存管理配置 =================
    # 保留用户设置，但如果没有设置，则走 "串行，常驻GPU，强力清理" 的最佳策略
    jdata["train_options"]["parallel_multi"] = bool(jdata["train_options"].get("parallel_multi", False))
    jdata["train_options"]["serial_offload_experts"] = bool(
        jdata["train_options"].get("serial_offload_experts", False))  # False: 常驻GPU，速度快
    jdata["train_options"]["serial_empty_cache_per_expert"] = bool(
        jdata["train_options"].get("serial_empty_cache_per_expert", True))  # True: 每次算完强力清理碎片，防OOM
    jdata["train_options"]["serial_zero_grad_immediately"] = bool(
        jdata["train_options"].get("serial_zero_grad_immediately", True))
    jdata["train_options"]["log_single_model_compatible_loss"] = True
    jdata["train_options"]["log_single_model_compatible_loss_mode"] = "reduce"
    # ==========================================================

    if restart:
        with entry_tagger.tag("trainer/restart"):
            trainer = MultiTrainer.restart(checkpoint=restart, train_datasets=train_datasets,
                                           train_options=jdata["train_options"], common_options=jdata["common_options"],
                                           reference_datasets=reference_datasets,
                                           validation_datasets=validation_datasets)
    else:
        with entry_tagger.tag("build_model"):
            model = build_model(checkpoint=init_model, model_options=jdata["model_options"],
                                common_options=jdata["common_options"], train_options=jdata["train_options"])
        if jdata["model_options"]["prediction"].get('scale_type', "scale_w_back_grad") != 'no_scale':
            train_datasets.E3statistics(model=model)
        with entry_tagger.tag("trainer/init"):
            trainer = MultiTrainer(distance_ranges=distance_ranges, train_options=jdata["train_options"],
                                   common_options=jdata["common_options"], model=model, train_datasets=train_datasets,
                                   validation_datasets=validation_datasets, reference_datasets=reference_datasets)

    with entry_tagger.tag("trainer/register_plugins"):
        log_field = ["train_loss", "lr"]
        if validation_datasets:
            trainer.register_plugin(
                Validationer(interval=[(jdata["train_options"]["validation_freq"], 'iteration'), (1, 'epoch')],
                             fast_mode=jdata["train_options"]["valid_fast"]))
            log_field.append("validation_loss")

        trainer.register_plugin(TrainLossMonitor(sliding_win_size=jdata["train_options"]["sliding_win_size"],
                                                 avg_per_iter=chk_avg_per_iter(jdata)))
        trainer.register_plugin(LearningRateMonitor())
        for monitor in [TrainOnsiteLossMonitor, TrainHoppingLossMonitor, TrainZLossMonitor, ExpertLoadCVMonitor]:
            trainer.register_plugin(monitor(interval=[(1, 'iteration'), (1, 'epoch')]))

        trainer.register_plugin(
            ScalarFieldMonitor(stat_name="train_loss_opt", interval=[(1, 'iteration'), (1, 'epoch')]))
        for i in range(trainer.num_experts):
            trainer.register_plugin(
                ScalarFieldMonitor(stat_name=f"expert_{i}_onsite", interval=[(1, 'iteration'), (1, 'epoch')]))
            trainer.register_plugin(
                ScalarFieldMonitor(stat_name=f"expert_{i}_hopping", interval=[(1, 'iteration'), (1, 'epoch')]))

        log_field.extend(["mean_max_prob", "expert_load_cv", "train_onsite_loss", "train_hopping_loss"])

        if jdata["train_options"].get("use_tensorboard"):
            trainer.register_plugin(
                TensorBoardMonitor(interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')],
                                   log_dir=os.path.join(output,
                                                        "tensorboard_logs") if output else "./tensorboard_logs"))

        trainer.register_plugin(
            Logger(log_field, interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))
        for q in trainer.plugin_queues.values(): heapq.heapify(q)

        if output:
            with open(os.path.join(output, "train_config.json"), "w") as fp:
                json.dump(jdata, fp, indent=4)
            if jdata["train_options"].get("save_freq"):
                trainer.register_plugin(
                    Saver(interval=[(jdata["train_options"].get("save_freq"), 'iteration'), (1, 'epoch')]),
                    checkpoint_path=run_opt["checkpoint_path"])

    print_multi_model_params_detailed(trainer.model, logger=log, max_depth=5)

    with entry_tagger.tag("trainer/run", device=torch.device(jdata["common_options"]["device"])):
        start_time = time.time()
        trainer.run(trainer.train_options["num_epoch"])
        log.info("finished training")
        log.info(f"wall time: {(time.time() - start_time):.3f} s")


if __name__ == "__main__":
    import shutil

    if os.path.exists('output_dir_multi'): shutil.rmtree('output_dir_multi')
    multi_train(INPUT=r'general_debug_Al_10_cubic.json', output=r'output_dir_multi', log_level=2, log_path=r'log.txt',
                init_model=None, restart=None)