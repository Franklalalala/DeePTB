import copy
import heapq
import logging
import torch
import json
import os
import time
from pathlib import Path
from dptb.checkpoint_config import merge_checkpoint_common_options
from dptb.configuration import migrate_legacy_checkpoint_model_options
from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from typing import Optional
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_loader, setup_seed
from dptb.nnops.tester import Tester
from dptb.utils.argcheck import normalize_test, collect_cutoffs
from dptb.plugins.monitor import TestLossMonitor, TensorBoardMonitor, ScalarFieldMonitor
from dptb.plugins.train_logger import Logger

__all__ = ["test"]

log = logging.getLogger(__name__)

def _test(
        INPUT: str,
        init_model: str,
        output: str,
        log_level: int,
        log_path: Optional[str],
        use_correction: Optional[str] = None,
        **kwargs
):
    # TODO: permit commandline init_model and config file init.
    run_opt = {
        "init_model": init_model,
        "log_path": log_path,
        "log_level": log_level,
        "use_correction": use_correction,
        "freeze":True,
        "train_soc":False
    }
    
    # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        results_path = os.path.join(str(output), "results")
        Path(results_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
                        "output": str(Path(output).absolute()),
                        "results_path": str(Path(results_path).absolute()),
                        "log_path": str(Path(log_path).absolute())
                        })
    
    raw_jdata = j_loader(INPUT)
    explicit_common_options = copy.deepcopy(raw_jdata.get("common_options", {}))
    jdata = normalize_test(raw_jdata)

    checkpoint = torch.load(init_model, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    checkpoint_common_options = checkpoint_config["common_options"]
    basis = checkpoint_common_options["basis"]
    for asym, orb in jdata["common_options"]["basis"].items():
        if asym not in basis:
            raise ValueError(f"Atom {asym} not found in model's basis")
        if orb != basis[asym]:
            raise ValueError(
                f"Orbital {orb} of Atom {asym} not consistent with the model's basis"
            )

    jdata["common_options"] = merge_checkpoint_common_options(
        jdata["common_options"],
        checkpoint_common_options,
        explicit_common_options,
        preserve_runtime_defaults=True,
    )
    setup_seed(seed=jdata["common_options"]["seed"])

    set_log_handles(log_level, Path(log_path) if log_path else None)

    jdata["model_options"] = migrate_legacy_checkpoint_model_options(
        checkpoint_config["model_options"]
    )
    del checkpoint
    
    cutoff_options = collect_cutoffs(jdata)
    cutoff_options = {
        key: cutoff_options.get(key, jdata["data_options"].get(key))
        for key in ("r_max", "oer_max", "er_max")
    }
    test_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["test"], **jdata["common_options"])
    model = build_model(
        run_opt["init_model"],
        model_options=jdata["model_options"],
        common_options=jdata["common_options"],
        explicit_common_options=explicit_common_options,
    )
    build_dataset.check_cutoffs(model=model)
    model.eval()
    tester = Tester(
        test_options=jdata["test_options"],
        common_options=jdata["common_options"],
        model = model,
        test_datasets=test_datasets,
    )

    # register the plugin in tester, to tract training info
    tester.register_plugin(TestLossMonitor())
    for test_field in ("test_onsite_loss", "test_hopping_loss"):
        tester.register_plugin(
            ScalarFieldMonitor(
                stat_name=test_field,
                interval=[(1, 'iteration'), (1, 'epoch')],
            )
        )

    if bool(jdata.get("test_options", {}).get("use_tensorboard", False)):
        tb_dir = os.path.join(str(output), "tensorboard_logs") if output else "./tensorboard_logs"
        tester.register_plugin(TensorBoardMonitor(interval=[(1, 'epoch')], log_dir=tb_dir))

    tester.register_plugin(Logger(["test_loss", "test_onsite_loss", "test_hopping_loss"],
        interval=[(1, 'iteration'), (1, 'epoch')]))
    
    for q in tester.plugin_queues.values():
        heapq.heapify(q)
    
    tester.build()

    if output:
        # output training configurations:
        with open(os.path.join(output, "test_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

    start_time = time.time()

    tester.run()

    end_time = time.time()
    log.info("finished testing")
    log.info(f"wall time: {(end_time - start_time):.3f} s")
