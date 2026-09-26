#!/usr/bin/env python
"""Calibrate from config.data_options.train exactly once; never opens validation.

PYTHONPATH=. python tools/calibrate_structure_mole.py input.json train_stats.pt
The input must enable structure_mole. Dense initialization is deferred until
the later fresh training build; this tool only constructs the descriptor.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.nn.structure_mole import fit_structure_stats
from dptb.utils.argcheck import collect_cutoffs, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = Path(args.config)
    original = normalize(json.loads(path.read_text()))
    cfg = copy.deepcopy(original)
    cfg["common_options"]["device"] = args.device
    opt = cfg["model_options"]["embedding"]["structure_mole"]
    opt.update(init_from="", stats_path="")
    model = build_model(model_options=cfg["model_options"], common_options=cfg["common_options"],
                        train_options=cfg["train_options"])
    dataset = build_dataset(**collect_cutoffs(cfg), **cfg["data_options"]["train"], **cfg["common_options"])
    bundle = fit_structure_stats(model, (dataset[i] for i in range(len(dataset))), split="train")
    bundle["source_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle["train_data_options"] = original["data_options"]["train"]
    torch.save(bundle, args.output)
    print(f"Calibrated {len(dataset)} training structures -> {args.output}")


if __name__ == "__main__":
    main()
