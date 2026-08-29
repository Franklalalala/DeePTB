#!/usr/bin/env python
"""Config for arm A: mu-gauged hamil_abs, no k-space term, no band labels.

Built from the baseline checkpoint's own config so the architecture is
verbatim; only the loss, the schedule and the data source differ.

The smoke run deliberately uses the 138-structure set, which already carries S.
It is not meant to produce a result -- it checks that the gauge solve behaves:
|mu| in the meV range and the loss going DOWN after the shift. The real run
waits for S to reach the 12k training LMDB.
"""
import argparse
import copy
import json
import re
import sys

import torch

sys.path.insert(0, "/data/wgh/0828_band_finetune/code/DeePTB_bandft")

CKPT = "/data/wgh/relay_1089717/nnenv.iter100000.pth"
WS = "/data/wgh/0828_band_finetune"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--freq", type=int, default=20)
    ap.add_argument("--no-gauge", action="store_true",
                    help="control arm: same loss without the mu shift")
    ap.add_argument("--train-root", default=f"{WS}/data/ft_train")
    ap.add_argument("--valid-root", default=f"{WS}/data/ft_valid")
    args = ap.parse_args()

    cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["config"]
    train = dict(cfg["train_options"])

    train["optimizer"] = dict(train["optimizer"])
    train["optimizer"]["lr"] = args.peak_lr
    train["lr_scheduler"] = {
        "type": "wsd", "total_steps": args.total_steps,
        "warmup_steps": max(10, args.total_steps // 20),
        "warmup_lr": 1e-05, "min_lr": 1e-05,
        "decay_ratio": 0.65, "decay_type": "cosine",
        "decay_steps": None, "last_epoch": -1,
    }
    train["num_epoch"] = 10 ** 9
    train["batch_size"] = args.batch_size
    train["val_batch_size"] = 1
    train["ref_batch_size"] = 1
    train["dynamic_batch"] = dict(train.get("dynamic_batch", {}))
    train["dynamic_batch"]["enabled"] = False
    for k in ("train_num_workers", "val_num_workers", "ref_num_workers"):
        train[k] = 1
    train["display_freq"] = args.freq
    train["validation_freq"] = args.freq
    train["save_freq"] = args.freq
    train["max_ckpt"] = 2
    train["use_ddp"] = False
    train["expert_data_parallel_size"] = 1

    loss = {"method": "hamil_abs_gauged", "onsite_shift": False,
            "gauge": not args.no_gauge, "gauge_clip": 1.0}
    train["loss_options"] = {"train": dict(loss), "validation": dict(loss)}

    data_common = {
        "prefix": "data", "separator": ".", "type": "LMDBDataset",
        "get_Hamiltonian": True, "get_H0": True, "get_DM": False,
        "get_overlap": True,          # arm A needs S, but NOT eigenvalues
        "get_eigenvalues": False,
        "residual_hamiltonian": False,
    }
    out = {
        "common_options": cfg["common_options"],
        "model_options": cfg["model_options"],
        "train_options": train,
        "data_options": {
            "train": dict(data_common, root=args.train_root),
            "validation": dict(data_common, root=args.valid_root),
            "r_max": None, "er_max": None, "oer_max": None,
        },
    }

    from dptb.utils.argcheck import normalize
    dropped = []
    for _ in range(40):
        try:
            normalize(copy.deepcopy(out))
            break
        except Exception as exc:
            m = re.search(r"undefined key `([^`]+)`", str(exc))
            loc = re.search(r"at location `([^`]+)`", str(exc))
            if not m:
                raise
            key = m.group(1)
            sec = (loc.group(1) if loc else "train_options").split("/")[0]
            if key not in out.get(sec, {}):
                raise
            out[sec].pop(key)
            dropped.append("%s/%s" % (sec, key))
    else:
        raise SystemExit("could not normalize after 40 drops")

    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    if dropped:
        print("  dropped:", ", ".join(dropped))
    print("  loss=hamil_abs_gauged gauge=%s steps=%d lr=%s batch=%d"
          % (not args.no_gauge, args.total_steps, args.peak_lr, args.batch_size))


if __name__ == "__main__":
    main()
