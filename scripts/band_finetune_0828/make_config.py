#!/usr/bin/env python
"""Emit the fine-tune input.json from the pretrained arm's own config.

model_options and common_options are lifted verbatim from the 1089717
checkpoint -- a fine-tune that silently changes architecture is not a
fine-tune. Only the data source, the loss and the schedule are new.
"""
import argparse
import json

import sys

import torch

sys.path.insert(0, "/data/wgh/0828_band_finetune/code/DeePTB_bandft")

CKPT = "/data/wgh/relay_1089717/nnenv.iter100000.pth"
WS = "/data/wgh/0828_band_finetune"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=10000)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--coeff-ham", type=float, default=0.9)
    ap.add_argument("--band-emax", type=float, default=123.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--peak-lr", type=float, default=None,
                    help="Override the checkpoint's peak lr. The pretrained "
                         "0.01 was tuned for batch 96; at batch 1 it diverges "
                         "(grad_norm 7.8e5 in epoch 1).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    train = dict(cfg["train_options"])



    total = 200 if args.smoke else args.total_steps
    warmup = max(10, int(total * args.warmup_frac))

    if args.peak_lr is not None:
        train["optimizer"] = dict(train["optimizer"])
        train["optimizer"]["lr"] = args.peak_lr

    train["lr_scheduler"] = {
        "type": "wsd", "total_steps": total, "warmup_steps": warmup,
        "warmup_lr": 1e-05, "min_lr": 1e-05,
        "decay_ratio": 0.65, "decay_type": "cosine",
        "decay_steps": None, "last_epoch": -1,
    }
    train["num_epoch"] = 10 ** 9

    # k-points and reference bands are ragged across structures (nk 102..244,
    # nband 15..736), so they cannot be collated into a rectangular batch.
    train["batch_size"] = args.batch_size
    train["val_batch_size"] = 1
    train["ref_batch_size"] = 1
    train["dynamic_batch"] = dict(train.get("dynamic_batch", {}))
    train["dynamic_batch"]["enabled"] = False
    train["train_num_workers"] = 1
    train["val_num_workers"] = 1
    train["ref_num_workers"] = 1
    train["data_persistent_workers"] = True
    train["data_prefetch_factor"] = 2

    # train.py asserts display_freq >= validation_freq.
    freq = 20 if args.smoke else 200
    train["display_freq"] = freq
    train["validation_freq"] = freq
    train["save_freq"] = freq
    train["max_ckpt"] = 2
    train["use_ddp"] = False
    train["expert_data_parallel_size"] = 1

    band_loss = {
        "method": "eig_ham_h0res",
        "coeff_ham": args.coeff_ham,
        "band_overlap": True,
        "band_emin": 0.0,
        "band_emax": args.band_emax,
        "eout_weight": 0.01,
        "onsite_shift": False,
    }
    train["loss_options"] = {
        "train": dict(band_loss),
        "validation": dict(band_loss),
    }

    data_common = {
        "prefix": "data", "separator": ".", "type": "LMDBDataset",
        "get_Hamiltonian": True, "get_H0": True, "get_DM": False,
        "get_overlap": True, "get_eigenvalues": True,
        "residual_hamiltonian": False,
    }
    out = {
        "common_options": cfg["common_options"],
        "model_options": cfg["model_options"],
        "train_options": train,
        "data_options": {
            "train": dict(data_common, root=f"{WS}/data/ft_train"),
            "validation": dict(data_common, root=f"{WS}/data/ft_valid"),
            "r_max": None, "er_max": None, "oer_max": None,
        },
    }
    # The checkpoint's train_options carries keys the runtime injected at
    # launch (ddp_world_size, ddp_rank, ...). dargs runs in strict mode and
    # rejects anything it does not define, so drop exactly what it names
    # rather than guessing a whitelist.
    import copy
    import re as _re
    from dptb.utils.argcheck import normalize as _normalize

    dropped = []
    for _ in range(40):
        try:
            _normalize(copy.deepcopy(out))
            break
        except Exception as exc:
            m = _re.search(r"undefined key `([^`]+)`", str(exc))
            loc = _re.search(r"at location `([^`]+)`", str(exc))
            if not m:
                raise
            key = m.group(1)
            section = (loc.group(1) if loc else "train_options").split("/")[0]
            target = out.get(section, {})
            if key not in target:
                raise
            target.pop(key)
            dropped.append("%s/%s" % (section, key))
    else:
        raise SystemExit("could not normalize config after 40 key drops")
    if dropped:
        print("  dropped runtime-only keys:", ", ".join(dropped))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)
    print("  total_steps=%d warmup=%d peak_lr=%s coeff_ham=%.2f band_emax=%.0f batch=%d"
          % (total, warmup, train["optimizer"]["lr"], args.coeff_ham,
             args.band_emax, args.batch_size))


if __name__ == "__main__":
    main()
