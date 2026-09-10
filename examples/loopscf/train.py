#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train non-SOC LoopSCF with head or MoE adaptation."""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("DPTB_REPO_ROOT", os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)


def _has_wm(sd):
    return any(("wm_node" in k) or ("wm_edge" in k) for k in sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-model")
    source.add_argument("--restart")
    ap.add_argument(
        "--base-model",
        default=os.environ.get("LOOPSCF_BASE_CKPT"),
        help="Matching base checkpoint used to construct a model with saved LoopSCF adapters",
    )
    ap.add_argument("--log", default="log.txt")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--mode", choices=("head", "moe"), required=True)
    ap.add_argument("--n-k-train", type=int, default=5)
    args = ap.parse_args()
    if args.K < 1 or args.n_k_train < 1:
        ap.error("--K and --n-k-train must be positive")
    if args.restart:
        if not os.path.isfile(args.restart):
            raise SystemExit("restart missing: %s" % args.restart)
    elif not os.path.isfile(args.init_model):
        ap.error("initial checkpoint does not exist: %s" % args.init_model)

    import importlib
    import torch
    from dptb.nnops.loopscf import (
        ARM1_PATTERNS,
        ARM2_PATTERNS,
        freeze_by_patterns,
        install_working_memory_true_diag,
        patch_fw10_per_graph,
        patch_stepwise_loss,
    )

    train_mod = importlib.import_module("dptb.entrypoints.train")
    nn_build = importlib.import_module("dptb.nn.build")
    trainer_mod = importlib.import_module("dptb.nnops.trainer")
    dptb_nn = importlib.import_module("dptb.nn")

    print(
        "[LOOPSCF] mode=%s K=%d n_k=%d restart=%s REPO=%s"
        % (args.mode, args.K, args.n_k_train, args.restart, REPO),
        flush=True,
    )
    patch_stepwise_loss(K=args.K)
    patch_fw10_per_graph()

    with open(args.input, encoding="utf-8") as f:
        cfg = json.load(f)
    basis = cfg["common_options"]["basis"]
    orig_build = nn_build.build_model
    patterns = ARM1_PATTERNS if args.mode == "head" else ARM2_PATTERNS

    def _install(model):
        device = next(model.parameters()).device
        from dptb.data import AtomicDataDict
        from dptb.data.transforms import OrbitalMapper
        from dptb.nn.hr2hk import HR2HK

        idp = OrbitalMapper(basis, method="e3tb", device=device)
        h2k = HR2HK(
            idp=idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.HAMILTONIAN_KEY,
            device=device,
        )
        s2k = HR2HK(
            idp=idp,
            overlap=True,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            device=device,
        )
        install_working_memory_true_diag(
            model,
            mode=args.mode,
            idp=idp,
            h2k=h2k,
            s2k=s2k,
            K=args.K,
            n_k_train=args.n_k_train,
        )
        freeze_by_patterns(model, patterns)
        return model

    def build_model_wm(checkpoint=None, *a, **kw):
        src = checkpoint
        wm_sd = None
        if isinstance(src, str) and os.path.isfile(src):
            raw = torch.load(src, map_location="cpu", weights_only=False)
            sd = raw.get("model_state_dict", raw)
            if isinstance(sd, dict) and _has_wm(sd):
                print(
                    "[WM-TrueDiag] checkpoint has WM keys; build BASE then load %s"
                    % src,
                    flush=True,
                )
                wm_sd = sd
                if not args.base_model or not os.path.isfile(args.base_model):
                    raise ValueError(
                        "A saved LoopSCF checkpoint requires --base-model pointing to its matching base checkpoint"
                    )
                src = args.base_model
        model = orig_build(src, *a, **kw)
        model = _install(model)
        if wm_sd is not None:
            model.load_state_dict(wm_sd, strict=True)
        return model

    nn_build.build_model = build_model_wm
    trainer_mod.build_model = build_model_wm
    dptb_nn.build_model = build_model_wm
    train_mod.build_model = build_model_wm

    train_mod.train(
        INPUT=args.input,
        init_model=(None if args.restart else args.init_model),
        restart=args.restart,
        train_soc=False,
        output=args.output,
        log_level=20,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
