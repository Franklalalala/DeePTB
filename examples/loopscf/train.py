#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train non-SOC LoopSCF with head or MoE adaptation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import hashlib
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("DPTB_REPO_ROOT", os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)


def _has_wm(sd):
    return any(("wm_node" in k) or ("wm_edge" in k) or ("latent_core" in k) for k in sd)


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
    ap.add_argument("--architecture", choices=("scalar", "latent"), default="scalar")
    ap.add_argument(
        "--latent-strategy", choices=("bptt", "detach", "reset"), default="bptt"
    )
    ap.add_argument("--n-k-train", type=int, default=5)
    ap.add_argument(
        "--latent-readout", choices=("frozen", "residual"), default="residual"
    )
    ap.add_argument(
        "--trace-batches",
        action="store_true",
        help="Record graph and random-k hashes to verify matched arms",
    )
    ap.add_argument(
        "--loader-workers",
        type=int,
        default=1,
        help="Workers per loader; 0 is for diagnostics",
    )
    ap.add_argument(
        "--no-feedback",
        action="store_true",
        help="Train a matched control with identical repeated forwards and no WM injection",
    )
    args = ap.parse_args()
    if args.architecture == "latent" and (args.no_feedback or args.mode != "head"):
        ap.error("latent uses --mode head and --latent-strategy, not --no-feedback")
    if args.K < 1 or args.n_k_train < 1:
        ap.error("--K and --n-k-train must be positive")
    if args.loader_workers < 0:
        ap.error("--loader-workers must be nonnegative")
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
    from dptb.nnops.loopscf import CORRECTNESS_VERSION

    try:
        revision = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except OSError:
        revision = None
    protocol = {
        "architecture": args.architecture,
        "latent_readout": (
            args.latent_readout if args.architecture == "latent" else None
        ),
        "latent_strategy": (
            args.latent_strategy if args.architecture == "latent" else None
        ),
        "version": CORRECTNESS_VERSION,
        "mode": args.mode,
        "K": args.K,
        "feedback": not args.no_feedback,
        "n_k_train": args.n_k_train,
        "source_revision": (
            revision.stdout.strip()
            if revision is not None and revision.returncode == 0
            else None
        ),
        "torch_version": torch.__version__,
        "init_model": args.init_model,
        "restart": args.restart,
        "base_model": args.base_model,
        "spectral_metric": "legacy_vbm_aligned_fw10",
        "occupation": (
            "not_applicable"
            if args.architecture == "latent"
            else "global_zero_temperature"
        ),
        "overlap_cutoff": 1e-5,
        "physical_h0_basis": "ao",
        "network_h0_basis": "inverse_cg_rme",
        "loader_workers": args.loader_workers,
        "trace_batches": args.trace_batches,
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    }
    protocol_sources = list((Path(REPO) / "dptb/nnops/loopscf").glob("*.py"))
    protocol_sources += [Path(REPO) / "dptb/nnops/trainer.py"]
    protocol["source_sha256"] = {
        str(p.resolve().relative_to(Path(REPO).resolve())): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in protocol_sources
    }
    protocol["launcher_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep each invocation, including restarts, alongside the training config.
    import time

    invocation = time.time_ns()

    (output / ("loopscf_protocol_%d.json" % invocation)).write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    basis = cfg["common_options"]["basis"]
    orig_build = nn_build.build_model
    patterns = ARM1_PATTERNS if args.mode == "head" else ARM2_PATTERNS

    # Single Trainer currently omits worker kwargs. Apply them explicitly here
    # to all three loaders; this launcher uses the same worker policy per split.
    from functools import partial

    loader_kwargs = dict(num_workers=args.loader_workers, pin_memory=True)
    if args.loader_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    trainer_mod.DataLoader = partial(trainer_mod.DataLoader, **loader_kwargs)

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
        if args.architecture == "latent":
            from dptb.nnops.loopscf.latent import install_latent_corrector

            install_latent_corrector(
                model,
                idp,
                K=args.K,
                strategy=args.latent_strategy,
                n_k_train=args.n_k_train,
                readout=args.latent_readout,
            )
        else:
            install_working_memory_true_diag(
                model,
                mode=args.mode,
                idp=idp,
                h2k=h2k,
                s2k=s2k,
                K=args.K,
                n_k_train=args.n_k_train,
                feedback=not args.no_feedback,
            )
            freeze_by_patterns(model, patterns)
        trainable = {
            name: list(p.shape)
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        (output / ("trainable_%d.json" % invocation)).write_text(
            json.dumps(trainable, indent=2)
        )
        if args.trace_batches:
            original_prepare = model._wm_prepare
            trace_path = output / ("batch_trace_%d.jsonl" % invocation)
            trace_index = 0

            def traced_prepare(batch):
                nonlocal trace_index
                state = original_prepare(batch)
                if model.training:

                    def digest(keys):
                        h = hashlib.sha256()
                        for key, value in keys:
                            t = value.detach().cpu().contiguous()
                            h.update(str((key, str(t.dtype), tuple(t.shape))).encode())
                            h.update(t.numpy().tobytes())
                        return h.hexdigest()

                    graph_keys = (
                        AtomicDataDict.POSITIONS_KEY,
                        AtomicDataDict.CELL_KEY,
                        AtomicDataDict.ATOM_TYPE_KEY,
                        AtomicDataDict.EDGE_INDEX_KEY,
                        AtomicDataDict.EDGE_CELL_SHIFT_KEY,
                        "nelec",
                    )
                    row = {
                        "batch": trace_index,
                        "graph_sha256": digest((k, batch[k]) for k in graph_keys),
                        "kpoints_sha256": digest([("kpts", state["kpts"])]),
                        "n_atoms": state["n_atom"],
                    }
                    with trace_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    trace_index += 1
                return state

            model._wm_prepare = traced_prepare
        return model

    def build_model_wm(checkpoint=None, *a, **kw):
        src = checkpoint
        wm_sd = None
        if isinstance(src, str) and os.path.isfile(src):
            raw = torch.load(src, map_location="cpu", weights_only=False)
            sd = raw.get("model_state_dict", raw)
            if isinstance(sd, dict) and _has_wm(sd):
                is_latent = any("latent_core" in k for k in sd)
                if is_latent != (args.architecture == "latent"):
                    raise ValueError(
                        "cannot load adapters from a different LoopSCF architecture"
                    )
                if is_latent:
                    from dptb.nnops.loopscf.latent import validate_latent_checkpoint

                    validate_latent_checkpoint(
                        sd, args.latent_strategy, args.latent_readout
                    )
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
        if args.architecture == "latent":
            # Record the actual starting weights, including a strict-loaded
            # adapter checkpoint, rather than its temporary construction init.
            initial = {
                name: hashlib.sha256(
                    p.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            (output / ("initial_trainable_%d.json" % invocation)).write_text(
                json.dumps(initial, indent=2)
            )
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
