"""Bounded whole-stack annealing, graph-level exit learning, and gate calibration.

Uses explicit AO residual matrix supervision without inferred electron counts.
The canonical pretrained model is strictly loaded before adding the recurrence.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.data.build import build_dataset
from dptb.data.dataloader import DataLoader
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nnops.loopscf.stack import (
    install_stack_loop,
    graph_hamiltonian_losses,
    adaptive_objective,
)
from dptb.utils.argcheck import normalize


def batch_dict(batch):
    return AtomicData.to_AtomicDataDict(batch.to("cuda:0"))


def losses_for(out, ref, idp):
    return torch.stack(
        [graph_hamiltonian_losses(n, e, ref, idp) for n, e in out["_loop_preds"]], -1
    )


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hours", type=float, default=9)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=5e-4)
    ap.add_argument("--uniform-aux", type=float, default=0.2)
    ap.add_argument("--strategy", choices=["bptt", "detach"], default="bptt")
    ap.add_argument("--gate-calibration", type=int, default=100)
    ap.add_argument("--calibration-size", type=int, default=512)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--validation-limit", type=int, default=400)
    args = ap.parse_args()
    if (
        min(args.K, args.steps, args.batch_size) < 1
        or args.hours <= 0
        or not 0 <= args.uniform_aux <= 1
    ):
        ap.error("invalid budget or objective settings")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "history.jsonl").exists():
        raise RuntimeError(
            "run directory already contains a run; use a fresh directory"
        )
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = normalize(json.loads(Path(args.input).read_text()))
    common = dict(cfg["common_options"], device="cuda:0")
    data = cfg["data_options"]
    datasets = {}
    for split in ["train", "validation"]:
        datasets[split] = build_dataset(
            **data[split],
            r_max=data.get("r_max"),
            er_max=data.get("er_max"),
            oer_max=data.get("oer_max"),
            **common,
        )
    if args.calibration_size >= len(datasets["train"]):
        raise ValueError("calibration set consumes all training data")
    # Same fixed calibration partition across arms/seeds, independent of validation.
    partition = np.random.default_rng(20260910).permutation(len(datasets["train"]))
    cal_indices = partition[: args.calibration_size].tolist()
    train_indices = partition[args.calibration_size :].tolist()
    atomic_json(
        root / "partition.json",
        {"train": train_indices, "gate_calibration": cal_indices},
    )
    model = build_model(
        checkpoint=args.base_model,
        model_options=cfg["model_options"],
        common_options=common,
    ).cuda()
    idp = OrbitalMapper(common["basis"], method="e3tb", device="cuda:0")
    # RNG reset makes newly introduced bridge/gate tensors matched across depths.
    torch.manual_seed(args.seed)
    install_stack_loop(model, idp, K=args.K, strategy=args.strategy)
    groups = {"body": [], "bridge": [], "gate": []}
    initial_hashes = {k: hashlib.sha256() for k in groups}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = (
                "bridge"
                if ".stack_bridge." in name
                else "gate" if ".stack_exit." in name else "body"
            )
            groups[group].append(p)
            initial_hashes[group].update(name.encode())
            initial_hashes[group].update(
                p.detach().contiguous().cpu().numpy().tobytes()
            )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": p,
                "lr": args.lr * (10 if name == "bridge" else 1),
                "multiplier": 10 if name == "bridge" else 1,
            }
            for name, p in groups.items()
        ],
        weight_decay=0.01,
    )
    atomic_json(
        root / "protocol.json",
        dict(
            vars(args),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            train_count=len(train_indices),
            calibration_count=len(cal_indices),
            validation_count=len(datasets["validation"]),
            trainable_parameters={
                k: sum(p.numel() for p in v) for k, v in groups.items()
            },
            initial_trainable_sha256={
                k: v.hexdigest() for k, v in initial_hashes.items()
            },
            loss="per_graph_half_onsite_hopping_each_half_L1_RMSE_eV",
            h0_network="inverse_CG_from_AO",
            objective="(1-uniform_aux)*expected_loss+uniform_aux*uniform_loss-beta*entropy",
            config_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            source_sha256={
                str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in [
                    Path(__file__),
                    Path(sys.modules[install_stack_loop.__module__].__file__),
                ]
            },
        ),
    )
    atomic_json(root / "train_config.json", cfg)
    exclude = [A.KPOINT_KEY, A.ENERGY_EIGENVALUE_KEY]
    generator = torch.Generator().manual_seed(args.seed)

    def loader(indices, split="train", shuffle=False):
        subset = torch.utils.data.Subset(datasets[split], indices)
        return DataLoader(
            dataset=subset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            exclude_keys=exclude,
            generator=generator,
        )

    train_loader = loader(train_indices, shuffle=True)
    # Validation is RNG-neutral so matching training order does not depend on eval cadence.
    val_loader = loader(
        list(range(min(args.validation_limit, len(datasets["validation"])))),
        split="validation",
    )
    step = 0
    started = time.monotonic()

    def save(name, stage):
        target = root / (name + ".pth")
        tmp = target.with_suffix(".tmp")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "stage": stage,
                "stack_protocol": vars(args),
                "config": cfg,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            tmp,
        )
        tmp.replace(target)
        if name.startswith("step_"):
            periodic = sorted(root.glob("step_*.pth"))
            for old in periodic[:-3]:
                if old.name != "step_002000.pth":
                    old.unlink()

    def validate(tag):
        rng = torch.get_rng_state()
        crng = torch.cuda.get_rng_state_all()
        grng = generator.get_state()
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in val_loader:
                ref = batch_dict(batch)
                out = model(dict(ref))
                ls = losses_for(out, ref, idp)
                p = out["_exit_probabilities"]
                chosen = (p.cumsum(-1) < 0.5).sum(-1).clamp_max(args.K - 1)
                for loss, prob, c in zip(
                    ls.cpu().tolist(), p.cpu().tolist(), chosen.cpu().tolist()
                ):
                    rows.append(
                        {
                            "per_step_ev": loss,
                            "p": prob,
                            "exit_step": c + 1,
                            "exit_loss_ev": loss[c],
                        }
                    )
        atomic_json(
            root / ("validation_" + tag + ".json"),
            {
                "step": step,
                "results": rows,
                "mean_per_step_ev": np.mean(
                    [r["per_step_ev"] for r in rows], axis=0
                ).tolist(),
                "mean_exit_loss_ev": float(np.mean([r["exit_loss_ev"] for r in rows])),
            },
        )
        torch.set_rng_state(rng)
        torch.cuda.set_rng_state_all(crng)
        generator.set_state(grng)
        model.train()

    validate("initial")
    started = time.monotonic()
    history = (root / "history.jsonl").open("a", buffering=1)
    done = False
    while not done:
        for batch in train_loader:
            elapsed = time.monotonic() - started
            if step >= args.steps or elapsed >= args.hours * 3600:
                done = True
                break
            tick = time.monotonic()
            step += 1
            progress = max(step / args.steps, elapsed / (args.hours * 3600))
            warm = min(1.0, step / 30)
            factor = warm * (
                0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
            )
            for g in optimizer.param_groups:
                g["lr"] = args.lr * g["multiplier"] * factor
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ref = batch_dict(batch)
            out = model(dict(ref))
            ls = losses_for(out, ref, idp)
            expected, ent = adaptive_objective(
                ls, out["_exit_probabilities"], args.beta
            )
            total = (
                (1 - args.uniform_aux) * expected
                + args.uniform_aux * ls.mean()
                - args.uniform_aux * args.beta * ent.mean()
            )
            if not bool(torch.isfinite(total)):
                raise RuntimeError("nonfinite task loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            digest = hashlib.sha256()
            for key in [A.ATOM_TYPE_KEY, A.POSITIONS_KEY, A.EDGE_INDEX_KEY]:
                digest.update(ref[key].detach().contiguous().cpu().numpy().tobytes())
            row = {
                "step": step,
                "seconds": time.monotonic() - tick,
                "elapsed": time.monotonic() - started,
                "graph_hash": digest.hexdigest(),
                "loss": float(total.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "per_step_ev": ls.detach().mean(0).cpu().tolist(),
                "p": out["_exit_probabilities"].detach().mean(0).cpu().tolist(),
                "entropy": float(ent.detach().mean()),
                "grad_norm": float(norm),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "graphs": ls.shape[0],
                "counts": out["_stack_counts"],
            }
            history.write(json.dumps(row) + "\n")
            if step <= 3 or step % 10 == 0:
                print("TRAIN", json.dumps(row), flush=True)
            if step % args.checkpoint_every == 0:
                save("step_%06d" % step, "joint")
    save("joint_final", "joint")
    validate("joint_final")
    # Stage II uses a separate slice of the original TRAIN split. No validation labels.
    if args.K > 1 and args.gate_calibration > 0 and cal_indices:
        for name, p in model.named_parameters():
            p.requires_grad_(".stack_exit." in name)
        model.eval()
        gate_opt = torch.optim.AdamW(groups["gate"], lr=1e-3, weight_decay=0)
        cal_loader = loader(cal_indices, shuffle=True)
        calibration = []
        for i, batch in enumerate(cal_loader):
            if i >= args.gate_calibration:
                break
            ref = batch_dict(batch)
            gate_opt.zero_grad(set_to_none=True)
            out = model(dict(ref))
            ls = losses_for(out, ref, idp).detach()
            # Reviewer/Ouro retrospective gain I_t=max(0,L_(t-1)-L_t), at t>=2.
            # Loss units here are eV, so gamma/slope are explicitly adapted.
            gain = (ls[:, :-1] - ls[:, 1:]).clamp_min(0)
            target = torch.sigmoid(5000 * (gain - 0.0002))
            gate_logits = out["_stack_logits"][:, 1:-1]
            if gate_logits.shape[1] == 0:
                break
            offsets = gate_logits.new_tensor(
                [math.log(args.K - t - 1) for t in range(1, args.K - 1)]
            )
            # Continuation logit is minus the exit-hazard logit. Final hazard is unused.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                -(gate_logits - offsets), target[:, :-1]
            )
            loss.backward()
            gate_opt.step()
            calibration.append(float(loss.detach()))
        atomic_json(
            root / "calibration.json",
            {
                "losses": calibration,
                "gamma_ev": 0.0002,
                "slope_per_ev": 5000,
                "target": "retrospective positive loss improvement; first/final hazard not calibrated",
            },
        )
        save("calibrated_final", "gate_calibrated")
        validate("calibrated_final")
    atomic_json(
        root / "complete.json",
        {
            "steps": step,
            "training_budget_hours": args.hours,
            "completed_at": time.time(),
        },
    )
    print("ANNEAL_COMPLETE", step, flush=True)


if __name__ == "__main__":
    main()
