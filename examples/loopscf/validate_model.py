"""Real-checkpoint GPU check: AO rotation, populations, and one update per arm."""

from pathlib import Path
import sys, json, time, gc

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input", required=True, help="Training config with readable real data"
)
parser.add_argument(
    "--checkpoint", required=True, help="Original base checkpoint without WM"
)
parser.add_argument("--output", required=True)
parser.add_argument("--indices", type=int, nargs=2, default=[0, 11])
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
stage = Path(args.output)
stage.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))
import torch

if torch.device(args.device).type != "cuda":
    parser.error("this real-model check requires a CUDA device")
torch.cuda.set_device(args.device)
import dptb
import dptb.nnops.loopscf as wm
import dptb.nnops.trainer as trainer_module

assert Path(dptb.__file__).resolve().is_relative_to(repo)
assert Path(wm.__file__).resolve().is_relative_to(repo)
torch.set_num_threads(6)
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.data.build import build_dataset
from dptb.data.dataloader import DataLoader
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nnops.loss import Loss
from dptb.utils.argcheck import normalize


def rotate_ao(feat, idp, rotation):
    from e3nn import o3
    from dptb.utils.constants import anglrMId
    import re

    # e3nn 0.4 builds Wigner generators on CPU; convert the small D blocks after.
    rotation = rotation.cpu()
    yzx = rotation.new_tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    basis_rotation = yzx @ rotation @ yzx.T
    out = feat.clone()
    for pair, sl in idp.orbpair_maps.items():
        ls = [anglrMId[re.findall(r"[a-zA-Z]", t)[0]] for t in pair.split("-")]
        dl, dr = [
            o3.Irrep(l, (-1) ** l).D_from_matrix(basis_rotation).to(feat) for l in ls
        ]
        block = feat[:, sl].reshape(-1, dl.shape[0], dr.shape[0])
        out[:, sl] = (dl @ block @ dr.T).flatten(1)
    return out


def rotated_batch(data, idp, rotation):
    result = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
    for k in (A.POSITIONS_KEY, A.CELL_KEY, A.EDGE_VECTORS_KEY):
        if k in result:
            result[k] = result[k] @ rotation.to(result[k]).T
    for k in (
        A.NODE_FEATURES_KEY,
        A.EDGE_FEATURES_KEY,
        A.NODE_H0_KEY,
        A.EDGE_H0_KEY,
        A.NODE_OVERLAP_KEY,
        A.EDGE_OVERLAP_KEY,
    ):
        if k in result:
            result[k] = rotate_ao(result[k], idp, rotation)
    return result


def manual_steps(model, data, K):
    state = model._wm_prepare(data)
    wm_n = wm_e = None
    outputs = []
    for k in range(1, K + 1):
        out = model._wm_one_k(data, k, wm_n, wm_e, state)
        outputs.append(
            (out[A.NODE_FEATURES_KEY].clone(), out[A.EDGE_FEATURES_KEY].clone())
        )
        wm_n, wm_e = model._wm_update(out, state)
    return outputs, state


cfg = normalize(json.loads(Path(args.input).read_text()))
common = dict(cfg["common_options"])
common["device"] = args.device
do = cfg["data_options"]
ds = build_dataset(
    **do["train"],
    r_max=do.get("r_max"),
    er_max=do.get("er_max"),
    oer_max=do.get("oer_max"),
    **common
)
items = [ds[i] for i in args.indices]
# Loop training recomputes reference bands at random k. Plot-path labels are
# ragged and not inputs to either WM or this training loss; retain them in ds.
batch = next(
    iter(
        DataLoader(
            dataset=items,
            batch_size=2,
            shuffle=False,
            exclude_keys=[A.ENERGY_EIGENVALUE_KEY, A.KPOINT_KEY],
        )
    )
)
ref = AtomicData.to_AtomicDataDict(batch.to(args.device))
wm.patch_stepwise_loss(K=2)
results = []
original_backward = torch.Tensor.backward
for mode in ("head", "moe"):
    torch.manual_seed(20260910)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(
        checkpoint=args.checkpoint,
        model_options=cfg["model_options"],
        common_options=common,
    ).to(args.device)
    idp = OrbitalMapper(common["basis"], method="e3tb", device=args.device)
    wm.install_working_memory_true_diag(
        model, mode=mode, idp=idp, K=2, collect_diagnostics=True
    )
    # Fresh base, v3 adapter state; deliberately nonzero to exercise feedback.
    for name, p in model.named_parameters():
        if "wm_" in name:
            torch.nn.init.normal_(p, std=0.01)
    sd = model.state_dict()
    model.load_state_dict(sd, strict=True)
    del sd
    wm.freeze_by_patterns(
        model, wm.ARM1_PATTERNS if mode == "head" else wm.ARM2_PATTERNS
    )
    model.eval()
    with torch.no_grad():
        invalid = dict(ref)
        invalid["nelec"] = ref["nelec"].reshape(-1)[:1]
        try:
            model._wm_prepare(invalid)
        except ValueError as exc:
            assert "electron count per graph" in str(exc)
        else:
            raise AssertionError("one electron count broadcast to different structures")
        state = model._wm_prepare(ref)
        ptr = state["ptr"]
        qs = [float(state["q0"][ptr[g] : ptr[g + 1]].sum()) for g in range(2)]
        for n, q in zip(ref["nelec"].reshape(-1).tolist(), qs):
            assert abs(n - q) < 1e-4, (n, q)
        out = model(dict(ref))
        out2 = model(dict(ref))
        for key in [A.NODE_FEATURES_KEY, A.EDGE_FEATURES_KEY]:
            torch.testing.assert_close(out[key], out2[key], rtol=1e-6, atol=1e-6)
        assert len(out["_loop_q"]) == 3
        for q in out["_loop_q"]:
            for g, n in enumerate(ref["nelec"].reshape(-1)):
                assert abs(float(q[ptr[g] : ptr[g + 1]].sum() - n)) < 1e-4
        from e3nn import o3

        rotation = o3.rand_matrix(dtype=torch.float64, device=args.device)
        rotated = rotated_batch(ref, idp, rotation)
        expected, rs = manual_steps(model, ref, 4)
        actual, _ = manual_steps(model, rotated, 4)
        rotation_errors = []
        for k, (pair, pair_r) in enumerate(zip(expected, actual), start=1):
            errors = []
            for original, got in zip(pair, pair_r):
                target = rotate_ao(original, idp, rotation)
                errors.append(
                    float((target - got).norm() / target.norm().clamp_min(1e-9))
                )
            rotation_errors.append(
                {"K": k, "node_relative": errors[0], "edge_relative": errors[1]}
            )
        print("ROTATION", json.dumps(rotation_errors), flush=True)
        # Float32 backbone has its own numerical equivariance floor.
        rotation_ok = (
            max(max(r["node_relative"], r["edge_relative"]) for r in rotation_errors)
            < 2e-3
        )
        # Collect the remaining diagnostics even when the inherited base fails.
        # Never turn a baseline symmetry failure into a successful validation.
        overlap = [
            {k: v.tolist() if torch.is_tensor(v) else v for k, v in d.items()}
            for d in out["_loop_overlap"]
        ]
        del rotated, expected, actual, rs
        del state, out, out2
    model.train()
    c = {
        k: v for k, v in common.items() if k in ("basis", "dtype", "overlap", "has_soc")
    }
    lossfunc = Loss(
        **cfg["train_options"]["loss_options"]["train"], device=args.device, **c
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    optimizer.zero_grad(set_to_none=True)
    saved = {
        key: ref[key].clone()
        for key in [
            A.NODE_FEATURES_KEY,
            A.EDGE_FEATURES_KEY,
            A.NODE_OVERLAP_KEY,
            A.EDGE_OVERLAP_KEY,
        ]
    }
    t = time.monotonic()
    loss, parts = wm.stepwise_train_loss(model, dict(ref), dict(ref), lossfunc)
    trainer_module.Trainer._backward_loss(None, loss)
    assert torch.Tensor.backward is original_backward
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6))
    assert grad_norm > 0
    for key, prior in saved.items():
        torch.testing.assert_close(prior, ref[key])
    optimizer.step()
    result = {
        "mode": mode,
        "rotation": rotation_errors,
        "rotation_ok": rotation_ok,
        "overlap": overlap,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "strict_load": True,
        "indices": args.indices,
        "nelec": ref["nelec"].reshape(-1).tolist(),
        "q_prior_sum": qs,
        "eval_repeatable": True,
        "loss": float(loss),
        "step_losses": parts,
        "finite_gradients": True,
        "grad_norm": grad_norm,
        "optimizer_step": True,
        "batch_unchanged": True,
        "seconds": time.monotonic() - t,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    results.append(result)
    print(json.dumps(result), flush=True)
    del model, optimizer, lossfunc, grads, saved, loss
    gc.collect()
    torch.cuda.empty_cache()
report = {
    "version": wm.CORRECTNESS_VERSION,
    "device": torch.cuda.get_device_name(),
    "tests": results,
    "success": all(r["rotation_ok"] for r in results),
    "limits": "Two diagnostic structures, nonzero random v3 adapters, K1-4 rotation and one update per arm; no generalization claim.",
}
(stage / "integration_gpu_results.json").write_text(json.dumps(report, indent=2))
print(
    "INTEGRATION_PASS" if report["success"] else "INTEGRATION_ROTATION_FAILED",
    flush=True,
)
raise SystemExit(0 if report["success"] else 1)
