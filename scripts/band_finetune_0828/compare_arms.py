#!/usr/bin/env python
"""Per-structure comparison: untuned baseline vs the fine-tuned arms.

The held-out mean is dominated by one structure (#10 contributes 2706 of the
28-structure mean 96.76), so an improvement in the mean could mean either
"the model got better everywhere" or "one catastrophic structure got patched".
Those two have completely different implications, and only a per-structure
table separates them.

Also reports fw_10 -- the reporting metric -- which the training objective only
correlates with, since the loss windows relative to the band bottom while fw_10
windows relative to the VBM.
"""
import json
import sys

import numpy as np
import torch

REPO = "/data/wgh/0828_band_finetune/code/DeePTB_bandft"
sys.path.insert(0, REPO)

from dptb.data import AtomicData, AtomicDataDict  # noqa: E402
from dptb.data.build import build_dataset  # noqa: E402
from dptb.data.dataloader import DataLoader  # noqa: E402
from dptb.nn.build import build_model  # noqa: E402
from dptb.nnops.loss import Loss  # noqa: E402

WS = "/data/wgh/0828_band_finetune"
ARMS = [
    ("base", "/data/wgh/deltah_wsd_band_0806/ckpt/dhwsd.nnenv.iter100000.metafix.pth"),
    ("B_lr1e-3", f"{WS}/out/prod_lr1e3/checkpoint/nnenv.latest_resumable.pth"),
    ("C_lr1e-5", f"{WS}/out/prod_lr1e5/checkpoint/nnenv.latest.pth"),
]
DEV = "cuda:0"

cfg = json.load(open(f"{WS}/configs/finetune.json"))
common = cfg["common_options"]
do = cfg["data_options"]

ds = build_dataset(**do["validation"], r_max=do.get("r_max"),
                   er_max=do.get("er_max"), oer_max=do.get("oer_max"), **common)

lo = dict(cfg["train_options"]["loss_options"]["validation"])
lo.update({k: v for k, v in common.items()
           if k in ("basis", "dtype", "overlap", "has_soc")})
lo["device"] = DEV
crit = Loss(**lo)


def fw10(pred, ref, nocc):
    p = pred - pred[:, :nocc].max()
    r = ref - ref[:, :nocc].max()
    m = (r >= -10.0) & (r <= 10.0)
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(p[m] - r[m]).mean())


results = {}
for name, ckpt in ARMS:
    mc = dict(common); mc["device"] = DEV
    model = build_model(checkpoint=ckpt, model_options=cfg["model_options"],
                        common_options=mc)
    model.eval()
    tot, fw = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset=ds, batch_size=1, shuffle=False):
            batch = batch.to(DEV)
            d = AtomicData.to_AtomicDataDict(batch)
            ref = d.copy()
            out = model(d)
            tot.append(float(crit(out, ref)))

            phys = crit._add_h0(out)
            sd = crit._solver_dict(phys, ref)
            eig = crit.eigen(sd)[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
            if eig.dim() == 3:
                eig = eig[0]
            ep = eig.cpu().numpy()
            er = ref[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
            er = (er[0] if er.is_nested else er).cpu().numpy()
            if er.ndim == 3:
                er = er[0]
            nb = min(ep.shape[1], er.shape[1])
            nelec = float(d["nelec"]) if "nelec" in d else 2.0 * (nb // 2)
            nocc = max(1, min(int(np.ceil(nelec / 2.0)), nb - 1))
            fw.append(fw10(ep[:, :nb], er[:, :nb], nocc))
    results[name] = {"total": np.array(tot), "fw10": np.array(fw)}
    print("%s done" % name, flush=True)
    del model
    torch.cuda.empty_cache()

b, B, C = results["base"], results["B_lr1e-3"], results["C_lr1e-5"]
print("\n idx | base_total   B_total     C_total   | base_fw10  B_fw10   C_fw10")
print("-" * 78)
for i in range(len(b["total"])):
    print("%4d | %10.4g %10.4g %10.4g | %8.4f %8.4f %8.4f"
          % (i, b["total"][i], B["total"][i], C["total"][i],
             b["fw10"][i], B["fw10"][i], C["fw10"][i]))

print("\n=== aggregate over %d held-out structures ===" % len(b["total"]))
print("%-10s %12s %12s %12s %12s" % ("", "total_mean", "total_median", "fw10_mean", "fw10_median"))
for name in ("base", "B_lr1e-3", "C_lr1e-5"):
    r = results[name]
    print("%-10s %12.4g %12.4g %12.4f %12.4f"
          % (name, r["total"].mean(), np.median(r["total"]),
             np.nanmean(r["fw10"]), np.nanmedian(r["fw10"])))

print("\n=== win/loss vs base, per structure ===")
for name in ("B_lr1e-3", "C_lr1e-5"):
    r = results[name]
    w_t = int((r["total"] < b["total"]).sum())
    w_f = int((r["fw10"] < b["fw10"]).sum())
    print("  %-10s total better on %2d/%d   fw10 better on %2d/%d"
          % (name, w_t, len(b["total"]), w_f, len(b["fw10"])))

print("\n=== excluding the worst baseline structure (idx %d) ===" % int(b["total"].argmax()))
keep = np.ones(len(b["total"]), dtype=bool)
keep[b["total"].argmax()] = False
for name in ("base", "B_lr1e-3", "C_lr1e-5"):
    r = results[name]
    print("  %-10s total_mean=%10.4g  fw10_mean=%8.4f"
          % (name, r["total"][keep].mean(), np.nanmean(r["fw10"][keep])))

json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()},
          open(f"{WS}/out/arm_comparison.json", "w"), indent=1)
print("\n-> %s/out/arm_comparison.json" % WS)
