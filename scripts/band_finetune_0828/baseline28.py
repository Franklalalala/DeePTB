#!/usr/bin/env python
"""Starting-point baseline on the full held-out set, in the trainer's own units.

The earlier 0.065 figure was the MEDIAN over the FIRST 6 held-out structures,
while validation_loss is the MEAN over ALL 28. With a heavy-tailed per-structure
loss (train instantaneous values span 0.008 to 1.9e5) those two statistics are
not comparable. This recomputes the untuned checkpoint the way the trainer's
validation does -- every structure, arithmetic mean -- and also reports the
median and the per-structure spread so the tail is visible rather than hidden.
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

CKPT = "/data/wgh/deltah_wsd_band_0806/ckpt/dhwsd.nnenv.iter100000.metafix.pth"
cfg = json.load(open("/data/wgh/0828_band_finetune/configs/finetune.json"))
common = cfg["common_options"]
do = cfg["data_options"]
DEV = "cuda:0"

ds = build_dataset(**do["validation"], r_max=do.get("r_max"),
                   er_max=do.get("er_max"), oer_max=do.get("oer_max"), **common)
mc = dict(common); mc["device"] = DEV
model = build_model(checkpoint=CKPT, model_options=cfg["model_options"],
                    common_options=mc)
model.eval()

lo = dict(cfg["train_options"]["loss_options"]["validation"])
lo.update({k: v for k, v in common.items()
           if k in ("basis", "dtype", "overlap", "has_soc")})
lo["device"] = DEV
crit = Loss(**lo)

totals, hams, eigs = [], [], []
with torch.no_grad():
    for i, batch in enumerate(DataLoader(dataset=ds, batch_size=1, shuffle=False)):
        batch = batch.to(DEV)
        d = AtomicData.to_AtomicDataDict(batch)
        ref = d.copy()
        out = model(d)
        val = crit(out, ref)
        totals.append(float(val))
        hams.append(crit._last_parts["ham"])
        eigs.append(crit._last_parts["eig"])
        print("[%2d] total=%.4e ham=%.4e eig=%.4e" % (i, totals[-1], hams[-1], eigs[-1]),
              flush=True)

t = np.array(totals); h = np.array(hams); e = np.array(eigs)
print("\n=== untuned baseline, n=%d held-out structures ===" % len(t))
print("                    mean        median      min         max")
for name, v in (("total", t), ("ham", h), ("eig", e)):
    print("  %-6s %11.4e %11.4e %11.4e %11.4e"
          % (name, v.mean(), np.median(v), v.min(), v.max()))
print("\nmean/median ratio for total: %.1fx  (>2 means the mean is tail-driven)"
      % (t.mean() / np.median(t)))
print("top-3 structures by total: %s" % np.sort(t)[-3:][::-1])
print("\nvalidation_loss is the MEAN row -> compare arms against %.4f" % t.mean())
