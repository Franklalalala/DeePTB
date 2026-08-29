#!/usr/bin/env python
"""Unit-check the arm A loss before spending a training run on it.

Three things must hold, and each has a distinct failure meaning:

  1. mu is in the meV range.  A large |mu| means the label's energy zero is
     off, not that the gauge is doing work.
  2. the loss DROPS after the shift.  If it rises, S and dH are not aligned
     (wrong field, wrong width, wrong sign) -- the gauge is then actively
     harmful and nothing else downstream would tell us.
  3. self-consistency: feeding the label as the prediction must give mu ~ 0
     and a near-zero loss. This catches a mis-wired reference/prediction side.
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
BASE = "/data/wgh/deltah_wsd_band_0806/ckpt/dhwsd.nnenv.iter100000.metafix.pth"
DEV = "cuda:0"

cfg = json.load(open(f"{WS}/configs/armA_smoke.json"))
common = cfg["common_options"]
do = cfg["data_options"]

ds = build_dataset(**do["validation"], r_max=do.get("r_max"), er_max=do.get("er_max"),
                   oer_max=do.get("oer_max"), **common)
mc = dict(common); mc["device"] = DEV
model = build_model(checkpoint=BASE, model_options=cfg["model_options"], common_options=mc)
model.eval()

lo = dict(cfg["train_options"]["loss_options"]["train"])
lo.update({k: v for k, v in common.items() if k in ("basis", "dtype", "overlap", "has_soc")})
lo["device"] = DEV
crit = Loss(**lo)
print("criterion:", type(crit).__name__)

mus, gains, ratios = [], [], []
print("\n idx |        mu (eV) | ham_ungauged |   ham_gauged | gain")
print("-" * 66)
for i, batch in enumerate(DataLoader(dataset=ds, batch_size=1, shuffle=False)):
    batch = batch.to(DEV)
    d = AtomicData.to_AtomicDataDict(batch)
    ref = d.copy()
    with torch.no_grad():
        out = model(d)
        loss = crit(out, ref)
    p = crit._last_parts
    mus.append(p["mu"]); gains.append(p["gauge_gain"])
    ratios.append(p["ham"] / max(p["ham_ungauged"], 1e-30))
    print("%4d | %+14.6e | %12.6e | %12.6e | %.4f"
          % (i, p["mu"], p["ham_ungauged"], p["ham"], p["gauge_gain"]))
    if i >= 11:
        break

mu = np.array(mus); gn = np.array(gains)
print("\n=== gauge behaviour over %d structures ===" % len(mu))
print("  |mu|      : median %.3e  max %.3e eV" % (np.median(np.abs(mu)), np.abs(mu).max()))
print("  gauge gain: median %.4f  min %.4f  (>1 means the shift helped)"
      % (np.median(gn), gn.min()))

ok_mag = np.median(np.abs(mu)) < 0.1
ok_gain = (gn >= 0.999).all()
print("\n  [%s] |mu| is small (median < 0.1 eV)" % ("PASS" if ok_mag else "FAIL"))
print("  [%s] gauge never hurts (all gains >= 1)" % ("PASS" if ok_gain else "FAIL"))

# Self-consistency: label as its own prediction.
print("\n=== self-consistency: predict the label ===")
batch = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False))).to(DEV)
d = AtomicData.to_AtomicDataDict(batch)
ref = d.copy()
with torch.no_grad():
    loss_self = crit(d, ref)
p = crit._last_parts
print("  loss=%.6e  mu=%+.3e eV" % (float(loss_self), p["mu"]))
ok_self = float(loss_self) < 1e-6 and abs(p["mu"]) < 1e-6
print("  [%s] label-vs-label gives ~0 loss and ~0 mu" % ("PASS" if ok_self else "FAIL"))

print("\nVERDICT: %s" % ("arm A loss behaves correctly"
                         if (ok_mag and ok_gain and ok_self) else
                         "SOMETHING IS WRONG -- do not start training"))
