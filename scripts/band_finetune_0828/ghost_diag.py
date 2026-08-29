#!/usr/bin/env python
"""Step 0 of the NextHAM plan: how much of our band error is ghost states?

This decides whether the k-space term is worth building at all. NextHAM's own
ablation says the k-space loss barely moves mean H accuracy (1.615 -> 1.417 meV
Gauge MAE) -- what it buys is the suppression of isolated blow-ups at a few
k-points. If our baseline does not have those, there is nothing for it to fix
and we should only run arm A (mu-gauged hamil_abs).

Per structure, on the baseline checkpoint:
  * diagonalize (H0 + dH_pred, S) at every stored k-point
  * compute the fw_10 error PER k-point (not pooled)
  * ghost score = max_k / median_k

A structure counts as ghosted when one k-point is dramatically worse than the
typical one. Needs no S transfer and no training -- band138.lmdb already has
everything.
"""
import argparse
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


def fw10_per_k(pred, ref, nocc, lo=-10.0, hi=10.0):
    """fw_10 error for each k-point separately.

    Pooling over k is what hides a ghost: one bad k-point among 200 barely
    moves the pooled mean, which is exactly the failure mode the k-space term
    targets.
    """
    e_p = pred[:, :nocc].max()
    e_r = ref[:, :nocc].max()
    p, r = pred - e_p, ref - e_r
    out = []
    for ik in range(r.shape[0]):
        m = (r[ik] >= lo) & (r[ik] <= hi)
        out.append(float(np.abs(p[ik][m] - r[ik][m]).mean()) if m.sum() else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["train", "validation", "both"])
    ap.add_argument("--ratio-thresh", type=float, default=5.0)
    ap.add_argument("--abs-thresh", type=float, default=0.3,
                    help="worst-k fw_10 (eV) above which a spike counts as real")
    ap.add_argument("--out", default=f"{WS}/out/ghost_diag.json")
    args = ap.parse_args()

    cfg = json.load(open(f"{WS}/configs/finetune.json"))
    common = cfg["common_options"]
    do = cfg["data_options"]
    mc = dict(common); mc["device"] = DEV
    model = build_model(checkpoint=BASE, model_options=cfg["model_options"],
                        common_options=mc)
    model.eval()

    lo = dict(cfg["train_options"]["loss_options"]["validation"])
    lo.update({k: v for k, v in common.items()
               if k in ("basis", "dtype", "overlap", "has_soc")})
    lo["device"] = DEV
    crit = Loss(**lo)

    splits = ["train", "validation"] if args.split == "both" else [args.split]
    rows = []
    for sp in splits:
        ds = build_dataset(**do[sp], r_max=do.get("r_max"), er_max=do.get("er_max"),
                           oer_max=do.get("oer_max"), **common)
        for i, batch in enumerate(DataLoader(dataset=ds, batch_size=1, shuffle=False)):
            batch = batch.to(DEV)
            d = AtomicData.to_AtomicDataDict(batch)
            ref_batch = d.copy()
            with torch.no_grad():
                out = model(d)
                phys = crit._add_h0(out)
                sd = crit._solver_dict(phys, ref_batch)
                eig = crit.eigen(sd)[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
            if eig.dim() == 3:
                eig = eig[0]
            ep = eig.cpu().numpy()
            er = ref_batch[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
            er = (er[0] if er.is_nested else er).cpu().numpy()
            if er.ndim == 3:
                er = er[0]
            nb = min(ep.shape[1], er.shape[1])
            nelec = float(d.get("nelec", 2.0 * (nb // 2)))
            nocc = max(1, min(int(np.ceil(nelec / 2.0)), nb - 1))

            per_k = fw10_per_k(ep[:, :nb], er[:, :nb], nocc)
            per_k = per_k[np.isfinite(per_k)]
            if per_k.size < 3:
                continue
            med, mx = float(np.median(per_k)), float(per_k.max())
            ratio = mx / max(med, 1e-12)
            rows.append({
                "split": sp, "idx": i, "nk": int(per_k.size),
                "fw10_median_k": med, "fw10_max_k": mx, "ratio": ratio,
                "fw10_pooled": float(np.mean(per_k)),
                "ghost": bool(ratio >= args.ratio_thresh and mx >= args.abs_thresh),
            })
            if i % 20 == 0:
                print("  [%s %3d] med=%.4f max=%.4f ratio=%.1f" % (sp, i, med, mx, ratio),
                      flush=True)

    r = np.array([x["ratio"] for x in rows])
    g = np.array([x["ghost"] for x in rows])
    mx = np.array([x["fw10_max_k"] for x in rows])
    md = np.array([x["fw10_median_k"] for x in rows])

    print("\n=== ghost diagnosis over %d structures ===" % len(rows))
    print("max_k / median_k ratio : median %.2f  p90 %.2f  max %.2f"
          % (np.median(r), np.percentile(r, 90), r.max()))
    print("worst-k fw_10 (eV)     : median %.4f  p90 %.4f  max %.4f"
          % (np.median(mx), np.percentile(mx, 90), mx.max()))
    print("typical-k fw_10 (eV)   : median %.4f" % np.median(md))
    print("\nGHOST RATE = %d/%d = %.1f%%  (ratio>=%.1f and worst-k>=%.2f eV)"
          % (g.sum(), len(g), 100.0 * g.mean(), args.ratio_thresh, args.abs_thresh))
    print("\nverdict: %s" % (
        "ghost is a real failure mode -> the k-space term has a target"
        if g.mean() >= 0.05 else
        "ghost is rare -> k-space term has little to fix; run arm A only"))

    json.dump({"ratio_thresh": args.ratio_thresh, "abs_thresh": args.abs_thresh,
               "ghost_rate": float(g.mean()), "n": len(rows), "rows": rows},
              open(args.out, "w"), indent=1)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
