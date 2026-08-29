#!/usr/bin/env python
"""Split the 138-structure band set into fine-tune / held-out shards.

These 138 are the pretrained arm's own *test* split, so every structure spent
on fine-tuning is one that can no longer measure it. The split is strided
(every 5th record held out) rather than a prefix, so both sides cover the same
range of cell sizes and band counts.

Also reports where the Fermi level sits once each structure's bands are shifted
by their own minimum -- that is the coordinate EigLoss's energy window uses,
which is NOT the same as the fw_10 convention (VBM-relative).
"""
import argparse
import json
import os
import pickle
import shutil

import lmdb
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--holdout-stride", type=int, default=5)
    args = ap.parse_args()

    src = lmdb.open(args.src, readonly=True, lock=False, subdir=True)
    with src.begin() as txn:
        n = txn.stat()["entries"]

    train_idx = [i for i in range(n) if i % args.holdout_stride != 0]
    valid_idx = [i for i in range(n) if i % args.holdout_stride == 0]
    print("total %d -> train %d / holdout %d" % (n, len(train_idx), len(valid_idx)))

    stats, manifest = [], {"train": [], "valid": []}
    for name, idxs in (("ft_train", train_idx), ("ft_valid", valid_idx)):
        root = os.path.join(args.outdir, name)
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root)
        dst = lmdb.open(os.path.join(root, "data.0000.lmdb"), map_size=1 << 40, subdir=True)
        for j, i in enumerate(idxs):
            with src.begin() as txn:
                rec = pickle.loads(txn.get(i.to_bytes(4, "big")))
            with dst.begin(write=True) as txn:
                txn.put(j.to_bytes(4, "big"), pickle.dumps(rec))

            eig = rec["eigenvalue"].numpy()
            nocc = int(np.ceil(rec["nelec"] / 2.0))
            nb = eig.shape[1]
            shifted = eig - eig.min()
            e_fermi_rel = float(shifted[:, :min(nocc, nb) - 1].max()) if nb >= 2 else float("nan")
            stats.append({"case_id": rec["case_id"], "split": name, "nocc": nocc,
                          "nband": nb, "e_vbm_rel": e_fermi_rel,
                          "espan": float(shifted.max())})
            manifest[name.split("_")[1]].append(rec["case_id"])
        dst.close()
        print("  %s -> %s (%d records)" % (name, root, len(idxs)))
    src.close()

    vbm = np.array([s["e_vbm_rel"] for s in stats if np.isfinite(s["e_vbm_rel"])])
    span = np.array([s["espan"] for s in stats])
    print("\n=== bottom-relative energies (the coordinate EigLoss windows in) ===")
    print("VBM - E_min : median %.1f  p10 %.1f  p90 %.1f  max %.1f eV"
          % (np.median(vbm), np.percentile(vbm, 10), np.percentile(vbm, 90), vbm.max()))
    print("full span   : median %.1f  max %.1f eV" % (np.median(span), span.max()))
    print("suggested band_emax = VBM_p90 + 10 = %.0f eV  (emin = 0)"
          % (np.percentile(vbm, 90) + 10))

    with open(os.path.join(args.outdir, "split_manifest.json"), "w") as fh:
        json.dump({"holdout_stride": args.holdout_stride,
                   "n_train": len(train_idx), "n_valid": len(valid_idx),
                   "vbm_rel_p90": float(np.percentile(vbm, 90)),
                   "manifest": manifest, "stats": stats}, fh, indent=1)
    print("manifest ->", os.path.join(args.outdir, "split_manifest.json"))


if __name__ == "__main__":
    main()
