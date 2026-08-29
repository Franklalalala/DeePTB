#!/usr/bin/env python
"""Add overlap RMEs to the 12k training LMDB, one shard at a time.

Parses each case's ABACUS overlap CSR with the log-free adapter (verified
bitwise against `_abacus_parse` on the 138 set) and writes a new LMDB shard
carrying `node_overlap` / `edge_overlap` alongside everything the record
already had.

The check that matters is coverage. `block_to_feature(..., "zero")` fills any
graph edge with no matching S block with zeros -- which is correct for the
blocks ABACUS legitimately drops (|S|max < 1e-10), and catastrophic-but-silent
if the CSR belongs to a different geometry than the record. So every structure
records how many of its edges got no S block, and a shard whose rate looks
nothing like the 138-set baseline is reported rather than quietly written.

Shards are independent, so several copies of this can run in parallel over
disjoint --shard-range slices.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/data/wgh/0828_band_finetune/src")
from csr_to_rme import build_meta, overlap_rme, parse_overlap_csr  # noqa: E402

sys.path.insert(0, "/data/wgh/0828_band_finetune/code/DeePTB_bandft")
import lmdb  # noqa: E402
import torch  # noqa: E402
from dptb.data.transforms import OrbitalMapper  # noqa: E402

CSR_DIR = "/data/wgh/0829_S_relay/csr"
OVERRIDE_DIR = "/data/wgh/0829_S_relay/csr_train_override"
BASIS_CFG = "/data/wgh/0828_band_finetune/configs/finetune.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/wgh/h0res_20260815/h0res/train")
    ap.add_argument("--dst", default="/data/wgh/0829_train_with_S/train")
    ap.add_argument("--shard-range", default="0:240")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.shard_range.split(":"))
    os.makedirs(args.dst, exist_ok=True)
    basis = json.load(open(BASIS_CFG))["common_options"]["basis"]
    idp = OrbitalMapper(basis=basis, method="e3tb")
    orbital_types, norb_of = build_meta(basis)

    shards = sorted(glob.glob(os.path.join(args.src, "*.lmdb")))[lo:hi]
    print("shards %d..%d -> %d to process" % (lo, hi, len(shards)), flush=True)

    rows, n_done, n_missing = [], 0, 0
    t0 = time.time()
    for si, sh in enumerate(shards):
        out = os.path.join(args.dst, os.path.basename(sh))
        if os.path.exists(os.path.join(out, "data.mdb")):
            print("  %s exists, skip" % os.path.basename(sh), flush=True)
            continue
        src_env = lmdb.open(sh, readonly=True, lock=False, subdir=True)
        dst_env = lmdb.open(out + ".tmp", map_size=1 << 38, subdir=True)
        with src_env.begin() as txn:
            n = txn.stat()["entries"]
            for i in range(n):
                rec = pickle.loads(txn.get(i.to_bytes(4, "big")))
                cid = rec.get("case_id", "")
                # 264 case ids mean different structures in train vs test;
                # the shared dir holds test's copy, so train looks here first.
                csr = os.path.join(OVERRIDE_DIR, cid + ".csr")
                if not os.path.exists(csr):
                    csr = os.path.join(CSR_DIR, cid + ".csr")
                if not os.path.exists(csr):
                    n_missing += 1
                    print("  MISSING CSR for %s" % cid, flush=True)
                    continue
                el = [int(z) for z in np.asarray(rec["atomic_numbers"]).flatten().tolist()]
                sn = np.array([norb_of[z] for z in el])
                blocks, norb_csr = parse_overlap_csr(csr, el, orbital_types, sn)
                if blocks is None or norb_csr != int(sn.sum()):
                    # The CSR belongs to a different structure than the record.
                    print("  ORBITAL COUNT MISMATCH %s: csr=%d record=%d"
                          % (cid, norb_csr, int(sn.sum())), flush=True)
                    n_missing += 1
                    continue
                node, edge = overlap_rme(rec, blocks, idp)
                ne = int(np.asarray(rec["edge_index"]).shape[1])
                zero_edges = int((edge.abs().sum(dim=1) == 0).sum())
                rec["node_overlap"] = node.to(torch.float32)
                rec["edge_overlap"] = edge.to(torch.float32)
                with dst_env.begin(write=True) as w:
                    w.put(i.to_bytes(4, "big"), pickle.dumps(rec))
                rows.append({"case_id": cid, "n_edges": ne,
                             "zero_S_edges": zero_edges,
                             "zero_frac": zero_edges / max(ne, 1),
                             "n_blocks": len(blocks)})
                n_done += 1
        src_env.close(); dst_env.close()
        os.rename(out + ".tmp", out)
        if si % 10 == 0:
            el_min = (time.time() - t0) / 60
            print("  [%3d/%3d] %s  done=%d  %.1f min" %
                  (si + 1, len(shards), os.path.basename(sh), n_done, el_min), flush=True)

    zf = np.array([r["zero_frac"] for r in rows]) if rows else np.array([0.0])
    print("\n=== overlap integration ===")
    print("records written: %d   missing/mismatched: %d" % (n_done, n_missing))
    print("edges with no S block: median %.4f  p90 %.4f  max %.4f  (fraction of edges)"
          % (np.median(zf), np.percentile(zf, 90), zf.max()))
    bad = [r for r in rows if r["zero_frac"] > 0.5]
    print("structures with >50%% edges missing S: %d %s"
          % (len(bad), [r["case_id"] for r in bad[:5]]))
    print("\nreference: the 138-set built the original way had a median of "
          "about 1.5%% zero-S edges; a median far above that means the CSR and "
          "the record disagree about geometry.")
    if args.report:
        json.dump({"n": n_done, "missing": n_missing, "rows": rows},
                  open(args.report, "w"), indent=1)
        print("-> %s" % args.report)


if __name__ == "__main__":
    main()
