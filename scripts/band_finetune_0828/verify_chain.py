#!/usr/bin/env python
"""End-to-end check of the band fine-tuning data chain.

Takes the label side only -- no model -- and asks one question:

    diagonalize (H0 + delta-H_label, S_parsed) on the record's own k-points;
    do we reproduce the cached DFT reference bands?

Agreement means the overlap parse, the H0 add-back, the fractional k-point
convention and the RME -> H(k) path are all consistent. Disagreement means the
band loss would be training against a mis-specified target, and nothing in the
training loop would say so.

Uses the repo's own Eigenvalues/HR2HK -- the same modules EigLoss calls -- so
what is verified here is the path training actually takes.
"""
import argparse
import json
import pickle
import sys

import lmdb
import numpy as np
import torch

REPO = "/data/wgh/0828_band_finetune/code/DeePTB_bandft"
sys.path.insert(0, REPO)

from dptb.data import AtomicDataDict  # noqa: E402
from dptb.data.transforms import OrbitalMapper  # noqa: E402
from dptb.nn.energy import Eigenvalues  # noqa: E402

BASIS_CFG = "/data/wgh/nacf_band_0826/configs/train_config.json"


def window_mae(pred, ref, nocc, lo=-10.0, hi=10.0):
    """fw_10: shift each side by its own VBM max, mask on ref, equal-weight MAE."""
    e_vbm_pred = pred[:, :nocc].max()
    e_vbm_ref = ref[:, :nocc].max()
    p = pred - e_vbm_pred
    r = ref - e_vbm_ref
    m = (r >= lo) & (r <= hi)
    if m.sum() == 0:
        return float("nan"), 0
    return float(np.abs(p[m] - r[m]).mean()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    basis = json.load(open(BASIS_CFG))["common_options"]["basis"]
    idp = OrbitalMapper(basis=basis, method="e3tb", device=args.device)
    idp.get_orbpair_maps()

    # float64 throughout: we are checking a convention, not a speed path, and
    # float32 storage noise (~1e-5 eV) would otherwise dominate the verdict.
    eig = Eigenvalues(
        idp=idp,
        h_edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        h_node_field=AtomicDataDict.NODE_FEATURES_KEY,
        h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
        out_field=AtomicDataDict.ENERGY_EIGENVALUE_KEY,
        s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
        s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
        s_out_field=AtomicDataDict.OVERLAP_KEY,
        dtype=torch.float64,
        device=args.device,
    )

    env = lmdb.open(args.lmdb, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n_total = txn.stat()["entries"]
    n = min(args.n, n_total) if args.n else n_total
    print("records: %d, checking %d" % (n_total, n), flush=True)

    rows = []
    for i in range(n):
        with env.begin() as txn:
            rec = pickle.loads(txn.get(i.to_bytes(4, "big")))
        cid = rec["case_id"]

        d = {
            AtomicDataDict.ATOMIC_NUMBERS_KEY: rec["atomic_numbers"].clone(),
            AtomicDataDict.EDGE_INDEX_KEY: rec["edge_index"].clone(),
            AtomicDataDict.EDGE_CELL_SHIFT_KEY: rec["edge_cell_shift"].to(torch.float64),
            AtomicDataDict.POSITIONS_KEY: rec["pos"].to(torch.float64),
            AtomicDataDict.CELL_KEY: rec["cell"].to(torch.float64),
            AtomicDataDict.PBC_KEY: rec["pbc"].clone(),
        }
        idp(d)
        # The stored target is the residual; the physical H is H0 + dH.
        d[AtomicDataDict.NODE_FEATURES_KEY] = (
            rec["node_h0"].to(torch.float64) + rec["node_features"].to(torch.float64))
        d[AtomicDataDict.EDGE_FEATURES_KEY] = (
            rec["edge_h0"].to(torch.float64) + rec["edge_features"].to(torch.float64))
        d[AtomicDataDict.NODE_OVERLAP_KEY] = rec["node_overlap"].to(torch.float64)
        d[AtomicDataDict.EDGE_OVERLAP_KEY] = rec["edge_overlap"].to(torch.float64)
        d[AtomicDataDict.KPOINT_KEY] = rec["kpoint"].to(torch.float64).reshape(-1, 3)
        d = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in d.items()}

        with torch.no_grad():
            out = eig(d)
        pred = out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].cpu().numpy()  # (nk, norb)
        ref = rec["eigenvalue"].numpy()                                    # (nk, nband)

        nb = min(pred.shape[1], ref.shape[1])
        nocc = int(np.ceil(rec["nelec"] / 2.0))
        p, r = pred[:, :nb], ref[:, :nb]

        absdiff = np.abs(p - r)
        fw, npts = window_mae(p, r, nocc)
        rows.append({
            "case_id": cid, "nk": int(p.shape[0]), "nband_used": int(nb),
            "norb_model": int(pred.shape[1]), "nband_ref": int(ref.shape[1]),
            "nocc": nocc,
            "max_abs_eV": float(absdiff.max()),
            "mean_abs_eV": float(absdiff.mean()),
            "fw10_mae_eV": fw, "fw10_points": npts,
        })
        print("[%3d/%3d] %-18s nk=%3d nb=%3d  max|d|=%.3e  mean|d|=%.3e  fw10=%.3e"
              % (i + 1, n, cid, p.shape[0], nb, absdiff.max(), absdiff.mean(), fw),
              flush=True)

    mx = np.array([r["max_abs_eV"] for r in rows])
    fw = np.array([r["fw10_mae_eV"] for r in rows])
    print("\n=== label-side chain check over %d structures ===" % len(rows))
    print("max|dE|   : median %.3e  worst %.3e eV" % (np.median(mx), mx.max()))
    print("fw10 MAE  : median %.3e  worst %.3e eV" % (np.median(fw), fw.max()))
    verdict = "PASS" if np.median(fw) < 1e-3 else "FAIL"
    print("verdict: %s (expect ~1e-5 eV float32 storage noise; >1e-3 means a "
          "convention is wrong)" % verdict)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"verdict": verdict, "rows": rows}, fh, indent=1)
        print("report ->", args.report)


if __name__ == "__main__":
    main()
