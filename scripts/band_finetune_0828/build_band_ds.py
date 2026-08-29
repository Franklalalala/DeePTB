#!/usr/bin/env python
"""Build the band fine-tuning LMDB: 138 structures + overlap S + DFT eigenvalues.

Each output record is the original h0dh_ds record (geometry, stored graph, H0,
delta-H target) plus four new fields:

    node_overlap / edge_overlap : overlap RMEs, row-aligned with the stored graph
    eigenvalue                  : DFT reference bands, (nk, nband)
    kpoint                      : fractional k-points, (nk, 3)
    nelec                       : electron count, for the Fermi window

Overlap comes from the ABACUS `srs1_nao.csr` of each case (S depends only on
geometry and basis, so the ceil-run copy is the same matrix as the ref run).
It is parsed with the repo's own `_abacus_parse`, never by hand -- the Ry->eV
factor, the ABACUS->DeePTB m reordering and the `i_j_Rx_Ry_Rz` key convention
all have to stay inherited.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import traceback

import ase.data
import h5py
import lmdb
import numpy as np
import torch

REPO = "/data/wgh/0828_band_finetune/code/DeePTB_bandft"
sys.path.insert(0, REPO)

from dptb.data.interfaces.abacus import _abacus_parse  # noqa: E402
from dptb.data.interfaces.ham_to_feature import block_to_feature  # noqa: E402
from dptb.data.transforms import OrbitalMapper  # noqa: E402
from dptb.data import AtomicDataDict  # noqa: E402

SRC_LMDB = "/data/wgh/deltah_wsd_band_0806/data138/h0dh_ds/data.0000.lmdb"
CEIL = "/data/wgh/deltah_wsd_band_0806/runner138/h0dh/{id}/ceil/OUT.ABACUS"
BANDS = "/data/wgh/deltah_wsd_band_0806/analysis138/bands_cache/{id}.npz"
BANDS_JSON = "/data/wgh/deltah_wsd_band_0806/analysis138/bands_cache/{id}.json"
BASIS_CFG = "/data/wgh/nacf_band_0826/configs/train_config.json"


# ABACUS v3.9.0.22 renamed things the stock parser still greps for in the old
# spelling. Rewriting them in a staged copy keeps the parser -- which carries
# the Ry->eV factor and the ABACUS->DeePTB m reordering -- byte-identical to
# the one the model was trained with.
#
#   "Lattice constant (Angstrom)"  <- was lower case
#   "Atom label ="                 <- was lower case
#   coordinate rows "Li  x y z"    <- were "tauc_Li1  x y z"
LOG_FIXUPS = (
    ("Lattice constant (Angstrom)", "lattice constant (Angstrom)"),
    ("Atom label =", "atom label ="),
)
COORD_HEADERS = ("CARTESIAN COORDINATES ( UNIT", "DIRECT COORDINATES")


def normalize_log(src_path, dst_path):
    """Rewrite a v3.9 log into the dialect the stock parser reads.

    Returns the element symbol sequence recovered from the coordinate block so
    the caller can cross-check it against the LMDB record's atomic numbers.
    """
    with open(src_path, "r") as fh:
        lines = fh.readlines()

    for old, new in LOG_FIXUPS:
        lines = [ln.replace(old, new) for ln in lines]

    nsites = None
    for ln in lines:
        if "TOTAL ATOM NUMBER" in ln:
            nsites = int(ln.split()[-1])
            break
    if nsites is None:
        raise ValueError("no TOTAL ATOM NUMBER in log")

    start = None
    for i, ln in enumerate(lines):
        if "K-POINTS" in ln:
            continue
        if any(h in ln for h in COORD_HEADERS):
            start = i
            break
    if start is None:
        raise ValueError("no atom coordinate header in log")
    if "atom" not in lines[start + 1]:
        raise ValueError("coordinate header not followed by 'atom' row")

    symbols, counts = [], {}
    for j in range(nsites):
        idx = start + 2 + j
        parts = lines[idx].split()
        if parts[0].startswith("tau"):        # already the old dialect
            symbols.append("".join(c for c in parts[0][5:] if c.isalpha()))
            continue
        sym = parts[0]
        counts[sym] = counts.get(sym, 0) + 1
        symbols.append(sym)
        # tauc_/taud_ prefix is 5 chars; the parser slices [5:] for the label.
        prefix = "tauc_" if "CARTESIAN" in lines[start] else "taud_"
        parts[0] = "%s%s%d" % (prefix, sym, counts[sym])
        lines[idx] = " " + "  ".join(parts) + "\n"

    with open(dst_path, "w") as fh:
        fh.writelines(lines)
    return symbols


def parse_overlap_blocks(case_id, workdir, expect_numbers):
    """Run the repo parser on a minimal staged folder; return the S block dict."""
    src = CEIL.format(id=case_id)
    folder = os.path.join(workdir, case_id)
    out = os.path.join(folder, "OUT.ABACUS")
    os.makedirs(out, exist_ok=True)
    symbols = normalize_log(os.path.join(src, "running_scf.log"),
                            os.path.join(out, "running_scf.log"))
    # The log's atom order must match the record's, or every S block lands on
    # the wrong atom pair without anything raising.
    got = [ase.data.atomic_numbers[s] for s in symbols]
    want = [int(z) for z in expect_numbers.flatten().tolist()]
    if got != want:
        raise ValueError("atom order mismatch: log=%s record=%s" % (got[:8], want[:8]))

    # LTS-outH0 names it srs1_nao.csr; the stock parser expects the upstream name.
    shutil.copy(os.path.join(src, "srs1_nao.csr"),
                os.path.join(out, "data-SR-sparse_SPIN0.csr"))

    parsed = os.path.join(workdir, "parsed_" + case_id)
    _abacus_parse(folder, parsed, "OUT.ABACUS", output_mode="conv",
                  get_Ham=False, get_overlap=True)
    with h5py.File(os.path.join(parsed, "overlaps.h5"), "r") as fid:
        grp = fid["0"]
        blocks = {k: np.array(grp[k]) for k in grp.keys()}
    shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(parsed, ignore_errors=True)
    return blocks


def overlap_rme(rec, blocks, idp):
    """Project S blocks onto the record's stored graph -> node/edge overlap RMEs."""
    data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: rec["atomic_numbers"].clone(),
        AtomicDataDict.EDGE_INDEX_KEY: rec["edge_index"].clone(),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: rec["edge_cell_shift"].clone(),
        AtomicDataDict.POSITIONS_KEY: rec["pos"].clone(),
        AtomicDataDict.CELL_KEY: rec["cell"].clone(),
        AtomicDataDict.PBC_KEY: rec["pbc"].clone(),
    }
    idp(data)  # fills atom_types / edge_types
    block_to_feature(data, idp, blocks=False, overlap_blocks=blocks,
                     missing_block_policy="zero")
    return (data[AtomicDataDict.NODE_OVERLAP_KEY],
            data[AtomicDataDict.EDGE_OVERLAP_KEY])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    basis = json.load(open(BASIS_CFG))["common_options"]["basis"]
    idp = OrbitalMapper(basis=basis, method="e3tb")

    src_env = lmdb.open(SRC_LMDB, readonly=True, lock=False, subdir=True)
    with src_env.begin() as txn:
        n_total = txn.stat()["entries"]
    n = args.limit if args.limit else n_total
    print("source records:", n_total, "-> building:", n, flush=True)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    dst_env = lmdb.open(args.out, map_size=1 << 40, subdir=True)

    workdir = tempfile.mkdtemp(prefix="bandds_")
    rows, ok, failed = [], 0, []
    try:
        for i in range(n):
            key = i.to_bytes(4, "big")
            with src_env.begin() as txn:
                rec = pickle.loads(txn.get(key))
            case_id = rec["case_id"]
            try:
                blocks = parse_overlap_blocks(case_id, workdir, rec["atomic_numbers"])
                node_ovp, edge_ovp = overlap_rme(rec, blocks, idp)

                z = np.load(BANDS.format(id=case_id))
                meta = json.load(open(BANDS_JSON.format(id=case_id)))
                eig = np.asarray(z["ref"], dtype=np.float64)      # (nk, nband)
                klist = np.asarray(z["klist"], dtype=np.float64)  # (nk, 3)
                assert eig.shape[0] == klist.shape[0], (eig.shape, klist.shape)

                rec["node_overlap"] = node_ovp.to(torch.float32)
                rec["edge_overlap"] = edge_ovp.to(torch.float32)
                rec["eigenvalue"] = torch.from_numpy(eig).to(torch.float32)
                rec["kpoint"] = torch.from_numpy(klist).to(torch.float32)
                rec["nelec"] = float(meta["nelec"])

                with dst_env.begin(write=True) as txn:
                    txn.put(key, pickle.dumps(rec))

                # Self-check: the on-site S diagonal must sit near 1 for
                # normalized NAOs, and a graph edge whose S block is entirely
                # missing means the parse did not cover the stored graph.
                zero_edges = int((edge_ovp.abs().sum(dim=1) == 0).sum())
                rows.append({
                    "j": i, "case_id": case_id,
                    "n_atoms": int(rec["pos"].shape[0]),
                    "n_edges": int(rec["edge_index"].shape[1]),
                    "n_S_blocks": len(blocks),
                    "zero_S_edges": zero_edges,
                    "node_ovp_absmean": float(node_ovp.abs().mean()),
                    "edge_ovp_absmean": float(edge_ovp.abs().mean()),
                    "nk": int(klist.shape[0]), "nband": int(eig.shape[1]),
                    "nelec": float(meta["nelec"]),
                    "eig_min": float(eig.min()), "eig_max": float(eig.max()),
                })
                ok += 1
                if i % 10 == 0 or args.limit:
                    print("[%3d/%3d] %s edges=%d S_blocks=%d zeroS=%d nk=%d nband=%d"
                          % (i + 1, n, case_id, rec["edge_index"].shape[1],
                             len(blocks), zero_edges, klist.shape[0], eig.shape[1]),
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((case_id, repr(exc)))
                print("FAILED %s: %r" % (case_id, exc), flush=True)
                traceback.print_exc()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        dst_env.close()
        src_env.close()

    print("\n=== built %d / %d, failed %d ===" % (ok, n, len(failed)))
    for c, e in failed:
        print("  FAIL", c, e)
    if rows:
        arr = np.array([r["zero_S_edges"] for r in rows])
        print("zero_S_edges: total=%d  max_per_struct=%d  structs_affected=%d"
              % (arr.sum(), arr.max(), int((arr > 0).sum())))
        print("nband range: %d..%d   nk range: %d..%d"
              % (min(r["nband"] for r in rows), max(r["nband"] for r in rows),
                 min(r["nk"] for r in rows), max(r["nk"] for r in rows)))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"ok": ok, "n": n, "failed": failed, "rows": rows}, fh, indent=1)
        print("report ->", args.report)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
