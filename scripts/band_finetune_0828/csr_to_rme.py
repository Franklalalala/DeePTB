#!/usr/bin/env python
"""Parse an ABACUS overlap CSR into DeePTB RME blocks without the SCF log.

`_abacus_parse` reads running_scf.log to learn nsites / element /
site_norbits / orbital_types, then slices the CSR into atom-pair blocks and
applies the ABACUS->DeePTB m-reordering. We only shipped the CSR files, and
shipping 17k logs would be another ~6 GB. But every one of those quantities is
already determined by the record's atomic_numbers plus the basis, so the
metadata can be rebuilt locally.

What is NOT rebuilt is the physics: the block slicing, the `i_j_Rx_Ry_Rz` key
convention and `U_orbital.transform` are taken from the repo, unchanged. This
module only replaces where the metadata comes from.

Correctness is established by cross-check, not by argument: the 138-structure
set already has overlap RMEs produced the original way (log + _abacus_parse),
so `--verify` reruns those through this path and requires a bitwise match.
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

REPO = "/data/wgh/0828_band_finetune/code/DeePTB_bandft"
sys.path.insert(0, REPO)

import lmdb  # noqa: E402
import torch  # noqa: E402
from dptb.data.interfaces.abacus import OrbAbacus2DeepTB  # noqa: E402
from dptb.data.interfaces.ham_to_feature import block_to_feature  # noqa: E402
from dptb.data.transforms import OrbitalMapper  # noqa: E402
from dptb.data import AtomicDataDict  # noqa: E402
import ase.data  # noqa: E402

U_ORBITAL = OrbAbacus2DeepTB()
_LSYM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def basis_to_orbital_types(basis_str):
    """'4s2p2d1f' -> [0,0,0,0, 1,1, 2,2, 3]  (ABACUS orders by L, then zeta)."""
    out, num = [], ""
    for ch in basis_str:
        if ch.isdigit():
            num += ch
        else:
            out.extend([_LSYM[ch]] * int(num or 1))
            num = ""
    return out


def build_meta(basis):
    """Per-element orbital types and orbital counts, keyed by atomic number."""
    orbital_types, norbits = {}, {}
    for sym, bstr in basis.items():
        z = ase.data.atomic_numbers[sym]
        ls = basis_to_orbital_types(bstr)
        orbital_types[z] = ls
        norbits[z] = int(sum(2 * l + 1 for l in ls))
    return orbital_types, norbits


def parse_overlap_csr(path, element, orbital_types_dict, site_norbits):
    """Slice the CSR into atom-pair blocks, with the repo's m-reordering.

    Mirrors `_abacus_parse.parse_matrix` for the non-spinful overlap case:
    same key convention, same 1e-10 drop rule (ABACUS omits blocks whose max
    magnitude is below it), same U_orbital transform.
    """
    nsites = len(element)
    cumsum = np.cumsum(site_norbits)
    blocks = {}
    with open(path) as f:
        line = f.readline()
        if "Matrix Dimension of" not in line:
            line = f.readline()
            assert "Matrix Dimension of" in line, "unexpected CSR header in %s" % path
        norbits = int(line.split()[-1])
        expected = int(np.sum(site_norbits))
        if norbits != expected:
            # This CSR belongs to a different structure. Bail before slicing.
            return None, norbits
        f.readline()                              # "Matrix number of ..."
        for line in f:
            parts = line.split()
            if len(parts) == 0:
                break
            if int(parts[3]) == 0:
                continue
            R = np.array(parts[:3]).astype(int)
            vals = np.array(f.readline().split()).astype(np.float32)
            cols = np.array(f.readline().split()).astype(int)
            rowptr = np.array(f.readline().split()).astype(np.int32)
            dense = csr_matrix((vals, cols, rowptr), shape=(norbits, norbits),
                               dtype=np.float32).toarray()
            for i in range(nsites):
                for j in range(nsites):
                    mat = dense[cumsum[i] - site_norbits[i]:cumsum[i],
                                cumsum[j] - site_norbits[j]:cumsum[j]]
                    if abs(mat).max() < 1e-10:
                        continue
                    mat = U_ORBITAL.transform(mat, orbital_types_dict[element[i]],
                                              orbital_types_dict[element[j]])
                    blocks["%d_%d_%d_%d_%d" % (i, j, R[0], R[1], R[2])] = mat
    return blocks, norbits


def _t(x, dtype=None):
    """Records come as torch tensors in one dataset and numpy in another."""
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return t.to(dtype) if dtype is not None else t


def overlap_rme(rec, blocks, idp):
    data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: _t(rec["atomic_numbers"], torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: _t(rec["edge_index"], torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: _t(rec["edge_cell_shift"], torch.get_default_dtype()),
        AtomicDataDict.POSITIONS_KEY: _t(rec["pos"], torch.get_default_dtype()),
        AtomicDataDict.CELL_KEY: _t(rec["cell"], torch.get_default_dtype()),
        AtomicDataDict.PBC_KEY: _t(rec["pbc"], torch.bool),
    }
    idp(data)
    block_to_feature(data, idp, blocks=False, overlap_blocks=blocks,
                     missing_block_policy="zero")
    return (data[AtomicDataDict.NODE_OVERLAP_KEY],
            data[AtomicDataDict.EDGE_OVERLAP_KEY])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rerun the 138 set and require a bitwise match")
    ap.add_argument("--csr-dir", default="/data/wgh/0829_S_relay/csr")
    ap.add_argument("--basis-cfg",
                    default="/data/wgh/0828_band_finetune/configs/finetune.json")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    basis = json.load(open(args.basis_cfg))["common_options"]["basis"]
    idp = OrbitalMapper(basis=basis, method="e3tb")
    orbital_types, norb_of = build_meta(basis)

    if args.verify:
        src = "/data/wgh/0828_band_finetune/data/band138.lmdb"
        env = lmdb.open(src, readonly=True, lock=False, subdir=True)
        n_ok = n_bad = n_skip = 0
        worst = 0.0
        with env.begin() as txn:
            total = txn.stat()["entries"]
            for i in range(min(args.n, total)):
                rec = pickle.loads(txn.get(i.to_bytes(4, "big")))
                cid = rec["case_id"]
                csr = os.path.join(args.csr_dir, cid + ".csr")
                if not os.path.exists(csr):
                    # The 138 set's own ABACUS reruns are on this box already,
                    # so the adapter can be validated before the bulk pull ends.
                    alt = ("/data/wgh/deltah_wsd_band_0806/runner138/h0dh/"
                           "%s/ceil/OUT.ABACUS/srs1_nao.csr" % cid)
                    if os.path.exists(alt):
                        csr = alt
                    else:
                        n_skip += 1
                        continue
                el = [int(z) for z in rec["atomic_numbers"].flatten().tolist()]
                sn = np.array([norb_of[z] for z in el])
                blocks, _ = parse_overlap_csr(csr, el, orbital_types, sn)
                node, edge = overlap_rme(rec, blocks, idp)
                dn = (node - rec["node_overlap"]).abs().max().item()
                de = (edge - rec["edge_overlap"]).abs().max().item()
                worst = max(worst, dn, de)
                ok = dn == 0.0 and de == 0.0
                n_ok += ok; n_bad += (not ok)
                print("[%3d] %-18s node_dmax=%.3e edge_dmax=%.3e %s"
                      % (i, cid, dn, de, "OK" if ok else "MISMATCH"), flush=True)
        print("\n=== cross-check vs the log-based parse ===")
        print("bitwise identical: %d   differing: %d   csr missing: %d" % (n_ok, n_bad, n_skip))
        print("worst abs deviation: %.3e" % worst)
        print("VERDICT: %s" % ("adapter reproduces _abacus_parse exactly"
                               if n_bad == 0 and n_ok > 0 else
                               "DO NOT USE -- adapter disagrees with the reference path"))
        return

    print("nothing to do; pass --verify (integration script comes next)")


if __name__ == "__main__":
    main()
