#!/usr/bin/env python
"""Fit per-(bond-type x orbpair-slice) scales for the overlap-Hueckel hopping
prior and a per-(atom-type x column) calibrated onsite table, from a training
LMDB that carries overlap + target features.  Writes a calibration artifact
consumed by ``flow_options.prior_calibration`` (see
:mod:`dptb.nnops.prior_calibration`) and prints the explained-variance ladder

    L0  K*typemean*S   (the uncalibrated online prior, K fixed)
    L1  per (bond_type, orbpair slice) signed alpha   <- what the artifact stores
    H0  real H0 as prior (reference ceiling, when node_h0/edge_h0 are present)

so a single run answers "is the hopping gap scale or shape?" on the actual
training distribution before any training job is launched.

The fit is a closed-form least squares per group,
``alpha = sum(H*P)/sum(P*P)`` with ``P`` the K*energy*S prior entries of that
group, computed in the RME layout the dataset stores (records must carry
``edge_index``; overlap wider than the feature layout is compressed through the
record's ``soc_uureal_keep`` mask).

Usage (thin/regen-style LMDB with edge_index):
    python tools/calibrate_huckel_scales.py \
        --lmdb /path/data.train.s00.lmdb [more.lmdb ...] \
        --basis-json basis.json --has-soc --nextham-uureal-mask \
        --target-mode as_is --max-records 512 --out calib_soc729.pt

``--target-mode add_h0`` reconstructs full H = features + h0 for datasets that
store delta targets (features = H - H0).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from typing import Dict, Optional

import lmdb
import numpy as np
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nnops import prior_physical
from dptb.nnops.prior_calibration import CALIBRATION_VERSION, make_signature


def _open_records(path: str, max_records: int):
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    try:
        with env.begin() as txn:
            n = min(txn.stat()["entries"], max_records) if max_records > 0 else txn.stat()["entries"]
            for idx in range(n):
                raw = txn.get(idx.to_bytes(4, "big"))
                if raw is None:
                    continue
                yield pickle.loads(raw)
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", nargs="+", required=True, help="LMDB dataset dir(s) (subdir layout)")
    ap.add_argument("--basis-json", required=True, help="JSON file: {symbol: basis string/list} (common_options.basis)")
    ap.add_argument("--has-soc", action="store_true")
    ap.add_argument("--nextham-uureal-mask", action="store_true")
    ap.add_argument("--huckel-k", type=float, default=1.75)
    ap.add_argument("--energy-mode", choices=["type_mean", "orbital_pair"], default="type_mean",
                    help="which endpoint-energy convention the fitted scales correct")
    ap.add_argument("--target-mode", choices=["as_is", "add_h0"], default="as_is",
                    help="add_h0: targets stored as delta (features = H - H0); reconstruct H = features + h0")
    ap.add_argument("--max-records", type=int, default=512)
    ap.add_argument("--min-count", type=int, default=64,
                    help="groups with fewer entries keep scale 1.0 (no calibration)")
    ap.add_argument("--out", required=True, help="output artifact (.pt)")
    ap.add_argument("--report", default="", help="optional JSON report path")
    args = ap.parse_args()

    with open(args.basis_json, "r", encoding="utf-8") as f:
        basis = json.load(f)
    idp = OrbitalMapper(
        basis,
        method="e3tb",
        device="cpu",
        has_soc=args.has_soc,
        nextham_uureal_mask=args.nextham_uureal_mask,
    )
    orbpair_maps = idp.get_orbpair_maps() if callable(getattr(idp, "get_orbpair_maps", None)) else idp.orbpair_maps
    rme_dim = max(int(s.stop) for s in orbpair_maps.values())
    block_of_col = np.full(rme_dim, -1, dtype=np.int64)
    for bi, slc in enumerate(orbpair_maps.values()):
        block_of_col[slc.start:slc.stop] = bi
    if (block_of_col < 0).any():
        raise SystemExit("orbpair_maps leave unmapped RME columns; unsupported layout")
    n_blocks = len(orbpair_maps)
    indicator = np.zeros((rme_dim, n_blocks))
    indicator[np.arange(rme_dim), block_of_col] = 1.0

    sym_of_z = {}
    from ase.data import chemical_symbols

    for z, sym in enumerate(chemical_symbols):
        sym_of_z[z] = sym
    sym2type = dict(idp.chemical_symbol_to_type)
    bond2type = dict(idp.bond_to_type)
    n_bond = max(int(v) for v in bond2type.values()) + 1
    n_type = max(int(v) for v in sym2type.values()) + 1

    table = prior_physical.basis_onsite_table(idp, device="cpu", dtype=torch.float64)
    if table is None:
        raise SystemExit("idp does not expose basis_to_full_basis/orbpair_maps; cannot build onsite table")
    table = table.numpy()
    type_mean = prior_physical.basis_onsite_type_mean(torch.from_numpy(table), fallback=0.0).numpy()
    pair_table = None
    if args.energy_mode == "orbital_pair":
        pt = prior_physical.huckel_pair_energy_table(idp, device="cpu", dtype=torch.float64)
        if pt is None:
            raise SystemExit("energy-mode orbital_pair needs idp.bond_to_type/basis_to_full_basis")
        pair_table = pt.numpy()

    # streaming sums per (bond_type): [4, n_blocks] = hp, pp, hh, n over prior entries P
    sums: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((4, n_blocks)))
    ons_sum = np.zeros((n_type, rme_dim))
    ons_sq = np.zeros((n_type, rme_dim))
    ons_n = np.zeros(n_type)
    tot = {"h2": 0.0, "n": 0, "records": 0, "ssr_L0": 0.0, "h0_ssr": 0.0, "has_h0": True}

    for path in args.lmdb:
        for rec in _open_records(path, args.max_records):
            zs = np.asarray(rec["atomic_numbers"]).astype(int).reshape(-1)
            syms = [sym_of_z[z] for z in zs]
            try:
                ts = np.array([sym2type[s] for s in syms])
            except KeyError as exc:
                raise SystemExit(f"element {exc} in dataset but not in --basis-json") from exc
            ei = rec.get("edge_index")
            if ei is None:
                raise SystemExit(
                    "record lacks edge_index; this tool requires edge-graph datasets "
                    "(thin/regen style). Use the offline block-space script for legacy sets."
                )
            ei = np.asarray(ei).reshape(2, -1)
            h = np.asarray(rec["edge_features"], dtype=np.float64)
            nh = np.asarray(rec["node_features"], dtype=np.float64)
            h0 = rec.get("edge_h0")
            nh0 = rec.get("node_h0")
            if args.target_mode == "add_h0":
                if h0 is None or nh0 is None:
                    raise SystemExit("--target-mode add_h0 needs node_h0/edge_h0 in every record")
                h = h + np.asarray(h0, dtype=np.float64)
                nh = nh + np.asarray(nh0, dtype=np.float64)
            s = np.asarray(rec["edge_overlap"], dtype=np.float64)
            if s.shape[-1] != h.shape[-1]:
                keep = rec.get("soc_uureal_keep")
                if keep is None or np.asarray(keep).astype(bool).sum() != h.shape[-1]:
                    raise SystemExit(
                        f"edge_overlap width {s.shape[-1]} != feature width {h.shape[-1]} "
                        "and no usable soc_uureal_keep mask to compress it"
                    )
                s = s[:, np.asarray(keep).astype(bool)]
            if h.shape[-1] != rme_dim:
                raise SystemExit(f"feature width {h.shape[-1]} != idp rme_dim {rme_dim}; wrong basis/SOC flags?")
            ti, tj = ts[ei[0]], ts[ei[1]]
            bt = np.array(
                [bond2type[f"{syms[a]}-{syms[b]}"] for a, b in zip(ei[0], ei[1])],
                dtype=np.int64,
            )
            # prior entries P for the configured energy convention
            if pair_table is not None:
                p = args.huckel_k * pair_table[bt] * s
            else:
                emean = 0.5 * (type_mean[ti] + type_mean[tj])
                p = args.huckel_k * emean[:, None] * s
            tot["h2"] += float((h * h).sum())
            tot["n"] += h.size
            tot["ssr_L0"] += float(((h - p) ** 2).sum())
            if h0 is not None and args.target_mode == "as_is":
                tot["h0_ssr"] += float(((h - np.asarray(h0, dtype=np.float64)) ** 2).sum())
            elif args.target_mode == "add_h0":
                tot["h0_ssr"] += float((np.asarray(rec["edge_features"], dtype=np.float64) ** 2).sum())
            else:
                tot["has_h0"] = False
            for b in np.unique(bt):
                m = bt == b
                sums[int(b)] += np.stack([
                    ((h[m] * p[m]) @ indicator).sum(0),
                    ((p[m] * p[m]) @ indicator).sum(0),
                    ((h[m] * h[m]) @ indicator).sum(0),
                    (np.ones_like(h[m]) @ indicator).sum(0),
                ])
            for t in np.unique(ts):
                mt = ts == t
                ons_sum[t] += nh[mt].sum(0)
                ons_sq[t] += (nh[mt] ** 2).sum(0)
                ons_n[t] += mt.sum()
            tot["records"] += 1

    if tot["records"] == 0:
        raise SystemExit("no records read")

    # edge scale table (multiplicative correction on top of the configured prior)
    edge_scale = np.ones((n_bond, rme_dim))
    ssr_L1 = 0.0
    for b, arr in sums.items():
        hp, pp, hh, nn = arr
        alpha = np.where((pp > 0) & (nn >= args.min_count), hp / np.maximum(pp, 1e-300), 1.0)
        ssr_L1 += float((hh - 2 * alpha * hp + alpha * alpha * pp).sum())
        edge_scale[b] = alpha[block_of_col]

    node_table = np.zeros((n_type, rme_dim))
    seen = ons_n > 0
    node_table[seen] = ons_sum[seen] / ons_n[seen, None]
    ons_ssr = float((ons_sq - node_table * ons_sum).sum())
    ons_h2 = float(ons_sq.sum())

    report = {
        "records": tot["records"],
        "hopping_entries": tot["n"],
        "R2_L0_asconfigured": 1 - tot["ssr_L0"] / tot["h2"],
        "R2_L1_pair_block_calibrated": 1 - ssr_L1 / tot["h2"],
        "R2_H0": (1 - tot["h0_ssr"] / tot["h2"]) if tot["has_h0"] else None,
        "onsite_R2_datacal_table": 1 - ons_ssr / ons_h2 if ons_h2 > 0 else None,
        "energy_mode": args.energy_mode,
        "huckel_k": args.huckel_k,
        "target_mode": args.target_mode,
    }
    artifact = {
        "version": CALIBRATION_VERSION,
        "signature": make_signature(idp, rme_dim=rme_dim),
        "edge_scale": torch.from_numpy(edge_scale).to(torch.float32),
        "node_table": torch.from_numpy(node_table).to(torch.float32),
        "report": report,
        "meta": {
            "lmdb": list(args.lmdb),
            "min_count": args.min_count,
        },
    }
    torch.save(artifact, args.out)
    print(json.dumps(report, indent=1))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
