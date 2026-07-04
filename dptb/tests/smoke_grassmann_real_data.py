#!/usr/bin/env python3
"""Manual real-data smoke: pure-P (Grassmann) regression on REAL non-SOC DFT
Hamiltonians (NOT auto-collected by pytest -- name starts with `smoke_`).

Assembles dense H(Gamma) *basis-free* from the raw ``hamiltonian`` block dicts of a
non-SOC block LMDB (keys ``"i_j_Rx_Ry_Rz"``, values dense ``(norb_i, norb_j)``
blocks; Gamma point = sum over R). Picks structures with a clear central gap
(insulators/semiconductors), then verifies both the H-route (predict H, supervise
P) and the DM-route (``from_density=True``: regress a density, retract onto the
Grassmann manifold) converge ``P_pred -> P_ref`` under gradient descent.

Usage:
    python tests/smoke_grassmann_real_data.py <lmdb_root> [device]

Validated 2026-07-04 on natlan non-soc-nextham-test (torch 2.5.0+cu124): 3/3
insulating structures, both routes -> chordal ~1e-3 down to ~1e-21.
"""
import glob
import os
import pickle
import sys

import lmdb
import numpy as np
import torch

try:
    from dptb.nnops.grassmann import occupied_projector, chordal_distance_sq, grassmann_p_loss_single
except Exception:  # allow path-based standalone use
    import importlib.util
    _p = os.path.join(os.path.dirname(__file__), "..", "dptb", "nnops", "grassmann.py")
    _spec = importlib.util.spec_from_file_location("grassmann", _p)
    _m = importlib.util.module_from_spec(_spec)
    sys.modules["grassmann"] = _m
    _spec.loader.exec_module(_m)
    occupied_projector = _m.occupied_projector
    chordal_distance_sq = _m.chordal_distance_sq
    grassmann_p_loss_single = _m.grassmann_p_loss_single

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
DEV = torch.device(sys.argv[2] if len(sys.argv) > 2 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(0)


def assemble_gamma(Hdict):
    norb, parsed = {}, []
    for k, v in Hdict.items():
        p = k.split("_")
        i, j = int(p[0]), int(p[1])
        blk = np.asarray(v, dtype=np.float64)
        norb[i], norb[j] = blk.shape[0], blk.shape[1]
        parsed.append((i, j, blk))
    off, cur = {}, 0
    for a in sorted(norb):
        off[a] = cur
        cur += norb[a]
    H = np.zeros((cur, cur), dtype=np.float64)
    for i, j, blk in parsed:
        H[off[i]:off[i] + norb[i], off[j]:off[j] + norb[j]] += blk
    return 0.5 * (H + H.T), cur


def best_central_gap(eps):
    n = len(eps)
    lo, hi = max(1, int(0.2 * n)), min(n - 1, int(0.8 * n))
    bi, bg = lo, -1.0
    for i in range(lo, hi):
        if eps[i] - eps[i - 1] > bg:
            bg, bi = eps[i] - eps[i - 1], i
    return bi, bg


def descend(H_ref, n_occ, steps=400, from_density=False):
    h_ref = H_ref.to(DEV)
    with torch.no_grad():
        p_ref = occupied_projector(h_ref, None, n_occ=n_occ, from_density=from_density)
    start = h_ref + 0.3 * float(h_ref.abs().mean()) * torch.randn_like(h_ref)
    h_pred = (0.5 * (start + start.mH)).clone().requires_grad_(True)
    opt = torch.optim.Adam([h_pred], lr=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-3)

    def chordal_now():
        with torch.no_grad():
            pp = occupied_projector(h_pred.detach(), None, n_occ=n_occ, from_density=from_density)
            return float(chordal_distance_sq(pp, p_ref))

    c0 = chordal_now()
    for _ in range(steps):
        opt.zero_grad()
        r = grassmann_p_loss_single(h_pred, h_ref, None, n_occ=n_occ, lambda_chordal=1.0,
                                    lambda_eps=0.0, from_density=from_density)
        r.loss.backward()
        opt.step()
        sched.step()
    return c0, chordal_now(), float(r.gap)


def main():
    shards = sorted(glob.glob(os.path.join(ROOT, "*.lmdb")))
    recs = []
    for sh in shards[:3]:
        env = lmdb.open(sh, readonly=True, lock=False, subdir=True, max_readers=1, readahead=False)
        with env.begin() as txn:
            for _, raw in txn.cursor():
                recs.append(pickle.loads(raw))
                if len(recs) >= 12:
                    break
        env.close()
        if len(recs) >= 12:
            break
    print(f"# {len(recs)} real records from {ROOT}  device={DEV}")

    cand = []
    for idx, rec in enumerate(recs):
        H, N = assemble_gamma(rec["hamiltonian"])
        eps = np.linalg.eigvalsh(H)
        n_occ, gap = best_central_gap(eps)
        rel = gap / (float(eps.max() - eps.min()) + 1e-12)
        cand.append((rel, n_occ, N, idx))
    cand.sort(reverse=True)

    ok = 0
    top = cand[:3]
    for rel, n_occ, N, idx in top:
        H, _ = assemble_gamma(recs[idx]["hamiltonian"])
        Ht = torch.tensor(H, dtype=torch.float64)
        c0, cN, g = descend(Ht, n_occ, from_density=False)
        okH = cN < 1e-2 and cN < 1e-2 * max(c0, 1e-30)
        with torch.no_grad():
            D_ref = (2.0 * occupied_projector(Ht.to(DEV), None, n_occ=n_occ)).cpu()
        c0d, cNd, _ = descend(D_ref, n_occ, from_density=True)
        okD = cNd < 1e-2 and cNd < 1e-2 * max(c0d, 1e-30)
        print(f"  rec{idx} N={N} n_occ={n_occ} gap={g:.4f}  H-route {c0:.2e}->{cN:.2e} "
              f"{'OK' if okH else 'X'}  DM-route {c0d:.2e}->{cNd:.2e} {'OK' if okD else 'X'}")
        ok += int(okH and okD)
    print(f"===== REAL-DATA SMOKE {'PASS' if ok == len(top) else 'FAIL'}: {ok}/{len(top)} =====")
    sys.exit(0 if ok == len(top) else 1)


if __name__ == "__main__":
    main()
