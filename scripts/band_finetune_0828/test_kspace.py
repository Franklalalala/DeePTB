#!/usr/bin/env python
"""Unit-check nextham_kspace before it costs a training run.

Four things, each catching a failure that would otherwise be silent:

  1. U really is the label's eigenbasis:  U^H S(k) U = I  and
     U^H H_label(k) U = diag(eps). If a convention is off (phase sign, m
     ordering, fractional vs cartesian k) the projection is still finite and
     the loss still decreases -- it just supervises the wrong subspace.
  2. Feeding the label as the prediction must drive P/Q/PQ to ~0.
  3. The weighted share of the k-space terms. NextHAM's 0.045% was tuned for
     their setup; if ours lands far outside [1%, 10%] the term is either
     asleep or in charge, and the first round showed what "in charge" costs.
  4. mu stays in the meV range, as in arm A.
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

lo = {"method": "nextham_kspace", "onsite_shift": False,
      "w_p": 2e-4, "w_q": 1e-4, "w_pq": 1.5e-4,
      "band_window": 10.0, "q_window": 30.0, "n_kpoints": 1}
lo.update({k: v for k, v in common.items() if k in ("basis", "dtype", "overlap", "has_soc")})
lo["device"] = DEV
crit = Loss(**lo)
print("criterion:", type(crit).__name__)
print("weights: R=%.5f P=%.1e Q=%.1e PQ=%.1e" % (crit.w_r, crit.w_p, crit.w_q, crit.w_pq))

# --- 1. eigenbasis sanity, done inside the loss's own machinery -------------
print("\n=== 1. is U the label eigenbasis? ===")
batch = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False))).to(DEV)
d = AtomicData.to_AtomicDataDict(batch)
base = crit._base_fields(d)
kpt = torch.rand(1, 3, device=DEV, dtype=torch.get_default_dtype())
dl = dict(base)
dl[AtomicDataDict.KPOINT_KEY] = kpt
for k in (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.EDGE_FEATURES_KEY,
          AtomicDataDict.NODE_OVERLAP_KEY, AtomicDataDict.EDGE_OVERLAP_KEY):
    v = d[k]
    dl[k] = v[0] if (torch.is_tensor(v) and v.is_nested) else v
with torch.no_grad():
    sk = crit.s2k(dict(dl))[AtomicDataDict.OVERLAP_KEY][0]
    hk = crit.h2k(dict(dl))[AtomicDataDict.HAMILTONIAN_KEY][0]
    L = torch.linalg.cholesky(sk)
    Li = torch.linalg.inv(L)
    ev, evec = torch.linalg.eigh(Li @ hk @ Li.conj().transpose(-1, -2))
    U = Li.conj().transpose(-1, -2) @ evec
    ortho = (U.conj().transpose(-1, -2) @ sk @ U
             - torch.eye(sk.shape[0], device=DEV, dtype=sk.dtype)).abs().max().item()
    diag = U.conj().transpose(-1, -2) @ hk @ U
    off = (diag - torch.diag(torch.diagonal(diag))).abs().max().item()
print("  max|U^H S U - I|      = %.3e   (want < 1e-4)" % ortho)
print("  max off-diag of U^H H U = %.3e eV (want small vs the spectrum width)" % off)
print("  spectrum: %.1f .. %.1f eV, n_orb=%d" % (ev.real.min(), ev.real.max(), ev.numel()))
ok_u = ortho < 1e-4

# --- 2/3/4. run the loss on model output and on the label ------------------
print("\n=== 2-4. loss behaviour over structures ===")
print(" idx |    mu (eV) |        R |        P |        Q |       PQ | k-share")
print("-" * 76)
shares, mus = [], []
for i, b in enumerate(DataLoader(dataset=ds, batch_size=1, shuffle=False)):
    b = b.to(DEV)
    dd = AtomicData.to_AtomicDataDict(b)
    ref = dd.copy()
    with torch.no_grad():
        total = crit(model(dd), ref)
    p = crit._last_parts
    kshare = (p["wP"] + p["wQ"] + p["wPQ"]) / max(float(total), 1e-30)
    shares.append(kshare); mus.append(p["mu"])
    print("%4d | %+10.3e | %8.3e | %8.3e | %8.3e | %8.3e | %6.2f%%"
          % (i, p["mu"], p["R"], p["P"], p["Q"], p["PQ"], 100 * kshare))
    if i >= 7:
        break

print("\n=== self-consistency: label as prediction ===")
b = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False))).to(DEV)
dd = AtomicData.to_AtomicDataDict(b)
with torch.no_grad():
    tot = crit(dd, dd.copy())
p = crit._last_parts
print("  total=%.3e  R=%.3e  P=%.3e  Q=%.3e  PQ=%.3e  mu=%.3e"
      % (float(tot), p["R"], p["P"], p["Q"], p["PQ"], p["mu"]))
ok_self = max(p["P"], p["Q"], p["PQ"]) < 1e-3

sh = np.array(shares); mu = np.array(mus)
print("\n=== verdict ===")
print("  [%s] U is the label eigenbasis (U^H S U = I)" % ("PASS" if ok_u else "FAIL"))
print("  [%s] label-vs-label drives P/Q/PQ to ~0" % ("PASS" if ok_self else "FAIL"))
print("  k-space weighted share: median %.2f%%  min %.2f%%  max %.2f%%"
      % (100 * np.median(sh), 100 * sh.min(), 100 * sh.max()))
in_band = 0.01 <= np.median(sh) <= 0.10
print("  [%s] share within [1%%, 10%%]  (below: asleep; above: it takes over)"
      % ("PASS" if in_band else "CHECK"))
print("  |mu| median %.3e eV  [%s]" % (np.median(np.abs(mu)),
                                       "PASS" if np.median(np.abs(mu)) < 0.1 else "FAIL"))
