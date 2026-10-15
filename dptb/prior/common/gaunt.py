"""Real Gaunt coefficients of the ABACUS real spherical harmonics, on the device (FP64).

    G[l1(l1+1)+m1, l2(l2+1)+m2, l3(l3+1)+m3] = int Y_l1m1 Y_l2m2 Y_l3m3 dOmega

for l1, l2 <= lmax, filled only where |l1 - l2| <= l3 <= l1 + l2 and l1 + l2 + l3 is even (the layout ABACUS
extract_gaunt_table produces), by Gauss-Legendre (4 lmax + 8 nodes in cos theta) x uniform phi (8 lmax + 16 nodes)
quadrature, which is exact for these degrees.
"""
import math
from functools import lru_cache

import numpy as np
import torch

from dptb.prior.common.harmonics import real_harmonics


def _slot_columns(l):
    """Column of m = -l..l in the harmonics order 0, +1, -1, +2, -2, ..."""
    return [2 * m - 1 if m > 0 else -2 * m for m in range(-l, l + 1)]


@lru_cache(maxsize=None)
def _gaunt(lmax):
    nt, nphi = 4 * lmax + 8, 8 * lmax + 16
    mu, wmu = np.polynomial.legendre.leggauss(nt)
    phi = 2 * math.pi * np.arange(nphi) / nphi
    st = np.sqrt(1 - mu ** 2)
    v = np.stack([np.multiply.outer(st, np.cos(phi)), np.multiply.outer(st, np.sin(phi)),
                  np.multiply.outer(mu, np.ones(nphi))], -1).reshape(-1, 3)
    w = torch.as_tensor(np.multiply.outer(wmu, np.full(nphi, 2 * math.pi / nphi)).reshape(-1))
    v = torch.as_tensor(v)
    Y = torch.cat([real_harmonics(l, v)[:, _slot_columns(l)] for l in range(2 * lmax + 1)], -1)   # [p, (2lmax+1)^2]
    d1 = (lmax + 1) ** 2
    full = torch.einsum('pa,pb,pc->abc', Y[:, :d1] * w[:, None], Y[:, :d1], Y)
    l_of = torch.as_tensor([l for l in range(2 * lmax + 1) for _ in range(2 * l + 1)])
    l1, l2, l3 = l_of[:d1, None, None], l_of[None, :d1, None], l_of[None, None, :]
    keep = ((l1 - l2).abs() <= l3) & (l3 <= l1 + l2) & ((l1 + l2 + l3) % 2 == 0)
    return torch.where(keep, full, torch.zeros_like(full))


def gaunt_table(lmax):
    """[(lmax+1)^2, (lmax+1)^2, (2 lmax+1)^2] real Gaunt table on the host (a fresh copy)."""
    return _gaunt(int(lmax)).clone()
