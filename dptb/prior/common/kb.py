"""Kleinman-Bylander matrices of a UPF on its real-harmonic projector descriptors (Ry), shared by the H0 and GF builders.

Projectors a in file order, each with descriptors m = 0, +1, -1, ... (ABACUS real harmonics); D_ab couples equal l
(and equal j for fully relativistic files).
    spinor_d   D[(a, m, s), (b, m', s')] = D_ab P_(l, j)[(m, s), (m', s')], spin-major [up block, down block], with
               P_(l, l+1/2) = (l + 1 + L.sigma) / (2l + 1), P_(l, l-1/2) = (l - L.sigma) / (2l + 1) taken to the real
               harmonics (identity in spin for scalar-relativistic files)
    effective_d  the non-SOC spin trace: D_ab sqrt(w_a w_b) delta_mm', w = 1 (l = 0 or scalar), (l + 1)/(2l + 1)
               (j = l + 1/2), l/(2l + 1) (j = l - 1/2); equals (D_upup + D_dndn) / 2 of spinor_d.
"""
import math

import numpy as np


def spin_angular_projector(l, j):
    """[2(2l+1)]^2 projector onto j = l +/- 1/2 in [up real-AO, down real-AO] with m = 0, +1, -1, ..."""
    l = int(l)
    if l < 0 or not (abs(j - l - .5) < 1e-8 or (l > 0 and abs(j - l + .5) < 1e-8)):
        raise ValueError('invalid spin-angular l, j')
    n = 2 * l + 1
    m = np.arange(-l, l + 1)
    plus = np.zeros((n, n), complex)
    for c in range(n - 1):
        plus[c + 1, c] = np.sqrt(l * (l + 1) - m[c] * (m[c] + 1))
    ls = np.block([[np.diag(m), plus.T], [plus, -np.diag(m)]]).astype(complex)
    proj = ((l + 1) * np.eye(2 * n) + ls) / (2 * l + 1) if j > l else (l * np.eye(2 * n) - ls) / (2 * l + 1)
    c = np.zeros((n, n), complex)            # rows: real harmonics in the complex basis
    c[0, l] = 1
    for k in range(1, l + 1):
        c[2 * k - 1, l + k], c[2 * k - 1, l - k] = 1 / np.sqrt(2), (-1) ** k / np.sqrt(2)
        c[2 * k, l + k], c[2 * k, l - k] = -1j / np.sqrt(2), 1j * (-1) ** k / np.sqrt(2)
    t = np.kron(np.eye(2), c)
    return t.conj() @ proj @ t.T


def spinor_d(shells, js, dij, *, has_so):
    """Complex spinor D (Ry) of radial D_ij on projector shells with total angular momenta js."""
    shells, dij = [int(l) for l in shells], np.asarray(dij, dtype=float)
    if dij.shape != (len(shells),) * 2 or not np.allclose(dij, dij.T, atol=1e-12, rtol=1e-12):
        raise ValueError('UPF D matrix must be square and symmetric')
    if has_so and any(j is None for j in js):
        raise ValueError('fully relativistic projectors require explicit j')
    off = np.r_[0, np.cumsum([2 * l + 1 for l in shells])]
    w = int(off[-1])
    out = np.zeros((2 * w, 2 * w), complex)
    for a, l in enumerate(shells):
        ang = spin_angular_projector(l, float(js[a])) if has_so else np.eye(2 * (2 * l + 1))
        rows = np.r_[np.arange(off[a], off[a + 1]), w + np.arange(off[a], off[a + 1])]
        for b, lb in enumerate(shells):
            if l != lb or (has_so and abs(float(js[a]) - float(js[b])) > 1e-8):
                continue
            cols = np.r_[np.arange(off[b], off[b + 1]), w + np.arange(off[b], off[b + 1])]
            out[np.ix_(rows, cols)] = dij[a, b] * ang
    return out


def _spin_trace_weight(l, j, has_so):
    if not has_so or j is None or l == 0:
        return 1.0
    if abs(float(j) - (l + 0.5)) < 1e-8:
        return (l + 1.0) / (2.0 * l + 1.0)
    if abs(float(j) - (l - 0.5)) < 1e-8:
        return l / (2.0 * l + 1.0)
    raise ValueError(f'incompatible fully-relativistic projector l={l}, j={j}')


def effective_d(upf):
    """Non-SOC spin-traced KB matrix of a UPF (block diagonal in m)."""
    ps = list(upf.projectors)
    off = np.concatenate(([0], np.cumsum([2 * int(p.l) + 1 for p in ps]))).astype(int)
    out = np.zeros((off[-1], off[-1]))
    for a, pa in enumerate(ps):
        for b, pb in enumerate(ps):
            if pa.l != pb.l:
                continue
            if upf.has_so and ((pa.j is None) != (pb.j is None) or
                               (pa.j is not None and abs(float(pa.j) - float(pb.j)) > 1e-8)):
                continue
            w = math.sqrt(_spin_trace_weight(int(pa.l), pa.j, bool(upf.has_so)) *
                          _spin_trace_weight(int(pb.l), pb.j, bool(upf.has_so)))
            out[off[a]:off[a + 1], off[b]:off[b + 1]] = np.eye(2 * int(pa.l) + 1) * float(upf.dij_ry[a, b]) * w
    return out
