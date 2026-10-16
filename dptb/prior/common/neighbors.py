"""Periodic neighbour enumeration on the device.

All searches of the prior and inference paths reduce to one query: the directed image pairs (i, j, S) of a periodic
structure with |x_j + S @ cell - x_i| below a pair radius r_i + r_j + margin, where S is an integer lattice translation.
The query runs in FP64 on the device over all candidate translations, so positions never leave the GPU and no cell
list is kept on the host. On CUDA one fused Triton kernel tests every (row, atom, translation) in registers, first
counting the hits of each (row tile, atom tile, translation chunk) and then writing them at prefix-sum offsets, so no
[rows, atoms, translations, 3] intermediate is formed; small cells, with few tiles and hundreds of translations, get
as many translation chunks as fill the device. Elsewhere the same test runs in row chunks with torch. Both evaluate the same expression
((x_j + S @ cell) - x_i, squares summed in order without FMA contraction, correctly rounded sqrt), so they select the
same pairs. Results are sorted by the key (i, j, S0, S1, S2), the order of ``np.unique`` on key rows.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

BR, BJ = 16, 64     # rows x atoms per program of the fused pair test
PROGRAMS = 512      # translation chunks per call: enough programs for about four per SM
FUSED = True        # use the fused kernel on CUDA (False: the torch row-chunk path everywhere)


def key_codes(i, j, shift, n_atoms, span):
    """Monotone int64 code of keys (i, j, S) with |S| < span; lexicographic order of (i, j, S0, S1, S2)."""
    base = 2 * span + 1
    s = shift.to(torch.int64) + span
    return (((i.to(torch.int64) * n_atoms + j) * base + s[:, 0]) * base + s[:, 1]) * base + s[:, 2]


def translations(frac, radius, cell):
    """Integer translations S (rows, int64) that can bring any atom within ``radius`` of any other atom.

    ``frac`` are the fractional coordinates as given (not wrapped), so S applies to the original positions.
    """
    dual = torch.linalg.inv(cell).norm(dim=0)
    span = frac.max(0).values - frac.min(0).values if len(frac) else torch.zeros(3, dtype=cell.dtype, device=cell.device)
    hi = torch.ceil(span + radius * dual).to(torch.int64) + 1
    axes = [torch.arange(-int(h), int(h) + 1, device=cell.device) for h in hi.tolist()]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


@triton.jit(do_not_specialize=['NR', 'N', 'NT', 'T0', 'TB'])
def _pairs_kernel(pos, shift, rad_r, rad_j, rows, offsets, counts, oi, oj, ot, od, NR, N, NT, T0, TB, margin_t,
                  STRICT: tl.constexpr, SELF: tl.constexpr, F32LIM: tl.constexpr, WRITE: tl.constexpr,
                  VEC: tl.constexpr, BR: tl.constexpr, BJ: tl.constexpr):
    """Pairs (rows[a], b, t) with |(x_b + shift_t) - x_rows[a]| below the pair radius, for one BR x BJ tile and the
    translations t of chunk [TB * k, TB * k + TB). WRITE = False: store the program's hit count; WRITE = True: write
    the hits (row-major over the tile, translations outer) from the program's offset."""
    pr = tl.program_id(0)
    pj = tl.program_id(1)
    pt = tl.program_id(2)
    pid = (pr * tl.num_programs(1) + pj) * tl.num_programs(2) + pt
    a = pr * BR + tl.arange(0, BR)
    b = pj * BJ + tl.arange(0, BJ)
    ma = a < NR
    mb = b < N
    r = tl.load(rows + a, mask=ma, other=0)
    xr = tl.load(pos + r * 3 + 0, mask=ma, other=0.0)[:, None]
    yr = tl.load(pos + r * 3 + 1, mask=ma, other=0.0)[:, None]
    zr = tl.load(pos + r * 3 + 2, mask=ma, other=0.0)[:, None]
    xb = tl.load(pos + b * 3 + 0, mask=mb, other=0.0)[None, :]
    yb = tl.load(pos + b * 3 + 1, mask=mb, other=0.0)[None, :]
    zb = tl.load(pos + b * 3 + 2, mask=mb, other=0.0)[None, :]
    margin = tl.load(margin_t)                       # FP64 (a Python float argument would be FP32)
    ra = tl.load(rad_r + r, mask=ma, other=0.0)
    rb = tl.load(rad_j + b, mask=mb, other=0.0)
    if F32LIM:
        lim = (ra.to(tl.float32)[:, None] + rb.to(tl.float32)[None, :]).to(tl.float64) + margin
    else:
        lim = (ra[:, None] + rb[None, :]) + margin
    valid = ma[:, None] & mb[None, :]
    same = r[:, None] == b[None, :]
    rr = tl.reshape(tl.broadcast_to(r[:, None], (BR, BJ)), (BR * BJ,))
    bb = tl.reshape(tl.broadcast_to(b[None, :], (BR, BJ)), (BR * BJ,))
    cnt = tl.zeros((), dtype=tl.int64)
    base = tl.load(offsets + pid) if WRITE else cnt
    for t in range(pt * TB, tl.minimum(pt * TB + TB, NT)):
        dx = (xb + tl.load(shift + t * 3 + 0)) - xr
        dy = (yb + tl.load(shift + t * 3 + 1)) - yr
        dz = (zb + tl.load(shift + t * 3 + 2)) - zr
        dist = tl.sqrt((dx * dx + dy * dy) + dz * dz)       # FP64 sqrt is sqrt.rn.f64
        keep = (dist < lim) if STRICT else (dist <= lim)
        keep = keep & valid
        if not SELF:
            keep = keep & ~(same & (t == T0))
        k = tl.reshape(keep.to(tl.int64), (BR * BJ,))
        if WRITE:
            dst = base + cnt + tl.cumsum(k, 0) - k
            m = k > 0
            tl.store(oi + dst, rr, mask=m)
            tl.store(oj + dst, bb, mask=m)
            tl.store(ot + dst, tl.zeros_like(k) + t, mask=m)
            if VEC:
                tl.store(od + dst * 3 + 0, tl.reshape(dx, (BR * BJ,)), mask=m)
                tl.store(od + dst * 3 + 1, tl.reshape(dy, (BR * BJ,)), mask=m)
                tl.store(od + dst * 3 + 2, tl.reshape(dz, (BR * BJ,)), mask=m)
        cnt += tl.sum(k, 0)
    if not WRITE:
        tl.store(counts + pid, cnt)


def _pairs_cuda(pos, shift, T, radius, rj, rows, margin, strict, self_interaction, limit_dtype, with_vectors):
    """Fused pair test (see ``_pairs_kernel``): i, j, S and optionally the vectors, unsorted."""
    dev = pos.device
    pos = pos.contiguous()
    shift = shift.contiguous()
    nr, n, nt = len(rows), len(pos), len(T)
    t0 = (len(T) - 1) // 2          # translations() spans a symmetric box: (0, 0, 0) is the middle row
    tiles = triton.cdiv(nr, BR) * triton.cdiv(n, BJ)
    tb = triton.cdiv(nt, min(nt, triton.cdiv(PROGRAMS, tiles)))
    grid = (triton.cdiv(nr, BR), triton.cdiv(n, BJ), triton.cdiv(nt, tb))
    counts = torch.empty(grid[0] * grid[1] * grid[2], dtype=torch.int64, device=dev)
    ri = radius.to(pos.dtype).contiguous()
    rb = rj.to(pos.dtype).contiguous()
    rows = rows.contiguous()
    f32 = limit_dtype == torch.float32
    args = dict(STRICT=bool(strict), SELF=bool(self_interaction), F32LIM=f32, VEC=bool(with_vectors), BR=BR, BJ=BJ,
                num_warps=4, enable_fp_fusion=False)
    mt = torch.tensor([float(margin)], dtype=torch.float64, device=dev)
    _pairs_kernel[grid](pos, shift, ri, rb, rows, counts, counts, counts, counts, counts, pos, nr, n, nt, t0, tb, mt,
                        WRITE=False, **args)
    offsets = torch.cumsum(counts, 0) - counts
    m = int(counts.sum())
    i = torch.empty(m, dtype=torch.int64, device=dev)
    j = torch.empty(m, dtype=torch.int64, device=dev)
    t = torch.empty(m, dtype=torch.int64, device=dev)
    d = torch.empty((m, 3), dtype=pos.dtype, device=dev) if with_vectors else pos
    if m:
        _pairs_kernel[grid](pos, shift, ri, rb, rows, offsets, counts, i, j, t, d, nr, n, nt, t0, tb, mt,
                            WRITE=True, **args)
    out = [i, j, T[t]]
    if with_vectors:
        out.append(d)
    return out


def image_pairs(pos, cell, radius, *, margin=0.0, strict=False, self_interaction=False, centres=None,
                targets_radius=None, limit_dtype=None, max_candidates=1 << 24, with_vectors=False):
    """Directed image pairs within the pair radius, sorted by (i, j, S).

    pos [n, 3], cell [3, 3]: FP64 device tensors (any length unit, the same for radii).
    radius [n]: per-atom radius r_i; the pair radius is r_i + r_j + margin (``targets_radius`` gives r_j separately).
    strict: compare with ``<`` (ASE neighbour lists) instead of ``<=``.
    limit_dtype: dtype in which r_i + r_j is formed before the FP64 comparison (float32 for the model graph, whose
        per-species cutoffs DeePTB keeps in float32).
    self_interaction: keep (i, i, 0).
    centres: optional subset of rows i (int64 tensor); default all atoms.
    Returns i, j (int64), S [m, 3] (int64) and, with ``with_vectors``, the FP64 vectors x_j + S @ cell - x_i.
    """
    dev = pos.device
    n = len(pos)
    rj = radius if targets_radius is None else targets_radius
    rows = torch.arange(n, device=dev) if centres is None else centres.to(dev)
    empty = torch.zeros(0, dtype=torch.int64, device=dev)
    if n == 0 or len(rows) == 0:
        out = (empty, empty, torch.zeros((0, 3), dtype=torch.int64, device=dev))
        return out + (torch.zeros((0, 3), dtype=pos.dtype, device=dev),) if with_vectors else out
    # the translation box of ``translations`` from one host transfer (reach, fractional span, dual norms)
    inv = torch.linalg.inv(cell)
    frac = pos @ inv
    head = torch.cat([(radius[rows].max() + rj.max()).reshape(1), frac.max(0).values - frac.min(0).values,
                      inv.norm(dim=0)]).tolist()
    reach = head[0] + float(margin)
    # a pair within reach has |S_k| <= |f_j,k - f_i,k| + reach |inv[:, k]| <= span_k + reach dual_k (u = v inv): the
    # ceiling bounds S_k unless rounding moves the bound by a whole unit
    hi = [math.ceil(s + reach * d) for s, d in zip(head[1:4], head[4:7])]
    T = torch.as_tensor(np.stack(np.meshgrid(*[np.arange(-h, h + 1) for h in hi], indexing='ij'), -1).reshape(-1, 3),
                        device=dev)
    shift = T.to(pos.dtype) @ cell
    same_lim = limit_dtype == torch.float32 or (radius.dtype == rj.dtype == torch.float64
                                                 and limit_dtype in (None, torch.float64))
    if FUSED and pos.is_cuda and pos.dtype == torch.float64 and same_lim:
        parts = [_pairs_cuda(pos, shift, T, radius, rj, rows, margin, strict, self_interaction, limit_dtype,
                             with_vectors)]
        rows = rows[:0]
    else:
        parts = []
    step = max(1, max_candidates // max(1, n * len(T)))
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        d = pos[None, :, None, :] + shift[None, None, :, :] - pos[r][:, None, None, :]     # [b, n, nT, 3]
        dist = torch.sqrt((d * d).sum(-1))
        if limit_dtype is None:
            lim = radius[r][:, None, None] + rj[None, :, None] + margin
        else:
            lim = (radius[r].to(limit_dtype)[:, None, None] + rj.to(limit_dtype)[None, :, None]).to(pos.dtype) + margin
        keep = dist < lim if strict else dist <= lim
        if not self_interaction:
            keep &= ~((r[:, None, None] == torch.arange(n, device=dev)[None, :, None]) & (T == 0).all(1)[None, None, :])
        ii, jj, tt = torch.nonzero(keep, as_tuple=True)
        part = [r[ii], jj, T[tt]]
        if with_vectors:
            part.append(d[ii, jj, tt])
        parts.append(part)
        del d, dist, keep
    i = torch.cat([p[0] for p in parts])
    j = torch.cat([p[1] for p in parts])
    S = torch.cat([p[2] for p in parts])
    span = max(hi) + 1                  # |S| <= hi: any bound gives the same (unique) key order
    order = torch.argsort(key_codes(i, j, S, n, span))
    out = (i[order], j[order], S[order])
    if with_vectors:
        out = out + (torch.cat([p[3] for p in parts])[order],)
    return out


class PairBase:
    """One search serving several pair queries of a structure: the directed pairs (i, j, S) within
    row_radius[i] + col_radius[j] + margin (self pairs included), sorted by key, with vectors and distances.
    ``select`` returns what ``image_pairs`` returns for a query whose radii and margin are not larger, by testing the
    same distance expression ((x_j + S @ cell) - x_i, squares summed in order, correctly rounded sqrt) with the same
    comparison; rounding is monotone, so the base holds every pair of such a query."""

    def __init__(self, pos, cell, row_radius, col_radius, margin):
        self.n = len(pos)
        self.i, self.j, self.S, self.d = image_pairs(pos, cell, row_radius, targets_radius=col_radius, margin=margin,
                                                     self_interaction=True, with_vectors=True)
        d = self.d
        self.dist = torch.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2])
        self.self_pair = (self.i == self.j) & (self.S == 0).all(1)

    def select(self, radius, *, targets_radius=None, margin=0.0, strict=False, self_interaction=False, centres=None):
        rj = radius if targets_radius is None else targets_radius
        lim = (radius[self.i] + rj[self.j]) + margin
        keep = self.dist < lim if strict else self.dist <= lim
        if not self_interaction:
            keep &= ~self.self_pair
        if centres is not None:
            row = torch.zeros(self.n, dtype=torch.bool, device=keep.device)
            row[centres] = True
            keep &= row[self.i]
        idx = torch.nonzero(keep, as_tuple=True)[0]
        return self.i[idx], self.j[idx], self.S[idx]


def span_of(shift):
    """Smallest span with |S| < span for every row of ``shift``."""
    return int(shift.abs().max()) + 1 if shift.numel() else 1

