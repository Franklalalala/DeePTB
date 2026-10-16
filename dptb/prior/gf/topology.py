"""GF prior topology on the device: block support, centre queries and three-centre terms.

Conventions (bohr). An edge (i, j, R) is the block between AO atom i in the home cell and AO atom j at image R. A query
(a, k, T) pairs AO atom a (home cell) with the centre k at image T; its displacement is x_a - (x_k + T L), the
argument of the centre|AO tables. Two queries of the same centre atom, (a1, k, T1) and (a2, k, T2), see the same centre
image for the block (a1, a2, T1 - T2): a three-centre term of that block.
"""
import torch
import triton
import triton.language as tl

from dptb.prior.common.blocks import KeyIndex
from dptb.prior.common.neighbors import image_pairs, key_codes, span_of

BOHR = 0.5291772109
EPS = 1e-11 / BOHR          # margin of the direct AO-pair support


def unique_keys(i, j, S, n):
    """Sorted unique (i, j, S) rows as keys [m, 5]."""
    span = span_of(S)
    codes = torch.unique(key_codes(i, j, S, n, span))
    base = 2 * span + 1
    s2 = codes % base
    s1 = codes // base % base
    s0 = codes // base ** 2 % base
    ij = codes // base ** 3
    return torch.stack([ij // n, ij % n, s0 - span, s1 - span, s2 - span], 1)


class Queries:
    """Queries (a, k, T) of one centre family, grouped by centre atom k (rows sorted by k)."""

    def __init__(self, a, k, T, pos, cell):
        order = torch.argsort(k, stable=True)
        self.a, self.k, self.T = a[order], k[order], T[order]
        self.vec = pos[self.a] - pos[self.k] - self.T.to(pos.dtype) @ cell
        self.ptr = torch.searchsorted(self.k, torch.arange(len(pos) + 1, device=pos.device))

    def __len__(self):
        return len(self.a)

    def pairs(self, chunk=1 << 24):
        """All ordered query pairs (q1, q2) sharing the centre atom, in chunks of centres: yields (q1, q2)."""
        dev = self.a.device
        g = self.ptr[1:] - self.ptr[:-1]
        sq = g * g
        k = 0
        n = len(g)
        csum = torch.cumsum(sq, 0)
        total = int(csum[-1]) if n else 0
        if total <= chunk:          # one chunk (one host sync): the usual case below a few hundred atoms
            if total:
                cent = torch.repeat_interleave(torch.arange(n, device=dev), sq, output_size=total)
                local = torch.arange(total, device=dev) - (csum - sq)[cent]
                gk = g[cent]
                yield self.ptr[cent] + local // gk, self.ptr[cent] + local % gk
            return
        csum = csum.tolist()
        while k < n:
            # centres k .. k_end-1 with at most ``chunk`` pairs (at least one centre)
            limit = (csum[k - 1] if k else 0) + chunk
            k_end = k + 1
            while k_end < n and csum[k_end] <= limit:
                k_end += 1
            gs = g[k:k_end]
            total = int(sq[k:k_end].sum())
            if total:
                cent = torch.repeat_interleave(torch.arange(k, k_end, device=dev), gs * gs)
                off = torch.repeat_interleave(torch.cumsum(gs * gs, 0) - gs * gs, gs * gs)
                local = torch.arange(total, device=dev) - off
                gk = g[cent]
                q1 = self.ptr[cent] + local // gk
                q2 = self.ptr[cent] + local % gk
                yield q1, q2
            k = k_end


def centre_queries(pos, cell, centre_radius, ao_radius, *, strict, margin, include_self=False, base=None):
    """Queries (a, k, T) with |x_a - x_k - T L| below ao_radius[a] + centre_radius[k] (+ margin; strict: '<').

    Centres with a negative radius have no queries. include_self keeps (k, k, 0). base: a ``PairBase`` of the
    structure that covers this query (the same pairs, without a new search).
    """
    centres = torch.nonzero(centre_radius >= 0, as_tuple=True)[0]
    kw = dict(targets_radius=ao_radius, margin=margin, strict=strict, self_interaction=include_self, centres=centres)
    k, a, S = image_pairs(pos, cell, centre_radius, **kw) if base is None else base.select(centre_radius, **kw)
    # image_pairs gives x_a + S L - x_k; the query displacement x_a - (x_k + T L) uses T = -S
    return Queries(a, k, -S, pos, cell)


def support(pos, cell, rc, beta, model_edges, base=None):
    """Block support of the prior: direct AO pairs, model graph edges and pairs sharing a projector centre.

    rc [n]: AO cutoffs; beta [n]: largest projector cutoff (-1 without projectors); model_edges (i, j, S); base: an
    optional ``PairBase`` covering the searches. Returns sorted unique edge keys [E, 5] (no (i, i, 0)).
    """
    n = len(pos)
    i, j, S = image_pairs(pos, cell, rc + EPS, strict=True) if base is None else base.select(rc + EPS, strict=True)
    parts = [(i, j, S), model_edges]
    if bool((beta >= 0).any()):
        q = centre_queries(pos, cell, beta, rc, strict=False, margin=1e-10 / BOHR, include_self=True, base=base)
        for q1, q2 in q.pairs():
            a1, a2 = q.a[q1], q.a[q2]
            R = q.T[q1] - q.T[q2]
            sel = torch.nonzero((a1 != a2) | (R != 0).any(1), as_tuple=True)[0]
            parts.append((a1[sel], a2[sel], R[sel]))
    i = torch.cat([p[0] for p in parts])
    j = torch.cat([p[1] for p in parts])
    S = torch.cat([p[2] for p in parts])
    return unique_keys(i, j, S, n)


@triton.jit(do_not_specialize=['c0', 'NC', 'NP', 'N', 'SPAN', 'M', 'LOGC', 'LOGM'])
def _terms_kernel(rows_out, q1_out, q2_out, ptr, csum, qa, qk, qT, codes, order, blocks, c0, NC, NP, N, SPAN, M, LOGC,
                  LOGM, ENDPOINTS: tl.constexpr, BLOCK: tl.constexpr):
    """Query pairs p of centres [c0, c0 + NC) in the order of ``Queries.pairs`` (csum: inclusive cumsum of the
    centres' squared query counts): rows_out[p] = block row of (a1, a2, T1 - T2) in the sorted ``codes`` (``KeyIndex``)
    when it is a selected block (and, without ENDPOINTS, the centre image is neither block atom), else -1."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mp = p < NP
    pl = p.to(tl.int64)
    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.zeros([BLOCK], dtype=tl.int64) + NC
    for _ in range(LOGC):                       # centre: searchsorted(csum, p, right=True)
        mid = (lo + hi) // 2
        go = (lo < hi) & (tl.load(csum + tl.minimum(mid, NC - 1), mask=mp, other=0) <= pl)
        stay = (lo < hi) & ~go
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(stay, mid, hi)
    c = tl.minimum(lo, NC - 1)
    first = tl.load(ptr + c0 + c, mask=mp, other=0)
    g = tl.load(ptr + c0 + c + 1, mask=mp, other=1) - first
    local = pl - (tl.load(csum + c, mask=mp, other=0) - g * g)
    gs = tl.maximum(g, 1)
    q1 = first + local // gs
    q2 = first + local % gs
    a1 = tl.load(qa + q1, mask=mp, other=0)
    a2 = tl.load(qa + q2, mask=mp, other=0)
    t10 = tl.load(qT + q1 * 3 + 0, mask=mp, other=0)
    t11 = tl.load(qT + q1 * 3 + 1, mask=mp, other=0)
    t12 = tl.load(qT + q1 * 3 + 2, mask=mp, other=0)
    t20 = tl.load(qT + q2 * 3 + 0, mask=mp, other=0)
    t21 = tl.load(qT + q2 * 3 + 1, mask=mp, other=0)
    t22 = tl.load(qT + q2 * 3 + 2, mask=mp, other=0)
    r0 = t10 - t20
    r1 = t11 - t21
    r2 = t12 - t22
    inside = (tl.abs(r0) < SPAN) & (tl.abs(r1) < SPAN) & (tl.abs(r2) < SPAN)
    base = 2 * SPAN + 1
    code = (((a1 * N + a2) * base + (r0 + SPAN)) * base + (r1 + SPAN)) * base + (r2 + SPAN)
    code = tl.where(inside, code, ((a1 * N + a2) * base + SPAN) * base * base + SPAN * base + SPAN)
    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.zeros([BLOCK], dtype=tl.int64) + M
    for _ in range(LOGM):                       # searchsorted(codes, code) (left)
        mid = (lo + hi) // 2
        go = (lo < hi) & (tl.load(codes + tl.minimum(mid, M - 1), mask=mp, other=0) < code)
        stay = (lo < hi) & ~go
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(stay, mid, hi)
    pos = tl.minimum(lo, M - 1)
    hit = mp & inside & (tl.load(codes + pos, mask=mp, other=-1) == code)
    row = tl.load(order + pos, mask=hit, other=0)
    keep = hit & (tl.load(blocks + row, mask=hit, other=0) != 0)
    if not ENDPOINTS:
        k = tl.load(qk + q1, mask=mp, other=-1)
        keep = keep & ~((k == a1) & (t10 == 0) & (t11 == 0) & (t12 == 0))
        keep = keep & ~((k == a2) & (t20 == 0) & (t21 == 0) & (t22 == 0))
    tl.store(rows_out + p, tl.where(keep, row, -1), mask=mp)
    tl.store(q1_out + p, q1, mask=mp)
    tl.store(q2_out + p, q2, mask=mp)


def three_centre_terms(q, index, blocks, *, endpoints, chunk=1 << 24):
    """Terms (block row, q1, q2) of the blocks ``blocks`` (bool [rows of index]) from query pairs of shared centres,
    sorted by row (stable: centre, then q1, q2 within a row).

    endpoints=False drops the centre images that are the block's own atoms ((k, T1) = (a1, 0) or (k, T2) = (a2, 0)).
    The pairs of each chunk of centres (at most ``chunk`` pairs, at least one centre) are enumerated, looked up and
    filtered in one Triton kernel (``_terms_kernel``).
    """
    dev = q.a.device
    g = q.ptr[1:] - q.ptr[:-1]
    n = len(g)
    csum = torch.cumsum(g * g, 0)
    total = int(csum[-1]) if n else 0
    if not total or not len(index.sorted):
        return torch.zeros((0, 3), dtype=torch.int64, device=dev)
    spans = [(0, n)]
    if total > chunk:
        host = csum.tolist()
        spans, k = [], 0
        while k < n:
            limit = (host[k - 1] if k else 0) + chunk
            k_end = k + 1
            while k_end < n and host[k_end] <= limit:
                k_end += 1
            spans.append((k, k_end))
            k = k_end
    qT = q.T.to(torch.int64).contiguous()
    blk = blocks.to(torch.int8).contiguous()
    M = len(index.sorted)
    out = []
    for c0, c1 in spans:
        cs = csum[c0:c1] - (csum[c0 - 1] if c0 else 0)
        NP = total if (c0, c1) == (0, n) else int(cs[-1])
        if not NP:
            continue
        rows = torch.empty(NP, dtype=torch.int64, device=dev)
        q1 = torch.empty_like(rows)
        q2 = torch.empty_like(rows)
        BLOCK = 512
        _terms_kernel[(triton.cdiv(NP, BLOCK),)](
            rows, q1, q2, q.ptr, cs.contiguous(), q.a, q.k, qT, index.sorted, index.order, blk, c0, c1 - c0, NP,
            index.n_atoms, int(index.span), M, int(c1 - c0).bit_length() + 1, int(M).bit_length() + 1,
            ENDPOINTS=bool(endpoints), BLOCK=BLOCK, num_warps=4)
        sel = torch.nonzero(rows >= 0, as_tuple=True)[0]
        out.append(torch.stack([rows[sel], q1[sel], q2[sel]], 1))
    if not out:
        return torch.zeros((0, 3), dtype=torch.int64, device=dev)
    t = torch.cat(out) if len(out) > 1 else out[0]
    return t[torch.argsort(t[:, 0], stable=True)]


class Topology:
    """Blocks = n nodes, then the support edges; reverse rows and representative edges (r < reverse[r])."""

    def __init__(self, pos, cell, edges):
        n = len(pos)
        dev = pos.device
        nodes = torch.zeros((n, 5), dtype=torch.int64, device=dev)
        nodes[:, 0] = nodes[:, 1] = torch.arange(n, device=dev)
        self.n = n
        self.edges = edges
        self.keys = torch.cat([nodes, edges])
        self.index = KeyIndex(self.keys, n)
        self.reverse = self.index.reverse(self.keys)
        if bool((self.reverse < 0).any()):
            raise ValueError('block support is not closed under (i, j, R) -> (j, i, -R)')
        rows = torch.arange(len(self.keys), device=dev)
        self.is_node = rows < n
        self.representative = (~self.is_node) & (rows < self.reverse)
        self.vec = pos[self.keys[:, 1]] - pos[self.keys[:, 0]] + self.keys[:, 2:].to(pos.dtype) @ cell
