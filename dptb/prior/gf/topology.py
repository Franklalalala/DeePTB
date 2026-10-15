"""GF prior topology on the device: block support, centre queries and three-centre terms.

Conventions (bohr). An edge (i, j, R) is the block between AO atom i in the home cell and AO atom j at image R. A query
(a, k, T) pairs AO atom a (home cell) with the centre k at image T; its displacement is x_a - (x_k + T L), the
argument of the centre|AO tables. Two queries of the same centre atom, (a1, k, T1) and (a2, k, T2), see the same centre
image for the block (a1, a2, T1 - T2): a three-centre term of that block.
"""
import torch

from dptb.prior.common.blocks import KeyIndex
from dptb.prior.common.neighbors import image_pairs, key_codes, span_of

BOHR = 0.5291772109


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
        n = len(pos)
        counts = torch.bincount(self.k, minlength=n)
        self.ptr = torch.zeros(n + 1, dtype=torch.int64, device=pos.device)
        self.ptr[1:] = torch.cumsum(counts, 0)

    def __len__(self):
        return len(self.a)

    def pairs(self, chunk=1 << 24):
        """All ordered query pairs (q1, q2) sharing the centre atom, in chunks of centres: yields (q1, q2)."""
        dev = self.a.device
        g = self.ptr[1:] - self.ptr[:-1]
        sq = g * g
        k = 0
        n = len(g)
        csum = torch.cumsum(sq, 0).tolist()
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


def centre_queries(pos, cell, centre_radius, ao_radius, *, strict, margin, include_self=False):
    """Queries (a, k, T) with |x_a - x_k - T L| below ao_radius[a] + centre_radius[k] (+ margin; strict: '<').

    Centres with a negative radius have no queries. include_self keeps (k, k, 0).
    """
    centres = torch.nonzero(centre_radius >= 0, as_tuple=True)[0]
    k, a, S = image_pairs(pos, cell, centre_radius, targets_radius=ao_radius, margin=margin, strict=strict,
                          self_interaction=include_self, centres=centres)
    # image_pairs gives x_a + S L - x_k; the query displacement x_a - (x_k + T L) uses T = -S
    return Queries(a, k, -S, pos, cell)


def support(pos, cell, rc, beta, model_edges):
    """Block support of the prior: direct AO pairs, model graph edges and pairs sharing a projector centre.

    rc [n]: AO cutoffs; beta [n]: largest projector cutoff (-1 without projectors); model_edges (i, j, S).
    Returns sorted unique edge keys [E, 5] (no (i, i, 0)).
    """
    n = len(pos)
    eps = 1e-11 / BOHR
    i, j, S = image_pairs(pos, cell, rc + eps, strict=True)
    parts = [(i, j, S), model_edges]
    if bool((beta >= 0).any()):
        q = centre_queries(pos, cell, beta, rc, strict=False, margin=1e-10 / BOHR, include_self=True)
        for q1, q2 in q.pairs():
            a1, a2 = q.a[q1], q.a[q2]
            R = q.T[q1] - q.T[q2]
            keep = (a1 != a2) | (R != 0).any(1)
            parts.append((a1[keep], a2[keep], R[keep]))
    i = torch.cat([p[0] for p in parts])
    j = torch.cat([p[1] for p in parts])
    S = torch.cat([p[2] for p in parts])
    return unique_keys(i, j, S, n)


def three_centre_terms(q, index, blocks, *, endpoints):
    """Terms (block row, q1, q2) of the blocks ``blocks`` (bool [rows of index]) from query pairs of shared centres.

    endpoints=False drops the centre images that are the block's own atoms ((k, T1) = (a1, 0) or (k, T2) = (a2, 0)).
    """
    out = []
    for q1, q2 in q.pairs():
        a1, a2 = q.a[q1], q.a[q2]
        keys = torch.cat([a1[:, None], a2[:, None], q.T[q1] - q.T[q2]], 1)
        rows = index.find(keys)
        keep = rows >= 0
        keep[keep.clone()] = blocks[rows[keep]]
        if not endpoints:
            k = q.k[q1]
            keep &= ~((k == a1) & (q.T[q1] == 0).all(1))
            keep &= ~((k == a2) & (q.T[q2] == 0).all(1))
        out.append(torch.stack([rows[keep], q1[keep], q2[keep]], 1))
    if not out:
        return torch.zeros((0, 3), dtype=torch.int64, device=q.a.device)
    t = torch.cat(out)
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
