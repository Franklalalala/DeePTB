"""GF prior block support (bohr): the blocks the reference assembly keeps.

An edge (i, j, R) is the block between AO atom i in the home cell and AO atom j at image R. The support holds the
direct AO pairs (|d| < rc_i + rc_j + EPS), the model graph edges and the pairs that share a projector centre image
(three-centre projector terms, ``neighbors.centre_queries``).
"""
import torch

from dptb.prior.common.blocks import unique_keys
from dptb.prior.common.neighbors import centre_queries, image_pairs
from dptb.prior.common.units import BOHR_GF as BOHR

EPS = 1e-11 / BOHR          # margin of the direct AO-pair support


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
