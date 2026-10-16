"""GF prior two- and three-centre steps on the device (ABACUS gauge, blocks padded to the structure's AO width w).

Blocks are those of ``blocks.BlockKeys`` (the n nodes, then the support edges). Per block:

    base     edges with |d| <= rc_i + rc_j + 1e-10: P2 base table (T + V_NA,i + V_NA,j, Ry); nodes: onsite_base
    overlap  same rows: overlap table; nodes: onsite_overlap
    proj     the shared nonlocal step (``threecentre.projector``) on the P2 centre|AO projector tables: D_eff (Ry,
             real) and the channels of D_spinor (Ry, complex)
    vna      sum over VNA centres k other than the block's atoms of F_ki^T diag(eps_k) F_kj (P23 factors, eV) on the
             nodes (symmetrised (V + V^T) / 2) and representative edges (reverse edges take the transposes)
    pair_xc  two-centre XC table on every edge

The prior combines them as C = herm(Ry herm(base + proj) + V + pair_xc + ...) (``prior``), S = herm(overlap) and
the spinor projector part (``finish``), with herm(X) = (X + X_reverse^H) / 2.
"""
import torch

from dptb.prior.common import threecentre as tc
from dptb.prior.common.harmonics import rotation_matrices
from dptb.prior.common.neighbors import centre_queries
from dptb.prior.common.radial import pair_groups, rotation_pack, rotation_pack_of
from dptb.prior.common.units import RY  # noqa: F401  (the prior reads A.RY)


class Context:
    """Per-structure arrays shared by the assembly, environment-XC and pair-XC steps."""

    def __init__(self, tables, symbols, pos, cell, bk, cache):
        self.tables, self.cache = tables, cache
        self.symbols = symbols
        self.names = sorted(set(symbols))
        self.sp = {s: tables.species(s) for s in self.names}
        dev = pos.device
        self.spec = torch.as_tensor([self.names.index(s) for s in symbols], device=dev)
        self.pos, self.cell, self.bk = pos, cell, bk
        self.n = len(symbols)
        self.w = max(self.sp[s].norb for s in self.names)
        self.norb = torch.as_tensor([self.sp[s].norb for s in symbols], device=dev)
        self.rc = torch.as_tensor([self.sp[s].rc for s in symbols], dtype=pos.dtype, device=dev)
        # largest angular degree of the AO shells and of each centre family (projector, VNA, density projector):
        # a displacement set's rotation pack only goes up to the degrees of the tables evaluated on it
        top = lambda attr: max(max(getattr(self.sp[s], attr) or (0,)) for s in self.names)  # noqa: E731
        self.lmax_ao = top('shells')
        self.lmax_proj = max(self.lmax_ao, top('proj_shells'))
        self.lmax_vna = max(self.lmax_ao, top('vna_shells'))
        self.lmax_q = max(self.lmax_ao, top('q_shells'))
        self.lmax = max(self.lmax_proj, self.lmax_vna, self.lmax_q)
        self.base = None        # optional neighbors.PairBase serving the centre queries

    def column(self, attr):
        return torch.as_tensor([getattr(self.sp[s], attr) for s in self.symbols], dtype=self.pos.dtype,
                               device=self.pos.device)


class Displacements:
    """Displacements of one query set grouped by (left, right) species, with their rotation matrices computed once
    and shared by every table evaluated on them (rotations of -d for the transposed sorted-pair tables)."""

    def __init__(self, ctx, left_atoms, right_atoms, vectors, lmax):
        self.ctx, self.vectors, self.n = ctx, vectors.contiguous(), len(vectors)
        ns = len(ctx.names)
        self.groups = [(ctx.names[p // ns], ctx.names[p % ns], rows)
                       for p, rows in pair_groups(ctx.spec[left_atoms], ctx.spec[right_atoms], ns)]
        self.lmax = lmax
        self._pack = {}

    def pack(self, sign):
        """Rotation pack (``radial.rotation_pack``) of sign * vectors, computed once."""
        if sign not in self._pack:
            if self.vectors.is_cuda:
                self._pack[sign] = rotation_pack_of(self.vectors, self.lmax, sign)
            else:
                self._pack[sign] = rotation_pack(rotation_matrices(sign * self.vectors, range(self.lmax + 1)),
                                                 self.lmax)
        return self._pack[sign]


def evaluate(ctx, kind, disp, shape, transpose_pairs=False):
    """Table ``kind`` (left species x right species of each group) at the displacements -> [Q, *shape] (padded).

    transpose_pairs: tables keyed by sorted species pairs; blocks whose (left, right) species are out of order are
    evaluated at -d and transposed.
    """
    jobs = []
    for a, b, rows in disp.groups:
        flip = transpose_pairs and a > b
        table = ctx.tables.table(kind, b, a) if flip else ctx.tables.table(kind, a, b)
        if max(table.degrees) > disp.lmax:
            raise ValueError(f'{kind} {a}|{b}: degree {max(table.degrees)} above the rotation pack ({disp.lmax})')
        jobs.append((table, rows, flip))
    # the groups partition the rows and into() writes whole blocks: no padding left to zero when every block is full
    full = all((t.shape[::-1] if f else t.shape) == tuple(shape) for t, _, f in jobs)
    out = disp.vectors.new_empty((disp.n, *shape)) if full else disp.vectors.new_zeros((disp.n, *shape))
    for table, rows, flip in jobs:
        table.into(out, rows, disp.vectors, disp.pack(-1.0 if flip else 1.0), flip)
    return out


def two_centre(ctx):
    """P2 base (Ry) and overlap blocks [n + E, w, w] (unsymmetrised)."""
    bk, w = ctx.bk, ctx.w
    nb = len(bk.keys)
    base = ctx.pos.new_zeros((nb, w, w))
    ovl = ctx.pos.new_zeros((nb, w, w))
    for q, s in enumerate(ctx.names):
        rows = torch.nonzero(ctx.spec == q, as_tuple=True)[0]
        m = ctx.sp[s].norb
        base[rows, :m, :m] = ctx.sp[s].arrays['onsite_base']
        ovl[rows, :m, :m] = ctx.sp[s].arrays['onsite_overlap']
    e = torch.arange(bk.n, nb, device=ctx.pos.device)
    i, j = bk.keys[e, 0], bk.keys[e, 1]
    d = bk.vec[e]
    sel = torch.linalg.vector_norm(d, dim=-1) <= ctx.rc[i] + ctx.rc[j] + 1e-10
    e, i, j, d = e[sel], i[sel], j[sel], d[sel]
    disp = Displacements(ctx, i, j, d, ctx.lmax_ao)
    base[e] = evaluate(ctx, 'base', disp, (w, w))
    ovl[e] = evaluate(ctx, 'overlap', disp, (w, w))
    return base, ovl


def _arrays(ctx, key):
    """{species index: array ``key``} of the species that carry it."""
    return {q: ctx.sp[s].arrays[key] for q, s in enumerate(ctx.names) if key in ctx.sp[s].arrays}


def kb_matrices(ctx, width):
    """``threecentre.KBMatrices`` of the composition (D_eff diagonal, D_spinor), built once per composition."""
    key = (tuple(ctx.names), width)
    if key not in ctx.cache:
        ctx.cache[key] = tc.KBMatrices(_arrays(ctx, 'd_eff'), _arrays(ctx, 'd_spinor'), width)
    return ctx.cache[key]


def projector(ctx):
    """Projector blocks: real (D_eff) [m, w, w] on every row, spinor [m, 4, w, w] on the canonical rows and the
    query and term counts (``threecentre.projector``), from the P2 centre|AO projector tables masked per projector
    channel to |d| <= beta_channel + rc_AO + 1e-12."""
    bk, w = ctx.bk, ctx.w
    beta = ctx.column('proj_max')
    q = centre_queries(ctx.pos, ctx.cell, beta, ctx.rc, strict=False, margin=1e-12, include_self=True,
                       base=ctx.base)
    npj = max([ctx.sp[s].arrays['d_eff'].shape[0] for s in ctx.names if 'd_eff' in ctx.sp[s].arrays] or [1])
    if not len(q):
        return tc.projector(bk, q, ctx.pos.new_zeros((0, npj, w)), ctx.spec[q.k], kb_matrices(ctx, npj))
    vals = evaluate(ctx, 'projector', Displacements(ctx, q.k, q.a, q.vec, ctx.lmax_proj), (npj, w))
    cut = torch.full((len(ctx.names), npj), -1.0, dtype=ctx.pos.dtype, device=ctx.pos.device)
    for s_i, s in enumerate(ctx.names):
        sp = ctx.sp[s]
        if sp.proj_max >= 0:
            ch = torch.as_tensor(sp.proj_cut, dtype=ctx.pos.dtype).repeat_interleave(
                torch.as_tensor([2 * l + 1 for l in sp.proj_shells]))
            cut[s_i, :len(ch)] = ch.to(ctx.pos.device)
    spec_k = ctx.spec[q.k]
    r = torch.linalg.vector_norm(q.vec, dim=-1)
    vals = vals * (r[:, None] <= cut[spec_k] + ctx.rc[q.a][:, None] + 1e-12)[:, :, None]
    return tc.projector(bk, q, vals, spec_k, kb_matrices(ctx, npj))


def vna(ctx):
    """VNA [m, w, w] (eV): sum over the VNA centres k other than the block's atoms of F_ki^T diag(eps_k) F_kj on the
    nodes (symmetrised (V + V^T) / 2) and the representative edges (the reverse edges take the transposes); and the
    counts of queries and three-centre terms."""
    bk, w = ctx.bk, ctx.w
    q = centre_queries(ctx.pos, ctx.cell, ctx.column('vna_cut'), ctx.rc, strict=True, margin=-1e-12,
                       base=ctx.base)
    out = ctx.pos.new_zeros((len(bk), w, w))
    terms = q.a.new_zeros((0, 3))
    if len(q):
        nq = max(len(ctx.sp[s].arrays['eps_vna']) for s in ctx.names)
        vals = evaluate(ctx, 'vna', Displacements(ctx, q.k, q.a, q.vec, ctx.lmax_vna), (nq, w))
        terms = tc.three_centre_terms(q, bk.index, bk.canonical, endpoints=False)
        tc.contract(vals, ctx.spec[q.k], terms, tc.pad_matrices(_arrays(ctx, 'eps_vna'), nq), out)
    node = out[:bk.n]
    out[:bk.n] = (node + node.transpose(-1, -2)) * .5
    return bk.mirror(out), dict(queries=len(q), terms=len(terms))


def pair_xc(ctx):
    """Two-centre XC <mu|v_xc[rho_i + rho_j]|nu> [m, w, w] (eV) on every edge, zero on the nodes. The table of the
    sorted species pair is evaluated at -d and transposed for the other order; a same-species table is not an exact
    transpose of itself at -d (bipolar quadrature from one atom's frame), so both directions are evaluated and the
    prior averages them (``BlockKeys.herm``)."""
    bk = ctx.bk
    e = torch.arange(bk.n, len(bk), device=ctx.pos.device)
    disp = Displacements(ctx, bk.keys[e, 0], bk.keys[e, 1], bk.vec[e], ctx.lmax_ao)
    out = ctx.pos.new_zeros((len(bk), ctx.w, ctx.w))
    out[bk.n:] = evaluate(ctx, 'pairxc', disp, (ctx.w, ctx.w), transpose_pairs=True)
    return out
