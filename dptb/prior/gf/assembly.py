"""GF prior two- and three-centre assembly on the device (ABACUS gauge, blocks padded to the structure's AO width w).

Blocks are the n nodes followed by the support edges (``topology.Topology``). Per block:

    base     edges with |d| <= rc_i + rc_j + 1e-10: P2 base table (T + V_NA,i + V_NA,j, Ry); nodes: onsite_base
    overlap  same rows: overlap table; nodes: onsite_overlap
    proj     sum over projector centres k (endpoints included) of P_ki^T D P_kj, P the centre|AO projector table,
             masked per projector channel to |d| <= beta_ch + rc_AO + 1e-12; D = D_eff (Ry, real) and, for the full
             spinor part, the four spin-major channels of D_spinor (Ry, complex)
    vna      sum over VNA centres k other than the block's atoms of F_ki^T diag(eps_k) F_kj (P23 factors, eV): node
             blocks (symmetrised (V + V^T) / 2) and representative edges (reverse edges take the transpose)

Outputs, as the reference assembly and edge-VNA plans define them:
    node_p23 = Ry herm(onsite_base + proj) + sym(V_node),  edge_p2 = Ry herm(base + proj),  edge_vna,
    S = herm(overlap) and the spinor projector part Ry herm(proj_spinor) [*, 2w, 2w] (spin-major, padded per spin),
with herm(X)_node = (X + X^H) / 2 and herm(X)_edge = (X + X_reverse^H) / 2.
"""
import torch
import triton
import triton.language as tl

from dptb.prior.common.harmonics import rotation_matrices
from dptb.prior.common.radial import rotation_pack, rotation_pack_of
from dptb.prior.gf.topology import centre_queries, three_centre_terms

RY = 13.605693122994


class Context:
    """Per-structure arrays shared by the assembly, environment-XC and pair-XC steps."""

    def __init__(self, tables, symbols, pos, cell, topo):
        self.tables = tables
        self.symbols = symbols
        self.names = sorted(set(symbols))
        self.sp = {s: tables.species(s) for s in self.names}
        dev = pos.device
        self.spec = torch.as_tensor([self.names.index(s) for s in symbols], device=dev)
        self.pos, self.cell, self.topo = pos, cell, topo
        self.n = len(symbols)
        self.w = max(self.sp[s].norb for s in self.names)
        self.norb = torch.as_tensor([self.sp[s].norb for s in symbols], device=dev)
        self.rc = torch.as_tensor([self.sp[s].rc for s in symbols], dtype=pos.dtype, device=dev)
        # largest angular degree of any table of these species (AO, projector, VNA and density-projector shells)
        self.lmax = max(max(self.sp[s].shells + self.sp[s].proj_shells + self.sp[s].vna_shells + self.sp[s].q_shells)
                        for s in self.names)

    def column(self, attr):
        return torch.as_tensor([getattr(self.sp[s], attr) for s in self.symbols], dtype=self.pos.dtype,
                               device=self.pos.device)


class Displacements:
    """Displacements of one query set grouped by (left, right) species, with their rotation matrices computed once
    and shared by every table evaluated on them (rotations of -d for the transposed sorted-pair tables)."""

    def __init__(self, ctx, left_atoms, right_atoms, vectors, lmax):
        self.ctx, self.vectors, self.n = ctx, vectors.contiguous(), len(vectors)
        ns = len(ctx.names)
        pair = ctx.spec[left_atoms] * ns + ctx.spec[right_atoms]
        order = torch.argsort(pair)
        counts = torch.bincount(pair, minlength=ns * ns).tolist()
        self.groups, start = [], 0
        for p, c in enumerate(counts):
            if c:
                self.groups.append((ctx.names[p // ns], ctx.names[p % ns], order[start:start + c]))
                start += c
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
    out = disp.vectors.new_zeros((disp.n, *shape))
    for a, b, rows in disp.groups:
        flip = transpose_pairs and a > b
        table = ctx.tables.table(kind, b, a) if flip else ctx.tables.table(kind, a, b)
        table.into(out, rows, disp.vectors, disp.pack(-1.0 if flip else 1.0), flip)
    return out


def herm(blocks, topo):
    node, edge = blocks[:topo.n], blocks[topo.n:]
    rev = topo.reverse[topo.n:] - topo.n
    return torch.cat([(node + node.transpose(-1, -2).conj()) * .5, (edge + edge[rev].transpose(-1, -2).conj()) * .5])


def two_centre(ctx):
    """P2 base (Ry) and overlap blocks [n + E, w, w] (unsymmetrised)."""
    topo, w = ctx.topo, ctx.w
    nb = len(topo.keys)
    base = ctx.pos.new_zeros((nb, w, w))
    ovl = ctx.pos.new_zeros((nb, w, w))
    for q, s in enumerate(ctx.names):
        rows = torch.nonzero(ctx.spec == q, as_tuple=True)[0]
        m = ctx.sp[s].norb
        base[rows, :m, :m] = ctx.sp[s].arrays['onsite_base']
        ovl[rows, :m, :m] = ctx.sp[s].arrays['onsite_overlap']
    e = torch.arange(topo.n, nb, device=ctx.pos.device)
    i, j = topo.keys[e, 0], topo.keys[e, 1]
    d = topo.vec[e]
    sel = torch.linalg.vector_norm(d, dim=-1) <= ctx.rc[i] + ctx.rc[j] + 1e-10
    e, i, j, d = e[sel], i[sel], j[sel], d[sel]
    disp = Displacements(ctx, i, j, d, ctx.lmax)
    base[e] = evaluate(ctx, 'base', disp, (w, w))
    ovl[e] = evaluate(ctx, 'overlap', disp, (w, w))
    return base, ovl


@triton.jit(do_not_specialize=['NQ', 'W', 'RS', 'ES'])
def _pair_sum(out, X, Z, q1, q2, rows, starts, counts, NQ, W, RS, ES, BW: tl.constexpr):
    """out[row] += sum over the row's terms t of X[q1_t]^T Z[q2_t] (X, Z [Q, NQ, W]); one program per row, so
    every block is accumulated in registers and written once. out element (a, b) of a row sits at a * W + b times
    the element stride ES; rows are RS apart."""
    r = tl.program_id(0)
    row = tl.load(rows + r)
    st = tl.load(starts + r)
    n = tl.load(counts + r)
    a = tl.arange(0, BW)
    mw = a < W
    acc = tl.zeros([BW, BW], dtype=tl.float64)
    for t in range(st, st + n):
        i1 = tl.load(q1 + t) * NQ
        i2 = tl.load(q2 + t) * NQ
        for p in range(NQ):
            x = tl.load(X + (i1 + p) * W + a, mask=mw, other=0.0)
            z = tl.load(Z + (i2 + p) * W + a, mask=mw, other=0.0)
            acc += x[:, None] * z[None, :]
    ptr = out + row * RS + (a[:, None] * W + a[None, :]) * ES
    m2 = mw[:, None] & mw[None, :]
    tl.store(ptr, tl.load(ptr, mask=m2, other=0.0) + acc, mask=m2)


def _contract(values, q, terms, mats, out):
    """out[row] += values[q1]^T M values[q2] for terms (row, q1, q2) sorted by row, M the matrix of the shared centre's
    species: mats {species index: [nq] (diagonal; out [*, w, w]) or [c, nq, nq] (out [*, c, w, w], real or
    complex)}; centre species without a matrix contribute nothing. Z = M values is formed once per query, then
    ``_pair_sum`` contracts every row's terms."""
    if not len(terms) or not mats:
        return
    Q, nq, w = values.shape
    rows, counts = torch.unique_consecutive(terms[:, 0], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    q1, q2 = terms[:, 1].contiguous(), terms[:, 2].contiguous()
    X = values.contiguous()
    first = next(iter(mats.values()))
    BW = max(16, triton.next_power_of_2(w))
    launch = lambda o, Z, rs, es: _pair_sum[(len(rows),)](o, X, Z, q1, q2, rows, starts, counts, nq, w, rs, es,  # noqa: E731
                                                       BW=BW, num_warps=4)
    if first.ndim == 1:
        m = values.new_zeros((max(mats) + 1, nq))
        for k, v in mats.items():
            m[k] = v
        sk = q.spec_k.clamp_max(len(m) - 1)
        Z = (values * (m[sk] * (q.spec_k < len(m))[:, None].to(values.dtype))[:, :, None]).contiguous()
        launch(out, Z, w * w, 1)
        return
    c = first.shape[0]
    view = torch.view_as_real(out) if out.is_complex() else out[..., None]
    for ch in range(c):
        Z = torch.zeros((Q, nq, w), dtype=first.dtype, device=values.device)
        for k, M in mats.items():
            sel = torch.nonzero(q.spec_k == k, as_tuple=True)[0]
            if len(sel):
                Z[sel] = M[ch] @ values[sel].to(first.dtype)
        parts = (Z.real, Z.imag) if Z.is_complex() else (Z,)
        for part, Zp in enumerate(parts):
            launch(view[:, ch, :, :, part], Zp.contiguous(), view.stride(0), view.stride(3))


def _matrices(ctx, key, width):
    """{species index: centre matrix padded to ``width``} of the species that carry ``key``."""
    out = {}
    for q, s in enumerate(ctx.names):
        m = ctx.sp[s].arrays.get(key)
        if m is None:
            continue
        if key == 'd_spinor':
            n = m.shape[0] // 2
            m = m.reshape(2, n, 2, n).transpose(1, 2).reshape(4, n, n)
        if m.ndim == 1:
            out[q] = torch.nn.functional.pad(m, (0, width - len(m)))
        else:
            if m.ndim == 2:
                m = m[None]
            out[q] = torch.nn.functional.pad(m, (0, width - m.shape[-1], 0, width - m.shape[-2]))
    return out


def projector(ctx, spinor=True):
    """Projector contractions over all blocks: real (D_eff) [n + E, w, w] and spinor [n + E, 4, w, w]."""
    topo, w = ctx.topo, ctx.w
    beta = ctx.column('proj_max')
    q = centre_queries(ctx.pos, ctx.cell, beta, ctx.rc, strict=False, margin=1e-12, include_self=True)
    q.spec_k = ctx.spec[q.k]
    nb = len(topo.keys)
    real = ctx.pos.new_zeros((nb, 1, w, w))
    spin = torch.zeros((nb, 4, w, w), dtype=torch.complex128, device=ctx.pos.device) if spinor else None
    if not len(q):
        return real[:, 0], spin
    npj = max([ctx.sp[s].arrays['d_eff'].shape[0] for s in ctx.names if 'd_eff' in ctx.sp[s].arrays] or [1])
    vals = evaluate(ctx, 'projector', Displacements(ctx, q.k, q.a, q.vec, ctx.lmax), (npj, w))
    # per-channel support: |d| <= beta_channel + rc_AO + 1e-12
    cut = torch.full((len(ctx.names), npj), -1.0, dtype=ctx.pos.dtype, device=ctx.pos.device)
    for s_i, s in enumerate(ctx.names):
        sp = ctx.sp[s]
        if sp.proj_max >= 0:
            ch = torch.as_tensor(sp.proj_cut, dtype=ctx.pos.dtype).repeat_interleave(
                torch.as_tensor([2 * l + 1 for l in sp.proj_shells]))
            cut[s_i, :len(ch)] = ch.to(ctx.pos.device)
    r = torch.linalg.vector_norm(q.vec, dim=-1)
    vals = vals * (r[:, None] <= cut[q.spec_k] + ctx.rc[q.a][:, None] + 1e-12)[:, :, None]
    every = torch.ones(nb, dtype=torch.bool, device=ctx.pos.device)
    terms = three_centre_terms(q, topo.index, every, endpoints=True)
    _contract(vals, q, terms, _matrices(ctx, 'd_eff', npj), real)
    if spinor:
        _contract(vals, q, terms, _matrices(ctx, 'd_spinor', npj), spin)
    return real[:, 0], spin


def vna(ctx):
    """Onsite VNA [n, w, w] (symmetrised) and edge VNA [E, w, w] (eV)."""
    topo, w = ctx.topo, ctx.w
    q = centre_queries(ctx.pos, ctx.cell, ctx.column('vna_cut'), ctx.rc, strict=True, margin=-1e-12)
    q.spec_k = ctx.spec[q.k]
    nb = len(topo.keys)
    out = ctx.pos.new_zeros((nb, w, w))
    if len(q):
        nq = max(len(ctx.sp[s].arrays['eps_vna']) for s in ctx.names)
        vals = evaluate(ctx, 'vna', Displacements(ctx, q.k, q.a, q.vec, ctx.lmax), (nq, w))
        blocks = topo.is_node | topo.representative
        terms = three_centre_terms(q, topo.index, blocks, endpoints=False)
        _contract(vals, q, terms, _matrices(ctx, 'eps_vna', nq), out)
    node = out[:topo.n]
    node = (node + node.transpose(-1, -2)) * .5
    edge = out[topo.n:]
    rev = topo.reverse[topo.n:] - topo.n
    edge = torch.where(topo.representative[topo.n:, None, None], edge, edge[rev].transpose(-1, -2))
    return node, edge


def pair_xc(ctx):
    """Two-centre XC <mu|v_xc[rho_i + rho_j]|nu> on every support edge [E, w, w] (eV)."""
    topo = ctx.topo
    e = torch.arange(topo.n, len(topo.keys), device=ctx.pos.device)
    disp = Displacements(ctx, topo.keys[e, 0], topo.keys[e, 1], topo.vec[e], ctx.lmax)
    return evaluate(ctx, 'pairxc', disp, (ctx.w, ctx.w), transpose_pairs=True)
