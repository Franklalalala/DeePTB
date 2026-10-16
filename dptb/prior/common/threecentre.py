"""Three-centre sums on the device, shared by the GF and H0 priors.

A three-centre quantity of the block (i, j, R) sums, over the centre atoms k and their images, products of two
centre|AO two-centre values that see the same centre image:

    M_ij(R) = sum_k sum_(k images) P_k,i^T D_k P_k,j,

with P [n_c, w] the centre|AO table values of the query (i, k, T) and D_k [n_c, n_c] the centre matrix of k's species
(diagonal, or a few channels of full matrices). The queries come from one periodic search (``neighbors.Queries``);
``three_centre_terms`` pairs the queries of each centre and looks the block of every pair up in the block keys (one
Triton kernel per chunk of centres), ``contract`` forms Z = D P once per query and sums every block's terms on FP64
tensor cores (``dmma``; a Triton kernel without it), writing each block once. ``projector`` is the nonlocal
(Kleinman-Bylander) part of both priors.
"""
import torch
import triton
import triton.language as tl

from dptb.prior.common import dmma
from dptb.prior.common.blocks import key_code, search_sorted

@triton.jit(do_not_specialize=['c0', 'NC', 'NP', 'N', 'SPAN', 'M', 'LOGC', 'LOGM'])
def _terms_kernel(rows_out, q1_out, q2_out, ptr, csum, qa, qk, qT, codes, order, blocks, c0, NC, NP, N, SPAN, M, LOGC,
                  LOGM, ENDPOINTS: tl.constexpr, BLOCK: tl.constexpr):
    """Query pairs p of centres [c0, c0 + NC) in the order of ``Queries.pairs`` (csum: inclusive cumsum of the
    centres' squared query counts): rows_out[p] = block row of (a1, a2, T1 - T2) in the sorted ``codes`` (``KeyIndex``)
    when it is a selected block (and, without ENDPOINTS, the centre image is neither block atom), else -1."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mp = p < NP
    pl = p.to(tl.int64)
    c = search_sorted(csum, NC, pl, LOGC, mp, RIGHT=True)          # centre of each pair
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
    code = tl.where(inside, key_code(a1, a2, r0, r1, r2, N, SPAN), key_code(a1, a2, 0, 0, 0, N, SPAN))
    pos = search_sorted(codes, M, code, LOGM, mp, RIGHT=False)
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


@triton.jit(do_not_specialize=['NQ', 'W', 'RS', 'ES'])
def _pair_sum(out, X, Z, q1, q2, rows, starts, counts, NQ, W, RS, ES, BW: tl.constexpr, PC: tl.constexpr):
    """out[row] += sum over the row's terms t of X[q1_t]^T Z[q2_t] (X, Z [Q, NQ, W]); one program per row, so
    every block is accumulated in registers and written once. Channels are read PC at a time and their outer products
    summed in registers (more loads in flight). out element (a, b) of a row sits at a * W + b times the element
    stride ES; rows are RS apart."""
    r = tl.program_id(0)
    row = tl.load(rows + r)
    st = tl.load(starts + r)
    n = tl.load(counts + r)
    a = tl.arange(0, BW)
    pc = tl.arange(0, PC)
    mw = a < W
    acc = tl.zeros([BW, BW], dtype=tl.float64)
    for t in range(st, st + n):
        i1 = tl.load(q1 + t).to(tl.int64) * NQ
        i2 = tl.load(q2 + t).to(tl.int64) * NQ
        for p0 in range(0, NQ, PC):
            p = p0 + pc
            m = (p < NQ)[:, None] & mw[None, :]
            x = tl.load(X + (i1 + p)[:, None] * W + a[None, :], mask=m, other=0.0)
            z = tl.load(Z + (i2 + p)[:, None] * W + a[None, :], mask=m, other=0.0)
            acc += tl.sum(x[:, :, None] * z[:, None, :], axis=0)
    ptr = out + row * RS + (a[:, None] * W + a[None, :]) * ES
    m2 = mw[:, None] & mw[None, :]
    tl.store(ptr, tl.load(ptr, mask=m2, other=0.0) + acc, mask=m2)


def contract(values, spec_k, terms, mats, out, skip=()):
    """out[row] += values[q1]^T M values[q2] for terms (row, q1, q2) sorted by row, M the matrix of the shared centre's
    species: mats {species index: [nq] (diagonal; out [*, w, w]) or [c, nq, nq] (out [*, c, w, w], real or
    complex)}; centre species without a matrix contribute nothing. Z = M values is formed once per query, then the
    FP64 tensor-core kernel ``dmma.pair_sum`` (or, without it, ``_pair_sum``) contracts every row's terms. skip:
    (channel, part) pairs left at zero (part 0 real, 1 imaginary), for outputs the caller discards."""
    if not len(terms) or not mats:
        return
    Q, nq, w = values.shape
    rows, counts = torch.unique_consecutive(terms[:, 0], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    q1, q2 = terms[:, 1].contiguous(), terms[:, 2].contiguous()
    X = values.contiguous()
    first = next(iter(mats.values()))
    if X.is_cuda and w <= 32 and dmma.available(X.device):
        launch = lambda o, Z, rs, es: dmma.pair_sum(o, X, Z, q1, q2, rows, starts, counts, nq, w, rs, es)  # noqa: E731
    else:
        BW = max(8, triton.next_power_of_2(w))
        warps, PC = (1, 1) if BW <= 8 else (4, 4)
        launch = lambda o, Z, rs, es: _pair_sum[(len(rows),)](o, X, Z, q1, q2, rows, starts, counts, nq, w, rs,  # noqa: E731
                                                           es, BW=BW, PC=PC, num_warps=warps)
    if first.ndim == 1:
        m = values.new_zeros((max(mats) + 1, nq))
        for k, v in mats.items():
            m[k] = v
        sk = spec_k.clamp_max(len(m) - 1)
        Z = (values * (m[sk] * (spec_k < len(m))[:, None].to(values.dtype))[:, :, None]).contiguous()
        launch(out, Z, w * w, 1)
        return
    c = first.shape[0]
    view = torch.view_as_real(out) if out.is_complex() else out[..., None]
    sel = {k: torch.nonzero(spec_k == k, as_tuple=True)[0] for k in mats}
    for ch in range(c):
        Z = torch.zeros((Q, nq, w), dtype=first.dtype, device=values.device)
        for k, M in mats.items():
            if len(sel[k]):
                Z[sel[k]] = M[ch] @ values[sel[k]].to(first.dtype)
        parts = (Z.real, Z.imag) if Z.is_complex() else (Z,)
        for part, Zp in enumerate(parts):
            if (ch, part) not in skip:
                launch(view[:, ch, :, :, part], Zp.contiguous(), view.stride(0), view.stride(3))


def spin_channels(d):
    """Spin-major [2n, 2n] -> channels [4, n, n] (uu, ud, du, dd)."""
    n = d.shape[-1] // 2
    return d.reshape(2, n, 2, n).transpose(1, 2).reshape(4, n, n)


def pad_matrices(mats, width):
    """{species index: [n] (diagonal) or [n, n] / [c, n, n] matrix} padded to ``width`` channels ([c, width, width]
    for matrices)."""
    out = {}
    for q, m in mats.items():
        if m.ndim == 1:
            out[q] = torch.nn.functional.pad(m, (0, width - len(m)))
        else:
            m = m[None] if m.ndim == 2 else m
            out[q] = torch.nn.functional.pad(m, (0, width - m.shape[-1], 0, width - m.shape[-2]))
    return out


class KBMatrices:
    """Kleinman-Bylander matrices of the centre species of one composition, padded to ``width`` projector channels.

    eff {species index: [n] or [n, n]}: the spin-traced (non-SOC) D_eff, real; spinor {species index: [2n, 2n]}:
    the spinor D (spin-major, complex), kept only for species with a spin-orbit part (Im D_uu, D_ud or D_du nonzero;
    the others add nothing beyond D_eff to the blocks the prior keeps). du / dd: every kept species has
    D_du = -conj(D_ud) / Im D_dd = -Im D_uu exactly, so those channels follow from ud / uu by an exact negation.
    """

    def __init__(self, eff, spinor, width):
        self.width = width
        self.eff = pad_matrices(eff, width)
        ch = {q: spin_channels(d) for q, d in spinor.items()}
        ch = {q: c for q, c in ch.items() if bool(c[0].imag.any() | c[1].ne(0).any() | c[2].ne(0).any())}
        self.spinor = pad_matrices(ch, width)
        self.du = all(torch.equal(M[2], -M[1].conj()) for M in self.spinor.values())
        self.dd = all(torch.equal(M[3].imag, -M[0].imag) for M in self.spinor.values())


def projector(bk, q, values, spec_k, kb):
    """Nonlocal blocks of both priors, sum over the projector centres k (the block's own atoms included) of
    P_ki^T D_k P_kj on the nodes and representative edges of ``bk`` (``blocks.BlockKeys``).

    q: ``neighbors.Queries`` of the projector centres; values [Q, nc, w]: the centre|AO projector overlaps of the
    queries (channels padded to kb.width); spec_k [Q]: species index of each query's centre; kb: ``KBMatrices``.
    Returns real [m, w, w] (D_eff, on every row: the reverse edges are the transposes), spin [m, 4, w, w] complex
    (D_spinor channels uu, ud, du, dd on the canonical rows, as ``finish.finish`` reads them; the real parts of uu and
    dd are not formed: D_eff carries them) and the counts of queries and three-centre terms.
    """
    m, w = len(bk), values.shape[-1]
    real = values.new_zeros((m, 1, w, w))
    spin = torch.zeros((m, 4, w, w), dtype=torch.complex128, device=values.device)
    terms = q.a.new_zeros((0, 3))
    if len(q):
        terms = three_centre_terms(q, bk.index, bk.canonical, endpoints=True)
        contract(values, spec_k, terms, kb.eff, real)
        if kb.spinor:
            skip = {(0, 0), (3, 0)} | ({(2, 0), (2, 1)} if kb.du else set()) | ({(3, 1)} if kb.dd else set())
            contract(values, spec_k, terms, kb.spinor, spin, skip=skip)
            if kb.du:
                spin[:, 2] = -spin[:, 1].conj()
            if kb.dd:
                spin[:, 3].imag.copy_(-spin[:, 0].imag)
    return bk.mirror(real[:, 0]), spin, dict(queries=len(q), terms=len(terms))
