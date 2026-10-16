"""Local-potential blocks of the H0 prior by FFT-grid tiles, on the device (FP64).

    V_ij(R) = w sum_r phi_i(r - x_i) v(r) phi_j(r - x_j - R.cell),   w = Omega / N_grid,

over the FFT grid points r = (n / N) @ cell of the home cell. The grid is cut into tiles of TILE^3 points (TILE = 8 for
every grid: on H200 the fastest edge from 16 to 2,000 atoms, at the same peak memory as 12 or 16). An atom image
(wrapped home position + T) enters a tile when some grid point of the tile lies within
its cutoff (|r - c|^2 <= r_c^2 + 1e-12, the support test of the AO values): the candidate tiles of each image are the
box of tiles its cutoff sphere can reach, tested point by point on the device. Per tile, the AO values of its M images
(A [M w, P], one Triton kernel: uniform cubic spline of the radial channel times the ABACUS real harmonic
(``harmonics.real_ylm``, the expression of the GF rotation packs), zero beyond r_c) and A v (same kernel) give all
image pair blocks at once as A (v A)^T (batched GEMM over tiles of equal image count). The block of images (a, Ta),
(b, Tb) belongs to the key (a, b, Tb - Ta); it is added (one Triton kernel, binary search of the key, atomic add) only
in the orientation whose key code is not above its reverse (the canonical rows of ``BlockKeys``: nodes and
representative edges), and the other rows are mirrored (``BlockKeys.mirror``), so V_ji(-R) = V_ij(R)^T holds exactly.
Blocks of image pairs outside the pair support must vanish (checked).
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

from dptb.prior.common.blocks import key_code, search_sorted
from dptb.prior.common.harmonics import harmonic_norm, real_ylm
from dptb.prior.common.radial import horner, interval_uniform
from dptb.prior.common.neighbors import any_of, inside, translations

TILE = 8
_CONST = {}


def _consts(device):
    if device not in _CONST:
        # support margin of r_c^2, Y_00, radius below which a point is the image centre
        _CONST[device] = torch.tensor([1e-12, 0.5 / math.sqrt(math.pi), 1e-14], dtype=torch.float64, device=device)
    return _CONST[device]


class AOTables:
    """Radial spline coefficients [species, channel, 4, interval] and per-function (channel, l, m, norm), padded.

    Coefficients are the stored not-a-knot polynomials of the prepared species (powers 3, 2, 1, 0 of r - r_k), or
    scipy's not-a-knot fit of the radial values when the species carries none.
    """

    def __init__(self, orbs, device):
        from scipy.interpolate import CubicSpline
        coeffs, desc, dr, rc, nint, norb = [], [], [], [], [], []
        for orb in orbs:
            stored = orb.metadata.get('offline_spline_coefficients')
            if stored is None:
                c = np.stack([CubicSpline(orb.r, ch.radial, bc_type='not-a-knot', extrapolate=False).c
                              for ch in orb.channels])
            else:
                c = np.asarray(stored, dtype=np.float64)
            if c.shape != (len(orb.channels), 4, len(orb.r) - 1):
                raise ValueError('AO spline coefficients do not match the radial mesh')
            coeffs.append(c)
            desc.append(np.array([[d.channel_index, d.l, d.m] for d in orb.descriptors()], dtype=np.int64))
            dr.append(float(orb.dr))
            rc.append(float(orb.rcut))
            nint.append(len(orb.r) - 1)
            norb.append(orb.norb)
        ns = len(orbs)
        self.max_ch = max(c.shape[0] for c in coeffs)
        self.max_int = max(nint)
        self.wmax = max(norb)
        C = np.zeros((ns, self.max_ch, 4, self.max_int))
        lmi = np.zeros((3, ns, self.wmax), np.int32)
        K = np.zeros((ns, self.wmax))
        for q, (c, d) in enumerate(zip(coeffs, desc)):
            C[q, :c.shape[0], :, :c.shape[2]] = c
            lmi[:, q, :len(d)] = d.T
            for mu, (_, l, m) in enumerate(d):
                K[q, mu] = harmonic_norm(int(l), int(m))
        T = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device).contiguous()  # noqa: E731
        self.coef = T(C, torch.float64)
        self.chi, self.lq, self.mq = (T(x, torch.int32) for x in lmi)
        self.nq = T(K, torch.float64)
        self.nw = T(np.asarray(norb), torch.int32)
        self.dr = T(np.asarray(dr), torch.float64)
        self.idr = T(1.0 / np.asarray(dr), torch.float64)
        self.rcut = T(np.asarray(rc), torch.float64)
        self.nint = T(np.asarray(nint), torch.int32)


@triton.jit(do_not_specialize=['M', 'MAXCH', 'MAXINT', 'W'])
def _ao_kernel(outa, outv, pot, cen, sp, ok, org, coef, dr_t, idr_t, rc_t, nint_t, chi, lq, mq, nq, nw_t, cell, consts,
               grid, M, MAXCH, MAXINT, W, TS: tl.constexpr, BLOCK_P: tl.constexpr):
    """outa[b, m, mu, p] = phi_mu(r_p - cen[b, m]) and outv = outa * pot(r_p) for the points p of tile b (image slot m,
    species sp[b, m]); zero outside the grid, beyond the cutoff and for empty slots."""
    pid = tl.program_id(0)            # b * M + m
    pb = tl.program_id(1)
    b = pid // M
    P: tl.constexpr = TS * TS * TS
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = p < P
    okv = tl.load(ok + pid)
    s = tl.load(sp + pid)
    cx = tl.load(cen + pid * 3 + 0)
    cy = tl.load(cen + pid * 3 + 1)
    cz = tl.load(cen + pid * 3 + 2)
    ox = tl.load(org + b * 3 + 0)
    oy = tl.load(org + b * 3 + 1)
    oz = tl.load(org + b * 3 + 2)
    n0 = tl.load(grid + 0)
    n1 = tl.load(grid + 1)
    n2 = tl.load(grid + 2)
    gi = ox + (p // (TS * TS))
    gj = oy + ((p // TS) % TS)
    gk = oz + (p % TS)
    inb = (gi < n0) & (gj < n1) & (gk < n2) & pmask
    v = tl.load(pot + (gi * n1 + gj) * n2 + gk, mask=inb, other=0.0)
    fx = gi.to(tl.float64) / n0.to(tl.float64)
    fy = gj.to(tl.float64) / n1.to(tl.float64)
    fz = gk.to(tl.float64) / n2.to(tl.float64)
    c00 = tl.load(cell + 0)
    c01 = tl.load(cell + 1)
    c02 = tl.load(cell + 2)
    c10 = tl.load(cell + 3)
    c11 = tl.load(cell + 4)
    c12 = tl.load(cell + 5)
    c20 = tl.load(cell + 6)
    c21 = tl.load(cell + 7)
    c22 = tl.load(cell + 8)
    rx = fx * c00 + fy * c10 + fz * c20
    ry = fx * c01 + fy * c11 + fz * c21
    rz = fx * c02 + fy * c12 + fz * c22
    dx = rx - cx
    dy = ry - cy
    dz = rz - cz
    r2 = dx * dx + dy * dy + dz * dz
    r = tl.sqrt(r2)
    eps12 = tl.load(consts + 0)
    y00 = tl.load(consts + 1)
    tiny = tl.load(consts + 2)
    rc = tl.load(rc_t + s)
    drs = tl.load(dr_t + s)
    nint = tl.load(nint_t + s)
    keep = inside(r2, rc * rc + eps12, (okv != 0) & inb)
    kf, t = interval_uniform(r, tl.load(idr_t + s), drs, nint)
    kidx = kf.to(tl.int32)
    small = r < tiny
    inv = 1.0 / tl.where(small, 1.0, r)
    u = dx * inv
    w = dz * inv
    vv = dy * inv
    nwv = tl.load(nw_t + s)
    obase = (pid * W) * P
    prev = s * 0 - 1
    rad = tl.zeros([BLOCK_P], dtype=tl.float64)
    for mu in range(0, W):            # runtime bound: one compiled kernel per tile edge, whatever the basis width
        ch = tl.load(chi + s * W + mu)
        l = tl.load(lq + s * W + mu)
        mm = tl.load(mq + s * W + mu)
        if ch != prev:
            cb = ((s * MAXCH + ch) * 4) * MAXINT + kidx
            a0 = tl.load(coef + cb, mask=keep, other=0.0)
            a1 = tl.load(coef + cb + MAXINT, mask=keep, other=0.0)
            a2 = tl.load(coef + cb + 2 * MAXINT, mask=keep, other=0.0)
            a3 = tl.load(coef + cb + 3 * MAXINT, mask=keep, other=0.0)
            rad = horner(a0, a1, a2, a3, t)
            rad = tl.where(r > rc, 0.0, rad)
            prev = ch
        y = real_ylm(l, mm, u, vv, w, tl.load(nq + s * W + mu))
        y = tl.where(small, tl.where(l == 0, y00, 0.0), y)
        val = rad * y
        val = tl.where(keep & (mu < nwv), val, 0.0)
        tl.store(outa + obase + mu * P + p, val, mask=pmask)
        tl.store(outv + obase + mu * P + p, val * v, mask=pmask)


def ao_tiles(tab, pot, cen, sp, ok, org, grid, cell, ts, block_p=128):
    """AO values A [B, M, W, P] and A * pot on the tiles with origins org [B, 3] of the images cen [B, M, 3]."""
    B, M = sp.shape
    W = tab.wmax
    P = ts ** 3
    outa = torch.empty((B, M, W, P), dtype=torch.float64, device=cen.device)
    outv = torch.empty_like(outa)
    g = (B * M, triton.cdiv(P, block_p))
    _ao_kernel[g](outa, outv, pot, cen.contiguous(), sp.to(torch.int32).contiguous(), ok.to(torch.int32).contiguous(),
                  org.to(torch.int64).contiguous(), tab.coef, tab.dr, tab.idr, tab.rcut, tab.nint, tab.chi, tab.lq,
                  tab.mq, tab.nq, tab.nw, cell, _consts(cen.device), grid, M, tab.max_ch, tab.max_int, W, TS=ts,
                  BLOCK_P=block_p)
    return outa, outv


@triton.jit
def _touch_kernel(hit, tile, cen, rc_t, cell, consts, grid, NT1, NT2, TS: tl.constexpr, BLOCK_P: tl.constexpr):
    """hit[c] = 1 when some grid point of tile tile[c] lies within the cutoff of the image centre cen[c] (the AO
    support test |r - c|^2 <= r_c^2 + 1e-12, same arithmetic as _ao_kernel)."""
    c = tl.program_id(0)
    tt = tl.load(tile + c)
    ox = (tt // (NT1 * NT2)) * TS
    oy = ((tt // NT2) % NT1) * TS
    oz = (tt % NT2) * TS
    cx = tl.load(cen + c * 3 + 0)
    cy = tl.load(cen + c * 3 + 1)
    cz = tl.load(cen + c * 3 + 2)
    rc = tl.load(rc_t + c)
    eps12 = tl.load(consts + 0)
    lim = rc * rc + eps12
    n0 = tl.load(grid + 0)
    n1 = tl.load(grid + 1)
    n2 = tl.load(grid + 2)
    c00 = tl.load(cell + 0)
    c01 = tl.load(cell + 1)
    c02 = tl.load(cell + 2)
    c10 = tl.load(cell + 3)
    c11 = tl.load(cell + 4)
    c12 = tl.load(cell + 5)
    c20 = tl.load(cell + 6)
    c21 = tl.load(cell + 7)
    c22 = tl.load(cell + 8)
    P: tl.constexpr = TS * TS * TS
    hit_c = c * 0
    for p0 in range(0, P, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        gi = ox + (p // (TS * TS))
        gj = oy + ((p // TS) % TS)
        gk = oz + (p % TS)
        inb = (gi < n0) & (gj < n1) & (gk < n2) & (p < P)
        fx = gi.to(tl.float64) / n0.to(tl.float64)
        fy = gj.to(tl.float64) / n1.to(tl.float64)
        fz = gk.to(tl.float64) / n2.to(tl.float64)
        dx = fx * c00 + fy * c10 + fz * c20 - cx
        dy = fx * c01 + fy * c11 + fz * c21 - cy
        dz = fx * c02 + fy * c12 + fz * c22 - cz
        r2 = dx * dx + dy * dy + dz * dz
        hit_c = hit_c | any_of(inside(r2, lim, inb)).to(tl.int32)
    tl.store(hit + c, hit_c.to(tl.int8))


def tile_incidences(img_c, img_rc, C, shape, ts, grid_t, cell_t):
    """(tile, image) incidences sorted by (tile, image) and the tile grid (nt0, nt1, nt2): every image whose cutoff
    reaches a grid point of the tile. Candidates are the tiles of the box spanned by the cutoff sphere in fractional
    coordinates (|f_p - f_c| <= sqrt(r_c^2 + 1e-12) |dual| per axis, the radius of the point test); each candidate is
    tested point by point (_touch_kernel)."""
    dev = img_c.device
    N = torch.as_tensor(shape, device=dev)
    nt = [-(-s // ts) for s in shape]
    inv = torch.linalg.inv(C)
    dual = inv.norm(dim=0)
    f = img_c @ inv                                           # [n_img, 3] fractional
    reach = torch.sqrt(img_rc * img_rc + 1e-12)[:, None] * dual[None, :]
    lo = torch.ceil((f - reach) * N - 1e-6).to(torch.int64).clamp(min=0)
    hi = torch.floor((f + reach) * N + 1e-6).to(torch.int64).clamp(max=N - 1)
    tlo, thi = lo // ts, hi // ts
    span = (thi - tlo + 1).clamp(min=0)
    span = torch.where((hi >= lo).all(1, keepdim=True), span, torch.zeros_like(span))
    nc = span.prod(1)
    has = torch.nonzero(nc, as_tuple=True)[0]
    nc_h = nc[has]
    total = int(nc_h.sum())
    img = torch.repeat_interleave(has, nc_h, output_size=total)
    off = torch.repeat_interleave(torch.cumsum(nc_h, 0) - nc_h, nc_h, output_size=total)
    k = torch.arange(total, device=dev) - off
    sp_ = span[img]
    s2 = sp_[:, 2]
    s12 = sp_[:, 1] * s2
    t0 = tlo[img, 0] + k // s12
    t1 = tlo[img, 1] + (k // s2) % sp_[:, 1]
    t2 = tlo[img, 2] + k % s2
    tile = (t0 * nt[1] + t1) * nt[2] + t2
    hit = torch.empty(total, dtype=torch.int8, device=dev)
    if total:
        _touch_kernel[(total,)](hit, tile, img_c[img].contiguous(), img_rc[img].contiguous(), cell_t,
                                _consts(dev), grid_t, nt[1], nt[2], TS=ts, BLOCK_P=256)
    keep = hit.bool()
    tile, img = tile[keep], img[keep]
    code = tile * len(img_c) + img
    code, perm = torch.sort(code)
    return torch.stack([tile[perm], img[perm]], 1), nt


@triton.jit(do_not_specialize=['M', 'W', 'NCODE', 'NAT', 'SPAN', 'ITERS'])
def _scatter_kernel(V, G, img, cnt, img_atom, img_T, codes, order, stray, M, W, NCODE, NAT, SPAN, ITERS,
                    BW: tl.constexpr):
    """Add the block G[b, ma, :, mb, :] of the image pair (img[b, ma], img[b, mb]) to its key row of V when the key
    code is not above the reverse key's; a block whose key is absent sets stray[0] when it is not all zero."""
    pid = tl.program_id(0).to(tl.int64)   # (b * M + ma) * M + mb; 64-bit offsets into G
    b = pid // (M * M)
    rem = pid % (M * M)
    ma = rem // M
    mb = rem % M
    n = tl.load(cnt + b)
    xa = tl.load(img + b * M + ma)
    xb = tl.load(img + b * M + mb)
    a = tl.load(img_atom + xa).to(tl.int64)
    c = tl.load(img_atom + xb).to(tl.int64)
    r0 = tl.load(img_T + xb * 3 + 0) - tl.load(img_T + xa * 3 + 0)
    r1 = tl.load(img_T + xb * 3 + 1) - tl.load(img_T + xa * 3 + 1)
    r2 = tl.load(img_T + xb * 3 + 2) - tl.load(img_T + xa * 3 + 2)
    fwd = key_code(a, c, r0, r1, r2, NAT, SPAN)
    rev = key_code(c, a, -r0, -r1, -r2, NAT, SPAN)
    inside = (r0 > -SPAN) & (r0 < SPAN) & (r1 > -SPAN) & (r1 < SPAN) & (r2 > -SPAN) & (r2 < SPAN)
    act = (ma < n) & (mb < n) & (fwd <= rev)
    pos = search_sorted(codes, NCODE, fwd, ITERS, fwd == fwd, RIGHT=False)
    found = inside & (tl.load(codes + pos) == fwd)
    row = tl.load(order + pos)
    i = tl.arange(0, BW)[:, None]
    j = tl.arange(0, BW)[None, :]
    msk = (i < W) & (j < W) & act
    MW = M * W
    blk = tl.load(G + (b * MW + ma * W + i) * MW + mb * W + j, mask=msk, other=0.0)
    tl.atomic_add(V + (row * W + i) * W + j, blk, mask=msk & found)
    bad = tl.max(tl.max(tl.where(msk & (blk != 0.0), 1, 0), axis=1), axis=0)
    flag = tl.where(act & (~found), bad, 0)
    tl.atomic_max(stray, flag, mask=flag > 0)


def local_blocks(field, tab, spec, pos, frac, cell, bk, checks, *, phi_bytes=4 << 30):
    """Local-potential blocks [m, w, w] (Ry) on the blocks of ``bk`` (``BlockKeys``) and stats.

    field: Field (values on the device); tab: AOTables; spec [n]: species row of each atom in ``tab``; pos [n, 3],
    frac [n, 3], cell [3, 3]: FP64 device tensors (bohr, positions as given); checks: ``steps.Checks``; phi_bytes:
    memory of one batch of tiles, for the AO values (A, A v) and for the tile Gram matrices G alike.
    """
    dev = pos.device
    index = bk.index
    n_keys = len(bk)
    shape = field.shape
    N = torch.as_tensor(shape, device=dev)
    C = cell
    v = field.values.to(torch.float64).contiguous()
    W = tab.wmax
    ts = TILE
    corners = torch.as_tensor([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=torch.float64,
                              device=dev)
    step = C / N[:, None].to(torch.float64)                   # rows: grid step vectors
    cen_off = torch.full((3,), (ts - 1) / 2.0, dtype=torch.float64, device=dev) @ step
    rt = float(((corners * (ts - 1)) @ step - cen_off).norm(dim=1).max())
    # atom images: wrapped home positions + T; their translation in the given coordinates is T - floor(frac)
    wind = torch.floor(frac).to(torch.int64)
    home = pos - wind.to(torch.float64) @ C
    rc = tab.rcut[spec]
    reach = float(rc.max()) + rt
    dual = torch.linalg.inv(C).norm(dim=0)
    Tr = translations(torch.zeros((1, 3), dtype=torch.float64, device=dev), reach + 1e-9, C)
    Tr = Tr[(Tr.abs() <= torch.ceil(reach * dual).to(torch.int64) + 1).all(1)]
    n = len(pos)
    img_c = (home[:, None, :] + (Tr.to(torch.float64) @ C)[None]).reshape(-1, 3)
    img_atom = torch.arange(n, device=dev)[:, None].expand(n, len(Tr)).reshape(-1).contiguous()
    img_T = (Tr[None, :, :] - wind[:, None, :]).reshape(-1, 3).contiguous()
    img_rc = rc[img_atom]
    grid_t = torch.as_tensor(list(shape), dtype=torch.int64, device=dev)
    cell_t = C.reshape(-1).contiguous()
    inc, nt = tile_incidences(img_c, img_rc, C, shape, ts, grid_t, cell_t)
    ntiles = nt[0] * nt[1] * nt[2]
    ti = torch.stack(torch.meshgrid(*[torch.arange(x, device=dev) for x in nt], indexing='ij'), -1).reshape(-1, 3)
    origin = ti * ts                                          # integer grid index of each tile corner
    counts = torch.bincount(inc[:, 0], minlength=ntiles)
    starts = torch.cumsum(counts, 0) - counts
    order = torch.argsort(counts, descending=True)
    cnt_sorted = counts[order].cpu().numpy()
    P = ts ** 3
    V = torch.zeros((n_keys, W, W), dtype=torch.float64, device=dev)
    stray = torch.zeros(1, dtype=torch.int32, device=dev)
    codes, korder = index.sorted.contiguous(), index.order.contiguous()
    iters = max(1, int(math.ceil(math.log2(max(2, len(codes))))) + 1)
    stats = dict(tiles=int(ntiles), tile_edge=ts, incidences=int(len(inc)), tile_radius=rt,
                 max_images=int(cnt_sorted.max()) if len(cnt_sorted) else 0, flops=0.)
    pairs = torch.zeros((), dtype=torch.int64, device=dev)
    at = 0
    while at < len(order):
        M = int(cnt_sorted[at])
        if M == 0:
            break
        B = max(1, min(len(order) - at, phi_bytes // (M * P * W * 8 * 3), phi_bytes // ((M * W) ** 2 * 8)))
        tiles = order[at:at + B]
        at += B
        B = len(tiles)
        cnt = counts[tiles]
        idx = starts[tiles][:, None] + torch.arange(M, device=dev)[None]
        ok = torch.arange(M, device=dev)[None] < cnt[:, None]
        img = torch.where(ok, inc[idx.clamp(max=len(inc) - 1), 1], 0)
        A, Av = ao_tiles(tab, v, img_c[img], spec[img_atom[img]], ok, origin[tiles], grid_t, cell_t, ts)
        G = torch.bmm(A.reshape(B, M * W, P), Av.reshape(B, M * W, P).transpose(1, 2))
        del A, Av
        stats['flops'] += 2.0 * B * P * (M * W) ** 2
        _scatter_kernel[(B * M * M,)](V, G, img.to(torch.int64).contiguous(), cnt.contiguous(), img_atom, img_T, codes,
                                      korder, stray, M, W, len(codes), n, index.span, iters,
                                      BW=triton.next_power_of_2(W))
        pairs += (cnt * cnt).sum()
        del G
    checks.fail_if(stray[0], 'nonzero local-potential block outside the pair support')
    checks.value('local_blocks_scattered', pairs)
    return bk.mirror(V) * field.weight, stats

