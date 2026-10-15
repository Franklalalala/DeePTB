"""Local-potential blocks of the H0 prior by FFT-grid tiles, on the device (FP64).

    V_ij(R) = w sum_r phi_i(r - x_i) v(r) phi_j(r - x_j - R.cell),   w = Omega / N_grid,

over the FFT grid points r = (n / N) @ cell of the home cell. The grid is cut into tiles of TS^3 points (TS = 8, or 12
above 24M points); the tile list of every atom image (wrapped home position + T) whose sphere |r - c| <= r_c reaches
the tile's bounding sphere is formed on the device. Per tile, the AO values of its M images (A [M w, P], one Triton
kernel: uniform cubic spline of the radial channel times the ABACUS real harmonic, zero beyond r_c) give all image pair
blocks at once as A (v A)^T (batched GEMM over tiles of equal image count); the block of images (a, Ta), (b, Tb) is
added to the key (a, b, Tb - Ta). Blocks of image pairs outside the pair support must vanish (checked once at the end).
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from dptb.prior.common.neighbors import translations

_CONST = {}


def _consts(device):
    if device not in _CONST:
        _CONST[device] = torch.tensor([1e-12, 4.0 * math.pi, math.sqrt(2.0), 0.5 / math.sqrt(math.pi), 1e-14],
                                      dtype=torch.float64, device=device)
    return _CONST[device]


class AOTables:
    """Radial spline coefficients [species, channel, 4, interval] and per-function (channel, l, m), padded.

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
        for q, (c, d) in enumerate(zip(coeffs, desc)):
            C[q, :c.shape[0], :, :c.shape[2]] = c
            lmi[:, q, :len(d)] = d.T
        T = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device).contiguous()  # noqa: E731
        self.coef = T(C, torch.float64)
        self.chi, self.lq, self.mq = (T(x, torch.int32) for x in lmi)
        self.nw = T(np.asarray(norb), torch.int32)
        self.dr = T(np.asarray(dr), torch.float64)
        self.rcut = T(np.asarray(rc), torch.float64)
        self.nint = T(np.asarray(nint), torch.int32)


@triton.jit(do_not_specialize=['M', 'MAXCH', 'MAXINT', 'W'])
def _ao_kernel(out, cen, sp, ok, org, coef, dr_t, rc_t, nint_t, chi, lq, mq, nw_t, cell, consts, grid,
               M, MAXCH, MAXINT, W,
               TS: tl.constexpr, BLOCK_P: tl.constexpr):
    """out[b, m, mu, p] = phi_mu(r_p - cen[b, m]) for the points p of tile b (image slot m, species sp[b, m])."""
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
    fourpi = tl.load(consts + 1)
    sqrt2 = tl.load(consts + 2)
    y00 = tl.load(consts + 3)
    tiny = tl.load(consts + 4)
    rc = tl.load(rc_t + s)
    drs = tl.load(dr_t + s)
    nint = tl.load(nint_t + s)
    keep = (r2 <= rc * rc + eps12) & (okv != 0) & inb
    kf = libdevice.floor(r / drs)
    kidx = kf.to(tl.int32)
    kidx = tl.minimum(tl.maximum(kidx, 0), nint - 1)
    t = r - kidx.to(tl.float64) * drs
    small = r < tiny
    rs_ = tl.where(small, 1.0, r)
    ct = dz / rs_
    ct = tl.minimum(tl.maximum(ct, -1.0), 1.0)
    st = tl.sqrt(tl.maximum(1.0 - ct * ct, 0.0))
    phi = libdevice.atan2(dy, dx)
    nwv = tl.load(nw_t + s)
    obase = out + (pid * W) * P
    for mu in range(0, W):            # runtime bound: one compiled kernel per tile edge, whatever the basis width
        ch = tl.load(chi + s * W + mu)
        l = tl.load(lq + s * W + mu)
        mm = tl.load(mq + s * W + mu)
        cb = ((s * MAXCH + ch) * 4) * MAXINT + kidx
        a0 = tl.load(coef + cb, mask=keep, other=0.0)
        a1 = tl.load(coef + cb + MAXINT, mask=keep, other=0.0)
        a2 = tl.load(coef + cb + 2 * MAXINT, mask=keep, other=0.0)
        a3 = tl.load(coef + cb + 3 * MAXINT, mask=keep, other=0.0)
        rad = a3 + t * (a2 + t * (a1 + t * a0))
        rad = tl.where(r > rc, 0.0, rad)
        am = tl.where(mm >= 0, mm, -mm)
        pmm = tl.full([BLOCK_P], 1.0, tl.float64)
        fact = tl.full([BLOCK_P], 1.0, tl.float64)
        for i in range(0, am):
            pmm = pmm * (-fact * st)
            fact = fact + 2.0
        p_prev = pmm
        p_curr = ct * (2 * am + 1).to(tl.float64) * pmm
        for ll in range(am + 2, l + 1):
            p_next = (ct * (2 * ll - 1).to(tl.float64) * p_curr - (ll + am - 1).to(tl.float64) * p_prev) \
                / (ll - am).to(tl.float64)
            p_prev = p_curr
            p_curr = p_next
        plm = tl.where(l == am, pmm, p_curr)
        ratio = tl.full([BLOCK_P], 1.0, tl.float64)
        for kk in range(l - am + 1, l + am + 1):
            ratio = ratio / kk.to(tl.float64)
        norm = tl.sqrt((2 * l + 1).to(tl.float64) / fourpi * ratio)
        ang = am.to(tl.float64) * phi
        trig = tl.where(mm > 0, libdevice.cos(ang), libdevice.sin(ang))
        y = tl.where(mm == 0, norm * plm, sqrt2 * norm * plm * trig)
        y = tl.where(small, tl.where(l == 0, y00, 0.0), y)
        val = rad * y
        val = tl.where(keep & (mu < nwv), val, 0.0)
        tl.store(obase + mu * P + p, val, mask=pmask)


def ao_tiles(tab, cen, sp, ok, org, grid, cell, ts, block_p=128):
    """AO values [B, M, W, P] on the tiles with origins org [B, 3] of the images cen [B, M, 3] (species sp, valid ok)."""
    B, M = sp.shape
    W = tab.wmax
    P = ts ** 3
    out = torch.empty((B, M, W, P), dtype=torch.float64, device=cen.device)
    g = (B * M, triton.cdiv(P, block_p))
    _ao_kernel[g](out, cen.contiguous(), sp.to(torch.int32).contiguous(), ok.to(torch.int32).contiguous(),
                  org.to(torch.int64).contiguous(), tab.coef, tab.dr, tab.rcut, tab.nint, tab.chi, tab.lq, tab.mq,
                  tab.nw, cell, _consts(cen.device), grid, M, tab.max_ch, tab.max_int, W, TS=ts, BLOCK_P=block_p)
    return out


def local_blocks(field, tab, spec, pos, frac, cell, index, n_keys, *, phi_bytes=4 << 30, tile=8, tile_large=12):
    """Local-potential blocks [n_keys, w, w] (Ry) on the rows of ``index`` (KeyIndex of the pair support) and stats.

    field: Field (values on the device); tab: AOTables; spec [n]: species row of each atom in ``tab``; pos [n, 3],
    frac [n, 3], cell [3, 3]: FP64 device tensors (bohr, positions as given).
    """
    dev = pos.device
    shape = field.shape
    N = torch.as_tensor(shape, device=dev)
    C = cell
    v = field.values.to(torch.float64).contiguous()
    W = tab.wmax
    ts = tile_large if (tile == 8 and math.prod(shape) > 24_000_000) else tile
    nt = [-(-s // ts) for s in shape]
    corners = torch.as_tensor([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=torch.float64,
                              device=dev)
    step = C / N[:, None].to(torch.float64)                   # rows: grid step vectors
    cen_off = torch.full((3,), (ts - 1) / 2.0, dtype=torch.float64, device=dev) @ step
    rt = float(((corners * (ts - 1)) @ step - cen_off).norm(dim=1).max())
    ti = torch.stack(torch.meshgrid(*[torch.arange(x, device=dev) for x in nt], indexing='ij'), -1).reshape(-1, 3)
    origin = ti * ts                                          # integer grid index of each tile corner
    tcen = origin.to(torch.float64) @ step + cen_off          # Cartesian tile centres
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
    img_atom = torch.arange(n, device=dev)[:, None].expand(n, len(Tr)).reshape(-1)
    img_T = (Tr[None, :, :] - wind[:, None, :]).reshape(-1, 3)
    img_rc = rc[img_atom]
    # (tile, image) incidences, sorted by tile
    parts = []
    chunk = max(1, int(2e7 // max(1, len(img_c))))
    for a in range(0, len(tcen), chunk):
        d = (tcen[a:a + chunk, None, :] - img_c[None]).norm(dim=-1)
        tt, ii = torch.nonzero(d <= img_rc[None] + rt + 1e-9, as_tuple=True)
        parts.append(torch.stack([tt + a, ii], 1))
    inc = torch.cat(parts)
    counts = torch.bincount(inc[:, 0], minlength=len(tcen))
    starts = torch.cumsum(counts, 0) - counts
    order = torch.argsort(counts, descending=True)
    cnt_sorted = counts[order].cpu().numpy()
    P = ts ** 3
    loc = torch.stack(torch.meshgrid(*[torch.arange(ts, device=dev)] * 3, indexing='ij'), -1).reshape(-1, 3)
    grid_t = torch.as_tensor(list(shape), dtype=torch.int64, device=dev)
    cell_t = C.reshape(-1).contiguous()
    V = torch.zeros((n_keys, W, W), dtype=torch.float64, device=dev)
    stray = torch.zeros((), dtype=torch.float64, device=dev)
    stats = dict(tiles=int(len(tcen)), tile_edge=ts, incidences=int(len(inc)), tile_radius=rt,
                 max_images=int(cnt_sorted.max()) if len(cnt_sorted) else 0, blocks_scattered=0, flops=0.)
    at = 0
    while at < len(order):
        M = int(cnt_sorted[at])
        if M == 0:
            break
        B = max(1, min(len(order) - at, phi_bytes // (M * P * W * 8 * 4)))
        tiles = order[at:at + B]
        at += B
        cnt = counts[tiles]
        idx = starts[tiles][:, None] + torch.arange(M, device=dev)[None]
        ok = torch.arange(M, device=dev)[None] < cnt[:, None]
        img = torch.where(ok, inc[idx.clamp(max=len(inc) - 1), 1], 0)
        gidx = origin[tiles][:, None, :] + loc[None]          # [B, P, 3]
        inside = (gidx < N).all(-1)                           # partial tiles at the upper faces
        gw = torch.where(inside[..., None], gidx, torch.zeros_like(gidx))
        pot = v[gw[..., 0], gw[..., 1], gw[..., 2]] * inside  # [B, P]
        A = ao_tiles(tab, img_c[img], spec[img_atom[img]], ok, origin[tiles], grid_t, cell_t, ts)
        A = A.reshape(B, M * W, P)
        G = torch.bmm(A, (A * pot[:, None, :]).transpose(1, 2))
        del A
        stats['flops'] += 2.0 * B * P * (M * W) ** 2
        G = G.reshape(B, M, W, M, W).permute(0, 1, 3, 2, 4)  # [B, M, M, w, w]
        a_img = img[:, :, None].expand(B, M, M)
        b_img = img[:, None, :].expand(B, M, M)
        pair_ok = ok[:, :, None] & ok[:, None, :]
        R = img_T[b_img] - img_T[a_img]
        q = torch.stack([img_atom[a_img], img_atom[b_img], R[..., 0], R[..., 1], R[..., 2]], -1)[pair_ok]
        blocks = G[pair_ok]
        del G
        rows = index.find(q)
        miss = rows < 0
        stray = torch.maximum(stray, torch.where(miss[:, None, None], blocks.abs(), 0.).amax() if len(blocks) else stray)
        V.index_add_(0, rows.clamp(min=0), torch.where(miss[:, None, None], 0., blocks))
        stats['blocks_scattered'] += int(len(rows))
        del blocks
    if float(stray) != 0.0:
        raise ValueError('nonzero local-potential block outside the pair support')
    return V * field.weight, stats
