"""Radial block tables on the device: bond-frame two-centre blocks interpolated in |d| and rotated to the lab frame.

A table stores, on a knot grid of distances, the block C(|d|) [nL, nR] of a two-centre quantity for a displacement
along +z (shells ``left`` / ``right``, ABACUS real harmonics). The value at a displacement d is

    B(d) = D_L(d^) C(|d|) D_R(d^)^T,   zero for |d| >= support - 1e-12,

where D_L, D_R are block diagonal in the shells, with the real-harmonic rotation matrix D^l of the rotation taking +z
to d^ (``harmonics.rotation_matrices``). C is the not-a-knot cubic spline of the tabulated values (``scipy``'s
``CubicSpline`` default), or the linear interpolant of tables declared linear. Splines are fitted on the device: the
tridiagonal system depends on the knots only and is factorised once per grid.
"""
import hashlib
import math

import numpy as np
import torch
import triton
import triton.language as tl

from dptb.prior.common.harmonics import rotation_matrices

_LU = {}


def _not_a_knot_lu(x, grid_key):
    """LU factors of the slope system of a not-a-knot cubic spline on knots x (FP64 device tensor, len >= 4)."""
    key = (str(x.device), grid_key)
    if key not in _LU:
        n = len(x)
        dx = x[1:] - x[:-1]
        a = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        i = torch.arange(1, n - 1, device=x.device)
        a[i, i] = 2 * (dx[:-1] + dx[1:])
        a[i, i + 1] = dx[:-1]
        a[i, i - 1] = dx[1:]
        a[0, 0], a[0, 1] = dx[1], x[2] - x[0]
        a[n - 1, n - 1], a[n - 1, n - 2] = dx[-2], x[-1] - x[-3]
        _LU[key] = torch.linalg.lu_factor(a)
    return _LU[key]


def spline_coefficients(x, y, interpolation='cubic', grid_key=None):
    """Piecewise cubic coefficients [K - 1, 4, M] (powers 3, 2, 1, 0 in d - x_i) of values y [K, M] on knots x [K].

    ``grid_key`` identifies the knot grid for the factorisation cache (default: a digest of the knots).
    """
    dx = (x[1:] - x[:-1])[:, None]
    slope = (y[1:] - y[:-1]) / dx
    c = torch.zeros((len(x) - 1, 4, y.shape[1]), dtype=y.dtype, device=y.device)
    if interpolation == 'linear' or len(x) < 4:
        if interpolation != 'linear':
            raise ValueError('cubic radial tables need at least four knots')
        c[:, 2], c[:, 3] = slope, y[:-1]
        return c
    b = torch.empty_like(y)
    b[1:-1] = 3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:])
    d0 = x[2] - x[0]
    b[0] = ((dx[0] + 2 * d0) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]) / d0
    d1 = x[-1] - x[-3]
    b[-1] = (dx[-1] ** 2 * slope[-2] + (2 * d1 + dx[-1]) * dx[-2] * slope[-1]) / d1
    if grid_key is None:
        grid_key = hashlib.sha1(x.cpu().numpy().tobytes()).hexdigest()
    lu, piv = _not_a_knot_lu(x, grid_key)
    s = torch.linalg.lu_solve(lu, piv, b)
    t = (s[:-1] + s[1:] - 2 * slope) / dx
    c[:, 0] = t / dx
    c[:, 1] = (slope - s[:-1]) / dx - t
    c[:, 2] = s[:-1]
    c[:, 3] = y[:-1]
    return c


def _shell_index(shells):
    """{l: AO indices [n_shells_l, 2l + 1]} of a shell list."""
    out, o = {}, 0
    for l in shells:
        out.setdefault(l, []).append(list(range(o, o + 2 * l + 1)))
        o += 2 * l + 1
    return out, o


def rotation_pack(rot, lmax):
    """[Q, sum_l (2l + 1)^2] row-major D^0..D^lmax of each displacement (the layout ``RadialTable.into`` reads)."""
    return torch.cat([rot[l].reshape(len(rot[l]), -1) for l in range(lmax + 1)], 1).contiguous()


_PACK_PLAN = {}


def _pack_plan(lmax, device):
    """Per pack element e = (l, row, column a): degree, harmonic index, column, offsets of the collocation directions
    and of the inverse (``harmonics._collocation``), and the normalisation of every (l, |m|)."""
    key = (lmax, str(device))
    if key not in _PACK_PLAN:
        from dptb.prior.common.harmonics import _collocation
        dirs, inv, elems, norm = [], [], [], np.zeros((lmax + 1, lmax + 1))
        for l in range(lmax + 1):
            d, v = _collocation(l)
            n = 2 * l + 1
            for mi in range(n):
                for a in range(n):
                    elems.append((l, mi, a, len(dirs), len(inv)))
            dirs += list(d)
            inv += list(v.reshape(-1))
            for m in range(l + 1):
                norm[l, m] = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device).contiguous()  # noqa: E731
        _PACK_PLAN[key] = (t(elems, torch.int32), t(dirs, torch.float64), t(inv, torch.float64), t(norm, torch.float64))
    return _PACK_PLAN[key]


@triton.jit(do_not_specialize=['Q', 'RS', 'LM1'])
def _pack_kernel(out, vec, elems, dirs, inv, norm, Q, RS, LM1, sign, BQ: tl.constexpr):
    """out[q, e] = D^l[row, a] = sum_c inv_l[a, c] Y_l,row(R_q u_c) for pack element e (program axis 1), R_q the
    rotation taking +z to sign * vec[q] (``harmonics.rotation_z_to``), u_c the collocation directions of degree l."""
    e = tl.program_id(1)
    q = tl.program_id(0) * BQ + tl.arange(0, BQ)
    mq = q < Q
    l = tl.load(elems + e * 5 + 0)
    mi = tl.load(elems + e * 5 + 1)
    a = tl.load(elems + e * 5 + 2)
    od = tl.load(elems + e * 5 + 3)
    oi = tl.load(elems + e * 5 + 4)
    vx = sign * tl.load(vec + q * 3 + 0, mask=mq, other=0.0)
    vy = sign * tl.load(vec + q * 3 + 1, mask=mq, other=0.0)
    vz = sign * tl.load(vec + q * 3 + 2, mask=mq, other=1.0)
    radius = tl.sqrt(vx * vx + vy * vy + vz * vz)
    nr = tl.maximum(radius, 1e-30)
    x = vx / nr
    y = vy / nr
    z = vz / nr
    f = tl.where(z < 0, (1 - z) / tl.maximum(x * x + y * y, 1e-30), 1 / tl.maximum(1 + z, 1e-30))
    r00 = 1 + (-(x * x)) * f
    r01 = (-(x * y)) * f
    r11 = 1 + (-(y * y)) * f
    r22 = 1 + (-(x * x) - y * y) * f
    r02 = x
    r12 = y
    r20 = -x
    r21 = -y
    south = z <= -1 + 1e-14
    ident = (z >= 1 - 1e-14) | (radius <= 1e-14)
    r00 = tl.where(ident | south, 1.0, r00)
    r11 = tl.where(ident, 1.0, tl.where(south, -1.0, r11))
    r22 = tl.where(ident, 1.0, tl.where(south, -1.0, r22))
    zero = ident | south
    r01 = tl.where(zero, 0.0, r01)
    r02 = tl.where(zero, 0.0, r02)
    r12 = tl.where(zero, 0.0, r12)
    r20 = tl.where(zero, 0.0, r20)
    r21 = tl.where(zero, 0.0, r21)
    m = (mi + 1) // 2                           # ABACUS order: 0, +1, -1, +2, -2, ...
    nrm = tl.load(norm + l * LM1 + m)
    acc = tl.zeros([BQ], dtype=tl.float64)
    for c in range(2 * l + 1):
        dx = tl.load(dirs + (od + c) * 3 + 0)
        dy = tl.load(dirs + (od + c) * 3 + 1)
        dz = tl.load(dirs + (od + c) * 3 + 2)
        ux = r00 * dx + r01 * dy + r02 * dz
        uy = r01 * dx + r11 * dy + r12 * dz
        uz = r20 * dx + r21 * dy + r22 * dz
        un = tl.maximum(tl.sqrt(ux * ux + uy * uy + uz * uz), 1e-30)
        ux = ux / un
        uy = uy / un
        uz = uz / un
        re = tl.full([BQ], 1.0, tl.float64)
        im = tl.zeros([BQ], dtype=tl.float64)
        dfac = 1.0
        for k in range(m):                      # (x + i y)^m and (2m - 1)!!
            re, im = re * ux - im * uy, im * ux + re * uy
            dfac = dfac * (2 * k + 1)
        sgn = 1.0 - 2.0 * (m % 2)
        p = tl.full([BQ], 1.0, tl.float64) * (sgn * dfac)
        if l > m:
            prev = p
            p = (2 * m + 1) * uz * p
            for deg in range(m + 2, l + 1):
                prev, p = p, ((2 * deg - 1) * uz * p - (deg + m - 1) * prev) / (deg - m)
        val = tl.where(m == 0, nrm * p, tl.where(mi % 2 == 1, 1.4142135623730951 * nrm * p * re,
                                                  1.4142135623730951 * nrm * p * im))
        acc += tl.load(inv + oi + a * (2 * l + 1) + c) * val
    tl.store(out + q * RS + e, acc, mask=mq)


def rotation_pack_of(vectors, lmax, sign=1.0):
    """``rotation_pack(rotation_matrices(sign * vectors, range(lmax + 1)), lmax)`` in one Triton launch."""
    elems, dirs, inv, norm = _pack_plan(lmax, vectors.device)
    n = len(elems)
    out = torch.empty((len(vectors), n), dtype=torch.float64, device=vectors.device)
    if len(vectors):
        BQ = 64
        _pack_kernel[(triton.cdiv(len(vectors), BQ), n)](out, vectors.contiguous(), elems, dirs, inv, norm,
                                                         len(vectors), n, lmax + 1, float(sign), BQ=BQ, num_warps=2)
    return out


def _rot_map(shells, n):
    """[n, n] int32: offset of D^l[a - o, i - o] in a rotation pack row for AOs a, i of one shell (o its first AO),
    -1 elsewhere."""
    out = np.full((n, n), -1, dtype=np.int32)
    o = 0
    for l in shells:
        base = sum((2 * m + 1) ** 2 for m in range(l))
        for a in range(2 * l + 1):
            out[o + a, o:o + 2 * l + 1] = base + a * (2 * l + 1) + np.arange(2 * l + 1)
        o += 2 * l + 1
    return out


TALL_ROWS = 16


def _tall_maps(left, right, active, br):
    """Maps of the row-tiled kernel: canonical (k, j) -> channel [nL, br]; left rotation map [nL, nL]; right map
    [br, br]; per tile of TALL_ROWS rows the range [klo, khi) of canonical rows sharing a shell with the tile."""
    nl, nr = sum(2 * l + 1 for l in left), sum(2 * l + 1 for l in right)
    cmap = np.full((nl, br), -1, dtype=np.int32)
    cmap[active // nr, active % nr] = np.arange(len(active), dtype=np.int32)
    rmap = np.full((br, br), -1, dtype=np.int32)
    rmap[:nr, :nr] = _rot_map(right, nr)
    first, end, o = [], [], 0
    for l in left:
        first += [o] * (2 * l + 1)
        end += [o + 2 * l + 1] * (2 * l + 1)
        o += 2 * l + 1
    tiles = range(0, nl, TALL_ROWS)
    klo = np.asarray([first[t] for t in tiles], dtype=np.int32)
    khi = np.asarray([end[min(t + TALL_ROWS, nl) - 1] for t in tiles], dtype=np.int32)
    return cmap, _rot_map(left, nl), rmap, klo, khi


def _block_maps(left, right, active, bw):
    """Maps of the square kernel, each [bw, bw] int32 (-1 = zero): canonical (i, j) -> active channel; left and right
    rotation maps (``_rot_map``) padded to bw."""
    nl, nr = sum(2 * l + 1 for l in left), sum(2 * l + 1 for l in right)
    cmap = np.full((bw, bw), -1, dtype=np.int32)
    cmap[active // nr, active % nr] = np.arange(len(active), dtype=np.int32)
    lmap = np.full((bw, bw), -1, dtype=np.int32)
    rmap = np.full((bw, bw), -1, dtype=np.int32)
    lmap[:nl, :nl] = _rot_map(left, nl)
    rmap[:nr, :nr] = _rot_map(right, nr)
    return cmap, lmap, rmap


@triton.jit(do_not_specialize=['NA', 'K', 'RS', 'OS0', 'OS1', 'OS2', 'LOGK', 'NL', 'NR'])
def _into_kernel(out, rows, vec, rpack, knots, coef, cmap, lmap, rmap, consts, NA, K, NL, NR, RS, OS0, OS1, OS2, LOGK,
                 sign, FLIP: tl.constexpr, BW: tl.constexpr):
    """out[q] = D_L(d) C(|d|) D_R(d)^T for the displacement row q = rows[program] (transposed when FLIP): C the
    spline of the canonical block gathered row by row through ``cmap``, D_L, D_R gathered from the rotation pack
    through ``lmap``, ``rmap``; zero beyond the support (consts[0]); C itself at |d| <= 1e-14."""
    q = tl.load(rows + tl.program_id(0))
    x = sign * tl.load(vec + q * 3 + 0)
    y = sign * tl.load(vec + q * 3 + 1)
    z = sign * tl.load(vec + q * 3 + 2)
    r = tl.sqrt(x * x + y * y + z * z)
    lo = 0
    hi = K
    for _ in range(LOGK):                       # searchsorted(knots, r, right=True)
        mid = (lo + hi) // 2
        kv = tl.load(knots + tl.minimum(mid, K - 1))
        go = (lo < hi) & (kv <= r)
        stay = (lo < hi) & (kv > r)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(stay, mid, hi)
    i = tl.minimum(tl.maximum(lo, 1), K - 1) - 1
    d = r - tl.load(knots + i)
    a = tl.arange(0, BW)
    inside = r < tl.load(consts)
    centre = r <= 1e-14
    cb = coef + i.to(tl.int64) * 4 * NA
    rp = rpack + q * RS
    rm = tl.load(rmap + a[None, :] * BW + a[:, None])          # D_R^T [j, b]
    dr = tl.where(centre, (a[:, None] == a[None, :]).to(tl.float64), tl.load(rp + rm, mask=rm >= 0, other=0.0))
    acc = tl.zeros([BW, BW], dtype=tl.float64)
    for k in range(NL):                         # acc += D_L[:, k] (C[k, :] D_R^T)
        cm = tl.load(cmap + k * BW + a)
        mk = cm >= 0
        c = ((tl.load(cb + cm, mask=mk, other=0.0) * d + tl.load(cb + NA + cm, mask=mk, other=0.0)) * d +
             tl.load(cb + 2 * NA + cm, mask=mk, other=0.0)) * d + tl.load(cb + 3 * NA + cm, mask=mk, other=0.0)
        c = tl.where(inside, c, 0.0)
        zr = tl.sum(c[:, None] * dr, axis=0)
        lm = tl.load(lmap + a * BW + k)
        dl = tl.where(centre, (a == k).to(tl.float64), tl.load(rp + lm, mask=lm >= 0, other=0.0))
        acc += dl[:, None] * zr[None, :]
    if FLIP:
        ptr = out + q * OS0 + a[:, None] * OS2 + a[None, :] * OS1
    else:
        ptr = out + q * OS0 + a[:, None] * OS1 + a[None, :] * OS2
    tl.store(ptr, acc, mask=(a < NL)[:, None] & (a < NR)[None, :])


@triton.jit(do_not_specialize=['NA', 'K', 'RS', 'OS0', 'OS1', 'OS2', 'LOGK', 'NL', 'NR'])
def _into_tall_kernel(out, rows, vec, rpack, knots, coef, cmap, lmap, rmap, klo, khi, consts, NA, K, NL, NR, RS, OS0,
                      OS1, OS2, LOGK, sign, FLIP: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr):
    """``_into_kernel`` for tables with more than 32 rows: program (q, t) writes rows [t BT, t BT + BT) of block q,
    contracting only the canonical rows [klo[t], khi[t]) that share a shell with them."""
    q = tl.load(rows + tl.program_id(0))
    t = tl.program_id(1)
    x = sign * tl.load(vec + q * 3 + 0)
    y = sign * tl.load(vec + q * 3 + 1)
    z = sign * tl.load(vec + q * 3 + 2)
    r = tl.sqrt(x * x + y * y + z * z)
    lo = 0
    hi = K
    for _ in range(LOGK):                       # searchsorted(knots, r, right=True)
        mid = (lo + hi) // 2
        kv = tl.load(knots + tl.minimum(mid, K - 1))
        go = (lo < hi) & (kv <= r)
        stay = (lo < hi) & (kv > r)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(stay, mid, hi)
    i = tl.minimum(tl.maximum(lo, 1), K - 1) - 1
    d = r - tl.load(knots + i)
    j = tl.arange(0, BR)
    a = t * BT + tl.arange(0, BT)
    ma = a < NL
    inside = r < tl.load(consts)
    centre = r <= 1e-14
    cb = coef + i.to(tl.int64) * 4 * NA
    rp = rpack + q * RS
    rm = tl.load(rmap + j[None, :] * BR + j[:, None])          # D_R^T [j, b]
    dr = tl.where(centre, (j[:, None] == j[None, :]).to(tl.float64), tl.load(rp + rm, mask=rm >= 0, other=0.0))
    acc = tl.zeros([BT, BR], dtype=tl.float64)
    for k in range(tl.load(klo + t), tl.load(khi + t)):
        cm = tl.load(cmap + k * BR + j)
        mk = cm >= 0
        c = ((tl.load(cb + cm, mask=mk, other=0.0) * d + tl.load(cb + NA + cm, mask=mk, other=0.0)) * d +
             tl.load(cb + 2 * NA + cm, mask=mk, other=0.0)) * d + tl.load(cb + 3 * NA + cm, mask=mk, other=0.0)
        c = tl.where(inside, c, 0.0)
        zr = tl.sum(c[:, None] * dr, axis=0)
        lm = tl.load(lmap + a * NL + k, mask=ma, other=-1)
        dl = tl.where(centre, (a == k).to(tl.float64), tl.load(rp + lm, mask=lm >= 0, other=0.0))
        acc += dl[:, None] * zr[None, :]
    if FLIP:
        ptr = out + q * OS0 + a[:, None] * OS2 + j[None, :] * OS1
    else:
        ptr = out + q * OS0 + a[:, None] * OS1 + j[None, :] * OS2
    tl.store(ptr, acc, mask=ma[:, None] & (j < NR)[None, :])


class RadialTable:
    """One radial block table resident on the device (see module docstring)."""

    def __init__(self, distances, values, left, right, support, interpolation='cubic', device='cuda',
                 dtype=torch.float64):
        dev = torch.device(device)
        self.left, self.right = tuple(int(l) for l in left), tuple(int(l) for l in right)
        li, nl = _shell_index(self.left)
        ri, nr = _shell_index(self.right)
        values = np.asarray(values)
        if values.shape != (len(distances), nl, nr) or not np.isfinite(values).all():
            raise ValueError('radial table values do not match the shells')
        self.shape = (nl, nr)
        self.support = float(support)
        distances = np.asarray(distances, dtype=np.float64)
        self.knots = torch.as_tensor(distances, dtype=dtype, device=dev)
        y = torch.as_tensor(values.reshape(len(distances), -1), device=dev).to(dtype)
        c = spline_coefficients(self.knots, y, interpolation, hashlib.sha1(distances.tobytes()).hexdigest())
        active = torch.nonzero(c.ne(0).any(0).any(0), as_tuple=True)[0]
        self.active = active
        self.coefficients = c[:, :, active].contiguous()
        self.left_index = {l: torch.as_tensor(v, device=dev) for l, v in li.items()}
        self.right_index = {l: torch.as_tensor(v, device=dev) for l, v in ri.items()}
        self.degrees = sorted(set(self.left) | set(self.right))
        self._bw = max(16, triton.next_power_of_2(max(self.shape)))
        self._br = max(16, triton.next_power_of_2(self.shape[1]))
        act = active.cpu().numpy()
        if self._bw <= 32:
            self._maps = tuple(torch.as_tensor(m, device=dev) for m in _block_maps(self.left, self.right, act, self._bw))
        elif self._br <= 32:
            self._maps = tuple(torch.as_tensor(m, device=dev) for m in _tall_maps(self.left, self.right, act, self._br))
        else:
            self._maps = None
        self._consts = torch.tensor([self.support - 1e-12], dtype=torch.float64, device=dev)

    @property
    def nbytes(self):
        return self.coefficients.numel() * self.coefficients.element_size() + self.knots.numel() * 8

    def canonical(self, r):
        """Bond-frame blocks C(r) [Q, nL, nR] (zero beyond the support)."""
        i = (torch.searchsorted(self.knots, r.contiguous(), right=True).clamp(1, len(self.knots) - 1) - 1)
        d = (r - self.knots[i])[:, None]
        c = self.coefficients[i]
        v = ((c[:, 0] * d + c[:, 1]) * d + c[:, 2]) * d + c[:, 3]
        v = torch.where((r < self.support - 1e-12)[:, None], v, torch.zeros_like(v))
        out = v.new_zeros((len(r), self.shape[0] * self.shape[1]))
        out[:, self.active] = v
        return out.reshape(-1, *self.shape)

    def into(self, out, rows, vectors, rpack, flip=False):
        """out[rows] = blocks at ``vectors[rows]`` (bohr; at -vectors[rows], transposed, when ``flip``) in one fused
        kernel; ``rpack`` the rotation pack of the vectors used (``rotation_pack``; of -vectors when ``flip``). out
        is FP64 [Q, >= nL, >= nR] (>= nR, >= nL when ``flip``); entries outside the block are left untouched.
        Blocks up to 32 x 32 take the fused kernel, taller ones (up to 32 columns) its row-tiled variant, wider ones
        the per-degree einsum path."""
        if not len(rows):
            return
        K = len(self.knots)
        if self._bw > 32 and self._maps is not None:      # tall tables (projector, VNA and density factors)
            _into_tall_kernel[(len(rows), len(self._maps[3]))](
                out, rows, vectors, rpack, self.knots, self.coefficients, *self._maps, self._consts,
                self.coefficients.shape[-1], K, self.shape[0], self.shape[1], rpack.stride(0), out.stride(0),
                out.stride(1), out.stride(2), int(K).bit_length() + 1, -1.0 if flip else 1.0, FLIP=bool(flip),
                BT=TALL_ROWS, BR=self._br, num_warps=2)
            return
        if self._bw > 32:       # wide tables: per-degree path
            v = vectors[rows]
            rot, o = {}, 0
            for l in range(max(self.degrees) + 1):
                n = (2 * l + 1) ** 2
                rot[l] = rpack[rows, o:o + n].reshape(-1, 2 * l + 1, 2 * l + 1)
                o += n
            v = self(-v if flip else v, rot=rot)
            v = v.transpose(1, 2) if flip else v
            out[rows, :v.shape[1], :v.shape[2]] = v
            return
        _into_kernel[(len(rows),)](out, rows, vectors, rpack, self.knots, self.coefficients, *self._maps, self._consts,
                                   self.coefficients.shape[-1], K, self.shape[0], self.shape[1], rpack.stride(0),
                                   out.stride(0), out.stride(1), out.stride(2), int(K).bit_length() + 1,
                                   -1.0 if flip else 1.0, FLIP=bool(flip), BW=self._bw, num_warps=4)

    def __call__(self, vectors, rot=None, chunk=None):
        """Lab-frame blocks [Q, nL, nR] at displacements ``vectors`` [Q, 3] (bohr).

        ``rot``: rotation matrices of these vectors ({l: [Q, 2l + 1, 2l + 1]}, ``harmonics.rotation_matrices``) shared
        by several tables of the same displacements; computed here when absent.
        """
        if not len(vectors):
            return vectors.new_zeros((0, *self.shape))
        if chunk is None:   # about 512 MiB of temporaries per chunk
            chunk = max(1024, (512 << 20) // (8 * (4 * self.coefficients.shape[-1] + 3 * self.shape[0] * self.shape[1])))
        if rot is None:
            rot = rotation_matrices(vectors, self.degrees)     # all queries at once: few large kernels
        out = vectors.new_empty((len(vectors), *self.shape))
        for a in range(0, len(vectors), chunk):
            sl = slice(a, a + chunk)
            r = torch.linalg.vector_norm(vectors[sl], dim=-1)
            c = self.canonical(r)
            x = torch.empty_like(c)
            for l, idx in self.left_index.items():
                x[:, idx] = torch.einsum('qab,qsbn->qsan', rot[l][sl], c[:, idx])
            y = torch.empty_like(x)
            for l, idx in self.right_index.items():
                y[:, :, idx] = torch.einsum('qab,qmsb->qmsa', rot[l][sl], x[:, :, idx])
            out[sl] = torch.where((r <= 1e-14)[:, None, None], c, y)
        return out
