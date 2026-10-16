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

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from dptb.prior.common.blocks import search_sorted
from dptb.prior.common.harmonics import harmonic_norm, real_ylm, rotation_matrices

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


def spline_eval(knots, coef, r):
    """Values [len(r), M] of the piecewise cubics ``coef`` [K - 1, 4, M] (``spline_coefficients``) at r: the interval
    by ``torch.searchsorted`` and the cubic in ``horner`` order; points outside the knots take the end intervals
    (callers mask them). The tensor twin of the kernels' spline evaluation."""
    i = (torch.searchsorted(knots, r.contiguous(), right=True) - 1).clamp(0, len(knots) - 2)
    h = (r - knots[i])[:, None]
    c = coef[i]
    return ((c[:, 0] * h + c[:, 1]) * h + c[:, 2]) * h + c[:, 3]


@triton.jit
def horner(c3, c2, c1, c0, h):
    """c3 h^3 + c2 h^2 + c1 h + c0: the cubic piece of every radial spline of both priors (Horner, highest power
    first)."""
    return ((c3 * h + c2) * h + c1) * h + c0


@triton.jit
def interval_uniform(r, inv_dr, dr, n_int):
    """Interval k = min(max(floor(r / dr), 0), n_int - 1) of a uniform knot grid (as floor(r * inv_dr)) and the offset
    h = r - k dr, k as a float of r's dtype. Non-uniform knots use ``blocks.search_sorted``."""
    k = tl.minimum(tl.maximum(libdevice.floor(r * inv_dr), 0.0), (n_int - 1).to(r.dtype))
    return k, r - k * dr


def pair_groups(left, right, ns):
    """Rows grouped by species pair for table evaluation: [(p, rows)] over the nonempty pairs p = left * ns + right
    (left, right [Q]: species index of each row; one host read of the group sizes). Both priors' two-centre steps
    group their displacements this way."""
    pair = left * ns + right
    order = torch.argsort(pair)
    bounds = torch.searchsorted(pair[order], torch.arange(ns * ns + 1, device=pair.device)).tolist()
    return [(p, order[bounds[p]:bounds[p + 1]]) for p in range(ns * ns) if bounds[p + 1] > bounds[p]]


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
    """Per pack row (l, row): degree, harmonic index, pack offset of its first column, offsets of the collocation
    directions and of the inverse (``harmonics._collocation``); the normalisation of every (l, |m|); pack width."""
    key = (lmax, str(device))
    if key not in _PACK_PLAN:
        from dptb.prior.common.harmonics import _collocation
        dirs, inv, rows, norm, e = [], [], [], np.zeros((lmax + 1, lmax + 1)), 0
        for l in range(lmax + 1):
            d, v = _collocation(l)
            n = 2 * l + 1
            for mi in range(n):
                rows.append((l, mi, e, len(dirs), len(inv)))
                e += n
            dirs += list(d)
            inv += list(v.reshape(-1))
            for m in range(l + 1):
                norm[l, m] = harmonic_norm(l, m)
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device).contiguous()  # noqa: E731
        _PACK_PLAN[key] = (t(rows, torch.int32), t(dirs, torch.float64), t(inv, torch.float64), t(norm, torch.float64),
                           e)
    return _PACK_PLAN[key]


@triton.jit(do_not_specialize=['Q', 'RS', 'LM1'])
def _pack_kernel(out, vec, prow, dirs, inv, norm, Q, RS, LM1, sign, BQ: tl.constexpr, BA: tl.constexpr):
    """out[q, e0 + a] = D^l[row, a] = sum_c inv_l[a, c] Y_l,row(R_q u_c) for every column a of pack row (l, row)
    (program axis 1; e0 its first pack element), R_q the rotation taking +z to sign * vec[q]
    (``harmonics.rotation_z_to``), u_c the collocation directions of degree l: Y_l,row(R_q u_c) is formed once per
    direction for all columns."""
    w = tl.program_id(1)
    q = tl.program_id(0) * BQ + tl.arange(0, BQ)
    mq = q < Q
    l = tl.load(prow + w * 5 + 0)
    mi = tl.load(prow + w * 5 + 1)
    e0 = tl.load(prow + w * 5 + 2)
    od = tl.load(prow + w * 5 + 3)
    oi = tl.load(prow + w * 5 + 4)
    a = tl.arange(0, BA)
    ma = a < 2 * l + 1
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
    m = (mi + 1) // 2                           # ABACUS order: 0, +1, -1, +2, -2, ... (|m|)
    nrm = tl.load(norm + l * LM1 + m)
    acc = tl.zeros([BQ, BA], dtype=tl.float64)
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
        val = real_ylm(l, tl.where(mi % 2 == 1, m, -m), ux, uy, uz, nrm)
        acc += tl.load(inv + oi + a * (2 * l + 1) + c, mask=ma, other=0.0)[None, :] * val[:, None]
    tl.store(out + q[:, None] * RS + e0 + a[None, :], acc, mask=mq[:, None] & ma[None, :])


def rotation_pack_of(vectors, lmax, sign=1.0):
    """``rotation_pack(rotation_matrices(sign * vectors, range(lmax + 1)), lmax)`` in one Triton launch."""
    prow, dirs, inv, norm, n = _pack_plan(lmax, vectors.device)
    out = torch.empty((len(vectors), n), dtype=torch.float64, device=vectors.device)
    if len(vectors):
        BQ = 64
        _pack_kernel[(triton.cdiv(len(vectors), BQ), len(prow))](
            out, vectors.contiguous(), prow, dirs, inv, norm, len(vectors), n, lmax + 1, float(sign), BQ=BQ,
            BA=triton.next_power_of_2(2 * lmax + 1), num_warps=2)
    return out


TILE_ROWS = 8
MDIAG_WARPS = 2


def tiles_per_program(rows, tiles):
    """Tiles of one displacement row per ``_mdiag_kernel`` program (measured on H200: 2, or 4 for calls with at least
    2^15 rows; one row per program leaves small calls without enough programs)."""
    return min(tiles, 4 if rows >= 1 << 15 else 2)


def _mdiag_plan(left, right, active, nr, br):
    """Plan of ``_mdiag_kernel``, or None when an active channel (k, j) is off the m-diagonal (k, j at different
    positions in their shells; two-centre blocks never are). On the diagonal, row k of the lab block's right factor
    is z_k[b] = C[k, o2(b) + p(k)] D^l2(b)[b - o2(b), p(k)] (p = position in the shell: one channel per column), so

        B[a, b] = sum_k D^l1[a - o1, p(k)] z_k[b]   over the rows k of a's shell.

    Maps: zmap [nL, br] channel of (k, o2(b) + p(k)) (-1: none); dmap [nL, br] pack offset of D^l2(b)[., p(k)];
    lmap [nL, nL] pack offset of D^l1[a - o, k - o] at [k, a] (-1 outside the shell); pos [nL] p(k); cpos [br] p(b)
    (-1: padding); per tile of TILE_ROWS rows the range [klo, khi) of rows sharing a shell with the tile."""
    def shells(ls):
        out, o = [], 0
        for l in ls:
            out.append((o, 2 * l + 1, sum((2 * m + 1) ** 2 for m in range(l))))
            o += 2 * l + 1
        return out
    sl, sr = shells(left), shells(right)
    nl = sum(n for _, n, _ in sl)
    rsh = [s for s, (_, n, _) in enumerate(sl) for _ in range(n)]
    csh = [s for s, (_, n, _) in enumerate(sr) for _ in range(n)]
    pos = np.asarray([k - sl[rsh[k]][0] for k in range(nl)], dtype=np.int32)
    chan = {}
    for c, a in enumerate(np.asarray(active).tolist()):
        k, j = divmod(int(a), nr)
        if j - sr[csh[j]][0] != pos[k]:
            return None
        chan[(k, j)] = c
    zmap = np.full((nl, br), -1, dtype=np.int32)
    dmap = np.full((nl, br), -1, dtype=np.int32)
    for k in range(nl):
        for b in range(nr):
            o2, n2, b2 = sr[csh[b]]
            if pos[k] < n2 and (k, o2 + pos[k]) in chan:
                zmap[k, b] = chan[(k, o2 + pos[k])]
                dmap[k, b] = b2 + (b - o2) * n2 + pos[k]
    lmap = np.full((nl, nl), -1, dtype=np.int32)
    for o, n, b1 in sl:
        for a in range(n):
            lmap[o:o + n, o + a] = b1 + a * n + np.arange(n)
    cpos = np.full(br, -1, dtype=np.int32)
    cpos[:nr] = [b - sr[csh[b]][0] for b in range(nr)]
    tiles = range(0, nl, TILE_ROWS)
    klo = np.asarray([sl[rsh[t]][0] for t in tiles], dtype=np.int32)
    khi = np.asarray([sum(sl[rsh[min(t + TILE_ROWS, nl) - 1]][:2]) for t in tiles], dtype=np.int32)
    return zmap, dmap, lmap, pos, cpos, klo, khi


@triton.jit(do_not_specialize=['NA', 'K', 'RS', 'OS0', 'OS1', 'OS2', 'LOGK', 'NL', 'NR', 'NT', 'TP'])
def _mdiag_kernel(out, rows, vec, rpack, knots, coef, zmap, dmap, lmap, pos, cpos, klo, khi, consts, NA, K, NL, NR,
                  NT, TP, RS, OS0, OS1, OS2, LOGK, sign, FLIP: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr):
    """Tiles [g TP, g TP + TP) of BT rows (of NT) of out[q] = D_L(d) C(|d|) D_R(d)^T (transposed when FLIP) for the
    displacement row q = rows[p // G], g = p % G, G = cdiv(NT, TP) (p the program), for m-diagonal tables
    (``_mdiag_plan``): one channel per (row, column), so each row k of C costs BR products and one rank-1 update; zero
    beyond the support (consts[0]); C at |d| <= 1e-14. The tiles of a row run in consecutive programs (or in one), so
    the coefficients of its knot interval are read once from memory."""
    G = (NT + TP - 1) // TP
    p = tl.program_id(0)
    q = tl.load(rows + p // G)
    g = p % G
    x = sign * tl.load(vec + q * 3 + 0)
    y = sign * tl.load(vec + q * 3 + 1)
    z = sign * tl.load(vec + q * 3 + 2)
    r = tl.sqrt(x * x + y * y + z * z)
    i = tl.maximum(search_sorted(knots, K, r, LOGK, r == r, RIGHT=True), 1) - 1     # knot interval of r
    d = r - tl.load(knots + i)
    j = tl.arange(0, BR)
    inside = r < tl.load(consts)
    centre = r <= 1e-14
    cb = coef + i.to(tl.int64) * 4 * NA
    rp = rpack + q * RS
    cp = tl.load(cpos + j)
    for t in range(g * TP, tl.minimum(g * TP + TP, NT)):
        a = t * BT + tl.arange(0, BT)
        ma = a < NL
        acc = tl.zeros([BT, BR], dtype=tl.float64)
        for k in range(tl.load(klo + t), tl.load(khi + t)):
            cm = tl.load(zmap + k * BR + j)
            mk = cm >= 0
            c = horner(tl.load(cb + cm, mask=mk, other=0.0), tl.load(cb + NA + cm, mask=mk, other=0.0),
                       tl.load(cb + 2 * NA + cm, mask=mk, other=0.0), tl.load(cb + 3 * NA + cm, mask=mk, other=0.0), d)
            c = tl.where(inside, c, 0.0)
            dm = tl.load(dmap + k * BR + j)
            dr = tl.where(centre, (cp == tl.load(pos + k)).to(tl.float64), tl.load(rp + dm, mask=dm >= 0, other=0.0))
            lm = tl.load(lmap + k * NL + a, mask=ma, other=-1)
            dl = tl.where(centre, (a == k).to(tl.float64), tl.load(rp + lm, mask=lm >= 0, other=0.0))
            acc += dl[:, None] * (c * dr)[None, :]
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
        act = active.cpu().numpy()
        self._br = max(8, triton.next_power_of_2(nr))
        plan = _mdiag_plan(self.left, self.right, act, nr, self._br)
        self._mdiag = None if plan is None else tuple(torch.as_tensor(m, device=dev).contiguous() for m in plan)
        self._consts = torch.tensor([self.support - 1e-12], dtype=torch.float64, device=dev)

    @property
    def nbytes(self):
        return self.coefficients.numel() * self.coefficients.element_size() + self.knots.numel() * 8

    def canonical(self, r):
        """Bond-frame blocks C(r) [Q, nL, nR] (zero beyond the support)."""
        v = spline_eval(self.knots, self.coefficients, r)
        v = torch.where((r < self.support - 1e-12)[:, None], v, torch.zeros_like(v))
        out = v.new_zeros((len(r), self.shape[0] * self.shape[1]))
        out[:, self.active] = v
        return out.reshape(-1, *self.shape)

    def into(self, out, rows, vectors, rpack, flip=False):
        """out[rows] = blocks at ``vectors[rows]`` (bohr; at -vectors[rows], transposed, when ``flip``) in one fused
        kernel over the active channels only (``_mdiag_plan``); ``rpack`` the rotation pack of the vectors used
        (``rotation_pack``; of -vectors when ``flip``). out is FP64 [Q, >= nL, >= nR] (>= nR, >= nL when ``flip``);
        entries outside the block are left untouched."""
        if not len(rows):
            return
        K = len(self.knots)
        if self._mdiag is not None:
            nt = len(self._mdiag[5])
            tp = tiles_per_program(len(rows), nt)
            _mdiag_kernel[(len(rows) * triton.cdiv(nt, tp),)](
                out, rows, vectors, rpack, self.knots, self.coefficients, *self._mdiag, self._consts,
                self.coefficients.shape[-1], K, self.shape[0], self.shape[1], nt, tp, rpack.stride(0),
                out.stride(0), out.stride(1), out.stride(2), int(K).bit_length() + 1, -1.0 if flip else 1.0,
                FLIP=bool(flip), BT=TILE_ROWS, BR=self._br, num_warps=MDIAG_WARPS)
            return
        v = vectors[rows]          # off the m-diagonal (no two-centre table is): per-degree path
        rot, o = {}, 0
        for l in range(max(self.degrees) + 1):
            n = (2 * l + 1) ** 2
            rot[l] = rpack[rows, o:o + n].reshape(-1, 2 * l + 1, 2 * l + 1)
            o += n
        v = self(-v if flip else v, rot=rot)
        v = v.transpose(1, 2) if flip else v
        out[rows, :v.shape[1], :v.shape[2]] = v

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
