"""Onsite XC of the GF prior on atom-centred quadratures, on the device.

For atom a with neighbours J (all periodic images with |d_J| < 27 bohr, a itself included)

    V_a[k, l] = sum_g w_g phi_k(x_g) phi_l(x_g) v_xc(rho_a(x_g)),   rho_a(x) = sum_J rho_sJ(|x - d_J|),

on the product grid of order (nr, nmu, nphi) inside the orbital sphere r < r_c of a's species: Gauss-Legendre radial
nodes on (0, r_c), Gauss-Legendre mu = cos(theta), uniform phi; w = w_r r^2 w_mu 2 pi / nphi. phi_k = R_ch(r) Y_lm,
R_ch the not-a-knot spline of the normalised ORB radial, Y the ABACUS real harmonics. rho_s is the neutral valence
density of the UPF (PP_RHOATOM scaled to Z_val, r = 0 repaired) plus the unscaled NLCC, each a not-a-knot spline
clamped at 0, zero beyond the last knot (the NLCC spline also beyond its last interval that can exceed 1e-24
e/bohr^3: its ringing tail is below the FP64 resolution of any total density). v_xc is LDA exchange + PZ81
correlation (eV), zero for rho <= 1e-20.

Kernels: one Triton kernel evaluates rho and v_xc for every (atom, grid point), looping over the atom's neighbours
sorted by D - L (distance minus density reach): per radial node r, those with D + r <= L reach the whole shell and
need no test, those with D - r <= L are tested per angular tile and per point, the rest are never visited;
the Gram contraction uses the product structure of the grid: A_i[ab] = sum_j w_j Y_a Y_b v_ij (one GEMM over the
directions, only a <= b since A_i is symmetric), then V[k, l] = sum_i w_i R_k(r_i) R_l(r_i) A_i[lm(k), lm(l)] (one GEMM
over the radial nodes).
Mixed precision (``OnsiteXC(mixed=True)``, the default): neighbours at least ``SPLIT`` bohr from every point of a
(compact) angular tile are summed in FP32 from FP32 copies of the same tables; geometry tests, near neighbours, the
total density and v_xc stay FP64. On 147 structures (12,923 atoms) the onsite blocks moved by at most 1.1e-8 eV and
no order decision changed.

Order policy (accepted 1 meV rule, refined): every atom at medium and fine; atoms with max|medium - fine| < 1e-3 eV use
medium; the others use fine and are checked against finer; unresolved atoms walk the extra orders and take the lower
order of the first step below 1e-3 eV (fine when none converges).

Order screen (``OnsiteXC(screen=True)``, the default): the medium/fine difference comes from the sharp densities of the
atom itself and of neighbours inside or near its orbital sphere. The same difference evaluated with only the neighbours
within rc + SCREEN_DELTA (5-28 % of the rows) stayed within a factor 0.62-1.07 below the full one (full / near) on 2,651
atoms of 48 structures (crystals, disordered cells, molecular, layered, H-bonded, heavy-element and strained cells; a
4 bohr margin gave 0.75-1.08 with a third more rows). Atoms whose near-neighbour difference is below SCREEN_LO = 5e-4 eV
(half the 1e-3 eV threshold) take medium without the fine grid;
every other atom goes through the full policy above, so the output equals the unscreened one unless the screen misses
a fine atom (none in that set, nor in 147 structures with 12,923 atoms). The fine/finer check of the fine atoms is screened the same way (near-neighbour
difference within a factor 0.97-1.01 of the full one on 792 fine atoms; none needed an order above fine). The
indicator only selects atoms, so it is evaluated entirely in FP32 (points, neighbour rows, density and v_xc; the FP64
v_xc alone, two pow and a log per point, cost more than the ~30 neighbour rows of a point): its error (below 1e-5 eV)
is far inside the margin between SCREEN_LO and the smallest indicator of an atom at the threshold (0.75 x 1e-3 eV).
"""
import hashlib
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from dptb.prior.common.harmonics import real_harmonics
from dptb.prior.common.neighbors import image_pairs
from dptb.prior.common.radial import spline_coefficients

RY = 13.605693122994
RADIUS = 27.0
MEDIUM, FINE, FINER = (128, 24, 48), (256, 40, 80), (384, 56, 112)
EXTRA = ((512, 72, 144), (640, 88, 176), (768, 104, 208))
TOLERANCE = 1e-3


def _consts(device):
    v = [3.0 / (4.0 * math.pi), 1.0 / 3.0, 3.0 / math.pi, 0.0311, 0.048, 0.002, 0.0116, 0.1423, 1.0529, 0.3334,
         2.0 * RY, 1e-20]
    return torch.tensor(v, dtype=torch.float64, device=device)


TM, TP, WARPS = 4, 16, 1    # angular tile (mu x phi) and warps of the density kernel
REACH_EPS = 1e-9            # bohr; margin of the radial reach classes (the per-point test stays exact)
NLCC_FLOOR = 1e-24          # e/bohr^3; NLCC spline tail (ringing beyond the core data) treated as zero
# Mixed precision of the density sum: a neighbour at least SPLIT bohr from every point of a tile contributes in FP32
# (its density there is a smooth tail, a few per cent of the total); MIXED_TILE is the compact tile of that mode.
SPLIT = 6.0
MIXED_TILE = (8, 8)
MIXED_WARPS = 1
SCREEN_DELTA = 3.0          # bohr; near neighbours of the order screen: |d_J| <= rc + SCREEN_DELTA
SCREEN_LO = 5e-4            # eV; near-neighbour medium/fine difference below which an atom takes medium directly
SCREEN_TILE, SCREEN_WARPS = (8, 16), 1  # angular tile and warps of the indicator kernel


@triton.jit
def _density_add(rho, d2, s, idr_t, dr_t, kn_t, knl_t, coef, KM, inr):
    """rho + max(spline_val, 0) + max(spline_nlcc, 0) at the distance sqrt(d2) for the points with ``inr``;
    coefficients [species, channel, interval, 4] are read as 16-byte pairs, the NLCC ones only up to the species' last
    significant NLCC interval (most points see only the valence table, four intervals per cache line)."""
    idr = tl.load(idr_t + s)
    dr = tl.load(dr_t + s)
    nk = tl.load(kn_t + s)
    knl = tl.load(knl_t + s)
    d = tl.sqrt(d2)
    kf = tl.minimum(tl.maximum(libdevice.floor(d * idr), 0.0), (nk - 2).to(tl.float64))
    h = d - kf * dr
    base = coef + ((s * 2 * KM + kf.to(tl.int32)) * 4)[:, :, None] + tl.arange(0, 2)[None, None, :]
    m = inr[:, :, None]
    c0, c1 = tl.split(tl.load(base, mask=m, other=0.0))
    c2, c3 = tl.split(tl.load(base + 2, mask=m, other=0.0))
    v = tl.maximum(((c0 * h + c1) * h + c2) * h + c3, 0.0)
    m = (inr & (kf <= knl))[:, :, None]
    base = base + KM * 4
    c0, c1 = tl.split(tl.load(base, mask=m, other=0.0))
    c2, c3 = tl.split(tl.load(base + 2, mask=m, other=0.0))
    return rho + v + tl.maximum(((c0 * h + c1) * h + c2) * h + c3, 0.0)


@triton.jit
def _density_add32(rho, d2, s, idr_t, dr_t, kn_t, knl_t, coef, KM, inr, NLCC: tl.constexpr):
    """FP32 twin of ``_density_add`` for far neighbours (the FP32 copy of the same tables; see ``MIXED``).
    NLCC = False leaves out the NLCC term, for neighbours whose tile lies beyond its last significant interval (the
    term is then exactly zero)."""
    idr = tl.load(idr_t + s)
    dr = tl.load(dr_t + s)
    nk = tl.load(kn_t + s)
    knl = tl.load(knl_t + s)
    d = tl.sqrt(d2)
    kf = tl.minimum(tl.maximum(libdevice.floor(d * idr), 0.0), (nk - 2).to(tl.float32))
    h = d - kf * dr
    base = coef + ((s * 2 * KM + kf.to(tl.int32)) * 4)[:, :, None] + tl.arange(0, 4)[None, None, :]
    m = inr[:, :, None]
    c02, c13 = tl.split(tl.reshape(tl.load(base, mask=m, other=0.0), [base.shape[0], base.shape[1], 2, 2]))
    c0, c2 = tl.split(c02)
    c1, c3 = tl.split(c13)
    v = tl.maximum(((c0 * h + c1) * h + c2) * h + c3, 0.0)
    if NLCC:
        m = (inr & (kf <= knl))[:, :, None]
        base = base + KM * 4
        c02, c13 = tl.split(tl.reshape(tl.load(base, mask=m, other=0.0), [base.shape[0], base.shape[1], 2, 2]))
        c0, c2 = tl.split(c02)
        c1, c3 = tl.split(c13)
        out = rho + v + tl.maximum(((c0 * h + c1) * h + c2) * h + c3, 0.0)
    else:
        out = rho + v
    return out


@triton.jit
def _vxc(rho, consts):
    """LDA exchange + PZ81 correlation potential (eV) of the FP64 density rho, zero for rho <= consts[11]."""
    k34pi = tl.load(consts + 0)
    third = tl.load(consts + 1)
    k3pi = tl.load(consts + 2)
    a1 = tl.load(consts + 3)
    b1 = tl.load(consts + 4)
    c1_ = tl.load(consts + 5)
    d1 = tl.load(consts + 6)
    g1 = tl.load(consts + 7)
    be1 = tl.load(consts + 8)
    be2 = tl.load(consts + 9)
    two_ry = tl.load(consts + 10)
    floor_ = tl.load(consts + 11)
    ok = rho > floor_
    n = tl.where(ok, rho, 1.0)
    rs = libdevice.pow(k34pi / n, third)
    lr = libdevice.log(rs)
    e_hi = a1 * lr - b1 + c1_ * rs * lr - d1 * rs
    ep_hi = a1 / rs + c1_ * (lr + 1.0) - d1
    sr = tl.sqrt(rs)
    den = 1.0 + be1 * sr + be2 * rs
    dp = be1 / (2.0 * sr) + be2
    e = tl.where(rs < 1.0, e_hi, -g1 / den)
    ep = tl.where(rs < 1.0, ep_hi, g1 * dp / (den * den))
    vx = -libdevice.pow(k3pi * n, third)
    v = two_ry * (vx + e - rs * ep / 3.0)
    return tl.where(ok, v, 0.0)


@triton.jit
def _vxc32(rho, consts):
    """FP32 twin of ``_vxc`` (constants from the FP32 copy of ``_consts``)."""
    k34pi = tl.load(consts + 0)
    third = tl.load(consts + 1)
    k3pi = tl.load(consts + 2)
    a1 = tl.load(consts + 3)
    b1 = tl.load(consts + 4)
    c1_ = tl.load(consts + 5)
    d1 = tl.load(consts + 6)
    g1 = tl.load(consts + 7)
    be1 = tl.load(consts + 8)
    be2 = tl.load(consts + 9)
    two_ry = tl.load(consts + 10)
    floor_ = tl.load(consts + 11)
    ok = rho > floor_
    n = tl.where(ok, rho, 1.0)
    rs = libdevice.pow(k34pi / n, third)
    lr = libdevice.log(rs)
    e_hi = a1 * lr - b1 + c1_ * rs * lr - d1 * rs
    ep_hi = a1 / rs + c1_ * (lr + 1.0) - d1
    sr = tl.sqrt(rs)
    den = 1.0 + be1 * sr + be2 * rs
    dp = be1 / (2.0 * sr) + be2
    e = tl.where(rs < 1.0, e_hi, -g1 / den)
    ep = tl.where(rs < 1.0, ep_hi, g1 * dp / (den * den))
    vx = -libdevice.pow(k3pi * n, third)
    v = two_ry * (vx + e - rs * ep / 3.0)
    return tl.where(ok, v, 0.0)


@triton.jit(do_not_specialize=['P', 'NOM', 'NPHI', 'NT', 'NTP', 'KM', 'NR'])
def _potential_kernel(out, rad, dirs, ptr, ends, nd, ns, atoms, idr_t, dr_t, kn_t, knl_t, coef, consts, P, NOM, NPHI,
                      NT, NTP, KM, NR, idr32, dr32, knl32, coef32, split, nlr_t,
                      TM: tl.constexpr, TP: tl.constexpr, MIXED: tl.constexpr):
    """rho and v_xc on one (radial node, angular tile) of one atom's grid.

    The atom's neighbours are sorted by D - L (distance minus density reach); at radial node r the rows
    [ptr, full) reach every point of the shell (D + r <= L), [full, end) may reach some (D - r <= L) and are tested
    against the tile's bounding sphere and then per point; the rest cannot reach the shell.
    MIXED: a neighbour whose distance to the tile's bounding sphere is at least ``split`` is summed in FP32 (FP32
    copies of the points, centre and tables, FP32 partial sum added once per point); nearer ones as in FP64.
    An FP32 neighbour whose distance to the tile's bounding sphere exceeds its species' NLCC reach nlr_t (last
    significant NLCC interval + 0.01 bohr) skips the NLCC term, which is exactly zero there. The squared distances of
    the per-point reach test are the ones the splines use.
    """
    a = tl.program_id(1)
    pid = tl.program_id(0)
    i = pid // NT
    t = pid % NT
    jm = (t // NTP) * TM + tl.arange(0, TM)[:, None]
    jp = (t % NTP) * TP + tl.arange(0, TP)[None, :]
    j = jm * NPHI + jp
    gm = (jp < NPHI) & (j < NOM)
    r = tl.load(rad + i)
    x = r * tl.load(dirs + j * 3 + 0, mask=gm, other=0.0)
    y = r * tl.load(dirs + j * 3 + 1, mask=gm, other=0.0)
    z = r * tl.load(dirs + j * 3 + 2, mask=gm, other=0.0)
    atom = tl.load(atoms + a)
    start = tl.load(ptr + atom)
    row = ends + (atom.to(tl.int64) * NR + i) * 2
    full = tl.load(row + 0)
    end = tl.load(row + 1)
    rho = tl.zeros([TM, TP], dtype=tl.float64)
    if MIXED:
        cnt = tl.sum(tl.sum(gm.to(tl.float64), 1), 0)
        mx = tl.sum(tl.sum(tl.where(gm, x, 0.0), 1), 0) / cnt
        my = tl.sum(tl.sum(tl.where(gm, y, 0.0), 1), 0) / cnt
        mz = tl.sum(tl.sum(tl.where(gm, z, 0.0), 1), 0) / cnt
        ex = x - mx
        ey = y - my
        ez = z - mz
        rad_b = tl.sqrt(tl.max(tl.max(tl.where(gm, ex * ex + ey * ey + ez * ez, 0.0), 1), 0))
        x32 = x.to(tl.float32)
        y32 = y.to(tl.float32)
        z32 = z.to(tl.float32)
        far_r = split + rad_b
        far2 = far_r * far_r
        rho32 = tl.zeros([TM, TP], dtype=tl.float32)
        for q in range(start, full):
            cx = tl.load(nd + q * 4 + 0)
            cy = tl.load(nd + q * 4 + 1)
            cz = tl.load(nd + q * 4 + 2)
            bx = cx - mx
            by = cy - my
            bz = cz - mz
            b2 = bx * bx + by * by + bz * bz
            if b2 >= far2:
                sq = tl.load(ns + q)
                nl = tl.load(nlr_t + sq) + rad_b
                fx = x32 - cx.to(tl.float32)
                fy = y32 - cy.to(tl.float32)
                fz = z32 - cz.to(tl.float32)
                f2 = fx * fx + fy * fy + fz * fz
                if b2 >= nl * nl:
                    rho32 = _density_add32(rho32, f2, sq, idr32, dr32, kn_t, knl32, coef32, KM, gm, NLCC=False)
                else:
                    rho32 = _density_add32(rho32, f2, sq, idr32, dr32, kn_t, knl32, coef32, KM, gm, NLCC=True)
            else:
                dx = x - cx
                dy = y - cy
                dz = z - cz
                rho = _density_add(rho, dx * dx + dy * dy + dz * dz, tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef,
                                   KM, gm)
        for q in range(full, end):
            cx = tl.load(nd + q * 4 + 0)
            cy = tl.load(nd + q * 4 + 1)
            cz = tl.load(nd + q * 4 + 2)
            kl = tl.load(nd + q * 4 + 3)
            bx = cx - mx
            by = cy - my
            bz = cz - mz
            b2 = bx * bx + by * by + bz * bz
            reach = kl + rad_b
            if b2 <= reach * reach:
                if b2 >= far2:
                    k32 = kl.to(tl.float32)
                    fx = x32 - cx.to(tl.float32)
                    fy = y32 - cy.to(tl.float32)
                    fz = z32 - cz.to(tl.float32)
                    f2 = fx * fx + fy * fy + fz * fz
                    in32 = (f2 <= k32 * k32) & gm
                    sq = tl.load(ns + q)
                    nl = tl.load(nlr_t + sq) + rad_b
                    if b2 >= nl * nl:
                        rho32 = _density_add32(rho32, f2, sq, idr32, dr32, kn_t, knl32, coef32, KM, in32, NLCC=False)
                    else:
                        rho32 = _density_add32(rho32, f2, sq, idr32, dr32, kn_t, knl32, coef32, KM, in32, NLCC=True)
                else:
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    e2 = dx * dx + dy * dy + dz * dz
                    rho = _density_add(rho, e2, tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef, KM, (e2 <= kl * kl) & gm)
        rho += rho32.to(tl.float64)
    else:
        for q in range(start, full):
            dx = x - tl.load(nd + q * 4 + 0)
            dy = y - tl.load(nd + q * 4 + 1)
            dz = z - tl.load(nd + q * 4 + 2)
            rho = _density_add(rho, dx * dx + dy * dy + dz * dz, tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef, KM,
                               gm)
        if full < end:
            cnt = tl.sum(tl.sum(gm.to(tl.float64), 1), 0)
            mx = tl.sum(tl.sum(tl.where(gm, x, 0.0), 1), 0) / cnt
            my = tl.sum(tl.sum(tl.where(gm, y, 0.0), 1), 0) / cnt
            mz = tl.sum(tl.sum(tl.where(gm, z, 0.0), 1), 0) / cnt
            ex = x - mx
            ey = y - my
            ez = z - mz
            rad_b = tl.sqrt(tl.max(tl.max(tl.where(gm, ex * ex + ey * ey + ez * ez, 0.0), 1), 0))
            for q in range(full, end):
                cx = tl.load(nd + q * 4 + 0)
                cy = tl.load(nd + q * 4 + 1)
                cz = tl.load(nd + q * 4 + 2)
                kl = tl.load(nd + q * 4 + 3)
                bx = cx - mx
                by = cy - my
                bz = cz - mz
                reach = kl + rad_b
                if bx * bx + by * by + bz * bz <= reach * reach:
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    e2 = dx * dx + dy * dy + dz * dz
                    rho = _density_add(rho, e2, tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef, KM,
                                       (e2 <= kl * kl) & gm)
    g = i.to(tl.int64) * NOM + j
    tl.store(out + a.to(tl.int64) * P + g, _vxc(rho, consts), mask=gm)


@triton.jit(do_not_specialize=['P', 'NOM', 'NPHI', 'NT', 'NTP', 'KM', 'NR'])
def _indicator_kernel(out, rad, dirs, ptr, ends, nd, ns, atoms, idr_t, dr_t, kn_t, knl_t, coef, consts, nlr_t, P, NOM,
                      NPHI, NT, NTP, KM, NR, TM: tl.constexpr, TP: tl.constexpr):
    """v_xc of the order screen's indicator (see ``OnsiteXC``) on one (radial node, angular tile) of one atom's grid:
    ``_potential_kernel`` with every quantity in FP32 (grid points, neighbour rows nd [M, 4], tables, density, v_xc;
    the reach classes from ``ends`` are the same). A neighbour whose distance to the tile's bounding sphere exceeds its
    species' NLCC reach skips the NLCC term."""
    a = tl.program_id(1)
    pid = tl.program_id(0)
    i = pid // NT
    t = pid % NT
    jm = (t // NTP) * TM + tl.arange(0, TM)[:, None]
    jp = (t % NTP) * TP + tl.arange(0, TP)[None, :]
    j = jm * NPHI + jp
    gm = (jp < NPHI) & (j < NOM)
    r = tl.load(rad + i)
    x = r * tl.load(dirs + j * 3 + 0, mask=gm, other=0.0)
    y = r * tl.load(dirs + j * 3 + 1, mask=gm, other=0.0)
    z = r * tl.load(dirs + j * 3 + 2, mask=gm, other=0.0)
    atom = tl.load(atoms + a)
    start = tl.load(ptr + atom)
    row = ends + (atom.to(tl.int64) * NR + i) * 2
    full = tl.load(row + 0)
    end = tl.load(row + 1)
    cnt = tl.sum(tl.sum(gm.to(tl.float32), 1), 0)
    mx = tl.sum(tl.sum(tl.where(gm, x, 0.0), 1), 0) / cnt
    my = tl.sum(tl.sum(tl.where(gm, y, 0.0), 1), 0) / cnt
    mz = tl.sum(tl.sum(tl.where(gm, z, 0.0), 1), 0) / cnt
    ex = x - mx
    ey = y - my
    ez = z - mz
    rad_b = tl.sqrt(tl.max(tl.max(tl.where(gm, ex * ex + ey * ey + ez * ez, 0.0), 1), 0))
    rho = tl.zeros([TM, TP], dtype=tl.float32)
    for q in range(start, full):
        cx = tl.load(nd + q * 4 + 0)
        cy = tl.load(nd + q * 4 + 1)
        cz = tl.load(nd + q * 4 + 2)
        bx = cx - mx
        by = cy - my
        bz = cz - mz
        sq = tl.load(ns + q)
        nl = tl.load(nlr_t + sq) + rad_b
        fx = x - cx
        fy = y - cy
        fz = z - cz
        f2 = fx * fx + fy * fy + fz * fz
        if bx * bx + by * by + bz * bz >= nl * nl:
            rho = _density_add32(rho, f2, sq, idr_t, dr_t, kn_t, knl_t, coef, KM, gm, NLCC=False)
        else:
            rho = _density_add32(rho, f2, sq, idr_t, dr_t, kn_t, knl_t, coef, KM, gm, NLCC=True)
    for q in range(full, end):
        cx = tl.load(nd + q * 4 + 0)
        cy = tl.load(nd + q * 4 + 1)
        cz = tl.load(nd + q * 4 + 2)
        kl = tl.load(nd + q * 4 + 3)
        bx = cx - mx
        by = cy - my
        bz = cz - mz
        b2 = bx * bx + by * by + bz * bz
        reach = kl + rad_b
        if b2 <= reach * reach:
            fx = x - cx
            fy = y - cy
            fz = z - cz
            f2 = fx * fx + fy * fy + fz * fz
            inr = (f2 <= kl * kl) & gm
            sq = tl.load(ns + q)
            nl = tl.load(nlr_t + sq) + rad_b
            if b2 >= nl * nl:
                rho = _density_add32(rho, f2, sq, idr_t, dr_t, kn_t, knl_t, coef, KM, inr, NLCC=False)
            else:
                rho = _density_add32(rho, f2, sq, idr_t, dr_t, kn_t, knl_t, coef, KM, inr, NLCC=True)
    g = i.to(tl.int64) * NOM + j
    tl.store(out + a.to(tl.int64) * P + g, _vxc32(rho, consts).to(tl.float64), mask=gm)


class _Atom:
    """Species data of the onsite XC: ORB radial splines and the UPF density splines."""

    def __init__(self, s, pporb, meta, device):
        from dptb.prior.common.pporb import read_abacus_orb, read_upf, simpson_rab

        def find(role):
            name = Path(meta[role + '_file']).name
            path = Path(pporb) / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta[role + '_sha256']:
                raise ValueError(f'{s}: {role} file {path} differs from the table sources')
            return path

        o = read_abacus_orb(find('orbital'))
        u = read_upf(find('upf'))
        self.rc = float(o.r[-1])
        self.channels = [int(c.l) for c in o.channels]
        r = torch.as_tensor(np.asarray(o.r, dtype=np.float64), device=device)
        y = torch.as_tensor(np.stack([np.asarray(c.radial, dtype=np.float64) for c in o.channels], 1), device=device)
        self.orb_knots, self.orb_coef = r, spline_coefficients(r, y)
        q = np.asarray(u.rhoatom_q, dtype=np.float64)
        q = q * u.z_valence / float(simpson_rab(q, u.rab))
        ur = np.asarray(u.r, dtype=np.float64)
        rho = np.maximum(q, 0) / (4 * np.pi * np.maximum(ur, 1e-20) ** 2)
        if ur[0] < 1e-10:
            rho[0] = rho[1]
        d = np.diff(ur)
        if abs(ur[0]) > 1e-14 or not np.allclose(d, d[0], rtol=0, atol=1e-12):
            raise ValueError(f'{s}: the onsite density kernel needs a uniform UPF mesh from r = 0')
        cols = [rho] + ([np.asarray(u.nlcc, dtype=np.float64)] if u.nlcc is not None else [])
        knots = torch.as_tensor(ur, device=device)
        c = spline_coefficients(knots, torch.as_tensor(np.stack(cols, 1), device=device))   # [K - 1, 4, ch]
        self.dens = c.permute(2, 0, 1).contiguous()                                        # [ch, K - 1, 4]
        self.dr, self.K, self.last = float(d[0]), len(ur), float(ur[-1])
        # last interval whose NLCC cubic can exceed NLCC_FLOOR (bound sum_p |c_p| dr^(3-p)); beyond it the NLCC term
        # is below the FP64 resolution of any density it is added to and its coefficients are not read
        self.nlcc_last = -1
        if self.dens.shape[0] > 1:
            dr_pow = torch.tensor([self.dr ** 3, self.dr ** 2, self.dr, 1.0], dtype=torch.float64, device=device)
            big = torch.nonzero((self.dens[1].abs() * dr_pow).sum(1) >= NLCC_FLOOR, as_tuple=True)[0]
            self.nlcc_last = int(big[-1]) if len(big) else -1

    def radial(self, r):
        """R_ch(r) [len(r), nch] (not-a-knot spline, zero outside the mesh)."""
        k = self.orb_knots
        i = (torch.searchsorted(k, r, right=True).clamp(1, len(k) - 1) - 1)
        h = (r - k[i])[:, None]
        c = self.orb_coef[i]
        v = ((c[:, 0] * h + c[:, 1]) * h + c[:, 2]) * h + c[:, 3]
        return torch.where(((r >= k[0]) & (r <= k[-1]))[:, None], v, torch.zeros_like(v))


class Grid:
    """Product quadrature of one (species, order): radial nodes and weights, directions and weights, radial values."""

    def __init__(self, atom, order, device):
        from scipy.special import roots_legendre
        nr, nmu, nphi = order
        x, wr = roots_legendre(nr)
        mu, wm = roots_legendre(nmu)
        r = (x + 1) * atom.rc / 2
        wr = wr * atom.rc / 2
        phi = np.arange(nphi) * 2 * np.pi / nphi
        dirs = np.stack(np.broadcast_arrays(np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi),
                                            np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi), mu[:, None]), -1).reshape(-1, 3)
        aw = np.repeat(wm, nphi) * 2 * np.pi / nphi
        T = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64, device=device)  # noqa: E731
        self.order = tuple(order)
        self.nr, self.nmu, self.nphi = nr, nmu, nphi
        self.r, self.dirs = T(r), T(dirs)
        self.r32, self.dirs32 = self.r.to(torch.float32), self.dirs.to(torch.float32)
        self.P, self.nom = nr * len(dirs), len(dirs)
        lmax = max(atom.channels)
        ylm = torch.cat([real_harmonics(l, self.dirs) for l in range(lmax + 1)], 1)       # [nom, (lmax + 1)^2]
        self.nlm = nlm = (lmax + 1) ** 2
        # the angular Gram A[lm, lm'] is symmetric: only the columns lm <= lm' are formed
        u, v = np.triu_indices(nlm)
        self.yy = (T(aw)[:, None] * ylm[:, u] * ylm[:, v]).contiguous()                   # [nom, nlm (nlm + 1) / 2]
        upair = np.zeros((nlm, nlm), dtype=np.int64)
        upair[u, v] = upair[v, u] = np.arange(len(u))
        R = atom.radial(self.r)                                                            # [nr, nch]
        self.rr = (T(wr * r * r)[:, None, None] * R[:, :, None] * R[:, None, :]).reshape(nr, -1)
        # AO k -> (channel, lm index l^2 + m-order index); AO pair (k, l) -> (upper lm pair, channel pair)
        ch, lm = [], []
        for c, l in enumerate(atom.channels):
            ch += [c] * (2 * l + 1)
            lm += [l * l + t for t in range(2 * l + 1)]
        ch, lm = np.asarray(ch), np.asarray(lm)
        self.nch = nch = len(atom.channels)
        self.pair = torch.as_tensor(upair[lm[:, None], lm[None, :]], device=device)
        self.chp = torch.as_tensor(ch[:, None] * nch + ch[None, :], device=device)
        self.radius = float(r.max())

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.r, self.dirs, self.r32, self.dirs32, self.yy, self.rr))


class OnsiteXC:
    def __init__(self, pporb, species_meta, device='cuda', grid_bytes=1 << 30, mixed=True, screen=True):
        self.pporb, self.meta = Path(pporb), species_meta
        self.mixed, self.screen = mixed, screen
        self.stats = {}
        self.device = torch.device(device)
        self.atoms = {}
        self.grids = OrderedDict()
        self.grid_bytes = grid_bytes
        self.consts = _consts(self.device)
        self.consts32 = self.consts.to(torch.float32)

    def atom(self, s):
        if s not in self.atoms:
            self.atoms[s] = _Atom(s, self.pporb, self.meta[s], self.device)
        return self.atoms[s]

    def grid(self, s, order):
        key = (s, tuple(order))
        g = self.grids.get(key)
        if g is None:
            g = self.grids[key] = Grid(self.atom(s), order, self.device)
            while sum(x.nbytes for x in self.grids.values()) > self.grid_bytes and len(self.grids) > 1:
                self.grids.popitem(last=False)
        self.grids.move_to_end(key)
        return g

    def neighbours(self, symbols, pos, cell):
        """Every atom's neighbours that can reach its grid, sorted per atom by D - L (D = |d|, L = density reach of
        the neighbour's species): ptr [n + 1], rows (dx, dy, dz, L) [M, 4], species index [M], and the padded sorted
        keys D - L [n, max count] (+inf padding) from which ``ends`` derives the reach classes of each radial node."""
        names = sorted(set(symbols))
        dev = pos.device
        n = len(symbols)
        spec = torch.as_tensor([names.index(s) for s in symbols], device=dev)
        half = torch.full((n,), RADIUS / 2, dtype=pos.dtype, device=dev)
        a, j, S, d = image_pairs(pos, cell, half, strict=True, self_interaction=True, with_vectors=True)
        reach = torch.as_tensor([max(self.atom(s).rc for s in names)] * len(names), dtype=pos.dtype, device=dev)
        last = torch.as_tensor([self.atom(s).last for s in names], dtype=pos.dtype, device=dev)
        D = torch.linalg.vector_norm(d, dim=-1)
        keep = torch.nonzero(D <= reach[spec[a]] + last[spec[j]] + 1e-9, as_tuple=True)[0]
        a, j, d, D = a[keep], j[keep], d[keep], D[keep]
        L = last[spec[j]]
        key = D - L
        order = torch.argsort(key, stable=True)
        order = order[torch.argsort(a[order], stable=True)]
        a, j, d, L, key = a[order], j[order], d[order], L[order], key[order]
        ptr = torch.searchsorted(a, torch.arange(n + 1, device=dev))
        width = int((ptr[1:] - ptr[:-1]).max()) if n else 0
        keys = torch.full((n, max(width, 1)), float('inf'), dtype=pos.dtype, device=dev)
        keys[a, torch.arange(len(a), device=dev) - ptr[a]] = key
        # density tables of the species present, padded to a common knot count
        KM = max(self.atom(s).K for s in names)
        coef = torch.zeros((len(names), 2, KM, 4), dtype=torch.float64, device=dev)
        for q, s in enumerate(names):
            c = self.atom(s).dens
            coef[q, :len(c), :c.shape[1]] = c
        info = dict(names=names, ptr=ptr, d=torch.cat([d, L[:, None]], 1).contiguous(),
                    s=spec[j].to(torch.int32).contiguous(), keys=keys, KM=KM, coef=coef, ends={},
                    dr=torch.as_tensor([self.atom(s).dr for s in names], dtype=torch.float64, device=dev),
                    idr=torch.as_tensor([1.0 / self.atom(s).dr for s in names], dtype=torch.float64, device=dev),
                    K=torch.as_tensor([self.atom(s).K for s in names], dtype=torch.int32, device=dev),
                    knl=torch.as_tensor([float(self.atom(s).nlcc_last) for s in names], dtype=torch.float64,
                                        device=dev),
                    nlr=torch.as_tensor([(self.atom(s).nlcc_last + 1) * self.atom(s).dr + 0.01
                                         if self.atom(s).nlcc_last >= 0 else 0.0 for s in names],
                                        dtype=torch.float64, device=dev))
        for k in ('coef', 'dr', 'idr', 'knl', 'nlr'):
            info[k + '32'] = info[k].to(torch.float32)
        return info

    def near(self, nb, symbols):
        """The neighbour rows of ``nb`` within rc + SCREEN_DELTA of their atom (same layout, same order; also an FP32
        copy of the rows for the indicator)."""
        dev = nb['ptr'].device
        n = len(symbols)
        atom = torch.repeat_interleave(torch.arange(n, device=dev), nb['ptr'][1:] - nb['ptr'][:-1],
                                       output_size=len(nb['d']))
        rc = torch.as_tensor([self.atom(s).rc for s in symbols], dtype=torch.float64, device=dev)
        keep = torch.nonzero(torch.linalg.vector_norm(nb['d'][:, :3], dim=-1) <= rc[atom] + SCREEN_DELTA,
                             as_tuple=True)[0]
        a = atom[keep]
        ptr = torch.searchsorted(a, torch.arange(n + 1, device=dev))
        d = nb['d'][keep].contiguous()
        keys = torch.full((n, max(int((ptr[1:] - ptr[:-1]).max()), 1)), float('inf'), dtype=torch.float64, device=dev)
        keys[a, torch.arange(len(a), device=dev) - ptr[a]] = torch.linalg.vector_norm(d[:, :3], dim=-1) - d[:, 3]
        out = dict(nb)
        out.update(ptr=ptr, d=d, d32=d.to(torch.float32), s=nb['s'][keep].contiguous(), keys=keys, ends={})
        return out

    @staticmethod
    def ends(nb, g):
        """[n, nr, 2] int64 row ends per (atom, radial node): neighbours with D - L <= -r reach the whole shell,
        those with D - L <= r may reach part of it (margins REACH_EPS keep the classes conservative)."""
        if id(g) not in nb['ends']:
            keys, ptr = nb['keys'], nb['ptr']
            r = g.r[None, :].expand(len(keys), -1).contiguous()
            full = torch.searchsorted(keys, -r - REACH_EPS, right=True)
            part = torch.searchsorted(keys, r + REACH_EPS, right=True)
            nb['ends'][id(g)] = (g, (torch.stack([full, part], -1) + ptr[:-1, None, None]).contiguous())
        return nb['ends'][id(g)][1]

    def blocks(self, s, order, atoms, nb, chunk_bytes=1 << 29, all32=False):
        """V [len(atoms), norb, norb] (eV) of atoms of species s on the grid ``order``; all32: the order screen's
        indicator (``_indicator_kernel``, FP32; ``nb`` from ``near``)."""
        g = self.grid(s, order)
        dev = self.device
        ends = self.ends(nb, g)
        out = []
        per = max(1, chunk_bytes // (8 * g.P))
        mixed = self.mixed
        tm, tp = SCREEN_TILE if all32 else MIXED_TILE if mixed else (TM, TP)
        for a0 in range(0, len(atoms), per):
            ids = atoms[a0:a0 + per].contiguous()
            v = torch.empty((len(ids), g.P), dtype=torch.float64, device=dev)
            ntp = triton.cdiv(g.nphi, tp)
            nt = triton.cdiv(g.nmu, tm) * ntp
            if all32:
                _indicator_kernel[(g.nr * nt, len(ids))](
                    v, g.r32, g.dirs32, nb['ptr'], ends, nb['d32'], nb['s'], ids, nb['idr32'], nb['dr32'], nb['K'],
                    nb['knl32'], nb['coef32'], self.consts32, nb['nlr32'], g.P, g.nom, g.nphi, nt, ntp, nb['KM'], g.nr,
                    TM=tm, TP=tp, num_warps=SCREEN_WARPS)
            else:
                _potential_kernel[(g.nr * nt, len(ids))](
                    v, g.r, g.dirs, nb['ptr'], ends, nb['d'], nb['s'], ids, nb['idr'], nb['dr'], nb['K'], nb['knl'],
                    nb['coef'], self.consts, g.P, g.nom, g.nphi, nt, ntp, nb['KM'], g.nr, nb['idr32'], nb['dr32'],
                    nb['knl32'], nb['coef32'], SPLIT, nb['nlr'], TM=tm, TP=tp, MIXED=mixed,
                    num_warps=MIXED_WARPS if mixed else WARPS)
            A = (v.reshape(-1, g.nom) @ g.yy).reshape(len(ids), g.nr, -1)                  # [a, nr, upper lm pairs]
            F = (A.transpose(1, 2) @ g.rr)                                                 # [a, upper lm pairs, c c']
            out.append(F[:, g.pair, g.chp])
        return torch.cat(out)

    def __call__(self, symbols, pos, cell, width):
        """Onsite XC blocks [n, width, width] (eV, ABACUS gauge) with the refined order policy; also the orders."""
        dev = self.device
        n = len(symbols)
        nb = self.neighbours(symbols, pos, cell)
        near = self.near(nb, symbols) if self.screen else None
        names = nb['names']
        site = torch.zeros((n, width, width), dtype=torch.float64, device=dev)
        orders = [MEDIUM] * n
        screened = screened_finer = 0
        for s in names:
            ids = torch.as_tensor([i for i, x in enumerate(symbols) if x == s], device=dev)
            m = self.atom(s)
            norb = sum(2 * l + 1 for l in m.channels)
            med = self.blocks(s, MEDIUM, ids, nb)
            site[ids, :norb, :norb] = med
            check = torch.arange(len(ids), device=dev)
            if near is not None:
                fine_near = self.blocks(s, FINE, ids, near, all32=True)
                est = (self.blocks(s, MEDIUM, ids, near, all32=True) - fine_near).abs().flatten(1).amax(1)
                check = torch.nonzero(est >= SCREEN_LO, as_tuple=True)[0]
                screened += len(ids) - len(check)
                if not len(check):
                    continue
            fin = self.blocks(s, FINE, ids[check], nb)
            err = (med[check] - fin).abs().flatten(1).amax(1)
            sel = torch.nonzero(err >= TOLERANCE, as_tuple=True)[0]
            if not len(sel):
                continue
            bad, fin = check[sel], fin[sel]
            site[ids[bad], :norb, :norb] = fin
            for k in bad.tolist():
                orders[int(ids[k])] = FINE
            if near is not None:
                est = (fine_near[bad] - self.blocks(s, FINER, ids[bad], near, all32=True)).abs()
                keep = torch.nonzero(est.flatten(1).amax(1) >= SCREEN_LO, as_tuple=True)[0]
                screened_finer += len(bad) - len(keep)
                if not len(keep):
                    continue
                bad, fin = bad[keep], fin[keep]
            finer = self.blocks(s, FINER, ids[bad], nb)
            unresolved = (fin - finer).abs().flatten(1).amax(1) >= TOLERANCE
            for t in torch.nonzero(unresolved, as_tuple=True)[0].tolist():
                atom = ids[bad[t]:bad[t] + 1]
                prev, prev_order = finer[t:t + 1], FINER
                for order in EXTRA:
                    cur = self.blocks(s, order, atom, nb)
                    if float((prev - cur).abs().max()) < TOLERANCE:
                        site[atom, :norb, :norb] = prev
                        orders[int(atom)] = prev_order
                        break
                    prev, prev_order = cur, order
        self.stats = dict(screened=screened, screened_finer=screened_finer)
        return site, orders
