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
directions), then V[k, l] = sum_i w_i R_k(r_i) R_l(r_i) A_i[lm(k), lm(l)] (one GEMM over the radial nodes).

Order policy (accepted 1 meV rule, refined): every atom at medium and fine; atoms with max|medium - fine| < 1e-3 eV use
medium; the others use fine and are checked against finer; unresolved atoms walk the extra orders and take the lower
order of the first step below 1e-3 eV (fine when none converges).
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


@triton.jit
def _density_add(rho, x, y, z, cx, cy, cz, s, idr_t, dr_t, kn_t, knl_t, coef, KM, inr):
    """rho + max(spline_val, 0) + max(spline_nlcc, 0) at |p - c| for the points p with ``inr``; coefficients
    [species, channel, interval, 4] are read as 16-byte pairs, the NLCC ones only up to the species' last significant
    NLCC interval (most points see only the valence table, four intervals per cache line)."""
    idr = tl.load(idr_t + s)
    dr = tl.load(dr_t + s)
    nk = tl.load(kn_t + s)
    knl = tl.load(knl_t + s)
    dx = x - cx
    dy = y - cy
    dz = z - cz
    d = tl.sqrt(dx * dx + dy * dy + dz * dz)
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


@triton.jit(do_not_specialize=['P', 'NOM', 'NPHI', 'NT', 'NTP', 'KM', 'NR'])
def _potential_kernel(out, rad, dirs, ptr, ends, nd, ns, atoms, idr_t, dr_t, kn_t, knl_t, coef, consts, P, NOM, NPHI,
                      NT, NTP, KM, NR, TM: tl.constexpr, TP: tl.constexpr):
    """rho and v_xc on one (radial node, angular tile) of one atom's grid.

    The atom's neighbours are sorted by D - L (distance minus density reach); at radial node r the rows
    [ptr, full) reach every point of the shell (D + r <= L), [full, end) may reach some (D - r <= L) and are tested
    against the tile's bounding sphere and then per point; the rest cannot reach the shell.
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
    for q in range(start, full):
        rho = _density_add(rho, x, y, z, tl.load(nd + q * 4 + 0), tl.load(nd + q * 4 + 1), tl.load(nd + q * 4 + 2),
                           tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef, KM, gm)
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
                inr = (dx * dx + dy * dy + dz * dz <= kl * kl) & gm
                rho = _density_add(rho, x, y, z, cx, cy, cz, tl.load(ns + q), idr_t, dr_t, kn_t, knl_t, coef, KM,
                                   inr)
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
    g = i.to(tl.int64) * NOM + j
    tl.store(out + a.to(tl.int64) * P + g, tl.where(ok, v, 0.0), mask=gm)


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
        self.P, self.nom = nr * len(dirs), len(dirs)
        lmax = max(atom.channels)
        ylm = torch.cat([real_harmonics(l, self.dirs) for l in range(lmax + 1)], 1)       # [nom, (lmax + 1)^2]
        self.yy = (T(aw)[:, None, None] * ylm[:, :, None] * ylm[:, None, :]).reshape(len(dirs), -1)
        R = atom.radial(self.r)                                                            # [nr, nch]
        self.rr = (T(wr * r * r)[:, None, None] * R[:, :, None] * R[:, None, :]).reshape(nr, -1)
        self.nlm = (lmax + 1) ** 2
        # AO k -> (channel, lm index l^2 + m-order index)
        ch, lm = [], []
        for c, l in enumerate(atom.channels):
            ch += [c] * (2 * l + 1)
            lm += [l * l + t for t in range(2 * l + 1)]
        self.ch, self.lm = torch.as_tensor(ch, device=device), torch.as_tensor(lm, device=device)
        self.nch = len(atom.channels)
        self.radius = float(r.max())

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.r, self.dirs, self.yy, self.rr))


class OnsiteXC:
    def __init__(self, pporb, species_meta, device='cuda', grid_bytes=1 << 30):
        self.pporb, self.meta = Path(pporb), species_meta
        self.device = torch.device(device)
        self.atoms = {}
        self.grids = OrderedDict()
        self.grid_bytes = grid_bytes
        self.consts = _consts(self.device)

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
        keep = D <= reach[spec[a]] + last[spec[j]] + 1e-9
        a, j, d, D = a[keep], j[keep], d[keep], D[keep]
        L = last[spec[j]]
        key = D - L
        order = torch.argsort(key, stable=True)
        order = order[torch.argsort(a[order], stable=True)]
        a, j, d, L, key = a[order], j[order], d[order], L[order], key[order]
        counts = torch.bincount(a, minlength=n)
        ptr = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        ptr[1:] = torch.cumsum(counts, 0)
        width = int(counts.max()) if n else 0
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
                                        device=dev))
        return info

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

    def blocks(self, s, order, atoms, nb, chunk_bytes=1 << 29):
        """V [len(atoms), norb, norb] (eV) of atoms of species s on the grid ``order``."""
        g = self.grid(s, order)
        dev = self.device
        ends = self.ends(nb, g)
        out = []
        per = max(1, chunk_bytes // (8 * g.P))
        for a0 in range(0, len(atoms), per):
            ids = atoms[a0:a0 + per].contiguous()
            v = torch.empty((len(ids), g.P), dtype=torch.float64, device=dev)
            ntp = triton.cdiv(g.nphi, TP)
            nt = triton.cdiv(g.nmu, TM) * ntp
            _potential_kernel[(g.nr * nt, len(ids))](
                v, g.r, g.dirs, nb['ptr'], ends, nb['d'], nb['s'], ids, nb['idr'], nb['dr'], nb['K'], nb['knl'], nb['coef'],
                self.consts, g.P, g.nom, g.nphi, nt, ntp, nb['KM'], g.nr, TM=TM, TP=TP, num_warps=WARPS)
            A = (v.reshape(-1, g.nom) @ g.yy).reshape(len(ids), -1, g.nlm * g.nlm)          # [a, nr, nlm^2]
            F = (A.transpose(1, 2) @ g.rr).reshape(len(ids), g.nlm, g.nlm, g.nch, g.nch)     # [a, lm, lm', c, c']
            out.append(F[:, g.lm[:, None], g.lm[None, :], g.ch[:, None], g.ch[None, :]])
        return torch.cat(out)

    def __call__(self, symbols, pos, cell, width):
        """Onsite XC blocks [n, width, width] (eV, ABACUS gauge) with the refined order policy; also the orders."""
        dev = self.device
        n = len(symbols)
        nb = self.neighbours(symbols, pos, cell)
        names = nb['names']
        site = torch.zeros((n, width, width), dtype=torch.float64, device=dev)
        orders = [MEDIUM] * n
        for s in names:
            ids = torch.as_tensor([i for i, x in enumerate(symbols) if x == s], device=dev)
            m = self.atom(s)
            norb = sum(2 * l + 1 for l in m.channels)
            med = self.blocks(s, MEDIUM, ids, nb)
            fin = self.blocks(s, FINE, ids, nb)
            err = (med - fin).abs().flatten(1).amax(1)
            site[ids, :norb, :norb] = med
            bad = torch.nonzero(err >= TOLERANCE, as_tuple=True)[0]
            if not len(bad):
                continue
            site[ids[bad], :norb, :norb] = fin[bad]
            finer = self.blocks(s, FINER, ids[bad], nb)
            unresolved = (fin[bad] - finer).abs().flatten(1).amax(1) >= TOLERANCE
            for k in bad.tolist():
                orders[int(ids[k])] = FINE
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
        return site, orders
