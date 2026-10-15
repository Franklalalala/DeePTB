"""Local potential of the H0 prior on the FFT grid, on the device (FP64, Rydberg atomic units).

    grid      per axis the smallest 2,3,5-smooth 2 m + 1, m the largest |index| along that axis of a reciprocal vector with
              |G|^2 <= ecutrho; the integer sphere is tested on the device in z slabs and points within 1e-12 (relative)
              of the cutoff are re-tested on the host with the expressions of the reference loop.
    species   field view of the UPF (ABACUS): mesh cut at the first odd length beyond 15 bohr (the last sample of an
              even mesh dropped first), PP_RHOATOM extrapolated to r = 0 and scaled to Z_val.
    radial    at every distinct |G| (``torch.unique`` over the grid), with the Simpson/PP_RAB weights w of the mesh:
                  q(G) = sum_r q(r) sinc(G r) w,   c(G) likewise for the NLCC core charge 4 pi r^2 rho_c(r),
                  v(G) = 4 pi / Omega [sum_r (r V(r) + Z e^2 erf(r)) sin(G r) / G w - Z e^2 exp(-G^2 / 4) / G^2],
                  v(0) = 4 pi / Omega sum_r r (r V(r) + Z e^2) w        (alpha term kept).
              Optionally (``RadialTable``) the smooth parts are tabulated per species on a uniform |G| grid with exact
              derivatives and interpolated by cubic Hermite polynomials (a switchable approximation, off by default).
    density   rho(G) = sum_s S_s(G) q_s(|G|) / Omega cut to |G|^2 <= ecutrho and scaled to the electron count;
              S_s(G) = sum_{a in s} exp(-i G.x_a) is a type-1 NUFFT (cuFINUFFT, eps 1e-12; S(0) exact).
    potential V = V_loc + V_H + V_xc with V_loc(G) = sum_s S_s(G) v_s(|G|) (sphere), V_H = 4 pi e^2 rho(G) / G^2 (0 at
              G = 0), V_xc the PBE potential of rho + rho_core (``common.xc.pbe_potential``, gradient i G (rho + rho_c)),
              each brought to the grid as N ifft(.).
"""
import math

import numpy as np
import torch
from scipy.special import erf

from dptb.prior.common.pporb import simpson_rab, simpson_rab_weights
from dptb.prior.common.xc import pbe_potential

FOUR_PI = 4.0 * math.pi
E2 = 2.0                     # e^2 in Rydberg atomic units


def fft_grid(cell, ecutrho=480., device='cuda'):
    """FFT grid (n0, n1, n2) of the charge cutoff ``ecutrho`` (Ry) for the cell rows ``cell`` (bohr, host array)."""
    cell = np.asarray(cell, float)
    B = 2 * np.pi * np.linalg.inv(cell).T
    limits = (np.sqrt(ecutrho) * np.linalg.norm(cell, axis=1) / (2 * np.pi)).astype(int) + 1
    xs = np.arange(-limits[0], limits[0] + 1)
    ys = np.arange(-limits[1], limits[1] + 1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    xy = np.stack([xx.ravel(), yy.ravel()], 1)
    base = xy @ B[:2]
    bt = torch.as_tensor(base, device=device)
    axy = torch.as_tensor(np.abs(xy), device=device)
    zs = np.arange(-int(limits[2]), int(limits[2]) + 1)
    maxima = np.zeros(3, dtype=int)
    lo, hi = ecutrho * (1 - 1e-12), ecutrho * (1 + 1e-12)
    step = max(1, int(4e7 // max(1, len(base))))
    row_in = torch.zeros(len(base), dtype=torch.bool, device=device)
    for a in range(0, len(zs), step):
        zc = zs[a:a + step]
        zb = torch.as_tensor(np.stack([z * B[2] for z in zc]), device=device)
        g = bt[None] + zb[:, None, :]
        q = (g * g).sum(-1)
        sure = q <= lo
        amb = (q > lo) & (q <= hi)
        zin = sure.any(1)
        row_in |= sure.any(0)
        if bool(zin.any()):
            maxima[2] = max(maxima[2], int(np.abs(zc[zin.cpu().numpy()]).max()))
        if bool(amb.any()):
            zi, ri = [t.cpu().numpy() for t in torch.nonzero(amb, as_tuple=True)]
            for z in np.unique(zi):
                rows = ri[zi == z]
                gg = base[rows] + int(zc[z]) * B[2]
                inside = np.einsum('ij,ij->i', gg, gg) <= ecutrho
                if inside.any():
                    maxima[:2] = np.maximum(maxima[:2], np.abs(xy[rows[inside]]).max(0))
                    maxima[2] = max(maxima[2], abs(int(zc[z])))
    if bool(row_in.any()):
        maxima[:2] = np.maximum(maxima[:2], axy[row_in].max(0).values.cpu().numpy())

    def smooth(n):
        while True:
            b = n
            for p in (2, 3, 5):
                while b % p == 0:
                    b //= p
            if b == 1:
                return n
            n += 1
    return tuple(smooth(int(v) * 2 + 1) for v in maxima)


def field_upf(upf, cutoff_bohr=15.0):
    """Radial arrays of the field (ABACUS complete_default + Charge::atomic_rho): r, rab, q, vloc, nlcc, z (host)."""
    reader_n = len(upf.r) - (len(upf.r) % 2 == 0)
    above = np.flatnonzero(upf.r[:reader_n] > cutoff_bohr)
    n = min(reader_n, 2 * ((int(above[0]) + 2) // 2) - 1) if len(above) else reader_n
    if n < 3:
        raise ValueError('pseudopotential field cutoff leaves fewer than 3 samples')
    r, rab, q = upf.r[:n], upf.rab[:n], upf.rhoatom_q[:n].copy()
    charge = float(simpson_rab(q, rab))
    if not np.isfinite(charge) or charge <= 0:
        raise ValueError('non-positive atomic valence density integral')
    if r[0] != 0:
        rho = q / (4 * np.pi * r * r)
        ratio = (rho[2] / rho[1]) ** (r[1] / (r[2] - r[1]))
        rho0 = rho[1] if ratio < 1e-12 else rho[1] / ratio
        q[0] = rho0 * 4 * np.pi * r[0] ** 2
    q *= upf.z_valence / charge
    nlcc = None if upf.nlcc is None else upf.nlcc[:n]
    if nlcc is not None and upf.nlcc_is_radial_density:
        nlcc = FOUR_PI * r ** 2 * nlcc
    return dict(r=r, rab=rab, q=q, vloc=upf.vloc_ry[:n], core=nlcc, z=float(upf.z_valence))


class RadialSpecies:
    """Device arrays of one species' field view and its direct |G| transforms."""

    def __init__(self, upf, device):
        f = field_upf(upf)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)  # noqa: E731
        self.z = f['z']
        self.ze = f['z'] * E2
        self.r, self.w, self.q = T(f['r']), T(simpson_rab_weights(f['rab'])), T(f['q'])
        self.vloc = T(f['vloc'])
        self.comp = self.r * self.vloc + self.ze * T(erf(f['r']))
        self.core = None if f['core'] is None else T(f['core'])
        self.alpha_integral = torch.sum(self.r * (self.r * self.vloc + self.ze) * self.w) * FOUR_PI

    def transforms(self, g, volume, chunk=8192):
        """q(G), v(G), core(G) at the magnitudes g (direct quadrature)."""
        r, w = self.r, self.w
        qg, vg, cg = torch.empty_like(g), torch.empty_like(g), torch.zeros_like(g)
        alpha = self.alpha_integral / volume
        for a in range(0, g.numel(), chunk):
            gg = g[a:a + chunk]
            sinc = torch.sinc(gg[:, None] * r[None, :] / np.pi)
            qg[a:a + chunk] = torch.sum(self.q[None, :] * sinc * w[None, :], dim=1)
            if self.core is not None:
                cg[a:a + chunk] = torch.sum(self.core[None, :] * sinc * w[None, :], dim=1)
            zero = gg < 1e-13
            safe = torch.where(zero, torch.ones_like(gg), gg)
            integ = torch.sum(self.comp[None, :] * torch.sin(safe[:, None] * r[None, :]) / safe[:, None] * w[None, :], dim=1)
            vals = FOUR_PI * (integ - self.ze * torch.exp(-safe ** 2 / 4) / safe ** 2) / volume
            vg[a:a + chunk] = torch.where(zero, alpha, vals)
        return qg, vg, cg


class RadialTable:
    """The smooth parts of the transforms of one species, tabulated on G_k = k h and interpolated (cubic Hermite).

    q(G), core(G) and I(G) = sum_r (r V + Z e^2 erf(r)) sin(G r) / G w are all F(G) = sum_r a_r sinc(G r); the table
    holds F and F'(G) = sum_r a_r (cos(G r) - sinc(G r)) / G at the knots (exact derivatives, F'(0) = 0), so the
    interpolant is C1 and its error is at most h^4 max|F^(4)| / 384 <= (h r_max)^4 / 384 sum|a|. The singular part of v
    and the alpha term stay exact: v(G) = 4 pi / Omega (I(G) - Z e^2 exp(-G^2 / 4) / G^2), v(0) = alpha / Omega. Valid
    for |G| <= gmax (the charge sphere); beyond it the last interval is extended (those G are cut from every quantity).
    """

    def __init__(self, species, gmax, step=5e-4, chunk=2048):
        self.sp, self.step = species, float(step)
        k = int(math.ceil(gmax / self.step)) + 2
        g = torch.arange(k + 1, dtype=torch.float64, device=species.r.device) * self.step
        r = species.r
        core = species.core if species.core is not None else torch.zeros_like(r)
        a = torch.stack([species.q, species.comp * r, core], 0) * species.w           # [3, nr]
        F = torch.empty((len(g), 3), dtype=torch.float64, device=r.device)
        D = torch.empty_like(F)
        for s in range(0, len(g), chunk):
            gg = g[s:s + chunk]
            x = gg[:, None] * r[None, :]
            sinc = torch.sinc(x / np.pi)
            F[s:s + chunk] = sinc @ a.T
            safe = torch.where(gg > 0, gg, torch.ones_like(gg))[:, None]
            D[s:s + chunk] = torch.where(gg[:, None] > 0, ((torch.cos(x) - sinc) / safe) @ a.T, 0.)
        self.F, self.D, self.gmax = F, D, float(g[-1])

    def _interp(self, x):
        h = self.step
        i = torch.clamp((x / h).to(torch.int64), 0, len(self.F) - 2)
        t = (x / h - i.to(x.dtype))[:, None]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * self.F[i] + (t3 - 2 * t2 + t) * h * self.D[i]
                + (-2 * t3 + 3 * t2) * self.F[i + 1] + (t3 - t2) * h * self.D[i + 1])

    def transforms(self, g, volume):
        f = self._interp(g)
        zero = g < 1e-13
        safe = torch.where(zero, torch.ones_like(g), g)
        v = FOUR_PI * (f[:, 1] - self.sp.ze * torch.exp(-safe ** 2 / 4) / safe ** 2) / volume
        return f[:, 0], torch.where(zero, self.sp.alpha_integral / volume, v), f[:, 2]


class Field:
    def __init__(self, values, cell, shape, metadata):
        self.values, self.cell, self.shape, self.metadata = values, cell, shape, metadata
        self.weight = float(abs(np.linalg.det(cell)) / np.prod(shape))


_PLAN = {}
PLAN_CACHE_POINTS = 1 << 24      # keep the plan of grids up to 256^3 points (2 GiB upsampled work array)


def _nufft_plan(shape, eps, dev, stream):
    """Type-1 cufinufft plan of the grid ``shape``; plan creation (about 0.1 s) is paid once per grid shape: the most
    recent plan of a grid up to PLAN_CACHE_POINTS points is kept and reused with new points (``setpts``)."""
    import cufinufft
    key = (dev.index, tuple(shape), eps, stream.cuda_stream)
    plan = _PLAN.get(key)
    if plan is None:
        _PLAN.clear()
        plan = cufinufft.Plan(1, tuple(shape), n_trans=1, eps=eps, isign=-1, dtype='complex128', modeord=1,
                              upsampfac=2.0, gpu_device_id=dev.index if dev.index is not None else
                              torch.cuda.current_device(), gpu_stream=stream.cuda_stream)
        if int(np.prod(shape)) <= PLAN_CACHE_POINTS:
            _PLAN[key] = plan
    return plan


def structure_factor(frac, shape, eps=1e-12, plan=None):
    """S(G) = sum_a exp(-i G.x_a) on the FFT grid (FFT index order) for fractional coordinates frac [n, 3] (device);
    ``plan``: a plan of this grid (``_nufft_plan``) shared by the species of a structure."""
    dev = frac.device
    angles = (torch.remainder(frac + 0.5, 1.0) - 0.5) * (2 * np.pi)
    stream = torch.cuda.current_stream(dev)
    if plan is None:
        plan = _nufft_plan(shape, eps, dev, stream)
    plan.setpts(*(angles[:, d].contiguous() for d in range(3)))
    out = plan.execute(torch.ones(len(frac), dtype=torch.complex128, device=dev)).reshape(tuple(shape))
    out.reshape(-1)[0] = len(frac)
    stream.synchronize()
    return out


def build_field(radial, symbols, frac, cell, shape, *, ecutrho=480., total_electrons=None, tables=None):
    """Total local potential (Ry) on the FFT grid ``shape`` of the cell rows ``cell`` (bohr, host array).

    radial: {symbol: RadialSpecies}; frac: fractional coordinates [n, 3] (device); tables: optional
    {symbol: RadialTable} used instead of the direct transforms.
    """
    dev = frac.device
    volume = float(abs(np.linalg.det(cell)))
    labels = np.asarray(symbols, dtype=object)
    with torch.no_grad():
        b = torch.as_tensor(2.0 * np.pi * np.linalg.inv(cell).T, device=dev)
        freq = [torch.as_tensor(np.fft.fftfreq(n, d=1.0 / n), device=dev) for n in shape]
        h, k, l = torch.meshgrid(*freq, indexing='ij')
        gvec = h[..., None] * b[0] + k[..., None] * b[1] + l[..., None] * b[2]
        g2 = torch.einsum('...i,...i->...', gvec, gvec)
        flat = torch.sqrt(torch.clamp_min(g2.reshape(-1), 0.0))
        gm, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        del flat, h, k, l
        mask = g2 <= float(ecutrho) + 1e-12
        rg = torch.zeros(shape, dtype=torch.complex128, device=dev)
        cg = torch.zeros_like(rg)
        vg = torch.zeros_like(rg)
        plan = _nufft_plan(shape, 1e-12, dev, torch.cuda.current_stream(dev))
        for s in sorted(radial):
            ids = torch.as_tensor(np.where(labels == s)[0], device=dev)
            if not len(ids):
                continue
            sf = structure_factor(frac[ids], shape, plan=plan)
            src = radial[s] if tables is None else tables[s]
            q, v, c = src.transforms(gm, volume)
            rg += sf * q[inverse].reshape(shape) / volume
            vg += sf * v[inverse].reshape(shape)
            cg += sf * c[inverse].reshape(shape) / volume
        del plan
        rg = torch.where(mask, rg, 0.)
        cg = torch.where(mask, cg, 0.)
        if total_electrons is None:
            total_electrons = float(sum(radial[s].z for s in symbols))
        raw = float(rg[0, 0, 0].real * volume)
        if raw <= 0 or not np.isfinite(raw):
            raise ValueError(f'superposed PP_RHOATOM has a non-positive or non-finite electron count {raw}')
        scale = float(total_electrons) / raw
        rg *= scale
        vg = torch.where(mask, vg, 0.)
        hg = torch.zeros_like(rg)
        nz = g2 > 1e-14
        hg[nz] = FOUR_PI * E2 * rg[nz] / g2[nz]
        n = math.prod(shape)

        def ifft(c):
            return (torch.fft.ifftn(c) * n).real

        rho = ifft(rg)
        vloc = ifft(vg)
        vh = ifft(hg)
        tot = rg + cg
        grad = tuple(ifft(1j * gvec[..., axis] * tot) for axis in range(3))
        vxc = pbe_potential(ifft(tot), grad, gvec, mask)
        total = vloc + vh + vxc
        for a in (rho, rg, total):
            if not bool(torch.isfinite(a).all()):
                raise FloatingPointError('non-finite periodic field')
    meta = dict(fft_shape=list(shape), target_electrons=float(total_electrons), raw_superposed_electrons=raw,
                density_normalization_scale=scale, distinct_g=int(len(gm)), gmax=float(gm[-1]))
    return Field(total, np.asarray(cell, dtype=np.float64), tuple(int(x) for x in shape), meta)
