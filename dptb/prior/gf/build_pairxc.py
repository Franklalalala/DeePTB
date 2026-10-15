"""GPU builder of the pair-XC tables: two-centre LDA exchange-correlation blocks of a species pair (eV).

    C_ab(d) = <phi_a^A | v_xc[rho_A + rho_B] | phi_b^B>,   B displaced by d along +z,

with phi = f(r) Y_lm (ABACUS real harmonics), v_xc Slater + PZ81 (``common.xc.lda_pz81``) and rho the AtomicSource
valence plus non-linear-core density (each spline clipped at zero, zero beyond the last mesh point); orbitals are zero
outside their mesh and at r >= R_c. In bipolar coordinates around the bond (r from A, r_B from B, the azimuth done
analytically) the block is diagonal in the m index:

    C_ab(d) = 2 pi sum_{m <= min(l_a, l_b)} int_0^{R_A} dr int_{|r - d|}^{min(R_B, r + d)} dr_B  r r_B / d
              f_a(r) T_{l_a m}(cos t_A) v(r, r_B) f_b(r_B) T_{l_b m}(cos t_B)      (entries k = 0, or 2m - 1 and 2m),
    cos t_A = (r^2 + d^2 - r_B^2) / (2 r d),   cos t_B = (r^2 - d^2 - r_B^2) / (2 r_B d),
    T_lm = sqrt((2l + 1) / (4 pi) (l - m)! / (l + m)!) P_l^m,

C_ab(0) = delta_{l_a l_b} int_0^{min(R_A, R_B)} r^2 f_a f_b v dr, and C = 0 for d >= R_A + R_B. The r range is cut at
the kinks {d, R_B - d, d - R_B, d + R_B} inside (0, R_A) (and every 0.125 bohr in the partitioned variant); every
piece and every r_B range takes an n-point Gauss-Legendre rule.

Acceptance per pair: reference blocks at the held distances 0.137, 1.337 and six points from 2.313 to
R_A + R_B - 0.217 (reference order); the probe order must agree with them within 1e-4 eV; then the (order, knot step)
ladder is walked until the not-a-knot spline through the knots reproduces the references within 1e-4 eV.

    standard     reference 320, probe 192, ladder (128, 0.05), (256, 0.025), (384, 0.0125)
    partitioned  reference 128, probe 96,  ladder (64, 0.05), (96, 0.025)          (tried when standard fails)

Each table is one npz (distances, values_eV [K, n_A, n_B], left_shells, right_shells, support_bohr; partitioned
tables carry the suffix ``_partitioned``) listed under ``pairs`` with its SHA256 in the write-once store
(``dptb.prior.common.store``) manifest: the layout ``dptb.prior.gf.tables`` reads. All distances of a rung are evaluated together: the quadrature points of a chunk of
distances form one batch and every m is one batched FP64 GEMM.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.radial import spline_coefficients
from dptb.prior.common.store import TableStore, npz_bytes, sha256_file
from dptb.prior.common.xc import lda_pz81

SCHEMA = 'deeptb.pair_xc_table/v1'     # deployed manifests carry the legacy id (tables.LEGACY_SCHEMAS)
TOL_EV = 1e-4
VARIANTS = {
    'standard': dict(partition=0.0, reference=320, probe=192, ladder=((128, 0.05), (256, 0.025), (384, 0.0125))),
    'partitioned': dict(partition=0.125, reference=128, probe=96, ladder=((64, 0.05), (96, 0.025))),
}


class Splines:
    """Not-a-knot cubic splines (scipy ``CubicSpline``) of the rows of y [M, K] on knots x [K], on the device."""

    def __init__(self, x, y, device):
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)
        y = torch.as_tensor(np.asarray(y, dtype=np.float64).reshape(-1, len(x)).T.copy(), device=device)
        c = spline_coefficients(self.x, y)
        self.c = [c[:, :, m].contiguous() for m in range(c.shape[-1])]

    def __call__(self, r):
        """Values [M, *r.shape] (inside the knot range; callers mask outside)."""
        flat = r.reshape(-1)
        i = (torch.searchsorted(self.x, flat, right=True) - 1).clamp(0, len(self.x) - 2)
        t = flat - self.x[i]
        out = []
        for c in self.c:
            g = c[i]
            out.append(((g[:, 0] * t + g[:, 1]) * t + g[:, 2]) * t + g[:, 3])
        return torch.stack(out).reshape(-1, *r.shape)


class Radials:
    """Radial functions of one centre: l per function, common cutoff, values f(r) -> [n, *r.shape]."""

    def __init__(self, shells, cut, fn):
        self.shells = tuple(int(l) for l in shells)
        self.cut = float(cut)
        self.fn = fn
        self.offsets = np.cumsum([0] + [2 * l + 1 for l in self.shells])

    def __call__(self, r):
        return self.fn(r)


def atomic_source_arrays(symbol, orb, upf):
    """AtomicSource arrays of a species from its ABACUS ORB and UPF files: ORB channels (file order, normalised),
    the valence density q(r) Z / int q rab / (4 pi r^2) clipped at zero (rho(0) = rho(dr) when the mesh starts at 0)
    and the unscaled NLCC core density."""
    from dptb.prior.common.pporb import read_abacus_orb, read_upf, simpson_rab
    u, o = read_upf(upf), read_abacus_orb(orb)
    rawq = float(simpson_rab(u.rhoatom_q, u.rab))
    scale = u.z_valence / rawq
    rho = np.maximum(u.rhoatom_q * scale, 0) / (4 * np.pi * np.maximum(u.r, 1e-20) ** 2)
    if u.r[0] < 1e-10:
        rho[0] = rho[1]
    nlcc = np.asarray(u.nlcc, dtype=float) if u.nlcc is not None else np.zeros_like(u.r)
    ident = dict(NLCC=u.nlcc is not None, Z=float(u.z_valence),
                 channels=[dict(index=i, l=int(c.l)) for i, c in enumerate(o.channels)],
                 density_definition='normalized neutral valence (r=0 repair) + unscaled NLCC; AtomicDensity contract',
                 normalization=float(scale), orb_rcut=float(o.r[-1]), paths=dict(orbital=str(orb), upf=str(upf)),
                 raw_valence_Q=rawq, sha256=dict(orbital=sha256_file(orb), upf=sha256_file(upf)))
    return dict(symbol=np.array(symbol), r_orb=np.asarray(o.r, dtype=float),
                radial=np.asarray([c.radial for c in o.channels], dtype=float),
                shells=np.asarray([int(c.l) for c in o.channels], dtype=np.int64),
                orbital_cutoff_bohr=np.array(float(o.r[-1])), r_rho=np.asarray(u.r, dtype=float), rho_val=rho,
                rho_nlcc=nlcc, z_valence=np.array(float(u.z_valence)), identity=np.array(json.dumps(ident, sort_keys=True)))


class AtomicSource:
    """Radial orbitals and density of one species (AtomicSource arrays: symbol, shells, r_orb, radial,
    orbital_cutoff_bohr, r_rho, rho_val, rho_nlcc, z_valence, identity), splined on the device. Orbitals are zero
    outside their mesh and at r >= R_c; the density is max(valence, 0) + max(core, 0) on the clamped mesh, zero past
    its end."""

    def __init__(self, z, device='cuda', sha256=None):
        self.symbol = str(z['symbol'])
        self.shells = tuple(int(l) for l in z['shells'])
        self.cut = float(z['orbital_cutoff_bohr'])
        self.z_valence = float(z['z_valence'])
        self.identity = json.loads(str(z['identity']))
        self.sha256 = sha256 or hashlib.sha256(npz_bytes(dict(z))).hexdigest()
        self.r_rho = np.asarray(z['r_rho'], np.float64)
        r_orb = np.asarray(z['r_orb'], np.float64)
        rho = [z['rho_val']] + ([z['rho_nlcc']] if np.any(z['rho_nlcc']) else [])
        self.orb = Splines(r_orb, z['radial'], device)
        self.rho = Splines(self.r_rho, np.stack(rho), device)
        self.device = self.orb.x.device
        self.orb_range = (float(r_orb[0]), float(r_orb[-1]))
        self.rho_range = (float(self.r_rho[0]), float(self.r_rho[-1]))
        self.orbitals = Radials(self.shells, self.cut, self.radial)
        self.envelopes = Radials((0,) * len(self.shells), self.cut, lambda r: self.radial(r).abs())

    @classmethod
    def load(cls, path, device='cuda'):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files}, device, sha256_file(path))

    @property
    def norb(self):
        return sum(2 * l + 1 for l in self.shells)

    def radial(self, r):
        """f_a(r) [n_shells, *r.shape]."""
        ok = (r >= self.orb_range[0]) & (r <= self.orb_range[1]) & (r < self.cut)
        return torch.where(ok, self.orb(r), 0.0)

    def density(self, r):
        v = self.rho(r.clamp(*self.rho_range)).clamp_min(0.0).sum(0)
        return torch.where(r > self.rho_range[1], 0.0, v)


class Sources:
    """AtomicSource of each symbol: ``<sources_dir>/<S>.npz``, else built from ``species_files[S] = (orb, upf)``."""

    def __init__(self, sources_dir=None, species_files=None, device='cuda'):
        self.dir = Path(sources_dir) if sources_dir else None
        self.files = dict(species_files or {})
        self.device = device
        self._cache = {}

    def __getitem__(self, s):
        if s not in self._cache:
            if self.dir is not None and (self.dir / f'{s}.npz').exists():
                self._cache[s] = AtomicSource.load(self.dir / f'{s}.npz', self.device)
            elif s in self.files:
                self._cache[s] = AtomicSource(atomic_source_arrays(s, *self.files[s]), self.device)
            else:
                raise KeyError(f'no AtomicSource for {s}: give sources_dir/{s}.npz or species_files[{s!r}] = (orb, upf)')
        return self._cache[s]


def _theta(m, lmax, c):
    """{l: T_lm(c)} for m <= l <= lmax (associated Legendre with the Condon-Shortley phase, normalised)."""
    s = torch.sqrt((1.0 - c * c).clamp_min(0.0))
    p = (-1) ** m * math.prod(range(1, 2 * m, 2)) * s ** m if m else torch.ones_like(c)
    out, prev = {m: p}, None
    for l in range(m + 1, lmax + 1):
        if l == m + 1:
            prev, p = p, (2 * m + 1) * c * p
        else:
            prev, p = p, ((2 * l - 1) * c * p - (l + m - 1) * prev) / (l - m)
        out[l] = p
    return {l: math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) * v
            for l, v in out.items()}


class Bipolar:
    """Bond-frame blocks <f_a Y_{l_a m} | W(r_A, r_B) | g_b Y_{l_b m}> (right centre at +z distance d) by the bipolar
    quadrature of the module docstring; ``weight(r_A, r_B)`` is a multiplicative field of the two centre distances
    (None: 1). Shared by the pair-XC and environment-XC builders."""

    def __init__(self, left, right, weight=None, chunk_points=1 << 22, device='cuda'):
        self.left, self.right, self.weight = left, right, weight
        self.support = left.cut + right.cut
        self.shape = (int(left.offsets[-1]), int(right.offsets[-1]))
        self.chunk_points = int(chunk_points)
        self.device = torch.device(device)
        self._rules = {}
        la, lb = left.shells, right.shells
        self.mterms = []        # (m, left functions, right functions, [(rows, cols) per k])
        for m in range(min(max(la), max(lb)) + 1):
            ia = [i for i, l in enumerate(la) if l >= m]
            ib = [j for j, l in enumerate(lb) if l >= m]
            ks = [0] if m == 0 else [2 * m - 1, 2 * m]
            idx = [([int(left.offsets[i]) + k for i in ia], [int(right.offsets[j]) + k for j in ib]) for k in ks]
            self.mterms.append((m, ia, ib, idx))

    def rule(self, n):
        if n not in self._rules:
            from scipy.special import roots_legendre
            x, w = roots_legendre(n)
            self._rules[n] = (torch.as_tensor(x, device=self.device), torch.as_tensor(w, device=self.device))
        return self._rules[n]

    def cuts(self, d, partition):
        """Segment bounds (lo, hi) [D, S] of the r integration; padding segments have zero length."""
        aa, bb = self.left.cut, self.right.cut
        cand = np.stack([np.zeros_like(d), np.full_like(d, aa), d, bb - d, d - bb, d + bb], 1)
        if partition:
            q = np.arange(partition, aa, partition)
            cand = np.concatenate([cand, np.broadcast_to(q, (len(d), len(q)))], 1)
        valid = (cand > 0) & (cand < aa)
        valid[:, :2] = True
        cand = np.sort(np.where(valid, cand, aa), axis=1)
        return cand[:, :-1], cand[:, 1:]

    def _field(self, ra, rb):
        return None if self.weight is None else self.weight(ra, rb)

    def onsite(self, n):
        x, w = self.rule(n)
        A, B = self.left, self.right
        mn = min(A.cut, B.cut)
        rr = (x + 1) * mn / 2
        ww = w * mn / 2 * rr * rr
        v = self._field(rr, rr)
        fa, fb = A(rr), B(rr)
        block = torch.zeros(self.shape, dtype=torch.float64, device=self.device)
        for i, l in enumerate(A.shells):
            for j, ll in enumerate(B.shells):
                if l == ll:
                    z = ww * fa[i] * fb[j]
                    z = (z if v is None else z * v).sum()
                    k = torch.arange(2 * l + 1, device=self.device)
                    block[int(A.offsets[i]) + k, int(B.offsets[j]) + k] = z
        return block

    def bipolar(self, d, lo, hi, n):
        """Blocks [D, n_A, n_B] at distances d [D] (0 < d < support) from segments lo, hi [D, S] (device FP64)."""
        A, B = self.left, self.right
        x, w = self.rule(n)
        dd = d[:, None, None]
        r = lo[..., None] + (x + 1) * (hi - lo)[..., None] / 2                 # [D, S, n]
        wr = w * (hi - lo)[..., None] / 2
        lower = (r - dd).abs()
        span = ((r + dd).clamp_max(B.cut) - lower).clamp_min(0.0)
        rb = lower[..., None] + (x + 1) * span[..., None] / 2                  # [D, S, n, n]
        wb = w * span[..., None] / 2
        r4, d4 = r[..., None], dd[..., None]
        ca = ((r4 * r4 + d4 * d4 - rb * rb) / (2 * r4 * d4)).clamp(-1.0, 1.0)
        cb = ((r4 * r4 - d4 * d4 - rb * rb) / (2 * rb * d4)).clamp(-1.0, 1.0)
        D = len(d)
        ww = wr[..., None] * wb * r4 * rb / d4
        v = self._field(r4, rb)                                                # broadcasts to [D, S, n, n]
        if v is not None:
            ww = ww * v
        ww = ww.reshape(D, 1, -1)
        del v, wb
        fa = A(r)[..., None]                                                   # [nA, D, S, n, 1]
        fb = B(rb)                                                             # [nB, D, S, n, n]
        block = torch.zeros((D, *self.shape), dtype=torch.float64, device=self.device)
        for m, ia, ib, idx in self.mterms:
            ta, tb = _theta(m, max(A.shells), ca), _theta(m, max(B.shells), cb)
            left = torch.stack([(fa[i] * ta[A.shells[i]]).reshape(D, -1) for i in ia], 1) * ww
            right = torch.stack([(fb[j] * tb[B.shells[j]]).reshape(D, -1) for j in ib], 1)
            vals = torch.bmm(left, right.transpose(1, 2)) * (2 * math.pi)
            del left, right
            for rows, cols in idx:
                block[:, torch.as_tensor(rows, device=self.device)[:, None],
                      torch.as_tensor(cols, device=self.device)[None, :]] += vals
        return block

    def blocks(self, distances, n, partition=0.0):
        """C(d) [len(d), n_A, n_B] at host distances (numpy FP64)."""
        d = np.asarray(distances, dtype=np.float64)
        out = torch.zeros((len(d), *self.shape), dtype=torch.float64, device=self.device)
        zero = d < 1e-12
        if zero.any():
            out[torch.as_tensor(np.nonzero(zero)[0], device=self.device)] = self.onsite(n)
        gen = np.nonzero(~zero & (d < self.support - 1e-12))[0]
        if not len(gen):
            return out
        lo, hi = self.cuts(d[gen], partition)
        step = max(1, self.chunk_points // (lo.shape[1] * n * n))
        dev = self.device
        for a in range(0, len(gen), step):
            sl = slice(a, a + step)
            out[torch.as_tensor(gen[sl], device=dev)] = self.bipolar(
                torch.as_tensor(d[gen[sl]], device=dev), torch.as_tensor(lo[sl], device=dev),
                torch.as_tensor(hi[sl], device=dev), n)
        return out


def pair_xc(A, B, chunk_points=1 << 22):
    """Bipolar integrator of the pair-XC blocks of species A (left) and B (right)."""
    return Bipolar(A.orbitals, B.orbitals, lambda ra, rb: lda_pz81(A.density(ra) + B.density(rb))[1], chunk_points,
                   A.device)


def _spline_at(knots, values, at):
    """Not-a-knot spline through (knots, values [K, ...]) evaluated at host points ``at``."""
    x = torch.as_tensor(knots, device=values.device)
    c = spline_coefficients(x, values.reshape(len(knots), -1))
    q = torch.as_tensor(np.asarray(at, dtype=np.float64), device=values.device)
    i = (torch.searchsorted(x, q, right=True) - 1).clamp(0, len(x) - 2)
    t = (q - x[i])[:, None]
    g = c[i]
    return (((g[:, 0] * t + g[:, 1]) * t + g[:, 2]) * t + g[:, 3]).reshape(len(at), *values.shape[1:])


def converge(pxc, variant='standard'):
    """Walk the acceptance ladder of one variant -> (passed, knots, values [K, nA, nB] on device, attempts)."""
    v = VARIANTS[variant]
    support = pxc.support
    held = np.r_[.137, 1.337, np.linspace(2.313, support - .217, 6)]
    refs = pxc.blocks(held, v['reference'], v['partition'])
    qerr = float((pxc.blocks(held, v['probe'], v['partition']) - refs).abs().max())
    attempts, knots, values = [], None, None
    for order, step in v['ladder']:
        knots = np.linspace(0, support, int(np.ceil(support / step)) + 1)
        values = pxc.blocks(knots, order, v['partition'])
        ierr = float((_spline_at(knots, values, held) - refs).abs().max())
        row = dict(order=order, step_bohr_max=step, interpolation_max_eV=ierr, reference_change_eV=qerr)
        if v['partition']:
            row['radial_partition_bohr'] = v['partition']
        attempts.append(row)
        if ierr < TOL_EV and qerr < TOL_EV:
            return True, knots, values, attempts
        if qerr >= TOL_EV:
            break
    return False, knots, values, attempts


GENERATOR = ('dptb.prior.gf.build_pairxc', 'dptb.prior.common.xc', 'dptb.prior.common.radial')


def build(pairs, out_dir, sources_dir=None, device='cuda', variants=('standard', 'partitioned'),
          chunk_points=1 << 22, species_files=None):
    """Build the pair-XC tables of ``pairs`` ('A|B' or (A, B); stored sorted) into the store ``out_dir``.

    Each pair walks the variants in order until one passes; its manifest row records the attempts, the seconds and
    the source identities (a failing pair goes to ``failures`` and makes the manifest incomplete). Pairs the store
    already holds are skipped (tables are never rewritten). Inputs: ``sources_dir/<S>.npz`` or
    ``species_files[S] = (orb, upf)``. Returns [(pair, path or None, seconds)].
    """
    store = TableStore(out_dir, SCHEMA)
    gid = store.generator(*GENERATOR)
    src = Sources(sources_dir, species_files, device)
    failures = store.manifest.setdefault('failures', [])
    results = []
    for p in pairs:
        sa, sb = sorted(p.split('|') if isinstance(p, str) else p)
        key = f'{sa}|{sb}'
        if store.has('pairs', key):
            continue
        t0 = time.perf_counter()
        A, B = src[sa], src[sb]
        pxc = pair_xc(A, B, chunk_points)
        attempts, row = [], None
        for variant in variants:
            ok, knots, values, rows = converge(pxc, variant)
            attempts += rows
            if ok:
                name = f'{sa}_{sb}' + ('' if variant == 'standard' else f'_{variant}') + '.npz'
                row = store.write_npz(name, dict(distances=knots, values_eV=values.cpu().numpy(),
                                                 left_shells=np.asarray(A.shells, np.int64),
                                                 right_shells=np.asarray(B.shells, np.int64),
                                                 support_bohr=np.array(pxc.support)), compress=True)
                break
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        idents = {s: src[s].identity['sha256'] for s in (sa, sb)}
        failures[:] = [f for f in failures if f.get('pair') != key]
        if row is None:
            failures.append(dict(pair=key, passed=False, attempts=attempts, seconds=seconds, atomic_sources=idents))
        else:
            store.add('pairs', key, dict(file=row['path'], sha256=row['sha256'], bytes=row['bytes'], passed=True,
                                         atomic_sources=idents, attempts=attempts, seconds=seconds, generator=gid))
        results.append((key, None if row is None else store.root / row['path'], seconds))
    n = len(store.manifest.get('pairs', {}))
    store.flush(passed=n, total=n + len(failures), complete=not failures)
    return results
