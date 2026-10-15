"""Spherical Bessel transforms and two-centre radial tables on the device (FP64), shared by the H0 and GF table builders.

Bessel functions, all orders l <= lmax at once (``spherical_jn``): the power series of j_l(x)/x^l below x = 1, Miller's
downward recurrence normalised by sum_n (2n + 1) j_n(x)^2 = 1 for 1 <= x <= lmax, and upward recurrence from
j_0 = sin x / x, j_1 = sin x / x^2 - cos x / x above lmax (where it is stable).

Uniform grid (H0 pair bank, the ABACUS ModuleNAO convention; ``UniformSBT``): r_i = i dr (i < nr), cutoff = (nr - 1) dr,
k_j = j pi / cutoff (j < nr). For a radial function v with implicit power p (v = r^p f):
    F(k) = sqrt(2/pi) sum_i dr r_i^(2-p) v_i j_l(k r_i)
and for a pair of transforms with P = F1 F2 (overlap-like) or k^2 F1 F2 (kinetic), for every coupling order L:
    y(R)  =  4 pi dk sum_j k_j^(L+2) P_j g_L(k_j R)                 = S_L(R) / R^L
    y'(R) = -4 pi dk R sum_j k_j^(L+4) P_j g_(L+1)(k_j R)           = d/dR (S_L(R) / R^L)
with g_L(x) = j_L(x) / x^L (series at small x, so small R has no cancellation); ``hermite`` turns (y, y') on the grid
into per-interval cubic coefficients ((c3 w + c2) w + c1) w + c0.

Bond frame (GF radial block tables; ``BondFrameSBT``): source grids with trapezoid weights, k = linspace(0, kmax, n_k)
with trapezoid weights w_k, F_c(k) = int r^2 phi_c(r) j_lc(k r) dr, and for descriptors a = (c, l_a, m_a), b = (c', l_b, m_b)
at bond distance d along +z:
    V_ab(d) = sum_l (2l + 1) Re(i^(l_a - l_b - l)) C^l_ab sum_k (2/pi) w_k k^2 [k^2] F_c(k) F_c'(k) j_l(k d)
    C^l_ab  = int Y_a(Omega) P_l(cos theta) Y_b(Omega) dOmega   (Gauss-Legendre x uniform-phi quadrature, exact here)
then zeroed for m_a != m_b and for d >= rc_c + rc_c' (the exact axial rule and finite support).
"""
import math

import numpy as np
import torch

PI = math.pi


def _series(lmax, x, terms=24):
    """[lmax + 1, ...] j_l(x) / x^l by the power series (|x| < 1)."""
    x2 = -0.5 * x * x
    out = []
    for l in range(lmax + 1):
        term = torch.ones_like(x)
        total = torch.ones_like(x)
        for m in range(1, terms):
            term = term * x2 / (m * (2 * l + 2 * m + 1))
            total = total + term
        out.append(total / float(np.prod(np.arange(1, 2 * l + 2, 2, dtype=np.float64))))
    return torch.stack(out)


def spherical_jn(lmax, x, *, scaled=False):
    """[lmax + 1, *x.shape] j_l(x) for l = 0..lmax (x >= 0, FP64); ``scaled`` returns j_l(x) / x^l."""
    x = torch.as_tensor(x, dtype=torch.float64)
    small = x < 1.0
    xs = torch.where(small, torch.full_like(x, 2.0), x)       # safe argument for the recurrences
    s, c = torch.sin(xs), torch.cos(xs)
    up = [s / xs, s / (xs * xs) - c / xs]
    for n in range(1, lmax):
        up.append((2 * n + 1) / xs * up[n] - up[n - 1])
    out = torch.stack(up[:lmax + 1])
    if lmax >= 1:
        mid = (~small) & (x <= lmax)
        if bool(mid.any()):
            xm = torch.where(mid, x, torch.full_like(x, 1.5))
            top = 2 * lmax + 30
            f_next, f = torch.zeros_like(xm), torch.full_like(xm, 1e-100)
            down = [None] * (lmax + 1)
            norm = (2 * top + 1) * f * f
            for n in range(top, 0, -1):
                f_next, f = f, (2 * n + 1) / xm * f - f_next     # f_(n-1)
                norm = norm + (2 * n - 1) * f * f
                if n - 1 <= lmax:
                    down[n - 1] = f
            d = torch.stack(down)
            sign = torch.sign(d[0] * up[0] + d[1] * torch.where(mid, up[1], up[1]))
            d = d * (sign / torch.sqrt(norm))
            out = torch.where(mid, d, out)
    ser = _series(lmax, torch.where(small, x, torch.zeros_like(x)))
    if scaled:
        powers = torch.stack([xs ** l for l in range(lmax + 1)])
        return torch.where(small, ser, out / powers)
    powers = torch.stack([torch.where(small, x, xs) ** l for l in range(lmax + 1)])
    return torch.where(small, ser * powers, out)


def hermite(y, dy, dr):
    """Cubic Hermite coefficients [..., n - 1, 4] of grid values and derivatives [..., n]: ((c3 w + c2) w + c1) w + c0."""
    inv = 1.0 / dr
    dd = (y[..., 1:] - y[..., :-1]) * inv
    c0, c1 = y[..., :-1], dy[..., :-1]
    c3 = (c1 + dy[..., 1:] - 2.0 * dd) * inv * inv
    c2 = (dd - c1) * inv - c3 * dr
    return torch.stack([c0, c1, c2, c3], dim=-1)


def trapezoid_weights(r):
    """Weights w with sum_i w_i f_i = trapz(f, r)."""
    r = torch.as_tensor(r, dtype=torch.float64)
    w = torch.zeros_like(r)
    h = r[1:] - r[:-1]
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    return w


class UniformSBT:
    """Uniform r grid, FFT-compliant k grid and the per-order transform matrices on the device (built once)."""

    def __init__(self, cutoff, nr, device='cuda'):
        self.cutoff, self.nr, self.device = float(cutoff), int(nr), torch.device(device)
        self.dr = self.cutoff / (self.nr - 1)
        self.dk = (PI / self.dr) / (self.nr - 1)
        self.r = torch.arange(self.nr, dtype=torch.float64, device=self.device) * self.dr
        self.k = torch.arange(self.nr, dtype=torch.float64, device=self.device) * self.dk
        self._J, self._G = {}, {}

    def J(self, l):
        """[nr, nk] j_l(r_i k_j)."""
        if l not in self._J:
            self._J[l] = spherical_jn(l, torch.outer(self.r, self.k))[l]
        return self._J[l]

    def G(self, L):
        """[nk, nr] g_L(k_j R_i) = j_L(k_j R_i) / (k_j R_i)^L."""
        if L not in self._G:
            self._G[L] = spherical_jn(L, torch.outer(self.k, self.r), scaled=True)[L]
        return self._G[L]

    def forward(self, values, l, p):
        """[n, nk] F(k) of radial functions [n, nr] of order l with implicit power p."""
        w = self.r ** (2 - p) * (self.dr * math.sqrt(2.0 / PI))
        return (values.to(self.device, torch.float64) * w) @ self.J(l)

    def two_centre(self, P, L, kinetic=False):
        """y = S_L(R)/R^L and y' = d/dR y on the r grid for products P [n, nk] of transforms coupled to order L."""
        k = self.k
        if kinetic:
            P = P * k * k
        pref = 4.0 * PI * self.dk
        y = pref * ((P * k ** (L + 2)) @ self.G(L))
        dy = -pref * self.r * ((P * k ** (L + 4)) @ self.G(L + 1))
        return y, dy


def angular_grid(n_mu, n_phi, device='cuda'):
    """Gauss-Legendre x uniform (half-offset) phi directions [n_mu n_phi, 3] and weights (sum 4 pi)."""
    mu, wmu = np.polynomial.legendre.leggauss(int(n_mu))
    phi = (np.arange(int(n_phi)) + 0.5) * (2.0 * PI / int(n_phi))
    st = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    v = np.stack([np.multiply.outer(st, np.cos(phi)), np.multiply.outer(st, np.sin(phi)),
                  np.multiply.outer(mu, np.ones(int(n_phi)))], -1).reshape(-1, 3)
    w = np.multiply.outer(wmu, np.full(int(n_phi), 2.0 * PI / int(n_phi))).reshape(-1)
    return torch.as_tensor(v, device=device), torch.as_tensor(w, device=device)


def legendre(lmax, mu):
    """[lmax + 1, ...] P_l(mu)."""
    out = [torch.ones_like(mu), mu]
    for l in range(1, lmax):
        out.append(((2 * l + 1) * mu * out[l] - l * out[l - 1]) / (l + 1))
    return torch.stack(out[:lmax + 1])


class Channels:
    """Transforms F [n_c, n_k] of a channel set with its orders and cutoffs (descriptors: m = 0, +1, -1, ... per channel)."""

    def __init__(self, F, ls, cuts):
        self.F, self.ls, self.cuts = F, list(ls), list(cuts)


def distance_grid(support, step):
    """0, step, ... up to the first multiple of step at least support + step / 4 (the GF table distance grids)."""
    if support <= 0.0 or step <= 0.0:
        raise ValueError('support and distance step must be positive')
    end = math.ceil(float(support) / float(step)) * float(step)
    if end < support + 0.25 * step:
        end += float(step)
    return np.linspace(0.0, end, int(round(end / float(step))) + 1, dtype=np.float64)


class BondFrameSBT:
    """Radial block tables V_ab(d) of two channel sets (see the module docstring), all on the device."""

    def __init__(self, kmax, n_k, n_mu, n_phi, device='cuda'):
        self.device = torch.device(device)
        self.k = torch.linspace(0.0, float(kmax), int(n_k), dtype=torch.float64, device=self.device)
        wk = torch.full((int(n_k),), float(kmax) / (int(n_k) - 1), dtype=torch.float64, device=self.device)
        wk[[0, -1]] *= 0.5
        self.factor = (2.0 / PI) * wk * self.k * self.k
        self.n_mu, self.n_phi = int(n_mu), int(n_phi)
        self._ang = {}

    def transforms(self, r, channels):
        """Channels of [(l, values on r, cutoff)]: F_c(k) = trapz(r^2 phi_c j_lc(k r), r)."""
        r = torch.as_tensor(np.asarray(r, dtype=np.float64), device=self.device)
        w = trapezoid_weights(r) * r * r
        ls = [int(c[0]) for c in channels]
        jl = spherical_jn(max(ls), torch.outer(r, self.k))                  # [l, n_r, n_k]
        v = torch.as_tensor(np.stack([np.asarray(c[1], dtype=np.float64) for c in channels]), device=self.device) * w
        F = torch.stack([v[i] @ jl[l] for i, l in enumerate(ls)])
        return Channels(F, ls, [float(c[2]) for c in channels])

    def coupling(self, lmax_left, lmax_right):
        """[l, (lmax_left + 1)^2, (lmax_right + 1)^2] (2l + 1) Re(i^(la - lb - l)) C^l_ab over (l, m) slots
        indexed l^2 + (0, +1, -1, ...)."""
        key = (lmax_left, lmax_right)
        if key not in self._ang:
            from dptb.prior.common.harmonics import real_harmonics
            v, w = angular_grid(self.n_mu, self.n_phi, self.device)
            yl = torch.cat([real_harmonics(l, v) for l in range(lmax_left + 1)], -1)
            yr = torch.cat([real_harmonics(l, v) for l in range(lmax_right + 1)], -1)
            lt = lmax_left + lmax_right
            P = legendre(lt, v[:, 2]) * w
            C = torch.einsum('pa,lp,pb->lab', yl, P, yr)
            la = torch.cat([torch.full((2 * l + 1,), l) for l in range(lmax_left + 1)])
            lb = torch.cat([torch.full((2 * l + 1,), l) for l in range(lmax_right + 1)])
            e = (la[None, :, None] - lb[None, None, :] - torch.arange(lt + 1)[:, None, None]) % 4
            phase = torch.where(e == 0, 1.0, torch.where(e == 2, -1.0, 0.0)).to(torch.float64)
            phase = phase * (2 * torch.arange(lt + 1, dtype=torch.float64) + 1)[:, None, None]
            self._ang[key] = phase.to(self.device) * C
        return self._ang[key]

    def tables(self, left, right, distances, *, terms=None, kinetic=False):
        """[n_d, n_a, n_b] V_ab(d) on the descriptor layout of ``left``, ``right`` (Channels). ``terms``: a list of
        (F_left, F_right, kinetic) whose radial products are summed (default: left.F, right.F, ``kinetic``)."""
        dev = self.device
        terms = terms or [(left.F, right.F, kinetic)]
        d = torch.as_tensor(np.asarray(distances, dtype=np.float64), device=dev)
        ll, lr = left.ls, right.ls
        lt = max(ll) + max(lr)
        nk = len(self.k)
        prod = sum((Fa[:, None, :] * Fb[None, :, :]).reshape(-1, nk) * (self.factor * self.k * self.k if kin
                                                                          else self.factor)
                   for Fa, Fb, kin in terms)                               # [c1 c2, n_k]
        jd = spherical_jn(lt, torch.outer(self.k, d))                       # [l, n_k, n_d]
        radial = (prod @ jd).reshape(lt + 1, len(ll), len(lr), len(d))       # [l, c1, c2, n_d]
        idx = lambda ls, f: torch.as_tensor([f(i, l, q) for i, l in enumerate(ls) for q in range(2 * l + 1)],  # noqa: E731
                                            device=dev)
        ca, cb = idx(ll, lambda i, l, q: i), idx(lr, lambda i, l, q: i)
        sa, sb = idx(ll, lambda i, l, q: l * l + q), idx(lr, lambda i, l, q: l * l + q)
        ma, mb = (idx(ls, lambda i, l, q: (q + 1) // 2 * (1 if q % 2 else -1)) for ls in (ll, lr))
        coef = self.coupling(max(ll), max(lr))[:, sa][:, :, sb]             # [l, a, b]
        out = torch.einsum('lab,labd->dab', coef, radial[:, ca][:, :, cb])
        rca = torch.as_tensor(left.cuts, dtype=torch.float64, device=dev)[ca]
        rcb = torch.as_tensor(right.cuts, dtype=torch.float64, device=dev)[cb]
        keep = (ma[:, None] == mb[None, :])[None] & (d[:, None, None] < (rca[:, None] + rcb[None, :])[None] - 1e-12)
        return torch.where(keep, out, torch.zeros_like(out))
