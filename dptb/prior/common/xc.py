"""Exchange-correlation functionals on the device (FP64 torch), shared by the H0 and GF priors and their table builders.

PBE with the ABACUS built-in gates (H0 local-potential field; Hartree atomic units inside, Ry potential out). The
energy densities are libxc's unpolarised expressions:

    LDA_X       e = -3/4 (3/pi)^(1/3) n^(4/3)
    LDA_C_PW    e = n G(rs; PW92),        rs = (3 / (4 pi n))^(1/3)
    GGA_X_PBE   e = e_LDA_X(n) (1 + kappa - kappa / (1 + mu s^2 / kappa)),   s^2 = sigma / (4 (3 pi^2)^(2/3) n^(8/3))
    GGA_C_PBE   e = n (G(rs; PW92 modified) + gamma ln(1 + beta/gamma (t^2 + A t^4) / (1 + A t^2 + A^2 t^4))),
                t^2 = sigma / (4 k_s^2 n^2), k_s^2 = 4 (3 pi^2 n)^(1/3) / pi, A = beta/gamma / (exp(-eps_c / gamma) - 1)

with sigma = |grad n|^2; vrho = de/dn and vsigma = de/dsigma by autograd. Gates (ABACUS wrapper_xc / gradcorr): the
local part LDA_X + LDA_C_PW where n > 1e-10; the gradient correction (GGA value minus its sigma = 0 value) where
|n| > 1e-6 and sigma >= 1e-10. The potential is v = 2 (vrho - div(2 vsigma grad n)) Ry, the divergence taken as a
plane-wave derivative restricted to the charge sphere (``plane_wave_derivative``).

LDA (GF onsite exchange-correlation and tables; eV): Slater exchange plus Perdew-Zunger 81 correlation,

    eps_x = -3/4 (3 n / pi)^(1/3),   v_x = -(3 n / pi)^(1/3)
    eps_c = A ln rs + B + C rs ln rs + D rs  (rs < 1),   -gamma / (1 + beta1 sqrt(rs) + beta2 rs)  (rs >= 1)
    v_c = eps_c - rs/3 d eps_c / d rs

(Hartree, A = 0.0311, B = -0.048, C = 0.002, D = -0.0116, gamma = 0.1423, beta1 = 1.0529, beta2 = 0.3334), converted
with 2 Ry per Hartree and zero where n <= 1e-20; the constants are those of the GF onsite kernel.
"""
import math

import torch

PI = math.pi
RY = 13.605693122994
RS_FACTOR = (3.0 / (4.0 * PI)) ** (1.0 / 3.0)
PW = dict(a=0.031091, alpha1=0.21370, beta1=7.5957, beta2=3.5876, beta3=1.6382, beta4=0.49294, pp=1.0)
PW_MOD = dict(a=0.0310907, alpha1=0.21370, beta1=7.5957, beta2=3.5876, beta3=1.6382, beta4=0.49294, pp=1.0)
KAPPA = 0.8040
MU = 0.2195149727645171
BETA = 0.06672455060314922
GAMMA = (1.0 - math.log(2.0)) / PI ** 2
X2S = 1.0 / (2.0 * (6.0 * PI ** 2) ** (1.0 / 3.0))
PZ81 = dict(a=0.0311, b=0.048, c=0.002, d=0.0116, gamma=0.1423, beta1=1.0529, beta2=0.3334)
LDA_FLOOR = 1e-20


# ---------------------------------------------------------------------------------------------------------------- PBE
def _g_pw(rs, p):
    """PW92 G(rs) for the paramagnetic channel (libxc lda_c_pw.mpl g(1, rs))."""
    s = torch.sqrt(rs)
    aux = p['beta1'] * s + p['beta2'] * rs + p['beta3'] * rs * s + p['beta4'] * rs ** (p['pp'] + 1.0)
    return -2.0 * p['a'] * (1.0 + p['alpha1'] * rs) * torch.log(1.0 + 1.0 / (2.0 * p['a'] * aux))


def e_lda_x(n):
    return -0.75 * (3.0 / PI) ** (1.0 / 3.0) * n ** (4.0 / 3.0)


def e_lda_c_pw(n):
    rs = RS_FACTOR / n ** (1.0 / 3.0)
    return n * _g_pw(rs, PW)


def e_gga_x_pbe(n, sigma):
    """Unpolarised PBE exchange: e_x^LDA(n) F(s), s = X2S 2^(1/3) |grad n| / n^(4/3)."""
    s2 = (X2S * X2S) * (2.0 ** (2.0 / 3.0)) * sigma / n ** (8.0 / 3.0)
    f = 1.0 + KAPPA * (1.0 - KAPPA / (KAPPA + MU * s2))
    return e_lda_x(n) * f


def e_gga_c_pbe(n, sigma):
    """Unpolarised PBE correlation: n (eps_PW_mod(rs) + H(rs, t)), phi = 1."""
    rs = RS_FACTOR / n ** (1.0 / 3.0)
    ec = _g_pw(rs, PW_MOD)
    kf = (3.0 * PI ** 2 * n) ** (1.0 / 3.0)
    ks2 = 4.0 * kf / PI
    t2 = sigma / (4.0 * ks2 * n * n)
    A = (BETA / GAMMA) / (torch.exp(-ec / GAMMA) - 1.0)
    f1 = t2 + A * t2 * t2
    f2 = BETA * f1 / (GAMMA * (1.0 + A * f1))
    return n * (ec + GAMMA * torch.log(1.0 + f2))


def _derivatives(fn, n, sigma=None, chunk=1 << 22):
    """d fn / dn (and d fn / dsigma) elementwise by autograd, in chunks of ``chunk`` points."""
    vr = torch.empty_like(n)
    vs = torch.empty_like(n) if sigma is not None else None
    nf, vrf = n.reshape(-1), vr.reshape(-1)
    sf = sigma.reshape(-1) if sigma is not None else None
    vsf = vs.reshape(-1) if vs is not None else None
    with torch.enable_grad():
        for a in range(0, nf.numel(), chunk):
            x = nf[a:a + chunk].detach().requires_grad_(True)
            if sf is None:
                (g,) = torch.autograd.grad(fn(x).sum(), [x])
                vrf[a:a + chunk] = g
                continue
            s = sf[a:a + chunk].detach().requires_grad_(True)
            g, h = torch.autograd.grad(fn(x, s).sum(), [x, s])
            vrf[a:a + chunk] = g
            vsf[a:a + chunk] = h
    return vr, vs


def lda_pw_vrho(n):
    """vrho of LDA_X + LDA_C_PW (Hartree) where n > 1e-10, else 0 (the ABACUS local gate)."""
    active = n > 1e-10
    safe = torch.where(active, n, torch.ones_like(n))
    vx, _ = _derivatives(e_lda_x, safe)
    vc, _ = _derivatives(e_lda_c_pw, safe)
    return torch.where(active, vx + vc, torch.zeros_like(n))


def pbe_derivatives(rho, sigma):
    """(vrho, vsigma) of PBE with the ABACUS gates (Hartree); rho, sigma FP64 tensors of the same shape."""
    n = rho.abs()
    active = (n > 1e-6) & (sigma >= 1e-10)
    safe = torch.where(active, n, torch.ones_like(n))
    sg = torch.where(active, sigma, torch.zeros_like(sigma))
    vr = lda_pw_vrho(n)
    vs = torch.zeros_like(n)
    zero = torch.zeros_like(sg)
    for fn in (e_gga_x_pbe, e_gga_c_pbe):
        r, v = _derivatives(fn, safe, sg)
        r0, _ = _derivatives(fn, safe, zero)
        vr = vr + torch.where(active, r - r0, torch.zeros_like(r))
        vs = vs + torch.where(active, v, torch.zeros_like(v))
    return vr, vs


def plane_wave_derivative(values, gvec, mask, axis):
    """d values / d x_axis on the FFT grid, with the Fourier coefficients restricted to ``mask``."""
    coeff = torch.fft.fftn(values)
    return torch.fft.ifftn(1j * gvec[..., axis] * torch.where(mask, coeff, 0.)).real


def pbe_potential(rho, grad, gvec, mask):
    """PBE potential (Ry) on the FFT grid from the density rho and its gradient components grad (bohr^-3, bohr^-4)."""
    sigma = sum(g * g for g in grad)
    if not bool(torch.isfinite(rho).all() & torch.isfinite(sigma).all()):
        raise ValueError('non-finite density or gradient in the PBE potential')
    vrho, vsigma = pbe_derivatives(rho.to(torch.float64), sigma.to(torch.float64))
    v2 = 2 * vsigma
    div = sum(plane_wave_derivative(v2 * grad[axis], gvec, mask, axis) for axis in range(3))
    return 2 * (vrho - div)


# -------------------------------------------------------------------------------------------------------- LDA (PZ81)
def lda_pz81(rho):
    """(eps_xc, v_xc) in eV of Slater exchange + PZ81 correlation for densities rho (bohr^-3); 0 where rho <= 1e-20.

    eps_xc is the energy per electron; v_xc = d(rho eps_xc)/d rho.
    """
    p = PZ81
    ok = rho > LDA_FLOOR
    n = torch.where(ok, rho, torch.ones_like(rho))
    rs = torch.pow(3.0 / (4.0 * PI) / n, 1.0 / 3.0)
    lr = torch.log(rs)
    sr = torch.sqrt(rs)
    den = 1.0 + p['beta1'] * sr + p['beta2'] * rs
    hi = rs < 1.0
    e = torch.where(hi, p['a'] * lr - p['b'] + p['c'] * rs * lr - p['d'] * rs, -p['gamma'] / den)
    ep = torch.where(hi, p['a'] / rs + p['c'] * (lr + 1.0) - p['d'],
                     p['gamma'] * (p['beta1'] / (2.0 * sr) + p['beta2']) / (den * den))
    vx = -torch.pow(3.0 / PI * n, 1.0 / 3.0)
    zero = torch.zeros_like(rho)
    exc = torch.where(ok, 2.0 * RY * (0.75 * vx + e), zero)
    vxc = torch.where(ok, 2.0 * RY * (vx + e - rs * ep / 3.0), zero)
    return exc, vxc
