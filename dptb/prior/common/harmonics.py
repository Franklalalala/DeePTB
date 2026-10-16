"""Real spherical harmonics in the ABACUS convention, on the device (FP64).

Order within a degree l is m = 0, +1, -1, +2, -2, ...; m > 0 is sqrt(2) Re Y_l^m and m < 0 is sqrt(2) Im Y_l^|m|, with
the Condon-Shortley phase. ``rotation_matrices`` gives the matrices D^l(Q) with Y_l(Q^T x) = D^l(Q) Y_l(x) for the
rotation Q taking +z to a bond direction, which turns a bond-frame (Slater-Koster) block into the lab frame:
block_lab = D^l1 block_bond (D^l2)^T.

``real_harmonics`` (tensors) and ``real_ylm`` (Triton device function, used by the GF rotation-pack kernel and the H0
AO kernel) evaluate the same expression: (x + i y)^|m| and the Legendre recursion in z from (-1)^|m| (2|m| - 1)!!.
"""
import math
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl


def real_harmonics(l, vectors):
    """Y_lm(v / |v|) for the rows of ``vectors`` [..., 3] -> [..., 2l + 1]."""
    xyz = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-30)
    x, y, z = xyz.unbind(-1)
    real, imag = torch.ones_like(x), torch.zeros_like(x)
    out = []
    for m in range(l + 1):
        if m:
            real, imag = real * x - imag * y, imag * x + real * y
        q = torch.full_like(z, (-1) ** m * math.prod(range(1, 2 * m, 2)))
        if l > m:
            prev, q = q, (2 * m + 1) * z * q
            for degree in range(m + 2, l + 1):
                prev, q = q, ((2 * degree - 1) * z * q - (degree + m - 1) * prev) / (degree - m)
        norm = harmonic_norm(l, m)
        if m == 0:
            out.append(norm * q)
        else:
            out.extend((math.sqrt(2) * norm * q * real, math.sqrt(2) * norm * q * imag))
    return torch.stack(out, dim=-1)


@triton.jit
def real_ylm(l, m, x, y, z, nrm):
    """Y_lm at the unit vectors (x, y, z) (tensors of one shape; l and the signed ABACUS m are scalars):
    nrm P(z) Re / Im (x + i y)^|m| (sqrt(2) for m != 0) with P(z) = d^|m| P_l / dz^|m| times (-1)^|m| (the recursion of
    ``real_harmonics``); nrm = sqrt((2l + 1) / (4 pi) (l - |m|)! / (l + |m|)!)."""
    am = tl.where(m >= 0, m, -m)
    re = tl.zeros_like(x) + 1.0
    im = tl.zeros_like(x)
    dfac = 1.0
    for k in range(am):                         # (x + i y)^|m| and (2|m| - 1)!!
        re, im = re * x - im * y, im * x + re * y
        dfac = dfac * (2 * k + 1)
    p = tl.zeros_like(x) + (1.0 - 2.0 * (am % 2)) * dfac
    if l > am:
        prev = p
        p = (2 * am + 1) * z * p
        for deg in range(am + 2, l + 1):
            prev, p = p, ((2 * deg - 1) * z * p - (deg + am - 1) * prev) / (deg - am)
    return tl.where(m == 0, nrm * p, tl.where(m > 0, 1.4142135623730951 * nrm * p * re,
                                              1.4142135623730951 * nrm * p * im))


def harmonic_norm(l, m):
    """nrm of ``real_ylm``: sqrt((2l + 1) / (4 pi) (l - |m|)! / (l + |m|)!)."""
    m = abs(m)
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))


def solid_harmonics(lmax, vectors):
    """r^l Y_lm(v) for l <= lmax -> [..., (lmax + 1)^2], index l^2 + (2m - 1 if m > 0 else -2m)."""
    r = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return torch.cat([real_harmonics(l, vectors) * r ** l for l in range(lmax + 1)], dim=-1)


def rotation_z_to(vectors):
    """Rotation matrices [Q, 3, 3] taking +z to v / |v| (identity for v along +z or v = 0)."""
    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    n = vectors / radius.clamp_min(1e-30)
    x, y, z = n.unbind(-1)
    zero = torch.zeros_like(z)
    skew = torch.stack((zero, zero, x, zero, zero, y, -x, -y, zero), -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=vectors.device, dtype=vectors.dtype)
    # no cancellation of 1 + z near the south pole
    factor = torch.where(z < 0, (1 - z) / (x * x + y * y).clamp_min(1e-30), 1 / (1 + z).clamp_min(1e-30))
    rotation = eye + skew + (skew @ skew) * factor[:, None, None]
    south = torch.diag(vectors.new_tensor([1., -1., -1.]))
    rotation = torch.where((z <= -1 + 1e-14)[:, None, None], south, rotation)
    return torch.where(((z >= 1 - 1e-14) | (radius[:, 0] <= 1e-14))[:, None, None], eye, rotation)


@lru_cache(maxsize=None)
def _collocation(l):
    """2l + 1 well-conditioned directions and the inverse of Y_l on them (host FP64)."""
    from scipy.linalg import qr
    n_mu, n_phi = max(8, 2 * l + 5), max(16, 4 * l + 8)
    mu, _ = np.polynomial.legendre.leggauss(n_mu)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    s = np.sqrt(1 - mu ** 2)
    dirs = np.stack([np.outer(s, np.cos(phi)), np.outer(s, np.sin(phi)), np.outer(mu, np.ones(n_phi))], -1).reshape(-1, 3)
    base = real_harmonics(l, torch.as_tensor(dirs)).numpy()
    _, _, piv = qr(base.T, pivoting=True)
    rows = piv[:2 * l + 1]
    return dirs[rows], np.linalg.inv(base[rows])


def rotation_matrices(vectors, degrees):
    """{l: D^l [Q, 2l + 1, 2l + 1]} of the rotation taking +z to each displacement (identity at v = 0)."""
    cart = rotation_z_to(vectors)
    out = {}
    for l in degrees:
        dirs, inv = _collocation(l)
        d = torch.as_tensor(dirs, dtype=vectors.dtype, device=vectors.device)
        y = real_harmonics(l, d @ cart.transpose(-1, -2))          # [Q, 2l + 1 directions, 2l + 1]
        out[l] = (torch.as_tensor(inv, dtype=vectors.dtype, device=vectors.device) @ y).transpose(-1, -2)
    return out
