from __future__ import annotations

import numpy as np


def lda_pz81_vxc_ry(rho: np.ndarray) -> np.ndarray:
    """Unpolarized Perdew-Zunger 1981 LDA potential in Rydberg.

    Parameters reproduce the Ceperley-Alder fit in:
    J. P. Perdew and A. Zunger, Phys. Rev. B 23, 5048 (1981),
    DOI 10.1103/PhysRevB.23.5048.
    Input density is electrons / bohr^3.
    """
    n = np.asarray(rho, dtype=float)
    out = np.zeros_like(n)
    mask = n > 1e-20
    if not np.any(mask):
        return out
    nm = n[mask]
    rs = (3.0 / (4.0 * np.pi * nm)) ** (1.0 / 3.0)

    # Exchange potential in Hartree.
    vx = -((3.0 / np.pi) ** (1.0 / 3.0)) * nm ** (1.0 / 3.0)

    vc = np.empty_like(rs)
    high_density = rs < 1.0
    if np.any(high_density):
        r = rs[high_density]
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        eps = A * np.log(r) + B + C * r * np.log(r) + D * r
        deps = A / r + C * (np.log(r) + 1.0) + D
        vc[high_density] = eps - r * deps / 3.0
    if np.any(~high_density):
        r = rs[~high_density]
        gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
        den = 1.0 + beta1 * np.sqrt(r) + beta2 * r
        eps = gamma / den
        deps = -gamma * (beta1 / (2.0 * np.sqrt(r)) + beta2) / den**2
        vc[~high_density] = eps - r * deps / 3.0

    out[mask] = 2.0 * (vx + vc)  # hartree -> rydberg
    return out


def _periodic_divergence(vector: tuple[np.ndarray, np.ndarray, np.ndarray], gvec: np.ndarray) -> np.ndarray:
    """Spectral divergence on the same periodic FFT grid as ``gvec``."""
    div_g = np.zeros(vector[0].shape, dtype=complex)
    ngrid = vector[0].size
    for axis, component in enumerate(vector):
        coeff = np.fft.fftn(np.asarray(component, dtype=float)) / ngrid
        div_g += 1j * gvec[..., axis] * coeff
    return (np.fft.ifftn(div_g) * ngrid).real


def pbe_libxc_derivatives(rho, sigma, *, density_threshold=1.0e-6):
    """Built-in ABACUS PBE: local PW and separately gated GGA correction."""
    from .builtin_pbe import scalar
    if density_threshold != 1e-6:
        raise ValueError('Built-in PBE uses fixed reference gates; custom threshold is unsupported')
    n=np.asarray(rho,dtype=float);s=np.asarray(sigma,dtype=float)
    if n.shape!=s.shape or not np.isfinite(n).all() or not np.isfinite(s).all():
        raise ValueError('Finite matching rho/sigma arrays required')
    return scalar(n,s)


def pbe_vxc_ry(
    rho: np.ndarray,
    grad_rho: tuple[np.ndarray, np.ndarray, np.ndarray],
    gvec: np.ndarray,
    *,
    density_threshold: float = 1.0e-6,
    pw_mask=None,
) -> np.ndarray:
    """Built-in ABACUS PBE potential with a shared spectral divergence.

    Local Slater/PW survives above |rho|>1e-10. Gradient correction follows
    the separate built-in rho>1e-6 and sigma>=1e-10 gates. Custom scientific
    cutoffs are rejected for this reference contract.
    """
    n = np.asarray(rho, dtype=float)
    sigma = sum(np.asarray(g, dtype=float) ** 2 for g in grad_rho)
    vrho, vsigma_grid = pbe_libxc_derivatives(n, sigma, density_threshold=density_threshold)
    flux = tuple(2.0 * vsigma_grid * np.asarray(g, dtype=float) for g in grad_rho)
    from .pw_derivatives import derivative
    mask=np.ones(n.shape,dtype=bool) if pw_mask is None else pw_mask
    v_hartree = vrho - sum(derivative(f,gvec,mask,axis) for axis,f in enumerate(flux))
    return 2.0 * v_hartree  # hartree -> rydberg
