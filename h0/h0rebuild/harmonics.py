from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


def angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    r = np.linalg.norm(vectors, axis=-1)
    safe = np.where(r > 0.0, r, 1.0)
    zc = np.clip(vectors[..., 2] / safe, -1.0, 1.0)
    theta = np.arccos(zc)
    phi = np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), 2.0 * np.pi)
    theta = np.where(r > 0.0, theta, 0.0)
    phi = np.where(r > 0.0, phi, 0.0)
    return r, theta, phi


def complex_ylm(l: int, m: int, vectors: np.ndarray) -> np.ndarray:
    _, theta, phi = angles(vectors)
    return sph_harm_y(l, m, theta, phi)


def real_ylm_abacus(l: int, m: int, vectors: np.ndarray) -> np.ndarray:
    """ABACUS real spherical harmonics.

    Ordering is handled outside this function. For m>0 this is sqrt(2) Re Y_l^m;
    for m<0 it is sqrt(2) Im Y_l^{|m|}, with the Condon-Shortley phase used by
    SciPy and ABACUS. Thus p is ordered (p_z, p_x-like with ABACUS sign,
    p_y-like with ABACUS sign) as m=(0,+1,-1).
    """
    if m == 0:
        return complex_ylm(l, 0, vectors).real
    y = complex_ylm(l, abs(m), vectors)
    return np.sqrt(2.0) * (y.real if m > 0 else y.imag)


def spinor_cg(l: int, j: float, mj: float, spin: int) -> tuple[int, float]:
    """Return (m_l, coefficient) for |l,1/2;j,mj> and spin 0=up/1=down.

    The relative phase follows the same convention as ABACUS ``Soc::spinor``.
    A global phase for an entire (j,mj) spinor cancels in a KB projector.
    """
    den = 2.0 * l + 1.0
    if abs(j - (l + 0.5)) < 1e-8:
        if spin == 0:
            ml = int(round(mj - 0.5))
            coeff = np.sqrt(max(0.0, (l + mj + 0.5) / den))
        else:
            ml = int(round(mj + 0.5))
            coeff = np.sqrt(max(0.0, (l - mj + 0.5) / den))
    elif l > 0 and abs(j - (l - 0.5)) < 1e-8:
        if spin == 0:
            ml = int(round(mj - 0.5))
            coeff = np.sqrt(max(0.0, (l - mj + 0.5) / den))
        else:
            ml = int(round(mj + 0.5))
            coeff = -np.sqrt(max(0.0, (l + mj + 0.5) / den))
    else:
        raise ValueError(f"Incompatible l={l}, j={j}")
    if abs(ml) > l:
        return ml, 0.0
    return ml, float(coeff)


def mj_values(j: float) -> np.ndarray:
    n = int(round(2.0 * j)) + 1
    return -j + np.arange(n, dtype=float)
