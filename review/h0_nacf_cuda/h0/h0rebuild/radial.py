from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.interpolate import CubicSpline

from .harmonics import real_ylm_abacus
from .models import OrbitalBasis


class RadialSpline:
    def __init__(self, r: np.ndarray, y: np.ndarray, cutoff: float | None = None, coefficients=None):
        self.r = np.asarray(r, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.cutoff = float(self.r[-1] if cutoff is None else cutoff)
        if self.r.ndim != 1 or self.y.shape != self.r.shape:
            raise ValueError("Radial mesh and values must be one-dimensional and equal-sized")
        self._spline = (CubicSpline(self.r, self.y, bc_type="not-a-knot", extrapolate=False) if coefficients is None else
                        CubicSpline.construct_fast(np.asarray(coefficients), self.r, extrapolate=False))

    def __call__(self, x: np.ndarray, derivative: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = (x >= self.r[0]) & (x <= self.cutoff)
        if np.any(mask):
            vals = self._spline(x[mask], nu=derivative)
            out[mask] = np.nan_to_num(vals)
        return out


class OrbitalEvaluator:
    def __init__(self, basis: OrbitalBasis):
        self.basis = basis
        prepared = basis.metadata.get('offline_spline_coefficients', [None] * len(basis.channels))
        self.splines = [RadialSpline(basis.r, c.radial, basis.rcut, coeff) for c, coeff in zip(basis.channels, prepared)]
        self.descriptors = basis.descriptors()

    @property
    def norb(self) -> int:
        return len(self.descriptors)

    def values(self, points: np.ndarray, center: np.ndarray, laplacian: bool = False) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        center = np.asarray(center, dtype=float)
        vec = points - center
        radius = np.linalg.norm(vec, axis=1)
        out = np.zeros((len(points), self.norb), dtype=float)
        tiny = max(1e-8, self.basis.dr * 1e-4)
        rsafe = np.maximum(radius, tiny)

        for col, desc in enumerate(self.descriptors):
            spline = self.splines[desc.channel_index]
            ylm = real_ylm_abacus(desc.l, desc.m, vec)
            if not laplacian:
                radial = spline(radius)
            else:
                r0 = spline(rsafe)
                r1 = spline(rsafe, 1)
                r2 = spline(rsafe, 2)
                radial = r2 + 2.0 * r1 / rsafe - desc.l * (desc.l + 1.0) * r0 / (rsafe**2)
                radial = np.where(radius <= self.basis.rcut, radial, 0.0)
                # The r=0 point has measure zero; use a finite limiting sample.
                radial = np.nan_to_num(radial)
            out[:, col] = radial * ylm
        return out


def projector_beta_values(rmesh: np.ndarray, radial_u: np.ndarray, cutoff: float, radii: np.ndarray) -> np.ndarray:
    """Evaluate beta_l(r) from UPF u_l(r)=r*beta_l(r)."""
    spline = RadialSpline(rmesh, radial_u, cutoff)
    radii = np.asarray(radii, dtype=float)
    u = spline(radii)
    out = np.zeros_like(radii)
    mask = radii > max(1e-10, rmesh[1] * 1e-5)
    out[mask] = u[mask] / radii[mask]
    if np.any(~mask):
        out[~mask] = spline(np.full(np.count_nonzero(~mask), max(rmesh[1] * 1e-4, 1e-8)), 1)
    return out
