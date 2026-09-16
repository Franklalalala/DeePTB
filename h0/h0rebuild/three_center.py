from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from .constants import E2_RY_BOHR
from .models import UPFData
from .quadrature import intersection_grid
from .radial_quadrature import cumulative_simpson_rab
from .radial import OrbitalEvaluator


def neutral_atom_potential_radial(
    upf: UPFData,
    *,
    normalize_atomic_charge: bool = True,
    tail_gauge: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Return V_NA(r)=Vloc(r)+VH[rho_atom](r) in Rydberg.

    For q(r)=4*pi*r^2*rho(r),
      VH(r)=e^2[(1/r) int_0^r q(s)ds + int_r^inf q(s)/s ds].
    This neutral combination is finite-ranged up to numerical tails and is the
    natural two-/three-center tabulation quantity of Harris/Fireball/DFTB-style
    schemes. It is not used by the exact reciprocal-field path unless requested.
    """
    r = np.asarray(upf.r, dtype=float)
    q = np.asarray(upf.rhoatom_q, dtype=float).copy()
    if normalize_atomic_charge:
        raw_charge = float(cumulative_simpson_rab(q, upf.rab)[-1])
        if raw_charge <= 0.0:
            raise ValueError("PP_RHOATOM has non-positive radial integral")
        q *= float(upf.z_valence) / raw_charge
    enclosed = cumulative_simpson_rab(q, upf.rab)
    q_over_r = np.zeros_like(q)
    mask = r > 1e-14
    q_over_r[mask] = q[mask] / r[mask]
    outer_from_zero = cumulative_simpson_rab(q_over_r, upf.rab)
    outer = outer_from_zero[-1] - outer_from_zero
    vh = E2_RY_BOHR * (outer + np.divide(enclosed, r, out=np.zeros_like(r), where=mask))
    if len(r) > 1:
        vh[0] = E2_RY_BOHR * outer[0]
    vna = upf.vloc_ry + vh
    gauge = str(tail_gauge).lower()
    if gauge == "last_point_zero":
        vna = vna - float(vna[-1])
    elif gauge != "none":
        raise ValueError("tail_gauge must be 'none' or 'last_point_zero'")
    return r, vna


def sample_three_center_block(
    evaluator_a: OrbitalEvaluator,
    evaluator_b: OrbitalEvaluator,
    center_a: np.ndarray,
    center_b: np.ndarray,
    potential_center: np.ndarray,
    potential_r: np.ndarray,
    potential_values_ry: np.ndarray,
    step_bohr: float,
) -> np.ndarray:
    """Directly sample <phi_a|V_k(|r-R_k|)|phi_b> for a third center."""
    grid = intersection_grid(
        center_a,
        evaluator_a.basis.rcut,
        center_b,
        evaluator_b.basis.rcut,
        step_bohr,
    )
    out = np.zeros((evaluator_a.norb, evaluator_b.norb), dtype=float)
    if len(grid.points) == 0:
        return out
    spline = CubicSpline(potential_r, potential_values_ry, extrapolate=False)
    for start in range(0, len(grid.points), 100_000):
        pts = grid.points[start : start + 100_000]
        pa = evaluator_a.values(pts, center_a)
        pb = evaluator_b.values(pts, center_b)
        rk = np.linalg.norm(pts - potential_center, axis=1)
        vk = np.nan_to_num(spline(rk))
        out += pa.T @ (vk[:, None] * pb) * grid.weight
    return out


@dataclass
class ThreeCenterTable:
    """Horsfield-style table in (r_AK, r_BK, cos(theta_AKB))."""

    r_ak: np.ndarray
    r_bk: np.ndarray
    cos_theta: np.ndarray
    values_ry: np.ndarray  # shape (n1,n2,nc,norb_a,norb_b)

    def interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.r_ak, self.r_bk, self.cos_theta),
            self.values_ry,
            bounds_error=False,
            fill_value=0.0,
        )

    def evaluate(self, r_ak: float, r_bk: float, cos_theta: float) -> np.ndarray:
        value = self.interpolator()(np.array([[r_ak, r_bk, cos_theta]], dtype=float))[0]
        return np.asarray(value)


def build_three_center_table(
    evaluator_a: OrbitalEvaluator,
    evaluator_b: OrbitalEvaluator,
    potential_r: np.ndarray,
    potential_values_ry: np.ndarray,
    r_ak_grid: np.ndarray,
    r_bk_grid: np.ndarray,
    cos_theta_grid: np.ndarray,
    step_bohr: float,
) -> ThreeCenterTable:
    """Build a canonical-plane three-center table.

    K is at the origin, A=(r_AK,0,0), and
    B=(r_BK*cos(theta), r_BK*sin(theta),0). The stored full orbital block can
    later be rotated with Wigner/real-harmonic rotation matrices. The direct
    periodic-field production path avoids this rotation complexity; this builder
    is provided as the acceleration bridge.
    """
    r1 = np.asarray(r_ak_grid, dtype=float)
    r2 = np.asarray(r_bk_grid, dtype=float)
    ct = np.asarray(cos_theta_grid, dtype=float)
    values = np.zeros((len(r1), len(r2), len(ct), evaluator_a.norb, evaluator_b.norb))
    kcenter = np.zeros(3)
    for i, a in enumerate(r1):
        ca = np.array([a, 0.0, 0.0])
        for j, b in enumerate(r2):
            for k, c in enumerate(ct):
                c_clip = float(np.clip(c, -1.0, 1.0))
                cb = np.array([b * c_clip, b * np.sqrt(max(0.0, 1.0 - c_clip**2)), 0.0])
                values[i, j, k] = sample_three_center_block(
                    evaluator_a,
                    evaluator_b,
                    ca,
                    cb,
                    kcenter,
                    potential_r,
                    potential_values_ry,
                    step_bohr,
                )
    return ThreeCenterTable(r1, r2, ct, values)
