"""Offline builder library for grid-free environment-XC hopping tables (schema nacf-envxc/v1).

Written by Claude Fable 5.1, 2026-09-20. NumPy/SciPy only; nothing here runs online.

Physics (LDA-PZ81, frozen atomic densities: normalized neutral valence + unscaled NLCC in
their own channels; no SCF, no PBE claim). For a directed hopping edge (i, j, R) with
rho_pair = rho_i + rho_j and rho_env = sum of every other periodic atom image,

    X_mn = <m| v_xc[rho_pair + rho_env] |n>.

Everything geometry dependent that runs online is a table lookup, a rotation or a finite-rank
contraction. The tables built here are:

  species  : a finite-rank expansion of the multiplicative density operator of species K,
             rho_K ~= sum_h |q_h> eps_h <q_h|,  q_h = rho_K p_h,  eps_h = 1/<p_h|rho_K|p_h>,
             with radial seeds p_h orthogonalized in the rho_K-weighted metric (the P23/OpenMX
             construction with the potential replaced by the density; own definition, own
             version, no VNA numbers are reused).
  rhofac   : K|A  <q_{K,h} | phi_{A,mu}(R)>          -> signed AO density moments D_mn (GSN/McWEDA)
  envfac   : K|A  <q_{K,h} | w_{A,a}(R)>,  w_a=|R_a|  -> D2 numerator <w_a|rho_env|w_b>
  envnorm  : A|B  <w_a | w_b>                          -> D2 denominator
  envpair  : A|B  <w_a | rho_A + rho_B | w_b>          -> pair-density envelope average (expansion point)
  pairmom  : A|B  <phi_mu | rho_A + rho_B | phi_nu>    -> McWEDA pair moment D^p
  xcbg     : A|B  <phi_mu | v_xc[rho_A + rho_B + b] | phi_nu> for background nodes b (D2 layers)

All two-centre integrals use one bipolar quadrature in the bond frame (right centre on +z) and
the ABACUS real-harmonic m order (0,+1,-1,+2,-2,...), exactly as the accepted pair-XC tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import lpmv, roots_legendre

ENVXC_SCHEMA = "nacf-envxc/v1"
RY_TO_EV_XC = 13.605693122994          # the physics-line constant used by every accepted XC table
TABLE_KINDS = ("rhofac", "envfac", "envnorm", "envpair", "pairmom", "xcbg")
CENTRE_KINDS = ("rhofac", "envfac")     # keyed centre|AO (ordered)
PAIR_KINDS = ("envnorm", "envpair", "pairmom", "xcbg")  # keyed sorted A|B


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- build identity
class BuildIdentityError(RuntimeError):
    """An output root, species artifact or table does not carry the identity of the requested build."""


IDENTITY_SETTINGS = ("radial_rank", "l_buffer", "tail_seeds", "density_threshold", "projector_cutoff", "grid_step",
                     "distance_step", "order", "background_nodes", "import_background_sha256")


def numerical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of build settings that changes table numbers (structure coverage and kinds do not)."""
    out: dict[str, Any] = {}
    for key in IDENTITY_SETTINGS:
        value = settings.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            value = [float(x) for x in value]
        elif isinstance(value, (np.floating, float)):
            value = float(value)
        elif isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            value = int(value)
        out[key] = value
    return out


def build_identity(settings: Mapping[str, Any], code_identity: Mapping[str, str]) -> str:
    """Immutable identity of a table build: schema + numerical settings + code SHA256s. Sources are bound per artifact."""
    return sha256_json({"schema": ENVXC_SCHEMA, "settings": numerical_settings(settings), "code_identity": dict(code_identity)})


def read_identity(path: Path | str) -> dict[str, Any]:
    """Identity record embedded in a species or table npz ({} for artifacts built before identities existed)."""
    with np.load(Path(path), allow_pickle=False) as z:
        if "identity" not in z.files:
            return {}
        return json.loads(str(z["identity"]))


def check_identity(recorded: Mapping[str, Any], *, build_id: str, sources: Mapping[str, str], label: str) -> None:
    """Fail closed unless ``recorded`` names this build identity and exactly these source SHA256s."""
    if not recorded or recorded.get("build_identity") != build_id:
        raise BuildIdentityError(f"{label}: build identity {recorded.get('build_identity') if recorded else None!r} != requested {build_id!r}; "
                                 "choose a new --output root instead of reusing this artifact")
    for symbol, digest in sources.items():
        if recorded.get("sources", {}).get(symbol) != digest:
            raise BuildIdentityError(f"{label}: source {symbol} SHA256 {recorded.get('sources', {}).get(symbol)!r} != current {digest!r}; "
                                     "choose a new --output root instead of reusing this artifact")


# --------------------------------------------------------------------------- LDA-PZ81 (eV)
def lda_pz81_v_dv(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpolarized Perdew-Zunger 1981 potential (eV) and its density derivative (eV bohr^3).

    Same branch structure and constants as the accepted reference (audit_h0.xc.lda_pz81_vxc_ry
    and the torch ``v_and_dv`` used by every XC table so far). Densities <= 1e-20 give 0 for both.
    """
    n = np.asarray(rho, dtype=np.float64)
    v = np.zeros_like(n)
    dv = np.zeros_like(n)
    mask = n > 1e-20
    if not np.any(mask):
        return v, dv
    nm = n[mask]
    rs = (3.0 / (4.0 * np.pi * nm)) ** (1.0 / 3.0)
    vx = -((3.0 / np.pi) ** (1.0 / 3.0)) * nm ** (1.0 / 3.0)
    hi = rs < 1.0
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    gamma, b1, b2 = -0.1423, 1.0529, 0.3334
    with np.errstate(divide="ignore", invalid="ignore"):
        r = rs
        e_hi = A * np.log(r) + B + C * r * np.log(r) + D * r
        ep_hi = A / r + C * (np.log(r) + 1.0) + D
        epp_hi = -A / r**2 + C / r
        den = 1.0 + b1 * np.sqrt(r) + b2 * r
        dp = b1 / (2.0 * np.sqrt(r)) + b2
        dpp = -b1 / (4.0 * r**1.5)
        e_lo = gamma / den
        ep_lo = -gamma * dp / den**2
        epp_lo = -gamma * dpp / den**2 + 2.0 * gamma * dp**2 / den**3
    e = np.where(hi, e_hi, e_lo)
    ep = np.where(hi, ep_hi, ep_lo)
    epp = np.where(hi, epp_hi, epp_lo)
    v[mask] = 2.0 * RY_TO_EV_XC * (vx + e - r * ep / 3.0)
    dv[mask] = 2.0 * RY_TO_EV_XC * (vx / (3.0 * nm) - (2.0 * ep - r * epp) * r / (9.0 * nm))
    return v, dv


def lda_pz81_v(rho: np.ndarray) -> np.ndarray:
    return lda_pz81_v_dv(rho)[0]


# --------------------------------------------------------------------------- atomic sources
@dataclass
class AtomicSource:
    """Species radial data in the exact form used by the accepted XC pipeline.

    ``r_orb``/``radial`` are the ABACUS ORB mesh and channel radial functions (l per channel,
    common cutoff ``orbital_cutoff_bohr``). ``r_rho``/``rho_val``/``rho_nlcc`` are the UPF mesh
    with the normalized neutral valence density (with the r=0 repair) and the unscaled NLCC
    core charge (zeros when the UPF has none). ``identity`` carries file SHA256 and normalization.
    """
    symbol: str
    r_orb: np.ndarray
    radial: np.ndarray            # [channels, len(r_orb)]
    shells: tuple[int, ...]       # l per channel
    orbital_cutoff_bohr: float
    r_rho: np.ndarray
    rho_val: np.ndarray
    rho_nlcc: np.ndarray
    z_valence: float
    identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.r_orb = np.asarray(self.r_orb, dtype=np.float64)
        self.radial = np.asarray(self.radial, dtype=np.float64)
        self.shells = tuple(int(l) for l in self.shells)
        self.r_rho = np.asarray(self.r_rho, dtype=np.float64)
        self.rho_val = np.asarray(self.rho_val, dtype=np.float64)
        self.rho_nlcc = np.asarray(self.rho_nlcc, dtype=np.float64)
        if self.radial.shape != (len(self.shells), self.r_orb.size):
            raise ValueError(f"{self.symbol}: radial array shape {self.radial.shape} does not match shells/mesh")
        if self.rho_val.shape != self.r_rho.shape or self.rho_nlcc.shape != self.r_rho.shape:
            raise ValueError(f"{self.symbol}: density arrays do not match the UPF mesh")
        if not (np.all(np.diff(self.r_orb) > 0) and np.all(np.diff(self.r_rho) > 0)):
            raise ValueError(f"{self.symbol}: radial meshes must be strictly increasing")
        for name in ("r_orb", "radial", "r_rho", "rho_val", "rho_nlcc"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{self.symbol}: non-finite {name}")
        self._orb_splines = [CubicSpline(self.r_orb, row, extrapolate=False) for row in self.radial]
        self._val_spline = CubicSpline(self.r_rho, self.rho_val)
        self._nlcc_spline = CubicSpline(self.r_rho, self.rho_nlcc) if np.any(self.rho_nlcc) else None

    # radial functions -----------------------------------------------------------------
    @property
    def norb(self) -> int:
        return sum(2 * l + 1 for l in self.shells)

    @property
    def nshells(self) -> int:
        return len(self.shells)

    def radial_fn(self, channel: int) -> Callable[[np.ndarray], np.ndarray]:
        spline = self._orb_splines[channel]
        cutoff = self.orbital_cutoff_bohr

        def f(r):
            r = np.asarray(r, dtype=np.float64)
            out = np.nan_to_num(spline(r), nan=0.0)
            return np.where(r >= cutoff, 0.0, out)
        return f

    def envelope_fn(self, channel: int) -> Callable[[np.ndarray], np.ndarray]:
        f = self.radial_fn(channel)
        return lambda r: np.abs(f(r))

    def density(self, r: np.ndarray) -> np.ndarray:
        """Total density: max(valence,0) + max(nlcc,0) per channel, clamped mesh, 0 beyond the mesh.

        This is the AtomicDensity contract of the accepted onsite/pair/oracle codes.
        """
        r = np.asarray(r, dtype=np.float64)
        rr = np.clip(r, self.r_rho[0], self.r_rho[-1])
        out = np.maximum(self._val_spline(rr), 0.0)
        if self._nlcc_spline is not None:
            out = out + np.maximum(self._nlcc_spline(rr), 0.0)
        return np.where(r > self.r_rho[-1], 0.0, out)

    def density_cutoff(self, threshold: float) -> float:
        """Radius beyond which the total density stays below ``threshold`` (bohr^-3)."""
        rho = self.density(self.r_rho)
        active = np.flatnonzero(rho > threshold)
        if active.size == 0:
            raise ValueError(f"{self.symbol}: density never exceeds {threshold}")
        index = min(int(active[-1]) + 1, self.r_rho.size - 1)
        return float(self.r_rho[index])

    def density_beyond(self, radius: float) -> float:
        """Electrons of the total density outside ``radius`` (diagnostic of any truncation)."""
        r = self.r_rho
        rho = self.density(r)
        weight = 4.0 * np.pi * r * r * rho
        tail = r >= radius
        if tail.sum() < 2:
            return 0.0
        return float(np.trapz(weight[tail], r[tail]))

    # serialization ----------------------------------------------------------------------
    def save(self, path: Path | str) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, symbol=np.array(self.symbol), r_orb=self.r_orb, radial=self.radial,
                 shells=np.asarray(self.shells, dtype=np.int64),
                 orbital_cutoff_bohr=np.array(self.orbital_cutoff_bohr), r_rho=self.r_rho,
                 rho_val=self.rho_val, rho_nlcc=self.rho_nlcc, z_valence=np.array(self.z_valence),
                 identity=np.array(json.dumps(self.identity, sort_keys=True)))
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path | str) -> "AtomicSource":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(symbol=str(z["symbol"]), r_orb=z["r_orb"], radial=z["radial"],
                       shells=tuple(int(x) for x in z["shells"]),
                       orbital_cutoff_bohr=float(z["orbital_cutoff_bohr"]), r_rho=z["r_rho"],
                       rho_val=z["rho_val"], rho_nlcc=z["rho_nlcc"], z_valence=float(z["z_valence"]),
                       identity=json.loads(str(z["identity"])))


# --------------------------------------------------------------------------- density projectors
@dataclass
class DensityProjectors:
    symbol: str
    r: np.ndarray                     # projector radial grid (0 .. q cutoff)
    q_radial: np.ndarray              # [n_radial, len(r)]  q_h = rho * p_h
    q_l: np.ndarray                   # l per radial projector
    q_n: np.ndarray                   # radial index within l
    epsilon_radial: np.ndarray        # 1/<p_h|rho|p_h>  (bohr^3)
    cutoff_bohr: float
    metadata: dict[str, Any]
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def shells(self) -> tuple[int, ...]:
        return tuple(int(l) for l in self.q_l)

    @property
    def norb(self) -> int:
        return int(sum(2 * l + 1 for l in self.q_l))

    def epsilon_ao(self) -> np.ndarray:
        return np.concatenate([np.full(2 * int(l) + 1, e) for l, e in zip(self.q_l, self.epsilon_radial)])

    def radial_fn(self, index: int) -> Callable[[np.ndarray], np.ndarray]:
        spline = CubicSpline(self.r, self.q_radial[index], extrapolate=False)
        cutoff = self.cutoff_bohr

        def f(r):
            r = np.asarray(r, dtype=np.float64)
            return np.where(r >= cutoff, 0.0, np.nan_to_num(spline(r), nan=0.0))
        return f


def _radial_inner(r, a, b, weight=None):
    integrand = r * r * a * b if weight is None else r * r * a * b * weight
    return float(np.trapz(integrand, r))


def build_density_projectors(source: AtomicSource, *, radial_rank: int = 2, l_buffer: int = 1,
                             tail_seeds: int = 1, density_threshold: float = 1e-7,
                             grid_step: float = 0.005, standard_norm_tol: float = 1e-18,
                             metric_tol: float = 1e-14, reorthogonalization_passes: int = 2,
                             cutoff_bohr: float | None = None) -> DensityProjectors:
    """Finite-rank density expansion rho ~= sum_h |rho p_h> eps_h <p_h rho| in the rho metric.

    Seeds per l (in order, Gram-Schmidt in <.|rho|.>): the PAO radial functions of that l; the
    first PAO radial of that l times powers of rho/rho_max; for l above the PAO l_max the highest-l
    PAO radial times r^(l - l_max); and ``tail_seeds`` functions r^l rho^(1/2) r^n that live on the
    full density support, so the auxiliary space is not cut at the PAO cutoff. The projector cutoff
    is the density cutoff (``density_threshold``) unless ``cutoff_bohr`` is given explicitly; the
    electrons dropped beyond it are reported in the metadata.
    """
    if radial_rank <= 0 or l_buffer < 0 or tail_seeds < 0:
        raise ValueError("radial_rank must be positive; l_buffer and tail_seeds non-negative")
    rho_cut = source.density_cutoff(density_threshold)
    cutoff = float(rho_cut if cutoff_bohr is None else cutoff_bohr)
    r = np.arange(0.0, cutoff + grid_step, grid_step)
    r[0] = 1e-12
    rho = source.density(r)
    rho_hat = rho / max(float(rho.max()), 1e-300)
    seed_scale = rho_hat + 1e-13
    lmax_pao = max(source.shells)
    by_l: dict[int, list[int]] = {}
    for channel, l in enumerate(source.shells):
        by_l.setdefault(l, []).append(channel)
    pao = [source.radial_fn(c)(r) for c in range(source.nshells)]
    q_rows, q_l, q_n, eps, kinds = [], [], [], [], []
    offdiag_max = 0.0
    dropped = []
    for l in range(lmax_pao + l_buffer + 1):
        seeds: list[tuple[np.ndarray, str]] = []
        channels = by_l.get(l, [])
        if l <= lmax_pao and channels:
            for index in range(radial_rank):
                if index < len(channels):
                    seeds.append((pao[channels[index]].copy(), "pao"))
                else:
                    seeds.append((pao[channels[0]] * seed_scale ** (index - len(channels) + 1), "pao0_times_rho_power"))
        else:
            highest = by_l[lmax_pao]
            for index in range(radial_rank):
                base = pao[highest[min(index, len(highest) - 1)]] * r ** (l - lmax_pao)
                power = max(0, index - len(highest) + 1)
                seeds.append((base * seed_scale ** power, "highest_l_times_r_power"))
        for n in range(tail_seeds):
            seeds.append((r ** l * np.sqrt(np.maximum(rho, 0.0)) * (r / cutoff) ** n, "tail_r_l_sqrt_rho"))
        kept: list[np.ndarray] = []
        norms: list[float] = []
        for index, (candidate, kind) in enumerate(seeds):
            norm2 = _radial_inner(r, candidate, candidate)
            if not math.isfinite(norm2) or norm2 <= standard_norm_tol:
                dropped.append({"l": l, "seed": index, "kind": kind, "reason": "null standard norm"})
                continue
            p = candidate / math.sqrt(norm2)
            for _ in range(reorthogonalization_passes):
                for previous, previous_norm in zip(kept, norms):
                    p = p - (_radial_inner(r, previous, p, rho) / previous_norm) * previous
            vnorm = _radial_inner(r, p, p, rho)
            if not math.isfinite(vnorm) or vnorm <= metric_tol * max(norms, default=1.0):
                dropped.append({"l": l, "seed": index, "kind": kind, "reason": "near-null rho metric", "metric": vnorm})
                continue
            kept.append(p)
            norms.append(vnorm)
            q_rows.append(rho * p)
            q_l.append(l)
            q_n.append(len(kept) - 1)
            eps.append(1.0 / vnorm)
            kinds.append(kind)
        if kept:
            metric = np.array([[_radial_inner(r, a, b, rho) for b in kept] for a in kept])
            scale = np.sqrt(np.abs(np.diag(metric)))
            normalized = np.abs(metric) / np.maximum(scale[:, None] * scale[None, :], 1e-300)
            normalized -= np.diag(np.diag(normalized))
            offdiag_max = max(offdiag_max, float(normalized.max(initial=0.0)))
    if not q_rows:
        raise ValueError(f"{source.symbol}: no density projector survived")
    q_radial = np.asarray(q_rows)
    q_radial[:, r >= cutoff] = 0.0
    metadata = {
        "radial_rank": int(radial_rank), "l_buffer": int(l_buffer), "tail_seeds": int(tail_seeds),
        "lmax": int(lmax_pao + l_buffer), "density_threshold_bohr3": float(density_threshold),
        "density_cutoff_bohr": float(rho_cut), "projector_cutoff_bohr": cutoff,
        "orbital_cutoff_bohr": float(source.orbital_cutoff_bohr), "grid_step_bohr": float(grid_step),
        "radial_projectors": len(eps), "angular_projectors": int(sum(2 * l + 1 for l in q_l)),
        "normalized_rho_metric_offdiag_max": offdiag_max,
        "epsilon_min_bohr3": float(np.min(eps)), "epsilon_max_bohr3": float(np.max(eps)),
        "electrons_beyond_cutoff": source.density_beyond(cutoff),
        "electrons_beyond_orbital_cutoff": source.density_beyond(source.orbital_cutoff_bohr),
        "seed_kinds": kinds, "dropped_seeds": dropped,
    }
    return DensityProjectors(symbol=source.symbol, r=r, q_radial=q_radial, q_l=np.asarray(q_l, dtype=np.int64),
                             q_n=np.asarray(q_n, dtype=np.int64), epsilon_radial=np.asarray(eps), cutoff_bohr=cutoff,
                             metadata=metadata)


# --------------------------------------------------------------------------- two-centre quadrature
def _angular(l: int, m: int, c: np.ndarray) -> np.ndarray:
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) * lpmv(m, l, c)


def shell_offsets(shells: Sequence[int]) -> np.ndarray:
    return np.cumsum([0] + [2 * int(l) + 1 for l in shells])


def two_centre_block(left: Sequence[tuple[int, Callable]], right: Sequence[tuple[int, Callable]],
                     d: float, order: int, left_cutoff: float, right_cutoff: float,
                     weight: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """<f_A | W(r_A, r_B) | g_B> for the right centre at +z distance d, all (l,m) pairs.

    ``left``/``right`` are sequences of (l, radial callable); ``weight(r_A, r_B)`` is an optional
    multiplicative field (density sum, XC potential) that depends only on the two centre distances.
    Returns the block in the bond frame with the ABACUS real-harmonic order inside each shell
    (m=0, +1, -1, +2, -2, ...). Bipolar coordinates with Gauss-Legendre order ``order`` on each
    axis; the radial axis of A is split at every point where an integrand cutoff crosses.
    """
    la = [int(l) for l, _ in left]
    lb = [int(l) for l, _ in right]
    fa = [f for _, f in left]
    fb = [g for _, g in right]
    aa, bb = float(left_cutoff), float(right_cutoff)
    offseta, offsetb = shell_offsets(la), shell_offsets(lb)
    block = np.zeros((int(offseta[-1]), int(offsetb[-1])))
    if not la or not lb or d >= aa + bb - 1e-12:
        return block
    x, w = roots_legendre(int(order))
    if d < 1e-12:
        rr = (x + 1) * min(aa, bb) / 2
        ww = w * min(aa, bb) / 2 * rr * rr
        field = np.ones_like(rr) if weight is None else weight(rr, rr)
        for i, l in enumerate(la):
            for j, ll in enumerate(lb):
                if l == ll:
                    z = float(np.sum(ww * fa[i](rr) * fb[j](rr) * field))
                    for k in range(2 * l + 1):
                        block[offseta[i] + k, offsetb[j] + k] = z
        return block
    cuts = sorted(set([0.0, aa] + [v for v in (d, bb - d, d - bb, d + bb) if 0 < v < aa]))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        r = (lo + (x + 1) * (hi - lo) / 2)[:, None]
        wr = w[:, None] * (hi - lo) / 2
        lower = np.abs(r - d)
        upper = np.minimum(bb, r + d)
        span = np.maximum(upper - lower, 0.0)
        rb = lower + (x[None, :] + 1) * span / 2
        wb = w[None, :] * span / 2
        ca = np.clip((r * r + d * d - rb * rb) / (2 * r * d), -1.0, 1.0)
        cb = np.clip((r * r - d * d - rb * rb) / (2 * rb * d), -1.0, 1.0)
        field = 1.0 if weight is None else weight(np.broadcast_to(r, rb.shape), rb)
        ww = (wr * wb * r * rb / d * field).ravel()
        fa_r = [f(np.broadcast_to(r, rb.shape)).ravel() for f in fa]
        fb_r = [g(rb).ravel() for g in fb]
        for m in range(min(max(la), max(lb)) + 1):
            ia = [i for i, l in enumerate(la) if l >= m]
            ib = [j for j, l in enumerate(lb) if l >= m]
            left_m = np.array([fa_r[i] * _angular(la[i], m, ca).ravel() for i in ia])
            right_m = np.array([fb_r[j] * _angular(lb[j], m, cb).ravel() for j in ib])
            values = (left_m * ww) @ right_m.T * (2 * math.pi)
            for u, i in enumerate(ia):
                for v_, j in enumerate(ib):
                    for k in ([0] if m == 0 else [2 * m - 1, 2 * m]):
                        block[offseta[i] + k, offsetb[j] + k] += values[u, v_]
    return block


def distance_grid(support: float, step: float) -> np.ndarray:
    """Uniform knots from 0 covering ``support`` (the P2/P23 convention)."""
    if support <= 0.0 or step <= 0.0:
        raise ValueError("support and distance step must be positive")
    end = math.ceil(float(support) / float(step)) * float(step)
    if end < support + 0.25 * step:
        end += float(step)
    count = int(round(end / float(step))) + 1
    return np.linspace(0.0, end, count, dtype=np.float64)


# --------------------------------------------------------------------------- table construction
def _channels(functions: Sequence[Callable], shells: Sequence[int]):
    return [(int(l), f) for l, f in zip(shells, functions)]


def _envelopes(source: AtomicSource):
    return [(0, source.envelope_fn(c)) for c in range(source.nshells)]


def _orbitals(source: AtomicSource):
    return [(int(l), source.radial_fn(c)) for c, l in enumerate(source.shells)]


def _projectors(proj: DensityProjectors):
    return [(int(l), proj.radial_fn(h)) for h, l in enumerate(proj.q_l)]


def _pair_density(a: AtomicSource, b: AtomicSource):
    return lambda ra, rb: a.density(ra) + b.density(rb)


def _pair_xc(a: AtomicSource, b: AtomicSource, background: float):
    return lambda ra, rb: lda_pz81_v(a.density(ra) + b.density(rb) + background)


def build_table_values(kind: str, left_source, right_source, *, distances: np.ndarray, order: int,
                       background: float | None = None) -> dict[str, Any]:
    """Return dict(distances, values, left_shells, right_shells, support_bohr) for one table.

    ``left_source`` is a DensityProjectors for centre kinds (rhofac, envfac) and an AtomicSource
    for pair kinds; ``right_source`` is always an AtomicSource. Values are eV for ``xcbg`` and
    natural units otherwise (dimensionless for envnorm; bohr^-3 for density-weighted kinds).
    """
    if kind not in TABLE_KINDS:
        raise ValueError(f"unknown table kind {kind}")
    if kind in CENTRE_KINDS:
        if not isinstance(left_source, DensityProjectors):
            raise TypeError("centre kinds need DensityProjectors on the left")
        left = _projectors(left_source)
        left_cut = left_source.cutoff_bohr
        right = _orbitals(right_source) if kind == "rhofac" else _envelopes(right_source)
        weight = None
    else:
        left_cut = left_source.orbital_cutoff_bohr
        if kind == "envnorm":
            left, right, weight = _envelopes(left_source), _envelopes(right_source), None
        elif kind == "envpair":
            left, right, weight = _envelopes(left_source), _envelopes(right_source), _pair_density(left_source, right_source)
        elif kind == "pairmom":
            left, right, weight = _orbitals(left_source), _orbitals(right_source), _pair_density(left_source, right_source)
        else:
            if background is None or background < 0:
                raise ValueError("xcbg needs a non-negative background density")
            left, right, weight = _orbitals(left_source), _orbitals(right_source), _pair_xc(left_source, right_source, float(background))
    right_cut = right_source.orbital_cutoff_bohr
    support = float(left_cut + right_cut)
    distances = np.asarray(distances, dtype=np.float64)
    if distances[-1] + 1e-12 < support:
        raise ValueError("distance grid does not cover the table support")
    values = np.stack([two_centre_block(left, right, float(d), order, left_cut, right_cut, weight) for d in distances])
    values[distances >= support - 1e-12] = 0.0
    left_shells = tuple(int(l) for l, _ in left)
    right_shells = tuple(int(l) for l, _ in right)
    symmetrized = False
    if kind in PAIR_KINDS and left_source is right_source:
        # Same species on both centres: the exact block obeys M_mn(d) = (-1)^(l_m+l_n) M_nm(d) (point
        # reflection through the bond midpoint; the weight is symmetric under the swap). The bipolar
        # quadrature is built on the left centre and breaks this at the quadrature-error level, which
        # would otherwise leak into reverse-edge blocks. Enforce the exact symmetry.
        parity = np.repeat([(-1) ** l for l in left_shells], [2 * l + 1 for l in left_shells]).astype(np.float64)
        values = 0.5 * (values + parity[None, :, None] * parity[None, None, :] * values.transpose(0, 2, 1))
        symmetrized = True
    if not np.isfinite(values).all():
        raise ValueError(f"{kind} table produced non-finite values")
    return {"distances": distances, "values": values, "left_shells": left_shells,
            "right_shells": right_shells, "support_bohr": support, "symmetrized": symmetrized}


def save_table(path: Path | str, table: Mapping[str, Any], *, identity: Mapping[str, Any] | None = None, **extra) -> str:
    """Atomically write one table npz; ``identity`` (build identity + source SHA256s) is embedded when given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"distances": np.asarray(table["distances"]), "values": np.asarray(table["values"]),
               "left_shells": np.asarray(table["left_shells"], dtype=np.int64),
               "right_shells": np.asarray(table["right_shells"], dtype=np.int64),
               "support_bohr": np.array(float(table["support_bohr"]))}
    for key, value in extra.items():
        payload[key] = np.asarray(value)
    if identity is not None:
        payload["identity"] = np.array(json.dumps(dict(identity), sort_keys=True))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)
    return sha256_file(path)


def save_species(path: Path | str, proj: DensityProjectors, *, identity: Mapping[str, Any] | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(identity) if identity is not None else dict(proj.identity)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, r=proj.r, q_radial=proj.q_radial, q_l=proj.q_l, q_n=proj.q_n,
                 epsilon_radial=proj.epsilon_radial, epsilon_ao=proj.epsilon_ao(),
                 cutoff_bohr=np.array(proj.cutoff_bohr), metadata=np.array(json.dumps(proj.metadata, sort_keys=True)),
                 identity=np.array(json.dumps(record, sort_keys=True)))
    temporary.replace(path)
    return sha256_file(path)


def load_species(path: Path | str) -> DensityProjectors:
    with np.load(Path(path), allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        identity = json.loads(str(z["identity"])) if "identity" in z.files else {}
        return DensityProjectors(symbol=str(metadata.get("symbol", "")), r=z["r"], q_radial=z["q_radial"],
                                 q_l=z["q_l"], q_n=z["q_n"], epsilon_radial=z["epsilon_radial"],
                                 cutoff_bohr=float(z["cutoff_bohr"]), metadata=metadata, identity=identity)


def table_key(kind: str, left: str, right: str, index: int | None = None) -> str:
    if kind in PAIR_KINDS:
        left, right = sorted((left, right))
    key = f"{left}|{right}"
    if kind == "xcbg":
        if index is None:
            raise ValueError("xcbg tables need a background index")
        key += f"|{int(index)}"
    return key


def table_filename(kind: str, key: str) -> str:
    return f"{kind}/{key.replace('|', '__')}.npz"


def background_nodes(rho_bar_samples: np.ndarray, n_positive: int = 16, floor: float = 1e-6,
                     low_factor: float = 0.5, high_factor: float = 1.25) -> np.ndarray:
    """Zero node + geometric positive nodes covering [low_factor*P0.1, high_factor*max]."""
    s = np.asarray(rho_bar_samples, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    lo = max(floor, float(np.percentile(s, 0.1)) * low_factor) if s.size else floor
    hi = max(lo * 10, float(s.max()) * high_factor) if s.size else lo * 1e3
    return np.concatenate([[0.0], np.geomspace(lo, hi, int(n_positive))])


__all__ = [
    "ENVXC_SCHEMA", "RY_TO_EV_XC", "TABLE_KINDS", "CENTRE_KINDS", "PAIR_KINDS", "IDENTITY_SETTINGS",
    "AtomicSource", "DensityProjectors", "build_density_projectors", "two_centre_block",
    "build_table_values", "save_table", "save_species", "load_species", "table_key", "table_filename",
    "distance_grid", "background_nodes", "lda_pz81_v_dv", "lda_pz81_v", "sha256_file", "sha256_json",
    "shell_offsets", "BuildIdentityError", "numerical_settings", "build_identity", "read_identity", "check_identity",
]
