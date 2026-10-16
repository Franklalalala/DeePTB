"""Label-free charge/relative-potential cores. Requires only PyTorch.

Units: Angstrom, electron charge, eV. ``q`` is an effective excess-electron
coordinate (positive means more electrons), NOT a calibrated Mulliken charge. The periodic kernel is
Gaussian electrostatics with the G=0 mode removed; it is not a minimum-image
Coulomb approximation. Partial periodicity is deliberately unsupported.
"""
from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from itertools import product
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

COULOMB_EV_ANGSTROM = 14.3996454784255
# Select the periodic direct solver before importing the model. Both solvers
# preserve autograd through the learned electronegativity and hardness.
QEQ_SOLVER = os.environ.get("DPTB_QEQ_SOLVER", "ewald")
if QEQ_SOLVER not in ("dense", "ewald"):
    raise ValueError("DPTB_QEQ_SOLVER must be 'dense' or 'ewald'")
EWALD_X = 5.5  # real-space cutoff r_c = 2 sigma_E EWALD_X


_PERIODIC_SCOPE = threading.local()
_NO_PERIODIC_SCOPE = object()


@contextmanager
def periodic_gamma_scope(cache):
    """Bind a call-owned kernel cache to one head thread, restoring on failure."""
    previous = getattr(_PERIODIC_SCOPE, "cache", _NO_PERIODIC_SCOPE)
    _PERIODIC_SCOPE.cache = cache
    try:
        yield
    finally:
        if previous is _NO_PERIODIC_SCOPE:
            del _PERIODIC_SCOPE.cache
        else:
            _PERIODIC_SCOPE.cache = previous


class PeriodicGammaCache:
    """Share immutable kernels only within one paired-head inference call.

    Each entry checks the complete kernel options and exact geometry values.
    The heads retain independent hardness matrices and charge solves. A CUDA
    event orders readers on another stream, and record_stream keeps storage
    alive until those readers finish. The inference owner discards this object
    after both heads return; no geometry survives into the next call.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {}

    def get(self, pos, cell, *, sigma, g_cut, max_modes, solver):
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("shared periodic kernels require inference_mode")
        if solver not in ("dense", "ewald"):
            raise ValueError("only explicit periodic matrices can be shared")
        key = (solver, sigma, g_cut, max_modes, pos.shape, pos.dtype,
               pos.device, cell.shape, cell.dtype, cell.device)
        with self._lock:
            entries = self._entries.setdefault(key, [])
            for old_pos, old_cell, gamma, ready in entries:
                if ready is not None:
                    stream = torch.cuda.current_stream(pos.device)
                    stream.wait_event(ready)
                if torch.equal(pos, old_pos) and torch.equal(cell, old_cell):
                    if ready is not None:
                        gamma.record_stream(stream)
                    return gamma
            build = ewald_gamma if solver == "ewald" else periodic_gamma
            gamma = build(pos, cell, sigma=sigma, g_cut=g_cut, max_modes=max_modes)
            # Snapshots also reject a caller mutating a geometry tensor in place.
            saved_pos, saved_cell = pos.clone(), cell.clone()
            ready = None
            if pos.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(pos.device))
            entries.append((saved_pos, saved_cell, gamma, ready))
            return gamma


def graph_index(batch: Optional[torch.Tensor], n: int, device):
    if n == 0:
        raise ValueError("response requires at least one atom")
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=device)
    batch = batch.reshape(-1).to(device=device, dtype=torch.long)
    if batch.numel() != n or bool((batch < 0).any()):
        raise ValueError("invalid response batch")
    _, inv = torch.unique(batch, sorted=True, return_inverse=True)
    # Do not assume atoms of a graph occupy contiguous rows.
    return inv


def graph_mean(x: torch.Tensor, batch: torch.Tensor):
    ng = int(batch.max()) + 1
    total = x.new_zeros((ng, x.shape[1])).index_add(0, batch, x)
    count = torch.bincount(batch, minlength=ng).to(x).unsqueeze(1)
    return total / count.clamp_min(1)


def weighted_center(v: torch.Tensor, weights: torch.Tensor, batch: torch.Tensor):
    """Remove the graph gauge in the actual onsite S metric, not atom count."""
    v = v.reshape(-1, 1)
    w = weights.reshape(-1, 1).to(v)
    if v.shape != w.shape or batch.numel() != len(v):
        raise ValueError("centering shapes disagree")
    ng = int(batch.max()) + 1
    denom = w.new_zeros((ng, 1)).index_add(0, batch, w)
    if bool((denom <= 0).any()):
        raise ValueError("each graph needs nonzero physical onsite overlap")
    mean = v.new_zeros((ng, 1)).index_add(0, batch, v * w) / denom
    return v - mean[batch]


def onsite_coordinate(t: torch.Tensor, s: torch.Tensor, mask: torch.Tensor):
    """Least-squares coefficient of v_i S_ii in compact physical AO slots.

    This uses the same stored-slot metric as training, not an invented full-AO
    Frobenius weighting. Diagnostic dftdiag fits using BOTH onsite and hopping
    data; those fits need not equal this intentionally onsite-only target.
    """
    if t.shape != s.shape or mask.shape != s.shape:
        raise ValueError("onsite coordinate layout mismatch")
    sm = s * mask.to(s)
    w = sm.square().sum(-1, keepdim=True)
    if bool((w <= 0).any()):
        raise ValueError("empty/zero physical S onsite row")
    return (t * sm).sum(-1, keepdim=True) / w, w


def isolated_gamma(pos, sigma: float = 1.2):
    """Interaction of normalized Gaussians of standard deviation sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Avoid sqrt/0 derivatives at self and coincident atoms.
    r2 = (pos[:, None, :] - pos[None, :, :]).square().sum(-1)
    r = r2.clamp_min(1e-20).sqrt()
    return COULOMB_EV_ANGSTROM * torch.erf(r / (2 * sigma)) / r


def periodic_gamma(pos, cell, *, sigma=1.2, g_cut=4.0,
                   max_modes=30000, chunk=256):
    """3-D reciprocal Gaussian kernel, PSD by construction, float64 recommended.

    All +G and -G vectors are included exactly once. Cell vectors are ROWS.
    A spherical cutoff in Cartesian reciprocal space is rotation invariant.
    The integer search box is complete even for skew cells. No G=0 term is
    added (neutral system / conventional uniform background for nonzero Q).
    This release's model adapter permits only neutral delta-charge systems.
    """
    if sigma <= 0 or g_cut <= 0 or chunk < 1 or max_modes < 1:
        raise ValueError("invalid periodic-kernel options")
    if cell.shape != (3, 3) or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("expected pos[N,3], row-vector cell[3,3]")
    volume = torch.linalg.det(cell).abs()
    if not bool(torch.isfinite(volume)) or float(volume.detach()) < 1e-8:
        raise ValueError("periodic cell must be finite and nonsingular")
    bounds = torch.ceil(g_cut * cell.detach().norm(dim=1) / (2 * math.pi)).long()
    candidates = math.prod(int(2 * b + 1) for b in bounds)
    if candidates > 8 * max_modes:
        raise ValueError("reciprocal candidate budget exceeded; use a reduced cell or explicit larger budget")
    axes = [torch.arange(-int(b), int(b) + 1, device=pos.device, dtype=pos.dtype)
            for b in bounds]
    n = torch.cartesian_prod(*axes).reshape(-1, 3)
    g = (2 * math.pi) * n @ torch.linalg.inv(cell).T
    g2 = g.square().sum(-1)
    keep = (g2 > 1e-18) & (g2 <= g_cut ** 2 * (1 + 1e-12))
    g, g2 = g[keep], g2[keep]
    if not len(g):
        raise ValueError("no reciprocal modes inside cutoff; increase g_cut")
    if len(g) > max_modes:
        raise ValueError("reciprocal mode budget exceeded")
    gamma = pos.new_zeros((len(pos), len(pos)))
    # Global translation cancels analytically in the cosine difference. Use
    # relative positions to reduce phase error for unwrapped coordinates.
    relpos = pos - pos[:1]
    for start in range(0, len(g), chunk):
        modes, sq = g[start:start + chunk], g2[start:start + chunk]
        scale = ((4 * math.pi * COULOMB_EV_ANGSTROM / volume)
                 * torch.exp(-sigma ** 2 * sq) / sq).sqrt()
        phase = relpos @ modes.T
        c, s = phase.cos() * scale, phase.sin() * scale
        gamma = gamma + c @ c.T + s @ s.T
    return (gamma + gamma.T) * 0.5


def _ewald_reciprocal(pos, cell, sigma_e, g_e, max_modes, chunk=1024):
    """sum_{0 < |G| <= g_e} K_E(G) cos(G.(r_i - r_j)), the modes of periodic_gamma(sigma_E, g_e); per chunk one GEMM of
    [cos | sin] (scaled by sqrt K_E) accumulated in place of the two products and adds of periodic_gamma."""
    volume = torch.linalg.det(cell).abs()
    bounds = torch.ceil(g_e * cell.detach().norm(dim=1) / (2 * math.pi)).long().tolist()
    if math.prod(2 * b + 1 for b in bounds) > 8 * max_modes:
        raise ValueError("reciprocal candidate budget exceeded; use a reduced cell or explicit larger budget")
    axes = [torch.arange(-b, b + 1, device=pos.device, dtype=pos.dtype) for b in bounds]
    g = (2 * math.pi) * torch.cartesian_prod(*axes).reshape(-1, 3) @ torch.linalg.inv(cell).T
    g2 = g.square().sum(-1)
    keep = (g2 > 1e-18) & (g2 <= g_e ** 2 * (1 + 1e-12))
    g, g2 = g[keep], g2[keep]
    if len(g) > max_modes:
        raise ValueError("reciprocal mode budget exceeded")
    relpos = pos - pos[:1]
    gamma = pos.new_zeros((len(pos), len(pos)))
    for start in range(0, len(g), chunk):
        sq = g2[start:start + chunk]
        scale = ((4 * math.pi * COULOMB_EV_ANGSTROM / volume) * torch.exp(-sigma_e ** 2 * sq) / sq).sqrt()
        phase = relpos @ g[start:start + chunk].T
        cs = torch.cat((phase.cos() * scale, phase.sin() * scale), 1)
        gamma = torch.addmm(gamma, cs, cs.T)
    return gamma


def _ewald_real(pos, cell, sigma, sigma_e, volume, rows=2048):
    """Real-space part of ewald_gamma: sum over images of k_e [erf(r / 2 sigma) - erf(r / 2 sigma_E)] / r (the erf form
    has the r -> 0 limit k_e (1/sigma - 1/sigma_E) / sqrt(pi) without cancellation; at large r it differs from the
    erfc form by ~1e-16 k_e / r), minus the G = 0 limit (4 pi k_e / V)(sigma_E^2 - sigma^2) of the kernel difference.
    The images are those with fractional offsets d_k or d_k - sgn(d_k) per axis (d = f_j - f_i wrapped to
    [-1/2, 1/2]): complete within r_c whenever r_c is at most the smallest lattice-plane spacing. Image distances by
    r^2 = |x|^2 + sum_k m_k (2 s_k x.a_k + a_k.a_k) + sum_{k<l} 2 m_k m_l s_k s_l a_k.a_l (x the wrapped displacement,
    s_k = -sgn(d_k), m in {0,1}^3). ``rows`` rows of the matrix at a time (temporaries O(rows N))."""
    frac = pos @ torch.linalg.inv(cell)
    metric = cell @ cell.T
    a, b = 0.5 / sigma, 0.5 / sigma_e
    blocks = []
    for start in range(0, len(pos), rows):
        d = frac[None, :, :] - frac[start:start + rows, None, :]
        d = d - torch.round(d)
        x = d @ cell
        s = torch.where(d < 0, 1.0, -1.0).to(pos.dtype)
        lin = (2 * s * (x @ cell.T) + metric.diagonal()).permute(2, 0, 1).contiguous()
        cross = {(k, l): (2 * metric[k, l]) * s[..., k] * s[..., l] for k, l in ((0, 1), (0, 2), (1, 2))}
        r2 = {(0, 0, 0): x.square().sum(-1)}
        r2[(1, 0, 0)] = r2[(0, 0, 0)] + lin[0]
        r2[(0, 1, 0)] = r2[(0, 0, 0)] + lin[1]
        r2[(0, 0, 1)] = r2[(0, 0, 0)] + lin[2]
        r2[(1, 1, 0)] = r2[(1, 0, 0)] + lin[1] + cross[(0, 1)]
        r2[(1, 0, 1)] = r2[(1, 0, 0)] + lin[2] + cross[(0, 2)]
        r2[(0, 1, 1)] = r2[(0, 1, 0)] + lin[2] + cross[(1, 2)]
        r2[(1, 1, 1)] = r2[(1, 1, 0)] + lin[2] + cross[(0, 2)] + cross[(1, 2)]
        out = torch.zeros(d.shape[:2], dtype=pos.dtype, device=pos.device)
        for m in product((0, 1), repeat=3):
            squared = r2.pop(m).clamp_min(0)
            if pos.requires_grad or cell.requires_grad:
                # sqrt has an infinite derivative at coincident sites. Evaluate
                # the analytic even series there, retaining geometry gradients.
                r = squared.clamp_min(1e-12).sqrt()
                regular = (torch.erf(r * a) - torch.erf(r * b)) / r
                series = (2 / math.sqrt(math.pi)) * (
                    (a - b) - (a ** 3 - b ** 3) * squared / 3
                    + (a ** 5 - b ** 5) * squared.square() / 10)
                out = out + torch.where(squared < 1e-12, series, regular)
            else:
                r = squared.sqrt().clamp_min(1e-300)
                out = out + (torch.erf(r * a) - torch.erf(r * b)) / r
        blocks.append(out)
    real = blocks[0] if len(blocks) == 1 else torch.cat(blocks)
    return COULOMB_EV_ANGSTROM * real - 4 * math.pi * COULOMB_EV_ANGSTROM / volume * (sigma_e ** 2 - sigma ** 2)


def ewald_gamma(pos, cell, *, sigma=1.2, g_cut=4.0, max_modes=30000):
    """The matrix of periodic_gamma by Ewald splitting of its kernel at a wider Gaussian sigma_E >= sigma:
    K(G) = K_E(G) + [K(G) - K_E(G)], K_E(G) = (4 pi k_e / V) exp(-sigma_E^2 G^2) / G^2. The K_E sum runs over
    0 < |G| <= g_cut sigma / sigma_E (the same exp(-(sigma g_cut)^2) truncation as periodic_gamma); the difference is
    summed in real space (_ewald_real; _ewald_reciprocal for the K_E sum). G = 0 is excluded and the Gaussian
    self-interaction kept, as in periodic_gamma.
    sigma_E = max(sigma, h_min / (2 EWALD_X)), h_min the smallest lattice-plane spacing, so r_c = h_min: both parts
    cost O(N^2) (8 images per pair; K_E modes independent of N at fixed cell shape). Dense, then Cholesky as before.
    """
    # Lattice-plane spacings 1 / |b_k| (b_k the columns of inv(cell)), on the host: one transfer of the cell.
    h_min = float(1.0 / torch.linalg.inv(cell.detach().cpu()).norm(dim=0).max())
    sigma_e = max(sigma, h_min / (2 * EWALD_X))
    # Both truncations drop tails of relative size erfc(sigma g_cut): the split reproduces the finite-cutoff matrix
    # only where that is negligible (sigma g_cut = 4.8 for the production heads: 6e-12).
    if sigma_e <= sigma or math.erfc(sigma * g_cut) > 1e-10:
        return periodic_gamma(pos, cell, sigma=sigma, g_cut=g_cut, max_modes=max_modes)
    gamma = _ewald_reciprocal(pos, cell, sigma_e, g_cut * sigma / sigma_e, max_modes)
    gamma = gamma + _ewald_real(pos, cell, sigma, sigma_e, torch.linalg.det(cell).abs())
    return (gamma + gamma.T) * 0.5


def solve_periodic(chi, hardness, pos, cell, *, sigma, g_cut, max_modes, solver="dense", gamma_cache=None):
    """(q, Gamma q) of one periodic graph, both [n, 1] float64, by the solver of QEQ_SOLVER."""
    if gamma_cache is None:
        gamma_cache = getattr(_PERIODIC_SCOPE, "cache", None)
    if gamma_cache is not None and not torch.is_inference_mode_enabled():
        raise RuntimeError("shared periodic kernels require inference_mode")
    if gamma_cache is not None:
        gamma = gamma_cache.get(pos, cell, sigma=sigma, g_cut=g_cut,
                                max_modes=max_modes, solver=solver)
    elif solver == "ewald":
        gamma = ewald_gamma(pos, cell, sigma=sigma, g_cut=g_cut, max_modes=max_modes)
    elif solver == "dense":
        gamma = periodic_gamma(pos, cell, sigma=sigma, g_cut=g_cut, max_modes=max_modes)
    else:
        raise ValueError(f"unknown QEq solver {solver!r}")
    q = constrained_charge(chi, hardness, gamma, 0.0)
    return q, gamma @ q


def constrained_charge(chi, hardness, gamma, total_charge=0.0):
    """SPD constrained minimizer. Autograd flows through both chi and hardness.

    q = argmin chi.q + .5 q^T(diag(J)+Gamma)q, 1.q=Q.
    Never return chi+Jq+Gamma q as the potential: that is constant at stationarity.
    """
    shape = chi.shape
    chi, hardness = chi.reshape(-1, 1), hardness.reshape(-1)
    n = len(chi)
    if gamma.shape != (n, n) or hardness.numel() != n:
        raise ValueError("charge-system dimensions disagree")
    if bool((hardness <= 0).any()) or not bool(torch.isfinite(hardness).all()):
        raise ValueError("hardness must be finite and positive")
    if not bool(torch.isfinite(chi).all()) or not bool(torch.isfinite(gamma).all()):
        raise ValueError("nonfinite charge-system input")
    a = torch.diag(hardness) + (gamma + gamma.T) * 0.5
    chol = torch.linalg.cholesky(a)
    rhs = torch.cat((chi, torch.ones_like(chi)), 1)
    sol = torch.cholesky_solve(rhs, chol)
    x, y = sol[:, :1], sol[:, 1:]
    qtotal = torch.as_tensor(total_charge, dtype=chi.dtype, device=chi.device)
    q = -x + y * (qtotal + x.sum()) / y.sum()
    # Correct only numerical constraint round-off, preserving gradients.
    q = q + (qtotal - q.sum()) / n
    return q.reshape(shape)


class ResponseNetwork(nn.Module):
    """Hidden-scalar context readout or global QEq. No labels enter forward."""
    CONTEXT_WIDTH = {"none": 1, "graph": 3}

    def __init__(self, n_scalar, n_types, *, kind="context", hidden=64,
                 element_dim=8, local_only=False, sigma=1.2, g_cut=4.0,
                 hardness_min=5.0, hardness_max=40.0, max_atoms=512,
                 max_modes=30000, detach_features=False, context="auto",
                 output_scale=1.0, qeq_local=False, dtype=torch.float32, device="cpu"):
        super().__init__()
        if kind not in {"context", "qeq"}:
            raise ValueError("response.kind must be context or qeq")
        if not isinstance(qeq_local, bool):
            raise ValueError("response.qeq_local must be bool")
        if qeq_local and kind != "qeq":
            raise ValueError("response.qeq_local applies to kind=qeq only")
        if context == "auto":
            context = "graph" if kind == "context" else "none"
        if kind == "qeq":
            if context != "none" or local_only:
                raise ValueError("QEq requires local descriptors and context=none")
        elif context != "graph" or not local_only:
            raise ValueError("The context response is the local_only control")
        if detach_features:
            raise ValueError("Detached response features require an archived model")
        if isinstance(output_scale, bool) or not 0.0 < float(output_scale) < float("inf"):
            raise ValueError("response.output_scale must be a positive finite number")
        self.context = context
        self.output_scale = float(output_scale)
        if not 0 < hardness_min < hardness_max:
            raise ValueError("require 0 < hardness_min < hardness_max")
        if n_scalar < 1 or n_types < 1 or hidden < 1 or element_dim < 0:
            raise ValueError("invalid response network dimensions")
        self.kind, self.local_only = kind, bool(local_only)
        self.sigma, self.g_cut = float(sigma), float(g_cut)
        self.jmin, self.jmax = float(hardness_min), float(hardness_max)
        self.max_atoms, self.max_modes = int(max_atoms), int(max_modes)
        self.detach_features = bool(detach_features)
        self.element = nn.Embedding(n_types, element_dim, device=device, dtype=dtype) if element_dim else None
        # RMS-normalized scalar direction plus its log amplitude: normalization
        # must not discard the scalar norm, an informative invariant.
        dim = n_scalar + 1 + element_dim
        in_dim = dim * self.CONTEXT_WIDTH[context]
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden, dtype=dtype, device=device),
                                   nn.SiLU(), nn.Linear(hidden, hidden, dtype=dtype, device=device), nn.SiLU())
        self.readout = nn.Linear(hidden, 1, dtype=dtype, device=device)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.hardness = nn.Linear(hidden, 1, dtype=dtype, device=device) if kind == "qeq" else None
        if self.hardness is not None:
            nn.init.zeros_(self.hardness.weight)
            nn.init.zeros_(self.hardness.bias)
        # qeq_local=False (as delivered): v = Gamma q, the readout is chi and J lies in
        # [hardness_min, hardness_max]. qeq_local=True is the frozen-probe form F_Q/F_SQ that won
        # the lane-B/D2S ranking: v = readout(h) + kappa * Gamma q with its own chi head,
        # chi = chi_z[type] + chi_head(h), J = hardness_min + softplus(j_z[type]) + softplus(hardness(h))
        # (hardness_max unused). All new outputs start at zero, so v = 0 at initialisation.
        self.qeq_local = bool(qeq_local)
        if self.qeq_local:
            self.chi_head = nn.Linear(hidden, 1, dtype=dtype, device=device)
            nn.init.zeros_(self.chi_head.weight)
            nn.init.zeros_(self.chi_head.bias)
            self.chi_z = nn.Parameter(torch.zeros(n_types, dtype=dtype, device=device))
            self.j_z = nn.Parameter(torch.full((n_types,), 5.0, dtype=dtype, device=device))
            self.kappa = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))

    def forward(self, scalars, atom_types, batch=None, *, pos=None, cell=None, pbc=None):
        if scalars.ndim != 2 or not bool(torch.isfinite(scalars).all()):
            raise ValueError("response needs finite [N,C] scalars")
        batch = graph_index(batch, len(scalars), scalars.device)
        x = scalars
        rms = (x.square().mean(-1, keepdim=True) + 1e-8).sqrt()
        x = torch.cat((x / rms, rms.log()), -1)
        if self.element is not None:
            x = torch.cat((x, self.element(atom_types.reshape(-1).long())), -1)
        if self.context == "graph":
            # Preserve the local control's checkpoint layout and zero channels.
            x = torch.cat([x] + [torch.zeros_like(x)] * 2, -1)
        h = self.trunk(x)
        chi = self.readout(h)
        if self.kind == "context":
            if self.output_scale != 1.0:
                chi = chi * self.output_scale
            return chi, {"batch": batch}
        if pos is None or pos.shape != (len(x), 3):
            raise ValueError("QEq requires positions in Angstrom")
        ng = int(batch.max()) + 1
        if pbc is None:
            # Nonzero cell with omitted pbc is ambiguous, not an isolated system.
            if cell is not None and bool((cell != 0).any()):
                raise ValueError("QEq requires explicit pbc when a cell is supplied")
            pbc = torch.zeros((ng, 3), dtype=torch.bool, device=x.device)
        pbc = torch.as_tensor(pbc, device=x.device, dtype=torch.bool).reshape(-1, 3)
        if pbc.shape != (ng, 3):
            raise ValueError("one pbc row required per graph")
        cells = None if cell is None else cell.reshape(-1, 3, 3)
        if cells is not None and len(cells) != ng:
            raise ValueError("one cell required per graph")
        if self.qeq_local:
            t = atom_types.reshape(-1).long()
            chi_q = self.chi_z[t].unsqueeze(-1) + self.chi_head(h)
            j = self.jmin + F.softplus(self.j_z[t]).unsqueeze(-1) + F.softplus(self.hardness(h))
        else:
            chi_q = chi
            j = self.jmin + (self.jmax - self.jmin) * self.hardness(h).sigmoid()
        q = torch.zeros_like(chi, dtype=torch.float64)
        v = torch.zeros_like(q)
        # Solvers are never run in fp16/bf16. Geometry/cell gradients are supported
        # inside a fixed reciprocal-mode set; no force/stress claim at cutoffs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            for g in range(ng):
                rows = torch.where(batch == g)[0]
                if len(rows) > self.max_atoms:
                    raise ValueError("QEq max_atoms exceeded; no silent local fallback")
                pg = pos[rows].double()
                if bool(pbc[g].all()):
                    if cells is None:
                        raise ValueError("periodic QEq requires cell")
                    solver = QEQ_SOLVER
                    qg, vg = solve_periodic(chi_q[rows].double(), j[rows].double(), pg, cells[g].double(),
                                            sigma=self.sigma, g_cut=self.g_cut, max_modes=self.max_modes,
                                            solver=solver)
                elif not bool(pbc[g].any()):
                    gamma = isolated_gamma(pg, self.sigma)
                    qg = constrained_charge(chi_q[rows].double(), j[rows].double(), gamma, 0.0)
                    vg = gamma @ qg
                else:
                    raise ValueError("partial PBC needs a slab/wire kernel; 3D QEq is not a substitute")
                q = q.index_copy(0, rows, qg)
                v = v.index_copy(0, rows, vg)
        v = v.to(chi.dtype)
        if self.qeq_local:
            v = chi + self.kappa * v
        if self.output_scale != 1.0:
            v = v * self.output_scale
        return v, {"q": q, "chi": chi_q, "hardness": j, "batch": batch}
