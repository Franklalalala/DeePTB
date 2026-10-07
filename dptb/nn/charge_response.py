"""Label-free charge/relative-potential cores. Requires only PyTorch.

Units: Angstrom, electron charge, eV. ``q`` is an effective excess-electron
coordinate (positive means more electrons), NOT a calibrated Mulliken charge. The periodic kernel is
Gaussian electrostatics with the G=0 mode removed; it is not a minimum-image
Coulomb approximation. Partial periodicity is deliberately unsupported.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

COULOMB_EV_ANGSTROM = 14.3996454784255


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
                    gamma = periodic_gamma(pg, cells[g].double(), sigma=self.sigma,
                                           g_cut=self.g_cut, max_modes=self.max_modes)
                elif not bool(pbc[g].any()):
                    gamma = isolated_gamma(pg, self.sigma)
                else:
                    raise ValueError("partial PBC needs a slab/wire kernel; 3D QEq is not a substitute")
                qg = constrained_charge(chi_q[rows].double(), j[rows].double(), gamma, 0.0)
                q = q.index_copy(0, rows, qg)
                v = v.index_copy(0, rows, gamma @ qg)
        v = v.to(chi.dtype)
        if self.qeq_local:
            v = chi + self.kappa * v
        if self.output_scale != 1.0:
            v = v * self.output_scale
        return v, {"q": q, "chi": chi_q, "hardness": j, "batch": batch}
