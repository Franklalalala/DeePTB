"""Fast band solve with spectral clipping of the predicted Hamiltonian residual (the engine behind ``Eigenvalues`` when H0 is given).

Problem.  A network predicts the residual ``R = H - H0`` of the Hamiltonian over the non-self-consistent ``H0``; the bands are the
generalized eigenvalues of ``H(k) c = e S(k) c``.  In the S-orthonormal basis the residual is ``Rt = X^H R X`` (``X^H S X = 1``) and
its spectrum ``mu`` - the generalized eigenvalues of ``(R, S)`` - does not depend on which orthonormalisation is used.  The true
residual is a weighted average of the potential change, so ``|mu|`` is a few eV; a predicted residual can have a few components
of tens to hundreds of eV (directions where S is nearly singular amplify tiny element errors by ``1/lambda``) that pull spurious
levels into the band window.  Spectral clipping replaces ``mu`` by ``clip(mu, -b, b)`` (the eigenvectors are kept) and leaves
``H0``, ``S`` and every in-bound component untouched.

Layouts (the idp of the features decides):
  * 'soc'     full spin-orbital features (4 spin blocks x real/imag), ``S`` is ``N x N`` and ``H`` is ``2N x 2N``;
  * 'scalar'  non-SOC features, everything ``N x N``.
Paths (``SpectralBandSolver.kind``):
  * 'scalar'  ``N x N`` residual clipped as it is;
  * 'uu'      SOC with a spin-diagonal residual (ud = du = 0, uu = dd up to rounding): the spectrum of the ``2N x 2N`` residual is
              that of its ``N x N`` uu block, each eigenvalue twice, so only the uu block is assembled, transformed and clipped and
              the low-rank update is added to both spin blocks;
  * 'general' any other SOC residual: the whole ``2N x 2N`` residual is clipped in the ``blockdiag(L^-1, L^-1)`` basis.

Algorithm per chunk of k points: batched Cholesky ``S = L L^H`` and ``L^-1`` (the orthonormalisation; ``1/||L^-1||_F^2`` bounds
``lambda_min(S)`` from below); ``Ht = L^-1 H0 L^-H + Rt``; a batched Cholesky screen of ``(b - margin) 1 -+ Rt`` finds the k points
with an eigenvalue outside ``(-b, b)`` (positive definiteness is equivalent to all ``mu`` inside, up to rounding, which the margin
absorbs); only those k points get the eigendecomposition of ``Rt``, and the clipped eigenpairs enter as a low-rank update; the
levels are the batched eigenvalues of ``Ht``.

Near-singular S.  The production solver removes the directions of S with ``lambda <= ill_threshold`` and pads the missing levels
with 1e4 eV.  ``Eigenvalues`` keeps that contract: a k point whose smallest S eigenvalue is below the routing limit (float32:
``max(fp32_min_lambda, ill_threshold)``, float64: ``ill_threshold``) is re-assembled from the source features in float64 and solved
in the canonical basis of the remaining directions (``canonical_levels``), clipping included.  The cheap bound from the Cholesky
inverse selects the candidates; only they get an exact smallest eigenvalue.  Without ``ill_threshold`` (``None``) directions are
removed only when even the float64 matrix is not positive definite.

Assembly: ``DenseRAssembler`` turns the per-edge / per-atom orbital-pair blocks into one dense matrix per lattice shift once per
structure; every k chunk is then one matrix product ``exp(-2 pi i k.R) @ H_R``.  It reproduces ``HR2HK`` (same blocks, same 0.5
factors and ``H + H^H``; only the summation order differs).
"""
import logging
import math
import re
import time

import torch

from dptb.nn.cusolver_batched import CusolverBatchError
from dptb.utils.constants import anglrMId

log = logging.getLogger(__name__)

PAD_EV = 1.0e4


def feature_layout(idp):
    """'soc' (full spin-orbital layout), 'scalar' (non-SOC) or None (a layout the fast path does not handle)."""
    if getattr(idp, "method", "e3tb") != "e3tb":
        return None
    if getattr(idp, "has_soc", False):
        if getattr(idp, "soc_complex_doubling", False) and not getattr(idp, "nextham_uureal_mask", False):
            return "soc"
        return None
    return "scalar"


def _shell_dims(full_basis):
    dims = [2 * anglrMId[re.findall(r"[a-zA-Z]", o)[0]] + 1 for o in full_basis]
    offs = [0]
    for d in dims:
        offs.append(offs[-1] + d)
    return dims, offs


def _local_entries(idp, mode, uu_only=False):
    """Index tables of one orbital-pair block: feature column of the real / imaginary part, row / column inside the atom block,
    spin block (row spin, column spin) and the HR2HK weight.
      mode 'soc'     every ordered shell pair, 4 spin blocks (``uu_only``: the uu block only), weight 0.5 (``H + H^H`` follows);
      mode 'overlap' shell pairs i <= j, the uu part of a SOC layout or the whole real block of a scalar layout, weight 0.5 for
                     i == j and 1 otherwise (the ``_forward_scalar`` rule, used for S and for every non-SOC matrix)."""
    soc = bool(getattr(idp, "has_soc", False))
    full_basis = idp.full_basis
    dims, offs = _shell_dims(full_basis)
    cols = {k: [] for k in ("re", "im", "r", "c", "sr", "sc", "fac")}
    for i, io in enumerate(full_basis):
        for j, jo in enumerate(full_basis):
            sl = idp.orbpair_maps.get(io + "-" + jo)
            if sl is None:
                continue
            di, dj = dims[i], dims[j]
            w = di * dj
            if mode == "soc":
                half = (sl.stop - sl.start) // 2
                if half != 4 * w:
                    raise ValueError("orbital pair %s-%s has %d features, expected the full spin-orbital layout (%d)" % (io, jo, sl.stop - sl.start, 8 * w))
                for s in range(1 if uu_only else 4):
                    for q in range(w):
                        a, b = divmod(q, dj)
                        cols["re"].append(sl.start + s * w + q)
                        cols["im"].append(sl.start + half + s * w + q)
                        cols["r"].append(offs[i] + a)
                        cols["c"].append(offs[j] + b)
                        cols["sr"].append(s // 2)
                        cols["sc"].append(s % 2)
                        cols["fac"].append(0.5)
            elif mode == "overlap":
                if i > j:
                    continue
                half = (sl.stop - sl.start) // 2
                for q in range(w):
                    a, b = divmod(q, dj)
                    cols["re"].append(sl.start + q)
                    cols["im"].append(sl.start + half + q if soc else -1)
                    cols["r"].append(offs[i] + a)
                    cols["c"].append(offs[j] + b)
                    cols["sr"].append(0)
                    cols["sc"].append(0)
                    cols["fac"].append(0.5 if i == j else 1.0)
            else:
                raise ValueError(mode)
    long = lambda x: torch.tensor(x, dtype=torch.long)
    return dict(re=long(cols["re"]), im=long(cols["im"]), r=long(cols["r"]), c=long(cols["c"]), sr=long(cols["sr"]), sc=long(cols["sc"]),
                fac=torch.tensor(cols["fac"], dtype=torch.float64))


class DenseRAssembler:
    """Dense real-space matrices of one structure, one per lattice shift, and ``H(k)`` from them with one matrix product.

    ``mode``: 'soc' (``2N x 2N``; ``uu_only``: the ``N x N`` uu block) or 'overlap' (``N x N``), see ``_local_entries``."""

    def __init__(self, idp, atom_types, edge_index, edge_shift, mode, device="cpu", uu_only=False):
        entries = _local_entries(idp, mode, uu_only)
        types = atom_types.detach().cpu().long().flatten()
        norb = idp.atom_norb.detach().cpu().long()[types]
        offset = torch.cat([torch.zeros(1, dtype=torch.long), norb.cumsum(0)[:-1]])
        N = int(norb.sum())
        D = 2 * N if (mode == "soc" and not uu_only) else N
        masks = idp.mask_to_basis.detach().cpu().bool()
        position = masks.long().cumsum(1) - 1
        shifts = torch.round(edge_shift.detach().cpu().double()).long().reshape(-1, 3)
        uniq, inv = torch.unique(torch.cat([torch.zeros(1, 3, dtype=torch.long), shifts]), dim=0, return_inverse=True)
        zero_shift, edge_shift_id = int(inv[0]), inv[1:]
        u, v = edge_index.detach().cpu().long()
        parts = {"dest": [], "obj": [], "re": [], "im": [], "fac": []}
        counts = {"node": 0}

        def add(objs, rows_off, cols_off, rid, valid, mrow, mcol, is_edge):
            row = rows_off[:, None] + mrow[entries["r"][valid]][None, :] + entries["sr"][valid][None, :] * N
            col = cols_off[:, None] + mcol[entries["c"][valid]][None, :] + entries["sc"][valid][None, :] * N
            parts["dest"].append((rid[:, None] * D * D + row * D + col).flatten())
            parts["obj"].append(objs[:, None].expand_as(row).flatten())
            for name in ("re", "im"):
                parts[name].append(entries[name][valid][None, :].expand_as(row).flatten())
            parts["fac"].append(entries["fac"][valid][None, :].expand_as(row).flatten())
            if not is_edge:
                counts["node"] += row.numel()

        for t in types.unique().tolist():
            atoms = (types == t).nonzero().flatten()
            valid = masks[t][entries["r"]] & masks[t][entries["c"]]
            add(atoms, offset[atoms], offset[atoms], torch.full((len(atoms),), zero_shift, dtype=torch.long), valid, position[t], position[t], False)
        tu, tv = types[u], types[v]
        for a, b in torch.unique(torch.stack([tu, tv], 1), dim=0).tolist():
            es = ((tu == a) & (tv == b)).nonzero().flatten()
            valid = masks[a][entries["r"]] & masks[b][entries["c"]]
            add(es, offset[u[es]], offset[v[es]], edge_shift_id[es], valid, position[a], position[b], True)
        cat = {k: torch.cat(x) for k, x in parts.items()}
        self.N, self.D, self.mode, self.device = N, D, mode, torch.device(device)
        self.R = uniq.double().to(self.device)
        self.n_node = counts["node"]  # entries [:n_node] come from atoms (onsite), the rest from edges
        self.dest, self.obj, self.re = cat["dest"].to(self.device), cat["obj"].to(self.device), cat["re"].to(self.device)
        self.has_im = bool((cat["im"] >= 0).any())
        self.im = cat["im"].clamp(min=0).to(self.device)
        self.fac = cat["fac"].to(self.device)

    def real_space(self, Fn, Fe, dtype=torch.float64):
        """``[n_R, D, D]`` complex matrices of every lattice shift from the node and edge features."""
        cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        Fn, Fe = Fn.to(self.device, dtype), Fe.to(self.device, dtype)
        fac = self.fac.to(dtype)
        n = self.n_node

        def gather(column):
            out = torch.empty(self.dest.numel(), dtype=dtype, device=self.device)
            out[:n] = Fn[self.obj[:n], column[:n]]
            out[n:] = Fe[self.obj[n:], column[n:]]
            return out * fac

        size = self.R.shape[0] * self.D * self.D
        if self.has_im:
            HR = torch.zeros(size, dtype=cdt, device=self.device)
            HR.index_add_(0, self.dest, torch.complex(gather(self.re), gather(self.im)))
        else:
            HR = torch.zeros(size, dtype=dtype, device=self.device)
            HR.index_add_(0, self.dest, gather(self.re))
            HR = HR.to(cdt)
        return HR.view(self.R.shape[0], self.D, self.D)

    def hk(self, HR, kpts):
        """``H(k) = sum_R exp(-2 pi i k.R) H_R`` followed by ``H + H^H`` (as HR2HK)."""
        kpts = kpts.to(self.device, torch.float64)
        phase = torch.exp(-2j * math.pi * (kpts @ self.R.T)).to(HR.dtype)
        H = (phase @ HR.reshape(HR.shape[0], -1)).view(kpts.shape[0], self.D, self.D)
        return H + H.mH


def _absmax(x):
    return max(float(x.amax()), -float(x.amin())) if x.numel() else 0.0


def spin_pairs(idp):
    """``(first feature column, w)`` of every orbital-pair slice of a full spin-orbital layout."""
    dims, _ = _shell_dims(idp.full_basis)
    out = []
    for i, io in enumerate(idp.full_basis):
        for j, jo in enumerate(idp.full_basis):
            sl = idp.orbpair_maps.get(io + "-" + jo)
            if sl is not None:
                out.append((sl.start, dims[i] * dims[j]))
    return out


def detect_kind(idp, residual, scale, eps):
    """Which path a residual ``(node, edge)`` takes: ``('scalar' | 'uu' | 'general', info)``.

    'uu' needs ud = du = 0 and uu = dd (real and imaginary parts) within ``tol = 4 eps scale``; ``scale`` is the largest |H| / |H0|
    element and ``eps`` the machine epsilon of the feature dtype: ``H - H0`` of features that carry the spin-diagonal residual can
    differ between the two spin blocks by that rounding only."""
    if feature_layout(idp) == "scalar":
        return "scalar", {}
    off = uudd = None
    for F in residual:
        if F.shape[0] == 0:
            continue
        for start, w in spin_pairs(idp):
            blk = F[:, start:start + 8 * w].reshape(F.shape[0], 2, 4, w)
            o, d = blk[:, :, 1:3].abs().amax(), (blk[:, :, 0] - blk[:, :, 3]).abs().amax()
            off = o if off is None else torch.maximum(off, o)
            uudd = d if uudd is None else torch.maximum(uudd, d)
    off, uudd = (0.0 if off is None else float(off)), (0.0 if uudd is None else float(uudd))
    tol = 4.0 * eps * scale
    return ("uu" if (off <= tol and uudd <= tol) else "general"), dict(spin_offdiag_max=off, spin_uu_dd_max=uudd, spin_tol=tol)


def _transform_batched(Li, A):
    """``Li A Li^H`` on an ``N x N`` batch, or on the four spin blocks of a ``2N x 2N`` batch."""
    N = Li.shape[-1]
    Lih = Li.mH
    if A.shape[-1] == N:
        return Li @ A @ Lih
    out = torch.empty_like(A)
    for a in (0, 1):
        for c in (0, 1):
            out[:, a * N:(a + 1) * N, c * N:(c + 1) * N] = Li @ A[:, a * N:(a + 1) * N, c * N:(c + 1) * N] @ Lih
    return out


def _transform_basis(X, A, N):
    """``X^H A X`` for ``X [N, M]`` on an ``N x N`` matrix, or on the four spin blocks of a ``2N x 2N`` matrix."""
    Xh = X.mH
    if A.shape[-1] == N:
        return Xh @ A @ X
    M = X.shape[1]
    out = torch.empty((2 * M, 2 * M), dtype=A.dtype, device=A.device)
    for a in (0, 1):
        for c in (0, 1):
            out[a * M:(a + 1) * M, c * M:(c + 1) * M] = Xh @ A[a * N:(a + 1) * N, c * N:(c + 1) * N] @ X
    return out


def _add_update(Ht, Dl, kind, idx=None):
    """Add the low-rank update ``Dl`` of the clipped residual to ``Ht`` (rows ``idx`` of the batch, or all)."""
    N = Dl.shape[-1]
    if kind == "uu":
        if idx is None:
            Ht[:, :N, :N] += Dl
            Ht[:, N:, N:] += Dl
        else:
            Ht[idx, :N, :N] += Dl
            Ht[idx, N:, N:] += Dl
    elif idx is None:
        Ht += Dl
    else:
        Ht[idx] += Dl


def canonical_levels(S, H0, R, b, drop_threshold, kind):
    """Production semantics for a few k points in float64 (``S [B,N,N]``, ``H0 [B,D,D]``, ``R [B,Dr,Dr]``, complex128).

    The directions of ``S`` with ``lambda <= drop_threshold`` are removed, the canonical basis ``U lambda^-1/2`` of the others is
    used, the residual is clipped there (``b=None``: not clipped) and the levels are padded with 1e4 eV to ``D``.
    Returns ``(levels [B, D] float64, kept [B] (N x N directions kept), n_clipped [B])``."""
    B, N = S.shape[0], S.shape[-1]
    D = H0.shape[-1]
    lam, U = torch.linalg.eigh(0.5 * (S + S.mH))
    levels = torch.full((B, D), PAD_EV, dtype=torch.float64, device=S.device)
    kept = torch.zeros(B, dtype=torch.long)
    n_clipped = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        keep = lam[i] > drop_threshold
        M = int(keep.sum())
        kept[i] = M
        if M == 0:
            continue
        X = U[i][:, keep] * lam[i][keep].rsqrt().to(U.dtype)[None, :]  # X^H S X = 1
        Rt = _transform_basis(X, R[i], N)
        Ht = _transform_basis(X, H0[i], N)
        if kind == "uu":
            Ht[:M, :M] += Rt
            Ht[M:, M:] += Rt
        else:
            Ht += Rt
        Ht = 0.5 * (Ht + Ht.mH)
        if b is not None:
            mu, W = torch.linalg.eigh(0.5 * (Rt + Rt.mH))
            d = mu.clamp(-b, b) - mu
            n_clipped[i] = int((d != 0).sum())
            if int(n_clipped[i]):
                _add_update(Ht[None], ((W * d.to(W.dtype)[None, :]) @ W.mH)[None], kind)
        levels[i, :Ht.shape[-1]] = torch.linalg.eigvalsh(Ht)
    return levels, kept, n_clipped


class SpectralBandSolver:
    """Spectrally clipped bands of one structure and one Hamiltonian.

    ``S``, ``H0``, ``R`` are ``(node, edge)`` feature pairs in the layout of ``idp`` (``R = H - H0``).  ``kind`` is 'scalar', 'uu' or
    'general' (see the module docstring; ``detect_kind`` chooses it for a residual).  ``precision`` 'fp32' (complex64) or 'fp64'.
    ``batched`` is a ``BatchedEigh`` (CUDA) or None for ``torch.linalg``."""

    def __init__(self, idp, atom_types, edge_index, edge_shift, S, H0, R, device, precision="fp32", kind="uu", batched=None):
        layout = feature_layout(idp)
        if layout is None:
            raise ValueError("spectral clipping does not handle this feature layout")
        if (layout == "scalar") != (kind == "scalar"):
            raise ValueError("kind %r does not fit the %r feature layout" % (kind, layout))
        t0 = time.perf_counter()
        self.device = torch.device(device)
        self.kind, self.fp64, self.be = kind, precision == "fp64", batched
        self.rdt = torch.float64 if self.fp64 else torch.float32
        self.cdt = torch.complex128 if self.fp64 else torch.complex64
        asm = lambda mode, uu_only=False: DenseRAssembler(idp, atom_types, edge_index, edge_shift, mode, self.device, uu_only)
        self.pS = asm("overlap")
        self.N = self.pS.N
        if kind == "scalar":
            self.pH = self.pU = self.pS
        else:
            self.pH = asm("soc")
            self.pU = asm("soc", uu_only=True) if kind == "uu" else self.pH
        self.D = self.pH.D
        self.Dr = self.pU.D
        self._src = {"S": S, "H0": H0, "R": R}
        self._real = {}
        self.t_assemblers = time.perf_counter() - t0
        self.stats = {}

    # ---- real-space matrices (main precision on first use; float64 only when a k point needs it)
    def _R(self, key, fp64=None):
        dtype = torch.float64 if (self.fp64 if fp64 is None else fp64) else torch.float32
        if (key, dtype) not in self._real:
            asm = {"S": self.pS, "H0": self.pH, "R": self.pU}[key]
            self._real[(key, dtype)] = asm.real_space(*self._src[key], dtype)
        return self._real[(key, dtype)]

    def auto_kchunk(self, cap=128, floor=8, frac=0.5):
        """k points per chunk: at most ``cap``, from half of the free device memory (CUDA); small and memory-bounded on CPU."""
        per_k = (12 if self.kind == "general" else 6) * self.D * self.D * torch.empty(0, dtype=self.cdt).element_size()
        if self.device.type != "cuda":
            return int(max(1, min(32, 2e9 // per_k)))
        free, _ = torch.cuda.mem_get_info(self.device)
        return int(max(floor, min(cap, frac * free // per_k)))

    # ---- eigensolvers: the cuSOLVER input is a copy, so a failed call falls back on the original matrix
    def _count_fallback(self, error):
        self.stats["cusolver_fallback"] = self.stats.get("cusolver_fallback", 0) + int(error.bad.numel())

    def _eigh(self, A):
        if self.be is not None:
            try:
                return self.be.eigh(A.contiguous().clone(), vectors=True)
            except CusolverBatchError as error:
                self._count_fallback(error)
        return torch.linalg.eigh(A)

    def _eigvalsh(self, A):
        if self.be is not None:
            try:
                return self.be.eigvalsh(A.contiguous().clone())
            except CusolverBatchError as error:
                self._count_fallback(error)
        return torch.linalg.eigvalsh(A)

    # ---- clipping of a Hermitian batch Rh [B, Dr, Dr]: (update [B, Dr, Dr] or None, clipped eigenvalues, k points with a clipped one)
    def _clip_update(self, Rh, b):
        mu, W = self._eigh(Rh)
        d = mu.clamp(-b, b) - mu
        nz = d != 0
        m = int(nz.sum(dim=1).max())
        if m == 0:
            return None, 0, 0
        order = torch.argsort(d.abs(), dim=1, descending=True)[:, :m]
        Wm = torch.gather(W, 2, order[:, None, :].expand(-1, Rh.shape[-1], -1))
        Dl = (Wm * torch.gather(d, 1, order).to(Rh.dtype)[:, None, :]) @ Wm.mH
        return Dl, int(nz.sum()), int(nz.any(dim=1).sum())

    def _truth_mu(self, kp):
        """Eigenvalues of the residual in the Cholesky basis, float64 from the source features (diagnostics only)."""
        S = self.pS.hk(self._R("S", True), kp)
        S = 0.5 * (S + S.mH)
        L, info = torch.linalg.cholesky_ex(S)
        Li = torch.linalg.solve_triangular(L, torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand(L.shape).clone(), upper=False)
        Rt = _transform_batched(Li, self.pU.hk(self._R("R", True), kp))
        mu = self._eigvalsh(0.5 * (Rt + Rt.mH))
        mu[info != 0] = float("nan")
        return mu

    # ---- fast path on a batch of k points whose S is well conditioned (Li = L^-1 of the Cholesky factor)
    def _solve_fast(self, Li, kp, b, margin, counters, kglobal):
        N, kind = self.N, self.kind
        Rt = _transform_batched(Li, self.pU.hk(self._R("R"), kp))
        Ht = _transform_batched(Li, self.pH.hk(self._R("H0"), kp))
        if kind == "uu":
            Ht[:, :N, :N] += Rt
            Ht[:, N:, N:] += Rt
        else:
            Ht += Rt
        Ht = 0.5 * (Ht + Ht.mH)
        if b is None:
            return self._eigvalsh(Ht)
        Bk = Ht.shape[0]
        Rh = 0.5 * (Rt + Rt.mH)
        del Rt
        eye = torch.eye(Rh.shape[-1], dtype=self.cdt, device=self.device)
        _, i_plus = torch.linalg.cholesky_ex((b - margin) * eye - Rh)
        _, i_minus = torch.linalg.cholesky_ex((b - margin) * eye + Rh)
        flag = (i_plus != 0) | (i_minus != 0)
        idx = flag.nonzero().flatten()
        nf = int(idx.numel())
        counters["k_flagged"] += nf
        if nf:
            Dl, n_eig, n_k = self._clip_update(Rh if nf == Bk else Rh[idx], b)
            counters["n_clipped"] += n_eig
            if Dl is not None:
                counters["k_clipped"] += n_k
                _add_update(Ht, Dl, kind, None if nf == Bk else idx)
        if kglobal is not None:  # diagnostics: the screen against the exact float64 spectrum of the residual
            mu = self._truth_mu(kp)
            counters.setdefault("diag", []).append(dict(k=kglobal.cpu(), flag=flag.cpu(), mu_min=mu[:, 0].cpu(), mu_max=mu[:, -1].cpu()))
        return self._eigvalsh(Ht)

    # ---- float64 production rule on the near-singular k points
    def _solve_canonical(self, kp, b, drop_threshold):
        S = self.pS.hk(self._R("S", True), kp)
        H0 = self.pH.hk(self._R("H0", True), kp)
        R = self.pU.hk(self._R("R", True), kp)
        return canonical_levels(S, H0, R, b, drop_threshold, self.kind)

    def _solve_chunk(self, kp, b, ill_threshold, fp32_min_lambda, margin, kglobal):
        N, dev = self.N, self.device
        Bk = kp.shape[0]
        counters = dict(k_flagged=0, k_clipped=0, n_clipped=0, k_ill_candidates=0, k_fp64=0, k_dropped=0, n_dropped=0, n_effective_min=self.D)
        S = self.pS.hk(self._R("S"), kp)
        S = 0.5 * (S + S.mH)
        L, info = torch.linalg.cholesky_ex(S)
        Li = torch.linalg.solve_triangular(L, torch.eye(N, dtype=self.cdt, device=dev).expand(L.shape).clone(), upper=False)
        est = 1.0 / (Li.abs() ** 2).sum(dim=(-2, -1))  # <= lambda_min(S)
        # routing limit: float32 needs a floor for accuracy and must not undercut the removal threshold; float64 only needs the threshold
        if self.fp64:
            limit = ill_threshold
            candidate = (info != 0) | ~torch.isfinite(est)
            if limit is not None:
                candidate |= est <= limit
        else:
            limit = max(fp32_min_lambda, ill_threshold or 0.0)
            candidate = (info != 0) | ~torch.isfinite(est) | (est < limit)
        risky = torch.zeros(Bk, dtype=torch.bool, device=dev)
        if bool(candidate.any()):
            ci = candidate.nonzero().flatten()
            lam = self._eigvalsh(S[ci])[:, 0]  # exact smallest eigenvalue of S, candidates only
            bad = (info[ci] != 0) | ~torch.isfinite(lam)
            if limit is not None:
                bad |= (lam <= limit) if self.fp64 else (lam < limit + 2e-6)
            risky[ci] = bad
            counters["k_ill_candidates"] = int(ci.numel())
        n_risky = int(risky.sum())
        E = torch.full((Bk, self.D), PAD_EV, dtype=self.rdt, device=dev)
        if n_risky < Bk:
            rows = None if n_risky == 0 else (~risky).nonzero().flatten()  # near-singular rows never enter the fast solvers
            E_fast = self._solve_fast(Li if rows is None else Li[rows], kp if rows is None else kp[rows], b, margin, counters,
                                      None if kglobal is None else (kglobal if rows is None else kglobal[rows.cpu()]))
            if rows is None:
                E = E_fast
            else:
                E[rows] = E_fast
        if n_risky:
            ri = risky.nonzero().flatten()
            levels, kept, n_clip = self._solve_canonical(kp[ri], b, 0.0 if ill_threshold is None else ill_threshold)
            E[ri] = levels.to(self.rdt)
            scale = 1 if self.kind == "scalar" else 2
            counters["k_fp64"] = n_risky
            counters["k_dropped"] = int((kept < N).sum())
            counters["n_dropped"] = int((scale * (N - kept)).sum())
            counters["n_effective_min"] = min(counters["n_effective_min"], int(scale * kept.min()))
            counters["n_clipped"] += int(n_clip.sum())
            counters["k_clipped"] += int((n_clip > 0).sum())
        bad_rows = ~torch.isfinite(E).all(dim=1)
        if bool(bad_rows.any()):
            raise FloatingPointError("non-finite eigenvalues at k points %s of the chunk (non-finite input or a failed eigensolve)" % bad_rows.nonzero().flatten().tolist()[:10])
        return E, counters

    def solve(self, kpoints, b=10.0, ill_threshold=1e-5, fp32_min_lambda=1e-4, kchunk="auto", margin=0.01, diagnose=False, max_chunk=None):
        """Levels ``[nk, D]`` (ascending, ``self.rdt``) at ``kpoints`` (fractional, ``[nk, 3]``).  ``b=None``: no clipping.

        The statistics of the call are in ``self.stats`` (also while the call is running, so they survive an exception)."""
        if b is not None and not b > 0:
            raise ValueError("clip bound b must be positive, got %r" % (b,))
        margin = min(margin, 0.5 * b) if b is not None else margin
        kpts = torch.as_tensor(kpoints).detach().to(self.device, torch.float64).reshape(-1, 3)
        nk = int(kpts.shape[0])
        t0 = time.perf_counter()
        for key in ("S", "H0", "R"):
            self._R(key)
        chunk = self.auto_kchunk() if kchunk in (None, "auto") else int(kchunk)
        if max_chunk:
            chunk = min(chunk, int(max_chunk))
        chunk = max(1, chunk)
        st = self.stats = dict(path=self.kind, backend="cusolver" if self.be is not None else "torch", precision="fp64" if self.fp64 else "fp32",
                               k_total=nk, kchunk=chunk, k_flagged=0, k_clipped=0, n_clipped=0, k_ill_candidates=0, k_fp64=0, k_dropped=0,
                               n_dropped=0, n_effective_min=self.D, cusolver_fallback=0, oom_retries=0, clip_b=b, ill_threshold=ill_threshold,
                               t_prepare_s=self.t_assemblers + time.perf_counter() - t0)
        diag = []
        out = torch.empty((nk, self.D), dtype=self.rdt, device=self.device)
        t1 = time.perf_counter()
        c0 = 0
        while c0 < nk:
            kp = kpts[c0:c0 + chunk]
            if self.be is not None:
                self.be.Bmax_hint = chunk
            try:
                E, counters = self._solve_chunk(kp, b, ill_threshold, fp32_min_lambda, margin, torch.arange(c0, c0 + kp.shape[0]) if diagnose else None)
            except torch.cuda.OutOfMemoryError:
                if chunk <= 1:
                    raise
                chunk = max(1, chunk // 2)
                st["oom_retries"] += 1
                st["kchunk"] = chunk
                if self.be is not None:
                    self.be.release()
                torch.cuda.empty_cache()
                log.warning("out of GPU memory in the clipped band solve; retrying with %d k points per chunk", chunk)
                continue
            out[c0:c0 + kp.shape[0]] = E
            diag += counters.pop("diag", [])
            for key, value in counters.items():
                st[key] = min(st[key], value) if key == "n_effective_min" else st[key] + value
            c0 += kp.shape[0]
        st["t_solve_s"] = time.perf_counter() - t1
        if diagnose:
            cat = lambda key: torch.cat([d[key] for d in diag]).numpy() if diag else None
            st["diag"] = dict(k=cat("k"), flag=cat("flag"), mu_min=cat("mu_min"), mu_max=cat("mu_max"))
        return out
