"""Two-centre integrals from L-expanded radial tables (the H0 pair-bank layout), on the device.

A table row holds, per interval p of the uniform grid r_p = p dr (p < nr), the cubic coefficients (ascending powers of
R - r_p) of S_L(R) / R^L. For orbitals mu1 = (l1, z1, m1), mu2 = (l2, z2, m2) at displacement d (|d| = R <= rmax)

    B(d) = sum_L i^(l1 - l2 - L) [S_L(R) / R^L] sum_M G(l1 m1, l2 m2, L M) R^L Y_LM(d),

with L = |l1 - l2|, |l1 - l2| + 2, ..., l1 + l2 (so the phase is real), G the real Gaunt table indexed l (l + 1) + m and
R^L Y_LM the solid harmonics (``harmonics.solid_harmonics``); zero beyond rmax. ``index`` maps
[s1, l1, z1, s2, l2, z2, L] to a table row (-1 when absent); ``left`` / ``right`` list the (l, zeta, m) of every
function of each species, which also fixes the block layout.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from dptb.prior.common.harmonics import solid_harmonics


@triton.jit(do_not_specialize=['NS', 'P', 'N2', 'NOUT', 'NI', 'NT', 'W1', 'W2'])
def _accumulate(out, sel, coeffs, iv, wv, ok, Y, row, ycol, coef, NS, P, N2, NOUT, NI, NT, W1, W2,
                BP: tl.constexpr, BC: tl.constexpr):
    """out[q, a, b] = sum_t coef[c, t] S_row[c, t](R_q) Y[ycol[c, t], q] for the pairs q = sel[...] of one species pair
    and the output columns c = a * N2 + b (term lists padded per column to NT with coefficient 0); S from the cubic
    coefficients of the row on the interval iv[q] at offset wv[q]; zero where ok[q] is false (beyond the table)."""
    i = tl.program_id(0) * BP + tl.arange(0, BP)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    im = i < NS
    cm = c < NOUT
    m = im[:, None] & cm[None, :]
    q = tl.load(sel + i, mask=im, other=0)
    k = tl.load(iv + q, mask=im, other=0).to(tl.int64)
    w = tl.load(wv + q, mask=im, other=0.0)[:, None]
    acc = tl.zeros([BP, BC], dtype=tl.float64)
    for t in range(0, NT):
        r = tl.load(row + c * NT + t, mask=cm, other=0).to(tl.int64)
        y = tl.load(ycol + c * NT + t, mask=cm, other=0).to(tl.int64)
        f = tl.load(coef + c * NT + t, mask=cm, other=0.0)
        base = coeffs + (r[None, :] * NI + k[:, None]) * 4
        c0 = tl.load(base + 0, mask=m, other=0.0)
        c1 = tl.load(base + 1, mask=m, other=0.0)
        c2 = tl.load(base + 2, mask=m, other=0.0)
        c3 = tl.load(base + 3, mask=m, other=0.0)
        yv = tl.load(Y + y[None, :] * P + q[:, None], mask=m, other=0.0)
        acc += f[None, :] * (((c3 * w + c2) * w + c1) * w + c0) * yv
    keep = tl.load(ok + q, mask=im, other=0) != 0
    a = c // N2
    b = c % N2
    tl.store(out + q[:, None].to(tl.int64) * (W1 * W2) + (a * W2 + b)[None, :], tl.where(keep[:, None], acc, 0.0),
             mask=m)


class TwoCentre:
    def __init__(self, index, coeffs, gaunt, left, right, dr, nr, rmax):
        self.index, self.coeffs, self.gaunt = index, coeffs, gaunt
        self.left, self.right = left, right
        self.dr, self.nr, self.rmax = float(dr), int(nr), float(rmax)
        self.lmax = max(l for funcs in list(left) + list(right) for l, _, _ in funcs) * 2
        self._terms = {}

    def terms(self, s1, s2):
        """Per species pair: output column, table rows used, row of each term, solid-harmonic column, coefficient."""
        key = (s1, s2)
        if key not in self._terms:
            idx = self.index[s1, :, :, s2].cpu().numpy()                     # [l1, z1, l2, z2, L]
            g = self.gaunt.cpu().numpy()
            a = np.asarray(self.left[s1], dtype=np.int64)                      # (l, zeta, m) rows
            b = np.asarray(self.right[s2], dtype=np.int64)
            l1, z1, m1 = (a[:, k][:, None, None] for k in range(3))
            l2, z2, m2 = (b[:, k][None, :, None] for k in range(3))
            nL = idx.shape[-1]
            Ls = np.arange(nL)[None, None, :]
            t = idx[l1, z1, l2, z2, Ls]                                          # [n1, n2, L]
            lo = np.abs(l1 - l2)
            ok = (t >= 0) & (Ls >= lo) & (Ls <= l1 + l2) & ((Ls - lo) % 2 == 0)
            s0 = np.where((l1 - l2 - lo) % 4 == 0, 1.0, -1.0)
            sign = s0 * np.where(((Ls - lo) // 2) % 2 == 0, 1.0, -1.0)
            out, row, ylm, coef = [], [], [], []
            n2 = len(b)
            for L in range(nL):
                for M in range(-L, L + 1):
                    G = g[l1[:, :, 0] * (l1[:, :, 0] + 1) + m1[:, :, 0], l2[:, :, 0] * (l2[:, :, 0] + 1) + m2[:, :, 0],
                          L * (L + 1) + M] if L * (L + 1) + M < g.shape[2] else np.zeros((len(a), n2))
                    hit = ok[:, :, L] & (G != 0)
                    ia, ib = np.nonzero(hit)
                    out.append(ia * n2 + ib)
                    row.append(t[ia, ib, L])
                    ylm.append(np.full(len(ia), L * L + (2 * M - 1 if M > 0 else -2 * M)))
                    coef.append(sign[ia, ib, L] * G[ia, ib])
            dev = self.coeffs.device
            out, row, ylm, coef = (np.concatenate(x) for x in (out, row, ylm, coef))
            order = np.argsort(out, kind='stable')                               # terms grouped by output column
            out, row, ylm, coef = out[order], row[order], ylm[order], coef[order]
            ptr = np.zeros(len(a) * n2 + 1, dtype=np.int64)
            ptr[1:] = np.cumsum(np.bincount(out, minlength=len(a) * n2))
            # pad every column's term list to the same length (coefficient 0 on padding)
            ncol = len(a) * n2
            nt = max(1, int(np.diff(ptr).max())) if ncol else 1
            slot = np.arange(len(out)) - ptr[out]
            R, Yc, Cf = (np.zeros((ncol, nt), dtype=x) for x in (np.int32, np.int32, np.float64))
            R[out, slot], Yc[out, slot], Cf[out, slot] = row, ylm, coef
            T = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)  # noqa: E731
            self._terms[key] = (nt, T(R, torch.int32), T(Yc, torch.int32), T(Cf, torch.float64))
        return self._terms[key]

    def __call__(self, sp1, sp2, d, width1, width2):
        """Blocks [P, width1, width2] (zero padded) for per-pair species indices sp1, sp2 and displacements d [P, 3]."""
        P = len(d)
        out = d.new_zeros((P, width1, width2))
        if not P:
            return out
        R = torch.linalg.vector_norm(d, dim=-1)
        k = torch.clamp((R / self.dr).to(torch.int64), 0, self.nr - 2)
        w = (R - k.to(R.dtype) * self.dr).contiguous()
        ok = (R <= self.rmax).to(torch.int8)
        Y = solid_harmonics(self.lmax, d).T.contiguous()                         # [(lmax + 1)^2, P]
        k = k.to(torch.int32)
        ns = max(len(self.left), len(self.right))
        code = sp1 * ns + sp2
        order = torch.argsort(code)
        counts = torch.bincount(code, minlength=ns * ns).tolist()
        start = 0
        for c, cnt in enumerate(counts):
            if not cnt:
                continue
            s1, s2 = divmod(c, ns)
            n1, n2 = len(self.left[s1]), len(self.right[s2])
            nt, row, ylm, coef = self.terms(s1, s2)
            sel = order[start:start + cnt]
            grid = (triton.cdiv(cnt, 32), triton.cdiv(n1 * n2, 32))
            _accumulate[grid](out, sel, self.coeffs, k, w, ok, Y, row, ylm, coef, cnt, P, n2, n1 * n2, self.nr - 1, nt,
                              width1, width2, BP=32, BC=32)
            start += cnt
        return out
