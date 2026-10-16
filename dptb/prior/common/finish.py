"""Finish of both priors: Hermitian spinor blocks in the DeePTB gauge and spin layout, one fused kernel on CUDA.

Inputs on the blocks of ``BlockKeys`` (ABACUS gauge, padded to the width w):
    C   [m, w, w]      scalar block in output units, Hermitian over every block pair: the real part of the uu and dd
                       blocks
    Xs  [m, 4, w, w]   spinor part in table units, channels (uu, ud, du, dd), given on the canonical rows (nodes and
                       representative edges; the other rows are not read); the real parts of uu and dd are not read
    S   [m, w, w]      overlap, Hermitian over every block pair
Output, with X = [[uu, ud], [du, dd]] spin-major [2w, 2w] and X_b = Xs_b (representative edge), Xs_rb^H (the other
edges), (Xs_b + Xs_b^H) / 2 (nodes; rb the reverse row):

    H_b[r, c] = sign_i[r] sign_j[c] Y_b[hidx_i[r], hidx_j[c]],   Y = scale X except Re Y_uu = Re Y_dd = C_b,
    S_b[r, c] = sign_i[r] sign_j[c] S_b[sidx_i[r], sidx_j[c]]

(``blocks.gauge_tables``: DeePTB rows of each species, up orbitals then down). The two blocks of a pair come out exact
Hermitian conjugates of each other. A non-finite output element fails the prior's checks (``steps.Checks``; the kernel
sets a flag as it writes, so the test costs no extra pass).
"""
import sys

import torch
import triton
import triton.language as tl

FINISH_ROWS = 8


@triton.jit(do_not_specialize=['W'])
def _finish_kernel(H, S, C, spin, S_in, rev, si, sj, hidx, hsg, sidx, ssg, W, consts, nonfinite, B2: tl.constexpr,
                   BP: tl.constexpr, B1: tl.constexpr):
    """Rows [t BP, t BP + BP) (program 1) of output block b (program 0). H and spin are interleaved real/imaginary
    FP64; the reverse block is read as the slab [columns, rows] (consecutive rows contiguous) and transposed on chip.
    nonfinite[0] is set when a written element is NaN or infinite (|x| <= the largest double fails for both)."""
    b = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)
    SC = tl.load(consts)                            # FP64 scale (a Python float argument would be FP32)
    BIG = tl.load(consts + 1)                       # FP64 largest finite double
    rb = tl.load(rev + b).to(tl.int64)
    qi = tl.load(si + b)
    qj = tl.load(sj + b)
    r = t * BP + tl.arange(0, BP)
    k = tl.arange(0, B2)
    mr = r < 2 * W
    mk = k < 2 * W
    rs = tl.load(hidx + qi * 2 * W + r, mask=mr, other=0)
    cs = tl.load(hidx + qj * 2 * W + k, mask=mk, other=0)
    sr = tl.load(hsg + qi * 2 * W + r, mask=mr, other=0.0)
    sc = tl.load(hsg + qj * 2 * W + k, mask=mk, other=0.0)
    s1 = (rs // W)[:, None]
    a = (rs % W)[:, None]
    s2 = (cs // W)[None, :]
    c = (cs % W)[None, :]
    m2 = mr[:, None] & mk[None, :]
    mt = mk[:, None] & mr[None, :]
    y = spin + (((rb * 4 + (cs // W)[:, None] * 2 + (rs // W)[None, :]) * W + (cs % W)[:, None]) * W +
                (rs % W)[None, :]) * 2
    x = spin + (((b * 4 + s1 * 2 + s2) * W + a) * W + c) * 2
    if rb == b:                                     # node: (X + X^H) / 2
        xr = tl.load(x, mask=m2, other=0.0)
        xi = tl.load(x + 1, mask=m2, other=0.0)
        yr = tl.trans(tl.load(y, mask=mt, other=0.0))
        yi = tl.trans(tl.load(y + 1, mask=mt, other=0.0))
        hr = ((xr + yr) * 0.5) * SC
        hi = ((xi - yi) * 0.5) * SC
    elif b < rb:                                    # representative edge: X
        hr = tl.load(x, mask=m2, other=0.0) * SC
        hi = tl.load(x + 1, mask=m2, other=0.0) * SC
    else:                                           # reverse edge: X_rb^H
        hr = tl.trans(tl.load(y, mask=mt, other=0.0)) * SC
        hi = -tl.trans(tl.load(y + 1, mask=mt, other=0.0)) * SC
    cv = tl.load(C + (b * W + a) * W + c, mask=m2 & (s1 == s2), other=0.0)
    hr = tl.where(s1 == s2, cv, hr)
    sg = sr[:, None] * sc[None, :]
    o = H + ((b * 2 * W + r[:, None]) * 2 * W + k[None, :]) * 2
    tl.store(o, hr * sg, mask=m2)
    tl.store(o + 1, hi * sg, mask=m2)
    bad = tl.max(tl.max(tl.where(m2 & ~((tl.abs(hr) <= BIG) & (tl.abs(hi) <= BIG)), 1, 0), axis=1), axis=0)
    r1 = t * BP + tl.arange(0, BP)
    k1 = tl.arange(0, B1)
    m1r = r1 < W
    m1k = k1 < W
    ri = tl.load(sidx + qi * W + r1, mask=m1r, other=0)
    ci = tl.load(sidx + qj * W + k1, mask=m1k, other=0)
    g = tl.load(ssg + qi * W + r1, mask=m1r, other=0.0)[:, None] * tl.load(ssg + qj * W + k1, mask=m1k,
                                                                           other=0.0)[None, :]
    mm = m1r[:, None] & m1k[None, :]
    sv = tl.load(S_in + (b * W + ri[:, None]) * W + ci[None, :], mask=mm, other=0.0)
    tl.store(S + (b * W + r1[:, None]) * W + k1[None, :], sv * g, mask=mm)
    bad = tl.maximum(bad, tl.max(tl.max(tl.where(mm & ~(tl.abs(sv) <= BIG), 1, 0), axis=1), axis=0))
    tl.atomic_max(nonfinite, bad, mask=bad > 0)


def finish(bk, spec, gauge, C, Xs, S, scale, checks):
    """H [m, 2w, 2w] complex128 and S [m, w, w] (see module). bk: ``BlockKeys``; spec [n]: output species index of every
    atom; gauge: ``blocks.gauge_tables`` of the output species; scale: table units -> output units of Xs; checks:
    ``steps.Checks`` of the prior call (non-finite output)."""
    hidx, hsg, sidx, ssg = gauge
    m, w = len(bk), C.shape[-1]
    si, sj = spec[bk.keys[:, 0]], spec[bk.keys[:, 1]]
    if C.is_cuda:
        H = torch.empty((m, 2 * w, 2 * w), dtype=torch.complex128, device=C.device)
        S_out = torch.empty_like(S)
        nonfinite = torch.zeros(1, dtype=torch.int32, device=C.device)
        if m:
            _finish_kernel[(m, triton.cdiv(2 * w, FINISH_ROWS))](
                torch.view_as_real(H), S_out, C.contiguous(), torch.view_as_real(Xs.contiguous()), S.contiguous(),
                bk.reverse.contiguous(), si.contiguous(), sj.contiguous(), hidx, hsg, sidx, ssg, w,
                torch.tensor([float(scale), sys.float_info.max], dtype=torch.float64, device=C.device), nonfinite,
                B2=triton.next_power_of_2(2 * w), BP=FINISH_ROWS, B1=triton.next_power_of_2(w), num_warps=2)
        checks.fail_if(nonfinite[0], 'non-finite prior blocks')
        return H, S_out
    X = Xs.reshape(-1, 2, 2, w, w).transpose(2, 3).reshape(-1, 2 * w, 2 * w)
    Xr = X[bk.reverse].transpose(1, 2).conj()
    node = bk.is_node[:, None, None]
    X = torch.where(node, (X + Xr) * 0.5, torch.where(bk.representative[:, None, None], X, Xr)) * scale
    X[:, :w, :w] = C + 1j * X[:, :w, :w].imag
    X[:, w:, w:] = C + 1j * X[:, w:, w:].imag
    H = X.gather(1, hidx[si][:, :, None].expand(m, 2 * w, 2 * w))
    H = H.gather(2, hidx[sj][:, None, :].expand(m, 2 * w, 2 * w)) * (hsg[si][:, :, None] * hsg[sj][:, None, :])
    S = S.gather(1, sidx[si][:, :, None].expand(m, w, w))
    S = S.gather(2, sidx[sj][:, None, :].expand(m, w, w)) * (ssg[si][:, :, None] * ssg[sj][:, None, :])
    checks.fail_if(~(torch.isfinite(torch.view_as_real(H)).all() & torch.isfinite(S).all()), 'non-finite prior blocks')
    return H, S
