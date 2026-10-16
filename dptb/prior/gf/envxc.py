"""Environment XC of the GF prior (McWEDA arm, v2 stabilisation) on representative edges, on the device.

For a representative edge (i, j, R) and every density centre k other than the edge's atoms with
|x_a - x_k'| < rc_a + rq_k (a = i, j):
    N  = sum_k' G_ki^T diag(eps_k) G_kj    shell-level envelope moment   (G = envfac tables, q_k x shells_a)
    De = sum_k' F_ki^T diag(eps_k) F_kj    AO-level environment moment   (F = rhofac tables, q_k x AO_a)
and on the edge itself the pair tables Sw = <w_a|w_b>, Pw = <w_a|rho_i + rho_j|w_b> (shells) and
Dp = <phi_mu|rho_i + rho_j|phi_nu> (AO). Per shell pair (mask Sw > 1e-8): b = max(N / Sw, 0), rho_p = max(Pw / Sw, 0);
per AO element rho_t = rho_p + b and

    term = (v_t - v_p) S + (v'_t - v'_p) C_{2 kappa P}(Dp - rho_p S) + v'_t C_{2 kappa N}(De [N > 0] - b S)

with v, v' the LDA-PZ81 potential and its density derivative at rho_t, rho_p, S the edge overlap, and C_bound the
shell-block Frobenius rescaling min(1, bound / ||M_ab||). The term is zero where the shell pair is masked or either
density vanishes; reverse edges take the transpose. kappa_ab = sqrt((2 l_a + 1)(2 l_b + 1)).
"""
import torch
import triton
import triton.language as tl

from dptb.prior.common import threecentre as tc
from dptb.prior.common.neighbors import centre_queries
from dptb.prior.common.xc import lda_pz81, pz81, pz81_consts
from dptb.prior.gf.assembly import Displacements, _arrays, evaluate

OVERLAP_FLOOR = 1e-8


@triton.jit
def _pz81_kernel(rho, v_out, dv_out, consts, n, BLOCK: tl.constexpr):
    """``common.xc.pz81`` over a flat array."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v, dv = pz81(tl.load(rho + o, mask=m, other=1.0), consts)
    tl.store(v_out + o, v, mask=m)
    tl.store(dv_out + o, dv, mask=m)


def _pz81(rho):
    """LDA exchange + PZ81 correlation (unpolarised): v (eV) and dv/drho (eV bohr^3); zero for rho <= 1e-20."""
    if not (rho.is_cuda and rho.dtype == torch.float64):
        return lda_pz81(rho)[1:]
    x = rho.contiguous()
    v, dv = torch.empty_like(x), torch.empty_like(x)
    if x.numel():
        _pz81_kernel[(triton.cdiv(x.numel(), 256),)](x, v, dv, pz81_consts(x.device), x.numel(), BLOCK=256)
    return v, dv


def _rescale(M, bound, ia, ib, ns):
    """Scale every shell block of M [E, w, w] by min(1, bound / ||block||_F); ia, ib [E, w] shell of each row/column."""
    E, w, _ = M.shape
    rows = M.new_zeros((E, ns, w)).scatter_add_(1, ia[:, :, None].expand(E, w, w), M.square())
    norm = M.new_zeros((E, ns, ns)).scatter_add_(2, ib[:, None, :].expand(E, ns, w), rows).sqrt()
    over = norm > bound
    factor = torch.where(over, bound / torch.where(over, norm, torch.ones_like(norm)), torch.ones_like(norm))
    return M * _ao(factor, ia, ib)


def _ao(shell, ia, ib):
    """Shell-pair array [E, ns, ns] broadcast to AO elements [E, w, w]."""
    E, w = ia.shape
    x = shell.gather(1, ia[:, :, None].expand(E, w, shell.shape[2]))
    return x.gather(2, ib[:, None, :].expand(E, w, w))


def envxc(ctx, S):
    """McWEDA environment-XC blocks [m, w, w] (eV; zero on the nodes) and the counts of density-centre queries and
    three-centre terms; S the symmetrised overlap [m, w, w]."""
    bk, w, dev, dt = ctx.bk, ctx.w, ctx.pos.device, ctx.pos.dtype
    out = torch.zeros((len(bk), w, w), dtype=dt, device=dev)
    rep = torch.nonzero(bk.representative, as_tuple=True)[0]
    if not len(rep):
        return out, dict(queries=0, terms=0)
    nsh = max(len(ctx.sp[s].shells) for s in ctx.names)
    ns = nsh + 1                                        # one extra (empty) shell for AO padding
    # shell of every AO (padding -> the empty shell) and l of every shell, per species
    sao = torch.full((len(ctx.names), w), nsh, dtype=torch.int64, device=dev)
    lsh = torch.zeros((len(ctx.names), ns), dtype=dt, device=dev)
    for q, s in enumerate(ctx.names):
        sh = ctx.sp[s].shells
        sao[q, :ctx.sp[s].norb] = torch.repeat_interleave(torch.arange(len(sh)), torch.as_tensor([2 * l + 1 for l in sh])).to(dev)
        lsh[q, :len(sh)] = torch.as_tensor(sh, dtype=dt)
    # three-centre moments
    q = centre_queries(ctx.pos, ctx.cell, ctx.column('q_cut'), ctx.rc, strict=True, margin=-1e-12,
                       base=ctx.base)
    nb = len(bk)
    terms = q.a.new_zeros((0, 3))
    N = torch.zeros((nb, nsh, nsh), dtype=dt, device=dev)
    De = torch.zeros((nb, w, w), dtype=dt, device=dev)
    if len(q):
        nq = max(len(ctx.sp[s].arrays['eps_q']) for s in ctx.names)
        terms = tc.three_centre_terms(q, bk.index, bk.representative, endpoints=False)
        eps = tc.pad_matrices(_arrays(ctx, 'eps_q'), nq)
        spec_k = ctx.spec[q.k]
        disp = Displacements(ctx, q.k, q.a, q.vec, ctx.lmax_q)
        G = evaluate(ctx, 'envfac', disp, (nq, nsh))
        tc.contract(G, spec_k, terms, eps, N)
        del G
        F = evaluate(ctx, 'rhofac', disp, (nq, w))
        del disp
        tc.contract(F, spec_k, terms, eps, De)
        del F
    # pair tables on the representative edges
    i, j, vec = bk.keys[rep, 0], bk.keys[rep, 1], bk.vec[rep]
    disp = Displacements(ctx, i, j, vec, ctx.lmax_ao)
    Sw = evaluate(ctx, 'envnorm', disp, (nsh, nsh), transpose_pairs=True)
    Pw = evaluate(ctx, 'envpair', disp, (nsh, nsh), transpose_pairs=True)
    Dp = evaluate(ctx, 'pairmom', disp, (w, w), transpose_pairs=True)
    pad = lambda x: torch.nn.functional.pad(x, (0, 1, 0, 1))      # noqa: E731
    N, Sw, Pw = pad(N[rep]), pad(Sw), pad(Pw)
    mask = Sw > OVERLAP_FLOOR
    safe = torch.where(mask, Sw, torch.ones_like(Sw))
    b = torch.where(mask, N / safe, torch.zeros_like(N)).clamp_min(0.0)
    rp_sh = torch.where(mask, Pw / safe, torch.zeros_like(N)).clamp_min(0.0)
    ia, ib = sao[ctx.spec[i]], sao[ctx.spec[j]]
    li, lj = lsh[ctx.spec[i]], lsh[ctx.spec[j]]
    kappa = torch.sqrt((2 * li[:, :, None] + 1) * (2 * lj[:, None, :] + 1))
    # densities, masks and PZ81 are shell-pair quantities: evaluated per shell pair, then broadcast to AO elements
    rt_sh = rp_sh + b
    valid_sh = mask & (rt_sh > 0)
    validp_sh = mask & (rp_sh > 0)
    v_t, dv_t = (_ao(x, ia, ib) for x in _pz81(torch.where(valid_sh, rt_sh, torch.ones_like(rt_sh))))
    v_p, dv_p = (_ao(x, ia, ib) for x in _pz81(torch.where(validp_sh, rp_sh, torch.ones_like(rp_sh))))
    valid = _ao((valid_sh & validp_sh).to(dt), ia, ib) > 0
    bs, rp = _ao(b, ia, ib), _ao(rp_sh, ia, ib)
    Se = S[rep]
    De = De[rep] * (_ao(N, ia, ib) > 0)
    Nsh, Psh = N.clamp_min(0.0), Pw.clamp_min(0.0)
    M_env = _rescale(De - bs * Se, 2.0 * kappa * Nsh, ia, ib, ns)
    M_pair = _rescale(Dp - rp * Se, 2.0 * kappa * Psh, ia, ib, ns)
    term = (v_t - v_p) * Se + (dv_t - dv_p) * M_pair + dv_t * M_env
    out[rep] = torch.where(valid, term, torch.zeros_like(term))
    return bk.mirror(out), dict(queries=len(q), terms=len(terms))
