"""Nonlocal (Kleinman-Bylander) blocks of the H0 prior per projector centre, on the device (FP64).

    Vnl_ij(R) = sum_k Qs_ik^T D_k Qs_jk,   Q_ik[beta, mu] = <beta_k | phi_mu(i)>,

over the projector centres k (wrapped home positions) and the orbital images (atom a, translation T) within the reach
|x_a + T.cell - x_k| <= r_beta(k) + r_c(a). Q comes from the bank's projector-overlap tables (``TwoCentre``), Qs is the
spinor extension diag(Q, Q) [2 n_beta, 2 w] and D_k the spinor KB matrix of k's species (complex, [2 n_beta, 2 n_beta]
with the up/down projector blocks). For each centre all its incidences are contracted at once,
G = (Qs^T D) Qs over the M images of the centre (one batched GEMM over a batch of centres), and the block of images
(a, Ta), (b, Tb) is added to the key (a, b, Tb - Ta).
"""
import torch

from dptb.prior.common.neighbors import translations


def spinor_d(d, nproj, mp, device):
    """D_k [species, 2 mp, 2 mp] (complex128) from the bank's spinor KB matrices [2 n, 2 n] per species."""
    out = torch.zeros((len(d), 2 * mp, 2 * mp), dtype=torch.complex128, device=device)
    for q, (x, n) in enumerate(zip(d, nproj)):
        x = x.to(device, torch.complex128)
        out[q, :n, :n] = x[:n, :n]
        out[q, :n, mp:mp + n] = x[:n, n:]
        out[q, mp:mp + n, :n] = x[n:, :n]
        out[q, mp:mp + n, mp:mp + n] = x[n:, n:]
    return out


def nonlocal_blocks(Qfn, D, spec, pos, frac, cell, pcut, rcut, index, n_keys, wmax, *, budget=2 << 30, q_chunk=65536):
    """Nonlocal blocks [n_keys, 2w, 2w] (Ry, complex128) on the rows of ``index`` and stats.

    Qfn(sp_k, sp_a, d, mp, w): projector overlaps [P, mp, w] of centres of species sp_k with orbitals of species sp_a at
    displacements d = orbital centre - projector centre; D: spinor_d of the composition; spec [n]: species row of each
    atom; pos, frac [n, 3], cell [3, 3]: FP64 device tensors (bohr, positions as given); pcut, rcut [n]: projector and
    orbital cutoffs.
    """
    dev = pos.device
    n = len(pos)
    mp = D.shape[-1] // 2
    w2 = 2 * wmax
    home = pos - torch.floor(frac) @ cell                     # projector centres (wrapped)
    T = translations(frac, float(pcut.max() + rcut.max()), cell)
    shift = T.to(torch.float64) @ cell
    parts = []
    step = max(1, int(3e7 // max(1, n * len(T) * 3)))
    for a in range(0, n, step):
        b = min(n, a + step)
        d = pos[None, :, None, :] + shift[None, None] - home[a:b, None, None, :]
        dist = torch.sqrt((d * d).sum(-1))
        kk, aa, tt = torch.nonzero(dist <= pcut[a:b, None, None] + rcut[None, :, None], as_tuple=True)
        parts.append(torch.stack([kk + a, aa, tt], 1))
        del d, dist
    inc = torch.cat(parts)                                    # sorted by projector centre k
    k_of, a_of, t_of = inc[:, 0], inc[:, 1], inc[:, 2]
    disp = (pos[a_of] + shift[t_of] - home[k_of]).contiguous()
    Q = torch.cat([Qfn(spec[k_of[s:s + q_chunk]], spec[a_of[s:s + q_chunk]], disp[s:s + q_chunk], mp, wmax)
                   for s in range(0, len(inc), q_chunk)])     # [ninc, mp, w]
    ninc = len(inc)
    Qs = torch.zeros((ninc, 2 * mp, w2), dtype=torch.complex128, device=dev)
    Qs[:, :mp, :wmax] = Q
    Qs[:, mp:, wmax:] = Q
    del Q
    counts = torch.bincount(k_of, minlength=n)
    starts = torch.cumsum(counts, 0) - counts
    M = int(counts.max())
    T_inc = T[t_of]
    Vnl = torch.zeros((n_keys, w2, w2), dtype=torch.complex128, device=dev)
    kb = max(1, min(n, int(budget // max(1, M * M * w2 ** 2 * 16 * 3))))
    stray = torch.zeros((), dtype=torch.int64, device=dev)
    ar = torch.arange(M, device=dev)
    for k0 in range(0, n, kb):
        ks = torch.arange(k0, min(n, k0 + kb), device=dev)
        nb = len(ks)
        idx = starts[ks][:, None] + ar[None]
        ok = ar[None] < counts[ks][:, None]
        idx = torch.where(ok, idx, torch.zeros_like(idx))
        Qk = Qs[idx] * ok[:, :, None, None]                   # [nb, M, 2mp, 2w]
        L = torch.matmul(Qk.transpose(-1, -2), D[spec[ks]][:, None])   # [nb, M, 2w, 2mp]
        G = torch.bmm(L.reshape(nb, M * w2, 2 * mp), Qk.permute(0, 2, 1, 3).reshape(nb, 2 * mp, M * w2))
        G = G.reshape(nb, M, w2, M, w2).permute(0, 1, 3, 2, 4)
        del Qk, L
        pair_ok = ok[:, :, None] & ok[:, None, :]
        ai = a_of[idx][:, :, None].expand(nb, M, M)
        bi = a_of[idx][:, None, :].expand(nb, M, M)
        R = T_inc[idx][:, None, :, :] - T_inc[idx][:, :, None, :]
        q = torch.stack([ai, bi, R[..., 0], R[..., 1], R[..., 2]], -1)[pair_ok]
        rows = index.find(q)
        stray = stray + (rows < 0).sum()
        torch.view_as_real(Vnl).index_add_(0, rows.clamp(min=0), torch.view_as_real(G[pair_ok].contiguous()))
        del G
    if int(stray):
        raise ValueError('nonlocal block outside the pair support')
    return Vnl, dict(incidences=int(ninc), max_per_centre=M, triplets=int((counts.double() ** 2).sum()))
