"""Nonlocal (Kleinman-Bylander) step of the H0 prior: the pair bank's projector overlaps on the shared three-centre
operator (``common.threecentre.projector``, the same code as the GF prior's projector step).

    V_NL,ij(R) = sum_k Q_k,i^T D_k Q_k,j,   Q_k,a[beta, mu] = <beta_k | phi_mu(a)> at d = x_a - (x_k + T.cell),

over the projector centres k at the images T with |d| <= r_beta(k) + r_c(a) (``neighbors.centre_queries``). Q comes
from the bank's projector-overlap tables (``TwoCentre``); D_k is the spinor KB matrix of k's species (Ry, spin-major
[2 n_beta, 2 n_beta]), split into its spin trace D_eff = Re (D_uu + D_dd) / 2 and its spin-orbit channels.
"""
import torch

from dptb.prior.common.neighbors import centre_queries
from dptb.prior.common.threecentre import KBMatrices, projector, spin_channels


def kb_matrices(comp, names, width):
    """``KBMatrices`` of a bank composition (species in ``names`` order); D_eff as diagonals when every one is."""
    eff, spinor = {}, {}
    for q, s in enumerate(names):
        d = comp.d[s]
        if not d.numel():
            continue
        ch = spin_channels(d)
        eff[q] = 0.5 * (ch[0].real + ch[3].real)
        spinor[q] = d
    if all(torch.equal(m, torch.diag(torch.diagonal(m))) for m in eff.values()):
        eff = {q: torch.diagonal(m).clone() for q, m in eff.items()}
    return KBMatrices(eff, spinor, width)


def nonlocal_blocks(bk, comp, kb, spec, pos, cell, pcut, rcut, base, width, q_chunk=65536):
    """real [m, w, w], spin [m, 4, w, w] (Ry) and the query and term counts (``threecentre.projector``).

    bk: ``BlockKeys``; comp: bank composition; kb: ``kb_matrices``; spec [n]: species row of each atom in the
    composition; pos [n, 3], cell [3, 3]: FP64 device tensors (bohr); pcut, rcut [n]: projector and orbital cutoffs;
    base: the ``PairBase`` of the structure.
    """
    q = centre_queries(pos, cell, pcut, rcut, strict=False, margin=0.0, include_self=True, base=base)
    spec_k, spec_a = spec[q.k], spec[q.a]
    Q = torch.cat([comp.Q(comp.Q.displacements(spec_k[s:s + q_chunk], spec_a[s:s + q_chunk], q.vec[s:s + q_chunk]),
                          kb.width, width) for s in range(0, len(q), q_chunk)]) if len(q) else \
        pos.new_zeros((0, kb.width, width))
    return projector(bk, q, Q, spec_k, kb)
