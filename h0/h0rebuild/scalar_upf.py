"""ABACUS nspin=1 reduction of fully relativistic norm-conserving UPFs.

Port of Pseudopot_upf::average_p for lspinorb=false from the exact production
ABACUS commit ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f, read_pp.cpp.
This is not a trace/degeneracy average of an already assembled SOC matrix.
"""
from dataclasses import replace
import numpy as np
from .models import Projector, UPFData


def scalarize_upf(upf: UPFData) -> UPFData:
    if not upf.has_so:
        return upf
    old = upf.projectors
    # ABACUS overwrites the diagonal in place, then copies the leading square.
    d = upf.dij_ry.copy()
    new = []
    sources = []
    pos = 0
    while pos < len(old):
        p = old[pos]
        nb = len(new)
        l = p.l
        if l == 0:
            u = p.radial_u.copy()
            vion = float(d[pos, pos])
            cutoff = p.cutoff_index
            source = [pos]
            pos += 1
        else:
            if pos + 1 >= len(old):
                raise ValueError('Unpaired relativistic projector in ' + str(upf.source))
            q = old[pos + 1]
            if q.l != l or p.j is None or q.j is None:
                raise ValueError('Invalid adjacent l,j projector pair')
            if abs(p.j - (l - .5)) < 1e-6 and abs(q.j - (l + .5)) < 1e-6:
                minus, plus = pos, pos + 1
            elif abs(p.j - (l + .5)) < 1e-6 and abs(q.j - (l - .5)) < 1e-6:
                plus, minus = pos, pos + 1
            else:
                raise ValueError('Invalid adjacent j=l+/-1/2 projector pair')
            dp, dm = float(d[plus, plus]), float(d[minus, minus])
            vion = ((l + 1.) * dp + l * dm) / (2. * l + 1.)
            if abs(vion) < 1e-8:
                vion = .1
            u = ((l + 1.) * np.sqrt(abs(dp / vion)) * old[plus].radial_u
                 + l * np.sqrt(abs(dm / vion)) * old[minus].radial_u) / (2. * l + 1.)
            cutoff = max(p.cutoff_index, q.cutoff_index)
            source = [minus, plus]
            pos += 2
        d[nb, nb] = vion
        new.append(Projector(nb, l, None, u, cutoff, float(upf.r[cutoff - 1])))
        sources.append(source)
    return replace(upf, projectors=new, dij_ry=d[:len(new), :len(new)].copy(), has_so=False,
                   metadata={**upf.metadata, 'scalar_reduction': {
                       'algorithm':'ABACUS Pseudopot_upf::average_p lspinorb=false',
                       'source_commit':'ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f',
                       'source_projector_slots_zero_based':sources,
                       'original_projector_count':len(old),
                       'scalar_projector_count':len(new)}})
