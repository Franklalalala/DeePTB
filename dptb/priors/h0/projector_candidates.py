"""Conservative projector support anchored at one AO centre, without pruning."""
import numpy as np
def anchor_candidates(index, center, orbital_cutoff, maximum_projector_cutoff):
    return tuple((hit.atom_index, hit.center_bohr) for hit in
        index.query(center, orbital_cutoff + maximum_projector_cutoff))


class PreparedProjectorCandidates:
    """Private ordered snapshot for one source AO; no rounded geometry keys."""

    def __init__(self, candidates, structure, two_center, ci, rcut_i):
        self.center = np.asarray(ci,dtype=np.float64).copy()
        self.rcut_i = float(rcut_i)
        self.owner = getattr(two_center,'_base',two_center)
        self.structure = structure
        active=[]
        for atom,pcenter in candidates:
            p=np.asarray(pcenter,dtype=np.float64)
            if p.shape!=(3,) or not np.isfinite(p).all():
                raise ValueError('Projector centres must be finite three-vectors')
            symbol=structure.atoms[atom].species
            cutoff=two_center.sd[symbol].upf.max_projector_cutoff
            # Exactly the original left cutoff, performed once per anchor.
            if np.linalg.norm(self.center-p)>cutoff+self.rcut_i:
                continue
            active.append((two_center._check_species(symbol),p.copy(),cutoff))
        self.centers=np.asarray([p for _,p,_ in active],dtype=np.float64).reshape(-1,3)
        self.cutoffs=np.asarray([c for _,_,c in active],dtype=np.float64)
        self.species=tuple(s for s,_,_ in active)
        for a in (self.center,self.centers,self.cutoffs):a.setflags(write=False)

    def select(self, cj, rcut_j):
        displacement=np.asarray(cj,dtype=np.float64)-self.centers
        distances=np.linalg.norm(displacement,axis=1)
        limits=self.cutoffs+rcut_j
        selected=distances<=limits
        # axis reductions can round differently than norm(single_vector).
        # At the boundary repeat the exact old operation, not a tolerance.
        near=np.abs(distances-limits)<=32*np.finfo(float).eps*np.maximum(1.,np.maximum(distances,limits))
        for k in np.flatnonzero(near):
            selected[k]=np.linalg.norm(displacement[k])<=limits[k]
        return [(self.species[k],self.centers[k]) for k in np.flatnonzero(selected)]
