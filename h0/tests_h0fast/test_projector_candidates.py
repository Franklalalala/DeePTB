from types import SimpleNamespace
import numpy as np
import pytest
from h0rebuild.spatial import PeriodicAtomIndex
from h0rebuild.projector_candidates import anchor_candidates
from h0rebuild.projector_candidates import PreparedProjectorCandidates


@pytest.mark.parametrize('seed',range(8))
def test_anchor_keeps_identical_ordered_contributions_with_skew_and_images(seed):
    rng=np.random.default_rng(seed)
    cell=np.array([[7.,0,0],[2.8,6.4,0],[-1.9,1.5,8.1]])
    frac=rng.uniform(-1.5,2.5,(5,3))
    structure=SimpleNamespace(cell_bohr=cell,atoms=[SimpleNamespace(frac=f) for f in frac])
    index=PeriodicAtomIndex(structure,12.)
    pc=rng.uniform(.2,2.5,5)
    def active(candidates,ci,cj,ri,rj):
        return [(a,p.tobytes()) for a,p in candidates
            if np.linalg.norm(ci-p)<=ri+pc[a] and np.linalg.norm(cj-p)<=rj+pc[a]]
    for _ in range(8):
        ci,cj=rng.uniform(-7,15,(2,3));ri,rj=rng.uniform(.2,5,2)
        radius=max(pc)+max(ri,rj)+.5*np.linalg.norm(ci-cj)
        old=tuple((h.atom_index,h.center_bohr) for h in index.query(.5*(ci+cj),radius))
        new=anchor_candidates(index,ci,ri,max(pc))
        assert active(new,ci,cj,ri,rj)==active(old,ci,cj,ri,rj)


@pytest.mark.parametrize('x',[2.,np.nextafter(2.,0.),np.nextafter(2.,np.inf)])
def test_cutoff_ulp_and_duplicate_periodic_images(x):
    structure=SimpleNamespace(cell_bohr=np.diag([4.,4.,4.]),
        atoms=[SimpleNamespace(frac=np.array([x/4,0.,0.]))])
    index=PeriodicAtomIndex(structure,4.)
    got=anchor_candidates(index,np.zeros(3),1.,1.)
    active=[p[0] for _,p in got if np.linalg.norm(p)<=2.]
    expected=[x+r*4 for r in (-1,0) if abs(x+r*4)<=2.]
    assert active==expected


@pytest.mark.parametrize('seed',range(8))
def test_prepared_selection_matches_scalar_cutoffs_and_order(seed):
    rng=np.random.default_rng(seed);ci=rng.normal(size=3);ri=2.
    symbols=['A','B'];cutoffs={'A':1.,'B':2.}
    st=SimpleNamespace(atoms=[SimpleNamespace(species=s) for s in symbols])
    tc=SimpleNamespace(sd={s:SimpleNamespace(upf=SimpleNamespace(max_projector_cutoff=c)) for s,c in cutoffs.items()},_check_species=symbols.index)
    candidates=[(k%2,rng.normal(size=3)*3) for k in range(60)]
    # Include exactly-on-cutoff and adjacent representable values.
    candidates += [(0,np.array([x,0.,0.])) for x in (2.,np.nextafter(2.,0.),np.nextafter(2.,np.inf))]
    packed=PreparedProjectorCandidates(candidates,st,tc,ci,ri)
    for cj in [np.zeros(3),*rng.normal(size=(8,3))]:
        expected=[(a,p.tobytes()) for a,p in candidates if np.linalg.norm(ci-p)<=ri+cutoffs[symbols[a]] and np.linalg.norm(cj-p)<=1.+cutoffs[symbols[a]]]
        assert [(a,p.tobytes()) for a,p in packed.select(cj,1.)]==expected
    saved=packed.centers.copy()
    for _,p in candidates:p[:]=1000
    np.testing.assert_array_equal(packed.centers,saved)
