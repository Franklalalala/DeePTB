"""Optional scalar two-center backend using the official ABACUS ModuleNAO.

ABACUS's own spherical Bessel transforms and real Gaunt tables compute S, T
and KB overlaps. Local Veff remains the existing exact-FFT collocation path.
No target Hamiltonian, overlap labels, or fitted corrections are used here.
"""
from collections import Counter
from pathlib import Path
import tempfile
import time
import numpy as np
from .models import abacus_m_order


class PyAbacusTwoCenter:
    def __init__(self, species_data, *, dr_bohr=.01, cache_dir=None, nspin=1):
        from pyabacus import ModuleNAO as nao
        from pyabacus import ModuleBase as base
        import pyabacus
        if dr_bohr <= 0:
            raise ValueError('two-center grid step must be positive')
        if nspin not in (1,4):
            raise ValueError('nspin must be 1 or 4')
        self.nspin=nspin
        if nspin == 1 and any(d.upf.has_so for d in species_data.values()):
            raise ValueError('The pyabacus backend currently requires scalarized UPFs')
        start=time.perf_counter()
        self.species=list(species_data)
        self.index={s:i for i,s in enumerate(self.species)}
        self.sd=species_data
        self.desc={'orb':{},'proj':{}}
        self.collections={}
        self.d={}
        self.sbt=base.SphericalBesselTransformer()
        self._cache={}
        self.stats={'cache_hits':0,'cache_misses':0}
        self.tmp=tempfile.TemporaryDirectory(prefix='h0flash-pyabacus-',dir=cache_dir)
        try:
            files=[]
            for symbol,data in species_data.items():
                if data.orb.source is None:
                    raise ValueError('Official ORB source required for pyabacus collection layout')
                files.append(str(data.orb.source))
                self.desc['orb'][symbol]=[(x.l,x.zeta,m) for x in data.orb.channels for m in abacus_m_order(x.l)]
            orb=nao.RadialCollection()
            orb.build(len(files),files,'o')
            # Keep exactly the normalization and source-channel labels validated
            # by h0rebuild's ABACUS-compatible reader.
            for symbol,data in species_data.items():
                for c in data.orb.channels:
                    orb(self.index[symbol],c.l,c.zeta).build(c.l,True,len(data.orb.r),data.orb.r,c.radial,0,c.zeta,symbol,self.index[symbol],False)
            self.collections['orb']=orb
            files=[]
            for symbol,data in species_data.items():
                upf=data.upf
                counts=Counter(p.l for p in upf.projectors)
                if not counts:
                    raise NotImplementedError('pyabacus KB layout requires at least one projector per species')
                lmax=max(counts)
                # Seed only the upstream collection's shell layout. Its radial
                # arrays are replaced below with exact PP_R/PP_BETA data.
                cutoff=upf.max_projector_cutoff
                n=max(5,int(np.ceil(cutoff/.01))+1)
                step=cutoff/(n-1)
                rg=np.arange(n)*step
                header=['Element '+symbol,'Energy Cutoff(Ry) 100',f'Radius Cutoff(a.u.) {cutoff:.17g}',f'Lmax {lmax}']
                header += [f'Number of {"SPDFGH"[l]}orbital--> {counts[l]}' for l in range(lmax+1)]
                header += ['SUMMARY END',f'Mesh {n}',f'dr {step:.17g}']
                for l in range(lmax+1):
                    for z in range(counts[l]):
                        header += ['Type L N',f'0 {l} {z}',' '.join(f'{x:.17g}' for x in rg**l*np.exp(-rg)*(1-rg/cutoff)**2)]
                path=Path(self.tmp.name)/(symbol+'.projector-layout.orb')
                path.write_text('\n'.join(header)+'\n')
                files.append(str(path))
            proj=nao.RadialCollection()
            proj.build(len(files),files,'o')
            for symbol,data in species_data.items():
                upf=data.upf; seen=Counter(); slots={}; descriptors=[]
                for p in upf.projectors:
                    z=seen[p.l];seen[p.l]+=1
                    slots[p.index]=(p.l,z)
                    # pr=1 means the provided array is r*beta(r), exactly UPF.
                    radial_u=p.radial_u.copy()
                    radial_u[p.cutoff_index:]=0.
                    proj(self.index[symbol],p.l,z).build(p.l,True,len(upf.r),upf.r,radial_u,1,z,symbol,self.index[symbol],False)
                for l in sorted(seen):
                    for z in range(seen[l]):
                        descriptors += [(l,z,m) for m in abacus_m_order(l)]
                self.desc['proj'][symbol]=descriptors
                d=np.zeros((len(descriptors),len(descriptors)))
                desc_to_idx={x:i for i,x in enumerate(descriptors)}
                for p in upf.projectors:
                    for q in upf.projectors:
                        if p.l!=q.l:continue
                        for m in abacus_m_order(p.l):
                            i=desc_to_idx[(*slots[p.index],m)]
                            j=desc_to_idx[(*slots[q.index],m)]
                            d[i,j]=upf.dij_ry[p.index,q.index]
                self.d[symbol]=d
                if nspin == 4:
                    self.d[symbol]=spinor_projector_matrix(upf,slots,descriptors)
            self.collections['proj']=proj
            self.cutoff=2*max(max(d.orb.rcut,d.upf.max_projector_cutoff) for d in species_data.values())
            self.nr=int(np.ceil(self.cutoff/dr_bohr))+1
            self.dr=self.cutoff/(self.nr-1)
            for collection in self.collections.values():
                collection.set_transformer(self.sbt)
                collection.set_uniform_grid(True,self.nr,self.cutoff,'i',True)
            self.integrators={}
            for left,right,op in [('orb','orb','S'),('orb','orb','T'),('proj','orb','S')]:
                integrator=nao.TwoCenterIntegrator()
                integrator.tabulate(self.collections[left],self.collections[right],op,self.nr,self.cutoff)
                self.integrators[(left,right,op)]=integrator
            self.metadata={'backend':'official pyabacus.ModuleNAO','module_file':pyabacus.__file__,
                           'validated_upstream_commit':'1f16da12f8ae807ef87b2134683c64f99bc95d88',
                           'dr_bohr':self.dr,'table_radius_bohr':self.cutoff,'radial_points':self.nr,
                           'setup_seconds':time.perf_counter()-start,
                           'projectors':('full relativistic (l,j) spinor KB' if nspin==4 else 'ABACUS scalar average')+'; exact UPF radial_u with implicit exponent pr=1',
                           'nspin':nspin,
                           'harmonics':'native ABACUS real m=0,+1,-1,...; no fitted transform'}
        finally:
            self.tmp.cleanup()

    def overlap(self,left,symbol_left,right,symbol_right,r,op='S'):
        displacement=np.asarray(r,dtype=np.float64)
        key=(left,symbol_left,right,symbol_right,op,tuple(displacement))
        if key in self._cache:
            self.stats['cache_hits']+=1
            return self._cache[key]
        self.stats['cache_misses']+=1
        integrator=self.integrators[(left,right,op)]
        a=self.desc[left][symbol_left];b=self.desc[right][symbol_right]
        out=np.zeros((len(a),len(b)))
        it,jt=self.index[symbol_left],self.index[symbol_right]
        for i,(l,z,m) in enumerate(a):
            for j,(ll,zz,mm) in enumerate(b):
                out[i,j]=np.asarray(integrator.calculate(it,l,z,m,jt,ll,zz,mm,displacement,False)[0]).item()
        self._cache[key]=out
        return out

    def scalar_pair(self,symbol_i,symbol_j,ci,cj):
        r=np.asarray(cj)-np.asarray(ci)
        return (self.overlap('orb',symbol_i,'orb',symbol_j,r,'S'),
                self.overlap('orb',symbol_i,'orb',symbol_j,r,'T'))

    def nonlocal_block(self,structure,symbol_i,symbol_j,ci,cj,candidates):
        ni,nj=self.sd[symbol_i].orb.norb,self.sd[symbol_j].orb.norb
        mult=2 if self.nspin==4 else 1
        out=np.zeros((mult*ni,mult*nj),dtype=complex if self.nspin==4 else float)
        for atom_index,pcenter in candidates:
            symbol=structure.atoms[atom_index].species
            cutoff=self.sd[symbol].upf.max_projector_cutoff
            if np.linalg.norm(np.asarray(ci)-pcenter)>cutoff+self.sd[symbol_i].orb.rcut:continue
            if np.linalg.norm(np.asarray(cj)-pcenter)>cutoff+self.sd[symbol_j].orb.rcut:continue
            qi=self.overlap('proj',symbol,'orb',symbol_i,np.asarray(ci)-pcenter)
            qj=self.overlap('proj',symbol,'orb',symbol_j,np.asarray(cj)-pcenter)
            if self.nspin==1:
                out += qi.T @ self.d[symbol] @ qj
            else:
                d=self.d[symbol].reshape(2,len(qi),2,len(qj))
                out += np.einsum('pa,sptq,qb->satb',qi,d,qj,optimize=True).reshape(2*ni,2*nj)
        return out


def spinor_projector_matrix(upf, slots, descriptors):
    """Physical KB matrix in [up real-projectors, down real-projectors].

    Convert analytic ABACUS real harmonics to complex bra overlaps and apply
    the same (l,j) Clebsch-Gordan convention as the Cartesian reference.
    No scalar averaging is applied to fully relativistic input.
    """
    from .nonlocal_kb import _group_projectors, _scalar_group_block, _spinor_group_block
    lookup={x:i for i,x in enumerate(descriptors)}
    n=len(descriptors)
    overlaps={}
    for pos,p in enumerate(upf.projectors):
        l,z=slots[p.index]
        q=np.zeros((2*l+1,n),complex)
        q[l,lookup[(l,z,0)]]=1
        for m in range(1,l+1):
            plus,minus=lookup[(l,z,m)],lookup[(l,z,-m)]
            q[l+m,plus]=1/np.sqrt(2)
            q[l+m,minus]=-1j/np.sqrt(2)
            q[l-m,plus]=(-1)**m/np.sqrt(2)
            q[l-m,minus]=1j*(-1)**m/np.sqrt(2)
        overlaps[pos]=q
    out=np.zeros((2*n,2*n),complex)
    for (l,j),positions in _group_projectors(upf,list(overlaps)).items():
        d=upf.dij_ry[np.ix_(positions,positions)]
        if j is None:
            block=_scalar_group_block(positions,positions,d,overlaps,overlaps)
            out[:n,:n]+=block
            out[n:,n:]+=block
        else:
            out+=_spinor_group_block(positions,positions,l,j,d,overlaps,overlaps)
    return out
