"""Source-bound nonlocal spinor matrices for real NACF radial projectors.

Radial AO-projector overlaps remain real. Spin-orbit coupling lives in the
complex D matrix, expressed in ABACUS real harmonics and spin-major order.
Neither scalar spin traces nor uu-real training records determine full SOC.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


def spin_angular_projector(l, j):
    """Project onto j=l+/-1/2, independently of the radial projector index.

    P+ = (l+1+L.sigma)/(2l+1), P- = (l-L.sigma)/(2l+1).
    Output is [up real-AO, down real-AO], with m=0,+1,-1,... .
    """
    l=int(l)
    if l<0 or not (abs(j-l-.5)<1e-8 or (l>0 and abs(j-l+.5)<1e-8)):
        raise ValueError('invalid spin-angular l,j')
    n=2*l+1
    m=np.arange(-l,l+1)
    plus=np.zeros((n,n),complex)
    for column in range(n-1):
        plus[column+1,column]=np.sqrt(l*(l+1)-m[column]*(m[column]+1))
    ls=np.block([[np.diag(m),plus.T],[plus,-np.diag(m)]]).astype(complex)
    identity=np.eye(2*n)
    projector=((l+1)*identity+ls)/(2*l+1) if j>l else (l*identity-ls)/(2*l+1)
    # Rows are real ket coefficients in the complex spherical basis. Operator
    # conversion is C.conj() @ H @ C.T, not C @ H @ C.conj().T.
    c=np.zeros((n,n),complex)
    c[0,l]=1
    for k in range(1,l+1):
        c[2*k-1,l+k]=1/np.sqrt(2)
        c[2*k-1,l-k]=(-1)**k/np.sqrt(2)
        c[2*k,l+k]=-1j/np.sqrt(2)
        c[2*k,l-k]=1j*(-1)**k/np.sqrt(2)
    transform=np.kron(np.eye(2),c)
    return transform.conj()@projector@transform.T


def spinor_d_matrix(projector_shells, projector_j, dij_ry, *, has_so):
    """Expand UPF radial D into a full complex spinor operator in Ry.

    The radial shell list must be identical to the cached projector overlaps.
    Off-diagonal radial couplings with matching l,j are retained.
    """
    shells=list(map(int,projector_shells)); js=list(projector_j)
    dij=np.asarray(dij_ry,dtype=float)
    if len(js)!=len(shells) or dij.shape!=(len(shells),len(shells)) or not np.isfinite(dij).all():
        raise ValueError('invalid projector metadata or D matrix')
    if not np.allclose(dij,dij.T,atol=1e-12,rtol=1e-12):
        raise ValueError('UPF D matrix must be Hermitian')
    if has_so and any(j is None for j in js):
        raise ValueError('fully relativistic projectors require explicit j')
    offsets=np.r_[0,np.cumsum([2*l+1 for l in shells])]
    width=int(offsets[-1]); result=np.zeros((2*width,2*width),complex)
    for a,l in enumerate(shells):
        angular=spin_angular_projector(l,float(js[a])) if has_so else np.eye(2*(2*l+1))
        rows=np.r_[np.arange(offsets[a],offsets[a+1]),width+np.arange(offsets[a],offsets[a+1])]
        for b,lb in enumerate(shells):
            if l!=lb or (has_so and abs(float(js[a])-float(js[b]))>1e-8):
                continue
            cols=np.r_[np.arange(offsets[b],offsets[b+1]),width+np.arange(offsets[b],offsets[b+1])]
            result[np.ix_(rows,cols)]=dij[a,b]*angular
    return result


class SOCProjectorStore:
    """Immutable sidecar binding complex D matrices to the original P2 sources."""
    def __init__(self, root, p2):
        self.root=Path(root).resolve()
        raw=(self.root/'manifest.json').read_bytes()
        self.manifest=json.loads(raw)
        self.manifest_sha256=hashlib.sha256(raw).hexdigest()
        expected=hashlib.sha256((p2.root/'manifest.json').read_bytes()).hexdigest()
        if (self.manifest.get('schema')!='deeptb.soc_projector_table/v1' or
            self.manifest.get('source_p2_manifest_sha256')!=expected or
            self.manifest.get('spin_order')!='spin_major' or
            self.manifest.get('harmonic_convention')!='deeptb_abacus_real' or
            self.manifest.get('unit')!='Ry' or self.manifest.get('complete') is not True):
            raise ValueError('invalid SOC projector sidecar contract')
        self.p2=p2
        self._cache={}

    def d_spinor(self, symbol):
        if symbol not in self._cache:
            meta=self.manifest['species'][symbol]; source=self.p2.species[symbol]
            if meta['upf_sha256']!=source['upf_sha256'] or meta['projector_shells']!=source['projector_shells']:
                raise ValueError(f'SOC projector sources disagree for {symbol}')
            path=(self.root/meta['path']).resolve()
            if self.root not in path.parents: raise ValueError('SOC shard escapes table directory')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:
                raise ValueError(f'SOC shard checksum mismatch for {symbol}')
            with np.load(path,allow_pickle=False) as arrays:
                d=np.array(arrays['d_spinor_ry'],dtype=np.complex128)
            n=int(source['projector_norb'])
            if d.shape!=(2*n,2*n) or not np.isfinite(d).all() or not np.allclose(d,d.conj().T,atol=1e-11,rtol=1e-11):
                raise ValueError(f'invalid SOC D matrix for {symbol}')
            # This also binds normalization to the historical scalar table.
            if not np.allclose((d[:n,:n]+d[n:,n:])*.5,self.p2.d_eff(symbol),atol=1e-10,rtol=1e-10):
                raise ValueError(f'SOC spin trace does not reproduce scalar P2 for {symbol}')
            self._cache[symbol]=d
        return self._cache[symbol]
