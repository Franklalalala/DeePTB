"""Exact FFT AO contraction reusing explicitly supplied integer atom images."""
from collections import OrderedDict
import time
import numpy as np
from .grid_collocation import FFTGridAOCache


class PeriodicFFTGridAOCache:
    def __init__(self,field,*,max_bytes=512*1024**2,chunk_size=100_000,device='cpu'):
        self.field=field
        self.max_bytes=max_bytes
        self.device=str(device)
        self.builder=FFTGridAOCache(field,max_bytes=0,chunk_size=chunk_size)
        self._anchors=OrderedDict()
        self.resident=0
        self._stats={'anchor_builds':0,'anchor_hits':0,'evictions':0,'peak_bytes':0,
                     'anchor_seconds':0.,'contraction_seconds':0.,'pairs':0,'points':0}
        self.torch=None
        if self.device!='cpu':
            import torch
            if not self.device.startswith('cuda'):
                raise ValueError('Periodic collocation device must be cpu or cuda')
            self.torch=torch

    def _anchor(self,evaluator,center):
        key=(id(evaluator),np.asarray(center,dtype=np.float64).tobytes())
        if key in self._anchors:
            self._stats['anchor_hits']+=1
            self._anchors.move_to_end(key)
            return self._anchors[key]
        start=time.perf_counter()
        support=self.builder.support(evaluator,center)
        if not support.npoints:
            raise ValueError('No FFT nodes in an orbital support')
        indices=support.integer_indices
        lo=indices.min(axis=0);hi=indices.max(axis=0)
        shape=hi-lo+1
        lookup=np.full(tuple(shape),-1,dtype=np.int32)
        rel=indices-lo
        lookup[rel[:,0],rel[:,1],rel[:,2]]=np.arange(support.npoints,dtype=np.int32)
        wrapped=np.mod(indices,np.asarray(self.field.shape))
        potential=self.field.values_ry[wrapped[:,0],wrapped[:,1],wrapped[:,2]]
        values=support.values
        size=int(lookup.nbytes+values.nbytes+potential.nbytes)
        if self.torch is not None:
            values=self.torch.tensor(values,device=self.device,dtype=self.torch.float64)
            potential=self.torch.tensor(potential,device=self.device,dtype=self.torch.float64)
        entry={'lo':lo,'hi':hi,'lookup':lookup,'values':values,'potential':potential,'bytes':size}
        self._stats['anchor_builds']+=1
        self._stats['anchor_seconds']+=time.perf_counter()-start
        if self.max_bytes is None or size<=self.max_bytes:
            while self._anchors and self.max_bytes is not None and self.resident+size>self.max_bytes:
                _,old=self._anchors.popitem(last=False)
                self.resident-=old['bytes'];self._stats['evictions']+=1
            self._anchors[key]=entry;self.resident+=size
            self._stats['peak_bytes']=max(self._stats['peak_bytes'],self.resident)
        return entry

    def contract_pair(self,left,right,center_i,center_j_home,image):
        """Ket image is the exact integer translation, never inferred/rounded."""
        image=np.asarray(image)
        if image.shape!=(3,) or not np.array_equal(image,np.rint(image)):
            raise ValueError('Explicit integer lattice image required')
        center_j=np.asarray(center_j_home)+image@self.field.cell_bohr
        self._stats['pairs']+=1
        if np.linalg.norm(np.asarray(center_i)-center_j)>left.basis.rcut+right.basis.rcut:
            return np.zeros((left.norb,right.norb))
        a=self._anchor(left,center_i);b=self._anchor(right,center_j_home)
        start=time.perf_counter()
        shift=image.astype(np.int64)*np.asarray(self.field.shape,dtype=np.int64)
        lo=np.maximum(a['lo'],b['lo']+shift);hi=np.minimum(a['hi'],b['hi']+shift)
        if np.any(hi<lo):return np.zeros((left.norb,right.norb))
        sa=tuple(slice(int(x),int(y)+1) for x,y in zip(lo-a['lo'],hi-a['lo']))
        sb=tuple(slice(int(x),int(y)+1) for x,y in zip(lo-shift-b['lo'],hi-shift-b['lo']))
        ia=a['lookup'][sa].ravel();ib=b['lookup'][sb].ravel()
        keep=(ia>=0)&(ib>=0)
        ia=ia[keep].astype(np.int64);ib=ib[keep].astype(np.int64)
        self._stats['points']+=len(ia)
        if self.torch is None:
            out=a['values'][ia].T @ (a['potential'][ia,None]*b['values'][ib])*self.field.grid_weight
        else:
            ia=self.torch.as_tensor(ia,device=self.device)
            ib=self.torch.as_tensor(ib,device=self.device)
            out=(a['values'][ia].T @ (a['potential'][ia,None]*b['values'][ib])*self.field.grid_weight).cpu().numpy()
        self._stats['contraction_seconds']+=time.perf_counter()-start
        return out

    def stats(self):
        return {**self._stats,'resident_bytes':self.resident,'device':self.device,
                'identity':'canonical atom plus supplied integer lattice image; no rounded centers'}
