from dataclasses import replace
import numpy as np
import pytest
import torch
from h0rebuild.models import OrbitalBasis,OrbitalChannel
from h0rebuild.radial import OrbitalEvaluator
from h0rebuild.reciprocal import PeriodicField,reciprocal_grid


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('budget',[0,10**7])
@pytest.mark.parametrize('spin',[False,True])
def test_chunk_equals_separate_native_pairs_with_images_and_eviction(budget,spin):
    from h0rebuild.cuda_local_grid import CudaPeriodicFFTGridAOCache as Cache
    cell=np.array([[4.,0.,0.],[.8,4.,0.],[.2,.5,4.]])
    shape=(12,12,12);g,_=reciprocal_grid(cell,shape)
    xyz=np.indices(shape);v=np.cos(xyz[0]*.3)+np.sin(xyz[1]*.6)
    zero=np.zeros(shape);field=PeriodicField(cell,v,zero,zero,zero,zero,zero.astype(complex),g,{})
    zfield=replace(field,values_ry=v*.37+1.) if spin else None
    mesh=np.arange(31)*.1;radial=np.exp(-mesh)*(1-mesh/3)**2
    basis=OrbitalBasis('X',100.,mesh,.1,[OrbitalChannel(0,0,radial),OrbitalChannel(1,0,mesh*radial)])
    evaluator=OrbitalEvaluator(basis);ev={'X':evaluator};pos=np.array([[-1e-7,.5,.3],[2.,1.,.5]])
    pairs=[(i,j,R,pos[i],pos[j]+np.array(R)@cell) for i,j,R in [(0,0,(0,0,0)),(0,1,(0,0,0)),(1,0,(1,0,0)),(0,1,(9,0,0))]]
    separate=Cache(field,max_bytes=budget);zcache=Cache(zfield,max_bytes=budget) if spin else None
    ref=[(separate.contract_pair(evaluator,evaluator,ci,pos[j],R),
          zcache.contract_pair(evaluator,evaluator,ci,pos[j],R) if spin else None) for i,j,R,ci,cj in pairs]
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        combined=Cache(field,spin_z_field=zfield,max_bytes=budget)
        got=combined.contract_chunk(pairs,['X','X'],ev,pos)
    stream.synchronize()
    for (v,z),(rv,rz) in zip(got,ref):
        assert v.tobytes()==rv.tobytes() and v.flags.owndata
        if spin:assert z.tobytes()==rz.tobytes() and z.flags.owndata
    assert combined.resident<=budget
    assert combined.stats()['result_transfer_batches']==1
