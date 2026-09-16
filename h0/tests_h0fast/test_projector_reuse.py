from types import SimpleNamespace
import numpy as np
import pytest
import torch

from h0rebuild.projector_reuse import ProjectorReuseTwoCenter, factor_key


def test_exact_key_does_not_merge_close_geometry_or_species():
    a=np.array([1.,2.,3.]); b=a.copy(); b[0]=np.nextafter(b[0],np.inf)
    assert factor_key(0,1,a)!=factor_key(0,1,b)
    assert factor_key(0,1,a)!=factor_key(1,0,a)
    assert factor_key(0,1,a)==factor_key(0,1,a.copy())
    with pytest.raises(ValueError):factor_key(0,1,[np.nan,0,0])


def test_invalid_budget():
    for value in (0,-1,np.nan,np.inf):
        with pytest.raises(ValueError):ProjectorReuseTwoCenter(None,value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
@pytest.mark.parametrize('budget',[0.00001,0.0001,1.0])
def test_reuse_eviction_private_storage_and_side_stream(budget):
    base=SimpleNamespace(device=torch.device('cuda:0'),metadata={})
    obj=ProjectorReuseTwoCenter(base,budget,batch_edges=2)
    def evaluate(reqs):
        return torch.tensor([[[r[0]+2*r[1]+r[2][0],r[2][1]],
                              [r[2][2],sum(r[2])]] for r in reqs],device=base.device,dtype=torch.float64)
    obj._evaluate=evaluate
    requests=[(0,1,np.array([float(i),2.,3.])) for i in range(8)]
    reference=evaluate(requests+[requests[0]])
    got=obj._factors(requests+[requests[0]])
    assert torch.equal(got,reference)
    got.zero_()  # Public gather cannot mutate retained factors.
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual=obj._factors(requests+[requests[0]])
        # Eviction while a side stream has consumed masters.
        obj._factors([(1,0,np.array([float(i),1.,0.])) for i in range(8)])
    stream.synchronize()
    assert torch.equal(actual,reference)
    assert obj.stats['resident_bytes']<=obj.stats['budget_bytes']
    assert obj.stats['peak_resident_bytes']<=obj.stats['budget_bytes']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_exact_duplicate_elimination_and_distinct_displacement():
    obj=ProjectorReuseTwoCenter(SimpleNamespace(device=torch.device('cuda:0'),metadata={}),1.)
    calls=[]
    def evaluate(reqs):
        calls.append(len(reqs))
        return torch.tensor([[[r[2][0]]] for r in reqs],device=obj.device,dtype=torch.float64)
    obj._evaluate=evaluate
    a=(0,0,np.array([1.,0,0]));b=(0,0,np.array([np.nextafter(1.,2.),0,0]))
    obj._factors([a,a,b]);obj._factors([a,b,a])
    assert calls==[2]
