from types import SimpleNamespace
import numpy as np
import pytest
import torch
from h0rebuild.pair_batches import scalar_pair_batches


@pytest.mark.parametrize('size',[1,2,4,128])
def test_bounded_heterogeneous_pairs_preserve_bytes_order_and_ownership(size):
    base=SimpleNamespace(norb_per_species={'A':2,'B':3})
    calls=[];produced=[]
    def evaluate(symbols,disp):
        calls.append(len(symbols));s=np.zeros((len(symbols),3,3));t=s.copy()
        for k,(si,sj) in enumerate(symbols):
            ni,nj=base.norb_per_species[si],base.norb_per_species[sj]
            s[k,:ni,:nj]=disp[k,0];t[k,:ni,:nj]=disp[k,1]
        gs,gt=torch.from_numpy(s),torch.from_numpy(t);produced.extend([gs,gt]);return gs,gt
    base.eval_two_center_batch=evaluate
    def scalar(si,sj,ci,cj):
        s,t=evaluate([(si,sj)],np.array([cj-ci]));ni,nj=base.norb_per_species[si],base.norb_per_species[sj]
        return s[0,:ni,:nj].numpy().copy(),t[0,:ni,:nj].numpy().copy()
    base.scalar_pair=scalar
    edges=[(k%2,(k+1)%2,(k,0,0),np.array([-0.,1.,2.]),np.array([float(k),3.,4.])) for k in range(9)]
    got=list(scalar_pair_batches(base,iter(edges),['A','B'],size))
    assert max(calls)<=size and sum(calls)==9
    for edge,s,t in got:
        assert edge is edges[edge[2][0]]
        assert s.shape==(base.norb_per_species[['A','B'][edge[0]]],base.norb_per_species[['A','B'][edge[1]]])
        np.testing.assert_array_equal(s,edge[4][0]-edge[3][0]);np.testing.assert_array_equal(t,2.)
        assert s.flags.owndata and t.flags.owndata
    for a in produced:a.fill_(999)
    assert all(not np.any(s==999) for _,s,_ in got)


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError):list(scalar_pair_batches(None,[],[],size))
