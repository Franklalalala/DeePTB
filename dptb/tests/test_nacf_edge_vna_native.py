"""Opt-in behavioral checks: set DPTB_NACF_TOPOLOGY_LIBRARY after explicit build."""
import itertools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.nacf.topology import build_edge_topology
from dptb.nacf.edge_vna import NACFEdgeVNAPlan

pytestmark = pytest.mark.skipif(not os.environ.get('DPTB_NACF_TOPOLOGY_LIBRARY'), reason='native topology not built')


def geometry(pbc=(True,True,True)):
    return dict(symbols=['A','B','A'],positions_bohr=np.array([[.1,.3,.2],[1.2,.1,.7],[.7,1.2,.3]]),
                cell_bohr=np.array([[2.9,.1,.2],[.3,3.1,.1],[.2,.1,3.2]]),pbc=np.array(pbc),
                edge_index=np.array([[2,0,0,1,1,2],[0,2,1,0,2,1]]),edge_cell_shift=np.zeros((6,3),dtype=int))


def reference_topology(g):
    pos,cell,pbc=g['positions_bohr'],g['cell_bohr'],g['pbc']
    ao=np.array([1.3 if s=='A' else 1.1 for s in g['symbols']])
    cc=np.array([1.2 if s=='A' else .9 for s in g['symbols']])
    keys=set()
    ranges=[range(-5,6) if b else (0,) for b in pbc]
    for i,k,t in itertools.product(range(len(pos)),range(len(pos)),itertools.product(*ranges)):
        t=np.array(t)
        if i==k and not np.any(t):continue
        if np.linalg.norm(pos[i]-pos[k]-t@cell)<ao[i]+cc[k]-1e-12:
            keys.add((i,k,*map(int,-t)))
    edges=[(int(i),int(j),*map(int,r)) for (i,j),r in zip(g['edge_index'].T,g['edge_cell_shift'])]
    reverse=[edges.index((j,i,-x,-y,-z)) for i,j,x,y,z in edges]
    terms=set()
    for row,(i,j,*shift) in enumerate(edges):
        if row>reverse[row]:continue
        for q in keys:
            if q[0]!=i:continue
            r=(j,q[1],*(np.array(q[2:])+shift))
            if r in keys:terms.add((row,q,r))
    return keys,terms,reverse


def native(g,**kwargs):
    return build_edge_topology(g['positions_bohr'],g['cell_bohr'],g['pbc'],
            [1.3 if s=='A' else 1.1 for s in g['symbols']],
            [1.2 if s=='A' else .9 for s in g['symbols']],g['edge_index'],g['edge_cell_shift'],**kwargs)


@pytest.mark.parametrize('pbc',[(True,True,True),(True,False,True),(False,False,False)])
@pytest.mark.parametrize('rewrap',[False,True])
def test_exact_periodic_queries_and_terms(pbc,rewrap):
    g=geometry(pbc)
    if rewrap:
        wrap=np.array([[1,-1,0],[-2,0,1],[0,1,-1]])*g['pbc']
        g['positions_bohr']+=wrap@g['cell_bohr']
        i,j=g['edge_index'];g['edge_cell_shift']+=wrap[i]-wrap[j]
    q,t,r=reference_topology(g);got=native(g)
    assert set(map(tuple,got['queries']))==q
    assert {(int(row),tuple(got['queries'][a]),tuple(got['queries'][b])) for row,a,b in got['terms']}==t
    np.testing.assert_array_equal(got['reverse'],r)


@pytest.mark.parametrize('failure',['duplicate','missing_reverse','fractional','nonperiodic','onsite','nonfinite','budget'])
def test_invalid_graph_fails(failure):
    g=geometry();options={}
    if failure=='duplicate':g['edge_index'][:,1]=g['edge_index'][:,0]
    if failure=='missing_reverse':g['edge_index']=g['edge_index'][:,:-1];g['edge_cell_shift']=g['edge_cell_shift'][:-1]
    if failure=='fractional':g['edge_cell_shift']=g['edge_cell_shift'].astype(float)+.5
    if failure=='nonperiodic':g['pbc'][:]=False;g['edge_cell_shift'][0,0]=1
    if failure=='onsite':g['edge_index'][:,0]=0
    if failure=='nonfinite':g['positions_bohr'][0,0]=np.nan
    if failure=='budget':options['max_terms']=1
    with pytest.raises(ValueError):native(g,**options)


class AnalyticFactor(torch.nn.Module):
    def __init__(self,width):super().__init__();self.width=width
    def forward(self,d):
        v=torch.stack((torch.ones_like(d[:,0]),d[:,0]+.2*d[:,2],d[:,1]-.3*d[:,2]),dim=1)
        v=v*torch.exp(-.2*(d*d).sum(1))[:,None]
        return torch.stack([v+(k*.13)*v.roll(1,1) for k in range(self.width)],dim=-1)


class Bank(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('_anchor',torch.empty(0,dtype=torch.float64))
        self.tables=torch.nn.ModuleDict()
        self.p2=SimpleNamespace(species={'A':dict(orbital_cutoff_bohr=1.3,orbital_norb=2),
                                         'B':dict(orbital_cutoff_bohr=1.1,orbital_norb=1)})
        self.p23=SimpleNamespace(species={'A':dict(vna_cutoff_bohr=1.2),'B':dict(vna_cutoff_bohr=.9)},
                                  epsilon=lambda s:np.array([1.,-.3,.5])*(1. if s=='A' else .7))
    def p23_composition(self,symbols):return True,()
    def table(self,kind,centre,ao):
        key=centre+ao
        if key not in self.tables:self.tables[key]=AnalyticFactor(self.p2.species[ao]['orbital_norb'])
        return key


def independent_value(g,bank):
    _,terms,reverse=reference_topology(g);out=np.zeros((len(reverse),2,2))
    pos,cell=g['positions_bohr'],g['cell_bohr']
    def value(q):
        d=pos[q[0]]-pos[q[1]]+np.array(q[2:])@cell
        v=np.array([1.,d[0]+.2*d[2],d[1]-.3*d[2]])*np.exp(-.2*np.dot(d,d))
        return np.stack([v+(k*.13)*np.roll(v,1) for k in range(bank.p2.species[g['symbols'][q[0]]]['orbital_norb'])],axis=-1)
    for row,a,b in terms:
        x,y=value(a),value(b);z=x.T@np.diag(bank.p23.epsilon(g['symbols'][a[1]]))@y
        out[row,:z.shape[0],:z.shape[1]]+=z
    for row,r in enumerate(reverse):
        if row>r:out[row]=out[r].T
    return out


@pytest.mark.parametrize('prune',[True,False])
@pytest.mark.parametrize('chunk_bytes',[128,32*1024*1024])
def test_batch_scalar_oracle_and_row_order(prune,chunk_bytes):
    a,b=geometry(),geometry((True,False,False));b['positions_bohr'][2]+=[.2,-.1,.05]
    order=np.array([3,0,5,2,1,4]);b['edge_index']=b['edge_index'][:,order];b['edge_cell_shift']=b['edge_cell_shift'][order]
    bank=Bank();plan=NACFEdgeVNAPlan(bank,[a,b],prune_unused=prune,chunk_bytes=chunk_bytes)
    got=plan()['edge_vna_ao_ev'].numpy()
    want=np.concatenate([independent_value(g,bank) for g in [a,b]])
    np.testing.assert_allclose(got,want,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(plan.edge_ptr.numpy(),[0,6,12])
    np.testing.assert_array_equal(plan.edge_index.numpy()[:,6:]-3,b['edge_index'])
    for start,stop in plan.edge_slices:
        np.testing.assert_array_equal(got[start:stop],got[plan.reverse[start:stop].numpy()].transpose(0,2,1))


def test_empty_edges_and_no_silent_fallback():
    g=geometry();g['edge_index']=np.empty((2,0),dtype=int);g['edge_cell_shift']=np.empty((0,3),dtype=int)
    bank=Bank();plan=NACFEdgeVNAPlan(bank,[g])
    assert plan()['edge_vna_ao_ev'].shape==(0,2,2)
    assert len(bank.tables)==0
    bank.p23_composition=lambda symbols:(False,('A|B',))
    with pytest.raises(KeyError):NACFEdgeVNAPlan(bank,[g])


def test_periodic_self_images_exclude_only_endpoints():
    g=dict(symbols=['A'],positions_bohr=np.zeros((1,3)),cell_bohr=np.eye(3)*1.5,
           pbc=np.ones(3,dtype=bool),edge_index=np.zeros((2,2),dtype=int),
           edge_cell_shift=np.array([[1,0,0],[-1,0,0]]))
    q,t,_=reference_topology(g);got=native(g)
    assert len(t)>0
    assert set(map(tuple,got['queries']))==q
    assert {(int(row),tuple(got['queries'][a]),tuple(got['queries'][b])) for row,a,b in got['terms']}==t
    bank=Bank();plan=NACFEdgeVNAPlan(bank,[g])
    np.testing.assert_allclose(plan()['edge_vna_ao_ev'].numpy(),independent_value(g,bank),rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('distance,expected',[(2.-2e-12,True),(2.,False),(2.+2e-12,False)])
def test_strict_support_boundary(distance,expected):
    topo=build_edge_topology([[0.,0.,0.],[distance,0.,0.]],np.zeros((3,3)),[False]*3,
                             [1.,1.],[1.,1.],np.empty((2,0),int),np.empty((0,3),int))
    assert ((0,1,0,0,0) in set(map(tuple,topo['queries'])))==expected
