import itertools
import os
import numpy as np
import pytest
import torch
from dptb.nacf.density import density_topology,onsite_density_neighbors,probe_environment_density

native=pytest.mark.skipif(not os.environ.get('DPTB_NACF_TOPOLOGY_LIBRARY'),reason='native topology not built')
cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


@native
@pytest.mark.parametrize('periodic',[False,True])
def test_density_images_and_directed_edge_exclusion(periodic):
    pos=np.array([[.1,.2,.3],[1.,.2,.5]])
    cell=np.array([[2.6,0,0],[.4,2.9,0],[0,0,3.3]])
    radii=[1.4,1.8];supports=[.8,.9];edge=np.array([[0],[1]]);shift=np.zeros((1,3),int)
    got=density_topology(pos,cell,[periodic]*3,radii,supports,edge,shift)
    want=set()
    for i,k,s in itertools.product(range(2),range(2),itertools.product(range(-2,3) if periodic else (0,),repeat=3)):
        if i==k and s==(0,0,0):continue
        if np.linalg.norm(pos[k]+np.array(s)@cell-pos[i])<=radii[i]+supports[k]+1e-12:want.add((i,k,*s))
    assert set(map(tuple,got['queries']))==want
    edge_q=got['queries'][got['edge_queries']]
    assert set(map(tuple,edge_q))=={q for q in want if q[0]==0 and q!=(0,1,0,0,0)}
    assert got['edge_ptr'].tolist()==[0,len(edge_q)]


@native
def test_density_onsite_legacy_order_empty_and_batch():
    from dptb.data.interfaces.p2_batch import VectorizedNearbyImageEnumerator
    for pos in (np.array([[0.,0,0]]),np.array([[.1,.2,.3],[1.,.2,.5],[.5,.7,.8]])):
        g=dict(symbols=['X','Y','X'][:len(pos)],positions_bohr=pos,cell_bohr=np.eye(3)*2.5,pbc=[True]*3)
        got=onsite_density_neighbors(g,[1.]*len(pos),{'X':1.,'Y':1.},legacy_radius=3.2)
        for i in range(len(pos)):
            want={};enum=VectorizedNearbyImageEnumerator(g['cell_bohr'])
            for s,p in zip(g['symbols'],pos):
                _,centers=enum.query_arrays(p,pos[i],3.2);d=centers-pos[i];d=d[np.linalg.norm(d,axis=1)<3.2]
                if len(d):want.setdefault(s,[]).append(d)
            for s,v in want.items():np.testing.assert_array_equal(got[i][s],np.concatenate(v))
    q=density_topology([[0,0,0]],np.zeros((3,3)),[False]*3,[1.],[0.])
    assert q['queries'].shape==(0,5) and q['edge_ptr'].tolist()==[0]


def test_probe_density_weighted_empty_and_endpoint_contract():
    p=torch.tensor([[[0.,0,0],[1.,0,0]],[[0.,0,0],[2.,0,0]]],dtype=torch.float64)
    neighbors=np.array([[.5,0,0],[0,1,0]])
    bank={'X':lambda r:torch.exp(-r),'Y':lambda r:2*torch.exp(-r)}
    got=probe_environment_density(p,neighbors,['X','Y'],[0,2,2],bank,weights=torch.tensor([[1.,3.],[1.,1.]]))
    rho=torch.exp(-torch.linalg.vector_norm(p[0]-torch.tensor(neighbors[0]),dim=-1))+2*torch.exp(-torch.linalg.vector_norm(p[0]-torch.tensor(neighbors[1]),dim=-1))
    torch.testing.assert_close(got,torch.stack(((rho*torch.tensor([1.,3.])).sum()/4,rho.new_zeros(()))),atol=1e-15,rtol=0)
    assert probe_environment_density(p[:0],np.empty((0,3)),[],[0],bank).shape==(0,)


@cuda
def test_radial_multi_against_individual_near_south_and_empty():
    from dptb.data.interfaces.p2_table import RadialBlockTable
    from dptb.nacf.radial import TorchRadialBlockTable
    from dptb.nacf.fusion import RadialMultiPlan
    r=np.array([0.,.3,1.2,2.]);v=np.zeros((4,4,4));v[:,0,0]=[1,.8,.3,0]
    for i in range(1,4):v[:,i,i]=[2,1.5,.4,0]
    tables=[TorchRadialBlockTable(RadialBlockTable(r,v*c,(0,1),(0,1),2.),device='cuda',backend='cuda') for c in (1.,.8,1.7)]
    vec=torch.tensor([[0.,0.,0.],[1e-7,1e-6,-.7],[.2,.3,.9],[0.,0.,2.],[0.,0.,-1.]],device='cuda',dtype=torch.float64)
    plan=RadialMultiPlan([(tables,5),([tables[0]],0)])
    got=plan(vec)
    for a,t in zip(got,tables):torch.testing.assert_close(a,t(vec),atol=0,rtol=0)
    assert got[-1].shape==(0,4,4)
    with pytest.raises(NotImplementedError):RadialMultiPlan([(tables,5)],background_nodes=[0.,1.])


@cuda
@pytest.mark.parametrize('diagonal',[False,True])
def test_contraction_scatter_dense_diagonal_and_empty(diagonal):
    from dptb.nacf.fusion import contract_add
    torch.manual_seed(71)
    a=torch.randn(7,9,4,device='cuda',dtype=torch.float64);b=torch.randn(8,9,3,device='cuda',dtype=torch.float64)
    m=torch.randn((9,) if diagonal else (9,9),device='cuda',dtype=torch.float64)
    rows=torch.tensor([[0,1,2],[2,3,5],[0,2,4],[2,6,7]],device='cuda')
    got=torch.zeros(3,5,5,device='cuda',dtype=torch.float64);ref=got.clone()
    mat=torch.diag(m) if diagonal else m
    ref[:,:4,:3].index_add_(0,rows[:,0],a[rows[:,1]].transpose(-1,-2)@mat@b[rows[:,2]])
    contract_add(a,m,b,rows,got);torch.testing.assert_close(got,ref,atol=1e-12,rtol=0)
    old=got.clone();contract_add(a,m,b,rows[:0],got);torch.testing.assert_close(got,old,atol=0,rtol=0)


@native
@cuda
def test_fused_single_and_merged_periodic_assembly():
    from dptb.tests.test_nacf_gpu import stores
    from dptb.nacf.assembly import NACFTableBank,NACFBatchAssemblyPlan
    from dptb.nacf.fusion import enable_fusion
    p2,p23=stores();bank=NACFTableBank(p2,p23,device='cuda')
    g=dict(symbols=['X','Y'],positions_bohr=[[0.,0,0],[1.,.2,.1]],cell_bohr=np.eye(3)*3.,edge_index=[[0,1],[1,0]],edge_cell_shift=np.zeros((2,3),int))
    plans=[bank.prepare(**g,topology='native') for _ in range(2)]
    for plan in (plans[0],NACFBatchAssemblyPlan(plans),bank.prepare_edge_vna_batch([g,g])):
        ref=plan();enable_fusion(plan,contraction=True);got=plan()
        for k in ref:
            if ref[k].is_floating_point():torch.testing.assert_close(got[k],ref[k],atol=1e-12,rtol=0)


@cuda
def test_fused_density_channel_clipping_support_and_empty():
    from types import SimpleNamespace
    from dptb.nacf.density import density_sum_cuda
    torch.manual_seed(91)
    knots=torch.tensor([0.,.3,1.1,2.],device='cuda',dtype=torch.float64)
    coeff=[torch.randn(4,3,device='cuda',dtype=torch.float64) for _ in range(2)]
    points=torch.randn(57,3,device='cuda',dtype=torch.float64)
    neighbors={'X':torch.randn(37,3,device='cuda',dtype=torch.float64)}
    bank={'X':SimpleNamespace(knots=knots,coeff=coeff)}
    ref=points.new_zeros(len(points))
    for start in range(0,37,16):
        r=torch.linalg.vector_norm(points[:,None,:]-neighbors['X'][None,start:start+16,:],dim=-1)
        rr=r.clamp(knots[0],knots[-1]);idx=(torch.searchsorted(knots,rr.contiguous(),right=True)-1).clamp(0,len(knots)-2);d=rr-knots[idx]
        value=torch.zeros_like(r)
        for c in coeff:value+=(((c[0,idx]*d+c[1,idx])*d+c[2,idx])*d+c[3,idx]).clamp_min(0)
        ref+=torch.where(r>knots[-1],0.,value).sum(1)
    got=density_sum_cuda(points,neighbors,bank)
    torch.testing.assert_close(got,ref,atol=1e-12,rtol=0)
    assert density_sum_cuda(points[:0],neighbors,bank).shape==(0,)
