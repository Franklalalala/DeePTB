from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.nacf.soc import spin_angular_projector, spinor_d_matrix
from dptb.nacf.assembly import NACFTableBank, NACFBatchAssemblyPlan, NACFFeaturePlan


@pytest.mark.parametrize('l', [1,2,3])
def test_soc_projector_against_clebsch_gordan_and_sampled_harmonics(l):
    from sympy import S
    from sympy.physics.wigner import clebsch_gordan
    from dptb.data.interfaces.p2_table import real_sph_abacus, abacus_m_order, _complex_sph_harm
    rng=np.random.default_rng(67)
    xyz=rng.normal(size=(100,3)); xyz/=np.linalg.norm(xyz,axis=1)[:,None]
    theta=np.arccos(xyz[:,2]); phi=np.arctan2(xyz[:,1],xyz[:,0])
    yc=np.stack([_complex_sph_harm(l,m,theta,phi) for m in range(-l,l+1)],axis=1)
    yr=np.stack([real_sph_abacus(l,m,xyz) for m in abacus_m_order(l)],axis=1)
    coefficients=np.linalg.lstsq(yc,yr,rcond=None)[0]
    transform=np.kron(np.eye(2),coefficients)
    projectors=[]
    for twice_j in (2*l-1,2*l+1):
        j=S(twice_j)/2
        cg=np.array([[float(clebsch_gordan(l,S(1)/2,j,m,ms,S(mj)/2))
                      for mj in range(-twice_j,twice_j+1,2)]
                     for ms in (S(1)/2,-S(1)/2) for m in range(-l,l+1)])
        expected=transform.conj().T@(cg@cg.T)@transform
        actual=spin_angular_projector(l,float(j))
        np.testing.assert_allclose(actual,expected,atol=2e-14)
        np.testing.assert_allclose(actual@actual,actual,atol=2e-14)
        projectors.append(actual)
    np.testing.assert_allclose(sum(projectors),np.eye(2*(2*l+1)),atol=1e-14)


def test_soc_d_spin_trace_radial_couplings_and_time_reversal():
    shells=[1,1,1]; js=[.5,1.5,1.5]
    dij=np.array([[2.,0.,0.],[0.,3.,.4],[0.,.4,4.]])
    d=spinor_d_matrix(shells,js,dij,has_so=True)
    n=9
    scalar=np.kron(dij*np.array([[1/3,0,0],[0,2/3,2/3],[0,2/3,2/3]]),np.eye(3))
    np.testing.assert_allclose((d[:n,:n]+d[n:,n:])/2,scalar,atol=1e-14)
    # The training prior is scalar D_eff. In real harmonics Lz is imaginary,
    # so real uu/dd reproduce that prior even though the complex blocks differ.
    np.testing.assert_allclose(d[:n,:n].real,scalar,atol=1e-14)
    np.testing.assert_allclose(d[n:,n:].real,scalar,atol=1e-14)
    rng=np.random.default_rng(193)
    left,right=rng.normal(size=(n,7)),rng.normal(size=(n,5))
    np.testing.assert_allclose((left.T@d[:n,:n]@right).real,
                               left.T@scalar@right,atol=1e-13)
    np.testing.assert_allclose(d,d.conj().T,atol=1e-14)
    time_reversal=np.kron(np.array([[0,1],[-1,0]]),np.eye(n))
    np.testing.assert_allclose(time_reversal@d.conj()@time_reversal.T,d,atol=1e-14)
    assert np.max(abs(d.imag))>.1 and np.max(abs(d[:n,n:]))>.1


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_soc_assembly_matches_independent_complex_contractions_and_batch(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from dptb.tests.test_nacf_gpu import stores
    from dptb.data.interfaces.p23_table import P23VNAFactorAssembler
    p2,p23=stores()
    matrices={s:np.array([[2.,.3+.4j],[.3-.4j,3.]])*(1 if s=='X' else 1.4) for s in ('X','Y')}
    soc=SimpleNamespace(manifest={'source_p2_manifest_sha256':None},d_spinor=lambda s:matrices[s])
    bank=NACFTableBank(p2,p23,soc_store=soc,device=device)
    symbols=['X','Y']; pos=np.array([[0.,0.,0.],[1.3,.2,.1]])
    edges=np.array([[0,1],[1,0]]); shifts=np.zeros((2,3),int)
    plan=bank.prepare(symbols,pos,np.eye(3)*10,edges,shifts,pbc=(False,False,False))
    actual=plan()
    keys=[(0,0),(1,1),(0,1),(1,0)]
    expected=[]
    for i,j in keys:
        scalar=(p2.onsite_component(symbols[i],'p2_base') if i==j else
                p2.base_component(symbols[i],symbols[j],'p2_base').evaluate(pos[j]-pos[i]))[0,0]
        block=np.eye(2,dtype=complex)*scalar
        for k,s in enumerate(symbols):
            qi=p2.projector(s,symbols[i]).evaluate(pos[i]-pos[k])[0,0]
            qj=p2.projector(s,symbols[j]).evaluate(pos[j]-pos[k])[0,0]
            block+=qi*matrices[s]*qj
        expected.append(block*13.605698)
    vna,_,_=P23VNAFactorAssembler(p23,factor_dtype=np.float64).assemble_graph_addition(
        symbols=symbols,positions_bohr=pos,cell_bohr=np.eye(3)*10,edge_index=edges,edge_cell_shift=shifts,
        node_shapes=np.ones((2,2),int),edge_shapes=np.ones((2,2),int),node_pad_shape=(1,1),edge_pad_shape=(1,1))
    expected=np.stack(expected); expected[:2]+=vna*np.eye(2)
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(),expected[:2],atol=2e-8)
    np.testing.assert_allclose(actual['edge_p2_ao_ev'].cpu(),expected[2:],atol=2e-8)
    second=bank.prepare(symbols,pos*1.1,np.eye(3)*11,edges,shifts,pbc=(False,False,False))
    batch=NACFBatchAssemblyPlan([plan,second])()
    for key in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(batch[key],torch.cat([actual[key],second()[key]]))


@pytest.mark.parametrize('doubling',[True,False])
def test_soc_packing_four_spin_blocks_imaginary_and_unequal_widths(doubling):
    from dptb.data.transforms import OrbitalMapper
    from dptb.data.interfaces.ham_to_feature import block_to_feature
    from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
    idp=OrbitalMapper({'H':'1s','C':'1s1p'},method='e3tb',has_soc=True,
                      full_soc_prediction=True,nextham_uureal_mask=False,soc_complex_doubling=doubling)
    assembly=torch.nn.Module(); assembly.symbols=('H','C'); assembly.width=4
    assembly.positions=torch.zeros((2,3),dtype=torch.float64)
    assembly.edge_index=torch.tensor([[0,1],[1,0]])
    assembly.bank=SimpleNamespace(soc=object(),p2=SimpleNamespace(species={'H':{'orbital_shells':[0]},'C':{'orbital_shells':[0,1]}}))
    dtype=torch.float64 if doubling else torch.complex128
    plan=NACFFeaturePlan(assembly,idp,output_dtype=dtype)
    rng=torch.Generator().manual_seed(24)
    node=torch.randn((2,8,8),generator=rng,dtype=torch.complex128)
    edge=torch.randn((2,8,8),generator=rng,dtype=torch.complex128)
    actual=plan.pack(node,edge)
    converter=OrbAbacus2DeepTB(); blocks={}
    sizes=[1,4]; shells=[[0],[0,1]]
    for name,pairs,array in [('node',[(0,0),(1,1)],node),('edge',[(0,1),(1,0)],edge)]:
        for row,(i,j) in enumerate(pairs):
            ii=np.r_[np.arange(sizes[i]),4+np.arange(sizes[i])]
            jj=np.r_[np.arange(sizes[j]),4+np.arange(sizes[j])]
            block=array[row].numpy()[np.ix_(ii,jj)]
            blocks[f'{i}_{j}_0_0_0']=converter.transform(block,shells[i]*2,shells[j]*2)
    data={'atomic_numbers':torch.tensor([[1],[6]]),'edge_index':assembly.edge_index,'edge_cell_shift':torch.zeros((2,3))}
    idp(data)
    block_to_feature(data,idp,blocks,output_dtype=dtype)
    torch.testing.assert_close(actual[0],data['node_features'])
    torch.testing.assert_close(actual[1],data['edge_features'])
