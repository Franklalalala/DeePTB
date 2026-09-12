import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

def test_even_uncut_field_mesh():
    from h0rebuild.models import UPFData
    from h0rebuild.field_inputs import prepare_field_upf
    r=np.arange(832)*.01
    upf=UPFData('X',7.,r,np.full_like(r,.01),-2/(r+1),r*r*np.exp(-r),np.zeros((0,0)),[],False,nlcc=np.exp(-r))
    got=prepare_field_upf(upf,15.)
    assert len(got.r)==831 and got.r[-1]==8.30
    assert len(upf.r)==832

@pytest.mark.parametrize('attribute',['1.9','1.000001','nan','inf','1e0'])
def test_fractional_index_rejected(attribute):
    from h0rebuild.upf import _indexed_elements
    root=ET.fromstring(f'<UPF><PP_BETA.1 index="{attribute}"/></UPF>')
    with pytest.raises(ValueError):_indexed_elements(root,'PP_BETA')

def test_star_only_overflow_slot():
    from h0rebuild.upf import _indexed_elements
    with pytest.raises(ValueError):_indexed_elements(ET.fromstring('<UPF><PP_BETA.1 index="*"/></UPF>'),'PP_BETA')
    root=ET.fromstring('<UPF>'+''.join(f'<PP_BETA.{i} index="{i if i<10 else "*"}"/>' for i in range(1,11))+'</UPF>')
    rows,mismatches=_indexed_elements(root,'PP_BETA');assert len(rows)==10 and mismatches[-1]['slot']==10

@pytest.mark.parametrize('backend',['numpy','torch'])
def test_pw_projection_after_nonlinearity(backend):
    from h0rebuild.pw_derivatives import derivative
    n=32;x=2*np.pi*np.arange(n)/n
    g=np.zeros((n,1,1,3));g[:,0,0,0]=np.fft.fftfreq(n)*n
    mask=(g*g).sum(-1)<=1.
    v=(1+np.abs(np.cos(x)))[:,None,None]
    if backend=='torch':v,g,mask=map(torch.as_tensor,(v,g,mask))
    result=derivative(v,g,mask,0)
    np.testing.assert_allclose(result,0,atol=2e-15)
    # An in-band wave retains its analytic derivative.
    v=np.sin(x)[:,None,None]
    if backend=='torch':v=torch.as_tensor(v)
    np.testing.assert_allclose(derivative(v,g,mask,0),np.cos(x)[:,None,None],atol=3e-15)

def test_receipt_and_pause_contract():
    from acceptance import accept_receipt,elapsed_charge
    row=dict(id='Mn',attempt_id='new',identity='current',status='PASS',contract_verified_after=True)
    assert accept_receipt(row,'Mn','new','current',0)
    for field,value in [('attempt_id','old'),('identity','old'),('status','RUNNING'),('contract_verified_after',False)]:
        assert not accept_receipt({**row,field:value},'Mn','new','current',0)
    assert not accept_receipt(row,'Mn','new','current',-15)
    assert elapsed_charge(86400,False)==0 and elapsed_charge(.25,True)==0
    assert elapsed_charge(.25,False)==.25

def test_unmanifested_store_cannot_be_stamped(tmp_path):
    from h0rebuild.table_contract import ensure_store
    (tmp_path/'old.npz').write_bytes(b'untrusted old table')
    with pytest.raises(RuntimeError,match='Nonempty'):ensure_store(tmp_path)
    assert not (tmp_path/'source_contract.json').exists()

@pytest.fixture
def fixture_case(tmp_path):
    raw=Path('/home/mingkang_nt/codex/h0_cuda_random100_cell_gauge_v2_20260912/raw/SOC_mp-561353')
    if not raw.exists():pytest.skip('Liyue real fixture required')
    case=tmp_path/'case';(case/'OUT.ABACUS').mkdir(parents=True)
    for name in ['STRU','OUT.ABACUS/INPUT','OUT.ABACUS/running_scf.log']:shutil.copy2(raw/name,case/name)
    import production_io
    _,sd,_,_=production_io.load_case(raw)
    return case,sd

def test_magmom_equivalence(fixture_case):
    import production_io
    case,sd=fixture_case
    _,_,opts,_=production_io.load_case(case,prepared_species=sd)
    p=case/'STRU';p.write_text(p.read_text().replace(' mag ',' magmom '))
    _,_,got,_=production_io.load_case(case,prepared_species=sd)
    assert got['initial_moments_z']==opts['initial_moments_z']

@pytest.mark.parametrize('key,value',[('nelec','124'),('nelec_delta','1'),('dft_functional','lda')])
def test_unsupported_physics_rejected(fixture_case,key,value):
    import production_io,re
    case,sd=fixture_case;p=case/'OUT.ABACUS/INPUT';text=p.read_text()
    text=re.sub(r'^\s*'+key+r'\s+.*$', '',text,flags=re.M)+'\n'+key+' '+value+'\n';p.write_text(text)
    with pytest.raises(ValueError):production_io.load_case(case,prepared_species=sd)

def nacf_modules():
    root=Path(__file__).resolve().parents[2]/'nacf_overlay/dptb/nacf'
    pkg=types.ModuleType('review_nacf');pkg.__path__=[str(root)];sys.modules['review_nacf']=pkg
    annotation=types.ModuleType('dptb.data.interfaces.p2_table');annotation.RadialBlockTable=object
    with patch.dict(sys.modules,{'dptb.data.interfaces.p2_table':annotation}):
        return importlib.import_module('review_nacf.radial'),importlib.import_module('review_nacf.prepared')

def synthetic_table(radial):
    directions=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0],[0,-1.,0],[0,0,-1.]])
    base=radial._harmonics(1,torch.from_numpy(directions)).numpy()
    r=np.array([0.,.5,1.,2.,3.]);v=np.stack([np.diag([1.,2.,3.])*(1-x/3)**2 for x in r])
    return types.SimpleNamespace(distances=r,values=v,left_shells=(1,),right_shells=(1,),support_bohr=3.,
        _spline=CubicSpline(r,v,axis=0),_rotator=types.SimpleNamespace(directions=directions,_base={1:base}))

def test_nacf_rotation_identity_and_hit_validation(tmp_path):
    radial,prepared=nacf_modules();a=synthetic_table(radial);b=copy.deepcopy(a);b._rotator._base[1]*=1.01
    args=dict(device='cpu',dtype=torch.float64,backend='torch')
    prepared.cached_table(a,tmp_path,**args)
    cached=prepared.cached_table(b,tmp_path,**args);fresh=radial.TorchRadialBlockTable(b,**args)
    assert prepared.key_for(a)!=prepared.key_for(b)
    v=torch.tensor([[.3,.7,.2],[.8,-.4,.1]],dtype=torch.float64)
    torch.testing.assert_close(cached(v),fresh(v),rtol=0,atol=0)
    with pytest.raises(ValueError):prepared.cached_table(a,tmp_path,device='cpu',dtype=torch.float16,backend='torch')
    with pytest.raises(ValueError):prepared.cached_table(a,tmp_path,device='cpu',dtype=torch.float64,backend='bogus')

def test_nacf_concurrent_atomic_publication_and_corruption(tmp_path):
    radial,prepared=nacf_modules();source=synthetic_table(radial)
    def load(_):return prepared.cached_table(source,tmp_path,device='cpu',dtype=torch.float64,backend='torch')
    with ThreadPoolExecutor(max_workers=4) as pool:objects=list(pool.map(load,range(12)))
    for obj in objects:torch.testing.assert_close(obj.coefficients,objects[0].coefficients)
    p=tmp_path/(prepared.key_for(source)+'.pt');data=p.read_bytes();p.write_bytes(data[:-10]+b'corruption')
    with pytest.raises(ValueError,match='checksum'):load(0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_h0_private_resident_and_native_devices(tmp_path):
    import production_io
    from h0rebuild.offline import prepared_two_center,_resident
    from h0rebuild.models import SpeciesData
    from h0rebuild.scalar_upf import scalarize_upf
    from h0rebuild.pyabacus_integrals import PyAbacusTwoCenter
    from h0rebuild.precompiled import verify
    from h0rebuild.cuda_two_center import CUDATwoCenter,_C
    raw=Path('/home/mingkang_nt/codex/h0_cuda_random100_cell_gauge_v2_20260912/raw/nonSOC_db_seq_id_10868')
    _,sd,_,_=production_io.load_case(raw);sd={k:SpeciesData(v.orb,scalarize_upf(v.upf)) for k,v in sd.items()}
    binary=verify('_cuda_two_center')['binary_sha256']
    a=prepared_two_center(sd,store=tmp_path,prepare=True,device='cuda:0')
    symbol=next(iter(sd));pairs=[(symbol,symbol)]*3
    vectors=torch.tensor([[.1,.2,.3],[0.,0.,-1.],[.8,.9,.1]],dtype=torch.float64,device='cuda:0')
    expected=a.eval_two_center_batch(pairs,vectors)
    a.S_coeffs.zero_();a.sd[symbol].orb.channels[0].radial[:]=0
    with patch.object(CUDATwoCenter,'__init__',side_effect=AssertionError('No tabulation on hit')):
        b=prepared_two_center(sd,store=tmp_path,device='cuda:0')
        actual=b.eval_two_center_batch(pairs,vectors)
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
        _resident.clear();c=prepared_two_center(sd,store=tmp_path,device='cuda:0')
        for x,y in zip(c.eval_two_center_batch(pairs,vectors),expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
    # Real CPU integral oracle for the changed native boundary.
    ref=PyAbacusTwoCenter(sd,nspin=1)
    for k,v in enumerate(vectors.cpu().numpy()):
        rs,rt=ref.scalar_pair(symbol,symbol,np.zeros(3),v);n=rs.shape[0]
        np.testing.assert_allclose(actual[0][k,:n,:n].cpu(),rs,rtol=0,atol=1e-8)
        np.testing.assert_allclose(actual[1][k,:n,:n].cpu(),rt,rtol=0,atol=1e-8)
    noncontig=torch.zeros((3,6),device='cuda:0',dtype=torch.float64);noncontig[:,::2]=vectors
    if torch.cuda.device_count()>1:torch.cuda.set_device(1)
    ambient_result=b.eval_two_center_batch(pairs,vectors)
    for x,y in zip(ambient_result,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
    stream=torch.cuda.Stream(device=0)
    stream.wait_stream(torch.cuda.default_stream(0))
    with torch.cuda.stream(stream):result=b.eval_two_center_batch(pairs,noncontig[:,::2])
    stream.synchronize()
    for x,y in zip(result,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(ValueError):b.eval_two_center_batch(pairs[:1],vectors)
    with pytest.raises(ValueError):b.eval_two_center_batch([],torch.zeros(1,3))
    assert b.eval_two_center_batch([],torch.empty(0,3))[0].shape[0]==0
    # Direct native caller must reject bad layout, not silently read its strides.
    args=[torch.zeros(3,dtype=torch.int32,device='cuda:0')]*2+[b.S_coeffs,b.T_coeffs,b.S_index_map,b.T_index_map,b.index_map_strides,b.gaunt_table,b.gaunt_dims,b.orb_l,b.orb_zeta,b.orb_m,b.species_orb_offsets,b.dr,b.cutoff,b.nr,b.max_norb]
    with pytest.raises(RuntimeError):_C.eval_two_center_batch(noncontig[:,::2],*args)
    assert verify('_cuda_two_center')['binary_sha256']==binary
    torch.cuda.set_device(0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_local_grid_offline_spline_no_refit(fixture_case):
    from h0rebuild.cuda_local_grid import CudaPeriodicFFTGridAOCache
    _,sd=fixture_case;orb=next(iter(sd.values())).orb
    orb.metadata['offline_source_sha256']='test'
    orb.metadata['offline_spline_coefficients']=[CubicSpline(orb.r,c.radial,extrapolate=False).c for c in orb.channels]
    obj=CudaPeriodicFFTGridAOCache.__new__(CudaPeriodicFFTGridAOCache);obj._species_cache={};obj.device=torch.device('cuda:0')
    with patch('h0rebuild.cuda_local_grid.CubicSpline',side_effect=AssertionError('Runtime refit')):
        got=obj._get_species_table(types.SimpleNamespace(basis=orb))
    np.testing.assert_array_equal(got['spline_coeffs'].cpu(),np.asarray(orb.metadata['offline_spline_coefficients']))

def test_polarized_threshold_matches_direct_libxc():
    pylibxc=pytest.importorskip('pylibxc')
    from h0rebuild.spin_fields import _derivatives
    # Minority spin crossing the physical cutoff is not equivalent to only
    # masking the final total density. Compare actual LibXC entry points.
    rho=np.array([[2e-7,2e-7],[2e-7,2e-5],[3e-6,4e-6],[.1,.2]])
    sigma=np.array([[1e-12,0,1e-12]]*4)
    expected_r=np.zeros_like(rho);expected_s=np.zeros_like(sigma)
    for name in ('GGA_X_PBE','GGA_C_PBE'):
        f=pylibxc.LibXCFunctional(name,'polarized');f.set_dens_threshold(1e-6)
        answer=f.compute({'rho':rho.ravel(),'sigma':sigma.ravel()},do_vxc=True)
        r=answer['vrho'].reshape(-1,2);s=answer['vsigma'].reshape(-1,3)
        if name=='GGA_C_PBE':
            keep=(rho>=1e-6)&(np.sqrt(np.abs(sigma[:,[0,2]]))>=1e-10)
            r=r*keep;s=s*np.column_stack([keep[:,0],keep[:,0]&keep[:,1],keep[:,1]])
        expected_r+=r;expected_s+=s
    r,s=_derivatives(rho,sigma,1e-6)
    np.testing.assert_array_equal(r,expected_r);np.testing.assert_array_equal(s,expected_s)

@pytest.mark.skipif(sys.platform!='linux',reason='POSIX stopped-child semantics')
def test_real_stopped_worker_does_not_exhaust_budget():
    import os,signal,subprocess,time
    from acceptance import wait_active,process_state
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'],start_new_session=True)
    try:
        os.kill(child.pid,signal.SIGSTOP)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(wait_active,child,.75)
            time.sleep(1.5)
            assert process_state(child.pid)['state'] in ('T','t')
            assert not future.done()
            os.kill(child.pid,signal.SIGCONT)
            rc,timeout,spent,excluded=future.result(timeout=5)
        assert rc==0 and not timeout and excluded>=1.
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
