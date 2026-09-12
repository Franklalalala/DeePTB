import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from dptb.nacf.prepared import cached_table
from dptb.nacf.radial import TorchRadialBlockTable
from dptb.data.interfaces.p2_table import RadialBlockTable

def source():
    r=np.array([0.,.1,.3,.7,1.4,2.,3.])
    values=np.stack([np.diag(np.arange(1.,10.)*(1-x/3)**2) for x in r])
    return RadialBlockTable(r,values,(0,1,2),(0,1,2),3.)

def test_prepared_native_nondefault_stream_and_cache(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    table=source()
    fresh=cached_table(table,tmp_path,device='cuda',dtype=torch.float64,backend='cuda')
    def forbidden(*a,**k):raise AssertionError('Recompiled radial metadata or invoked compiler')
    monkeypatch.setattr(TorchRadialBlockTable,'__init__',forbidden)
    warm=cached_table(table,tmp_path,device='cuda',dtype=torch.float64,backend='cuda')
    assert warm.prepared_cache=='disk'
    real_popen=subprocess.Popen
    def no_compile(args,*a,**k):
        words=args if isinstance(args,(list,tuple)) else args.split()
        if any(Path(str(x)).name in ('nvcc','ninja','c++','g++','gcc') for x in words):forbidden()
        return real_popen(args,*a,**k)
    monkeypatch.setattr(subprocess,'Popen',no_compile)
    vectors=torch.tensor([[.3,.2,.7],[0,0,-1.],[1e-7,0,-1.],[0,0,0],[0,0,3.],[.1,.2,-.7]],device='cuda',dtype=torch.float64)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x=warm(vectors.clone());y=fresh(vectors.clone())
        warm.backend='torch';ref=warm(vectors.clone())
    stream.synchronize()
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(x,ref,atol=1e-9,rtol=1e-10)
    payload=next(tmp_path.glob('*.pt'));payload.write_bytes(payload.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):cached_table(table,tmp_path,device='cuda',dtype=torch.float64,backend='cuda')

def test_source_change_does_not_reuse_prepared_table(tmp_path):
    a=source();first=cached_table(a,tmp_path,device='cpu',dtype=torch.float64,backend='torch')
    b=source();b.support_bohr=2.9
    second=cached_table(b,tmp_path,device='cpu',dtype=torch.float64,backend='torch')
    assert len(list(tmp_path.glob('*.pt')))==2
    assert second.support_bohr!=first.support_bohr

def test_bank_cache_preserves_periodic_and_soc_contract(tmp_path):
    from dptb.tests.test_nacf_gpu import stores
    from dptb.nacf.assembly import NACFTableBank
    p2,p23=stores()
    def bank():return NACFTableBank(p2,p23,device='cuda',prepared_cache_dir=tmp_path)
    args=(['X','Y'],[[0,0,0],[1.3,.2,.1]],np.eye(3)*4.5,[[0,1],[1,0]],[[0,0,0],[0,0,0]])
    a=bank();x=a.prepare(*args)()
    b=bank();y=b.prepare(*args)()
    assert all(t.prepared_cache=='disk' for t in b.tables.values())
    for key in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
