import copy
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

def nacf(monkeypatch):
    from dptb.nacf import radial, prepared
    return radial, prepared


def source_table(radial):
    directions=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0],[0,-1.,0],[0,0,-1.]])
    r=np.array([0.,.5,1.,2.,3.]);v=np.stack([np.diag([1.,2.,3.])*(1-x/3)**2 for x in r])
    return types.SimpleNamespace(distances=r,values=v,left_shells=(1,),right_shells=(1,),support_bohr=3.,
        _spline=CubicSpline(r,v,axis=0),_rotator=types.SimpleNamespace(directions=directions,
        _base={1:radial._harmonics(1,torch.from_numpy(directions)).numpy()}))

@pytest.mark.parametrize('fault',['nan','inf','complex','shape','knots'])
def test_spline_semantics_cold_and_cache(monkeypatch,tmp_path,fault):
    radial,prepared=nacf(monkeypatch);source=source_table(radial)
    kw=dict(device='cpu',dtype=torch.float64,backend='torch')
    prepared.cached_table(source,tmp_path,**kw)
    if fault in ('nan','inf'):source._spline.c[0,0,0,0]=float(fault)
    elif fault=='complex':source._spline.c=source._spline.c.astype(complex)+1j
    elif fault=='shape':source._spline.c=source._spline.c[:2]
    else:source._spline.x=source.distances+.1
    with pytest.raises(ValueError,match='spline'):radial.TorchRadialBlockTable(source,**kw)
    with pytest.raises(ValueError,match='spline'):prepared.cached_table(source,tmp_path,**kw)

def test_comparison_empty_zero_and_missing():
    from h0rebuild.models import BlockKey
    from production_io import compare_blocks
    k=BlockKey(0,0,(0,0,0));h=BlockKey(0,0,(1,0,0));one={k:np.eye(1)}
    result=compare_blocks(one,one,[1]);assert result['total']['max_abs']==0
    assert result['hopping']['elements']==0 and result['hopping']['worst'] is None
    assert result['hopping']['max_abs'] is None
    zero=compare_blocks({k:np.zeros((1,1))},{k:np.zeros((1,1))},[1])
    assert zero['total']['mae_on_reference_nonzero'] is None
    missing=compare_blocks(one,{**one,h:np.ones((1,1))},[1])
    assert missing['support']['missing_reference_blocks']==1 and missing['hopping']['max_abs']==1
    with pytest.raises(ValueError,match='empty total'):compare_blocks({}, {}, [1])
    json.dumps(result,allow_nan=False);json.dumps(zero,allow_nan=False)

def test_dependency_change_and_relocation(monkeypatch,tmp_path):
    from h0rebuild.numerical_identity import capture
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    p=a/'numeric.py';q=b/'numeric.py';p.write_text('def numeric(x): return x*2\n');q.write_bytes(p.read_bytes())
    module=types.ModuleType('_numeric_probe');module.__file__=str(p);module.__version__='1'
    monkeypatch.setitem(sys.modules,'_numeric_probe',module)
    before=capture(('_numeric_probe',));module.__file__=str(q)
    relocated=capture(('_numeric_probe',))
    assert relocated['identity']==before['identity'] and relocated['provenance']!=before['provenance']
    stamp=q.stat();q.write_text('def numeric(x): return x*3\n');os.utime(q,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert capture(('_numeric_probe',))['identity']!=before['identity']

def test_deadline_exit_is_not_timeout():
    from acceptance import wait_active
    child=subprocess.Popen([sys.executable,'-S','-c','pass'],start_new_session=True)
    rc,timeout,_,_=wait_active(child,.2)
    assert rc==0 and not timeout

def test_environment_identity_ignores_library_mapping_order(monkeypatch):
    from h0rebuild.numerical_identity import capture
    from h0rebuild.offline import fingerprint
    original=Path.read_text
    before=capture(('numpy','scipy','torch'))['identity']
    def reversed_maps(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        return '\n'.join(reversed(text.splitlines())) if str(path)=='/proc/self/maps' else text
    monkeypatch.setattr(Path,'read_text',reversed_maps)
    after=capture(('numpy','scipy','torch'))['identity']
    assert fingerprint(before)==fingerprint(after)

def worker_script(path):
    root=str(Path(__file__).resolve().parents[1])
    script=path/'probe_worker.py'
    script.write_text("import argparse,os,sys,time\nfrom pathlib import Path\nsys.path.insert(0,"+repr(root)+")\n"
        "from lifecycle import register_worker\np=argparse.ArgumentParser();p.add_argument('--ready-fd',type=int);p.add_argument('--start-fd',type=int);p.add_argument('--attempt-dir');p.add_argument('--delay',type=float,default=0);p.add_argument('--sleep',type=float,default=.05);a=p.parse_args()\n"
        "d=Path(a.attempt_dir);(d/'seen_pid').write_text(str(os.getpid()));time.sleep(a.delay);register_worker(d,a.ready_fd,a.start_fd);(d/'started').write_text('yes');time.sleep(a.sleep);(d/'completed').write_text('yes')\n")
    return script

@pytest.mark.parametrize('duration,budget,expected',[(.05,2,False),(2,.15,True)])
def test_registered_execution(tmp_path,duration,budget,expected):
    from lifecycle import run_registered,write
    script=worker_script(tmp_path);write(tmp_path/'attempt.json',{'attempt_id':'probe','identity':'probe'})
    with (tmp_path/'lock').open('a') as f:
        execution=run_registered([sys.executable,'-S',str(script),'--attempt-dir',str(tmp_path),'--sleep',str(duration)],tmp_path,os.environ.copy(),[f.fileno()],budget,10)
    assert execution['timed_out']==expected and execution['registered']
    assert (execution['returncode']==0)==(not expected)

def test_case_lock_independent_of_gpu(tmp_path,monkeypatch):
    import fcntl
    import acceptance
    monkeypatch.setenv('H0_CASE_LOCK_DIR',str(tmp_path/'locks'));(tmp_path/'locks').mkdir()
    contract={'probe':True};name=acceptance.digest({'case':'X','identity':acceptance.digest(contract)})+'.lock'
    with (tmp_path/'locks'/name).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for gpu in (0,1):
            with pytest.raises(RuntimeError,match='owns case'):acceptance.run_attempt('X',gpu,tmp_path/'run',contract,2)

@pytest.mark.parametrize('window',['before_register','after_go'])
def test_dispatcher_crash_keeps_lock_and_worker_converges(tmp_path,window):
    import fcntl
    from lifecycle import process_state,write
    root=str(Path(__file__).resolve().parents[1]);worker=worker_script(tmp_path)
    write(tmp_path/'attempt.json',{'attempt_id':'crash','identity':'crash'})
    command=[sys.executable,'-S',str(worker),'--attempt-dir',str(tmp_path),'--sleep','20','--delay','1' if window=='before_register' else '0']
    supervisor=tmp_path/'supervisor.py'
    supervisor.write_text('import sys,os,fcntl\nsys.path.insert(0,'+repr(root)+')\nfrom lifecycle import run_registered\n'
        +'with open('+repr(str(tmp_path/'case.lock'))+',"a") as f:\n fcntl.flock(f,fcntl.LOCK_EX)\n run_registered('+repr(command)+','+repr(str(tmp_path))+',os.environ.copy(),[f.fileno()],.4,10)\n')
    parent=subprocess.Popen([sys.executable,'-S',str(supervisor)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        marker=tmp_path/('seen_pid' if window=='before_register' else 'started');deadline=time.monotonic()+10
        while not marker.exists():
            assert time.monotonic()<deadline and parent.poll() is None;time.sleep(.02)
        pid=int((tmp_path/'seen_pid').read_text());os.kill(parent.pid,signal.SIGKILL);parent.wait()
        with (tmp_path/'case.lock').open('a') as f:
            with pytest.raises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+8
        while (state:=process_state(pid)) and state['state']!='Z':
            assert time.monotonic()<deadline;time.sleep(.05)
        with (tmp_path/'case.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (tmp_path/'completed').exists()
        if window=='before_register':assert not (tmp_path/'started').exists()
        else:
            while not (tmp_path/'watchdog.json').exists():
                assert time.monotonic()<deadline;time.sleep(.02)
            assert json.loads((tmp_path/'watchdog.json').read_text())['timed_out']
    finally:
        if parent.poll() is None:os.killpg(parent.pid,signal.SIGKILL);parent.wait()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_private_copy_survives_eviction_on_side_stream():
    from h0rebuild.offline import private_copy
    def run(device):
        with torch.cuda.device(device):
            producer=torch.cuda.Stream();consumer=torch.cuda.Stream();copies=[]
            for index in range(8):
                with torch.cuda.stream(producer):master={'nested':[torch.full((1024*1024,),float(index),device=device)]}
                consumer.wait_stream(producer)
                with torch.cuda.stream(consumer):
                    torch.cuda._sleep(20000000)
                    copies.append(private_copy(master)['nested'][0])
                del master
                with torch.cuda.stream(producer):
                    trash=[torch.full((1024*1024,),-99.,device=device) for _ in range(8)]
                del trash
            consumer.synchronize()
            for index,value in enumerate(copies):assert torch.equal(value,torch.full_like(value,float(index)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,['cuda:'+str(i) for i in range(min(2,torch.cuda.device_count()))]))

@pytest.mark.parametrize('wall_timeout',[False,True])
def test_watchdog_survives_dispatcher_and_worker_pause(tmp_path,wall_timeout):
    from lifecycle import write
    root=str(Path(__file__).resolve().parents[1]);worker=worker_script(tmp_path)
    write(tmp_path/'attempt.json',{'attempt_id':'pause','identity':'pause'})
    command=[sys.executable,'-S',str(worker),'--attempt-dir',str(tmp_path),'--sleep','.3']
    supervisor=tmp_path/'supervisor.py'
    supervisor.write_text('import sys,os\nsys.path.insert(0,'+repr(root)+')\nfrom lifecycle import run_registered,write\n'
        +'with open('+repr(str(tmp_path/'lock'))+',"a") as f:\n result=run_registered('+repr(command)+','+repr(str(tmp_path))+',os.environ.copy(),[f.fileno()],2,'+('0.5' if wall_timeout else '10')+')\n write('+repr(str(tmp_path/'execution.json'))+',result)\n')
    parent=subprocess.Popen([sys.executable,'-S',str(supervisor)],start_new_session=True)
    pid=None
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'started').exists():
            assert time.monotonic()<deadline and parent.poll() is None;time.sleep(.01)
        pid=int((tmp_path/'seen_pid').read_text())
        os.kill(pid,signal.SIGSTOP);os.kill(parent.pid,signal.SIGSTOP);time.sleep(1.2)
        if wall_timeout:
            assert json.loads((tmp_path/'watchdog.json').read_text())['reason']=='WALL_TIMEOUT'
        else:
            assert not (tmp_path/'watchdog.json').exists();os.kill(pid,signal.SIGCONT)
        os.kill(parent.pid,signal.SIGCONT);assert parent.wait(timeout=10)==0
        result=json.loads((tmp_path/'execution.json').read_text())
        assert result['timed_out']==wall_timeout and result['paused_seconds']>.3
    finally:
        from lifecycle import signal_group,stop_child
        if pid is not None:signal_group(pid,signal.SIGCONT)
        signal_group(parent.pid,signal.SIGCONT);stop_child(parent)

def local_inputs(device='cuda:0'):
    def t(x,dtype):return torch.tensor(x,dtype=dtype,device=device)
    coeff=torch.zeros((1,4,3),dtype=torch.float64,device=device);coeff[:,3,:]=1
    return [t([0.,0,0],torch.float64),.3,t([-1,-1,-1],torch.int64),t([3,3,3],torch.int32),
        t([2,2,2],torch.int64),torch.eye(3,dtype=torch.float64,device=device),.1,3,1,coeff,1,
        t([[0,0,0]],torch.int32),torch.ones((2,2,2),device=device,dtype=torch.float64),False,
        torch.empty(0,device=device,dtype=torch.float64)]

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('fault',['negative_ch','large_ch','capacity','intmin_m','counts','shape','device','stride'])
def test_local_native_rejects_before_launch(fault):
    from h0rebuild.precompiled import load
    native=load('_cuda_local_grid');args=local_inputs()
    if fault=='negative_ch':args[11][0,0]=-1
    elif fault=='large_ch':args[11][0,0]=1
    elif fault=='capacity':args[8]=17
    elif fault=='intmin_m':args[11][0,2]=-2**31
    elif fault=='counts':args[3][:]=2**30
    elif fault=='shape':args[5]=torch.zeros(2,3,device='cuda:0',dtype=torch.float64)
    elif fault=='device':args[11]=args[11].cpu()
    else:args[12]=torch.ones(2,2,4,device='cuda:0',dtype=torch.float64)[:,:,::2]
    with pytest.raises(RuntimeError):native.build_atom_support_cuda(*args)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_local_native_valid_empty_pair_and_device():
    from h0rebuild.precompiled import load
    native=load('_cuda_local_grid');args=local_inputs()
    if torch.cuda.device_count()>1:torch.cuda.set_device(1)
    stream=torch.cuda.Stream(device=0)
    stream.wait_stream(torch.cuda.default_stream(0))
    with torch.cuda.stream(stream):anchor=native.build_atom_support_cuda(*args)
    stream.synchronize();torch.cuda.set_device(0)
    assert anchor[1].shape==(1,1)
    torch.testing.assert_close(anchor[1],torch.full_like(anchor[1],1/np.sqrt(4*np.pi)))
    empty=local_inputs();empty[3].zero_();out=native.build_atom_support_cuda(*empty)
    assert out[1].shape==(0,1)
    lo,hi,lookup=anchor[4:7];values,potential=anchor[1:3]
    i=torch.tensor([0],device='cuda:0',dtype=torch.int32);j=i.clone()
    pairs=[[lo],[hi],[lookup],[values],[potential],False,[],i,j,torch.zeros(1,3,device='cuda:0',dtype=torch.int64),args[4],.125,[1]]
    result=native.contract_pairs_batch_cuda(*pairs)[0][0]
    torch.testing.assert_close(result,torch.full_like(result,.125/(4*np.pi)))
    j[0]=-1
    with pytest.raises(RuntimeError,match='pair index'):native.contract_pairs_batch_cuda(*pairs)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_two_center_native_intmin_rejected():
    from h0rebuild.precompiled import load
    native=load('_cuda_two_center');dev='cuda:0'
    q=torch.zeros((1,3),device=dev,dtype=torch.float64);idx=torch.zeros(1,device=dev,dtype=torch.int32)
    coeff=torch.zeros(1,1,4,device=dev,dtype=torch.float64);mapping=torch.zeros((1,)*7,device=dev,dtype=torch.int32)
    gaunt=torch.zeros(25,25,81,device=dev,dtype=torch.float64);m=idx.clone();m[0]=-2**31
    args=[q,idx,idx,coeff,coeff,mapping,mapping,[1]*7,gaunt,[25,25,81],idx,idx,m,torch.tensor([0,1],device=dev,dtype=torch.int32),.1,.1,2,1]
    with pytest.raises(RuntimeError,match='harmonic m'):native.eval_two_center_batch(*args)
    m.zero_();s,t=native.eval_two_center_batch(*args);assert s.item()==0 and t.item()==0
