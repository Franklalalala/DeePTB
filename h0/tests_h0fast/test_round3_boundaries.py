"""Round-three audit regressions: allocation-free capacity, DSO identity, startup."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import pytest
import torch

def test_memory_budget_is_bound_to_acceptance_identity(tmp_path,monkeypatch):
 import acceptance
 from h0rebuild import numerical_identity
 from runtime_options import memory_budgets
 monkeypatch.setattr(acceptance,'ROOT',tmp_path)
 monkeypatch.setattr(numerical_identity,'execution',lambda:{'identity':{'test':True}})
 (tmp_path/'offline_tables').mkdir()
 for name in ('catalog.json','source_contract.json'):(tmp_path/'offline_tables'/name).write_text('{}')
 monkeypatch.delenv('H0_FIELD_MAX_WORK_MB',raising=False)
 monkeypatch.delenv('H0_STRUCTURE_FACTOR_MAX_WORK_MB',raising=False)
 old=acceptance.shared_contract();assert old['memory_budgets']['field_max_work_mb']==4096
 monkeypatch.setenv('H0_FIELD_MAX_WORK_MB','8192')
 new=acceptance.shared_contract();assert acceptance.digest(old)!=acceptance.digest(new)
 assert new['memory_budgets']==memory_budgets() and old['Hmax_eV']==new['Hmax_eV']==.005
 monkeypatch.setenv('H0_STRUCTURE_FACTOR_MAX_WORK_MB','12288')
 expanded=acceptance.shared_contract();assert acceptance.digest(expanded)!=acceptance.digest(new)
 assert expanded['memory_budgets']['structure_factor_max_work_mb']==12288
 for name in ('H0_FIELD_MAX_WORK_MB','H0_STRUCTURE_FACTOR_MAX_WORK_MB'):
  for value in ('nan','inf','0','-1'):
   monkeypatch.setenv(name,value)
   with pytest.raises(ValueError):acceptance.shared_contract()
  monkeypatch.delenv(name)

def mapping(p):
 s=p.stat();return f'1000-2000 r-xp 00000000 {os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x} {s.st_ino} {p}'

def test_library_collision_fails_closed_and_aliases_are_stable(tmp_path,monkeypatch):
 from h0rebuild import numerical_identity as ni
 a=tmp_path/'a/libopenblas_probe.so';b=tmp_path/'b/libopenblas_probe.so'
 a.parent.mkdir();b.parent.mkdir();a.write_bytes(b'A');b.write_bytes(b'B')
 lines=[mapping(a),mapping(b)];read=Path.read_text
 monkeypatch.setattr(Path,'read_text',lambda p,*x,**kw:'\n'.join(lines) if str(p)=='/proc/self/maps' else read(p,*x,**kw))
 for order in (lines[:],list(reversed(lines))):
  lines[:]=order
  with pytest.raises(RuntimeError,match='Conflicting loaded'):ni.capture(())
 b.write_bytes(b'A');lines[:]=[mapping(a),mapping(b),mapping(a)]
 first=ni.capture(());lines.reverse();assert ni.capture(())['identity']==first['identity']
 assert len(first['identity']['loaded_libraries'])==1
 a.write_bytes(b'A2')
 with pytest.raises(RuntimeError,match='Conflicting loaded'):ni.capture(())
 b.write_bytes(b'A2');assert ni.capture(())['identity']!=first['identity']
 lines[:]=[mapping(a),mapping(a)];calls=[];hash_file=ni.file_hash
 monkeypatch.setattr(ni,'file_hash',lambda p,**kw:(calls.append(str(p)),hash_file(p,**kw))[1])
 before=ni.loaded_libraries(r'/libopenblas[^/]*\.so');assert len(calls)==1
 moved=tmp_path/'moved';moved.mkdir();c=moved/a.name;c.write_bytes(a.read_bytes());lines[:]=[mapping(c)]
 after=ni.loaded_libraries(r'/libopenblas[^/]*\.so')
 assert before[0]==after[0] and before[1]!=after[1]

@pytest.mark.skipif(not torch.cuda.is_available(),reason='installed CUDA extension required')
@pytest.mark.parametrize('norb',[1,2,3,4])
def test_support_capacity_without_allocation(norb):
 from h0rebuild.precompiled import load
 native=load('_cuda_local_grid');limit=(2**31-1-256)//max(3,norb)
 native.validate_support_capacity(limit,norb)
 with pytest.raises(RuntimeError,match='capacity'):native.validate_support_capacity(limit+1,norb)
 with pytest.raises(RuntimeError):native.validate_support_capacity(-1,norb)
 with pytest.raises(RuntimeError):native.validate_support_capacity(0,0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='installed CUDA extension required')
def test_candidate_and_compacted_capacity_without_allocation():
 from h0rebuild.precompiled import load
 native=load('_cuda_local_grid');cap=2**31-1-256
 native.validate_candidate_capacity(cap//3)
 with pytest.raises(RuntimeError):native.validate_candidate_capacity(cap//3+1)
 with pytest.raises(RuntimeError):native.validate_candidate_capacity(800000000)
 # Real oblique case: candidate traversal is safe; a worst-case AO product
 # would reject, but a compacted spherical support can fit the output bound.
 native.validate_candidate_capacity(725*725*499)
 native.validate_support_capacity(12000000,25)
 with pytest.raises(RuntimeError):native.validate_support_capacity(cap//25+1,25)

def cpu_worker(tmp_path):
 root=str(Path(__file__).resolve().parents[1]);p=tmp_path/'worker.py'
 p.write_text('import argparse,sys,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(root)+')\nfrom lifecycle import register_worker\np=argparse.ArgumentParser();p.add_argument("--attempt-dir");p.add_argument("--ready-fd",type=int);p.add_argument("--start-fd",type=int);a=p.parse_args();register_worker(a.attempt_dir,a.ready_fd,a.start_fd);Path(a.attempt_dir,"computed").write_text("yes");time.sleep(.02)\n')
 return root,p

@pytest.mark.skipif(sys.platform!='linux',reason='POSIX lifecycle')
def test_parent_stopped_before_GO_does_not_spend_active_budget(tmp_path):
 from lifecycle import write,stop_child,signal_group
 root,worker=cpu_worker(tmp_path);write(tmp_path/'attempt.json',{'attempt_id':'preGO','identity':'preGO'})
 command=[sys.executable,'-S',str(worker),'--attempt-dir',str(tmp_path)]
 supervisor=tmp_path/'parent.py'
 supervisor.write_text('import sys,os,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(root)+')\nimport lifecycle\noriginal=os.write\ndef delay(fd,data):\n if data==b"G":\n  Path('+repr(str(tmp_path/'before_GO'))+').write_text("ready");time.sleep(.5)\n return original(fd,data)\nlifecycle.os.write=delay\nr=lifecycle.run_registered('+repr(command)+','+repr(str(tmp_path))+',os.environ.copy(),(),.15,5)\nlifecycle.write('+repr(str(tmp_path/'result.json'))+',r)\n')
 parent=subprocess.Popen([sys.executable,'-S',str(supervisor)],start_new_session=True)
 try:
  until=time.monotonic()+10
  while not (tmp_path/'before_GO').exists():
   assert time.monotonic()<until and parent.poll() is None;time.sleep(.01)
  os.kill(parent.pid,signal.SIGSTOP);time.sleep(.4)
  assert json.loads((tmp_path/'worker.json').read_text())['phase']=='READY'
  assert not (tmp_path/'computed').exists() and not (tmp_path/'watchdog.json').exists()
  os.kill(parent.pid,signal.SIGCONT);assert parent.wait(timeout=8)==0
  r=json.loads((tmp_path/'result.json').read_text());assert not r['timed_out'] and r['startup_seconds']>.4 and r['active_seconds']<.15
 finally:signal_group(parent.pid,signal.SIGCONT);stop_child(parent)

@pytest.mark.skipif(sys.platform!='linux',reason='POSIX lifecycle')
@pytest.mark.parametrize('kind',['startup','wall'])
def test_READY_has_separate_startup_and_wall_limits(tmp_path,kind):
 from lifecycle import watch,write,process_state,stop_child
 child=subprocess.Popen([sys.executable,'-S','-c','import time; time.sleep(10)'],start_new_session=True)
 try:
  ticks=process_state(child.pid)['start_ticks'];write(tmp_path/'worker.json',{'pid':child.pid,'start_ticks':ticks,'phase':'READY'})
  watch(child.pid,ticks,.01,.12 if kind=='wall' else 1.,tmp_path/'watchdog.json',startup_budget=.12 if kind=='startup' else 1.)
  child.wait(timeout=5);r=json.loads((tmp_path/'watchdog.json').read_text())
  assert r['active_seconds']==0 and r['reason']==('STARTUP_TIMEOUT' if kind=='startup' else 'WALL_TIMEOUT')
 finally:stop_child(child)
