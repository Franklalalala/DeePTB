import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

spec=importlib.util.spec_from_file_location('owner',Path(__file__).with_name('allocation_owner.py'))
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)

def test_cleanup_other_lease_and_exact_cwd(tmp_path,monkeypatch):
    path=Path('/tmp')/os.environ['USER']/('serial_fixture_'+str(os.getpid()))
    path.mkdir(exist_ok=False)
    lock=tmp_path/'stable.lock';lease=tmp_path/(path.name+'.self.lease')
    other=tmp_path/(path.name+'.other.lease');lease.touch();other.touch()
    assert owner.cleanup(path,path,lease,lock)=='retained-lease'
    other.unlink()
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],cwd=path)
    try:
        assert owner.cleanup(path,path,lease,lock)=='retained-consumer-or-owner'
    finally:child.terminate();child.wait()
    # PBS may hide other same-user /proc entries; the real path conservatively
    # retains those. Exercise deletion separately with a proven-idle fixture.
    monkeypatch.setattr(owner,'live_consumer',lambda p:False)
    assert owner.cleanup(path,path,lease,lock)=='cleaned'
    assert lock.exists()

def test_owner_stage_failure_and_term(tmp_path):
    for fail in (True,False):
        root=tmp_path/str(fail);root.mkdir()
        job='fixture_'+str(os.getpid())+'_'+str(fail)
        local=Path('/tmp')/os.environ['USER']/('h0_serial_'+job)
        (root/'stage_runtime.sh').write_text('exit '+('7' if fail else '0')+'\n')
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('allocation_owner.py')),'--root',str(root)],env=dict(os.environ,PBS_JOBID=job+'.fixture'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if not fail:
            deadline=time.time()+15
            while not (root/job/'stage.rc').exists():
                assert time.time()<deadline
                time.sleep(.1)
            p.send_signal(signal.SIGTERM)
        stdout=p.communicate(timeout=20)[0]
        assert p.returncode==(7 if fail else 143),stdout
        record=json.loads((root/job/'owner_exit.json').read_text())
        assert record['cleanup'] in {'cleaned','retained-consumer-or-owner'},record
        if record['cleanup']=='cleaned':assert not local.exists()
        else:
            assert local.exists()
            local.rmdir()  # fixture is empty; never recursively remove here
