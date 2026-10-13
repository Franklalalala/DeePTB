"""Registered workers with inherited single-instance locks and a separate watchdog.

The watchdog remains runnable when the dispatcher and worker are SIGSTOP'ed.
Active time is sampled at 50 ms; stopped intervals are excluded conservatively
only when both endpoints are stopped. Wall time includes all pauses. No long
polling gap is silently forgiven. Locks are released by descriptor lifetime,
never by LOCK_UN on an open description still inherited by a worker.
"""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid

def write(path, value):
    path=Path(path); pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)

def process_state(pid):
    try:
        row=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':row[0], 'start_ticks':row[19]}
    except FileNotFoundError:return None

def signal_group(pid, sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass

def stop_child(child):
    if child.poll() is not None:return
    signal_group(child.pid,signal.SIGTERM)
    signal_group(child.pid,signal.SIGCONT)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if child.poll() is None:signal_group(child.pid,signal.SIGKILL)
        child.wait()

def register_worker(attempt_dir, ready_fd, start_fd):
    """Publish own identity before the parent may authorize computation."""
    path=Path(attempt_dir);lease=json.loads((path/'attempt.json').read_text())
    state=process_state(os.getpid())
    record={**state,'pid':os.getpid(),'argv':sys.argv,'attempt_id':lease['attempt_id'],
            'identity':lease['identity'],'phase':'READY'}
    write(path/'worker.json',record)
    try:
        os.write(ready_fd,b'R');os.close(ready_fd)
        if not select.select([start_fd],[],[],120)[0] or os.read(start_fd,1)!=b'G':
            raise RuntimeError('Parent disappeared or did not authorize worker start')
    finally:os.close(start_fd)
    record['phase']='RUNNING';record['started_epoch']=time.time()
    record['started_monotonic']=time.monotonic();write(path/'worker.json',record)

def watch(pid, start_ticks, budget, wall_budget, output, startup_budget=120.):
    """Separate process: observes worker stop/exit even when dispatcher is paused."""
    path=Path(output);start=previous=time.monotonic();spent=excluded=0.
    prior=process_state(pid);timed_out=False;reason=None;started=None
    write(path.with_suffix('.ready.json'),{'pid':os.getpid(),'worker_pid':pid,
          'worker_start_ticks':start_ticks,'phase':'ARMED','startup_budget_seconds':startup_budget})
    while True:
        current=process_state(pid)
        if current is None or current['start_ticks']!=start_ticks or current['state']=='Z':break
        time.sleep(.05);now=time.monotonic();current=process_state(pid)
        if current is None or current['start_ticks']!=start_ticks or current['state']=='Z':break
        if started is None:
            worker=json.loads((path.parent/'worker.json').read_text())
            if worker.get('pid')!=pid or worker.get('start_ticks')!=start_ticks:
                raise RuntimeError('Watchdog start acknowledgement identity mismatch')
            if worker.get('phase')=='RUNNING':
                started=float(worker['started_monotonic'])
                if not start<=started<=now:raise RuntimeError('Invalid worker start timestamp')
        dt=0. if started is None else now-max(previous,started)
        previous=now
        stopped=prior and prior['state'] in ('T','t') and current['state'] in ('T','t')
        if stopped:excluded+=dt
        else:spent+=dt
        prior=current
        startup_timeout=started is None and now-start>=startup_budget
        if (started is not None and spent>=budget) or now-start>=wall_budget or startup_timeout:
            # Recheck identity and terminal state immediately before signalling.
            current=process_state(pid)
            if current is None or current['start_ticks']!=start_ticks or current['state']=='Z':break
            timed_out=True
            reason='WALL_TIMEOUT' if now-start>=wall_budget else ('STARTUP_TIMEOUT' if startup_timeout else 'ACTIVE_TIMEOUT')
            signal_group(pid,signal.SIGTERM);signal_group(pid,signal.SIGCONT)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                current=process_state(pid)
                if current is None or current['start_ticks']!=start_ticks or current['state']=='Z':break
                time.sleep(.05)
            else:signal_group(pid,signal.SIGKILL)
            break
    write(path,{'timed_out':timed_out,'reason':reason,'active_seconds':spent,
                'paused_seconds':excluded,'wall_seconds':time.monotonic()-start,
                'budget_seconds':budget,'wall_budget_seconds':wall_budget,
                'startup_seconds':(time.monotonic() if started is None else started)-start,
                'startup_budget_seconds':startup_budget,'phase_at_exit':'ARMED' if started is None else 'RUNNING',
                'sampling_seconds':.05,'worker_pid':pid,'worker_start_ticks':start_ticks})

def run_registered(command, attempt_dir, env, lock_fds, budget, wall_budget):
    path=Path(attempt_dir);ready_r,ready_w=os.pipe();start_r,start_w=os.pipe()
    child=guardian=None
    try:
        with (path/'worker.log').open('x') as log:
            child=subprocess.Popen(command+['--ready-fd',str(ready_w),'--start-fd',str(start_r)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=tuple(lock_fds)+(ready_w,start_r))
            os.close(ready_w);ready_w=None;os.close(start_r);start_r=None
            if not select.select([ready_r],[],[],120)[0] or os.read(ready_r,1)!=b'R':
                raise RuntimeError('Worker failed to register')
            worker=json.loads((path/'worker.json').read_text());lease=json.loads((path/'attempt.json').read_text())
            state=process_state(child.pid)
            if (worker['pid']!=child.pid or not state or state['start_ticks']!=worker['start_ticks']
                    or worker['attempt_id']!=lease['attempt_id'] or worker['identity']!=lease['identity']):
                raise RuntimeError('Worker handshake identity mismatch')
            guardian=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch',str(child.pid),
                '--start-ticks',worker['start_ticks'],'--budget',str(budget),'--wall-budget',str(wall_budget),
                '--output',str(path/'watchdog.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+120
            while not (path/'watchdog.ready.json').exists():
                if guardian.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Watchdog failed to start')
                time.sleep(.02)
            os.write(start_w,b'G');os.close(start_w);start_w=None
            while child.poll() is None:
                if guardian.poll() is not None and not (path/'watchdog.json').exists():
                    raise RuntimeError('Watchdog failed before worker completion')
                time.sleep(.05)
            guardian.wait(timeout=10)
            execution=json.loads((path/'watchdog.json').read_text())
            execution.update(returncode=child.returncode,finished=time.time(),registered=True)
            return execution
    finally:
        for fd in (ready_r,ready_w,start_r,start_w):
            if fd is not None:os.close(fd)
        if child is not None:stop_child(child)
        if guardian is not None:
            try:guardian.wait(timeout=10)
            except subprocess.TimeoutExpired:stop_child(guardian)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int,required=True)
    parser.add_argument('--start-ticks',required=True);parser.add_argument('--budget',type=float,required=True)
    parser.add_argument('--wall-budget',type=float,required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();watch(a.watch,a.start_ticks,a.budget,a.wall_budget,a.output)
