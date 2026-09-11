"""Own serial-training children and local staging while preserving a PBS allocation.

Commands are task-local JSON files written by the operator. No log-grep watchdog
or step-triggered signals are used. The training CLI owns its max_steps boundary.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time


def live_consumer(path):
    prefix = str(path.resolve()) + '/'
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid():
                continue
            if prefix in (proc / 'maps').read_text():
                return True
            cwd = str((proc / 'cwd').resolve())
            if cwd == str(path.resolve()) or cwd.startswith(prefix):
                return True
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target == str(path.resolve()) or target.startswith(prefix):
                        return True
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            return True
    return False


def cleanup(path, expected, lease, lock_path, identity=None):
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if lease.exists():
            lease.unlink()
        if not path.exists():
            return 'absent'
        if path.is_symlink() or path.resolve() != expected or expected.parent.parent != Path('/tmp'):
            return 'retained-unregistered'
        stat = path.stat()
        if identity is not None and (stat.st_dev, stat.st_ino, stat.st_uid) != identity:
            return 'retained-replaced'
        if list(lock_path.parent.glob(path.name + '.*.lease')):
            return 'retained-lease'
        if path.stat().st_uid != os.getuid() or live_consumer(path):
            return 'retained-consumer-or-owner'
        shutil.rmtree(path)
        return 'cleaned'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    job = os.environ['PBS_JOBID'].split('.')[0]
    account = os.environ['USER']
    local = Path('/tmp') / account / ('h0_serial_' + job)
    expected = local.resolve()
    control = root / job
    control.mkdir(parents=True, exist_ok=True)
    locks = local.parent / '.serial_locks'
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / (local.name + '.lock')
    lease = locks / (local.name + '.' + str(os.getpid()) + '.lease')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        local.mkdir(parents=True, exist_ok=True)
        stat = local.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_uid)
        registration = dict(pid=os.getpid(), job=job, root=str(expected), time=time.time(),
                            node=socket.gethostname(), start_time=Path('/proc/self/stat').read_text().split()[21],
                            identity=identity)
        lease.write_text(json.dumps(registration))
    child = None
    stopping = False
    stop_signal = None
    rc = 0
    def stop(sig, frame):
        nonlocal stopping, stop_signal
        stopping = True
        stop_signal = sig
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def run(command, tag):
        nonlocal child
        with (control / (tag + '.log')).open('w') as log:
            env = dict(os.environ, TEMP=str(local), TMP=str(local), TMPDIR=str(local))
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            (control / 'child.json').write_text(json.dumps(dict(pid=child.pid, tag=tag, command=command)))
            while child.poll() is None:
                if stopping:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                time.sleep(1)
            result = child.wait()
        (control / (tag + '.rc')).write_text(str(result))
        return result
    (control / 'owner.json').write_text(json.dumps(dict(**registration, local=str(local), lease=str(lease))))
    try:
        rc = run(['bash', str(root / 'stage_runtime.sh')], 'stage')
        if rc:
            print('RUNTIME_STAGE_FAILED', rc, flush=True)
            return 128 - rc if rc < 0 else rc
        print('OWNER_READY', job, flush=True)
        seen = set()
        while not stopping:
            request = control / 'request.json'
            if request.exists():
                task = json.loads(request.read_text())
                tag = task['tag']
                if tag not in seen:
                    seen.add(tag)
                    mode = task['mode']
                    if mode not in {'tests', 'smoke', 'production'}:
                        raise ValueError('unknown mode')
                    rc = run(['python3', str(root / 'node_mode.py'), mode], tag)
                    print('COMMAND_DONE', job, tag, rc, flush=True)
            time.sleep(2)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            result = cleanup(local, expected, lease, lock_path, identity)
        except Exception as exc:
            result = 'retained-cleanup-error: ' + repr(exc)
        (control / 'owner_exit.json').write_text(json.dumps(dict(rc=rc, cleanup=result)))
        print('OWNER_EXIT', rc, result, flush=True)
    return 128 + stop_signal if stop_signal is not None else (128 - rc if rc < 0 else rc)


if __name__ == '__main__':
    raise SystemExit(main())
