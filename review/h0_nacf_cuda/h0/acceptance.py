"""Finite acceptance with isolated attempts and pause-aware active budgets."""
import argparse, hashlib, json, os, signal, subprocess, sys, threading, time, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
TERMINAL=('PASS','NUMERICAL_FAIL')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_text(json.dumps(data,indent=2)+'\n');os.replace(pending,path)

def shared_contract():
    paths=[p for p in ROOT.rglob('*') if p.is_file() and
           ((p.parent==ROOT and p.suffix=='.py') or
            p.relative_to(ROOT).parts[0] in ('h0rebuild','csrc','prebuilt') and
            p.suffix in ('.py','.cpp','.cu','.cuh','.h','.so','.json'))]
    files={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(paths)}
    for name in ('catalog.json','source_contract.json'):
        files['offline_tables/'+name]=file_hash(ROOT/'offline_tables'/name)
    return {'schema':4,'files':files,'Hmax_eV':.005,'Smax':1e-6,
            'runtime_compilation':False,'runtime_tabulation':False,
            'assembly':'CUDA FP64 offline, PBE charge PW projection, nonlocal_complete, no hermitization'}

def case_contract(cid,shared,cat):
    folder=Path(cat['raw'])/cid
    names=['STRU','OUT.ABACUS/INPUT','OUT.ABACUS/running_scf.log',
           'OUT.ABACUS/data-HR0_SPIN0.csr','OUT.ABACUS/data-SR-sparse_SPIN0.csr']
    return {'shared':digest(shared),'case':cid,'entry':cat['cases'][cid],
            'inputs':{name:file_hash(folder/name) for name in names}}

def accept_receipt(row,cid,attempt,identity,returncode):
    return (returncode==0 and row.get('id')==cid and row.get('attempt_id')==attempt and
            row.get('identity')==identity and row.get('status') in TERMINAL and
            row.get('contract_verified_after') is True)

def process_state(pid):
    try:
        data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':data[0],'start_ticks':data[19]}
    except FileNotFoundError:return None

def elapsed_charge(delta,stopped):
    # An unobserved supervisor pause extends, never expires, the active budget.
    return 0. if stopped or delta>2. else delta

def wait_active(child,budget):
    spent=0.;previous=time.monotonic();excluded=0.
    while child.poll() is None:
        time.sleep(.25);now=time.monotonic();dt=now-previous;previous=now
        state=process_state(child.pid)
        charge=elapsed_charge(dt,bool(state and state['state'] in ('T','t')))
        spent+=charge;excluded+=dt-charge
        if spent>=budget:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            return child.returncode,True,spent,excluded
    return child.returncode,False,spent,excluded

def run_worker(cid,attempt_dir):
    p=Path(attempt_dir);lease=json.loads((p/'attempt.json').read_text());out=p/(cid+'.json')
    expected=lease['identity'];base={'id':cid,'attempt_id':lease['attempt_id'],'identity':expected}
    write(out,{**base,'status':'STARTING'})
    try:
        cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
        if digest(case_contract(cid,shared_contract(),cat))!=expected:raise RuntimeError('Contract changed before attempt')
        import new100
        new100.RUN=p;new100.worker(cid)
        row=json.loads(out.read_text())
        if digest(case_contract(cid,shared_contract(),cat))!=expected:raise RuntimeError('Contract changed during attempt')
        row.update(**base,contract_verified_after=True);write(out,row)
        return 0 if row['status'] in TERMINAL else 1
    except BaseException:
        import traceback
        write(out,{**base,'status':'ERROR','error':traceback.format_exc()});return 1

def run_attempt(cid,gpu,run,contract,budget):
    import fcntl
    identity=digest(contract);case_dir=run/'cases'/cid;case_dir.mkdir(parents=True,exist_ok=True)
    committed=case_dir/'accepted.json'
    if committed.exists():
        ref=json.loads(committed.read_text());p=case_dir/ref['attempt_id'];row=json.loads((p/(cid+'.json')).read_text())
        execution=json.loads((p/'execution.json').read_text())
        if accept_receipt(row,cid,ref['attempt_id'],identity,execution['returncode']) and not execution['timed_out']:
            return {'id':cid,'status':row['status'],'attempt_id':ref['attempt_id'],'reused':True}
    for p in case_dir.glob('*/attempt.json'):
        old=json.loads(p.read_text());live=process_state(old.get('pid',-1))
        if live and live['state']!='Z' and live['start_ticks']==old.get('start_ticks'):
            raise RuntimeError(f'Existing worker {old["pid"]} for {cid}; reconcile before restart')
    attempt=uuid.uuid4().hex;p=case_dir/attempt;p.mkdir()
    lease={'case':cid,'attempt_id':attempt,'identity':identity,'contract':contract,'gpu':gpu,'created':time.time()}
    write(p/'attempt.json',lease)
    lock_dir=Path(os.environ.get('H0_GPU_LOCK_DIR',str(ROOT/'gpu_locks')));lock_dir.mkdir(exist_ok=True)
    with (lock_dir/f'gpu{gpu}.lock').open('a') as lock:
        # Queue time does not consume the calculation budget. The worker keeps
        # the inherited lock even if its dispatcher exits.
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (p/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',cid,'--attempt-dir',str(p)],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
            state=process_state(child.pid)
            lease.update(pid=child.pid,start_ticks=state['start_ticks'] if state else None,started=time.time())
            write(p/'attempt.json',lease)
            rc,timeout,active,excluded=wait_active(child,budget)
        execution={'returncode':rc,'timed_out':timeout,'active_seconds':active,'paused_or_unobserved_seconds':excluded,'finished':time.time()}
        write(p/'execution.json',execution)
    out=p/(cid+'.json');row=json.loads(out.read_text()) if out.exists() else {}
    if timeout or not accept_receipt(row,cid,attempt,identity,rc):
        write(p/'rejected.json',{'reason':'Timeout, failed child or mismatched/incomplete receipt','execution':execution})
        return {'id':cid,'status':'ERROR','attempt_id':attempt}
    write(committed,{'attempt_id':attempt,'identity':identity})
    return {'id':cid,'status':row['status'],'attempt_id':attempt,'reused':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--attempt-dir')
    parser.add_argument('--ids',nargs='+');parser.add_argument('--budget',type=float,default=1800.)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1]);args=parser.parse_args()
    if args.worker:return run_worker(args.worker,args.attempt_dir)
    import fcntl
    shared=shared_contract();cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
    all_ids=sorted(p.name for p in Path(cat['raw']).iterdir() if (p/'STRU').exists())
    if len(all_ids)!=100 or len(set(all_ids))!=100:raise ValueError('Original cohort must contain exactly 100 structures')
    first=['SOC_mp-561353','nonSOC_db_seq_id_1773','SOC_mp-510294','nonSOC_db_seq_id_10868','nonSOC_db_seq_id_2251','nonSOC_db_seq_id_9278']
    order=args.ids or first+[x for x in all_ids if x not in first]
    if len(set(order))!=len(order) or not set(order)<=set(all_ids):raise ValueError('Invalid cohort selection')
    run=ROOT/'acceptance_v4'/digest(shared);run.mkdir(parents=True,exist_ok=True)
    with (run/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(run/'contract.json',shared)
        state={'status':'RUNNING','total':len(order),'completed':0,'cases':[],'run':str(run),
               'original_cohort':all_ids,'thresholds':{'Hmax_eV':.005,'Smax':1e-6}}
        write(ROOT/'acceptance_v4/current.json',{'run':str(run)});guard=threading.Lock()
        def lane(gpu,cohort):
            for cid in cohort:
                try:row=run_attempt(cid,gpu,run,case_contract(cid,shared,cat),args.budget)
                except Exception as exc:row={'id':cid,'status':'ERROR','error':repr(exc)}
                with guard:
                    state['cases'].append(row);state['completed']=len(state['cases']);write(run/'status.json',state)
                print(cid,row['status'],flush=True)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(lane,g,order[k::len(args.gpus)]) for k,g in enumerate(args.gpus)]
            for future in futures:future.result()
        state['status']='COMPLETE' if all(r['status'] in TERMINAL for r in state['cases']) else 'INCOMPLETE_ERRORS'
        write(run/'status.json',state)
    return 0 if state['status']=='COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
