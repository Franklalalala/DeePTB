"""Finite two-GPU validation of the current source; no legacy route execution."""
import os,sys,json,time,subprocess,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import new100
ROOT=Path(__file__).resolve().parent
new100.RUN=ROOT/'new100_v3';new100.RUN.mkdir(exist_ok=True)
RUN=new100.RUN
if len(sys.argv)>1:
    new100.worker(sys.argv[1]);sys.exit(0)
cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
ids=sorted(p.name for p in Path(cat['raw']).iterdir() if (p/'STRU').exists())
first=['nonSOC_db_seq_id_10868','nonSOC_db_seq_id_1773','SOC_mp-561353','SOC_mp-510294','nonSOC_db_seq_id_2251','nonSOC_db_seq_id_9278']
order=first+[x for x in ids if x not in first];assert len(order)==100 and len(set(order))==100
state={'status':'RUNNING','total':100,'completed':0,'cases':[], 'revision':'atomic-mag + per-species normalization + pseudo_rcut/v3',
       'old_version_execution':False,'thresholds':{'Hmax_eV':.005,'Smax':1e-6},'GPUs':[0,1],
       'FP64_parity':'Representative reviewed component tests; not newly measured for all 100'}
new100.write(RUN/'status.json',state)
def lane(gpu,cohort):
    results=[]
    lock=(ROOT/'gpu0.lock') if gpu==0 else ROOT.parent/'gpu1.lock'
    for cid in cohort:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
        with (RUN/(cid+'.log')).open('w') as log:
            child=subprocess.Popen(['flock',str(lock),sys.executable,__file__,cid],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        p=RUN/(cid+'.json');row=json.loads(p.read_text()) if p.exists() else {'id':cid,'status':'ERROR'}
        if row['status']=='RUNNING':row.update(status='ERROR',error='Worker timeout or process failure');new100.write(p,row)
        results.append(row);print(cid,row['status'],flush=True)
        # Derive one shared snapshot from terminal files, excluding partial writes.
        with guard:
            state['cases'].append({'id':cid,'status':row['status']});state['completed']=len(state['cases']);new100.write(RUN/'status.json',state)
    return results
import threading
guard=threading.Lock()
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(lane,g,order[g::2]) for g in (0,1)]
    for future in as_completed(futures):future.result()
state['status']='COMPLETE';new100.write(RUN/'status.json',state)
