"""Fresh S1 or S2 initialization, or an explicit within-stage restart."""
import argparse,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--s1-checkpoint');p.add_argument('--restart');p.add_argument('--smoke-steps',type=int,default=0);a=p.parse_args()
 import torch
 from dptb.entrypoints.multi_train import multi_train
 from dptb.nnops.multi_trainer import MultiTrainer
 d=json.loads(Path(a.input).read_text());e=d['model_options']['embedding'];assert torch.cuda.is_available()
 assert (e['num_experts'],e['top_k'],e['num_shared_experts'])==(256,1,0)
 assert not(a.restart and a.s1_checkpoint)
 if e['only2b']:assert not a.s1_checkpoint
 else:assert a.restart or a.s1_checkpoint,'S2 requires corresponding S1 checkpoint'
 if a.s1_checkpoint:
  ck=torch.load(a.s1_checkpoint,map_location='cpu',weights_only=False);c=ck['config'];ce=c['model_options']['embedding']
  assert ce['only2b'] and ce['method']==e['method']
  for k in ['num_experts','top_k','num_shared_experts','edge_router_prior_activate','edge_router_top1_mode','prior_kind','prior_node_key','prior_edge_key','h0_ao_cg']:assert ce.get(k)==e.get(k),('incompatible S1',k)
  assert c['common_options']['basis']==d['common_options']['basis']
  assert c['data_options']['train']['target_kind']==d['data_options']['train']['target_kind']
  assert c['train_options']['distance_ranges']==d['train_options']['distance_ranges']
  if not a.smoke_steps:assert ck['iteration']>=c['train_options']['max_steps'],'S1 not complete; production S2 requires completed S1'
  del ck
 out=Path(a.output).resolve()
 if not a.restart:out.mkdir(parents=True,exist_ok=False)
 else:assert out.is_dir()
 meta={'started':time.time(),'input':str(Path(a.input).resolve()),'s1':a.s1_checkpoint,'restart':a.restart,'device':torch.cuda.get_device_name(),'total_GiB':torch.cuda.get_device_properties(0).total_memory/2**30,'dptb':__import__('dptb').__file__,'smoke_steps':a.smoke_steps,'scheduler_id':os.environ.get('SLURM_JOB_ID')}
 (out/f'attempt_{int(time.time())}.json').write_text(json.dumps(meta,indent=2))
 class SmokeComplete(Exception):pass
 # A finite short run keeps the original WSD schedule and stops after a saved update.
 if a.smoke_steps:
  original=MultiTrainer.iteration;steps=[]
  def iteration(t,*args,**kw):
   before=t.iter;result=original(t,*args,**kw)
   if t.iter>before:
    loss=float(result);assert torch.isfinite(torch.tensor(loss))
    steps.append(int(before));(out/'SMOKE_PROGRESS.json').write_text(json.dumps({'successful_updates':len(steps),'committed_iteration':before,'loss':loss}))
    if len(steps)>=a.smoke_steps:
     from dptb.plugins.saver import Saver
     next_iter=t.iter
     try:
      t.iter=before;saver=next(x for x in t._registered_plugins if isinstance(x,Saver));saver.iteration()
     finally:t.iter=next_iter
     raise SmokeComplete()
   return result
  MultiTrainer.iteration=iteration
 try:multi_train(INPUT=a.input,init_model=a.s1_checkpoint,restart=a.restart,output=str(out),log_level=20,log_path=str(out/f'main.{int(time.time())}.log'))
 except SmokeComplete:pass
 from dptb.nn import top1_prior,so2_activation_routes
 routes=so2_activation_routes.STATS.snapshot()
 receipt={'status':'SMOKE_FINISHED' if a.smoke_steps else 'TRAIN_ENTRYPOINT_RETURNED','dispatch':dict(top1_prior.COUNTS),'so2_routes':routes,'so2_cuda_calls':sum(routes['calls'].get(r,0) for r in ('fused_p0','pack_scatter')),'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'checkpoint_verification_required':True}
 (out/'RUN_EXIT.json').write_text(json.dumps(receipt,indent=2))
 if not e['only2b']:assert receipt['dispatch'].get('grouped_cuda',0)>0 and receipt['so2_cuda_calls']>0,'selected CUDA route was not observed'
if __name__=='__main__':main()
