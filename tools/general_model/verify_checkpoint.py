import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--steps',required=True,type=int);p.add_argument('--output',required=True);a=p.parse_args()
 import torch
 path=Path(a.checkpoint);c=torch.load(path,map_location='cpu',weights_only=False);assert c['iteration']>=a.steps
 state=c['model_state_dict'];assert state
 for name,t in state.items():
  if torch.is_tensor(t) and (t.is_floating_point() or t.is_complex()):assert torch.isfinite(t).all(),name
 e=c['config']['model_options']['embedding'];assert (e['num_experts'],e['top_k'],e['num_shared_experts'])==(256,1,0)
 assert not any('weight_shared' in k or 'bias_shared' in k for k in state)
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 r={'status':'CHECKPOINT_FINITE','iteration':c['iteration'],'only2b':e['only2b'],'prior':e['prior_kind'],'target':c['config']['data_options']['train']['target_kind'],'sha256':h.hexdigest(),'path':str(path.resolve()),'optimizer_present':any('optimizer' in k for k in c),'scheduler_present':any('scheduler' in k for k in c),'does_not_prove_exact_sampler_resume':True}
 Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()
