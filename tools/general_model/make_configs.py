"""Generate all three independent prior/target routes; preserve scientific controls."""
import argparse,copy,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
ROUTES={'h0_h0res':('h0','h0res'),'nacf_nacfres':('na_cf','nacfres'),'nacf_h0res':('na_cf','h0res')}
def make(base,route,stage,root,max_samples=None):
 d=copy.deepcopy(base);prior,target=ROUTES[route];e=d['model_options']['embedding'];t=d['train_options'];data=d['data_options']['train']
 e.update(method='lem_moe_v3_edge_prior_2b',num_experts=256,top_k=1,num_shared_experts=0,edge_router_top1_mode='switch',edge_router_prior_activate=True,only2b=stage=='s1',two_b_seed_gnn=True,h0_ao_cg=True,prior_kind=prior,prior_node_key='node_h0' if prior=='h0' else 'node_p23',prior_edge_key='edge_h0' if prior=='h0' else 'edge_p2',so2_fusion_mode='streamed_m_major_cueq',mole_linear_mode='cublas_grouped')
 data.update(root=str(root),target_kind=target,prior_kind='p2' if prior=='h0' else 'na_cf',get_H0=prior=='h0',get_P2=prior!='h0',prefer_precomputed_h0=True,prefer_precomputed_p2=prior!='h0',require_prior_residual_rme_target=prior!='h0',residual_hamiltonian=False)
 t['dynamic_batch'].update(enabled=True,calibrate=True,max_cost=None,max_edge=None,calibration_quantile=.95,calibration_batches=128,min_samples=1,max_samples=96,oom_fallback=False)
 if max_samples is not None:
  assert 1<=max_samples<=96;t['dynamic_batch']['max_samples']=max_samples;t['batch_size']=max_samples
 t['skip_nonfinite_batch']=True;t['flow_options']['enabled']=False
 d['data_options'].pop('validation',None)
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--max-samples',type=int,help='Explicit deployment experiment; default preserves 96');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);manifest={}
 for route in ROUTES:
  for arm in ['onsite','hopping']:
   base=json.loads((P/'configs/base.json').read_text())
   base['train_options']['distance_ranges']=[[0., 1e-6]] if arm=='onsite' else [[1e-6, 10.]]
   base['train_options']['clip_last_expert_range']=arm=='onsite'
   for stage in ['s1','s2']:
    d=make(base,route,stage,a.data,a.max_samples);path=out/f'{route}.{arm}.{stage}.json';path.write_text(json.dumps(d,indent=2)+'\n');manifest[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'route':route,'stage':stage,'arm':arm,'max_samples':d['train_options']['dynamic_batch']['max_samples'],'validation':'generated_not_hanhai_gpu_validated'}
 (out/'CONFIG_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('generated',len(manifest),'configs')
if __name__=='__main__':main()
