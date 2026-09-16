"""Check raw named slots, edge identity and the actual selected DeePTB loader."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--samples',type=int,default=3);p.add_argument('--output',required=True);a=p.parse_args()
 import numpy as np,lmdb,pickle,zstandard,torch
 from dptb.utils.argcheck import normalize
 from dptb.entrypoints.multi_train import collect_cutoffs
 from dptb.data.build import build_dataset
 d=normalize(json.loads(Path(a.input).read_text()));data=d['data_options']['train'];prior=d['model_options']['embedding']['prior_kind'];checks=[]
 def decode(b):return pickle.loads(zstandard.ZstdDecompressor().decompress(b[4:]) if b[:4]==b'ZST1' else b)
 for path in sorted(Path(data['root']).rglob('data*.lmdb')):
  env=lmdb.open(str(path),readonly=True,lock=False,readahead=False)
  with env.begin() as tx:
   for key,blob in tx.cursor():
    r=decode(blob)
    for k in ['node_features','edge_features','node_h0','edge_h0','edge_index','edge_cell_shift']:
     assert k in r and np.isfinite(np.asarray(r[k])).all(),(path,key.hex(),k)
    assert np.asarray(r['edge_index']).shape[1]==np.asarray(r['edge_features']).shape[0]
    shifts=np.asarray(r['edge_cell_shift']);assert np.allclose(shifts,np.rint(shifts),atol=1e-8,rtol=0)
    if prior=='na_cf':
     for q in ['node_p23','edge_p2','node_delta_nacf','edge_delta_nacf']:assert q in r
     for pre,k in [('node','node_p23'),('edge','edge_p2')]:
      h=np.asarray(r[pre+'_features'],float)+np.asarray(r[pre+'_h0'],float);hp=np.asarray(r[k],float)+np.asarray(r[pre+'_delta_nacf'],float)
      assert np.max(np.abs(h-hp))<=2e-5
    checks.append({'shard':str(path),'key':key.hex()})
    if len(checks)>=a.samples:break
  env.close()
  if len(checks)>=a.samples:break
 assert checks,'No LMDB records found'
 common=dict(d['common_options']);common['device']='cpu';ds=build_dataset(**collect_cutoffs(d),**data,**common)
 for i in range(min(a.samples,len(ds))):
  raw=ds._load_data_dict(i);graph=ds[i]
  for prefix in ['node','edge']:
   k=prefix+'_features';expected=prefix+('_delta_nacf' if data['target_kind']=='nacfres' else '_features')
   assert torch.isfinite(graph[k]).all()
   assert torch.equal(graph[k],torch.as_tensor(raw[expected],dtype=graph[k].dtype)),('wrong target slot',i,k,expected)
   pk=prefix+'_h0' if prior=='h0' else ('node_p23' if prefix=='node' else 'edge_p2')
   assert torch.equal(graph[pk],torch.as_tensor(raw[pk],dtype=graph[pk].dtype)),('wrong prior slot',i,pk)
  assert torch.equal(graph['edge_index'],torch.as_tensor(raw['edge_index']))
 Path(a.output).write_text(json.dumps({'status':'SCHEMA_AND_LOADER_PASS','count':len(ds),'samples':checks,'target':data['target_kind'],'prior':prior,'dptb':__import__('dptb').__file__},indent=2))
if __name__=='__main__':main()
