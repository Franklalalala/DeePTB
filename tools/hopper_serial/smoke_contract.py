"""Actual CLI smoke and serial production using the same resolved configs."""
import json, os, sys, subprocess, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import faulthandler
faulthandler.enable()

R=Path('/scratch/sheng.lei/0912_h0_serial_edgemoe')
JOB=os.environ['PBS_JOBID'].split('.')[0]
CONTROL=R/JOB
LOCAL=Path('/tmp/sheng.lei/h0_serial_'+JOB)
ARM={'612077':'onsite','612078':'hopping'}[JOB]

def cost_item(i):
    return i,_COST_DS.get_dynamic_batch_cost_parts(i)

def cost_identity(ds):
    return [[Path(p).name,int(i)] for p,i in zip(ds._lmdb_path_map,ds.index_map)]

def prepare_cost_cache():
    global _COST_DS
    from dptb.data.build import build_dataset
    from dptb.utils.argcheck import normalize,get_cutoffs_from_model_options
    cfg=normalize(json.loads((CONTROL/'production_s1.json').read_text()))
    common=dict(cfg['common_options'],device='cpu')
    r,er,oer=get_cutoffs_from_model_options(cfg['model_options'])
    for split in ['train','validation']:
        path=CONTROL/f'{split}_costs.json'
        ds=build_dataset(**cfg['data_options'][split],r_max=r,er_max=er,oer_max=oer,**common)
        identity=cost_identity(ds)
        if path.exists() and json.loads(path.read_text())['identity']==identity:continue
        _COST_DS=ds
        print('COST_CACHE_BEGIN',split,len(ds),flush=True)
        values={}
        with multiprocessing.get_context('fork').Pool(12) as pool:
            for i,parts in pool.imap_unordered(cost_item,range(len(ds)),chunksize=32):
                values[str(i)]=parts
                if len(values)%4096==0:print('COST_CACHE_PROGRESS',split,len(values),flush=True)
        path.write_text(json.dumps(dict(identity=identity,parts=values)))
        print('COST_CACHE_COMPLETE',split,len(values),flush=True)

def inject_cost_cache(t):
    for split in ['train','validation']:
        ds=getattr(t,split+'_datasets',None)
        if ds is None:continue
        cached=json.loads((CONTROL/f'{split}_costs.json').read_text())
        assert cached['identity']==cost_identity(ds),'cost-cache dataset order differs'
        assert len(cached['parts'])==len(ds)
        ds._dynamic_batch_cost_parts_cache={int(k):v for k,v in cached['parts'].items()}
        print('COST_CACHE_LOADED',split,len(ds),flush=True)

def stage_data():
    cfg=json.loads((R/'configs'/f'{ARM}_s1.json').read_text())
    src=Path(cfg['data_options']['train']['root']); dst=LOCAL/'train'
    files=[Path(folder)/name for folder,dirs,names in os.walk(src,followlinks=True) for name in names]
    size=sum(p.stat().st_size for p in files)
    free=shutil.disk_usage(LOCAL).free
    marker=dst/'.STAGE_DONE'
    if marker.exists() and json.loads(marker.read_text()).get('bytes')==size:
        root=str(dst)
    elif free > size + 20*1024**3:
        print('FULL_STAGE_BEGIN',size,free,flush=True)
        def copy(p):
            out=dst/p.relative_to(src);out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,out)
            assert out.stat().st_size==p.stat().st_size
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy,files))
        marker.write_text(json.dumps(dict(source=str(src),bytes=size,files=len(files))))
        root=str(dst)
    else:
        print('PREAD_FALLBACK_INSUFFICIENT_LOCAL_DISK',size,free,flush=True)
        root=str(src)
    for stage in (1,2):
        cfg=json.loads((R/'configs'/f'{ARM}_s{stage}.json').read_text())
        cfg['data_options']['train']['root']=root
        (CONTROL/f'production_s{stage}.json').write_text(json.dumps(cfg,indent=2))
    print('DATA_RESOLVED',root,flush=True)

def checkpoint(out, steps):
    import torch
    matches=list((out/'checkpoint').glob('*.latest.pth'))
    if not matches:matches=list((out/'checkpoints').glob('*.latest.pth'))
    assert len(matches)==1, (out,matches)
    p=matches[0].resolve();d=torch.load(p,map_location='cpu',weights_only=False)
    assert d['iteration']==steps, (p,d['iteration'],steps)
    assert d['config']['model_options']['embedding']['only2b']
    return p

def worker():
    import torch
    from dptb.nnops.multi_trainer import MultiTrainer
    from dptb.nnops.objective import Objective
    from dptb.entrypoints.main import main
    if sys.argv[1]=='--cached-cli':
        original=MultiTrainer.run
        def cached_run(t,*a,**kw):
            inject_cost_cache(t)
            return original(t,*a,**kw)
        MultiTrainer.run=cached_run
        sys.argv=['dptb']+sys.argv[2:]
        return main()
    report={'arm':ARM,'steps':[],'mask_checks':[]}
    orig_run=MultiTrainer.run
    orig_obj=Objective.run
    def obj(self,**kw):
        d=kw['batch_copy'];n=int(kw['active_nodes']);e=int(kw['active_edges'])
        assert 'node_h0' in d and 'edge_h0' in d
        assert torch.isfinite(d['node_h0']).all() and torch.isfinite(d['edge_h0']).all()
        if ARM=='onsite':assert n>0 and e==0,(n,e)
        else:assert n==0 and e>0,(n,e)
        out=orig_obj(self,**kw)
        excluded='hopping' if ARM=='onsite' else 'onsite'
        if out.get(excluded) is not None:assert float(out[excluded])==0,out
        report['mask_checks'].append(dict(nodes=n,edges=e,excluded=excluded))
        return out
    Objective.run=obj
    def run(t,*a,**kw):
        inject_cost_cache(t)
        report['data_contract']=[]
        for split in ['train','validation']:
            ds=getattr(t,split+'_datasets')
            for index in (0,1):
                raw=ds._load_data_dict(index)
                sample=ds[index]
                assert ds.target_kind=='h0res' and ds.get_H0 and not ds.get_P2
                for prefix in ['node','edge']:
                    prior=sample[prefix+'_h0']
                    assert torch.isfinite(prior).all()
                    named=prefix+'_delta_h0'
                    if named in raw:
                        expected=torch.as_tensor(raw[named],dtype=sample[prefix+'_features'].dtype)
                        assert torch.equal(sample[prefix+'_features'],expected),(split,index,named)
                report['data_contract'].append(dict(split=split,index=index,schema=raw.get('sample_schema'),named_h0=('node_delta_h0' in raw),nodes=sample['node_h0'].shape[0],edges=sample['edge_h0'].shape[0]))
        base=t.model.experts[0];emb=base.embedding
        stage=1 if emb.only2b else 2
        frozen={n:p.detach().clone() for n,p in t.model.named_parameters() if not p.requires_grad}
        if stage==2:
            assert bool(emb.two_b_gnn_seeded)
            assert all(not p.requires_grad for m in emb._two_b_modules() for p in m.parameters())
        seen=set()
        def hook(name):
            def record(g):
                assert torch.isfinite(g).all(),name
                if bool(g.abs().sum()>0):seen.add(name)
            return record
        handles=[p.register_hook(hook(n)) for n,p in t.model.named_parameters() if p.requires_grad]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            result=orig_run(t,*a,**kw)
        for h in handles:h.remove()
        assert t.iter==4,t.iter
        assert report['mask_checks']
        assert all(torch.equal(dict(t.model.named_parameters())[n],p) for n,p in frozen.items())
        assert any('two_b_' in n for n in seen) if stage==1 else any('layers.' in n for n in seen)
        if stage==2:assert any('router.' in n for n in seen)
        names=sorted({event.name for event in prof.events()})
        (CONTROL/f'smoke_s{stage}_dispatch.json').write_text(json.dumps(names))
        if stage==2:assert any('FusedM0Function' in n or 'FusedPairFunction' in n or 'Sandwich' in n for n in names),names[:40]
        mapped=[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/scratch/' in line and ('.so' in line or 'data.mdb' in line)]
        assert not mapped,mapped
        report.update(stage=stage,final_next_step=t.iter,gradient_parameters=sorted(seen),dispatch_events=names,peak_allocated_bytes=torch.cuda.max_memory_allocated(),scratch_mappings=mapped)
        (CONTROL/f'smoke_s{stage}_evidence.json').write_text(json.dumps(report,indent=2))
        print('SMOKE_STAGE_CONTRACT_OK',stage,flush=True)
        return result
    MultiTrainer.run=run
    sys.argv=['dptb']+sys.argv[2:]
    main()

def main(mode):
    if mode=='diagnose':
        from dptb.data.build import build_dataset
        from dptb.utils.argcheck import normalize,get_cutoffs_from_model_options
        cfg=normalize(json.loads((CONTROL/'production_s1.json').read_text()))
        r,er,oer=get_cutoffs_from_model_options(cfg['model_options'])
        for split in ['train','validation']:
            ds=build_dataset(**cfg['data_options'][split],r_max=r,er_max=er,oer_max=oer,**dict(cfg['common_options'],device='cpu'))
            raw=ds._load_data_dict(0)
            print('RAW_CONTRACT',split,{k:(type(v).__name__,str(v)[:180]) for k,v in raw.items() if 'fingerprint' in k or 'schema' in k or 'semantics' in k},flush=True)
        return
    if mode in ['--worker','--cached-cli']:return worker()
    if mode=='smoke':
        stage_data()
        prepare_cost_cache()
    elif not (CONTROL/'SMOKE_OK.json').exists():raise RuntimeError('smoke has not passed')
    stamp=time.strftime('%Y%m%d_%H%M%S')
    parent=CONTROL/(mode+'_'+stamp);parent.mkdir()
    (CONTROL/(mode+'_current.json')).write_text(json.dumps(dict(output=str(parent))))
    ckpt=None
    stages=(1,2)
    resume=CONTROL/'resume_smoke_s1.json'
    if mode=='smoke' and resume.exists():
        ckpt=checkpoint(Path(json.loads(resume.read_text())['output']),3)
        resume.unlink()
        stages=(2,)
    for stage in stages:
        cfg=json.loads((CONTROL/f'production_s{stage}.json').read_text())
        cfg['train_options']['display_freq']=100
        if mode=='smoke':
            cfg['train_options'].update(max_steps=3,display_freq=1,save_freq=2,validation_freq=0,validation_epoch_freq=0)
        path=parent/f's{stage}.json';path.write_text(json.dumps(cfg,indent=2))
        out=parent/f's{stage}'
        cli=['multi-train',str(path),'-o',str(out),'-lp',str(parent/f's{stage}.log')]
        if ckpt:cli+=['-i',str(ckpt)]
        cmd=[sys.executable,str(Path(__file__)),'--worker' if mode=='smoke' else '--cached-cli']+cli
        print('SERIAL_STAGE_START',stage,cmd,flush=True)
        subprocess.run(cmd,check=True)
        if stage==1:ckpt=checkpoint(out,3 if mode=='smoke' else 100000)
        print('SERIAL_STAGE_COMPLETE',stage,flush=True)
    (CONTROL/('SMOKE_OK.json' if mode=='smoke' else 'PRODUCTION_COMPLETE.json')).write_text(json.dumps(dict(output=str(parent),time=time.time())))

if __name__=='__main__':main(sys.argv[1])
