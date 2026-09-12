"""Full spinor NACF/S lookup validation and timing, without a learned forward.

The full eight-channel target must not be confused with a trained uu-real SOC
model. Its mapper and cutoffs come from explicit basis/model configuration;
only lookup is measured here, so no synthetic neural-network timing is emitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase.io import read

from dptb.data.transforms import OrbitalMapper
from dptb.data.interfaces.p2_table import P2TableStore
from dptb.data.interfaces.p23_table import P23VNAFactorTableStore
from dptb.utils.argcheck import get_cutoffs_from_model_options
from .assembly import NACFTableBank
from .soc import SOCProjectorStore
from .overlap import OverlapTableStore
from .inference import prepare_geometry
from .production_benchmark import memory_snapshot, storage_bytes, json_default


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('config','p2','p23','overlap','soc','geometry','output'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--batch-sizes',type=int,nargs='+',default=[2,4,8])
    ap.add_argument('--allow-shared-gpu',action='store_true',help='Allow explicitly labelled diagnostic timing on a shared GPU')
    args=ap.parse_args()
    if args.repeats<3: ap.error('use at least three repeats')
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    config=json.loads(Path(args.config).read_text())
    idp=OrbitalMapper(config['common_options']['basis'],method='e3tb',has_soc=True,
                      full_soc_prediction=True,nextham_uureal_mask=False,soc_complex_doubling=True)
    cutoffs=get_cutoffs_from_model_options(config['model_options'])
    p2=P2TableStore(args.p2)
    bank=NACFTableBank(p2,P23VNAFactorTableStore(args.p23),overlap_store=OverlapTableStore(args.overlap),
                       soc_store=SOCProjectorStore(args.soc,p2),device=args.device,backend='cuda')
    device=bank._anchor.device
    before=memory_snapshot(device)
    if before['other_processes'] != [] and not args.allow_shared_gpu: raise RuntimeError('GPU occupied or occupancy unknown')
    report=dict(schema='deeptb.nacf_full_soc_benchmark/v1',scope=__doc__,rows=[],
                geometry_sha256=hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
                rme_width=int(idp.reduced_matrix_element),gpu=torch.cuda.get_device_name(device),
                model_forward_measured=False,soc_sha256=bank.soc.manifest_sha256,
                p2_sha256=bank.p2_manifest_sha256,overlap_sha256=bank.overlap.manifest_sha256)
    structures=read(args.geometry,index=':')
    if args.batch_sizes and (min(args.batch_sizes)<1 or max(args.batch_sizes)>len(structures)):
        ap.error('batch sizes must fit the supplied geometries')
    references=[]
    for i,atoms in enumerate(structures):
        plan,geometry=prepare_geometry(bank,idp,atoms,cutoffs)
        baseline={}; samples={backend:[] for backend in ('torch','cuda')}; peaks={backend:[] for backend in samples}
        for repeat in range(-1,args.repeats):
            for backend in (('torch','cuda') if repeat%2 else ('cuda','torch')):
                for table in bank.tables.values(): table.backend=backend
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                start=time.perf_counter(); value=plan(); torch.cuda.synchronize(device)
                elapsed=time.perf_counter()-start
                if repeat>=0:
                    samples[backend].append(elapsed)
                    peaks[backend].append(torch.cuda.max_memory_allocated(device))
                baseline[backend]={k:v.cpu() for k,v in value.items()}
                del value
        errors={}
        for key in baseline['cuda']:
            x,y=baseline['cuda'][key],baseline['torch'][key]
            if not torch.isfinite(x).all(): raise AssertionError('nonfinite full SOC feature')
            errors[key]=float((x-y).abs().max()) if x.numel() else 0.
            torch.testing.assert_close(x,y,atol=2e-4 if 'overlap' not in key else 2e-6,rtol=1e-5)
        if i<max(args.batch_sizes,default=0): references.append(baseline['cuda'])
        blocks=plan.assembly()
        for key in ('node_p23_ao_ev','edge_p2_ao_ev'):
            x=blocks[key]
            reverse=x if key.startswith('node') else x[plan.assembly.reverse]
            torch.testing.assert_close(x,reverse.transpose(-1,-2).conj(),atol=1e-10,rtol=1e-10)
        width=plan.assembly.width
        spin_flip=float(blocks['node_p23_ao_ev'][:,:width,width:].abs().max())
        imaginary=float(blocks['node_p23_ao_ev'].imag.abs().max())
        snapshot=memory_snapshot(device)
        row=dict(index=i,identity=dict(atoms.info),atoms=len(atoms),edges=geometry['edge_index'].shape[1],
                 max_abs=errors,node_spin_flip_max_abs_ev=spin_flip,node_imaginary_max_abs_ev=imaginary,
                 median_seconds={k:float(np.median(v)) for k,v in samples.items()},samples_seconds=samples,
                 peak_allocated_bytes=peaks,table_bytes=storage_bytes(bank.tables.buffers()),
                 interference=snapshot['other_processes'] != [],snapshot=snapshot)
        report['rows'].append(row)
        (output/'report.json').write_text(json.dumps(report,indent=2,default=json_default)+'\n')
        print(json.dumps({k:row[k] for k in ('index','atoms','edges','median_seconds','interference')}),flush=True)
        del plan,geometry,baseline,blocks
    report['batches']=[]
    for size in args.batch_sizes:
        subset=structures[:size]
        plan,geometry=prepare_geometry(bank,idp,subset,cutoffs)
        owner=geometry['batch'].reshape(-1)
        for key in ('edge_index','env_index','onsitenv_index'):
            if key in geometry:
                ii,jj=geometry[key]
                if not torch.equal(owner[ii],owner[jj]): raise AssertionError('cross-structure '+key)
        samples={k:[] for k in ('torch_prior','cuda_prior','torch_fresh_prior','cuda_fresh_prior')}
        errors={}
        for repeat in range(-1,args.repeats):
            names=list(samples)
            if repeat%2: names.reverse()
            for name in names:
                backend=name.split('_')[0]
                bank.backend=backend
                for table in bank.tables.values(): table.backend=backend
                torch.cuda.synchronize(device)
                start=time.perf_counter()
                value=prepare_geometry(bank,idp,subset,cutoffs)[0]() if 'fresh' in name else plan()
                torch.cuda.synchronize(device)
                elapsed=time.perf_counter()-start
                if repeat>=0: samples[name].append(elapsed)
                if repeat==-1:
                    errors[name]={}
                    for key,tensor in value.items():
                        actual=tensor.cpu(); expected=torch.cat([v[key] for v in references[:size]])
                        torch.testing.assert_close(actual,expected,atol=2e-4 if 'overlap' not in key else 2e-6,rtol=1e-5)
                        errors[name][key]=float((actual-expected).abs().max()) if actual.numel() else 0.
                del value
        medians={k:float(np.median(v)) for k,v in samples.items()}
        snapshot=memory_snapshot(device)
        row=dict(structures=size,indices=list(range(size)),atoms=sum(map(len,subset)),
                 batch_vs_singleton_max_abs=errors,samples_seconds=samples,median_seconds=medians,
                 median_seconds_per_structure={k:v/size for k,v in medians.items()},
                 interference=snapshot['other_processes'] != [],snapshot=snapshot)
        report['batches'].append(row)
        (output/'report.json').write_text(json.dumps(report,indent=2,default=json_default)+'\n')
        print('BATCH '+json.dumps(row,default=json_default),flush=True)
        del plan,geometry
    report['table_bytes']=storage_bytes(bank.tables.buffers())
    report['complete']=not any(r['interference'] for r in report['rows']+report['batches'])
    (output/'report.json').write_text(json.dumps(report,indent=2,default=json_default)+'\n')
    if not report['complete'] and not args.allow_shared_gpu: raise RuntimeError('GPU interference during SOC benchmark')


if __name__=='__main__': main()
