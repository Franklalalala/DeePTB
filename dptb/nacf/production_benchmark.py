"""Three-route NACF timing and memory on a fixed set of production geometries.

Ratios are prior time / pure AI forward time, not fractions of total latency.
CPU is the existing production SciPy assembly (including its CPU batching), GPU torch is the device table
reference, and GPU CUDA uses fused native radial evaluation. Warm GPU prior
excludes compiled geometry topology; fresh-geometry prior includes graph,
topology, packing and transfer. Loading/checksums/JIT and disk output are cold.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from ase.io import read

from dptb.data import AtomicData
from .benchmark import cpu_features
from .inference import load_predictor


def json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    raise TypeError(f'unsupported report value: {type(value).__name__}')


def storage_bytes(tensors):
    unique={}
    for tensor in tensors:
        storage=tensor.untyped_storage()
        unique[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return sum(unique.values())


def memory_snapshot(device):
    torch.cuda.synchronize(device)
    process_mib=None
    other_processes=None
    uuid=str(torch.cuda.get_device_properties(device).uuid)
    def normalized_uuid(value): return value.lower().replace('gpu-','').replace('-','')
    try:
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        records=[row.split(',') for row in text.splitlines()]
        records=[r for r in records if normalized_uuid(r[1].strip())==normalized_uuid(uuid)]
        values=[int(r[2].strip()) for r in records if r[0].strip()==str(os.getpid())]
        process_mib=sum(values) if values else None
        other_processes=[dict(pid=int(r[0]),gpu_mib=int(r[2])) for r in records if r[0].strip()!=str(os.getpid())]
    except (OSError,ValueError,subprocess.CalledProcessError):
        pass
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                process_gpu_mib=process_mib,gpu_uuid=uuid,other_processes=other_processes)


@torch.inference_mode()
def measure_case(predictor, atoms, repeats=3, warmup=1, allow_shared_gpu=False):
    device=predictor.device
    structures=[atoms] if hasattr(atoms,'get_positions') else list(atoms)
    if not structures: raise ValueError('empty benchmark batch')
    def graph_for(item):
        r,e,o=predictor.cutoffs
        return AtomicData.from_points(pos=item.positions,cell=item.cell.array,pbc=item.pbc,
                                      atomic_numbers=item.numbers,r_max=r,er_max=e,oer_max=o)
    def sync(): torch.cuda.synchronize(device)
    def select(backend):
        predictor.bank.backend=backend
        for table in predictor.bank.tables.values(): table.backend=backend
    select('cuda')
    before=memory_snapshot(device)
    if before['other_processes'] != [] and not allow_shared_gpu:
        raise RuntimeError('GPU is occupied by another process; choose an idle device')
    prepared=predictor.prepare(atoms)
    # Retain the original CPU graphs: a CPU lookup never needs a GPU-to-CPU
    # graph round trip. The numerical CPU assembler batches queries internally.
    cpu_graphs=[graph_for(item) for item in structures]
    native=prepared.plan()
    geometry=prepared.geometry
    def clone(data): return {k:v.clone() for k,v in data.items()}
    baseline={**clone(geometry),**clone(native)}
    def ai_input(): return clone(baseline)
    def cpu_from_graphs(graphs):
        values=[cpu_features(predictor,item,SimpleNamespace(geometry=graph))
                for item,graph in zip(structures,graphs)]
        return values[0] if len(values)==1 else {k:torch.cat([v[k] for v in values]) for k in values[0]}
    def cpu_prior(): return cpu_from_graphs(cpu_graphs)
    def cpu_fresh():
        return cpu_from_graphs([graph_for(item) for item in structures])
    def gpu_prior(backend):
        select(backend)
        return prepared.plan()
    def gpu_fresh(backend):
        select(backend)
        return predictor.prepare(atoms).plan()
    def gpu_full(backend):
        select(backend)
        return predictor(atoms)
    functions=dict(ai_forward=lambda:predictor.model(current_inputs),cpu_prior=cpu_prior,
                   torch_prior=lambda:gpu_prior('torch'),cuda_prior=lambda:gpu_prior('cuda'),
                   cpu_fresh_prior=cpu_fresh,torch_fresh_prior=lambda:gpu_fresh('torch'),cuda_fresh_prior=lambda:gpu_fresh('cuda'),
                   cuda_geometry_to_full=lambda:gpu_full('cuda'))
    errors={}
    # Validate prior features independently before reporting any timing ratio.
    for backend,fn in [('cpu',cpu_prior),('torch',lambda:gpu_prior('torch'))]:
        value=fn()
        errors[backend]={}
        for key in native:
            errors[backend][key]=float((value[key]-native[key]).abs().max()) if native[key].numel() else 0.
            torch.testing.assert_close(value[key],native[key],atol=2e-4 if 'overlap' not in key else 2e-6,rtol=1e-5)
        del value
    samples={k:[] for k in functions}; peaks={k:[] for k in functions}
    for repeat in range(-warmup,repeats):
        names=list(functions)
        if repeat%2: names.reverse()
        for name in names:
            current_inputs=ai_input() if name=='ai_forward' else None
            sync()
            torch.cuda.reset_peak_memory_stats(device)
            allocated=torch.cuda.memory_allocated(device)
            start=time.perf_counter(); result=functions[name](); sync()
            elapsed=time.perf_counter()-start
            if repeat>=0:
                samples[name].append(elapsed)
                peaks[name].append(dict(baseline_allocated_bytes=allocated,
                                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                        incremental_peak_bytes=torch.cuda.max_memory_allocated(device)-allocated))
            for key in ('node_features','edge_features') if name=='ai_forward' else ('node_p23','edge_p2'):
                if key in result and not torch.isfinite(result[key]).all(): raise AssertionError('nonfinite '+key)
            del result,current_inputs
    medians={k:float(np.median(v)) for k,v in samples.items()}
    after=memory_snapshot(device)
    return dict(formula=[item.get_chemical_formula() for item in structures],structures=len(structures),
                atoms=sum(map(len,structures)),edges=geometry['edge_index'].shape[1],
                interference=before['other_processes'] != [] or after['other_processes'] != [],snapshot_before=before,
                prior_max_abs=errors,samples_seconds=samples,median_seconds=medians,memory_peaks=peaks,
                median_seconds_per_structure={k:v/len(structures) for k,v in medians.items()},
                structures_per_second={k:len(structures)/v for k,v in medians.items()},
                prior_to_ai_ratio={k:medians[k]/medians['ai_forward'] for k in medians if 'prior' in k},
                cached_table_bytes=storage_bytes(predictor.bank.tables.buffers()),
                resident_graph_bytes=storage_bytes(geometry.values()),
                resident_model_input_bytes=storage_bytes(baseline.values()),
                geometry_plan_bytes=storage_bytes(v for k,v in prepared.plan.named_buffers() if not k.startswith('assembly.bank.')),
                snapshot=after)


def aggregate(rows, include_shared=False):
    good=[r for r in rows if r['status']=='ok' and (include_shared or not r.get('interference',False))]
    if not good: return dict(success=0,failed=len(rows))
    keys=good[0]['median_seconds']
    total={k:sum(r['median_seconds'][k] for r in good) for k in keys}
    return dict(success=len(good),failed=len(rows)-len(good),sum_seconds=total,
                prior_to_ai_ratio_of_sums={k:total[k]/total['ai_forward'] for k in keys if 'prior' in k},
                per_structure_ratio_quantiles={k:{str(q):float(np.quantile([r['prior_to_ai_ratio'][k] for r in good],q))
                                                  for q in (.1,.5,.9)} for k in keys if 'prior' in k})


@torch.inference_mode()
def validate_batch(predictor, structures):
    references=[{k:v.cpu() for k,v in predictor(item).items()
                 if k in ('node_features','edge_features','node_overlap','edge_overlap')}
                for item in structures]
    result=predictor(structures)
    owner=result['batch'].reshape(-1)
    for key in ('edge_index','env_index','onsitenv_index'):
        if key in result:
            i,j=result[key]
            if not torch.equal(owner[i],owner[j]): raise AssertionError('cross-structure '+key)
    errors={}
    for key in references[0]:
        expected=torch.cat([ref[key] for ref in references])
        actual=result[key].cpu()
        torch.testing.assert_close(actual,expected,atol=2e-4 if 'overlap' not in key else 2e-6,rtol=1e-5)
        errors[key]=float((actual-expected).abs().max()) if actual.numel() else 0.
    return errors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','p2','p23','overlap','expected-p2-sha256','geometry','output'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--allow-shared-gpu',action='store_true',help='Diagnostic run only; shared rows remain excluded from the primary aggregate')
    ap.add_argument('--batch-sizes',type=int,nargs='+',default=[])
    ap.add_argument('--skip-singletons',action='store_true',help='Run only the explicitly requested mixed batches')
    args=ap.parse_args()
    if args.repeats<3 or args.warmup<1: ap.error('at least 3 repeats and 1 warmup required')
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    device=torch.device(args.device)
    torch.cuda.init(); torch.cuda.synchronize(device)
    stages={'context':memory_snapshot(device)}
    if stages['context']['other_processes'] != [] and not args.allow_shared_gpu:
        raise RuntimeError('requested GPU is already occupied')
    from ._cuda import extension
    extension()
    stages['extension_imported']=memory_snapshot(device)
    predictor=load_predictor(args.checkpoint,args.p2,args.p23,args.overlap,args.expected_p2_sha256,args.device,backend='cuda')
    if predictor.idp.has_soc: raise ValueError('this matched model benchmark requires the non-SOC checkpoint; measure full SOC tables separately')
    stages['model_loaded_empty_table_cache']=memory_snapshot(device)
    model_bytes=storage_bytes(list(predictor.model.parameters())+list(predictor.model.buffers()))
    structures=read(args.geometry,index=':')
    if args.batch_sizes and (min(args.batch_sizes)<1 or max(args.batch_sizes)>len(structures)):
        ap.error('batch sizes must fit the geometry file')
    if args.skip_singletons and not args.batch_sizes: ap.error('no work requested')
    report=dict(schema='deeptb.nacf_production_benchmark/v1',scope=__doc__,checkpoint=args.checkpoint,
                checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                runtime_model_overrides={'so2_fusion_mode':'streamed_m_major_ref','mole_linear_mode':'split_loop'},
                model_dtype=str(predictor.dtype),prior_dtype=str(predictor.bank._anchor.dtype),
                geometry_sha256=hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,threads=torch.get_num_threads(),
                p2_sha256=predictor.bank.p2_manifest_sha256,p23_sha256=predictor.bank.p23.manifest_sha256,
                overlap_sha256=predictor.bank.overlap.manifest_sha256,model_bytes=model_bytes,
                memory_stages=stages,repeats=args.repeats,warmup=args.warmup,requested=len(structures),rows=[],batches=[])
    def save():
        report['aggregate']=aggregate(report['rows'])
        report['shared_diagnostic_aggregate']=aggregate(report['rows'],include_shared=True)
        temporary=out/'report.json.tmp'; temporary.write_text(json.dumps(report,indent=2,default=json_default)+'\n')
        temporary.replace(out/'report.json')
    save()
    for i,atoms in enumerate([] if args.skip_singletons else structures):
        try:
            row=measure_case(predictor,atoms,args.repeats,args.warmup,args.allow_shared_gpu)
            row.update(status='ok',index=i,identity=dict(atoms.info))
        except Exception as exc:
            row=dict(status='failed',index=i,identity=dict(atoms.info),error=repr(exc))
            gc.collect(); torch.cuda.empty_cache()
        report['rows'].append(row); save()
        print(json.dumps({k:row[k] for k in ('index','status','atoms','edges','median_seconds','error') if k in row},default=json_default),flush=True)
    for size in args.batch_sizes:
        try:
            errors=validate_batch(predictor,structures[:size])
            row=measure_case(predictor,structures[:size],args.repeats,args.warmup,args.allow_shared_gpu)
            row.update(status='ok',indices=list(range(size)),batch_vs_singleton_max_abs=errors)
        except Exception as exc:
            row=dict(status='failed',structures=size,error=repr(exc))
            gc.collect(); torch.cuda.empty_cache()
        report['batches'].append(row); save()
        print('BATCH '+json.dumps({k:row[k] for k in ('structures','status','atoms','edges','median_seconds','error') if k in row}),flush=True)
    report['memory_stages']['finished']=memory_snapshot(device)
    report['table_cache_bytes']=storage_bytes(predictor.bank.tables.buffers())
    save()
    checked=report['shared_diagnostic_aggregate'] if args.allow_shared_gpu else report['aggregate']
    if checked['failed'] or any(r['status']!='ok' or (r.get('interference') and not args.allow_shared_gpu) for r in report['batches']):
        raise RuntimeError('benchmark has failed or interfered structures; see report')


if __name__=='__main__': main()
