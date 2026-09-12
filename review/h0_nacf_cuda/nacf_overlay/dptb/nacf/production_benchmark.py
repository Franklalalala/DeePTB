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
import uuid
from types import SimpleNamespace

import numpy as np
import torch
from ase.io import read

from dptb.data import AtomicData
from .benchmark import cpu_features
from .inference import load_predictor


class GPUUnavailable(RuntimeError):
    """A benchmark cannot establish the requested unshared device condition."""


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
def measure_case(predictor, atoms, repeats=3, warmup=1, allow_shared_gpu=False,
                 cpu_feature_function=cpu_features):
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
        raise GPUUnavailable('GPU is occupied or occupancy is unknown; retry on an idle device')
    prepared=predictor.prepare(atoms)
    # Retain the original CPU graphs: a CPU lookup never needs a GPU-to-CPU
    # graph round trip. The numerical CPU assembler batches queries internally.
    cpu_graphs=[graph_for(item) for item in structures]
    native=prepared.plan()
    geometry=prepared.geometry
    def clone(data): return {k:v.clone() for k,v in data.items()}
    baseline=(prepared.model_inputs(native) if hasattr(prepared,'model_inputs')
              else {**clone(geometry),**clone(native)})
    def ai_input():
        if hasattr(predictor.model,'prepare_arm_inputs'):
            return predictor.model.prepare_arm_inputs(baseline)
        return clone(baseline)
    def ai_forward():
        if hasattr(predictor.model,'forward_arms'):
            return predictor.model.forward_arms(current_inputs)
        return predictor.model(current_inputs)
    def cpu_from_graphs(graphs):
        values=[cpu_feature_function(predictor,item,SimpleNamespace(geometry=graph))
                for item,graph in zip(structures,graphs)]
        return values[0] if len(values)==1 else {k:torch.cat([v[k] for v in values]) for k in values[0]}
    def cpu_prior(): return cpu_from_graphs(cpu_graphs)
    def cpu_fresh():
        return cpu_from_graphs([graph_for(item) for item in structures])
    functions=dict(ai_forward=ai_forward,cpu_prior=cpu_prior,
                   torch_prior=prepared.plan,cuda_prior=prepared.plan,
                   cpu_fresh_prior=cpu_fresh,torch_fresh_prior=lambda:predictor.prepare(atoms).plan(),
                   cuda_fresh_prior=lambda:predictor.prepare(atoms).plan(),
                   cuda_geometry_to_full=lambda:predictor(atoms))
    errors={}
    # Validate prior features independently before reporting any timing ratio.
    for backend,fn in [('cpu',cpu_prior),('torch',prepared.plan)]:
        if backend=='torch': select('torch')
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
            # Backend switching is benchmark instrumentation, not inference.
            if name.startswith(('torch_','cuda_')): select(name.split('_',1)[0])
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
            for key in ('node_features','edge_features') if name in ('ai_forward','cuda_geometry_to_full') else ('node_p23','edge_p2'):
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
    ap.add_argument('--model-backend',choices=('checkpoint','reference'),default='checkpoint',
                    help='Use the checkpoint model backend; reference is an explicit non-SOC diagnostic override')
    ap.add_argument('--p23-missing-policy',choices=('error','p2_if_missing_pairs'),default='error')
    ap.add_argument('--expected-p23-sha256',help='Trusted P23 fingerprint required for composition fallback')
    ap.add_argument('--allow-shared-gpu',action='store_true',help='Diagnostic run only; shared rows remain excluded from the primary aggregate')
    ap.add_argument('--batch-sizes',type=int,nargs='+',default=[])
    ap.add_argument('--skip-singletons',action='store_true',help='Run only the explicitly requested mixed batches')
    ap.add_argument('--resume',action='store_true',help='Resume matching output, preserving earlier failed attempts')
    ap.add_argument('--hopping-checkpoint',help='SOC hopping arm; --checkpoint is then the onsite arm')
    ap.add_argument('--soc',help='Full SOC projector sidecar, required with --hopping-checkpoint')
    ap.add_argument('--onsite-config',help='Verified same-run SOC training config when checkpoint omits dataset settings')
    ap.add_argument('--hopping-config',help='Verified same-run SOC hopping training config')
    ap.add_argument('--soc-ry-to-ev',type=float,default=13.605693122994,
                    help='Verified SOC training materializer conversion; default matches SOC29303')
    args=ap.parse_args()
    if bool(args.soc) != bool(args.hopping_checkpoint): ap.error('SOC requires both --soc and --hopping-checkpoint')
    if args.soc and args.model_backend != 'checkpoint': ap.error('SOC benchmarking requires the checkpoint model backend')
    if args.repeats<3 or args.warmup<1: ap.error('at least 3 repeats and 1 warmup required')
    out=Path(args.output)
    previous=None
    if out.exists():
        if not args.resume: raise FileExistsError(out)
        previous=json.loads((out/'report.json').read_text())
    elif args.resume: ap.error('resume requires an existing report')
    else: out.mkdir(parents=True)
    device=torch.device(args.device)
    torch.cuda.init(); torch.cuda.synchronize(device)
    stages={'context':memory_snapshot(device)}
    if stages['context']['other_processes'] != [] and not args.allow_shared_gpu:
        raise GPUUnavailable('requested GPU is occupied or occupancy is unknown')
    from ._cuda import extension
    extension()
    stages['extension_imported']=memory_snapshot(device)
    cpu_function = cpu_features
    if args.soc:
        from .spinor_inference import load_soc_predictor
        from .soc_cpu_reference import cpu_soc_features
        predictor=load_soc_predictor(args.checkpoint,args.hopping_checkpoint,p2=args.p2,p23=args.p23,
            overlap=args.overlap,soc=args.soc,expected_p2_sha256=args.expected_p2_sha256,
            device=args.device,backend='cuda',ry_to_ev=args.soc_ry_to_ev,
            onsite_config=args.onsite_config,hopping_config=args.hopping_config,
            p23_missing_policy=args.p23_missing_policy,expected_p23_sha256=args.expected_p23_sha256)
        cpu_function=cpu_soc_features
    else:
        predictor=load_predictor(args.checkpoint,args.p2,args.p23,args.overlap,args.expected_p2_sha256,args.device,
                                 backend='cuda',model_backend=args.model_backend,
                                 p23_missing_policy=args.p23_missing_policy,expected_p23_sha256=args.expected_p23_sha256)
        if predictor.idp.has_soc: raise ValueError('compact SOC inference requires a hopping checkpoint and complete SOC tables')
    stages['model_loaded_empty_table_cache']=memory_snapshot(device)
    model_bytes=storage_bytes(list(predictor.model.parameters())+list(predictor.model.buffers()))
    structures=read(args.geometry,index=':')
    if args.batch_sizes and (min(args.batch_sizes)<1 or max(args.batch_sizes)>len(structures)):
        ap.error('batch sizes must fit the geometry file')
    if args.skip_singletons and not args.batch_sizes: ap.error('no work requested')
    report=dict(schema='deeptb.nacf_production_benchmark/v1',scope=__doc__,checkpoint=args.checkpoint,
                checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                runtime_model_overrides=predictor.runtime_model_overrides,
                p23_missing_policy=predictor.bank.p23_missing_policy,
                model_dtype=str(predictor.dtype),prior_dtype=str(predictor.bank._anchor.dtype),
                geometry_sha256=hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,threads=torch.get_num_threads(),
                p2_sha256=predictor.bank.p2_manifest_sha256,p23_sha256=predictor.bank.p23.manifest_sha256,
                overlap_sha256=predictor.bank.overlap.manifest_sha256,model_bytes=model_bytes,
                memory_stages=stages,repeats=args.repeats,warmup=args.warmup,requested=len(structures),rows=[],batches=[])
    report['source_sha256']['interfaces/p2_table.py']=hashlib.sha256(
        (Path(__file__).parents[1]/'data/interfaces/p2_table.py').read_bytes()).hexdigest()
    report.update(hopping_checkpoint=args.hopping_checkpoint,
        hopping_checkpoint_sha256=hashlib.sha256(Path(args.hopping_checkpoint).read_bytes()).hexdigest() if args.soc else None,
        soc_sha256=predictor.bank.soc.manifest_sha256 if args.soc else None,
        output_contract='full_soc_nacf_plus_uu_real_residual' if args.soc else 'full_h_minus_nacf_addback',
        cpu_algorithm='legacy_scalar_algorithm_lifted_to_complex_spinor' if args.soc else 'legacy_scalar_algorithm',
        cpu_rotation='stable_south_pole',
        ry_to_ev=predictor.bank.ry_to_ev,
        training_config_sha256={role:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for role,path in (('onsite',args.onsite_config),('hopping',args.hopping_config)) if path},
        ai_forward_scope='both_onsite_and_hopping_arms_copies_excluded' if args.soc else 'single_model_copies_excluded')
    run_id=uuid.uuid4().hex
    invocation=dict(run_id=run_id,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    allow_shared_gpu=args.allow_shared_gpu,source_sha256=report['source_sha256'])
    if previous is not None:
        for key in ('checkpoint_sha256','geometry_sha256','gpu','torch_version','threads','p2_sha256','p23_sha256','overlap_sha256',
                    'model_dtype','prior_dtype','runtime_model_overrides','p23_missing_policy','repeats','warmup','requested','source_sha256',
                    'hopping_checkpoint_sha256','soc_sha256','output_contract','cpu_algorithm','cpu_rotation','ai_forward_scope','ry_to_ev','training_config_sha256'):
            if previous.get(key)!=report[key]: raise ValueError('resume provenance mismatch: '+key)
        report['rows']=previous['rows']
        report['batches']=previous.get('batches',[])
        report['attempt_history']=previous.get('attempt_history',[])
        report['invocations']=previous.get('invocations',[])
    report.setdefault('invocations',[]).append(invocation)
    def accepted(row):
        return row['status']=='ok' and (args.allow_shared_gpu or not row.get('interference',True))
    def replace_row(collection,row,key):
        old=next((r for r in collection if r[key]==row[key]),None)
        if old is not None:
            report.setdefault('attempt_history',[]).append(old)
            collection.remove(old)
        collection.append(row)
        collection.sort(key=lambda r:r[key])
    def save():
        report['aggregate']=aggregate(report['rows'])
        report['shared_diagnostic_aggregate']=aggregate(report['rows'],include_shared=True)
        temporary=out/'report.json.tmp'; temporary.write_text(json.dumps(report,indent=2,default=json_default)+'\n')
        temporary.replace(out/'report.json')
    save()
    for i,atoms in enumerate([] if args.skip_singletons else structures):
        if any(r['index']==i and accepted(r) for r in report['rows']): continue
        unavailable=False
        try:
            row=measure_case(predictor,atoms,args.repeats,args.warmup,args.allow_shared_gpu,cpu_feature_function=cpu_function)
            row.update(status='ok',index=i,identity=dict(atoms.info))
        except Exception as exc:
            row=dict(status='failed',index=i,identity=dict(atoms.info),error=repr(exc))
            unavailable=isinstance(exc,GPUUnavailable)
            gc.collect(); torch.cuda.empty_cache()
        row['run_id']=run_id
        replace_row(report['rows'],row,'index'); save()
        print(json.dumps({k:row[k] for k in ('index','status','atoms','edges','median_seconds','error') if k in row},default=json_default),flush=True)
        if unavailable: raise GPUUnavailable('stopped before further cases; resume when the device is idle')
    for size in args.batch_sizes:
        if any(r['structures']==size and accepted(r) for r in report['batches']): continue
        try:
            errors=validate_batch(predictor,structures[:size])
            row=measure_case(predictor,structures[:size],args.repeats,args.warmup,args.allow_shared_gpu,cpu_feature_function=cpu_function)
            row.update(status='ok',indices=list(range(size)),batch_vs_singleton_max_abs=errors)
        except Exception as exc:
            row=dict(status='failed',structures=size,error=repr(exc))
            gc.collect(); torch.cuda.empty_cache()
        row['run_id']=run_id
        replace_row(report['batches'],row,'structures'); save()
        print('BATCH '+json.dumps({k:row[k] for k in ('structures','status','atoms','edges','median_seconds','error') if k in row}),flush=True)
    report['memory_stages']['finished']=memory_snapshot(device)
    report['table_cache_bytes']=storage_bytes(predictor.bank.tables.buffers())
    report['table_cache_scope']='resident cache of final invocation; after resume this may cover only remaining structures'
    save()
    checked=report['shared_diagnostic_aggregate'] if args.allow_shared_gpu else report['aggregate']
    if checked['failed'] or any(r['status']!='ok' or (r.get('interference') and not args.allow_shared_gpu) for r in report['batches']):
        raise RuntimeError('benchmark has failed or interfered structures; see report')


if __name__=='__main__': main()
