#!/usr/bin/env python3
"""Matched warm CPU-table versus GPU-table inference on periodic geometries.

Both routes use the same checkpoint, graph, precision, orbital gauge and S.
CPU timing includes historical assembly, packing and host-to-device transfer.
GPU prepared timing excludes CPU topology compilation; geometry-to-output
timing is separately reported and includes that cost. No labels are loaded.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase.io import read

from .inference import load_predictor
from dptb.data.interfaces.p2_table import P2TableAssembler
from dptb.data.interfaces.p23_table import P23VNAFactorAssembler
from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
from dptb.data.interfaces.blockwise_tensor import block_tensors_to_feature_tensors
from dptb.utils.constants import Bohr2Ang


def cpu_features(predictor, atoms, prepared):
    if not np.all(atoms.pbc):
        raise ValueError('this independent legacy CPU benchmark requires fully periodic cells')
    bank = predictor.bank
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    sizes = [int(bank.p2.species[s]['orbital_norb']) for s in symbols]
    width = max(sizes)
    edges = prepared.geometry['edge_index'].cpu().numpy()
    shifts = prepared.geometry['edge_cell_shift'].cpu().numpy().astype(int)
    keys = [(i,i,0,0,0) for i in range(n)] + [(int(i),int(j),*map(int,s)) for (i,j),s in zip(edges.T,shifts)]
    positions = atoms.positions / Bohr2Ang
    cell = atoms.cell.array / Bohr2Ang
    args = dict(symbols=symbols,positions_bohr=positions,cell_bohr=cell)
    blocks = P2TableAssembler(bank.p2).assemble_sparse_blocks(**args,block_keys=keys)
    prior = np.zeros((len(keys),width,width))
    overlap = np.zeros_like(prior)
    for row,(i,j,*shift) in enumerate(keys):
        ni,nj = sizes[i],sizes[j]
        prior[row,:ni,:nj] = blocks[keys[row]] * 13.605698
        overlap[row,:ni,:nj] = (bank.overlap.onsite_component(symbols[i],'overlap') if row < n else
            bank.overlap.base_component(symbols[i],symbols[j],'overlap').evaluate(positions[j]+np.asarray(shift)@cell-positions[i]))
    use_p23,_ = bank.p23_composition(symbols)
    if use_p23:
        addition,_,_ = P23VNAFactorAssembler(bank.p23,factor_dtype=np.float64).assemble_graph_addition(
            **args,edge_index=np.empty((2,0),dtype=int),edge_cell_shift=np.empty((0,3),dtype=int),
            node_shapes=np.array([[s,s] for s in sizes]),edge_shapes=np.empty((0,2),dtype=int),
            node_pad_shape=(width,width),edge_pad_shape=(width,width))
        prior[:n] += addition
    lookup = {k:r for r,k in enumerate(keys[n:])}
    reverse = [lookup[(j,i,-x,-y,-z)] for i,j,x,y,z in keys[n:]]
    converter = OrbAbacus2DeepTB()
    packed = []
    for array in (prior,overlap):
        array[:n] = (array[:n]+array[:n].transpose(0,2,1))*.5
        array[n:] = (array[n:]+array[n:][reverse].transpose(0,2,1))*.5
        gauge = np.zeros_like(array)
        for row,(i,j,*_) in enumerate(keys):
            ni,nj = sizes[i],sizes[j]
            gauge[row,:ni,:nj] = converter.transform(array[row,:ni,:nj],bank.p2.species[symbols[i]]['orbital_shells'],bank.p2.species[symbols[j]]['orbital_shells'])
        data = {'atomic_numbers':torch.tensor(atoms.numbers[:,None]),'edge_index':torch.tensor(edges)}
        nr,er = block_tensors_to_feature_tensors(data,predictor.idp,
            node_blocks=torch.tensor(gauge[:n],device=predictor.device,dtype=predictor.dtype),
            edge_blocks=torch.tensor(gauge[n:],device=predictor.device,dtype=predictor.dtype))
        packed.extend((nr,er))
    return dict(zip(('node_p23','edge_p2','node_overlap','edge_overlap'),packed))


@torch.inference_mode()
def benchmark(predictor, atoms, repeats=5, warmup=1):
    prepared = predictor.prepare(atoms)
    def cpu_prior():
        return cpu_features(predictor,atoms,prepared)
    def cpu_inference():
        features = cpu_prior()
        inputs = {k:v.clone() for k,v in prepared.geometry.items()}
        inputs.update({k:v.clone() for k,v in features.items()})
        result = predictor.model(inputs)
        return dict(node_features=result['node_features']+features['node_p23'],
                    edge_features=result['edge_features']+features['edge_p2'],
                    node_overlap=features['node_overlap'],edge_overlap=features['edge_overlap'])
    def sync():
        if predictor.device.type == 'cuda': torch.cuda.synchronize(predictor.device)
    functions = dict(cpu_prior_and_s=cpu_prior,gpu_prior_and_s=prepared.plan,
                     cpu_prior_gpu_model=cpu_inference,gpu_prepared=prepared,
                     gpu_geometry_to_output=lambda:predictor(atoms))
    values = {k:fn() for k,fn in functions.items()}
    sync()
    errors = {k:float((values['cpu_prior_gpu_model'][k]-values['gpu_prepared'][k]).abs().max())
              for k in ('node_features','edge_features','node_overlap','edge_overlap')}
    for key,value in errors.items():
        if not np.isfinite(value) or value > (2e-5 if 'features' in key else 2e-6):
            raise AssertionError(f'CPU/GPU disagreement: {key}={value}')
    samples = {k:[] for k in functions}
    for _ in range(max(0,warmup-1)):
        for fn in functions.values(): fn()
    # Alternate order to reduce directional bias from shared-machine load.
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2: names.reverse()
        for name in names:
            sync(); start=time.perf_counter(); functions[name](); sync()
            samples[name].append(time.perf_counter()-start)
    return dict(atoms=len(atoms),edges=prepared.geometry['edge_index'].shape[1],
                cpu_vs_gpu_max_abs=errors,samples_seconds=samples,
                median_seconds={k:float(np.median(v)) for k,v in samples.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','p2','p23','overlap','expected-p2-sha256','geometry','output'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--compare-native',action='store_true',help='Compare fused CUDA against the torch table backend')
    ap.add_argument('--forward-batches',type=int,nargs='+',help='Measure pure AI forward on ordered geometry prefixes, e.g. 1 2 4 8 16')
    ap.add_argument('--warmup',type=int,default=3)
    args=ap.parse_args()
    if args.repeats < 3: ap.error('use at least three repeats')
    if args.warmup < 1: ap.error('use at least one warmup')
    if args.forward_batches and (args.compare_native or min(args.forward_batches)<1):
        ap.error('forward batch sizes must be positive and cannot combine with compare-native')
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    predictor=load_predictor(args.checkpoint,args.p2,args.p23,args.overlap,args.expected_p2_sha256,args.device)
    report=dict(torch_version=torch.__version__,device=str(predictor.device),
                gpu=torch.cuda.get_device_name(predictor.device) if predictor.device.type=='cuda' else None,
                threads=torch.get_num_threads(),p2_sha256=predictor.bank.p2_manifest_sha256,
                p23_sha256=predictor.bank.p23.manifest_sha256,overlap_sha256=predictor.bank.overlap.manifest_sha256,
                model_dtype=str(predictor.dtype),prior_dtype=str(predictor.bank._anchor.dtype),
                checkpoint=args.checkpoint,geometry=args.geometry,
                mode='cuda_vs_torch' if args.compare_native else 'cpu_vs_gpu',
                timing_scope=benchmark_native.__doc__ if args.compare_native else __doc__,cases=[])
    structures=read(args.geometry,index=':')
    if args.forward_batches:
        if max(args.forward_batches)>len(structures):
            ap.error('geometry file must contain at least the largest requested batch')
        report.update(mode='ai_forward_batches',timing_scope=benchmark_forward_batches.__doc__)
        report['cases']=benchmark_forward_batches(predictor,structures,args.forward_batches,args.repeats,args.warmup)
    for atoms in ([] if args.forward_batches else structures):
        row=(benchmark_native if args.compare_native else benchmark)(predictor,atoms,args.repeats,args.warmup)
        report['cases'].append(row)
        print('MATCHED',json.dumps(row),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')


@torch.inference_mode()
def benchmark_forward_batches(predictor, structures, batch_sizes=(1,2,4,8,16), repeats=5, warmup=3):
    """Warm synchronized wall-clock latency on ordered prefixes of geometries.

    ai_forward times only model(inputs), including its Python dispatch and GPU
    execution. Geometry, graph and NACF/S tensors are resident on device; fresh
    input clones and synchronization happen before its timer. No table assembly
    or NACF add-back is inside ai_forward. prepared_full includes recomputing
    priors, packing, input copies, model, add-back and output copies on a fixed
    prepared graph. geometry_to_full additionally includes CPU preparation and
    transfer. Model/table loading and CUDA compilation are excluded from all.
    Each batch uses the same ordered prefix; compare throughput only for this
    geometry family. Parity uses independently prepared singleton predictions.
    """
    if repeats < 1 or warmup < 1 or not batch_sizes or min(batch_sizes)<1 or max(batch_sizes)>len(structures):
        raise ValueError('invalid batch sizes, repeat count or warmup')
    def sync():
        if predictor.device.type=='cuda': torch.cuda.synchronize(predictor.device)
    def clone(data):
        return {k:v.clone() for k,v in data.items()}
    def residual_input(prepared):
        return {**clone(prepared.geometry),**clone(prepared.plan())}
    def values(prepared):
        inputs=residual_input(prepared)
        pn,pe=inputs['node_p23'].clone(),inputs['edge_p2'].clone()
        sn,se=inputs['node_overlap'].clone(),inputs['edge_overlap'].clone()
        result=predictor.model(inputs)
        return dict(node_residual=result['node_features'].clone(),edge_residual=result['edge_features'].clone(),
                    node_full=result['node_features']+pn,edge_full=result['edge_features']+pe,
                    node_overlap=sn,edge_overlap=se)
    # Keep validation references on CPU so they do not inflate batch GPU memory.
    reference=[]
    for atoms in structures[:max(batch_sizes)]:
        reference.append({k:v.cpu() for k,v in values(predictor.prepare(atoms)).items()})
    rows=[]
    for size in batch_sizes:
        subset=structures[:size]
        prepared=predictor.prepare(subset)
        # Validate every neighbour relation against the collated graph ownership.
        owner=prepared.geometry['batch'].reshape(-1)
        for key in ('edge_index','env_index','onsitenv_index'):
            if key in prepared.geometry:
                i,j=prepared.geometry[key]
                if not torch.equal(owner[i],owner[j]): raise AssertionError(f'cross-structure {key}')
        actual=values(prepared)
        errors={}
        for key,value in actual.items():
            target=torch.cat([item[key] for item in reference[:size]],dim=0)
            value=value.cpu()
            if not torch.isfinite(value).all(): raise AssertionError(f'nonfinite {key}')
            errors[key]=float((value-target).abs().max()) if value.numel() else 0.
            torch.testing.assert_close(value,target,atol=2e-4 if 'overlap' not in key else 2e-6,rtol=1e-5)
        del actual
        baseline=residual_input(prepared)
        def ai_forward():
            # This setup is deliberately outside the model-only timed region.
            inputs=clone(baseline)
            sync()
            start=time.perf_counter()
            result=predictor.model(inputs)
            sync()
            return time.perf_counter()-start,result
        def timed(fn):
            sync(); start=time.perf_counter(); result=fn(); sync()
            return time.perf_counter()-start,result
        functions=dict(ai_forward=ai_forward,prepared_full=lambda:timed(prepared),
                       geometry_to_full=lambda:timed(lambda:predictor(subset)))
        for _ in range(warmup):
            for fn in functions.values(): fn()
        samples={k:[] for k in functions}
        for repeat in range(repeats):
            names=list(functions)
            if repeat%2: names.reverse()
            for name in names:
                duration,result=functions[name]()
                samples[name].append(duration)
                del result
        medians={k:float(np.median(v)) for k,v in samples.items()}
        row=dict(structures=size,atoms=sum(map(len,subset)),edges=prepared.geometry['edge_index'].shape[1],
                 warmup=warmup,repeats=repeats,batch_vs_singleton_max_abs=errors,
                 samples_seconds=samples,median_seconds=medians,
                 median_seconds_per_structure={k:v/size for k,v in medians.items()},
                 structures_per_second={k:size/v for k,v in medians.items()})
        rows.append(row)
        print('FORWARD_BATCH',json.dumps(row),flush=True)
    return rows


@torch.inference_mode()
def benchmark_native(predictor, atoms, repeats=5, warmup=1):
    """Alternate backends on the exact same tables, plan and checkpoint."""
    prepared = predictor.prepare(atoms)
    original = {key:table.backend for key,table in predictor.bank.tables.items()}
    def select(backend):
        for table in predictor.bank.tables.values(): table.backend=backend
    def timed(fn):
        torch.cuda.synchronize(predictor.device)
        start=time.perf_counter(); result=fn(); torch.cuda.synchronize(predictor.device)
        return time.perf_counter()-start,result
    samples={k:[] for k in ('torch_prior','cuda_prior','torch_inference','cuda_inference')}
    try:
        reference={}
        for backend in ('torch','cuda'):
            select(backend)
            timed(prepared.plan)
            _,reference[backend]=timed(prepared)
        errors={k:float((reference['torch'][k]-reference['cuda'][k]).abs().max())
                for k in ('node_features','edge_features','node_overlap','edge_overlap')}
        if any(not np.isfinite(v) or v>2e-5 for v in errors.values()):
            raise AssertionError(errors)
        for _ in range(max(0,warmup-1)):
            for backend in ('torch','cuda'):
                select(backend)
                timed(prepared.plan); timed(prepared)
        for repeat in range(repeats):
            order=['torch','cuda'] if repeat%2==0 else ['cuda','torch']
            for backend in order:
                select(backend)
                for label,fn in [('prior',prepared.plan),('inference',prepared)]:
                    duration,_=timed(fn);samples[backend+'_'+label].append(duration)
        radial=[]
        assembly=prepared.plan.assembly
        for number,pair,key in assembly.query_specs:
            query=getattr(assembly,f'query_{number}')
            vectors=assembly.positions[query[:,0]]-assembly.positions[query[:,1]]+query[:,2:].to(assembly.cell.dtype)@assembly.cell
            table=predictor.bank.tables[key]
            durations={}
            values={}
            for backend in ('torch','cuda'):
                table.backend=backend
                _,values[backend]=timed(lambda:table(vectors))
                durations[backend]=[timed(lambda:table(vectors))[0] for _ in range(repeats)]
            diff=float((values['torch']-values['cuda']).abs().max())
            if not np.isfinite(diff) or diff>1e-8: raise AssertionError((key,diff))
            radial.append(dict(table=key,queries=len(vectors),shape=table.shape,max_abs=diff,
                               median_seconds={k:float(np.median(v)) for k,v in durations.items()}))
        return dict(atoms=len(atoms),edges=prepared.geometry['edge_index'].shape[1],
                    cuda_vs_torch_max_abs=errors,radial=radial,samples_seconds=samples,
                    median_seconds={k:float(np.median(v)) for k,v in samples.items()})
    finally:
        for key,table in predictor.bank.tables.items():table.backend=original[key]


if __name__ == '__main__':
    main()
