"""Matched full H0/S checks; one pass each, not a steady-state speed benchmark."""
import gc, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from production_io import load_case, read_csr, compare_blocks, RY_TO_EV
from h0rebuild.assemble import assemble_h0
from h0rebuild.precompiled import ROOT, verify

original_popen = subprocess.Popen
def no_compilers(args,*a,**kw):
    words=args if isinstance(args,(list,tuple)) else args.split()
    if any(Path(str(v)).name in ('ninja','nvcc','g++','c++','gcc','cc','cmake') for v in words):
        raise AssertionError('Runtime compilation attempted')
    return original_popen(args,*a,**kw)
subprocess.Popen=no_compilers
rows=[]
for case in ('nonSOC_db_seq_id_15674','SOC_mp-31055'):
    path=Path('/home/mingkang_nt/codex/h0_cuda_random100_cell_gauge_v2_20260912/raw')/case
    st,sd,opts,contract=load_case(path)
    common=dict(two_center_backend='pyabacus',two_center_dr_bohr=.01,
                two_center_cache_dir=str(ROOT/'tmp'),local_integration='fft_grid_periodic',
                local_cache_max_mb=4096,pair_support='nonlocal_complete',
                spatial_backend='indexed',hermitize=False,initial_moments_z=None)
    t=time.perf_counter()
    ref=assemble_h0(st,sd,**opts,**common,compute_device='cpu',field_backend='numpy',structure_factor_backend='direct',strict_reproduction=True)
    ref_seconds=time.perf_counter()-t
    torch.cuda.synchronize()
    t=time.perf_counter()
    gpu=assemble_h0(st,sd,**opts,**common,compute_device='cuda:0',field_backend='torch',structure_factor_backend='cufinufft',strict_reproduction=False,cuda_precompiled=True)
    torch.cuda.synchronize()
    gpu_seconds=time.perf_counter()-t
    counts=[sd[a.species].orb.norb*(2 if opts['nspin']==4 else 1) for a in st.atoms]
    hdiff=compare_blocks(gpu.h_blocks_ry,ref.h_blocks_ry,counts,scale=RY_TO_EV)['total']['max_abs']
    sdiff=compare_blocks(gpu.s_blocks,ref.s_blocks,counts)['total']['max_abs']
    hraw,_=read_csr(path/'OUT.ABACUS/data-HR0_SPIN0.csr',counts,nspin=opts['nspin'])
    sraw,_=read_csr(path/'OUT.ABACUS/data-SR-sparse_SPIN0.csr',counts,nspin=opts['nspin'])
    hm=compare_blocks(gpu.h_blocks_ry,hraw,counts,scale=RY_TO_EV)['total']['max_abs']
    sm=compare_blocks(gpu.s_blocks,sraw,counts)['total']['max_abs']
    row=dict(case=case,nspin=opts['nspin'],blocks=len(gpu.h_blocks_ry),
             candidate_vs_cpu_Hmax_eV=hdiff,candidate_vs_cpu_Smax=sdiff,
             raw_Hmax_meV=hm*1000,raw_Smax=sm,one_pass_cpu_seconds=ref_seconds,
             one_pass_candidate_seconds=gpu_seconds,steady_speedup=None,
             backend=gpu.metadata['two_center_backend'])
    print(json.dumps(row),flush=True)
    assert hdiff<1e-7 and sdiff<1e-7 and hm<.005 and sm<1e-6,row
    assert gpu.metadata['cuda_precompiled']
    rows.append(row)
    (ROOT/'work/full_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
    del ref,gpu
    gc.collect(); torch.cuda.empty_cache()
print('FULL_TWO_CASE_PASS',flush=True)
