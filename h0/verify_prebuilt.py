"""Real component regressions in a fresh process with compiler execution forbidden."""
import json, os, sys, time, unittest, subprocess
from pathlib import Path
import numpy as np
import torch
from h0rebuild.precompiled import verify, sha256, ROOT

# Deny external compilation, including accidental JIT through future code paths.
real_popen = subprocess.Popen
def no_compilers(args, *a, **kw):
    words = args if isinstance(args, (list,tuple)) else args.split()
    if any(Path(str(v)).name in ('ninja','nvcc','g++','c++','gcc','cc','cmake') for v in words):
        raise AssertionError('Runtime attempted to compile: '+str(words))
    return real_popen(args,*a,**kw)
subprocess.Popen = no_compilers
for name in ('_cuda_two_center','_cuda_local_grid'):
    verify(name, check_device=True)
from h0rebuild.cuda_two_center import CUDATwoCenter
from h0rebuild.pyabacus_integrals import PyAbacusTwoCenter
from tests_h0fast.test_cuda_two_center import load_simple_structure, TestCUDATwoCenter
from tests_h0fast.test_local_grid import TestCudaLocalGrid

report={'runtime_compilation':'forbidden','hardware':torch.cuda.get_device_name(), 'two_center':[]}
before={name:verify(name)['binary_sha256'] for name in ('_cuda_two_center','_cuda_local_grid')}
base=Path('/home/mingkang_nt/codex/h0_flash_20260912/production_reference')
cases=[(base/'db_seq_id_10667',1),(base/'db_seq_id_10282',1),
 (Path('/home/mingkang_nt/codex/h0_cuda_benchmark_20260912/oracle/db_seq_id_10282_nspin4'),4),
 (Path('/home/mingkang_nt/codex/h0_cuda_random100_cell_gauge_v2_20260912/raw/nonSOC_db_seq_id_11083'),1)]
rng=np.random.default_rng(20260912)
for case,spin in cases:
    st,sd=load_simple_structure(case,scalarize=spin==1)
    ref=PyAbacusTwoCenter(sd,nspin=spin)
    gpu=CUDATwoCenter(sd,nspin=spin)
    max_s=max_t=max_v=0.
    syms=list(sd)
    disps=np.vstack([np.zeros(3),[1e-8,0,0],[0,0,-1.5],rng.normal(size=(12,3))*2])
    for i,disp in enumerate(disps):
        a,b=syms[i%len(syms)],syms[(i+1)%len(syms)]
        rs,rt=ref.scalar_pair(a,b,[0,0,0],disp)
        gs,gt=gpu.scalar_pair(a,b,[0,0,0],disp)
        max_s=max(max_s,float(np.max(np.abs(rs-gs))))
        max_t=max(max_t,float(np.max(np.abs(rt-gt)))*13.605693122994)
        if i in (3,4):
            cand=[(j,np.asarray(atom.frac)@st.cell_bohr) for j,atom in enumerate(st.atoms)]
            rv=ref.nonlocal_block(st,a,b,[0,0,0],disp,cand)
            gv=gpu.nonlocal_block(st,a,b,[0,0,0],disp,cand)
            max_v=max(max_v,float(np.max(np.abs(rv-gv)))*13.605693122994)
    row=dict(case=case.name,nspin=spin,S_max=max_s,T_max_eV=max_t,Vnl_max_eV=max_v)
    print(json.dumps(row),flush=True)
    report['two_center'].append(row)
    assert max_s<1e-7 and max_t<1e-7 and max_v<1e-7, row
    del ref,gpu
    torch.cuda.empty_cache()

# Run local-grid regressions on a non-default stream to exercise the fixed boundary.
suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestCudaLocalGrid)
stream=torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    result=unittest.TextTestRunner(verbosity=2).run(suite)
stream.synchronize()
assert result.wasSuccessful(), 'Local-grid regression failed'
report['local_grid_tests']=result.testsRun
report['nondefault_stream']=True
assert before=={name:verify(name)['binary_sha256'] for name in before}
report['binary_hashes_unchanged']=before
report['status']='PASS'
(ROOT/'work/verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','binary_hashes_unchanged':before}),flush=True)
