"""New precompiled/offline route only, same immutable 100-case cohort."""
import os, sys, json, time, traceback, subprocess, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'new100'

def write(path,data):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(path)

def worker(cid):
    import numpy as np
    import torch
    from h0rebuild.offline import load_species
    from h0rebuild.precompiled import verify,sha256
    from h0rebuild.assemble import assemble_h0
    import h0rebuild.upf, h0rebuild.orb, h0rebuild.cuda_two_center
    import production_io
    cat=json.loads((ROOT/'offline_tables/catalog.json').read_text()); entry=cat['cases'][cid]
    out=RUN/(cid+'.json');report={'id':cid,'status':'RUNNING','backend':'reviewed CUDA + offline tables','started':time.time()}
    write(out,report)
    try:
        torch.set_num_threads(1)
        sd={s:load_species(cat['store'],i) for s,i in entry['species'].items()}
        def forbidden(*a,**k):raise AssertionError('Runtime attempted UPF/ORB parsing or radial tabulation')
        h0rebuild.upf.read_upf=forbidden;h0rebuild.orb.read_abacus_orb=forbidden
        production_io.read_upf=forbidden;production_io.read_abacus_orb=forbidden
        h0rebuild.cuda_two_center.CUDATwoCenter.__init__=forbidden
        real_popen=subprocess.Popen
        def guarded(args,*a,**k):
            words=args if isinstance(args,(tuple,list)) else args.split()
            if any(Path(str(x)).name in ('ninja','nvcc','g++','c++','gcc','cc','cmake') for x in words):raise AssertionError('Runtime compiler attempted')
            return real_popen(args,*a,**k)
        subprocess.Popen=guarded
        folder=Path(cat['raw'])/cid
        if sha256(folder/'STRU')!=entry['STRU_sha256']: raise ValueError('STRU changed after preparation')
        st,sd,opts,contract=production_io.load_case(folder,prepared_species=sd)
        counts=[sd[a.species].orb.norb*(2 if opts['nspin']==4 else 1) for a in st.atoms]
        report.update(atoms=len(st.atoms),nspin=opts['nspin'],species=list(sd),table_key=entry['table_key'])
        report.update(reader_revision='atomic-mag/v2', field_revision='ABACUS-reader-odd-PW-libxc-threshold/v4',
            initial_moments_z=opts.get('initial_moments_z'), pseudo_rcut_bohr=opts['pseudo_rcut_bohr'],
            reader_sha256=sha256(ROOT/'production_io.py'),assembly_sha256=sha256(ROOT/'h0rebuild/assemble.py'),field_preparation_sha256=sha256(ROOT/'h0rebuild/field_inputs.py'))
        write(out,report)
        torch.cuda.synchronize();start=time.perf_counter()
        got=assemble_h0(st,sd,**opts,two_center_backend='pyabacus',two_center_cache_dir=str(ROOT/'tmp'),
            offline_table_dir=cat['store'],local_integration='fft_grid_periodic',compute_device='cuda:0',
            field_backend='torch',structure_factor_backend='cufinufft',strict_reproduction=False,
            cuda_precompiled=True,pair_support='nonlocal_complete',spatial_backend='indexed',hermitize=False,
            local_cache_max_mb=4096,structure_factor_nthreads=1,structure_factor_max_work_mb=2048,field_max_work_mb=4096)
        torch.cuda.synchronize();report['assembly_seconds']=time.perf_counter()-start
        h,_=production_io.read_csr(folder/'OUT.ABACUS/data-HR0_SPIN0.csr',counts,opts['nspin'])
        s,_=production_io.read_csr(folder/'OUT.ABACUS/data-SR-sparse_SPIN0.csr',counts,opts['nspin'])
        report['H_error_eV']=production_io.compare_blocks(got.h_blocks_ry,h,counts,production_io.RY_TO_EV)
        report['S_error']=production_io.compare_blocks(got.s_blocks,s,counts)
        finite=all(np.isfinite(x).all() for blocks in (got.h_blocks_ry,got.s_blocks) for x in blocks.values())
        report['status']='PASS' if finite and report['H_error_eV']['total']['max_abs']<.005 and report['S_error']['total']['max_abs']<1e-6 else 'NUMERICAL_FAIL'
        report['runtime_forbidden']=['compilers','UPF parsing','ORB parsing','two-center tabulation']
        report['binary']={n:verify(n)['binary_sha256'] for n in ('_cuda_two_center','_cuda_local_grid')}
        report['assembly_metadata']=got.metadata
        if report['status']!='PASS':
            # Keep new component blocks for causal diagnostics, not just scalar maxima.
            arrays={};meta=[]
            for key,block in got.h_blocks_ry.items():
                if key.i==key.j and key.R==(0,0,0):
                    tag=str(key.i);arrays['h_'+tag]=block;arrays['ref_'+tag]=h.get(key,np.zeros_like(block));arrays['s_'+tag]=got.s_blocks[key]
                    for name,blocks in got.components_ry.items():
                        if key in blocks: arrays[name+'_'+tag]=blocks[key]
                    meta.append({'atom':key.i,'species':st.atoms[key.i].species})
            np.savez(RUN/(cid+'_onsite.npz'),**arrays);report['onsite_atoms']=meta
    except BaseException:
        report['status']='ERROR';report['error']=traceback.format_exc()
    report['finished']=time.time();write(out,report);print(cid,report['status'],report.get('assembly_seconds'),flush=True)

if __name__=='__main__':
    from acceptance import main
    raise SystemExit(main())
