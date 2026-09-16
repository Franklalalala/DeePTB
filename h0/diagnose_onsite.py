import json,time
from pathlib import Path
import numpy as np,torch
import h0rebuild.assemble as asm
from h0rebuild.offline import load_species
from production_io import load_case,read_csr,RY_TO_EV
ROOT=Path(__file__).resolve().parent
cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
original=asm.iter_pair_images
def only_onsite(*a,**k):
    for row in original(*a,**k):
        if row[0]==row[1] and tuple(row[2])==(0,0,0):yield row
asm.iter_pair_images=only_onsite
torch.set_num_threads(1)
report=[]
for cid in ['SOC_mp-561353','SOC_mp-510294','nonSOC_db_seq_id_2251','nonSOC_db_seq_id_9278']:
    entry=cat['cases'][cid];sd={s:load_species(cat['store'],i) for s,i in entry['species'].items()}
    folder=Path(cat['raw'])/cid;st,sd,opts,contract=load_case(folder,prepared_species=sd)
    counts=[sd[a.species].orb.norb*(2 if opts['nspin']==4 else 1) for a in st.atoms]
    t=time.perf_counter()
    got=asm.assemble_h0(st,sd,**opts,two_center_backend='pyabacus',offline_table_dir=cat['store'],
        local_integration='fft_grid_periodic',compute_device='cuda:0',field_backend='torch',
        structure_factor_backend='cufinufft',strict_reproduction=False,cuda_precompiled=True,
        pair_support='nonlocal_complete',spatial_backend='indexed',hermitize=False,local_cache_max_mb=4096)
    ref,_=read_csr(folder/'OUT.ABACUS/data-HR0_SPIN0.csr',counts,opts['nspin'])
    rows=[];arrays={}
    for key,h in got.h_blocks_ry.items():
        delta=h-ref[key];ix=np.unravel_index(np.argmax(abs(delta)),delta.shape)
        row={'atom':key.i,'species':st.atoms[key.i].species,'max_eV':float(abs(delta[ix])*RY_TO_EV),
             'index':list(map(int,ix)),'signed_error_eV':float(delta[ix].real*RY_TO_EV),
             'diag_eV':(np.diag(delta).real*RY_TO_EV).tolist()}
        rows.append(row)
        arrays['h_'+str(key.i)]=h;arrays['ref_'+str(key.i)]=ref[key]
        for name,blocks in got.components_ry.items():arrays[name+'_'+str(key.i)]=blocks[key]
    np.savez(ROOT/'work'/(cid+'_diagnostic.npz'),**arrays)
    report.append({'id':cid,'scope':'onsite only diagnostic; not full acceptance','seconds':time.perf_counter()-t,'rows':rows})
    (ROOT/'work/onsite_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(cid,'DONE',max(rows,key=lambda r:r['max_eV']),flush=True)
