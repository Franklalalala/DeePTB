"""Causal probe: ABACUS normalizes each species rho_at before superposition."""
import json,time,os
from pathlib import Path
import numpy as np,torch
import h0rebuild.assemble as asm
import h0rebuild.torch_fields as fields
from h0rebuild.offline import load_species
from h0rebuild.radial_quadrature import simpson_rab
from production_io import load_case,read_csr,RY_TO_EV
ROOT=Path(__file__).resolve().parent;cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
original_pairs=asm.iter_pair_images
def only_onsite(*a,**k):
    for row in original_pairs(*a,**k):
        if row[0]==row[1] and tuple(row[2])==(0,0,0):yield row
asm.iter_pair_images=only_onsite
original_transform=fields.radial_transforms_torch
def corrected(upf,*a,**k):
    if os.environ.get('H0_DIAG_CUTOFF'):
        from h0rebuild.field_inputs import prepare_field_upf
        return original_transform(prepare_field_upf(upf,float(os.environ['H0_DIAG_CUTOFF'])),*a,**k)
    q,v,c=original_transform(upf,*a,**k)
    return q*(upf.z_valence/float(simpson_rab(upf.rhoatom_q,upf.rab))),v,c
fields.radial_transforms_torch=corrected
torch.set_num_threads(1);report=[]
for cid in ['nonSOC_db_seq_id_10868','nonSOC_db_seq_id_1773','nonSOC_db_seq_id_2251','nonSOC_db_seq_id_9278','SOC_mp-561353','nonSOC_db_seq_id_15674']:
    e=cat['cases'][cid];sd={s:load_species(cat['store'],i) for s,i in e['species'].items()}
    folder=Path(cat['raw'])/cid;st,sd,opts,_=load_case(folder,prepared_species=sd)
    counts=[sd[a.species].orb.norb*(2 if opts['nspin']==4 else 1) for a in st.atoms]
    got=asm.assemble_h0(st,sd,**opts,two_center_backend='pyabacus',offline_table_dir=cat['store'],
        local_integration='fft_grid_periodic',compute_device='cuda:0',field_backend='torch',
        structure_factor_backend='cufinufft',structure_factor_max_work_mb=2048,field_max_work_mb=4096,
        strict_reproduction=False,cuda_precompiled=True,pair_support='nonlocal_complete',spatial_backend='indexed',hermitize=False,local_cache_max_mb=4096)
    ref,_=read_csr(folder/'OUT.ABACUS/data-HR0_SPIN0.csr',counts,opts['nspin'])
    maxerr=max(float(abs(v-ref[k]).max()*RY_TO_EV) for k,v in got.h_blocks_ry.items())
    row={'id':cid,'scope':'onsite causal probe only','Hmax_eV':maxerr,'charge_factors':{s:d.upf.z_valence/float(simpson_rab(d.upf.rhoatom_q,d.upf.rab)) for s,d in sd.items()}}
    name='cutoff_diagnosis.json' if os.environ.get('H0_DIAG_CUTOFF') else 'charge_diagnosis.json'
    report.append(row);(ROOT/'work'/name).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
