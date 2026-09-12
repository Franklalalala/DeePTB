"""Fresh-process and retained-object tests using only the reviewed new route."""
import builtins, copy, io, json, subprocess
from pathlib import Path
import numpy as np,torch
from h0rebuild.offline import load_species,prepared_two_center,_resident
from h0rebuild.models import SpeciesData
from h0rebuild.scalar_upf import scalarize_upf
from h0rebuild.cuda_two_center import CUDATwoCenter
from h0rebuild.precompiled import ROOT,verify

cat=json.loads((ROOT/'offline_tables/catalog.json').read_text())
torch.set_num_threads(1)
names=['nonSOC_db_seq_id_15674','SOC_mp-31055','nonSOC_db_seq_id_2279','SOC_mp-1094757']
report=[]
for cid in names:
    e=cat['cases'][cid];sd={s:load_species(cat['store'],i) for s,i in e['species'].items()}
    if e['nspin']==1:sd={s:SpeciesData(d.orb,scalarize_upf(d.upf)) for s,d in sd.items()}
    # Newly prepared candidate constructor, not the legacy backend.
    fresh=CUDATwoCenter(dict(sorted(sd.items())),nspin=e['nspin'],device='cuda:0',cache_dir=str(ROOT/'tmp'))
    disk=prepared_two_center(sd,store=cat['store'],nspin=e['nspin'],device='cuda:0')
    mx=0.
    for name in ('S_coeffs','T_coeffs','Q_coeffs','D_padded_cuda','gaunt_table'):
        torch.testing.assert_close(getattr(fresh,name),getattr(disk,name),atol=0,rtol=0)
    a,b=list(sd)[0],list(sd)[-1];vec=np.array([1.2,.8,-.5])
    old=disk.scalar_pair(a,b,[0,0,0],vec)
    warm=prepared_two_center(dict(reversed(list(sd.items()))),store=cat['store'],nspin=e['nspin'],device='cuda:0')
    assert warm is disk
    for aa,bb in zip(old,warm.scalar_pair(a,b,[0,0,0],vec)):assert np.array_equal(aa,bb)
    changed=copy.deepcopy(sd);changed[a].orb.channels[0].radial[2]+=1e-8
    try:prepared_two_center(changed,store=cat['store'],nspin=e['nspin'],device='cuda:0')
    except FileNotFoundError:pass
    else:raise AssertionError('Changed radial content silently reused a table')
    report.append({'id':cid,'fresh_vs_disk_coefficients':'bitwise equal','reordered_species_reuse':True,'mutation_rejected':True})

def forbidden(*a,**k):raise AssertionError('Runtime attempted table preparation')
CUDATwoCenter.__init__=forbidden
_resident.clear()
original_open=builtins.open;original_io_open=io.open
def check_file(file):
    if isinstance(file,(str,Path)) and str(file).lower().endswith(('.upf','.orb')):raise AssertionError('Runtime read original UPF/ORB')
def guarded(file,*a,**k):check_file(file);return original_open(file,*a,**k)
def guarded_io(file,*a,**k):check_file(file);return original_io_open(file,*a,**k)
builtins.open=guarded;io.open=guarded_io
from production_io import load_case
from h0rebuild.assemble import assemble_h0
cid=names[0];st,sd,opts,contract=load_case(Path(cat['raw'])/cid)
out=assemble_h0(st,sd,**opts,two_center_backend='pyabacus',local_integration='fft_grid_periodic',compute_device='cuda:0',
    field_backend='torch',structure_factor_backend='cufinufft',strict_reproduction=False,cuda_precompiled=True,
    pair_support='nonlocal_complete',spatial_backend='indexed',hermitize=False)
assert all(np.isfinite(v).all() for v in out.h_blocks_ry.values())
result={'status':'PASS','checks':report,'full_H0_without_UPF_ORB_reads_or_tabulation':cid,
        'binary':{n:verify(n)['binary_sha256'] for n in ('_cuda_two_center','_cuda_local_grid')}}
(ROOT/'work/offline_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
