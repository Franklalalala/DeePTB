"""Explicit offline preparation for an immutable cohort; no H/S oracle reads."""
import argparse, json, time, traceback
from acceptance import write
from pathlib import Path
from h0rebuild.offline import prepare_species, load_species, prepared_two_center
from h0rebuild.models import SpeciesData
from h0rebuild.scalar_upf import scalarize_upf
from h0rebuild.precompiled import sha256

def prepare(raw, store):
    raw, store = Path(raw), Path(store); store.mkdir(parents=True, exist_ok=True)
    from h0rebuild.table_contract import current, verify_contract, write_contract, ensure_store
    ensure_store(store)
    numerical_sources = current()
    catalog = {'schema': 1, 'raw': str(raw), 'store': str(store), 'cases': {}, 'errors': {}}
    memo = {}
    for folder in sorted(raw.iterdir()):
        if not (folder/'STRU').exists(): continue
        try:
            lines = [x.split('#')[0].strip() for x in (folder/'STRU').read_text().splitlines()]
            lines = [x for x in lines if x]
            a,b,c = [lines.index(x) for x in ('ATOMIC_SPECIES','NUMERICAL_ORBITAL','LATTICE_CONSTANT')]
            ids = {}; sd = {}
            for entry,orb in zip(lines[a+1:b],lines[b+1:c]):
                symbol,_,upf = entry.split(); op,up = folder/'PP_ORB'/orb,folder/'PP_ORB'/upf
                content = (sha256(op),sha256(up))
                if content not in memo: memo[content] = prepare_species(op,up,store)
                ids[symbol] = memo[content]; sd[symbol] = load_species(store,ids[symbol])
            spin = 4 if folder.name.startswith('SOC_') else 1
            reduced = {s: SpeciesData(d.orb,scalarize_upf(d.upf)) for s,d in sd.items()} if spin==1 else sd
            obj = prepared_two_center(reduced,store=store,nspin=spin,device='cuda:0',prepare=True)
            catalog['cases'][folder.name] = {'species': ids,'nspin':spin,'table_key':obj.metadata['offline_key'], 'STRU_sha256':sha256(folder/'STRU')}
            print(folder.name, 'PREPARED', flush=True)
        except Exception:
            catalog['errors'][folder.name] = traceback.format_exc(); print(folder.name, catalog['errors'][folder.name], flush=True)
        write(store/'catalog.json',catalog)
    print(json.dumps({'prepared':len(catalog['cases']), 'errors':len(catalog['errors']), 'unique_species_inputs':len(memo)}),flush=True)
    if current() != numerical_sources: raise RuntimeError('Numerical source changed during offline preparation')
    write_contract(store)
    return catalog

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('raw');p.add_argument('store');a=p.parse_args();prepare(a.raw,a.store)
