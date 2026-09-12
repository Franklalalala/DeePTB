"""Numerical generator identity, enforced at the lowest table API."""
import json, os, tempfile
from pathlib import Path
from .precompiled import sha256
FILES=('pyabacus_integrals.py','models.py','nonlocal_kb.py','harmonics.py','upf.py','orb.py',
       'radial_quadrature.py','radial.py','cuda_two_center.py','scalar_upf.py','offline.py','table_contract.py')

def current():
    root=Path(__file__).parent
    return {name:sha256(root/name) for name in FILES}

def write_contract(store):
    # A manifest is an immutable generator claim, never an upgrade stamp.
    path=Path(store)/'source_contract.json'
    if path.exists():
        verify_contract(store); return
    if any(p.is_file() for p in Path(store).rglob('*')):
        raise RuntimeError('Nonempty unmanifested H0 store: rebuild in a new namespace')
    with tempfile.NamedTemporaryFile(mode='w',dir=store,delete=False) as f:
        json.dump(current(),f,sort_keys=True); temp=f.name
    os.replace(temp,path)

def verify_contract(store):
    path=Path(store)/'source_contract.json'
    if not path.exists() or json.loads(path.read_text())!=current():
        raise RuntimeError('Prepared H0 numerical source contract missing/changed; explicitly prepare a new store')

def ensure_store(store):
    Path(store).mkdir(parents=True,exist_ok=True)
    if not (Path(store)/'source_contract.json').exists(): write_contract(store)
    verify_contract(store)
