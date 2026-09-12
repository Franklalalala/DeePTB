"""Bind prepared tables to their Python numerical dependencies as well as ABI."""
import json
from pathlib import Path
from .precompiled import sha256
FILES=('pyabacus_integrals.py','models.py','nonlocal_kb.py','harmonics.py','upf.py','orb.py','radial_quadrature.py','radial.py')
def current():
    root=Path(__file__).parent
    return {name:sha256(root/name) for name in FILES}
def write_contract(store):
    path=Path(store)/'source_contract.json';path.write_text(json.dumps(current(),indent=2)+'\n')
def verify_contract(store):
    path=Path(store)/'source_contract.json'
    if not path.exists() or json.loads(path.read_text())!=current():
        raise RuntimeError('Prepared H0 numerical source contract missing/changed; explicitly prepare tables for the selected implementation')
