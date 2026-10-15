"""SOC projector tables (schema deeptb.soc_projector_table/v1): the complex spinor KB matrix of each species.

Not a two-centre family: the radial AO-projector overlaps stay the real P2 projector tables, and spin-orbit coupling
lives in a per-species complex D in Ry on [up real projectors, down real projectors] (``dptb.prior.common.kb.spinor_d``).
Its spin trace (D_upup + D_dndn) / 2 equals the P2 d_eff (``effective_d``), which is checked.
UPF files are found by name in the ``pporb`` directories through the reference P2 species rows (SHA256 checked).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dptb.prior.common.kb import effective_d, spinor_d
from dptb.prior.common.store import TableStore
from dptb.prior.gf.build_p2 import readers, source_files

SCHEMA = 'deeptb.soc_projector_table/v1'
ORDERED = False
PER_SPECIES = True


def _p2_rows(reference):
    ref = Path(reference)
    for m in (ref.parent / 'p2' / 'manifest.json', ref / 'p2' / 'manifest.json', ref / 'manifest.json'):
        if m.exists():
            rows = json.loads(m.read_text()).get('species', {})
            if rows and 'upf_file' in next(iter(rows.values())):
                return rows, hashlib.sha256(m.read_bytes()).hexdigest()
    return {}, None


def build(species, out_dir, *, reference, pporb, species_files=None, device=None, compress=False, log=print):
    """Spinor D shards of ``species`` into the SOC root ``out_dir`` (created or extended, files never replaced)."""
    rows, p2_sha = _p2_rows(reference)
    store = TableStore(out_dir, SCHEMA, spin_order='spin_major', harmonic_convention='deeptb_abacus_real', unit='Ry',
                       source_p2_manifest_sha256=p2_sha)
    gid = store.generator('dptb.prior.gf.build_soc')
    seconds = {}
    for s in species:
        if store.has('species', s):
            continue
        t0 = time.perf_counter()
        upf_path = source_files([s], rows, pporb, species_files)[s][1]
        upf = readers()[1](upf_path)
        shells, js = [int(p.l) for p in upf.projectors], [p.j for p in upf.projectors]
        d = spinor_d(shells, js, upf.dij_ry, has_so=bool(upf.has_so))
        n = d.shape[0] // 2
        if not np.allclose((d[:n, :n] + d[n:, n:]) * .5, effective_d(upf), atol=1e-10, rtol=1e-10):
            raise ValueError(f'SOC spin trace does not reproduce the scalar d_eff for {s}')
        row = store.write_npz(Path(f'{s}.npz'), dict(d_spinor_ry=d), compress)
        digest = hashlib.sha256(Path(upf_path).read_bytes()).hexdigest()
        store.add('species', s, dict(path=row['path'], sha256=row['sha256'], upf_sha256=digest, projector_shells=shells,
                                     projector_j=js, upf_has_so=bool(upf.has_so), upf_metadata_repair=None,
                                     generator=gid))
        seconds[s] = time.perf_counter() - t0
        log(f'SOC_BUILT {s} {d.shape}')
    store.flush()
    return seconds


def species_of(reference):
    rows, _ = _p2_rows(reference)
    return sorted(rows or json.loads((Path(reference) / 'manifest.json').read_text())['species'])


def present(out_dir):
    path = Path(out_dir) / 'manifest.json'
    return set(json.loads(path.read_text()).get('species', {})) if path.exists() else set()
