"""Overlap radial tables (schema deeptb.overlap_radial_table/v1) on the device.

For an ordered species pair L|R on distances 0, step, ... (``dptb.prior.common.sbt.distance_grid`` of rc_L + rc_R):
    S_ab(d) = <phi_L,a | phi_R,b>(d along +z)
by the bond-frame transform of ``dptb.prior.common.sbt.BondFrameSBT`` (the construction of the P2 tables without the
kinetic and V_NA terms), zero for d >= rc_L + rc_R and for m_a != m_b; L == R also stores onsite_overlap = S(0) in FP64.
Values are float32 as the deployed tables. ORB files are found by name in the ``pporb`` directories and checked against
the reference manifest's SHA256 (``species_files`` gives files for species the reference lacks).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.sbt import BondFrameSBT, distance_grid
from dptb.prior.common.store import TableStore, sha256_file
from dptb.prior.gf.build_p2 import readers, source_files

SCHEMA = 'deeptb.overlap_radial_table/v1'
ORDERED = True
SETTINGS = dict(kmax=70.0, n_k=600, n_mu=18, n_phi=36, distance_step_bohr=0.02)


class Builder:
    def __init__(self, reference, pporb, *, species_files=None, device='cuda'):
        self.ref = json.loads((Path(reference) / 'manifest.json').read_text())
        self.settings = dict(SETTINGS, **{k: self.ref['build_settings'][k] for k in SETTINGS
                                          if k in self.ref.get('build_settings', {})})
        s = self.settings
        self.sbt = BondFrameSBT(s['kmax'], s['n_k'], s['n_mu'], s['n_phi'], device)
        self.pporb, self.files, self.sp = pporb, dict(species_files or {}), {}

    def species(self, s):
        if s not in self.sp:
            orb_path, upf_path = source_files([s], self.ref.get('species', {}), self.pporb, self.files)[s]
            orb = readers()[0](orb_path)
            rc = float(orb.rcut)
            ch = self.sbt.transforms(np.asarray(orb.r, dtype=np.float64), [(c.l, c.radial, rc) for c in orb.channels])
            row = dict(orbital_file=str(orb_path), orbital_sha256=sha256_file(orb_path), orbital_cutoff_bohr=rc,
                       orbital_shells=[int(c.l) for c in orb.channels],
                       orbital_norb=sum(2 * int(c.l) + 1 for c in orb.channels))
            self.sp[s] = (ch, rc, row)
        return self.sp[s]

    def table(self, left, right):
        (a, ra, _), (b, rb, _) = self.species(left), self.species(right)
        support = ra + rb
        d = distance_grid(support, self.settings['distance_step_bohr'])
        return d, self.sbt.tables(a, b, d), support


def build(pairs, out_dir, *, reference, pporb, species_files=None, device='cuda', compress=False, log=print):
    """Overlap tables of ordered pairs 'L|R' into the overlap root ``out_dir`` (created or extended, files never
    replaced). Returns {key: seconds}."""
    b = Builder(reference, pporb, species_files=species_files, device=device)
    store = TableStore(out_dir, SCHEMA, length_unit='bohr', value_unit='dimensionless',
                       harmonic_convention='deeptb_abacus_real',
                       build_settings={k: b.settings[k] for k in SETTINGS})
    gid = store.generator('dptb.prior.gf.build_overlap', 'dptb.prior.common.sbt')
    keys = [tuple(p.split('|')) if isinstance(p, str) else tuple(p) for p in pairs]
    seconds = {}
    for left, right in keys:
        key = f'{left}|{right}'
        if store.has('tables', key):
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d, v, support = b.table(left, right)
        values = v.to(torch.float32).cpu().numpy()
        onsite = v[0].cpu().numpy() if left == right else np.empty((0, 0))
        torch.cuda.synchronize()
        seconds[key] = dt = time.perf_counter() - t0
        row = store.write_npz(Path(f'{left}__{right}.npz'), dict(
            distances=d, values=values, left_shells=np.asarray(b.species(left)[2]['orbital_shells'], dtype=np.int64),
            right_shells=np.asarray(b.species(right)[2]['orbital_shells'], dtype=np.int64),
            support_bohr=np.asarray(support, dtype=np.float64), onsite_overlap=onsite), compress)
        store.add('tables', key, dict(path=row['path'], sha256=row['sha256'], generator=gid, seconds=dt))
        log(f'OVERLAP_BUILT {key} {dt:.3f}s')
    for s in sorted({x for k in keys for x in k}):
        store.manifest.setdefault('species', {}).setdefault(s, b.species(s)[2])
    store.flush()
    return seconds


def species_of(reference):
    return sorted(json.loads((Path(reference) / 'manifest.json').read_text())['species'])


def present(out_dir):
    path = Path(out_dir) / 'manifest.json'
    return set(json.loads(path.read_text()).get('tables', {})) if path.exists() else set()
