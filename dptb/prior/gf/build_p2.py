"""P2 radial tables (schema deeptb.p2_radial_table/v1) on the device: the deployed non-SOC P2 construction, batched.

For an ordered species pair L|R, on distances 0, step, ... (``dptb.prior.common.sbt.distance_grid`` of the support)
and with the bond-frame transform of ``dptb.prior.common.sbt.BondFrameSBT`` (trapezoid source quadrature,
k = linspace(0, kmax, n_k), exact angular coupling):
    base       <phi_L| T + V_NA,L + V_NA,R |phi_R>   (Ry; the three radial products summed before one transform)
               zero for d >= rc_L + rc_R; L == R also stores onsite_base = <T + V_NA> at d = 0 in FP64
    projector  <beta_P | phi_O>                      (Ry^(1/2); support max projector cutoff + rc_O)
V_NA is the neutral-atom potential of the UPF (local part + Hartree of the charge-normalised atomic density,
``dptb.prior.common.pporb.neutral_atom_potential_radial``) linearly interpolated onto the ORB grid and zero beyond the
UPF grid; beta = u / r of the UPF projectors (``dptb.prior.common.pporb.projector_beta_values``) inside their cutoff.
Species arrays: d_eff (the non-SOC spin-traced KB matrix, ``dptb.prior.common.kb.effective_d``) and onsite_base.
Values are stored in float32 as the deployed tables. ORB/UPF files are found by name in the ``pporb`` directories and
checked against the reference manifest's SHA256 (``species_files`` gives files for species the reference lacks).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.kb import effective_d
from dptb.prior.common.sbt import BondFrameSBT, distance_grid
from dptb.prior.common.store import TableStore, sha256_file

SCHEMA = 'deeptb.p2_radial_table/v1'
ORDERED = True
SETTINGS = dict(distance_step_bohr=0.05, kmax_bohr_inv=70.0, n_k=600, n_mu=18, n_phi=36)


def readers():
    from dptb.prior.common.pporb import read_abacus_orb, read_upf
    return read_abacus_orb, read_upf


def source_files(symbols, rows, pporb, species_files=None):
    """{symbol: (orb, upf)}: explicit files, else the reference row's file names in the pporb dirs (SHA256 checked)."""
    out = {}
    dirs = [Path(p) for p in ([pporb] if isinstance(pporb, (str, Path)) else (pporb or []))]
    for s in symbols:
        if species_files and s in species_files:
            out[s] = tuple(Path(f) for f in species_files[s])
            continue
        row = rows.get(s)
        if row is None:
            raise KeyError(f'{s}: not in the reference tables; give its ORB/UPF files')
        pick = []
        for name, digest in ((row['orbital_file'], row['orbital_sha256']), (row['upf_file'], row['upf_sha256'])):
            cands = [d / Path(name).name for d in dirs if (d / Path(name).name).is_file()]
            match = [c for c in cands if sha256_file(c) == digest]
            if not match:
                raise FileNotFoundError(f'{s}: no {Path(name).name} with the reference SHA256 in {dirs}')
            pick.append(match[0])
        out[s] = tuple(pick)
    return out


class Species:
    """ORB channels, V_NA-weighted channels and UPF projectors of one species as device transforms."""

    def __init__(self, symbol, orb_path, upf_path, sbt):
        from dptb.prior.common.pporb import neutral_atom_potential_radial, projector_beta_values
        read_abacus_orb, read_upf = readers()
        orb, upf = read_abacus_orb(orb_path), read_upf(upf_path)
        self.symbol, self.orb_path, self.upf_path, self.upf = symbol, Path(orb_path), Path(upf_path), upf
        r = np.asarray(orb.r, dtype=np.float64)
        self.rc = float(orb.rcut)
        self.shells = [int(c.l) for c in orb.channels]
        self.norb = sum(2 * l + 1 for l in self.shells)
        r_vna, vna = neutral_atom_potential_radial(upf)
        pot = np.interp(r, np.asarray(r_vna, dtype=np.float64), np.asarray(vna, dtype=np.float64),
                        left=float(np.asarray(vna)[0]), right=0.0)
        self.orb = sbt.transforms(r, [(c.l, c.radial, self.rc) for c in orb.channels])
        self.vna = sbt.transforms(r, [(c.l, np.asarray(c.radial) * pot, self.rc) for c in orb.channels])
        ru = np.asarray(upf.r, dtype=np.float64)
        proj = []
        for p in upf.projectors:
            beta = projector_beta_values(ru, np.asarray(p.radial_u, dtype=np.float64), float(p.cutoff_radius), ru)
            proj.append((int(p.l), np.where(ru <= float(p.cutoff_radius) + 1e-12, beta, 0.0), float(p.cutoff_radius)))
        self.proj = sbt.transforms(ru, proj) if proj else None
        self.proj_shells = [int(p.l) for p in upf.projectors]
        self.proj_cuts = [float(p.cutoff_radius) for p in upf.projectors]
        self.nproj = sum(2 * l + 1 for l in self.proj_shells)

    def row(self, array_path=None):
        out = dict(orbital_shells=self.shells, orbital_norb=self.norb, orbital_cutoff_bohr=self.rc,
                   projector_shells=self.proj_shells, projector_norb=self.nproj, projector_cutoffs_bohr=self.proj_cuts,
                   projector_max_cutoff_bohr=max(self.proj_cuts, default=0.0), orbital_file=str(self.orb_path),
                   orbital_sha256=sha256_file(self.orb_path), upf_file=str(self.upf_path),
                   upf_sha256=sha256_file(self.upf_path), upf_has_so=bool(self.upf.has_so), upf_metadata_repair=None)
        if array_path:
            out['array_path'] = array_path
        return out


class Builder:
    def __init__(self, reference, pporb, *, species_files=None, settings=None, device='cuda'):
        self.ref = json.loads((Path(reference) / 'manifest.json').read_text())
        self.settings = dict(SETTINGS, **{k: self.ref['build_settings'][k] for k in SETTINGS
                                          if k in self.ref.get('build_settings', {})}, **(settings or {}))
        s = self.settings
        self.sbt = BondFrameSBT(s['kmax_bohr_inv'], s['n_k'], s['n_mu'], s['n_phi'], device)
        self.pporb, self.files, self.sp = pporb, dict(species_files or {}), {}

    def species(self, s):
        if s not in self.sp:
            orb, upf = source_files([s], self.ref.get('species', {}), self.pporb, self.files)[s]
            self.sp[s] = Species(s, orb, upf, self.sbt)
        return self.sp[s]

    def base(self, left, right):
        a, b = self.species(left), self.species(right)
        support = a.rc + b.rc
        d = distance_grid(support, self.settings['distance_step_bohr'])
        terms = [(a.orb.F, b.orb.F, True), (a.vna.F, b.orb.F, False), (a.orb.F, b.vna.F, False)]
        v = self.sbt.tables(a.orb, b.orb, d, terms=terms)
        v[torch.as_tensor(d, device=v.device) >= support - 1e-12] = 0.0
        return d, v, support

    def onsite(self, s):
        a = self.species(s)
        return self.sbt.tables(a.orb, a.orb, [0.0], terms=[(a.orb.F, a.orb.F, True), (a.vna.F, a.orb.F, False)])[0]

    def projector(self, p, o):
        a, b = self.species(p), self.species(o)
        support = max(a.proj_cuts) + b.rc
        d = distance_grid(support, self.settings['distance_step_bohr'])
        v = self.sbt.tables(a.proj, b.orb, d)
        v[torch.as_tensor(d, device=v.device) >= support - 1e-12] = 0.0
        return d, v, support


def build(pairs, out_dir, *, reference, pporb, species_files=None, device='cuda', kinds=('base', 'projector'),
          compress=False, log=print):
    """Base and projector tables of ordered pairs 'L|R' into the P2 root ``out_dir`` (created or extended, files never
    replaced), plus species arrays (d_eff, onsite_base) of every species involved. Returns {kind:key: seconds}."""
    b = Builder(reference, pporb, species_files=species_files, device=device)
    st = b.settings
    store = TableStore(out_dir, SCHEMA, length_unit='bohr', energy_unit='Ry',
                       build_settings=dict(st, interpolation='cubic', dtype='float32',
                                           operator='T + unique(VNA_i,VNA_j) + factorized all-K Vnl',
                                           soc_mode='non-SOC scalar spin trace'))
    gid = store.generator('dptb.prior.gf.build_p2', 'dptb.prior.common.sbt')
    keys = [tuple(p.split('|')) if isinstance(p, str) else tuple(p) for p in pairs]
    seconds = {}
    for left, right in keys:
        key = f'{left}|{right}'
        for kind in kinds:
            section = kind + '_tables'
            if store.has(section, key) or (kind == 'projector' and not b.species(left).nproj):
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d, v, support = (b.base if kind == 'base' else b.projector)(left, right)
            values = v.to(torch.float32).cpu().numpy()
            torch.cuda.synchronize()
            seconds[f'{kind}:{key}'] = dt = time.perf_counter() - t0
            sl = b.species(left).shells if kind == 'base' else b.species(left).proj_shells
            arrays = dict(distances=d, values=values, left_shells=np.asarray(sl, dtype=np.int16),
                          right_shells=np.asarray(b.species(right).shells, dtype=np.int16),
                          support_bohr=np.asarray(support, dtype=np.float64))
            if kind == 'base' and left == right:
                arrays['onsite_base'] = b.onsite(left).cpu().numpy()
            row = store.write_npz(Path(kind) / f'{left}__{right}.npz', arrays, compress)
            store.add(section, key, dict(path=row['path'], sha256=row['sha256'], support_bohr=float(support),
                                         distance_count=len(d), interpolation='cubic', generator=gid, seconds=dt))
            log(f'P2_BUILT {kind} {key} {dt:.3f}s')
    for s in sorted({x for k in keys for x in k}):
        if s in store.manifest.get('species', {}):
            continue
        sp = b.species(s)
        rel = Path('species') / f'{s}.npz'
        row = store.write_npz(rel, dict(d_eff=effective_d(sp.upf), onsite_base=b.onsite(s).cpu().numpy()))
        store.manifest.setdefault('species', {})[s] = dict(sp.row(row['path']), array_sha256=row['sha256'],
                                                           generator=gid)
    store.flush()
    return seconds


def species_of(reference):
    return sorted(json.loads((Path(reference) / 'manifest.json').read_text())['species'])


def present(out_dir):
    path = Path(out_dir) / 'manifest.json'
    if not path.exists():
        return set()
    m = json.loads(path.read_text())
    return set(m.get('base_tables', {}))
