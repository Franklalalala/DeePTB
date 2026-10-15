"""GF prior tables resident on the device.

Every family is a set of radial block tables (``dptb.prior.common.radial.RadialTable``) plus per-species arrays, read
from the deployed table roots:

    kind        root / manifest rows                     key           block                       unit
    base        p2 / base_tables                         L|R           AO_L x AO_R                 Ry
    projector   p2 / projector_tables                    P|A           beta_P x AO_A               Ry^(1/2) (D_eff in Ry)
    vna         p23 (+ extensions, first root wins)      V|A           q_V x AO_A                  eV (epsilon in 1/eV)
    overlap     overlap / tables                         L|R           AO_L x AO_R                 1
    envfac      envxc / tables.envfac                    K|A           q_K x shells_A              -
    rhofac      envxc / tables.rhofac                    K|A           q_K x AO_A                  -
    envnorm, envpair, pairmom   envxc / tables.<kind>    sorted pair   shells x shells, AO x AO    -
    pairxc      pair-XC manifest / pairs                 sorted pair   AO x AO                     eV

Shards are checked against their manifest SHA256 once per process, decoded once, splined on the device and kept in
an LRU bounded in bytes. ``prefetch(symbols)`` reads (and hashes) the shards of a composition in a background thread
so the next structure's cold tables load while the current one runs.
"""
import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.radial import RadialTable

SORTED_KINDS = ('envnorm', 'envpair', 'pairmom', 'pairxc')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


# schema ids of tables written before the prior package (read entry only; stored manifests are not rewritten)
LEGACY_SCHEMAS = {'nacf-envxc/v1': 'deeptb.envxc_table/v1', 'deeptb.nacf.pair_xc/v1': 'deeptb.pair_xc_table/v1'}


def _manifest(root, schema=None):
    m = json.loads((Path(root) / 'manifest.json').read_text())
    m['schema'] = LEGACY_SCHEMAS.get(m.get('schema'), m.get('schema'))
    if schema is not None and m.get('schema') != schema:
        raise ValueError(f'{root}: table schema {m.get("schema")!r} is not {schema!r}')
    if m.get('complete') is False:
        raise ValueError(f'{root}: table manifest is not complete')
    return m


@dataclass
class Species:
    symbol: str
    shells: tuple
    norb: int
    rc: float                      # AO cutoff (bohr)
    z_valence: float
    m2: float                      # valence second moment (bohr^2)
    proj_shells: tuple = ()
    proj_cut: np.ndarray = None    # per projector channel (bohr)
    proj_max: float = -1.0
    vna_shells: tuple = ()
    vna_cut: float = 0.0
    q_shells: tuple = ()
    q_cut: float = 0.0
    arrays: dict = field(default_factory=dict)


FAMILY_SCHEMAS = dict(p2='deeptb.p2_radial_table/v1', overlap='deeptb.overlap_radial_table/v1',
                      envxc='deeptb.envxc_table/v1', pairxc='deeptb.pair_xc_table/v1',
                      soc='deeptb.soc_projector_table/v1', p23='deeptb.p23_factorized_vna_table/v1',
                      moments='deeptb.atomic_moments/v1')


class GFTables:
    """GF prior tables of the deployed roots, with an optional extension root stacked after them (``extension``:
    ``<extension>/<family>`` stores of ``dptb prior build``; the first layer holding a key wins). build_missing (off by
    default): keys of a composition no layer holds are built on the GPU into the extension before use (ORB/UPF from
    ``pporb`` directories or ``species_files``)."""

    def __init__(self, root, *, soc, pairxc, moments, p23_extensions=(), device='cuda', budget_bytes=6 << 30,
                 verify=True, extension=None, build_missing=False, pporb=(), species_files=None):
        self.root = Path(root)
        self.device = torch.device(device)
        self.verify = verify
        self.budget = int(budget_bytes)
        self.p2 = _manifest(self.root / 'p2', 'deeptb.p2_radial_table/v1')
        if self.p2.get('length_unit') != 'bohr' or self.p2.get('energy_unit') != 'Ry':
            raise ValueError('P2 tables must be in bohr and Ry')
        p23_roots = [self.root / 'p23', *map(Path, p23_extensions)]
        self.p23 = [(r, _manifest(r, 'deeptb.p23_factorized_vna_table/v1')) for r in p23_roots]
        for r, m in self.p23:
            if m.get('factor_energy_unit') != 'eV' or m.get('epsilon_unit') != '1/eV' or \
                    m.get('endpoint_policy') != 'exclude_i0_and_jR':
                raise ValueError(f'{r}: P23 contract mismatch')
        self.overlap = _manifest(self.root / 'overlap', 'deeptb.overlap_radial_table/v1')
        self.envxc = _manifest(self.root / 'envxc', 'deeptb.envxc_table/v1')
        if self.envxc.get('background_nodes') is not None:
            raise ValueError('environment-XC background tables are not used by this recipe')
        self.pairxc_path = Path(pairxc)
        self.pairxc = json.loads(self.pairxc_path.read_text())
        self.soc_root = Path(soc)
        self.soc = _manifest(self.soc_root, 'deeptb.soc_projector_table/v1')
        if self.soc.get('spin_order') != 'spin_major' or self.soc.get('unit') != 'Ry':
            raise ValueError('SOC projector tables must be spin-major and in Ry')
        # the deployed moments file is a flat {species: row} JSON; extension stores keep the rows under 'species'
        self.layers = dict(p2=[(self.root / 'p2', self.p2)], overlap=[(self.root / 'overlap', self.overlap)],
                           envxc=[(self.root / 'envxc', self.envxc)], pairxc=[(self.pairxc_path.parent, self.pairxc)],
                           soc=[(self.soc_root, self.soc)], p23=list(self.p23),
                           moments=[(Path(moments).parent, dict(species=json.loads(Path(moments).read_text())))])
        self.extension = Path(extension) if extension else None
        self.build_missing, self.pporb, self.species_files = build_missing, list(pporb), species_files
        self._stack_extension()
        self._tables = OrderedDict()
        self._bytes = 0
        self._species = {}
        self._raw = {}                   # key -> decoded host arrays (read ahead by prefetch)
        self._verified = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='gf-tables')
        self._pending = {}

    # ------------------------------------------------------------------ shards
    def _read(self, path, digest):
        path = Path(path)
        if self.verify and path not in self._verified:
            if sha256(path) != digest:
                raise ValueError(f'table shard checksum mismatch: {path}')
            self._verified.add(path)
        with np.load(path, allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in z.files if k not in ('metadata', 'identity')}

    # ------------------------------------------------------------------ layers
    def _stack_extension(self):
        if self.extension is None:
            return
        for fam, schema in FAMILY_SCHEMAS.items():
            r = self.extension / fam
            if (r / 'manifest.json').exists():
                self.layers[fam] = [x for x in self.layers[fam] if x[0] != r] + [(r, _manifest(r, schema))]

    def _find(self, fam, section, key):
        """(layer root, row, layer manifest) of the first layer of family ``fam`` holding ``key`` in ``section`` (a
        manifest key or a path of keys)."""
        for root, m in self.layers[fam]:
            rows = m
            for part in (section if isinstance(section, tuple) else (section,)):
                rows = rows.get(part, {})
            if key in rows:
                return root, rows[key], m
        raise KeyError(f'{fam} table missing: {section} {key}')

    def _has(self, fam, section, key):
        try:
            self._find(fam, section, key)
            return True
        except KeyError:
            return False

    def _build_missing(self, symbols):
        """Build every table of the composition no layer holds into the extension (``dptb.prior.build``)."""
        from dptb.prior.build import build_missing
        want = {}
        for a in symbols:
            for fam in ('soc', 'moments'):
                if not self._has(fam, 'species', a):
                    want.setdefault(fam, []).append(a)
            for b in symbols:
                k = f'{a}|{b}'
                for fam, section, key in (('p2', 'base_tables', k), ('overlap', 'tables', k),
                                          ('p23', 'factor_tables', k), ('envxc', ('tables', 'envfac'), k),
                                          ('envxc', ('tables', 'rhofac'), k)):
                    if not self._has(fam, section, key):
                        want.setdefault(fam, []).append(key if fam != 'envxc' else f'{section[1]}:{key}')
                if a <= b:
                    if not self._has('pairxc', 'pairs', k):
                        want.setdefault('pairxc', []).append(k)
                    for kind in ('envnorm', 'envpair', 'pairmom'):
                        if not self._has('envxc', ('tables', kind), k):
                            want.setdefault('envxc', []).append(f'{kind}:{k}')
        if not want:
            return
        refs = dict(p2=self.root, overlap=self.root, p23=self.root, envxc=self.root, soc=self.root / 'p2',
                    moments=self.root / 'p2', pairxc=self.pairxc_path.parent)
        with self._lock:
            for fam, keys in want.items():
                build_missing(fam, list(dict.fromkeys(keys)), self.extension, reference=refs[fam], pporb=self.pporb,
                              species_files=self.species_files, device=self.device)
            self._stack_extension()
            self._species = {s: v for s, v in self._species.items() if s not in symbols}

    def _row(self, kind, left, right):
        if kind in SORTED_KINDS:
            left, right = sorted((left, right))
        key = f'{left}|{right}'
        if kind in ('base', 'projector'):
            root, row, _ = self._find('p2', f'{kind}_tables', key)
            return root, row, 'values', row.get('interpolation', 'cubic')
        if kind == 'vna':
            root, row, m = self._find('p23', 'factor_tables', key)
            return root, row, 'values', m.get('interpolation', 'cubic')
        if kind == 'overlap':
            root, row, _ = self._find('overlap', 'tables', key)
            return root, row, 'values', 'cubic'
        if kind == 'pairxc':
            root, row, _ = self._find('pairxc', 'pairs', key)
            if row.get('passed') is False:
                raise ValueError(f'pair-XC table {key} did not pass its build check')
            return root, dict(path=row['file'], sha256=row['sha256']), 'values_eV', 'cubic'
        root, row, m = self._find('envxc', ('tables', kind), key)
        return root, row, None, m.get('interpolation', 'cubic')

    def _decode(self, kind, left, right):
        base, row, name, interp = self._row(kind, left, right)
        z = self._read(Path(base) / row['path'], row['sha256'])
        if name is None:
            name = 'values_eV' if 'values_eV' in z else 'values'
        out = dict(distances=z['distances'], values=z[name], left=tuple(int(x) for x in z['left_shells']),
                   right=tuple(int(x) for x in z['right_shells']), support=float(z['support_bohr']), interp=interp)
        if kind == 'overlap' and left == right:
            out['onsite'] = np.asarray(z['onsite_overlap'], dtype=np.float64)
        return out

    def _raw_entry(self, key):
        with self._lock:
            fut = self._pending.pop(key, None)
        if fut is not None:
            return fut.result()
        return self._decode(*key)

    # ------------------------------------------------------------------ tables
    def table(self, kind, left, right):
        if kind in SORTED_KINDS:
            left, right = sorted((left, right))
        key = (kind, left, right)
        t = self._tables.get(key)
        if t is not None:
            self._tables.move_to_end(key)
            return t
        raw = self._raw_entry(key)
        t = RadialTable(raw['distances'], raw['values'], raw['left'], raw['right'], raw['support'], raw['interp'],
                        device=self.device)
        if 'onsite' in raw:
            t.onsite = torch.as_tensor(raw['onsite'], device=self.device)
        self._tables[key] = t
        self._bytes += t.nbytes
        while self._bytes > self.budget and len(self._tables) > 1:
            _, old = self._tables.popitem(last=False)
            self._bytes -= old.nbytes
        return t

    def kinds_for(self, symbols):
        """(kind, left, right) of every table a structure of these species can use."""
        symbols = sorted(set(symbols))
        out = []
        for a in symbols:
            for b in symbols:
                out += [('base', a, b), ('overlap', a, b), ('vna', a, b), ('envfac', a, b), ('rhofac', a, b)]
                if int(self._find('p2', 'species', a)[1].get('projector_norb', 0)):
                    out.append(('projector', a, b))
                if a <= b:
                    out += [(k, a, b) for k in SORTED_KINDS]
        return out

    def prefetch(self, symbols):
        """Read and hash the shards of a composition in the background (decoded host arrays wait for ``table``)."""
        with self._lock:
            for key in self.kinds_for(symbols):
                if key not in self._tables and key not in self._pending:
                    self._pending[key] = self._pool.submit(self._decode, *key)

    def load(self, symbols):
        """Every table of a composition resident on the device (missing ones built first when switched on)."""
        if self.build_missing and self.extension is not None:
            self._build_missing(sorted(set(symbols)))
        self.prefetch(symbols)
        for key in self.kinds_for(symbols):
            self.table(*key)

    # ------------------------------------------------------------------ species
    def species(self, s):
        if s in self._species:
            return self._species[s]
        p2_root, p2, _ = self._find('p2', 'species', s)
        p23_root, p23, _ = self._find('p23', 'species', s)
        ex_root, ex, _ = self._find('envxc', 'species', s)
        shells = tuple(int(l) for l in p2['orbital_shells'])
        for other in (p23, ex):
            if tuple(int(l) for l in other['orbital_shells']) != shells or \
                    float(other['orbital_cutoff_bohr']) != float(p2['orbital_cutoff_bohr']):
                raise ValueError(f'{s}: orbital contract differs between table families')
        arr = self._read(p2_root / p2['array_path'], p2['array_sha256'])
        nproj = int(p2.get('projector_norb', 0))
        sp = Species(symbol=s, shells=shells, norb=int(p2['orbital_norb']), rc=float(p2['orbital_cutoff_bohr']),
                     z_valence=float(ex['z_valence']), m2=float(self._find('moments', 'species', s)[1]['M2_bohr2']),
                     proj_shells=tuple(int(l) for l in p2.get('projector_shells', ())),
                     proj_cut=np.asarray(p2.get('projector_cutoffs_bohr', ()), dtype=np.float64),
                     proj_max=float(p2['projector_max_cutoff_bohr']) if nproj else -1.0,
                     vna_shells=tuple(int(l) for l in p23['vna_projector_shells']), vna_cut=float(p23['vna_cutoff_bohr']),
                     q_shells=tuple(int(l) for l in ex['q_shells']), q_cut=float(ex['q_cutoff_bohr']))
        dev = self.device
        sp.arrays['onsite_base'] = torch.as_tensor(np.asarray(arr['onsite_base'], dtype=np.float64), device=dev)
        if nproj:
            sp.arrays['d_eff'] = torch.as_tensor(np.asarray(arr['d_eff'], dtype=np.float64), device=dev)
            soc_root, row, _ = self._find('soc', 'species', s)
            if row['upf_sha256'] != p2['upf_sha256'] or row['projector_shells'] != p2['projector_shells']:
                raise ValueError(f'SOC projector sources disagree for {s}')
            d = self._read(soc_root / row['path'], row['sha256'])['d_spinor_ry'].astype(np.complex128)
            if d.shape != (2 * nproj, 2 * nproj) or not np.allclose(d, d.conj().T, atol=1e-11, rtol=1e-11):
                raise ValueError(f'invalid SOC D matrix for {s}')
            sp.arrays['d_spinor'] = torch.as_tensor(d, device=dev)
        a23 = self._read(p23_root / p23['array_path'], p23['array_sha256'])
        # the P23 reader keeps epsilon in float32: the factorised VNA uses the rounded values
        sp.arrays['eps_vna'] = torch.as_tensor(np.asarray(a23['epsilon_ao'], dtype=np.float32).astype(np.float64), device=dev)
        aex = self._read(ex_root / ex['array_path'], ex['array_sha256'])
        eps = np.asarray(aex['epsilon_ao'], dtype=np.float64)
        if eps.shape != (int(ex['q_norb']),) or np.any(eps <= 0):
            raise ValueError(f'{s}: invalid environment-XC epsilon')
        sp.arrays['eps_q'] = torch.as_tensor(eps, device=dev)
        sp.arrays['onsite_overlap'] = self.table('overlap', s, s).onsite
        self._species[s] = sp
        return sp
