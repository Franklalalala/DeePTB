"""H0 pair bank (schema h0-physical-pair-bank/v1): the two-centre tables of a composition on the device.

The bank holds, per unordered species pair, the S, T and projector-overlap Q tables (index maps [l1, z1, l2, z2, L] into
rows of cubic coefficients of S_L(R)/R^L on the common grid), the orbital and projector descriptors and the spinor KB
matrices of each species, and the real Gaunt table; ``species/`` holds the prepared species data (ORB, aligned UPF).
Pair files are checked against their manifest SHA256 and the bank's physical contract once per process, and decoded
records stay in a host LRU. ``composition`` stitches the records of a species set into one index map and coefficient
array per kind, as ``TwoCentre`` evaluators (``dptb.prior.common.twocentre``).
"""
import hashlib
import itertools
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.twocentre import TwoCentre

SCHEMA = 'h0-physical-pair-bank/v1'


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Composition:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class PairBank:
    """A deployed pair bank plus extension banks of the same contract stacked after it (``extensions``,
    ``add_layer``); the first layer holding a pair or species wins."""

    def __init__(self, root, *, device='cuda', host_bytes=16 << 30, nspin=4, dr_bohr=0.02, extensions=()):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if m.get('schema') != SCHEMA:
            raise ValueError(f'{root}: not an H0 pair bank ({m.get("schema")!r})')
        if m['nspin'] != nspin or abs(m.get('actual_dr_bohr', m['dr_bohr']) - dr_bohr) > 1e-14:
            raise ValueError('H0 pair bank spin or grid differs from the requested contract')
        self.catalog = json.loads((self.root / 'catalog.json').read_text())
        self.device = torch.device(device)
        self.dr, self.nr, self.cutoff = float(m.get('actual_dr_bohr', m['dr_bohr'])), int(m['nr']), float(m['cutoff_bohr'])
        self.host_bytes = host_bytes
        self._records = OrderedDict()
        self._bytes = 0
        self._species = {}
        self._compositions = OrderedDict()
        self.layers = [(self.root, self.manifest, self.catalog)]
        for e in extensions:
            self.add_layer(e)

    def add_layer(self, root):
        """Stack (or re-read) the extension bank at ``root``; absent roots are skipped."""
        root = Path(root)
        if not (root / 'manifest.json').exists():
            return
        m = json.loads((root / 'manifest.json').read_text())
        if m.get('schema') != SCHEMA or m.get('physical_contract_sha256') != self.manifest['physical_contract_sha256'] \
                or int(m['nr']) != self.nr or float(m['cutoff_bohr']) != self.cutoff or m['nspin'] != self.manifest['nspin']:
            raise ValueError(f'{root}: extension bank contract differs from {self.root}')
        cat = json.loads((root / 'catalog.json').read_text())
        self.layers = [x for x in self.layers if x[0] != root] + [(root, m, cat)]
        self._compositions.clear()

    def has(self, key):
        return any(key in m['pairs'] for _, m, _ in self.layers)

    def _layer(self, section, key):
        for layer in self.layers:
            if key in layer[1].get(section, {}):
                return layer
        raise FileNotFoundError(f'H0 pair bank has no {section} entry {key}')

    def species(self, s):
        """Prepared species data (ORB, aligned UPF) of the bank."""
        if s not in self._species:
            from dptb.prior.h0.records import bank_species
            sd = bank_species(*self._layer('species', s), s)   # file sha, fingerprint, ORB/UPF sources
            self._species[s] = sd
        return self._species[s]

    def record(self, a, b):
        from dptb.prior.h0.records import fingerprint, read
        key = '|'.join(sorted({a, b}))
        if key in self._records:
            self._records.move_to_end(key)
            return self._records[key][0]
        root, manifest, _ = self._layer('pairs', key)
        row = manifest['pairs'][key]
        path = root / 'pairs' / row['file']
        if _sha256(path) != row['sha256']:
            raise ValueError(f'H0 pair bank file checksum mismatch: {key}')
        rec = read(path)
        checksum = rec.pop('checksum')
        if fingerprint(rec) != checksum or rec['physical_contract_sha256'] != self.manifest['physical_contract_sha256']:
            raise ValueError(f'H0 pair bank contract mismatch: {key}')
        nbytes = sum(d['index'].numel() * d['index'].element_size() + d['coeffs'].numel() * d['coeffs'].element_size()
                     for blocks in rec['blocks'].values() for d in blocks.values())
        self._records[key] = (rec, nbytes)
        self._bytes += nbytes
        while self._bytes > self.host_bytes and len(self._records) > 1:
            _, (_, old) = self._records.popitem(last=False)
            self._bytes -= old
        return rec

    def composition(self, symbols):
        """Two-centre evaluators (S, T, Q), descriptors and spinor D matrices of a species set (kept for 8 sets)."""
        species = sorted(set(symbols))
        key = tuple(species)
        if key in self._compositions:
            self._compositions.move_to_end(key)
            return self._compositions[key]
        index = {s: i for i, s in enumerate(species)}
        atoms, groups, gaunt = {}, {k: [] for k in ('S', 'T', 'Q')}, None
        for a, b in itertools.combinations_with_replacement(species, 2):
            rec = self.record(a, b)
            atoms.update(rec['atoms'])
            gaunt = rec['gaunt_table']
            for kind in groups:
                groups[kind].extend(rec['blocks'][kind].items())
        dev = self.device
        maps = {}
        for kind, blocks in groups.items():
            shape = np.max(np.asarray([list(x['index'].shape) for _, x in blocks]), axis=0).tolist()
            idx = torch.full((len(species), shape[0], shape[1], len(species), *shape[2:]), -1, dtype=torch.int64)
            coefficients, offset = [], 0
            for pair, data in blocks:
                a, b = pair.split('|')
                v = data['index'].to(torch.int64).clone()
                v[v >= 0] += offset
                z = list(v.shape)
                idx[index[a], :z[0], :z[1], index[b], :z[2], :z[3], :z[4]] = v
                coefficients.append(data['coeffs'])
                offset += data['coeffs'].shape[0]
            maps[kind] = (idx.to(dev), torch.cat(coefficients).to(dev, torch.float64))
        orb = [[tuple(int(x) for x in d) for d in atoms[s]['desc']['orb']] for s in species]
        proj = [[tuple(int(x) for x in d) for d in atoms[s]['desc']['proj']] for s in species]
        g = gaunt.to(dev, torch.float64)
        two = lambda kind, left, right: TwoCentre(maps[kind][0], maps[kind][1], g, left, right,  # noqa: E731
                                                  self.dr, self.nr, self.cutoff)
        comp = Composition(species=species, S=two('S', orb, orb), T=two('T', orb, orb), Q=two('Q', proj, orb),
                           norb=[len(x) for x in orb], nproj=[len(x) for x in proj],
                           d={s: torch.as_tensor(np.asarray(atoms[s]['d']), dtype=torch.complex128, device=dev)
                              for s in species})
        self._compositions[key] = comp
        while len(self._compositions) > 8:
            self._compositions.popitem(last=False)
        return comp
