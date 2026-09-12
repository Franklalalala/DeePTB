"""Immutable overlap-only sidecar for legacy P2 tables without S arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

from dptb.data.interfaces.p2_table import RadialBlockTable


class OverlapTableStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        path = self.root / 'manifest.json'
        self.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = json.loads(path.read_text())
        if self.manifest.get('schema') != 'deeptb.overlap_radial_table/v1' or self.manifest.get('complete') is not True:
            raise ValueError('incomplete or unsupported overlap table manifest')
        if self.manifest.get('length_unit') != 'bohr' or self.manifest.get('value_unit') != 'dimensionless':
            raise ValueError('invalid overlap table units')
        if self.manifest.get('harmonic_convention') != 'deeptb_abacus_real':
            raise ValueError('invalid overlap harmonic convention')
        self.species = self.manifest['species']
        self._tables, self._onsite = {}, {}

    def base_component(self, left, right, component):
        if component != 'overlap':
            raise ValueError('overlap sidecar only contains overlap')
        key = f'{left}|{right}'
        if key not in self._tables:
            meta = self.manifest['tables'][key]
            path = (self.root / meta['path']).resolve()
            if self.root not in path.parents:
                raise ValueError('table path escapes its root')
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError('overlap shard checksum mismatch')
            with np.load(path, allow_pickle=False) as z:
                for side, symbol in (('left', left), ('right', right)):
                    if tuple(z[f'{side}_shells']) != tuple(self.species[symbol]['orbital_shells']):
                        raise ValueError('overlap shard shell order disagrees with manifest')
                self._tables[key] = RadialBlockTable(z['distances'], z['values'], tuple(z['left_shells']), tuple(z['right_shells']), float(z['support_bohr']), 'cubic')
                if left == right:
                    onsite = np.asarray(z['onsite_overlap'], dtype=np.float64)
                    width = int(self.species[left]['orbital_norb'])
                    if onsite.shape != (width, width) or not np.isfinite(onsite).all():
                        raise ValueError('invalid onsite overlap array')
                    self._onsite[left] = onsite
        return self._tables[key]

    def onsite_component(self, symbol, component):
        self.base_component(symbol, symbol, component)
        return self._onsite[symbol]
