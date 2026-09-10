#!/usr/bin/env python3
"""Build S from the exact ORB files bound by a production P2 manifest.

Uses the existing qualified SBT builder; never modifies or recomputes P2/P23.
Requires the offline h0rebuild ORB reader. Inference has no such dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_nonsoc_p2_tables import _as_orbital_like, _sbt_context, _build_values, _distance_grid
from h0rebuild.orb import read_abacus_orb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p2-manifest', required=True)
    ap.add_argument('--orb-root', required=True)
    ap.add_argument('--species', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--distance-step', type=float, default=.02)
    args = ap.parse_args()
    raw = Path(args.p2_manifest).read_bytes()
    p2 = json.loads(raw)
    output = Path(args.output)
    if (output / 'manifest.json').exists():
        raise FileExistsError('use a new output directory for an immutable sidecar')
    output.mkdir(parents=True, exist_ok=True)
    species, orbitals = {}, {}
    for symbol in sorted(set(args.species)):
        row = p2['species'][symbol]
        path = Path(args.orb_root) / Path(row['orbital_file']).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['orbital_sha256']:
            raise ValueError(f'ORB hash mismatch for {symbol}')
        orbital = _as_orbital_like(read_abacus_orb(path), path)
        if orbital.shells != tuple(row['orbital_shells']):
            raise ValueError(f'ORB shell order mismatch for {symbol}')
        orbitals[symbol], species[symbol] = orbital, row
    settings = p2['build_settings']
    sbt = dict(kmax=settings['kmax_bohr_inv'], n_k=settings['n_k'], n_mu=settings['n_mu'], n_phi=settings['n_phi'])
    tables = {}
    for left, a in orbitals.items():
        for right, b in orbitals.items():
            support = a.rcut + b.rcut
            distances = _distance_grid(support, args.distance_step)
            values = _build_values(_sbt_context(a, b, **sbt), distances)
            path = output / f'{left}__{right}.npz'
            np.savez_compressed(path, distances=distances, values=values.astype(np.float32), left_shells=a.shells, right_shells=b.shells, support_bohr=support, onsite_overlap=values[0] if left == right else np.empty((0, 0)))
            tables[f'{left}|{right}'] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            print('BUILT', left, right, values.shape, flush=True)
    manifest = dict(schema='deeptb.overlap_radial_table/v1', complete=True,
                    source_p2_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    length_unit='bohr', value_unit='dimensionless', harmonic_convention='deeptb_abacus_real',
                    build_settings={**sbt, 'distance_step_bohr':args.distance_step},
                    species=species, tables=tables,
                    qualification='SBT construction only; validate versus independent overlap oracle before production')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
