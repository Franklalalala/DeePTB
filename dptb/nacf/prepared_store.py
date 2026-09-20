"""Explicit offline compact radial snapshots, with no dense runtime rebuild.

The source manifests identify immutable numerical inputs. A snapshot contains
losslessly compressed FP64 buffers or compact nodal values. Cold loading the
nodal encoding reconstructs spline coefficients and checks their exact hash;
it does not read dense source arrays or perform spatial quadrature. Preparing
a snapshot is a separate, explicit operation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from .prepared import ATTRS
from .radial import TorchRadialBlockTable

SCHEMA = 'nacf-prepared-radial-store/v2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def radial_identity():
    return sha256(Path(__file__).with_name('radial.py'))


def source_bindings(bank):
    return {name: sha256(store.root / 'manifest.json') for name, store in
            (('p2', bank.p2), ('p23', bank.p23), ('overlap', bank.overlap))}


def write_table(root, key, table, *, source=None):
    """Save a compiled CPU table once; caller publishes the final manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if table.knots.dtype != torch.float64:
        raise ValueError('offline snapshots require FP64 source buffers')
    arrays = {name: value.detach().cpu().numpy() for name, value in table.named_buffers()}
    # Keep only the nonzero axial channels. Reconstruct the exact source spline
    # from compact nodal values when possible, rather than storing four FP64
    # polynomial coefficients per interval. The bitwise gate is uniform and
    # never consults a species name or a Hamiltonian label.
    encoding = 'coefficients'
    if source is not None:
        nodes = np.asarray(source.values).reshape(len(source.distances), -1)[:, arrays['active_columns']]
        mode = 'linear' if source._spline is None else 'not-a-knot'
        candidate = _coefficients(source.distances, nodes, mode)
        if np.array_equal(candidate, arrays['coefficients']):
            small = nodes.astype(np.float32)
            if np.array_equal(small.astype(np.float64), nodes):
                nodes = small
            arrays['__nodes__'] = nodes
            arrays['__spline_mode__'] = np.asarray(mode)
            arrays['__coefficient_sha256__'] = np.asarray(hashlib.sha256(arrays['coefficients'].tobytes()).hexdigest())
            del arrays['coefficients']
            encoding = 'exact_compact_nodes'
    arrays['__attrs__'] = np.asarray(json.dumps({name: getattr(table, name) for name in ATTRS}))
    identity = hashlib.sha256(key.encode()).hexdigest()
    path = root / (identity + '.npz')
    if path.exists():
        raise FileExistsError('prepared table already exists; verify and reuse its receipt')
    pending = path.with_suffix('.pending')
    owned_pending = False
    try:
        with pending.open('xb') as handle:
            owned_pending = True
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish without replacing an immutable table created by another writer.
        # Both names are on the same filesystem; link creation is atomic.
        os.link(pending, path)
    finally:
        if owned_pending and pending.exists():
            pending.unlink()
    return {'path': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'buffer_bytes': sum(v.numel() * v.element_size() for v in table.buffers()),
            'encoding': encoding}


def _coefficients(knots, nodes, mode):
    nodes = np.asarray(nodes, dtype=np.float64)
    if mode == 'not-a-knot':
        from scipy.interpolate import CubicSpline
        return np.ascontiguousarray(CubicSpline(knots, nodes, axis=0, extrapolate=False).c.transpose(1, 0, 2))
    if mode == 'linear':
        c = np.zeros((len(knots) - 1, 4, nodes.shape[1]), dtype=np.float64)
        c[:, 2] = np.diff(nodes, axis=0) / np.diff(knots)[:, None]
        c[:, 3] = nodes[:-1]
        return c
    raise ValueError('unsupported compact spline encoding')


class PreparedRadialStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('schema') not in (SCHEMA, 'nacf-prepared-radial-store/v1') or self.manifest.get('complete') is not True:
            raise ValueError('incomplete or unsupported prepared radial store')
        if self.manifest.get('radial_source_sha256') != radial_identity():
            raise ValueError('prepared radial implementation changed; explicitly rebuild')

    def bind(self, bank):
        if self.manifest.get('source_manifests') != source_bindings(bank):
            raise ValueError('prepared radial source manifests disagree with the bank')

    def table(self, kind, left, right, *, device, dtype, backend):
        if dtype not in (torch.float32, torch.float64) or backend not in ('auto', 'torch', 'cuda'):
            raise ValueError('unsupported prepared radial dtype/backend')
        key = '|'.join((kind, left, right))
        entry = self.manifest['tables'].get(key)
        if entry is None:
            raise KeyError(f'prepared radial table missing: {key}; explicitly prepare it')
        name = entry['path']
        if Path(name).name != name:
            raise ValueError('prepared payload must be inside its store')
        payload = (self.root / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('prepared radial payload checksum mismatch')
        obj = TorchRadialBlockTable.__new__(TorchRadialBlockTable)
        torch.nn.Module.__init__(obj)
        obj.backend = backend
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            attrs = json.loads(str(arrays['__attrs__']))
            if set(attrs) != set(ATTRS):
                raise ValueError('prepared radial attributes disagree with schema')
            for name, value in attrs.items():
                setattr(obj, name, tuple(value) if isinstance(value, list) else value)
            buffers = {name: arrays[name] for name in arrays.files if not name.startswith('__')}
            if '__nodes__' in arrays:
                c = _coefficients(arrays['knots'], arrays['__nodes__'], str(arrays['__spline_mode__']))
                if hashlib.sha256(c.tobytes()).hexdigest() != str(arrays['__coefficient_sha256__']):
                    raise ValueError('compact spline arithmetic changed; explicitly prepare on this runtime')
                buffers['coefficients'] = c
            for name, value in buffers.items():
                if name.startswith('__'):
                    continue
                if value.dtype.kind == 'f':
                    value = value.astype(np.float64 if dtype == torch.float64 else np.float32)
                if not np.isfinite(value).all():
                    raise ValueError('nonfinite prepared radial buffer')
                obj.register_buffer(name, torch.from_numpy(value.copy()).to(device=device))
        obj.prepared_cache = 'immutable_compact_store'
        return obj


__all__ = ['PreparedRadialStore', 'write_table', 'source_bindings', 'radial_identity', 'SCHEMA']
