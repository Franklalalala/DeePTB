"""Prepared H0 records (schema h0-offline/v2): pickle-free npz files of nested data, and their content fingerprints.

A record is a JSON tree stored as ``__meta__`` next to the arrays it names: tensors and arrays become {'tensor' | 'array':
key}, dataclasses of ``dptb.prior.common.pporb`` {'class', 'fields'}, paths, tuples and dicts tagged likewise. The
fingerprint is the SHA256 of the canonical JSON tree followed by every array's key, dtype, shape and bytes, so it names
the content independently of the file. Pair records of the pair bank and the prepared species (ORB + aligned UPF) of
``species/`` use this format; ``bank_species`` loads a species the way the bank's manifest pins it.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common import pporb

SCHEMA = 'h0-offline/v2'
_CLASSES = {c.__name__: c for c in (pporb.SpeciesData, pporb.OrbitalBasis, pporb.OrbitalChannel, pporb.UPFData,
                                    pporb.Projector)}


def encode(value, arrays):
    if isinstance(value, torch.Tensor):
        name = str(len(arrays))
        arrays[name] = value.detach().cpu().numpy()
        return {'tensor': name}
    if isinstance(value, np.ndarray):
        name = str(len(arrays))
        arrays[name] = value
        return {'array': name}
    if dataclasses.is_dataclass(value):
        return {'class': type(value).__name__,
                'fields': {f.name: encode(getattr(value, f.name), arrays) for f in dataclasses.fields(value)}}
    if isinstance(value, Path):
        return {'path': str(value)}
    if isinstance(value, dict):
        return {'dict': [[encode(k, arrays), encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'tuple': [encode(v, arrays) for v in value]}
    if isinstance(value, list):
        return [encode(v, arrays) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def decode(value, arrays):
    if isinstance(value, list):
        return [decode(v, arrays) for v in value]
    if not isinstance(value, dict):
        return value
    if 'tensor' in value:
        return torch.from_numpy(arrays[value['tensor']].copy())
    if 'array' in value:
        return arrays[value['array']].copy()
    if 'path' in value:
        return Path(value['path'])
    if 'tuple' in value:
        return tuple(decode(v, arrays) for v in value['tuple'])
    if 'dict' in value:
        return {decode(k, arrays): decode(v, arrays) for k, v in value['dict']}
    if 'class' in value:
        return _CLASSES[value['class']](**{k: decode(v, arrays) for k, v in value['fields'].items()})
    raise ValueError('unknown record node')


def fingerprint(value):
    arrays = {}
    meta = encode(value, arrays)
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
    for name, a in arrays.items():
        h.update(name.encode())
        h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays['__meta__']))
        if meta['schema'] != SCHEMA:
            raise ValueError(f'{path}: record schema {meta["schema"]!r}, expected {SCHEMA!r}')
        return decode(meta['data'], arrays)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bank_species(root, manifest, catalog, symbol):
    """Prepared species data of a pair bank: file SHA256 and content fingerprint against the manifest row."""
    identity = catalog['species'][symbol]
    row = manifest['species'][symbol]
    if row['key'] != identity:
        raise ValueError(f'{symbol}: catalog and manifest name different species records')
    path = Path(root) / 'species' / f'{identity}.npz'
    if sha256(path) != row['sha256']:
        raise ValueError(f'{symbol}: species record checksum mismatch')
    data = read(path)
    if fingerprint(data) != identity:
        raise ValueError(f'{symbol}: species record content mismatch')
    for obj, key in ((data.orb, 'orb'), (data.upf, 'upf')):
        if obj.metadata.get('offline_source_sha256') != row[key + '_sha256']:
            raise ValueError(f'{symbol}: {key} source differs from the pair bank')
    return data
