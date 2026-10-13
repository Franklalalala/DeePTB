"""Exact shared radial storage, explicitly converted from an immutable v2 store.

The generator contract and original per-composition checksums are retained, never
restamped. SQLite holds unique curves keyed by exact grid and coefficient bytes;
small NPZ composition records preserve all other state, including complex D and
not-a-knot AO splines. CUDA consumes the original expanded tensors unchanged.
Runtime is read-only and never tabulates, compiles, or repairs a failed checksum.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import time
from contextlib import closing

import numpy as np
from . import offline
from .radial_codec import pack, unpack

SCHEMA = 'h0-shared-radial/v1'
MARKER = 'shared_radial.json'
COEFFS = ('S_coeffs', 'T_coeffs', 'Q_coeffs')


def is_shared_store(store):
    return (Path(store) / MARKER).is_file()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _grid(dr, cutoff, nr):
    return struct.pack('<ddq', float(dr), float(cutoff), int(nr))


def _connection(store):
    # No journal or temp writes are permitted in an inference process.
    db = sqlite3.connect((Path(store)/'curves.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-8192')
    return db


def read_record(store, key, *, _manifest=None):
    """Reconstruct exactly the original CPU record, in bounded curve chunks."""
    import torch
    store = Path(store)
    manifest = _manifest if _manifest is not None else json.loads((store/MARKER).read_text())
    if manifest['schema'] != SCHEMA or manifest['byteorder'] != sys.byteorder:
        raise ValueError('Shared radial schema/byte order mismatch')
    if key not in manifest['tables']:
        raise FileNotFoundError(f'Missing prepared shared table {key}; explicitly prepare')
    path = store/'two_center'/f'{key}.npz'
    if _digest(path.read_bytes()) != manifest['tables'][key]['index_sha256']:
        raise ValueError('Shared composition index checksum mismatch')
    record = offline.read(path)
    refs = record.pop('shared_coefficients')
    state = record['state']
    grid = _grid(state['dr'], state['cutoff'], state['nr'])
    with closing(_connection(store)) as db:
        for name in COEFFS:
            ref = refs[name]
            shape = tuple(ref['shape'])
            if len(shape) != 3 or shape[2] != 4 or shape[1] < 1 or len(ref['ids']) != shape[0]:
                raise ValueError('Invalid shared coefficient shape')
            coeffs = np.empty(shape, dtype=np.float64)
            ids = ref['ids']
            for start in range(0, len(ids), 128):
                chunk = ids[start:start+128]
                rows = dict((r[0], r[1:]) for r in db.execute(
                    'SELECT id,grid,nodes,tail,indices,bits,checksum FROM curves WHERE id IN (' +
                    ','.join('?' for _ in chunk) + ')', chunk))
                for offset, cid in enumerate(chunk):
                    if cid not in rows:
                        raise ValueError(f'Missing shared curve {cid}')
                    g, nodes, tail, indices, bits, checksum = rows[cid]
                    if g != grid:
                        raise ValueError('Shared curve grid mismatch')
                    c = unpack(dict(schema=2, dr=state['dr'], checksum=checksum,
                        nodes=np.frombuffer(nodes, dtype=np.float64).reshape(1, shape[1], 2),
                        tail=np.frombuffer(tail, dtype=np.float64).reshape(1, 2),
                        correction_indices=np.frombuffer(indices, dtype=np.int64),
                        correction_bits=np.frombuffer(bits, dtype=np.uint64)))
                    if _digest(grid + bytes.fromhex(checksum)) != cid:
                        raise ValueError('Shared curve identity mismatch')
                    coeffs[start+offset] = c[0]
            state[name] = torch.from_numpy(coeffs)
    if record['key'] != key or offline.fingerprint(state) != record['checksum']:
        raise ValueError('Reconstructed two-center checksum mismatch')
    return record


def convert_store(source, destination):
    """Explicit offline conversion. Publish marker only after every table reloads.

    An incomplete destination is retained for diagnosis and never accepted by the
    runtime. The source is read-only. No source or binary provenance is changed.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if sys.byteorder != 'little':
        raise ValueError('Initial shared format requires little-endian FP64')
    if destination.exists():
        raise FileExistsError('Use a new immutable destination')
    from .table_contract import verify_contract
    verify_contract(source)
    destination.mkdir(parents=True)
    (destination/'two_center').mkdir()
    shutil.copytree(source/'species', destination/'species')
    shutil.copyfile(source/'source_contract.json', destination/'source_contract.json')
    started = time.perf_counter()
    manifest = dict(schema=SCHEMA, byteorder=sys.byteorder, codec=2,
        source_store=str(source), source_contract_sha256=_digest((source/'source_contract.json').read_bytes()),
        representation_sources={p.name:_digest(p.read_bytes()) for p in
            (Path(__file__), Path(__file__).with_name('radial_codec.py'))}, tables={})
    db = sqlite3.connect(destination/'curves.sqlite')
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE curves(id TEXT PRIMARY KEY, grid BLOB, nodes BLOB, tail BLOB, indices BLOB, bits BLOB, checksum TEXT)')
    known = set()
    coefficient_bytes = 0
    for path in sorted((source/'two_center').glob('*.npz')):
        record = offline.read(path)
        state = record['state']
        if record['key'] != path.stem or offline.fingerprint(state) != record['checksum']:
            raise ValueError(f'Source checksum mismatch: {path}')
        if record['sources'] != json.loads((source/'source_contract.json').read_text()):
            raise ValueError('Source generator identity mismatch')
        grid = _grid(state['dr'], state['cutoff'], state['nr'])
        refs = {}
        for name in COEFFS:
            coefficients = state[name].numpy()
            # Original fingerprint includes dictionary order. Keep the slot.
            state[name] = None
            coefficient_bytes += coefficients.nbytes
            ids = []
            for curve in coefficients:
                checksum = _digest(curve.tobytes())
                cid = _digest(grid + bytes.fromhex(checksum))
                ids.append(cid)
                if cid in known:
                    continue
                packed = pack(curve[None], state['dr'])
                unpack(packed)  # Reject incompatible arithmetic at the producer too.
                db.execute('INSERT INTO curves VALUES (?,?,?,?,?,?,?)',
                    (cid, grid, packed['nodes'].tobytes(), packed['tail'].tobytes(),
                     packed['correction_indices'].tobytes(), packed['correction_bits'].tobytes(), checksum))
                known.add(cid)
            refs[name] = dict(shape=coefficients.shape, ids=ids)
        record['shared_coefficients'] = refs
        target = destination/'two_center'/path.name
        offline.save(target, record)
        manifest['tables'][path.stem] = dict(index_sha256=_digest(target.read_bytes()),
            source_sha256=_digest(path.read_bytes()), state_checksum=record['checksum'])
        db.commit()
        print(json.dumps({'converted':len(manifest['tables']), 'unique_curves':len(known)}), flush=True)
    db.close()
    if (source/'catalog.json').exists():
        catalog = json.loads((source/'catalog.json').read_text())
        catalog['store'] = str(destination)
        (destination/'catalog.json').write_text(json.dumps(catalog, indent=2))
    manifest.update(unique_curves=len(known), original_coefficient_bytes=coefficient_bytes,
                    conversion_seconds=time.perf_counter()-started)
    # Runtime cannot discover this store until every composition has reloaded.
    marker = destination/MARKER
    for key in manifest['tables']:
        read_record(destination, key, _manifest=manifest)
    verify_contract(source)
    if _digest((source/'source_contract.json').read_bytes()) != manifest['source_contract_sha256']:
        raise ValueError('Source contract changed during conversion')
    manifest.update(reloaded_tables=len(manifest['tables']),
                    total_seconds=time.perf_counter()-started,
                    source_store_bytes=sum(p.stat().st_size for p in source.rglob('*') if p.is_file()),
                    shared_store_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()))
    pending = marker.with_suffix('.pending')
    pending.write_text(json.dumps(manifest, indent=2))
    pending.replace(marker)
    return manifest


def prepared_shared_two_center(species_data, *, store, dr_bohr=.01, nspin=1, device='cuda'):
    """Compiler-free native runtime using verified expansion of the shared pool."""
    import torch
    from .cuda_two_center import CUDATwoCenter
    from .precompiled import verify
    from .table_contract import current
    with offline._resident_lock:
        sources = current()
        # One fresh environment capture per load, shared by verification and
        # key construction. Never cache or restamp the generator identity.
        if json.loads((Path(store)/'source_contract.json').read_text()) != sources:
            raise RuntimeError('Prepared H0 numerical source contract missing/changed; explicitly prepare a new store')
        sd = copy.deepcopy(dict(sorted(species_data.items())))
        key = offline.fingerprint({'species':sd, 'dr':dr_bohr, 'nspin':nspin, 'schema':offline.SCHEMA,
            'binary':verify('_cuda_two_center')['binary_sha256'], 'sources':sources})
        dev = torch.device(device)
        if dev.type != 'cuda':
            raise ValueError('Two-center runtime requires CUDA')
        if dev.index is None:
            dev = torch.device('cuda', torch.cuda.current_device())
        with torch.cuda.device(dev):
            verify('_cuda_two_center', check_device=True)
        resident_key = (str(Path(store).resolve()), key, str(dev))
        if resident_key in offline._resident:
            state, ready = offline._resident[resident_key]
            offline._resident.move_to_end(resident_key)
            torch.cuda.current_stream(dev).wait_event(ready)
            hit = 'shared_memory'
        else:
            record = read_record(store, key)
            if record['sources'] != sources:
                raise ValueError('Shared table generator identity mismatch')
            def upload(v):
                if isinstance(v, torch.Tensor): return v.to(dev)
                if isinstance(v, dict): return {k:upload(x) for k,x in v.items()}
                if isinstance(v, list): return [upload(x) for x in v]
                if isinstance(v, tuple): return tuple(upload(x) for x in v)
                return v
            state = upload(record['state'])
            ready = torch.cuda.Event(); ready.record(torch.cuda.current_stream(dev))
            offline._resident[resident_key] = (state, ready)
            while len(offline._resident) > 2:
                offline._resident.popitem(last=False)
            hit = 'shared_disk'
        obj = CUDATwoCenter.__new__(CUDATwoCenter)
        with torch.cuda.device(dev):
            obj.__dict__.update(offline.private_copy(state))
        obj.device, obj.sd = dev, sd
        obj.metadata.update(offline_cache=hit, offline_key=key, radial_representation=SCHEMA)
        return obj
