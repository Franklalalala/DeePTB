"""Content-addressed, pickle-free prepared data. Only explicit preparation writes.

An offline species ID is an immutable snapshot, not a promise to watch its source
files. Re-run preparation to select changed UPF/ORB contents. Runtime needs no
UPF/ORB files. Geometry and reciprocal G grids are deliberately not cached here.
"""
import dataclasses
import copy
import threading
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections import OrderedDict
import numpy as np
from . import models

SCHEMA = 'h0-offline/v2'

def encode(value, arrays):
    import torch
    if isinstance(value, torch.Tensor):
        name = str(len(arrays)); arrays[name] = value.detach().cpu().numpy()
        return {'tensor': name}
    if isinstance(value, np.ndarray):
        name = str(len(arrays)); arrays[name] = value
        return {'array': name}
    if dataclasses.is_dataclass(value):
        return {'class': type(value).__name__, 'fields': {f.name: encode(getattr(value, f.name), arrays) for f in dataclasses.fields(value)}}
    if isinstance(value, Path): return {'path': str(value)}
    if isinstance(value, dict): return {'dict': [[encode(k, arrays), encode(v, arrays)] for k,v in value.items()]}
    if isinstance(value, tuple): return {'tuple': [encode(v, arrays) for v in value]}
    if isinstance(value, list): return [encode(v, arrays) for v in value]
    if isinstance(value, np.generic): return value.item()
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(type(value))

def decode(value, arrays, device='cpu'):
    if isinstance(value, list): return [decode(v, arrays, device) for v in value]
    if not isinstance(value, dict): return value
    if 'tensor' in value:
        import torch
        return torch.from_numpy(arrays[value['tensor']].copy()).to(device)
    if 'array' in value: return arrays[value['array']].copy()
    if 'path' in value: return Path(value['path'])
    if 'tuple' in value: return tuple(decode(v, arrays, device) for v in value['tuple'])
    if 'dict' in value: return {decode(k, arrays, device): decode(v, arrays, device) for k,v in value['dict']}
    if 'class' in value:
        allowed = {c.__name__: c for c in (models.SpeciesData, models.OrbitalBasis, models.OrbitalChannel, models.UPFData, models.Projector)}
        return allowed[value['class']](**{k: decode(v, arrays, device) for k,v in value['fields'].items()})
    raise ValueError('Unknown offline record')

def fingerprint(value):
    arrays = {}; meta = encode(value, arrays)
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
    for name,a in arrays.items():
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}; meta = encode(value, arrays)
    arrays['__meta__'] = np.asarray(json.dumps({'schema': SCHEMA, 'data': meta}))
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        temp = Path(f.name)
        np.savez(f, **arrays)
    os.replace(temp, path)

def read(path, device='cpu'):
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays['__meta__']))
        if meta['schema'] != SCHEMA: raise ValueError('Offline schema mismatch')
        return decode(meta['data'], arrays, device)

def prepare_species(orb_path, upf_path, store):
    from .table_contract import ensure_store
    ensure_store(store)
    from .orb import read_abacus_orb
    from .upf import read_upf
    from .provenance import sha256_file
    from scipy.interpolate import CubicSpline
    data = models.SpeciesData(read_abacus_orb(orb_path), read_upf(upf_path))
    for obj in (data.orb, data.upf):
        obj.metadata['offline_source_sha256'] = sha256_file(obj.source)
    # Persist the exact not-a-knot AO polynomials, with no interpolation refit.
    data.orb.metadata['offline_spline_coefficients'] = [CubicSpline(data.orb.r, c.radial, bc_type='not-a-knot', extrapolate=False).c for c in data.orb.channels]
    identity = fingerprint(data)
    save(Path(store)/'species'/f'{identity}.npz', data)
    return identity

def load_species(store, identity):
    from .table_contract import verify_contract
    verify_contract(store)
    if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity): raise ValueError('Invalid species ID')
    data = read(Path(store)/'species'/f'{identity}.npz')
    if fingerprint(data) != identity: raise ValueError('Offline species checksum mismatch')
    return data

_resident = OrderedDict()
_resident_lock = threading.RLock()

def prepared_two_center(species_data, *, store, dr_bohr=.01, nspin=1, device='cuda', prepare=False):
    """Private resident masters; every consumer owns independent tensor buffers.

    Copying is device-to-device. No warm buffer hash or GPU-to-CPU round trip.
    Native pyabacus handles are not part of either runtime copies or disk state.
    """
    with _resident_lock:
        return _prepared_two_center(species_data,store=store,dr_bohr=dr_bohr,nspin=nspin,device=device,prepare=prepare)

def _prepared_two_center(species_data, *, store, dr_bohr, nspin, device, prepare):
    import torch
    from .cuda_two_center import CUDATwoCenter
    from .precompiled import verify
    from .table_contract import current, verify_contract, ensure_store
    if prepare: ensure_store(store)
    else: verify_contract(store)
    sd = copy.deepcopy(dict(sorted(species_data.items())))
    sources=current()
    key=fingerprint({'species':sd,'dr':dr_bohr,'nspin':nspin,'schema':SCHEMA,
        'binary':verify('_cuda_two_center')['binary_sha256'],'sources':sources})
    dev=torch.device(device)
    if dev.type!='cuda': raise ValueError('Two-center runtime requires CUDA')
    if dev.index is None: dev=torch.device('cuda',torch.cuda.current_device())
    with torch.cuda.device(dev): verify('_cuda_two_center',check_device=True)
    resident_key=(str(Path(store).resolve()),key,str(dev))
    if resident_key in _resident:
        state,ready=_resident[resident_key];_resident.move_to_end(resident_key);hit='memory'
        torch.cuda.current_stream(dev).wait_event(ready)
    else:
        path=Path(store)/'two_center'/f'{key}.npz'
        if path.exists():
            # Verify CPU bytes before uploading; previously verification copied
            # an already-uploaded table back to the CPU.
            record=read(path,'cpu')
            if (record['key']!=key or record.get('sources')!=sources or
                fingerprint(record['state'])!=record['checksum']):
                raise ValueError('Two-center offline checksum/identity mismatch')
            def upload(v):
                if isinstance(v,torch.Tensor): return v.to(dev)
                if isinstance(v,dict): return {k:upload(x) for k,x in v.items()}
                if isinstance(v,list): return [upload(x) for x in v]
                if isinstance(v,tuple): return tuple(upload(x) for x in v)
                return v
            state=upload(record['state']);hit='disk'
        elif prepare:
            obj=CUDATwoCenter(sd,dr_bohr=dr_bohr,nspin=nspin,device=str(dev),cache_dir=str(Path(store)))
            state={k:v for k,v in obj.__dict__.items() if k not in ('collections','integrators','sbt','device','sd')}
            if current()!=sources: raise RuntimeError('Generator changed during table preparation')
            save(path,{'key':key,'sources':sources,'checksum':fingerprint(state),'state':state});hit='prepared'
        else:
            raise FileNotFoundError(f'Missing prepared two-center table {key}; explicitly prepare; runtime never tabulates')
        ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(dev))
        _resident[resident_key]=(state,ready)
        while len(_resident)>2: _resident.popitem(last=False)
    obj=CUDATwoCenter.__new__(CUDATwoCenter)
    obj.__dict__.update(copy.deepcopy(state));obj.device=dev;obj.sd=sd
    obj.metadata.update(offline_cache=hit,offline_key=key)
    return obj
