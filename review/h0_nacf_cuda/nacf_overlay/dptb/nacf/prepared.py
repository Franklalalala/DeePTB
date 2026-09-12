"""Atomic self-checksummed radial artifacts; every hit validates input options."""
import hashlib, io, os, tempfile
from pathlib import Path
import numpy as np
import torch

ATTRS=('left_shells','right_shells','support_bohr','shape','angular_degrees')

def key_for(table):
    h=hashlib.sha256(b'nacf-prepared/v2')
    for name in ('radial.py','prepared.py'):
        h.update(Path(__file__).with_name(name).read_bytes())
    arrays=[table.distances,table.values,table._spline.c if table._spline is not None else np.empty(0),table._rotator.directions]
    for l in sorted(set(table.left_shells+table.right_shells)):
        arrays.append(table._rotator._base[l])
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(repr((table.left_shells,table.right_shells,table.support_bohr)).encode())
    return h.hexdigest()

def cached_table(source,cache_dir,*,device,dtype,backend):
    from .radial import TorchRadialBlockTable,validate_table_options
    validate_table_options(source,dtype,backend)
    key=key_for(source);root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.pt')
    if path.exists():
        # Read one atomic generation: no separately published checksum sidecar.
        artifact=path.read_bytes();digest,sep,payload=artifact.partition(b'\n')
        if not sep or hashlib.sha256(payload).hexdigest().encode()!=digest:
            raise ValueError('NACF prepared table checksum mismatch')
        record=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
        if record['key']!=key:raise ValueError('NACF prepared table identity mismatch')
        obj=TorchRadialBlockTable.__new__(TorchRadialBlockTable);torch.nn.Module.__init__(obj)
        obj.backend=backend
        for name,value in record['attrs'].items():setattr(obj,name,value)
        for name,value in record['buffers'].items():obj.register_buffer(name,value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype))
        obj.prepared_cache='disk';return obj
    obj=TorchRadialBlockTable(source,device='cpu',dtype=torch.float64,backend=backend)
    if key_for(source)!=key:raise ValueError('NACF source mutated during preparation')
    record={'key':key,'attrs':{k:getattr(obj,k) for k in ATTRS},'buffers':dict(obj.named_buffers())}
    buf=io.BytesIO();torch.save(record,buf);payload=buf.getvalue()
    pending=None
    try:
        with tempfile.NamedTemporaryFile(dir=root,suffix='.pending',delete=False) as f:
            pending=Path(f.name);f.write(hashlib.sha256(payload).hexdigest().encode()+b'\n'+payload);f.flush();os.fsync(f.fileno())
        os.replace(pending,path)
    finally:
        if pending is not None and pending.exists():pending.unlink()
    obj=obj.to(device=device,dtype=dtype);obj.prepared_cache='prepared';return obj
