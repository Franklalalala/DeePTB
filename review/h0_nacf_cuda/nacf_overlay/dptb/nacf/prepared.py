"""Persist GPU-ready radial buffers, including rotation/packing metadata.

The source table is validated by its normal store before cache selection. This
does not bypass P23 coverage or corrupt-source errors. Geometry stays online.
"""
import hashlib, os, tempfile
from pathlib import Path
import numpy as np
import torch

ATTRS=('left_shells','right_shells','support_bohr','shape','angular_degrees')
def key_for(table):
    h=hashlib.sha256(Path(__file__).with_name('radial.py').read_bytes())
    h.update(b'nacf-prepared/v1')
    for a in (table.distances,table.values,table._spline.c if table._spline is not None else np.empty(0)):
        a=np.ascontiguousarray(a);h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(repr((table.left_shells,table.right_shells,table.support_bohr)).encode())
    return h.hexdigest()

def cached_table(source,cache_dir,*,device,dtype,backend):
    from .radial import TorchRadialBlockTable
    key=key_for(source);root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.pt');checksum=path.with_suffix('.sha256')
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum.read_text().strip():raise ValueError('NACF prepared table checksum mismatch')
        record=torch.load(path,map_location='cpu',weights_only=True)
        if record['key']!=key:raise ValueError('NACF prepared table identity mismatch')
        obj=TorchRadialBlockTable.__new__(TorchRadialBlockTable);torch.nn.Module.__init__(obj)
        obj.backend=backend
        for name,value in record['attrs'].items():setattr(obj,name,value)
        for name,value in record['buffers'].items():obj.register_buffer(name,value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype))
        obj.prepared_cache='disk';return obj
    # Persist FP64 master buffers; each caller selects FP64/FP32 on load.
    obj=TorchRadialBlockTable(source,device='cpu',dtype=torch.float64,backend=backend)
    record={'key':key,'attrs':{k:getattr(obj,k) for k in ATTRS},'buffers':dict(obj.named_buffers())}
    with tempfile.NamedTemporaryFile(dir=root,suffix='.pt',delete=False) as f:
        pending=Path(f.name);torch.save(record,f)
    digest=hashlib.sha256(pending.read_bytes()).hexdigest()
    # Per-bank preparation is sequential. Publishing checksum before data means
    # readers never accept an unverified artifact; a race fails closed.
    checksum.write_text(digest+'\n');os.replace(pending,path)
    obj=obj.to(device=device,dtype=dtype);obj.prepared_cache='prepared';return obj
