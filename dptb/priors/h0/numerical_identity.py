"""Strict numerical dependency identities; paths are retained as provenance only.

This file is mirrored in the standalone NACF overlay. Hashes are reused only
while inode, size, mtime and ctime still match. Identity checks still stat the
installed files, so replacing a dependency invalidates an existing table/run.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading

_hashes={}
_lock=threading.RLock()

def file_hash(path, *, fresh=False):
    path=Path(path).resolve();s=path.stat()
    token=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with _lock:
        old=_hashes.get(str(path))
        if not fresh and old and old[0]==token:return old[1]
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        end=path.stat()
        if token!=(end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns):
            raise RuntimeError('Numerical dependency changed while hashing')
        _hashes[str(path)]=(token,h.hexdigest());return h.hexdigest()

def loaded_libraries(pattern):
    """Content identity without basename overwrite or mapping-order dependence.

    Different implementations under the same basename are rejected: symbol
    resolution/load order is not qualified by a sorted bag of file hashes.
    Multiple segments of an inode are read once; paths remain provenance only.
    """
    maps=Path('/proc/self/maps');objects={};by_name={};paths={}
    if not maps.exists():return [], {}
    for line in maps.read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)<6 or not re.search(pattern,parts[5]):continue
        raw=parts[5]
        if raw.endswith(' (deleted)'):raise RuntimeError('A loaded numerical library was replaced')
        p=Path(raw);stat=p.stat();major,minor=(int(x,16) for x in parts[3].split(':'))
        if (major,minor,int(parts[4]))!=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
            raise RuntimeError('Loaded numerical library identity changed')
        key=(stat.st_dev,stat.st_ino)
        # Mapped DSOs are hashed once per capture. The target filesystem can
        # give rapid same-size writes identical mtime/ctime (observed at 4 ms
        # resolution), so a cross-capture stat-token cache is insufficient.
        if key not in objects:objects[key]=file_hash(p,fresh=True)
        digest=objects[key]
        if p.name in by_name and by_name[p.name]!=digest:
            raise RuntimeError('Conflicting loaded numerical libraries with basename '+p.name)
        by_name[p.name]=digest;paths.setdefault(p.name,set()).add(str(p.resolve()))
    identity=[{'name':name,'sha256':digest} for name,digest in sorted(by_name.items())]
    return identity, {'library:'+name:sorted(items) for name,items in sorted(paths.items())}


def capture(names):
    importlib.import_module('numpy.linalg');importlib.import_module('scipy.linalg')
    if 'pyabacus' in names:
        importlib.import_module('pyabacus.ModuleNAO');importlib.import_module('pyabacus.ModuleBase')
    packages={};provenance={}
    for name in names:
        module=importlib.import_module(name);origin=Path(module.__file__).resolve()
        roots=list(map(Path,getattr(module,'__path__',[])))
        files={}
        if roots:
            for root in roots:
                for path in sorted(root.rglob('*')):
                    if path.is_file() and (path.suffix=='.py' or '.so' in path.name or path.suffix in ('.dll','.pyd')):
                        files[path.relative_to(root).as_posix()]=file_hash(path)
                extra=root.with_name(root.name+'.libs')
                if extra.is_dir():
                    for path in sorted(extra.rglob('*')):
                        if path.is_file():files['../'+extra.name+'/'+path.relative_to(extra).as_posix()]=file_hash(path)
        else:files[origin.name]=file_hash(origin)
        packages[name]={'version':str(getattr(module,'__version__','unversioned')),
                        'files_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
                        'file_count':len(files)}
        provenance[name]=str(origin)
    # These libraries are selected through dlopen / loader search paths and may
    # live outside the Python package. CUDA runtime libraries reside in the
    # qualified Torch/NVIDIA installation; the driver is recorded separately.
    allowed=['openblas','mkl','gfortran','quadmath','stdc\\+\\+','m(?=\\.so(?:\\.|$))']
    if 'pyabacus' in names:allowed.append('naopack')
    if 'pylibxc' in names:allowed.append('xc')
    if 'cufinufft' in names:allowed.extend(['finufft','cufinufft'])
    libraries,locations=loaded_libraries(r'/lib(?:'+'|'.join(allowed)+r')[^/]*(?:\.so|\.so\.[^/]*)')
    provenance.update(locations)
    identity={'schema':'numerical-environment/v2','python':platform.python_version(),
              'machine':platform.machine(),'packages':packages,'loaded_libraries':libraries}
    # /proc mappings reflect import/address order, not numerical identity.
    # Offline fingerprint encodes mapping order, so canonicalize recursively.
    identity=json.loads(json.dumps(identity,sort_keys=True))
    return {'identity':identity,'provenance':provenance}

def execution():
    import torch
    # Resolve the actual LibXC / NUFFT backend before binding their identities.
    importlib.import_module('pylibxc').LibXCFunctional('GGA_C_PBE','polarized')
    importlib.import_module('cufinufft')
    # Resolve lazy CUDA math libraries before taking the execution snapshot.
    if torch.cuda.is_available():
        x=torch.eye(2,dtype=torch.float64,device='cuda')
        torch.fft.fftn(x);torch.mm(x,x);torch.linalg.inv(x);torch.cuda.synchronize()
    record=capture(('numpy','scipy','torch','pylibxc','cufinufft','pyabacus'))
    driver=Path('/proc/driver/nvidia/version')
    record['identity']['cuda']={'torch_cuda':torch.version.cuda,
        'driver':driver.read_text() if driver.exists() else None,
        'capabilities':sorted({tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())})}
    libraries,locations=loaded_libraries(r'/lib(?:cuda|cufft|cublas|cusolver|cusparse|curand|nvJitLink)[^/]*\.so')
    record['provenance'].update(locations)
    record['identity']['cuda']['loaded_libraries']=libraries
    # Vendor CUDA wheels can sit outside torch/lib and are also content-bound.
    vendor=Path(torch.__file__).resolve().parent.parent/'nvidia'
    if vendor.is_dir():
        hashes={str(p.relative_to(vendor)):file_hash(p) for p in sorted(vendor.rglob('*.so*')) if p.is_file()}
        record['identity']['cuda']['vendor_sha256']=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return record
