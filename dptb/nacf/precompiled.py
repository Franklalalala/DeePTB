"""NACF native runtime: load a verified binary, never compile at inference."""
import hashlib, importlib, json, platform, sysconfig
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity():
    return {'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),'system':platform.system(),'machine':platform.machine(),
            'suffix':sysconfig.get_config_var('EXT_SUFFIX')}
def verify(device=None):
    path=ROOT/'prebuilt.json'
    if not path.exists():raise RuntimeError('Missing NACF prebuilt binary; run python -m dptb.nacf.precompile once during installation. Runtime never compiles.')
    m=json.loads(path.read_text())
    if m['runtime']!=identity():raise RuntimeError('NACF prebuilt Python/Torch/CUDA ABI mismatch')
    for name,digest in m['sources'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('NACF native source mismatch: '+name)
    if sha(ROOT/m['binary'])!=m['sha256']:raise RuntimeError('NACF prebuilt binary checksum mismatch')
    if device is not None:
        cap='.'.join(map(str,torch.cuda.get_device_capability(device)))
        if cap not in m['architectures']:raise RuntimeError('NACF binary does not contain GPU architecture '+cap)
    return m
def load():
    verify()
    return importlib.import_module('._nacf_radial','dptb.nacf')
