"""Strict, compiler-free validation and loading of the packaged CUDA extensions."""
import hashlib
import importlib
import json
import platform
import sysconfig
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def runtime_identity():
    return dict(python=platform.python_version(), machine=platform.machine(),
                system=platform.system(), torch=str(torch.__version__), cuda=torch.version.cuda,
                cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                ext_suffix=sysconfig.get_config_var('EXT_SUFFIX'))

def verify(name, check_device=False):
    path = ROOT / 'prebuilt' / (name + '.json')
    if not path.exists():
        raise RuntimeError(f'Missing precompiled manifest {path}. Run precompile.py during installation; runtime never compiles.')
    m = json.loads(path.read_text())
    if m['runtime'] != runtime_identity():
        raise RuntimeError('Precompiled H0 environment/ABI mismatch; select a matching prebuilt package or explicitly run precompile.py during installation.')
    for rel, expected in m['sources'].items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f'Precompiled H0 source mismatch: {rel}; explicitly rebuild once during installation.')
    for dep, expected in m['dependencies'].items():
        if not Path(dep).is_file() or sha256(dep) != expected:
            raise RuntimeError(f'Precompiled H0 dependency changed: {dep}; rebuild for the new ABI during installation.')
    binary = ROOT / m['binary']
    if not binary.is_file() or sha256(binary) != m['binary_sha256']:
        raise RuntimeError(f'Precompiled H0 binary missing or modified: {binary}')
    if check_device:
        cap = '.'.join(map(str, torch.cuda.get_device_capability()))
        if cap not in m['architectures']:
            raise RuntimeError(f'GPU architecture {cap} is not in this prebuilt package: {m["architectures"]}; runtime never compiles.')
    return m

def load(name):
    verify(name)
    return importlib.import_module('.' + name, 'h0rebuild')
