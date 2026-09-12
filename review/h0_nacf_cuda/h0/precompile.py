"""Explicit installation-time build. --check and runtime never invoke a compiler."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from h0rebuild.precompiled import ROOT, sha256, runtime_identity, verify

UPSTREAM = Path('/home/mingkang_nt/codex/h0_flash_20260912/upstream_abacus/source')
NAO = Path('/home/mingkang_nt/codex/h0_flash_20260912/deps/pyabacus/ModuleNAO')
ARCHS = ['8.0', '8.6', '8.9', '9.0']

def build(name, force=False, check=False):
    try:
        m = verify(name)
        for path, digest in m.get('upstream_headers', {}).items():
            if not Path(path).exists() or sha256(path) != digest:
                raise RuntimeError('Upstream C++ headers changed; rebuild during installation')
        if not force:
            print(json.dumps(dict(extension=name, status='REUSED', binary_sha256=m['binary_sha256'])), flush=True)
            return
    except (RuntimeError, OSError, ValueError) as exc:
        if check:
            raise SystemExit(str(exc))
        print(str(exc), flush=True)
    if check:
        return
    # Deliberately lazy: importing the runtime never imports cpp_extension.
    from torch.utils.cpp_extension import load
    import torch
    lane = 'two_center' if name == '_cuda_two_center' else 'local_grid'
    csrc = ROOT / 'csrc' / lane
    units = (['bindings.cpp', 'two_center_tables.cpp', 'two_center_cuda.cu'] if lane == 'two_center'
             else ['binding.cpp', 'support_kernel.cu', 'pair_kernel.cu'])
    inputs = sorted(csrc.glob('*.h')) + sorted(csrc.glob('*.cuh')) + [csrc / f for f in units]
    inputs += [ROOT / 'precompile.py']
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in inputs if p.is_file()}
    deps = {}
    includes = [str(csrc)]
    ldflags = []
    if lane == 'two_center':
        includes += [str(UPSTREAM), str(UPSTREAM/'source_base'), str(UPSTREAM/'source_base/module_container'), '/home/mingkang_nt/anaconda3/envs/abacus_dev/include']
        for p in sorted(NAO.glob('*.so*')):
            if p.is_file():
                deps[str(p)] = sha256(p)
        ldflags += [f'-L{NAO}', f'-Wl,-rpath,{NAO}', '-lnaopack', '-lopenblas', '-L/home/mingkang_nt/anaconda3/envs/new_soc/lib']
    build_dir = ROOT / 'work' / ('build_' + lane)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(ARCHS)
    os.environ.setdefault('MAX_JOBS', '2')
    started = time.perf_counter()
    mod = load(name=name, sources=[str(p) for p in inputs if p.suffix in ('.cpp', '.cu')],
               extra_cflags=['-O3', '-std=c++17'], extra_cuda_cflags=['-O3', '-std=c++17'],
               extra_include_paths=includes, extra_ldflags=ldflags,
               build_directory=str(build_dir), verbose=True)
    target = ROOT / 'h0rebuild' / (name + '.so')
    temp = target.with_suffix('.so.pending')
    shutil.copy2(mod.__file__, temp)
    os.replace(temp, target)
    headers = {str(p): sha256(p) for p in UPSTREAM.rglob('*.h')} if lane == 'two_center' else {}
    m = dict(schema=1, extension=name, runtime=runtime_identity(), architectures=ARCHS,
             sources=sources, dependencies=deps, upstream_headers=headers,
             binary=str(target.relative_to(ROOT)), binary_sha256=sha256(target),
             build_seconds=time.perf_counter()-started,
             flags={'cxx':['-O3','-std=c++17'],'nvcc':['-O3','-std=c++17']})
    out = ROOT/'prebuilt'/(name+'.json')
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix('.pending')
    tmp.write_text(json.dumps(m, indent=2)+'\n')
    os.replace(tmp, out)
    print(json.dumps(dict(extension=name, status='BUILT', seconds=m['build_seconds'], binary_sha256=m['binary_sha256'])), flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--force', action='store_true')
    args=parser.parse_args()
    for name in ('_cuda_two_center', '_cuda_local_grid'):
        build(name, force=args.force, check=args.check)
