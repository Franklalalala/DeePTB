"""Explicit installation build; normal imports do not import a compiler helper.

``--check`` never builds: it verifies the prebuilt manifest, binary and sources and, when ``--arch`` is
given, that the recorded native/PTX targets are exactly the requested ones. Any mismatch raises, so
the process exits nonzero; deployment automation can rely on ``--check --arch`` as a strict gate.
"""
import argparse,json,os,shutil,tempfile
from pathlib import Path
from .precompiled import ROOT,identity,sha,verify

SOURCES=['csrc/bindings.cpp','csrc/radial.cu','csrc/packing.cu','csrc/contraction.cu','csrc/density.cu','csrc/onsite_density.cu']


def split_targets(archs):
    """Requested CUDA targets -> (native list, PTX list); ``8.9+PTX`` records both."""
    native=[x.removesuffix('+PTX') for x in archs];ptx=[x.removesuffix('+PTX') for x in archs if x.endswith('+PTX')]
    return native,ptx


def targets_match(manifest,archs):
    native,ptx=split_targets(archs)
    return list(manifest.get('architectures',[]))==native and list(manifest.get('ptx_architectures',[]))==ptx


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--force',action='store_true')
    p.add_argument('--arch',action='append',help='CUDA target, repeatable; e.g. 8.9+PTX enables forward JIT on newer GPUs')
    a=p.parse_args(argv);archs=a.arch or ['8.0','8.6','8.9','9.0']
    if a.check:
        m=verify()   # raises on a missing/inconsistent prebuilt: nonzero exit
        if a.arch is not None and not targets_match(m,archs):
            raise RuntimeError(f"NACF prebuilt targets native={m.get('architectures')} ptx={m.get('ptx_architectures',[])} "
                               f"do not match the requested {archs}; rebuild with python -m dptb.nacf.precompile --arch ...")
        print('OK',m['sha256']);return m
    try:
        m=verify()
        if not a.force and (a.arch is None or targets_match(m,archs)): print('REUSED',m['sha256']);return m
    except (OSError,RuntimeError,ValueError):
        pass
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST']=';'.join(archs)
    build=Path(os.environ['TORCH_EXTENSIONS_DIR'])/'nacf_prebuilt';build.mkdir(parents=True,exist_ok=True)
    mod=load(name='_nacf_radial',sources=[str(ROOT/n) for n in SOURCES],extra_cflags=['-O3'],
             extra_cuda_cflags=['-O3','--fmad=false'],build_directory=str(build),verbose=True)
    binary=ROOT/'_nacf_radial.so';pending=binary.with_suffix('.pending');shutil.copy2(mod.__file__,pending);os.replace(pending,binary)
    native,ptx=split_targets(archs)
    m={'schema':1,'runtime':identity(),'architectures':native,'ptx_architectures':ptx,'sources':{n:sha(ROOT/n) for n in SOURCES+['precompile.py']},
       'binary':binary.name,'sha256':sha(binary),'flags':['-O3','--fmad=false']}
    pending=ROOT/'prebuilt.pending';pending.write_text(json.dumps(m,indent=2)+'\n');os.replace(pending,ROOT/'prebuilt.json');print('BUILT',m['sha256']);return m
if __name__=='__main__':main()
