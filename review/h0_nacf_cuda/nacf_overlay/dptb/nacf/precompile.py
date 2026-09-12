"""Explicit installation build; normal imports do not import a compiler helper."""
import argparse,json,os,shutil,tempfile
from pathlib import Path
from .precompiled import ROOT,identity,sha,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--force',action='store_true');a=p.parse_args()
    try:
        m=verify()
        if not a.force: print('REUSED',m['sha256']);return
    except (OSError,RuntimeError,ValueError):
        if a.check:raise
    if a.check:return
    from torch.utils.cpp_extension import load
    archs=['8.0','8.6','8.9','9.0'];os.environ['TORCH_CUDA_ARCH_LIST']=';'.join(archs)
    build=Path(os.environ['TORCH_EXTENSIONS_DIR'])/'nacf_prebuilt';build.mkdir(parents=True,exist_ok=True)
    sources=['csrc/bindings.cpp','csrc/radial.cu']
    mod=load(name='_nacf_radial',sources=[str(ROOT/n) for n in sources],extra_cflags=['-O3'],
             extra_cuda_cflags=['-O3','--fmad=false'],build_directory=str(build),verbose=True)
    binary=ROOT/'_nacf_radial.so';pending=binary.with_suffix('.pending');shutil.copy2(mod.__file__,pending);os.replace(pending,binary)
    m={'schema':1,'runtime':identity(),'architectures':archs,'sources':{n:sha(ROOT/n) for n in sources+['precompile.py']},
       'binary':binary.name,'sha256':sha(binary),'flags':['-O3','--fmad=false']}
    pending=ROOT/'prebuilt.pending';pending.write_text(json.dumps(m,indent=2)+'\n');os.replace(pending,ROOT/'prebuilt.json');print('BUILT',m['sha256'])
if __name__=='__main__':main()
