"""Explicit standalone C++20 build; no Torch ABI or Tonari Python dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'dptb/nacf/csrc'
    vendor = root / 'vendor/tonari'
    sources = [root/'topology.cpp'] + [vendor/name for name in ('geometry.cpp','neighbors_cpu.cpp','thread_pool.cpp')]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.cxx) + ['-std=c++20','-O3','-shared','-fPIC','-pthread',*[str(p) for p in sources],'-o',str(output)]
    subprocess.run(command, check=True)
    receipt = {'command':command,'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root/'topology.cpp',*sorted(vendor.glob('*'))] if p.is_file()}}
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
