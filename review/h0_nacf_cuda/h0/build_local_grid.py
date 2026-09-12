"""Compatibility entrypoint for the unified explicit precompiler."""
import argparse
from precompile import build
from h0rebuild.precompiled import ROOT, sha256 as compute_file_sha256

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    build('_cuda_local_grid', force=args.force, check=args.check)
