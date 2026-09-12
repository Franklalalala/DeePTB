#!/bin/bash
set -euo pipefail
cd /home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate
source ./env.sh
exec 200>/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/precompile.lock
flock -x 200
# Build only; explicit fatbin architecture list avoids probing or occupying GPU0/GPU1.
export CUDA_VISIBLE_DEVICES=""
/home/mingkang_nt/anaconda3/envs/new_soc/bin/python precompile.py
/home/mingkang_nt/anaconda3/envs/new_soc/bin/python precompile.py --check
touch work/PRECOMPILED
