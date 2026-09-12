#!/bin/bash
set -euo pipefail
cd /home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate
source ./env.sh
exec 201>/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/gpu1.lock
flock -x 201
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
/home/mingkang_nt/anaconda3/envs/new_soc/bin/python verify_prebuilt.py
# A second Python process checks startup reuse, with build tools absent from PATH.
PATH=/bin /home/mingkang_nt/anaconda3/envs/new_soc/bin/python precompile.py --check
touch work/VERIFIED
