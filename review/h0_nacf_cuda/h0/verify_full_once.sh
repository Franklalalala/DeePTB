#!/bin/bash
set -euo pipefail
cd /home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate
source ./env.sh
exec 201>/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/gpu1.lock
flock -x 201
test -f work/VERIFIED
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python verify_full.py
touch work/FULL_VERIFIED
