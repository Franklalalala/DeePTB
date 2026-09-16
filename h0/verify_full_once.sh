#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
exec 201>"${H0_GPU_LOCK:-$H0_WORK_ROOT/gpu.lock}"
flock -x 201
test -f work/VERIFIED
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python verify_full.py
touch work/FULL_VERIFIED
