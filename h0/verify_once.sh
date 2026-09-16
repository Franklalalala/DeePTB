#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
exec 201>"${H0_GPU_LOCK:-$H0_WORK_ROOT/gpu.lock}"
flock -x 201
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python verify_prebuilt.py
# A second Python process checks startup reuse, with build tools absent from PATH.
"$(command -v python)" precompile.py --check
touch work/VERIFIED
