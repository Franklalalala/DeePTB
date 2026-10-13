#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
exec 200>"$H0_WORK_ROOT/precompile.lock"
flock -x 200
export CUDA_VISIBLE_DEVICES=""
python -m dptb.priors.h0.precompile
python -m dptb.priors.h0.precompile --check
