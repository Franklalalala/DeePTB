# Source after activating an environment with torch, pyabacus and CUDA libraries.
export H0_CUDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${H0_WORK_ROOT:?Set H0_WORK_ROOT to an external writable task directory}"
export H0_WORK_ROOT
export H0_OFFLINE_TABLE_DIR="${H0_OFFLINE_TABLE_DIR:-$H0_WORK_ROOT/offline_tables}"
export TEMP="$H0_WORK_ROOT/tmp" TMP="$H0_WORK_ROOT/tmp" TMPDIR="$H0_WORK_ROOT/tmp"
export XDG_CACHE_HOME="$H0_WORK_ROOT/cache" CUDA_CACHE_PATH="$H0_WORK_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$H0_WORK_ROOT/cache/torch_extensions" PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$H0_WORK_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$H0_WORK_ROOT/cache/numba"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$TEMP" "$XDG_CACHE_HOME"
