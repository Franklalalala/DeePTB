export H0_CUDA_ROOT=/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate
export H0_OFFLINE_TABLE_DIR="$H0_CUDA_ROOT/offline_tables"
source /home/mingkang_nt/codex/h0_cuda_benchmark_20260912/scripts/env.sh
export CUDA_VISIBLE_DEVICES=1
export TEMP="$H0_CUDA_ROOT/tmp" TMP="$H0_CUDA_ROOT/tmp" TMPDIR="$H0_CUDA_ROOT/tmp"
export XDG_CACHE_HOME="$H0_CUDA_ROOT/cache" CUDA_CACHE_PATH="$H0_CUDA_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$H0_CUDA_ROOT/cache/torch_extensions" PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$H0_CUDA_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$H0_CUDA_ROOT/cache/numba" PIP_CACHE_DIR="$H0_CUDA_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$H0_CUDA_ROOT/cache/pycache" MAX_JOBS=2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/home/mingkang_nt/anaconda3/envs/new_soc/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/home/mingkang_nt/anaconda3/envs/new_soc/lib:/home/mingkang_nt/codex/h0_flash_20260912/deps/pyabacus/ModuleNAO:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$H0_CUDA_ROOT:$PYTHONPATH"
mkdir -p "$TEMP" "$XDG_CACHE_HOME"
