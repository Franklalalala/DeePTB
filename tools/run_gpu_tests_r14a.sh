#!/usr/bin/env bash
# Local-only qualification. No downloads, cluster commands, or deployment.
set -eo pipefail
R14A_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TMPDIR="$R14A_ROOT/tmp/r14a/gpu" TEMP="$R14A_ROOT/tmp/r14a/gpu" TMP="$R14A_ROOT/tmp/r14a/gpu"
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$TMPDIR/cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions" TRITON_CACHE_DIR="$TMPDIR/triton" CUDA_CACHE_PATH="$TMPDIR/cuda_cache"
mkdir -p "$TMPDIR" "$R14A_ROOT/outputs/r14a/gpu"
source "${R14A_SO2_ENV:?set R14A_SO2_ENV to the SO2CUDA environment script}"
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$R14A_ROOT/code:$SO2PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR="$TMPDIR/so2build/cublas_grouped"
export SO2_CUDA_PACK_SCATTER_BUILD_DIR="$TMPDIR/so2build/pack_scatter"
export DPTB_CUBLAS_GROUPED_BUILD_DIR="$TMPDIR/so2build/dptb_cublas"
export DPTB_SO2_MOE_FUSED_P0_BUILD_DIR="$TMPDIR/so2build/pack_scatter"
export SO2_CUDA_SCHEDULER_BUILD_DIR="$TMPDIR/so2build/scheduler"
export SO2_CUDA_CUTLASS_GROUPED_BUILD_DIR="$TMPDIR/so2build/cutlass_grouped"
export SO2_CUDA_CUTLASS_GEMM_SMOKE_BUILD_DIR="$TMPDIR/so2build/cutlass_smoke"
export R14A_TEST_DEVICE=cuda:0 R14A_MINI_ROOT="${R14A_MINI_ROOT:-$R14A_ROOT/tmp/r12b/mini}"
export DPTB_SO2_ACTIVATION_FUSED_P0=1 DPTB_CUBLAS_GROUPED_FAST_TF32=0 SO2_CUDA_FAST_TF32=0 NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Remove backend overrides that could turn a qualification into another route.
unset DPTB_SO2_FUSION_MODE DPTB_SO2_FUSE_M_CUBLAS DPTB_SO2_SORTED_EDGE_VIEW
cd "$R14A_ROOT/code"
python - <<'PY' 2>&1 | tee "$R14A_ROOT/outputs/r14a/gpu/environment.log"
import sys,torch,dptb,os
from pathlib import Path
print('Python:',sys.executable,'torch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
print('dptb:',dptb.__file__,'visible devices:',os.environ['CUDA_VISIBLE_DEVICES'],flush=True)
assert torch.cuda.is_available(), 'GPU unavailable; qualification fails, never skips'
import so2_cuda_ops
print('GPU:',torch.cuda.get_device_name(0),'SO2CUDA:',so2_cuda_ops.__file__,flush=True)
assert Path(os.environ['R14A_MINI_ROOT'],'basis.json').is_file(), 'Local real-record fixture is missing'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
PY
python -m pytest dptb/tests/test_r14a_structure_mole_cuda.py dptb/tests/test_r14a_real_training.py \
    -q -p no:cacheprovider --basetemp="$TMPDIR/pytest" -rs --tb=short \
    --junitxml="$R14A_ROOT/outputs/r14a/gpu/tests.xml" -o junit_family=legacy \
    2>&1 | tee "$R14A_ROOT/outputs/r14a/gpu/tests.log"
python - "$R14A_ROOT/outputs/r14a/gpu/tests.xml" <<'PY'
import sys,xml.etree.ElementTree as ET
r=ET.parse(sys.argv[1]).getroot()
assert len(r.findall('.//testcase')) == 6
assert not r.findall('.//failure') and not r.findall('.//error') and not r.findall('.//skipped')
print('All six GPU checks executed without skips.')
PY
python tools/benchmark_structure_mole.py --device cuda:0 --steps 20 \
    --output "$R14A_ROOT/outputs/r14a/gpu/benchmark.json" \
    2>&1 | tee "$R14A_ROOT/outputs/r14a/gpu/benchmark.log"
mkdir -p "$TMPDIR/baseline"
git archive 677e3c1 | tar -x -C "$TMPDIR/baseline"
PYTHONPATH="$TMPDIR/baseline:$SO2PATH" python tools/check_structure_mole_disabled.py --device cuda:0 \
    --collect "$R14A_ROOT/outputs/r14a/gpu/disabled_original.pt" \
    > "$R14A_ROOT/outputs/r14a/gpu/disabled_original.log" 2>&1
python tools/check_structure_mole_disabled.py --device cuda:0 \
    --collect "$R14A_ROOT/outputs/r14a/gpu/disabled_current.pt" \
    > "$R14A_ROOT/outputs/r14a/gpu/disabled_current.log" 2>&1
python tools/check_structure_mole_disabled.py --compare \
    "$R14A_ROOT/outputs/r14a/gpu/disabled_original.pt" "$R14A_ROOT/outputs/r14a/gpu/disabled_current.pt" \
    | tee "$R14A_ROOT/outputs/r14a/gpu/disabled_comparison.json"
