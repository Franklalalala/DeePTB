set -e
R=/scratch/sheng.lei/0912_h0_serial_edgemoe
JID=${PBS_JOBID%%.*}
LOCAL=/tmp/sheng.lei/h0_serial_$JID
mkdir -p "$LOCAL" "$R/$JID"
export TEMP=$LOCAL TMP=$LOCAL TMPDIR=$LOCAL
export STAGE_ENV_DST=$LOCAL/env
export E2G_CHECKOUT=$R/DeePTB
. /scratch/Projects/CFP-04/CFP04-CF-019/p23_h0res_shared/e2g_fused_0825_scripts/env.sh
test "$ENV_ROOT" = "$LOCAL/env"
export ENV_ROOT
export TEMP=$LOCAL TMP=$LOCAL TMPDIR=$LOCAL
export TORCH_EXTENSIONS_DIR=$LOCAL/cache/torch_ext
export SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR=$LOCAL/cache/cublas_grouped
export SO2_CUDA_PACK_SCATTER_BUILD_DIR=$LOCAL/cache/pack_scatter
mkdir -p "$SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR" "$SO2_CUDA_PACK_SCATTER_BUILD_DIR" "$TORCH_EXTENSIONS_DIR"
cp -n /scratch/Projects/CFP-04/CFP04-CF-019/p23_h0res_shared/so2_cache/cublas_grouped/*.so "$SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR/"
cp -n /scratch/Projects/CFP-04/CFP04-CF-019/p23_h0res_shared/so2_cache/pack_scatter/*.so "$SO2_CUDA_PACK_SCATTER_BUILD_DIR/"
export DPTB_PREAD_INDEX_DIRS=/scratch/e1373662/0911_hopper_audit/index_soc29303_train:/scratch/e1373662/0911_hopper_audit/index_soc29303_valid
export DPTB_PREAD_REQUIRED=1
# The existing allocations exhausted local full-copy staging before takeover.
# Preserve the verified indexed-pread fallback; never mmap shared LMDB.
python3 - <<'PY'
import os,json
from pathlib import Path
r=Path('/scratch/sheng.lei/0912_h0_serial_edgemoe')/os.environ['PBS_JOBID'].split('.')[0]
keys=['PATH','PYTHONPATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','PBS_JOBID','PBS_NODEFILE','TMPDIR','TEMP','TMP','CUDA_HOME','ENV_ROOT','TORCH_EXTENSIONS_DIR','DPTB_PREAD_INDEX_DIRS','DPTB_PREAD_REQUIRED']
keys += [k for k in os.environ if k.startswith(('SO2_CUDA_','DPTB_CUBLAS_','DPTB_CUDA_','DPTB_SO2_','DPTB_MOLE_'))]
(r/'new_runtime.json').write_text(json.dumps({k:os.environ[k] for k in keys if k in os.environ},indent=2))
print('RUNTIME_STAGED',os.environ['ENV_ROOT'],flush=True)
PY
