#!/usr/bin/env bash
set -eo pipefail

ROOT="${ROOT:-${1:-}}"
if [[ -z "${ROOT}" ]]; then
  echo "Usage: ROOT=/path/to/monitor_run bash $0"
  exit 2
fi

echo "== host =="
hostname || true
date || true

echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

echo "== compute processes =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo "== matching processes =="
ps -eo pid,ppid,etime,cmd --sort=start_time \
  | grep -E 'run_monitor|multi_train|dptb|0427_cuda_jump_monitor|torchrun|python' \
  | grep -v grep || true

echo "== run root =="
ls -lh "${ROOT}" || true

echo "== monitor files =="
find "${ROOT}" -maxdepth 4 -type f \
  \( -name '*.log' -o -name '*.csv' -o -name '*.pid' -o -name 'events.out*' \) \
  -printf '%p %s\n' 2>/dev/null | sort || true

for f in \
  "${ROOT}/smoke_console.log" \
  "${ROOT}/formal_console.log" \
  "${ROOT}/output_smoke/log/log.txt" \
  "${ROOT}/output_smoke/log/cuda_monitor.log" \
  "${ROOT}/output_formal/log/log.txt" \
  "${ROOT}/output_formal/log/cuda_monitor.log" \
  "${ROOT}/output_smoke/rank0/cuda_module_memory.csv" \
  "${ROOT}/output_smoke/rank1/cuda_module_memory.csv"; do
  if [[ -f "${f}" ]]; then
    echo "== tail ${f} =="
    tail -80 "${f}" || true
  fi
done

echo "== key smoke/formal markers =="
grep -R --line-buffered \
  -E 'CUDA_CACHE_MEMORY|CUDA Module Memory Monitor|iteration:|Traceback|RuntimeError|CUDA out of memory|SMOKE_END|FORMAL_START' \
  "${ROOT}"/*.log "${ROOT}"/output_smoke/log/*.log "${ROOT}"/output_formal/log/*.log \
  2>/dev/null | tail -160 || true
