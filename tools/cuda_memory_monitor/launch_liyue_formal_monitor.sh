#!/usr/bin/env bash
set -eo pipefail

ROOT="${ROOT:-${1:-}}"
if [[ -z "${ROOT}" ]]; then
  echo "Usage: ROOT=/path/to/monitor_run bash $0"
  exit 2
fi

cd "${ROOT}"

if [[ ! -x run_formal.sh && ! -f run_formal.sh ]]; then
  echo "run_formal.sh not found in ${ROOT}"
  exit 2
fi

if [[ -f formal.pid ]]; then
  pid="$(cat formal.pid || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "formal monitor already running: pid=${pid}"
    exit 0
  fi
fi

echo "== gpu before launch =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo "FORMAL_START $(date -Is)" >> formal_console.log
nohup bash run_formal.sh >> formal_console.log 2>&1 &
pid="$!"
echo "${pid}" > formal.pid
sleep 8

echo "formal monitor pid=${pid}"
if [[ -f status.sh ]]; then
  bash status.sh || true
fi
tail -80 formal_console.log || true
