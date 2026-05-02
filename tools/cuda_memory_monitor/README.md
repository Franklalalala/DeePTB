# CUDA jump memory monitor

This directory contains operational helpers for the `0427-cuda-jump-monitor`
branch. The branch adds low-overhead CUDA memory attribution for long DeePTB
training runs where GPU usage rises in step-like jumps.

## What is monitored

- Iteration-window CUDA peaks already reported by `monitor_cuda_memory`.
- Cache-miss probes for SO2/Wigner static tensors and cuEquivariance indexed
  linear module construction.
- Expert forward and validation forward context tags around cache probes.
- Optional `monitor_cuda_module_memory` CUDA module CSV for SO2 linear,
  MOLELinear, S2/FFN, and non-TorchScript TensorProduct wrappers. This can be
  enabled without `monitor_flag`, so production runs do not also enable the
  heavier DeepDoctor/SO2 gradient monitors.

TorchScript `RecursiveScriptModule` and `ScriptModule` instances cannot accept
Python forward hooks, so they are intentionally skipped. Their allocation jumps
should still be visible through the enclosing module rows and cache-miss probes.

Recommended production monitor options:

```json
{
  "monitor_cuda_memory": true,
  "monitor_cuda_cache_memory": true,
  "monitor_cuda_module_memory": true,
  "monitor_cuda_module_memory_min_delta_mb": 64.0,
  "debug_tags": true,
  "debug_tag_freq": 500,
  "monitor_flag": false
}
```

## Liyue helpers

Run these on liyue from a prepared monitor run directory:

```bash
ROOT=/home/mingkang_nt/codex/0427_cuda_jump_monitor_YYYYMMDD_HHMMSS \
  bash tools/cuda_memory_monitor/inspect_liyue_cuda_monitor.sh

ROOT=/home/mingkang_nt/codex/0427_cuda_jump_monitor_YYYYMMDD_HHMMSS \
  bash tools/cuda_memory_monitor/launch_liyue_formal_monitor.sh
```

The scripts do not contain credentials and do not assume a scheduler. They
inspect GPU/process/log state and start the formal monitor with the run
directory's `run_formal.sh`.
