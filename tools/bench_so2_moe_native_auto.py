#!/usr/bin/env python3
"""Compare indexed_sandwich_multi with no-sync persistent/CUTLASS modes.

This script is intentionally a module-level training smoke.  It does not replace
the production bs=32 smoke, but it keeps the mode/env matrix reproducible.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tools" / "bench_so2_moe_fused_train.py"


MODES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "indexed_sandwich_multi",
        {
            "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE": "indexed_sandwich_multi",
            "DPTB_SO2_MOE_FUSED_P0_LOG_SCHEDULE": "1",
        },
    ),
    (
        "persistent_warp_nosync_all_m",
        {
            "DPTB_SO2_FUSION_MODE": "streamed_m_major_persistent_grouped_p1",
            "DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP": "warp_collective",
            "DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0": "1",
            "DPTB_SO2_MOE_PERSISTENT_P1_NOSYNC_LAYOUT": "1",
            "DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_N": "16",
            "DPTB_SO2_MOE_PERSISTENT_P1_LOG_ONCE": "1",
        },
    ),
    (
        "cutlass_native_nosync_auto",
        {
            "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE": "indexed_sandwich_multi_cutlass_native",
            "DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP": "cutlass_native",
            "DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_TILE": "auto",
            "DPTB_SO2_MOE_PERSISTENT_P1_NOSYNC_LAYOUT": "1",
            "DPTB_SO2_MOE_PERSISTENT_P1_LOG_DESCRIPTORS": "1",
            "DPTB_SO2_MOE_PERSISTENT_P1_FORCE_CUTLASS_NATIVE": "1",
        },
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--experts", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--only", choices=[name for name, _env in MODES], action="append")
    args = parser.parse_args()

    common_env = os.environ.copy()
    common_env.setdefault("DPTB_CUBLAS_GROUPED_FAST_TF32", "0")
    common_env.setdefault("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", "cuda_cublas_segmented")
    common_env.setdefault("DPTB_SO2_MOE_PERSISTENT_P1_BACKWARD_MODE", "cuda_cublas_segmented")
    common_env.setdefault("DPTB_MOLE_LINEAR_MODE", "cublas_grouped")

    cmd_base = [
        sys.executable,
        str(BENCH),
        "--n",
        str(args.n),
        "--routes",
        str(args.routes),
        "--experts",
        str(args.experts),
        "--top-k",
        str(args.top_k),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--mole-linear-mode",
        "cublas_grouped",
    ]

    selected = set(args.only or [])
    for name, extra_env in MODES:
        if selected and name not in selected:
            continue
        env = common_env.copy()
        env.update(extra_env)
        if name == "persistent_warp_nosync_all_m":
            fusion_mode = "streamed_m_major_persistent_grouped_p1"
        else:
            fusion_mode = "streamed_m_major_fused_p0"
        cmd = cmd_base + ["--fusion-mode", fusion_mode]
        print(f"=== {name} ===", flush=True)
        print(
            "env "
            + " ".join(f"{k}={v}" for k, v in sorted(extra_env.items())),
            flush=True,
        )
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
