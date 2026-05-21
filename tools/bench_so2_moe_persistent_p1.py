#!/usr/bin/env python3
"""Run the existing SO2/MoE fused benchmark with P1 enabled.

This is a thin wrapper around DeePTB's ``tools/bench_so2_moe_fused_p0.py``.
Place this script in the DeePTB checkout or run it from the checkout root.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


os.environ.setdefault("DPTB_SO2_FUSION_MODE", "streamed_m_major_persistent_grouped_p1")
os.environ.setdefault("DPTB_MOLE_LINEAR_MODE", "cublas_grouped")
os.environ.setdefault("DPTB_SO2_MOE_PERSISTENT_P1_LOG_ONCE", "1")

bench = Path("tools") / "bench_so2_moe_fused_p0.py"
if not bench.exists():
    print("tools/bench_so2_moe_fused_p0.py not found; run from the DeePTB checkout root", file=sys.stderr)
    raise SystemExit(2)

sys.argv = [str(bench)] + sys.argv[1:]
runpy.run_path(str(bench), run_name="__main__")
