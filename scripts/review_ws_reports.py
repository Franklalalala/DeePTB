#!/usr/bin/env python3
"""Summarize WS0/WS4 report decisions for gate review."""
from __future__ import annotations

import argparse
from pathlib import Path


def contains(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws0", type=Path, required=True)
    ap.add_argument("--ws4", type=Path, required=True)
    args = ap.parse_args()
    ws0 = args.ws0.read_text(encoding="utf-8", errors="ignore")
    ws4 = args.ws4.read_text(encoding="utf-8", errors="ignore")

    checks = [
        ("WS0: PQ/κ 非主导", "部分证实，非主导机制" in ws0 or "非主导机制" in ws0),
        ("WS0: ep697 已正式评估", "ep697" in ws0 and "0.007075" in ws0),
        ("WS0: full-H 输出语义", "全量 H，非 ΔH 残差头" in ws0),
        ("WS0: edge flow-time 已在 HEAD 修复", "flow_time_condition_edges" in ws0),
        ("WS4: restart_dh 基座可用", "restart_dh" in ws4 and "9b10c72d5" in ws4),
        ("WS4: BTO/GaAs 不动点通过", "BTO" in ws4 and "GaAs" in ws4 and "1e-5" in ws4),
        ("WS4: 水分子不可直接 ABACUS 修复", "Hartree 制、GTO 血统" in ws4),
    ]
    print("# WS gate review")
    for name, ok in checks:
        print(f"- [{'x' if ok else ' '}] {name}")


if __name__ == "__main__":
    main()
