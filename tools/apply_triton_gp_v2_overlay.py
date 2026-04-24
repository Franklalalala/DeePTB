#!/usr/bin/env python3
"""Apply the Triton graph-persistent exact MoE V2 overlay to DeePTB.

Run from the DeePTB repository root after copying the overlay files:

    python3 tools/apply_triton_gp_v2_overlay.py

The script is intentionally idempotent and only edits
`dptb/nn/so2_triton_grouped_linear_ops.py` by adding guarded imports plus two
public function dispatch hooks.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path.cwd()
TARGET = ROOT / "dptb" / "nn" / "so2_triton_grouped_linear_ops.py"
MARKER = "# Triton exact graph-persistent V2 overlay"

IMPORT_BLOCK = f"""
{MARKER}
try:
    from .so2_triton_exact_gp_v2 import (
        complex_exact_moe_linear_v2 as _complex_exact_moe_linear_v2,
        exact_moe_linear_v2 as _exact_moe_linear_v2,
        use_complex_exact_gp_v2 as _use_complex_exact_gp_v2,
        use_exact_gp_v2 as _use_exact_gp_v2,
    )
except Exception:  # pragma: no cover - additive experimental route must not break default imports
    _exact_moe_linear_v2 = None
    _complex_exact_moe_linear_v2 = None

    def _use_exact_gp_v2() -> bool:
        return os.environ.get("DPTB_TRITON_EXACT_GP_V2", "0").strip().lower() in {{"1", "true", "yes", "on"}}

    def _use_complex_exact_gp_v2() -> bool:
        return os.environ.get(
            "DPTB_TRITON_COMPLEX_EXACT_GP_V2",
            os.environ.get("DPTB_TRITON_EXACT_GP_V2", "0"),
        ).strip().lower() in {{"1", "true", "yes", "on"}}
"""

REAL_HOOK = '''    if _use_exact_gp_v2():
        if _exact_moe_linear_v2 is None:
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V2 is enabled, but dptb.nn.so2_triton_exact_gp_v2 could not be imported."
            )
        return _exact_moe_linear_v2(
            x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            split_sizes,
        )
'''

COMPLEX_HOOK = '''    if _use_complex_exact_gp_v2():
        if _complex_exact_moe_linear_v2 is None:
            raise RuntimeError(
                "DPTB_TRITON_COMPLEX_EXACT_GP_V2 is enabled, but dptb.nn.so2_triton_exact_gp_v2 could not be imported."
            )
        return _complex_exact_moe_linear_v2(
            x_pair,
            coefficients,
            weight_experts,
            shared_weight,
            split_sizes,
        )
'''


def insert_import_block(text: str) -> str:
    if MARKER in text:
        return text
    anchor = "import torch.nn.functional as F\n"
    if anchor not in text:
        raise RuntimeError(f"Could not find import anchor {anchor!r} in {TARGET}")
    return text.replace(anchor, anchor + IMPORT_BLOCK, 1)


def insert_real_hook(text: str) -> str:
    if "_exact_moe_linear_v2(" in text:
        return text
    pattern = re.compile(
        r"(def grouped_exact_moe_linear\([\s\S]*?\) -> torch\.Tensor:\n"
        r"\s+split_sizes = _canonical_split_sizes\(split_sizes\)\n)"
        r"(\s+return _GroupedExactMoELinearFn\.apply\()",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find grouped_exact_moe_linear dispatch point")
    return text[: match.start()] + match.group(1) + REAL_HOOK + match.group(2) + text[match.end() :]


def insert_complex_hook(text: str) -> str:
    if "_complex_exact_moe_linear_v2(" in text:
        return text
    pattern = re.compile(
        r"(def grouped_complex_exact_moe_linear\([\s\S]*?\) -> torch\.Tensor:\n"
        r"\s+split_sizes = _canonical_split_sizes\(split_sizes\)\n)"
        r"(\s+return _GroupedComplexExactMoELinearFn\.apply\()",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find grouped_complex_exact_moe_linear dispatch point")
    return text[: match.start()] + match.group(1) + COMPLEX_HOOK + match.group(2) + text[match.end() :]


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} does not exist. Run from the DeePTB repo root.", file=sys.stderr)
        return 2
    original = TARGET.read_text()
    text = insert_import_block(original)
    text = insert_real_hook(text)
    text = insert_complex_hook(text)
    if text == original:
        print("Triton GP V2 overlay already applied.")
        return 0
    TARGET.write_text(text)
    print(f"Applied Triton GP V2 overlay to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
