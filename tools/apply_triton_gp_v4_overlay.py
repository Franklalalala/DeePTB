#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Apply the Triton exact graph-persistent V4 route hook.

Run this script from the DeePTB repository root after applying the V4 additive
patch.  It is idempotent and only inserts guarded dispatch code into
``dptb/nn/so2_triton_grouped_linear_ops.py``.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path.cwd()
TARGET = ROOT / "dptb" / "nn" / "so2_triton_grouped_linear_ops.py"
MARKER = "# Triton exact graph-persistent V4 overlay"

IMPORT_BLOCK = f'''
{MARKER}
try:
    from .so2_triton_exact_gp_v4 import (
        complex_exact_moe_linear_v4 as _complex_exact_moe_linear_v4,
        exact_moe_linear_v4 as _exact_moe_linear_v4,
        use_complex_exact_gp_v4 as _use_complex_exact_gp_v4,
        use_exact_gp_v4 as _use_exact_gp_v4,
    )
except Exception:  # pragma: no cover - additive experimental route must not break default imports
    _exact_moe_linear_v4 = None
    _complex_exact_moe_linear_v4 = None

    def _use_exact_gp_v4() -> bool:
        return os.environ.get("DPTB_TRITON_EXACT_GP_V4", "0").strip().lower() in {{"1", "true", "yes", "on"}}

    def _use_complex_exact_gp_v4() -> bool:
        return os.environ.get(
            "DPTB_TRITON_COMPLEX_EXACT_GP_V4",
            os.environ.get("DPTB_TRITON_EXACT_GP_V4", "0"),
        ).strip().lower() in {{"1", "true", "yes", "on"}}
'''

REAL_HOOK = '''    if _use_exact_gp_v4():
        if _exact_moe_linear_v4 is None:
            raise RuntimeError(
                "DPTB_TRITON_EXACT_GP_V4 is enabled, but dptb.nn.so2_triton_exact_gp_v4 could not be imported."
            )
        return _exact_moe_linear_v4(
            x,
            coefficients,
            weight_experts,
            bias_experts,
            shared_weight,
            shared_bias,
            split_sizes,
        )
'''

COMPLEX_HOOK = '''    if _use_complex_exact_gp_v4():
        if _complex_exact_moe_linear_v4 is None:
            raise RuntimeError(
                "DPTB_TRITON_COMPLEX_EXACT_GP_V4 is enabled, but dptb.nn.so2_triton_exact_gp_v4 could not be imported."
            )
        return _complex_exact_moe_linear_v4(
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
    # Place V4 before V3/V2 so V4 wins when multiple experimental flags are enabled.
    for marker in ("# Triton exact graph-persistent V3 overlay\n", "# Triton exact graph-persistent V2 overlay\n"):
        if marker in text:
            return text.replace(marker, IMPORT_BLOCK + "\n" + marker, 1)
    anchor = "import torch.nn.functional as F\n"
    if anchor not in text:
        raise RuntimeError(f"Could not find import anchor {anchor!r} in {TARGET}")
    return text.replace(anchor, anchor + IMPORT_BLOCK, 1)


def insert_real_hook(text: str) -> str:
    if "_exact_moe_linear_v4(" in text:
        return text
    pattern = re.compile(
        r"(def grouped_exact_moe_linear\([\s\S]*?\) -> torch\.Tensor:\n"
        r"\s+split_sizes = _canonical_split_sizes\(split_sizes\)\n)"
        r"(\s+if _use_exact_gp_v3\(\):|\s+if _use_exact_gp_v2\(\):|\s+return _GroupedExactMoELinearFn\.apply\()",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find grouped_exact_moe_linear dispatch point")
    return text[: match.start()] + match.group(1) + REAL_HOOK + match.group(2) + text[match.end() :]


def insert_complex_hook(text: str) -> str:
    if "_complex_exact_moe_linear_v4(" in text:
        return text
    pattern = re.compile(
        r"(def grouped_complex_exact_moe_linear\([\s\S]*?\) -> torch\.Tensor:\n"
        r"\s+split_sizes = _canonical_split_sizes\(split_sizes\)\n)"
        r"(\s+if _use_complex_exact_gp_v3\(\):|\s+if _use_complex_exact_gp_v2\(\):|\s+return _GroupedComplexExactMoELinearFn\.apply\()",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find grouped_complex_exact_moe_linear dispatch point")
    return text[: match.start()] + match.group(1) + COMPLEX_HOOK + match.group(2) + text[match.end() :]


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} does not exist. Run from the DeePTB repository root.", file=sys.stderr)
        return 2
    text = TARGET.read_text()
    updated = insert_complex_hook(insert_real_hook(insert_import_block(text)))
    if updated != text:
        TARGET.write_text(updated)
        print(f"Updated {TARGET}")
    else:
        print(f"No changes needed for {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
