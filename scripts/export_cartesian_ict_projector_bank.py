#!/usr/bin/env python3
"""Export DeePTB AO projectors through the explicit Cartesian/STF generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dptb.nn.embedding.cartesian_ict_bank import (
    export_cartesian_ict_projector_bank,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument(
        "--shell",
        action="append",
        required=True,
        help="AO shell in OrbitalMapper order; repeat for every super-basis shell.",
    )
    parser.add_argument("--validation-atol", type=float, default=2.0e-10)
    args = parser.parse_args()
    path = export_cartesian_ict_projector_bank(
        args.output,
        args.shell,
        validation_atol=args.validation_atol,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
