#!/usr/bin/env python3
"""Export the Route-B angular-projector interchange schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dptb.nn.embedding.ao_projector_bank import export_projector_bank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument(
        "--shell",
        action="append",
        required=True,
        help="AO shell in OrbitalMapper order; repeat, e.g. --shell 1s --shell 2s",
    )
    parser.add_argument(
        "--source",
        default="reference_wigner",
        help="provenance label; a Cartesian/ICT exporter should set its own label",
    )
    args = parser.parse_args()
    path = export_projector_bank(args.output, args.shell, source=args.source)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
