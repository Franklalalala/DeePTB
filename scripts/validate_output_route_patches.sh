#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD=${PYTHON_CMD:-python}

${PYTHON_CMD} -m pytest -q \
  dptb/tests/test_cartesian_ict_projector_bank.py \
  dptb/tests/test_output_route_registry.py \
  dptb/tests/test_output_head_equivariance_semantics.py \
  dptb/tests/test_output_route_atomic_forward.py \
  dptb/tests/test_output_route_config_argcheck.py

${PYTHON_CMD} scripts/smoke_output_head_route_matrix.py
${PYTHON_CMD} -m compileall -q dptb scripts

${PYTHON_CMD} - <<'PY'
from pathlib import Path
import yaml
from dptb.utils.argcheck import model_options

model_arg = model_options()
for path in sorted(Path("configs").glob("route_*.yaml")):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    normalized = model_arg.normalize_value(payload["model_options"])
    model_arg.check_value(normalized, strict=True)
    print(f"ARGCHECK_PASS {path}")
PY
