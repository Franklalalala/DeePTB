#!/usr/bin/env python
from __future__ import annotations

"""Probe pMF/CFM flow-geometry diagnostics from tensors.

With no input this runs a deterministic synthetic smoke.  With ``--npz`` it
expects NumPy arrays named after the diagnostic tensors, for example:

  target_v, du_dt, flow_grad, jvp_grad
  t, node_current, node_target, node_endpoint
  node_t, edge_current, edge_target, edge_endpoint, edge_t
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dptb.nnops.flow_diagnostics import (
    cfm_chord_cosine_diagnostics,
    cosine_similarity_tensors,
    pixel_meanflow_du_dt_diagnostics,
)


def _to_jsonable(state: Dict[str, torch.Tensor]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            out[key] = float(value)
            continue
        scalar = value.detach().cpu().reshape(())
        item = float(scalar.item())
        out[key] = item
    return out


def _synthetic_probe() -> Dict[str, object]:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    pmf = pixel_meanflow_du_dt_diagnostics(
        target_v=torch.tensor([3.0, 4.0]),
        du_dt=torch.tensor([0.0, 5.0]),
        flow_loss=(param.square()).sum(),
        jvp_loss=-(param.square()).sum(),
        parameters=[param],
    )
    cfm = cfm_chord_cosine_diagnostics(
        node_current=torch.zeros(1, 2),
        node_target=torch.tensor([[1.0, 0.0]]),
        node_endpoint=torch.tensor([[0.0, 1.0]]),
        edge_current=torch.zeros(1, 2),
        edge_target=torch.tensor([[0.0, 2.0]]),
        edge_endpoint=torch.tensor([[0.0, 4.0]]),
        t=torch.tensor([0.25]),
    )
    return {
        "source": "synthetic",
        "pixel_meanflow": _to_jsonable(pmf),
        "cfm": _to_jsonable(cfm),
    }


def _load_tensor(arrays, key: str) -> Optional[torch.Tensor]:
    if key not in arrays:
        return None
    return torch.as_tensor(arrays[key])


def _npz_probe(path: Path) -> Dict[str, object]:
    import numpy as np

    result: Dict[str, object] = {"source": str(path), "keys": []}
    with np.load(path) as arrays:
        result["keys"] = sorted(str(k) for k in arrays.files)
        target_v = _load_tensor(arrays, "target_v")
        du_dt = _load_tensor(arrays, "du_dt")
        if target_v is not None and du_dt is not None:
            pmf = pixel_meanflow_du_dt_diagnostics(target_v=target_v, du_dt=du_dt)
            flow_grad = _load_tensor(arrays, "flow_grad")
            jvp_grad = _load_tensor(arrays, "jvp_grad")
            if flow_grad is not None and jvp_grad is not None:
                pmf["grad_cos_flow_jvp"] = cosine_similarity_tensors(flow_grad, jvp_grad)
            result["pixel_meanflow"] = _to_jsonable(pmf)

        t = _load_tensor(arrays, "t")
        if t is not None:
            cfm = cfm_chord_cosine_diagnostics(
                t=t,
                node_t=_load_tensor(arrays, "node_t"),
                node_current=_load_tensor(arrays, "node_current"),
                node_target=_load_tensor(arrays, "node_target"),
                node_endpoint=_load_tensor(arrays, "node_endpoint"),
                edge_t=_load_tensor(arrays, "edge_t"),
                edge_current=_load_tensor(arrays, "edge_current"),
                edge_target=_load_tensor(arrays, "edge_target"),
                edge_endpoint=_load_tensor(arrays, "edge_endpoint"),
            )
            result["cfm"] = _to_jsonable(cfm)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, help="optional tensor bundle to probe")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    result = _npz_probe(args.npz) if args.npz else _synthetic_probe()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
