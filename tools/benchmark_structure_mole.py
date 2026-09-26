#!/usr/bin/env python
"""Small complete forward/backward/Adam-step comparison, serial GPU residency."""
import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path

import torch

from dptb.nn import so2_activation_routes as dispatch
from dptb.tests.structure_mole_helpers import calibrated_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cuda = args.device.startswith("cuda")
    if cuda:
        assert torch.cuda.is_available(), "Requested CUDA benchmark must execute on CUDA"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    assert args.steps > 0
    records = {}
    for execution in ("reference", "merged_core"):
        model, data = calibrated_model(device=args.device, execution=execution,
                                        mole_linear_mode="cublas_grouped" if cuda else "split_loop",
                                        n_layers=3, tp_radial_emb=True, use_interpolation_out=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        rows = []
        before_calls = dispatch.STATS.calls.get(dispatch.FUSED_P0, 0)
        for step in range(args.steps + 3):
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out = model(copy.deepcopy(data))
            loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if step >= 3:
                rows.append(dict(step=step-3, seconds=seconds, loss=float(loss.detach()),
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None))
        calls = dispatch.STATS.calls.get(dispatch.FUSED_P0, 0) - before_calls
        if cuda:
            assert (calls > 0) == (execution == "reference"), "unexpected backend dispatch"
        records[execution] = dict(rows=rows, mean_seconds=sum(r["seconds"] for r in rows)/len(rows),
                                  max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows) if cuda else None,
                                  observed_fused_p0_calls=calls,
                                  structures=3, nodes=data["pos"].shape[0], edges=data["edge_index"].shape[1],
                                  parameters=sum(p.numel() for p in model.parameters()),
                                  embedding_options=model.model_options["embedding"])
        del out, loss, opt, model, data
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    torch.testing.assert_close(torch.tensor([r["loss"] for r in records["reference"]["rows"]]),
                               torch.tensor([r["loss"] for r in records["merged_core"]["rows"]]), atol=2e-5, rtol=2e-4)
    Path(args.output).write_text(json.dumps(dict(device=args.device, torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if cuda else None, warmup_steps=3, records=records), indent=2))
    print(json.dumps({k: {kk:v for kk,v in r.items() if kk in ("mean_seconds","max_peak_allocated_bytes","observed_fused_p0_calls")}
                      for k,r in records.items()},indent=2))


if __name__ == "__main__":
    main()
