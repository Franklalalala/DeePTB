#!/usr/bin/env python
"""Smoke benchmark for DeePTB Pixel MeanFlow du/dt backends.

Compares finite_difference vs jvp on a synthetic Hamiltonian-feature batch and
a counting endpoint MLP: per-step wall clock (forward+backward), explicit model
calls, whether jvp silently fell back, and CUDA peak memory.

    python tools/bench_pixel_meanflow_du_dt_backend.py \
        --device cuda --batch-size 96 --backend both \
        --hidden-dim 512 --num-layers 4 --node-dim 64 --edge-dim 64

It answers whether the backend path is exercised and how the derivative
strategy moves time/memory on plain PyTorch ops. It is not a substitute for a
Hanhai production run on the real SO2/CUDA stack.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from typing import Dict, Iterable, Tuple

import torch

from dptb.nnops.flow import HamiltonianPixelMeanFlow


class CountingMLPEndpoint(torch.nn.Module):
    """Time-conditioned x-prediction MLP with a call counter."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()

        def _mlp(in_dim: int, out_dim: int) -> torch.nn.Sequential:
            dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(torch.nn.SiLU())
            return torch.nn.Sequential(*layers)

        self.node = _mlp(node_dim + 2, node_dim)
        self.edge = _mlp(edge_dim + 2, edge_dim)
        self.calls = 0

    @staticmethod
    def _node_time(data: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        t = data[key]
        batch = data["batch"].to(device=t.device, dtype=torch.long)
        return t.index_select(0, batch)

    @staticmethod
    def _edge_time(data: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        t = data[key]
        batch = data["batch"].to(device=t.device, dtype=torch.long)
        src = data["edge_index"][0].to(device=t.device, dtype=torch.long)
        return t.index_select(0, batch.index_select(0, src))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.calls += 1
        out = data.copy()
        nt = self._node_time(data, "flow_time_t").reshape(-1, 1)
        nh = self._node_time(data, "flow_time_h").reshape(-1, 1)
        et = self._edge_time(data, "flow_time_t").reshape(-1, 1)
        eh = self._edge_time(data, "flow_time_h").reshape(-1, 1)
        out["node_features"] = self.node(torch.cat([data["node_h0"], nt, nh], dim=-1))
        out["edge_features"] = self.edge(torch.cat([data["edge_h0"], et, eh], dim=-1))
        return out


def _cycle_edges(num_nodes: int, num_edges: int, offset: int, device: torch.device) -> torch.Tensor:
    src = torch.arange(num_edges, device=device, dtype=torch.long) % num_nodes + offset
    dst = (torch.arange(num_edges, device=device, dtype=torch.long) + 1) % num_nodes + offset
    return torch.stack([src, dst], dim=0)


def make_batch(
    *,
    num_graphs: int,
    nodes_per_graph: int,
    edges_per_graph: int,
    node_dim: int,
    edge_dim: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    node_count = num_graphs * nodes_per_graph
    edge_count = num_graphs * edges_per_graph
    batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph)
    edge_index = torch.cat(
        [
            _cycle_edges(nodes_per_graph, edges_per_graph, g * nodes_per_graph, device)
            for g in range(num_graphs)
        ],
        dim=1,
    )
    data = {
        "batch": batch,
        "edge_index": edge_index,
        "node_h0": torch.randn(node_count, node_dim, device=device) * 0.1,
        "edge_h0": torch.randn(edge_count, edge_dim, device=device) * 0.1,
        "node_features": torch.zeros(node_count, node_dim, device=device),
        "edge_features": torch.zeros(edge_count, edge_dim, device=device),
    }
    ref = {
        "batch": batch,
        "edge_index": edge_index,
        "node_features": torch.randn(node_count, node_dim, device=device),
        "edge_features": torch.randn(edge_count, edge_dim, device=device),
    }
    return data, ref


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_backend(args: argparse.Namespace, backend: str) -> Dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = CountingMLPEndpoint(
        args.node_dim, args.edge_dim, args.hidden_dim, args.num_layers
    ).to(device)
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "loss_type": "mse",
            "meanflow": {
                "du_dt_backend": backend,
                "jvp_tangent": args.jvp_tangent,
                "fd_eps": args.fd_eps,
                "aux_endpoint_weight": 0.0,
                "aux_boundary_v_weight": args.aux_boundary_v_weight,
                "jvp_fallback": not args.jvp_fail_fast,
            },
        }
    )
    data, ref = make_batch(
        num_graphs=args.batch_size,
        nodes_per_graph=args.nodes_per_graph,
        edges_per_graph=args.edges_per_graph,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        device=device,
    )
    r = torch.full((args.batch_size,), 0.25, device=device)
    t = torch.full((args.batch_size,), 0.75, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    calls = []
    explicit_calls = []
    jvp_used = []
    for step in range(args.warmup_steps + args.steps):
        model.calls = 0
        start = time.perf_counter()
        loss, state = flow.loss_with_model(model, data, ref, r=r, t=t)
        loss.backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if step >= args.warmup_steps:
            times.append(elapsed_ms)
            calls.append(float(model.calls))
            explicit_calls.append(float(state["train_flow_explicit_model_calls"].item()))
            jvp_used.append(float(state["train_flow_du_dt_backend_jvp"].item()))

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "backend": backend,
        "jvp_tangent": args.jvp_tangent,
        "batch_size": args.batch_size,
        "nodes_per_graph": args.nodes_per_graph,
        "edges_per_graph": args.edges_per_graph,
        "node_dim": args.node_dim,
        "edge_dim": args.edge_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "model_calls_mean": statistics.mean(calls),
        "explicit_model_calls_mean": statistics.mean(explicit_calls),
        "jvp_used_fraction": statistics.mean(jvp_used),
        "peak_memory_mb": peak_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["finite_difference", "jvp", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--nodes-per-graph", type=int, default=12)
    parser.add_argument("--edges-per-graph", type=int, default=48)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--fd-eps", type=float, default=1.0e-4)
    parser.add_argument("--jvp-tangent", choices=["path", "boundary"], default="boundary")
    parser.add_argument("--aux-boundary-v-weight", type=float, default=0.0)
    parser.add_argument(
        "--jvp-fail-fast",
        action="store_true",
        help="disable the finite_difference fallback so jvp failures raise",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    backends: Iterable[str]
    if args.backend == "both":
        backends = ("finite_difference", "jvp")
    else:
        backends = (args.backend,)

    result = {
        "commit": _git_commit(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "python_platform": platform.platform(),
        "results": [run_backend(args, backend) for backend in backends],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
