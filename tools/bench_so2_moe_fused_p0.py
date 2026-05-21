from __future__ import annotations

import argparse
import os

import torch

from dptb.nn.so2_moe_fused_p0 import try_forward_so2_moe_fused_p0
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear


def _cuda_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _make_globals(n: int, routes: int, experts: int, top_k: int, device: torch.device) -> MOLEGlobals:
    graph_index = torch.arange(n, device=device, dtype=torch.long) % routes
    scores = torch.rand((routes, experts), device=device, dtype=torch.float32)
    if top_k >= experts:
        coeffs = scores / scores.sum(dim=-1, keepdim=True)
        topk_indices = None
        topk_values = None
    else:
        raw, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        topk_values = raw / raw.sum(dim=-1, keepdim=True)
        coeffs = torch.zeros_like(scores)
        coeffs.scatter_(1, topk_indices, topk_values)
    return MOLEGlobals(
        coefficients=coeffs,
        graph_index=graph_index,
        topk_indices=topk_indices,
        topk_values=topk_values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--experts", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--irreps-in", default="32x0e + 24x1e + 16x2e + 8x3e")
    parser.add_argument("--irreps-out", default="32x0e + 24x1e + 16x2e + 8x3e")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fused P0 smoke benchmark.")

    os.environ.setdefault("DPTB_CUBLAS_GROUPED_FAST_TF32", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    dtype = torch.float32
    module = SO2_Linear(
        irreps_in=args.irreps_in,
        irreps_out=args.irreps_out,
        radial_emb=True,
        latent_dim=32,
        radial_channels=[64],
        num_experts=args.experts,
        num_shared_experts=1,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="full_dense",
        mole_linear_mode="cublas_grouped",
        so2_fusion_mode="streamed_m_major_fused_p0",
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn((args.n, module.irreps_in.dim), device=device, dtype=dtype)
    R = torch.randn((args.n, 3), device=device, dtype=dtype)
    latents = torch.randn((args.n, 32), device=device, dtype=dtype)
    globals_ = _make_globals(args.n, args.routes, args.experts, args.top_k, device)

    with torch.no_grad():
        ref, wigner = module._forward_streamed_m_major_grouped(
            x,
            R,
            globals_,
            latents,
            None,
            route="streamed_m_major_fused_p0_ref",
        )
        fused_result = try_forward_so2_moe_fused_p0(module, x, R, globals_, latents, wigner)
        if fused_result is None:
            raise RuntimeError("Fused P0 helper declined the benchmark case.")
        fused, _ = fused_result
        max_abs = float((fused - ref).abs().max().detach().cpu())
        max_rel = float(((fused - ref).abs() / (ref.abs() + 1e-6)).max().detach().cpu())

        ref_ms = _cuda_ms(
            lambda: module._forward_streamed_m_major_grouped(
                x,
                R,
                globals_,
                latents,
                wigner,
                route="streamed_m_major_fused_p0_ref",
            ),
            args.warmup,
            args.iters,
        )
        fused_ms = _cuda_ms(
            lambda: try_forward_so2_moe_fused_p0(module, x, R, globals_, latents, wigner),
            args.warmup,
            args.iters,
        )

    print("case,n,routes,experts,top_k,ms,max_abs,max_rel")
    print(f"streamed_grouped,{args.n},{args.routes},{args.experts},{args.top_k},{ref_ms:.6f},0,0")
    print(f"fused_p0,{args.n},{args.routes},{args.experts},{args.top_k},{fused_ms:.6f},{max_abs:.6e},{max_rel:.6e}")
    print(f"speedup_vs_streamed_grouped,{ref_ms / fused_ms:.6f}")


if __name__ == "__main__":
    main()
