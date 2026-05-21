from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear


def _cuda_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
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


def _make_module(args, mode: str, device: torch.device) -> SO2_Linear:
    return SO2_Linear(
        irreps_in=args.irreps_in,
        irreps_out=args.irreps_out,
        radial_emb=True,
        latent_dim=args.latent_dim,
        radial_channels=[args.radial_hidden],
        num_experts=args.experts,
        num_shared_experts=args.shared,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
        mole_linear_mode=args.mole_linear_mode,
        so2_fusion_mode=mode,
    ).to(device=device, dtype=torch.float32).train()


def _train_step(module, x, R, globals_, latents, target, wigner) -> torch.Tensor:
    module.zero_grad(set_to_none=True)
    x.grad = None
    latents.grad = None
    out, _ = module(x, R, globals_, latents, wigner)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--experts", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--shared", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--radial-hidden", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mole-linear-mode", default="cublas_grouped",
                        choices=("cublas_grouped", "cueq_indexed_linear"))
    parser.add_argument("--fusion-mode", default="streamed_m_major_fused_p0")
    parser.add_argument("--irreps-in", default="32x0e + 24x1e + 16x2e + 8x3e")
    parser.add_argument("--irreps-out", default="32x0e + 24x1e + 16x2e + 8x3e")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fused P0 training benchmark.")

    os.environ.setdefault("DPTB_CUBLAS_GROUPED_FAST_TF32", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    ref = _make_module(args, "streamed_m_major_cueq", device)
    fused = _make_module(args, args.fusion_mode, device)
    fused.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn((args.n, ref.irreps_in.dim), device=device, dtype=torch.float32, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    R = torch.randn((args.n, 3), device=device, dtype=torch.float32)
    latents_ref = torch.randn((args.n, args.latent_dim), device=device, dtype=torch.float32, requires_grad=True)
    latents_fused = latents_ref.detach().clone().requires_grad_(True)
    target = torch.randn((args.n, ref.irreps_out.dim), device=device, dtype=torch.float32)
    globals_ = _make_globals(args.n, args.routes, args.experts, args.top_k, device)
    wigner = ref._ensure_wigner_rotation(R, None)

    loss_ref = _train_step(ref, x_ref, R, globals_, latents_ref, target, wigner)
    loss_fused = _train_step(fused, x_fused, R, globals_, latents_fused, target, wigner)
    max_abs = float((loss_fused.detach() - loss_ref.detach()).abs().cpu())
    grad_abs = float((x_fused.grad - x_ref.grad).abs().max().detach().cpu())

    def ref_step() -> None:
        _train_step(ref, x_ref, R, globals_, latents_ref, target, wigner)

    def fused_step() -> None:
        _train_step(fused, x_fused, R, globals_, latents_fused, target, wigner)

    ref_ms = _cuda_ms(ref_step, args.warmup, args.iters)
    fused_ms = _cuda_ms(fused_step, args.warmup, args.iters)

    print("case,n,routes,experts,top_k,mole_linear_mode,fusion_mode,ms,loss_abs_diff,x_grad_max_abs")
    print(f"streamed_grouped_train,{args.n},{args.routes},{args.experts},{args.top_k},{args.mole_linear_mode},streamed_m_major_cueq,{ref_ms:.6f},0,0")
    print(f"fused_train,{args.n},{args.routes},{args.experts},{args.top_k},{args.mole_linear_mode},{args.fusion_mode},{fused_ms:.6f},{max_abs:.6e},{grad_abs:.6e}")
    print(f"speedup_vs_streamed_grouped_train,{ref_ms / fused_ms:.6f}")


if __name__ == "__main__":
    main()
