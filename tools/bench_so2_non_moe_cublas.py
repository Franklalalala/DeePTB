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

from dptb.nn.tensor_product import SO2_Linear


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


def _train_step(module, x, r, latents, target):
    module.zero_grad(set_to_none=True)
    x.grad = None
    if latents is not None:
        latents.grad = None
    out, _ = module(x, r, latents)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--radial-hidden", type=int, default=64)
    parser.add_argument("--irreps-in", default="16x0e + 16x1o + 12x2e + 8x3o")
    parser.add_argument("--irreps-out", default="16x0e + 16x1o + 12x2e + 8x3o")
    parser.add_argument("--no-radial", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the non-MoE SO2 benchmark.")

    os.environ.setdefault("DPTB_CUBLAS_GROUPED_FAST_TF32", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(20260523)
    device = torch.device("cuda")
    common = dict(
        irreps_in=args.irreps_in,
        irreps_out=args.irreps_out,
        radial_emb=not args.no_radial,
        latent_dim=args.latent_dim,
        radial_channels=[args.radial_hidden],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**common, so2_m_linear_mode="standard").to(device=device, dtype=torch.float32).train()
    grouped = SO2_Linear(**common, so2_m_linear_mode="indexed_sandwich_multi").to(device=device, dtype=torch.float32).train()
    cuda_pack = SO2_Linear(**common, so2_m_linear_mode="indexed_sandwich_cuda").to(device=device, dtype=torch.float32).train()
    grouped.load_state_dict(ref.state_dict(), strict=True)
    cuda_pack.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn((args.n, ref.irreps_in.dim), device=device, dtype=torch.float32, requires_grad=True)
    x_grouped = x_ref.detach().clone().requires_grad_(True)
    x_cuda_pack = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn((args.n, 3), device=device, dtype=torch.float32)
    if args.no_radial:
        latents_ref = None
        latents_grouped = None
        latents_cuda_pack = None
    else:
        latents_ref = torch.randn((args.n, args.latent_dim), device=device, dtype=torch.float32, requires_grad=True)
        latents_grouped = latents_ref.detach().clone().requires_grad_(True)
        latents_cuda_pack = latents_ref.detach().clone().requires_grad_(True)
    target = torch.randn((args.n, ref.irreps_out.dim), device=device, dtype=torch.float32)

    loss_ref = _train_step(ref, x_ref, r, latents_ref, target)
    loss_grouped = _train_step(grouped, x_grouped, r, latents_grouped, target)
    loss_cuda_pack = _train_step(cuda_pack, x_cuda_pack, r, latents_cuda_pack, target)
    loss_abs = float((loss_grouped.detach() - loss_ref.detach()).abs().cpu())
    cuda_pack_loss_abs = float((loss_cuda_pack.detach() - loss_ref.detach()).abs().cpu())
    x_grad_abs = float((x_grouped.grad - x_ref.grad).abs().max().detach().cpu())
    cuda_pack_x_grad_abs = float((x_cuda_pack.grad - x_ref.grad).abs().max().detach().cpu())

    def ref_step() -> None:
        _train_step(ref, x_ref, r, latents_ref, target)

    def grouped_step() -> None:
        _train_step(grouped, x_grouped, r, latents_grouped, target)

    def cuda_pack_step() -> None:
        _train_step(cuda_pack, x_cuda_pack, r, latents_cuda_pack, target)

    ref_ms = _cuda_ms(ref_step, args.warmup, args.iters)
    grouped_ms = _cuda_ms(grouped_step, args.warmup, args.iters)
    cuda_pack_ms = _cuda_ms(cuda_pack_step, args.warmup, args.iters)

    print("case,n,radial,mode,ms,loss_abs_diff,x_grad_max_abs")
    print(f"standard,{args.n},{not args.no_radial},standard,{ref_ms:.6f},0,0")
    print(f"indexed_sandwich_multi,{args.n},{not args.no_radial},indexed_sandwich_multi,{grouped_ms:.6f},{loss_abs:.6e},{x_grad_abs:.6e}")
    print(f"indexed_sandwich_cuda,{args.n},{not args.no_radial},indexed_sandwich_cuda,{cuda_pack_ms:.6f},{cuda_pack_loss_abs:.6e},{cuda_pack_x_grad_abs:.6e}")
    print(f"speedup_vs_standard,{ref_ms / grouped_ms:.6f}")
    print(f"speedup_vs_standard_indexed_sandwich_cuda,{ref_ms / cuda_pack_ms:.6f}")


if __name__ == "__main__":
    main()
