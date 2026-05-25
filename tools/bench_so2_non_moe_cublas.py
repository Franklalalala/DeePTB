from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dptb.nn.tensor_product import SO2_Linear, _Jd, batch_wigner_D
from dptb.nn.tensor_product_moe_v3 import batch_wigner_D_blocks
from e3nn.o3 import xyz_to_angles


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    env: tuple[tuple[str, str], ...] = ()
    wigner: str = "auto"
    strict: bool = True


DEFAULT_VARIANTS = (
    "standard",
    "indexed_sandwich_multi",
    "indexed_sandwich_cuda_multi:output_major",
    "indexed_sandwich_cuda_multi:per_m",
    "indexed_sandwich_cuda_multi:output_major:raw_cached",
    "indexed_sandwich_cuda_multi:output_major:raw_pack_v2",
    "indexed_sandwich_cuda_multi:output_major:raw_pack_v2_m0_cuda",
    "indexed_sandwich_cuda_multi:output_major:grouped_raw_v2",
    "materialized:grouped",
    "materialized:block_dense",
    "materialized_scheduled:scheduler",
    "materialized_scheduled:block_dense",
)


def _cuda_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end) / max(1, iters))
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return ms, peak_mib


def _train_step(module, x, r, latents, target, wigner_D_all=None) -> torch.Tensor:
    module.zero_grad(set_to_none=True)
    x.grad = None
    if latents is not None:
        latents.grad = None
    out, _ = module(x, r, latents, wigner_D_all=wigner_D_all)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    return loss


def _forward_step(module, x, r, latents, wigner_D_all=None) -> torch.Tensor:
    out, _ = module(x, r, latents, wigner_D_all=wigner_D_all)
    return out


def _parse_variant(spec: str) -> Variant:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty variant")
    parts = spec.split(":")
    name = parts[0]
    option = parts[1] if len(parts) > 1 else None
    wigner = parts[2] if len(parts) > 2 else "auto"

    if name == "standard":
        return Variant(spec, "standard", wigner="dense")
    if name == "indexed_sandwich_multi":
        return Variant(spec, "indexed_sandwich_multi", wigner=wigner)
    if name == "indexed_sandwich_cuda":
        return Variant(spec, "indexed_sandwich_cuda", wigner=wigner)
    if name == "indexed_sandwich_cuda_multi":
        schedule = option or "output_major"
        layout = "raw"
        layout_aliases = {
            "raw_output_major_v2_cached": "raw_cached",
            "output_major_v2_cached": "raw_cached",
            "v2_cached": "raw_cached",
            "cached_raw": "raw_cached",
            "raw_output_major_v2_grouped": "grouped_raw_v2",
            "output_major_v2_grouped": "grouped_raw_v2",
            "v2_grouped": "grouped_raw_v2",
            "grouped_v2": "grouped_raw_v2",
            "raw_output_major_v3_pack": "raw_pack_v2",
            "raw_output_major_v3_pack_v2": "raw_pack_v2",
            "pack_v2": "raw_pack_v2",
            "desc_pack": "raw_pack_v2",
            "raw_output_major_v3_pack_m0_cuda": "raw_pack_v2_m0_cuda",
            "raw_output_major_v3_pack_v2_m0_cuda": "raw_pack_v2_m0_cuda",
            "pack_v2_m0_cuda": "raw_pack_v2_m0_cuda",
            "m0_cuda_pack_v2": "raw_pack_v2_m0_cuda",
        }
        allowed_layouts = (
            "raw",
            "raw_cached",
            "raw_pack_v2",
            "raw_pack_v2_m0_cuda",
            "grouped_raw",
            "grouped_raw_v2",
            "cublas_grouped",
            "grouped_gemm",
            "block",
            "block_complex",
            "block_direct",
            "direct_block_complex",
            "compact_block",
            "fairchem_block",
        )
        if len(parts) > 2 and (parts[2] in allowed_layouts or parts[2] in layout_aliases):
            layout = layout_aliases.get(parts[2], parts[2])
            wigner = parts[3] if len(parts) > 3 else "auto"
        return Variant(
            spec,
            "indexed_sandwich_cuda_multi",
            (
                ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", schedule),
                ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", layout),
            ),
            wigner=wigner,
        )
    if name in ("materialized", "indexed_sandwich_materialized"):
        strategy = option or "block_dense"
        env = [("DPTB_SO2_MATERIALIZED_GEMM_STRATEGY", strategy)]
        if len(parts) > 2 and parts[2] in ("per_m", "output_major"):
            env.append(("DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE", parts[2]))
            wigner = parts[3] if len(parts) > 3 else "auto"
        return Variant(
            spec,
            "indexed_sandwich_materialized",
            tuple(env),
            wigner=wigner,
        )
    if name in ("scheduled", "indexed_sandwich_scheduled"):
        mainloop = option or "warp_collective"
        return Variant(
            spec,
            "indexed_sandwich_scheduled",
            (("DPTB_SO2_SCHEDULED_SANDWICH_MAINLOOP", mainloop),),
            wigner=wigner,
        )
    if name in ("materialized_scheduled", "indexed_sandwich_materialized_scheduled"):
        strategy = option or "scheduler"
        env = [("DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY", strategy)]
        if strategy in ("block_dense", "grouped") and len(parts) > 2 and parts[2] in ("per_m", "output_major"):
            env.append(("DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE", parts[2]))
            wigner = parts[3] if len(parts) > 3 else "auto"
        elif strategy == "scheduler" and len(parts) > 2:
            env.append(("DPTB_SO2_MATERIALIZED_SCHEDULED_MAINLOOP", parts[2]))
            wigner = parts[3] if len(parts) > 3 else "auto"
        return Variant(spec, "indexed_sandwich_materialized_scheduled", tuple(env), wigner=wigner)
    raise ValueError(f"unknown variant {spec!r}")


def _variant_list(value: str | None) -> list[Variant]:
    if value:
        specs = [v.strip() for v in value.split(",") if v.strip()]
    else:
        specs = list(DEFAULT_VARIANTS)
    return [_parse_variant(v) for v in specs]


def _set_default_env() -> None:
    os.environ["DPTB_CUBLAS_GROUPED_FAST_TF32"] = "0"
    os.environ["SO2_CUDA_FAST_TF32"] = "0"
    os.environ.setdefault("DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT", "1")
    os.environ.setdefault("DPTB_SO2_MATERIALIZED_STRICT", "1")
    os.environ.setdefault("DPTB_SO2_MATERIALIZED_SCHEDULED_STRICT", "1")
    os.environ.setdefault("DPTB_SO2_SCHEDULED_SANDWICH_STRICT", "1")
    os.environ.setdefault("DPTB_SO2_MATERIALIZED_SCHEDULED_NOSYNC_LAYOUT", "1")
    os.environ.setdefault("DPTB_SO2_SCHEDULED_SANDWICH_NOSYNC_LAYOUT", "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _set_profile_env(enabled: bool, print_every: int) -> None:
    if not enabled:
        return
    os.environ["SO2_CUDA_PROFILE"] = "1"
    os.environ["DPTB_SO2_PROFILE"] = "1"
    os.environ["SO2_CUDA_PROFILE_PRINT_EVERY"] = str(int(print_every))
    os.environ["DPTB_SO2_PROFILE_PRINT_EVERY"] = str(int(print_every))


def _reset_so2_profile(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from so2_cuda_ops import reset_profile_summary
    except Exception:
        return
    reset_profile_summary()


def _take_so2_profile(enabled: bool) -> dict | None:
    if not enabled:
        return None
    try:
        from so2_cuda_ops import get_profile_summary
    except Exception as exc:
        return {"error": repr(exc)}
    return get_profile_summary(reset=True)


def _profiled_so2_cuda_ms(summary: dict | None) -> float | None:
    if not summary or "error" in summary:
        return None
    cuda = summary.get("cuda_ms", {})
    total = float(cuda.get("so2.forward_total", {}).get("mean", 0.0))
    for label, row in cuda.items():
        if label.startswith("so2.backward."):
            total += float(row.get("mean", 0.0))
    return total


def _non_so2_surrounding_ms(step_ms: float | None, summary: dict | None) -> float | None:
    if step_ms is None:
        return None
    profiled = _profiled_so2_cuda_ms(summary)
    if profiled is None:
        return None
    return max(0.0, float(step_ms) - float(profiled))


def _profile_json(summary: dict | None) -> str | None:
    if summary is None:
        return None
    return json.dumps(summary, sort_keys=True)


def _reset_variant_env() -> None:
    os.environ["DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE"] = "per_m"
    os.environ["DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT"] = "raw"
    os.environ["DPTB_SO2_MATERIALIZED_GEMM_STRATEGY"] = "block_dense"
    os.environ["DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE"] = "per_m"
    os.environ["DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY"] = "scheduler"
    os.environ["DPTB_SO2_MATERIALIZED_SCHEDULED_MAINLOOP"] = "warp_collective"
    os.environ["DPTB_SO2_SCHEDULED_SANDWICH_MAINLOOP"] = "warp_collective"


def _apply_env(env: tuple[tuple[str, str], ...]) -> None:
    for key, value in env:
        os.environ[key] = value


def _make_wigner(kind: str, l_max: int, r: torch.Tensor):
    if kind == "none":
        return None
    angle = xyz_to_angles(r[:, [1, 2, 0]])
    gamma = torch.zeros_like(angle[0])
    if kind in ("dense", "auto"):
        return batch_wigner_D(l_max, angle[0], angle[1], gamma, _Jd)
    if kind == "compact":
        return batch_wigner_D_blocks(l_max, angle[0], angle[1], gamma, _Jd)
    raise ValueError(f"unknown wigner kind {kind!r}")


def _clone_input(x_ref, latents_ref):
    x = x_ref.detach().clone().requires_grad_(x_ref.requires_grad)
    latents = None if latents_ref is None else latents_ref.detach().clone().requires_grad_(latents_ref.requires_grad)
    return x, latents


def _make_module(common, mode: str, state_dict, device) -> SO2_Linear:
    module = SO2_Linear(**common, so2_m_linear_mode=mode).to(device=device, dtype=torch.float32).train()
    module.load_state_dict(state_dict, strict=True)
    return module


def _wigner_bytes(wigner) -> int:
    if wigner is None:
        return 0
    if torch.is_tensor(wigner):
        return int(wigner.numel() * wigner.element_size())
    if hasattr(wigner, "blocks"):
        return int(sum(block.numel() * block.element_size() for block in wigner.blocks))
    return 0


def _check_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> tuple[bool, float]:
    max_abs = float((got - ref).abs().max().detach().cpu())
    ok = bool(torch.allclose(got, ref, atol=atol, rtol=rtol))
    if not ok:
        print(f"[warn] {name} differs from reference: max_abs={max_abs:.6e}", file=sys.stderr)
    return ok, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--n-list", default=None, help="Comma-separated row counts. Overrides --n.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--radial-hidden", type=int, default=64)
    parser.add_argument("--irreps-in", default="16x0e + 16x1o + 12x2e + 8x3o")
    parser.add_argument("--irreps-out", default="16x0e + 16x1o + 12x2e + 8x3o")
    parser.add_argument("--no-radial", action="store_true")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--phase", choices=("train", "forward"), default="train")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--so2-profile", action="store_true")
    parser.add_argument("--so2-profile-print-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the non-MoE SO2 benchmark.")

    _set_default_env()
    _set_profile_env(args.so2_profile, args.so2_profile_print_every)
    variants = _variant_list(args.variants)
    n_values = [int(v) for v in args.n_list.split(",")] if args.n_list else [int(args.n)]

    rows = []
    device = torch.device("cuda")
    for n in n_values:
        torch.manual_seed(args.seed + n)
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
        state = ref.state_dict()
        x_ref = torch.randn((n, ref.irreps_in.dim), device=device, dtype=torch.float32, requires_grad=(args.phase == "train"))
        r = torch.randn((n, 3), device=device, dtype=torch.float32)
        latents_ref = None
        if not args.no_radial:
            latents_ref = torch.randn((n, args.latent_dim), device=device, dtype=torch.float32, requires_grad=(args.phase == "train"))
        target = torch.randn((n, ref.irreps_out.dim), device=device, dtype=torch.float32)
        wigner_ref = _make_wigner("dense", ref.l_max, r)

        if args.phase == "train":
            ref_loss = _train_step(ref, x_ref, r, latents_ref, target, wigner_D_all=wigner_ref)
            ref_out = ref_loss.detach()

            def ref_step() -> torch.Tensor:
                return _train_step(ref, x_ref, r, latents_ref, target, wigner_D_all=wigner_ref)

        else:
            with torch.no_grad():
                ref_out, _ = ref(x_ref, r, latents_ref, wigner_D_all=wigner_ref)

            def ref_step() -> torch.Tensor:
                with torch.no_grad():
                    return _forward_step(ref, x_ref, r, latents_ref, wigner_D_all=wigner_ref)

        _reset_so2_profile(args.so2_profile)
        ref_ms, ref_peak = _cuda_ms(ref_step, args.warmup, args.iters)
        ref_profile = _take_so2_profile(args.so2_profile)
        ref_row = {
            "n": n,
            "phase": args.phase,
            "variant": "standard",
            "mode": "standard",
            "wigner": "dense",
            "ms": ref_ms,
            "speedup_vs_standard": 1.0,
            "peak_mib": ref_peak,
            "wigner_mib": _wigner_bytes(wigner_ref) / (1024 * 1024),
            "correct": True,
            "max_abs": 0.0,
            "env": {},
            "so2_profile": _profile_json(ref_profile),
            "so2_profiled_cuda_ms": _profiled_so2_cuda_ms(ref_profile),
            "non_so2_surrounding_cuda_ms": _non_so2_surrounding_ms(ref_ms, ref_profile),
        }
        rows.append(ref_row)
        print(json.dumps(ref_row, sort_keys=True), flush=True)

        for variant in variants:
            if variant.mode == "standard":
                continue
            _reset_variant_env()
            _apply_env(variant.env)
            module = _make_module(common, variant.mode, state, device)
            x, latents = _clone_input(x_ref, latents_ref)
            wigner_kind = "dense" if variant.wigner == "auto" else variant.wigner
            if variant.mode == "standard" and wigner_kind == "compact":
                wigner_kind = "dense"
            wigner = _make_wigner(wigner_kind, module.l_max, r)
            try:
                if args.phase == "train":
                    got = _train_step(module, x, r, latents, target, wigner_D_all=wigner).detach()
                    correct, max_abs = _check_close(variant.name, got, ref_out, atol=args.atol, rtol=args.rtol)

                    def step() -> torch.Tensor:
                        return _train_step(module, x, r, latents, target, wigner_D_all=wigner)

                else:
                    with torch.no_grad():
                        got = _forward_step(module, x, r, latents, wigner_D_all=wigner)
                    correct, max_abs = _check_close(variant.name, got, ref_out, atol=args.atol, rtol=args.rtol)

                    def step() -> torch.Tensor:
                        with torch.no_grad():
                            return _forward_step(module, x, r, latents, wigner_D_all=wigner)

                _reset_so2_profile(args.so2_profile)
                ms, peak = _cuda_ms(step, args.warmup, args.iters)
                so2_profile = _take_so2_profile(args.so2_profile)
                row = {
                    "n": n,
                    "phase": args.phase,
                    "variant": variant.name,
                    "mode": variant.mode,
                    "wigner": wigner_kind,
                    "ms": ms,
                    "speedup_vs_standard": ref_ms / ms if ms else float("inf"),
                    "peak_mib": peak,
                    "wigner_mib": _wigner_bytes(wigner) / (1024 * 1024),
                    "correct": correct,
                    "max_abs": max_abs,
                    "env": dict(variant.env),
                    "so2_profile": _profile_json(so2_profile),
                    "so2_profiled_cuda_ms": _profiled_so2_cuda_ms(so2_profile),
                    "non_so2_surrounding_cuda_ms": _non_so2_surrounding_ms(ms, so2_profile),
                }
            except Exception as exc:
                so2_profile = _take_so2_profile(args.so2_profile)
                row = {
                    "n": n,
                    "phase": args.phase,
                    "variant": variant.name,
                    "mode": variant.mode,
                    "wigner": wigner_kind,
                    "ms": None,
                    "speedup_vs_standard": None,
                    "peak_mib": None,
                    "wigner_mib": _wigner_bytes(wigner) / (1024 * 1024),
                    "correct": False,
                    "max_abs": None,
                    "env": dict(variant.env),
                    "so2_profile": _profile_json(so2_profile),
                    "so2_profiled_cuda_ms": _profiled_so2_cuda_ms(so2_profile),
                    "non_so2_surrounding_cuda_ms": _non_so2_surrounding_ms(None, so2_profile),
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    if args.jsonl:
        path = Path(args.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
