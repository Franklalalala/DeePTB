# Triton Fused SO2 Prototype

Date: 2026-04-23

## Scope

This prototype follows the package at
`E:\thu\re_label\0423_read\pr9_fused_kernel_package`.

It adds an opt-in route:

```text
so2_fusion_mode=streamed_m_major_triton_fused
```

The route only fuses the streamed SO2 bridge operations:

```text
rotate-in pack -> SO2_m / MoLELinear -> rotate-out scatter
```

The middle `SO2_m_Linear` / `MOLELinear` path is unchanged. The intended
comparison setting is:

```text
mole_linear_mode=cueq_indexed_linear
so2_m_linear_mode=standard
```

## Validation

Remote env:

```text
host: natlan 172.27.73.246
env: dptb_p2_wigner_cu12_py310
torch: 2.8.0+cu128
triton: 3.4.0
GPU: NVIDIA L40S
checkout: /home/mingkang_nt/codex/0422_tests/pr9_triton_fused/DeePTB
```

Command:

```text
pytest dptb/tests/test_so2_triton_fused_ops.py \
       dptb/tests/test_so2_streamed_lmax_bounds.py -q --tb=short --maxfail=1
```

Result:

```text
57 passed, 20 warnings in 6.37s
```

## Module Benchmark

Setup: fp32, CUDA L40S, `B=32`, `num_experts=24`,
`num_shared_experts=0`, irreps `32x0e + ... + 32x6e`, forward+backward.

| E | config | mean ms | peak allocated GB |
|---:|---|---:|---:|
| 4096 | staged all cuEq | 31.71 | 0.454 |
| 4096 | streamed all cuEq | 46.52 | 0.475 |
| 4096 | Triton fused all cuEq | 55.24 | 0.509 |
| 8192 | staged all cuEq | 45.51 | 0.794 |
| 8192 | streamed all cuEq | 49.80 | 0.844 |
| 8192 | Triton fused all cuEq | 56.84 | 0.916 |
| 16384 | staged all cuEq | 86.76 | 1.425 |
| 16384 | streamed all cuEq | 65.33 | 1.543 |
| 16384 | Triton fused all cuEq | 62.86 | 1.676 |

Readout:

- The Triton route only wins at `E=16384`, where it is about 3.8% faster
  than `streamed all cuEq`.
- It is slower at `E=4096` and `E=8192`.
- It increases peak allocated memory in all module benchmark cases.

## Production-like bs32 A/B

Setup: dataset `/home/mingkang_nt/data/0422_test`, DDP on 2 L40S,
`batch_size=32`, `num_epoch=1`, `display_freq=2`, `validation_freq=20`,
`save_freq=1000`, `sliding_win_size=5`, `monitor_cuda_memory=true`,
`num_experts=24`, `top_k=24`, `num_shared_experts=0`.

Result file:

```text
/home/mingkang_nt/codex/0422_tests/pr9_triton_fused/runs_bs32_triton/bs32_triton_ab_results.json
```

| config | result | train wall time | peak allocated | peak reserved |
|---|---|---:|---:|---:|
| `streamed_m_major_cueq + cueq_indexed_linear` | pass | 51.647 s | 29282.3 MB | 42192.0 MB |
| `streamed_m_major_triton_fused + cueq_indexed_linear` | pass | 51.178 s | 32660.0 MB | 41590.0 MB |

## Decision

This is a reviewable prototype, not a recommended production default.

The production-like bs32 result shows only a 0.91% wall-time improvement while
increasing peak allocated memory by about 3.38 GB. The current recommended
setting for this dataset remains:

```text
so2_wigner_apply_mode=compact_blocks
so2_fusion_mode=streamed_m_major_cueq
mole_linear_mode=cueq_indexed_linear
so2_m_linear_mode=standard
mole_full_expert_fast_path=true
```

The next higher-value fusion attempt should move deeper into
`SO2_m_Linear` / `so2_m_linear_mode`, rather than only fusing the outer
streamed SO2 pack/scatter bridge.
