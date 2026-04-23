# Triton Complex Grouped Linear Intake

Date: 2026-04-23

## Source Reviewed

Reviewed `E:\deeptb\codex\0422\pr11_triton_next_deep_patch_package\pr11_triton_next_deep`.

The package proposes a deeper Triton route:

- keep the stronger outer SO2 route (`streamed_m_major_cueq` or `staged`)
- use Triton grouped real linear for generic `MOLELinear`
- use Triton grouped complex linear for `SO2_m_Linear(m>0)`

## What Was Absorbed

Only the useful `m>0` complex grouped idea was integrated.

The package's overlay and duplicate real mode were not copied. The current branch already has `mole_linear_mode=triton_grouped_linear`, so adding `triton_grouped_linear_deep` would duplicate semantics and increase config ambiguity.

Implemented opt-in mode:

```text
so2_m_linear_mode=triton_complex_grouped_linear
```

This mode:

- reuses `dptb/nn/so2_triton_grouped_linear_ops.py`
- keeps defaults unchanged
- bypasses the standard `MOLELinear -> [N, 2, 2*Cout] -> real/imag combine` path for `m>0`
- directly computes:

```text
y_real = x_real @ W_real.T - x_imag @ W_imag.T
y_imag = x_real @ W_imag.T + x_imag @ W_real.T
```

The Triton kernels are fp32-only through the existing `DPTB_TRITON_LINEAR_REQUIRE` / `DPTB_TRITON_LINEAR_DISABLE` guard style.

## Why The Overlay Was Not Absorbed

The PR11 overlay is not suitable for direct merge:

- monkey-patches `MOLELinear.forward` and `SO2_m_Linear.forward`
- redefines mode normalizers
- introduces duplicate mode names: `triton_grouped_linear_deep`, `triton_complex_grouped_deep`
- package tests only validate CPU fallback, not CUDA Triton execution
- original Triton scheduling uses `while True` / `break` style that was already problematic in prior Triton attempts
- runtime guard allowed fp16/bf16 without enough validation

## Verification

Local Windows:

```text
python -m py_compile dptb\nn\so2_triton_grouped_linear_ops.py dptb\nn\tensor_product_moe_v3.py dptb\tests\test_so2_triton_grouped_linear_ops.py dptb\utils\argcheck.py
PASS

pytest dptb\tests\test_so2_triton_grouped_linear_ops.py -q
1 skipped in 0.04s
```

The local pytest skip is expected because the local Windows environment does not provide the required torch test runtime.

natlan CUDA:

```text
python -m py_compile dptb/nn/so2_triton_grouped_linear_ops.py dptb/nn/tensor_product_moe_v3.py dptb/tests/test_so2_triton_grouped_linear_ops.py dptb/utils/argcheck.py
pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q --tb=short --maxfail=1
14 passed, 1 warning in 33.34s
```

Additional `SO2_m_Linear` CUDA fp32 parity smoke with `DPTB_TRITON_LINEAR_REQUIRE=1`:

```text
forward_max      8.940696716308594e-08
x_grad_max       1.1920928955078125e-07
coeff_grad_max   1.9073486328125e-06
weight_grad_max  9.5367431640625e-07
```

## Micro Benchmark

Shape:

- CUDA fp32
- `n_graphs=32`
- `rows_per_graph=256`
- `N=8192`
- `num_experts=24`
- irreps: `32x1o + 32x2e + 32x3o + 32x4e + 32x5o + 32x6e`
- forward + backward, 3 warmup, 10 measured iterations

Compared:

| route | mean ms | min ms | max ms | peak allocated MB | peak reserved MB |
|---|---:|---:|---:|---:|---:|
| `standard + cueq_indexed_linear` | 3.935 | 3.655 | 4.248 | 262.6 | 304.0 |
| `triton_complex_grouped_linear` | 7.033 | 6.704 | 7.196 | 203.3 | 476.0 |

Observed:

- Forward parity max diff: `5.960464477539062e-07`
- `triton_complex_grouped_linear` reduced allocated memory by about `59 MB` in this micro-benchmark
- It was about `1.79x` slower than the current cuEq indexed-linear standard path
- Reserved memory increased because Triton runtime/cache overhead dominates this small benchmark

## Current Guidance

Do not make `triton_complex_grouped_linear` a production default.

It is useful as an isolated experimental backend and as a correctness reference for future deeper fusion, but current performance does not justify replacing:

```text
so2_fusion_mode=streamed_m_major_cueq
mole_linear_mode=cueq_indexed_linear
so2_m_linear_mode=standard
```

Next performance work should avoid just replacing the middle complex linear. The remaining potential is in a larger fused boundary that removes surrounding layout materialization and also avoids slow `grad_weight` torch reductions.

## Aggressive Fused-MoE Follow-Up

An additional experimental backend was prototyped:

```text
so2_m_linear_mode=triton_complex_moe_fused_linear
```

This route fuses full-expert coefficient mixing with the `m>0` complex SO2 linear:

```text
y_real = sum_e c[e] * (x_real @ W_real[e].T - x_imag @ W_imag[e].T)
y_imag = sum_e c[e] * (x_real @ W_imag[e].T + x_imag @ W_real[e].T)
```

It avoids materializing:

```text
mixed_weights: [num_graphs, 2*Cout, Cin]
```

and still avoids the standard path's:

```text
x_proj: [N, 2, 2*Cout]
```

The first naive version performed one complex dot per expert inside the Triton tile and measured about `164.85 ms`, which was an immediate no-go. The kernel was then changed to first build tile-local mixed `Wr/Wi` in registers and perform one complex dot per tile. Correctness still passed, but performance remained poor.

Verification on natlan:

```text
pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q --tb=short --maxfail=1
18 passed, 1 warning in 28.97s
```

Micro-benchmark shape:

- CUDA fp32
- `n_graphs=32`
- `rows_per_graph=256`
- `N=8192`
- `num_experts=24`
- irreps: `32x1o + 32x2e + 32x3o + 32x4e + 32x5o + 32x6e`
- forward + backward, 3 warmup, 10 measured iterations

| route | mean ms | min ms | max ms | peak allocated MB | peak reserved MB |
|---|---:|---:|---:|---:|---:|
| `standard + cueq_indexed_linear` | 4.247 | 3.756 | 4.625 | 239.3 | 288.0 |
| `triton_complex_grouped_linear` | 7.465 | 7.107 | 7.639 | 180.1 | 460.0 |
| `triton_complex_moe_fused_linear` | 144.147 | 143.626 | 145.577 | 177.8 | 460.0 |

Observed:

- Forward max diff vs standard: `5.960464477539062e-07`
- The fused-MoE backend only saves about `2.25 MB` allocated relative to `triton_complex_grouped_linear`
- It is about `19.3x` slower than `triton_complex_grouped_linear`
- It is about `33.9x` slower than `standard + cueq_indexed_linear`

Decision:

```text
Do not run production bs32/bs48 A/B for triton_complex_moe_fused_linear.
```

The bottleneck is not merely `mixed_weights` materialization. Fusing dense 24-expert mixing into the same Triton tile introduces too much extra weight traffic and register work. The next useful Triton direction should move to a larger layout-fusion boundary only if it can also reduce surrounding pack/scatter and avoid Python/Torch reductions in backward.

## Fused Expert Linear Intake From PR12 Next Package

Reviewed:

```text
E:\deeptb\codex\0422\pr12_triton_next_fused_expert_patch_package\pr12_triton_next_fused_expert
```

Useful idea absorbed:

```text
mole_linear_mode=triton_fused_expert_linear
```

This is a general `MOLELinear` full-expert route. It accepts raw graph coefficients and expert banks directly instead of materializing:

```text
mixed_weights: [num_graphs, out_features, in_features]
mixed_bias:    [num_graphs, out_features]
```

The safe default path is a PyTorch oracle/fallback that computes each graph's mixed weight locally. The actual Triton execution is guarded behind:

```text
DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT=1
```

Reason: the package itself had no CUDA validation. In this branch, the direct Triton kernel was made compile-safe by replacing the persistent dynamic tile scanner with a simpler group-major 3D grid, and small CUDA parity passed. However, larger 8192-row MOLELinear smoke exposed an illegal-memory-access failure for module-initialized weights. Because of that, the Triton execution path is not enabled by default and should not be used in production.

Validation:

```text
python -m py_compile dptb\nn\so2_triton_grouped_linear_ops.py dptb\nn\tensor_product_moe_v3.py dptb\tests\test_so2_triton_grouped_linear_ops.py dptb\utils\argcheck.py
PASS

local pytest:
1 skipped in 0.06s

natlan:
pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q --tb=short --maxfail=1
32 passed, 1 skipped, 1 warning in 52.15s
```

Micro signal before disabling by default:

```text
MOLELinear 2D, N=8192, n_graphs=32, rows_per_graph=256, experts=24, in=64, out=64

split_loop:            mean 8.623 ms, peak allocated 43.9 MB
triton_grouped_linear: mean 5.373 ms, peak allocated 43.9 MB
triton_fused_expert:   illegal memory access on larger module smoke
```

Decision:

```text
Do not recommend triton_fused_expert_linear as an optimization.
Keep it as a correctness oracle / isolated experiment only.
Current production guidance remains unchanged.
```
