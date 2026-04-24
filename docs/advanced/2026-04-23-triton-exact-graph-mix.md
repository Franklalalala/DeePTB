# Triton Exact Graph-Mix Route

This note records the next Triton experiment after the earlier row-tile fused
expert path.

## Route A: Exact Graph Mix + Grouped Apply/Reduce

New opt-in modes:

```text
mole_linear_mode=triton_exact_grouped_linear
so2_m_linear_mode=triton_complex_exact_grouped_linear
```

Compatibility aliases are also accepted for package handoff scripts:

```text
mole_linear_mode=triton_exact_graph_mix_grouped
so2_m_linear_mode=triton_complex_exact_graph_mix_grouped
```

The intended production experiment still keeps the outer SO2 route on the
current best bridge:

```text
so2_fusion_mode=streamed_m_major_cueq
mole_linear_m0_mode=triton_exact_grouped_linear
so2_m_linear_mode=triton_complex_exact_grouped_linear
```

The scalar path follows the exact graph-level formulation:

```text
Wmix = coeff @ Wexp_flat
Bmix = coeff @ Bexp
Y    = grouped_linear(X, Wmix, Bmix, graph_ptr)
```

Backward does not save `Wmix` in `ctx`. It saves only `X`, `coeff`, expert
weights/biases, optional shared weights/biases, and graph splits. Backward
recomputes `Wmix`, computes `dX` through grouped apply, computes graph-level
`dWmix/dBmix` through grouped reduce, then maps those gradients back:

```text
dCoeff = dWmix_flat @ Wexp_flat.T + dBmix @ Bexp.T
dWexp  = coeff.T @ dWmix_flat
dBexp  = coeff.T @ dBmix
```

The complex `m > 0` path keeps the same graph-level exact mix, then applies:

```text
Yr = Xr @ Wr.T - Xi @ Wi.T
Yi = Xr @ Wi.T + Xi @ Wr.T
```

Both scalar and complex exact routes now use Triton grouped reduce in backward.
The implementation remains opt-in because the exact graph-mix route reduces
activation/cache pressure but adds extra graph-level weight mixing work.

## Route B: Graph-Persistent Full Fusion

This is the next terminal design, not implemented in this patch:

```text
for graph g, output tile, input tile:
    Wmix_tile = sum_e coeff[g,e] * Wexp[e,tile]
    reuse Wmix_tile across multiple row tiles in graph g
```

This differs from the previous `triton_fused_expert_linear` path, which mixes
expert weights inside each row tile and repeats the dense expert loop for every
row tile. The graph-persistent route should reuse a mixed weight tile across
multiple row tiles, but it has higher register/shared-memory pressure and a
harder backward reduction problem.

## CUDA Stack Constraint

Do not change the production CUDA version for this route.

Official documentation confirms useful future primitives, but they are not used
here:

- CUDA 13.1 introduced experimental cuBLASLt grouped GEMM with grouped matrix
  layouts and device-array shapes. The documented initial support targets
  newer GPU capability requirements, so it is not a drop-in dependency for the
  current L40S stack.
- cuDNN backend release notes show MoE grouped matmul support in the runtime
  fusion engine, focused on newer Blackwell configurations. It is a future
  reference point, not this branch's implementation dependency.

## Validation

Local Windows:

```text
conda run -n dptb python -m py_compile dptb/nn/tensor_product_moe_v3.py dptb/nn/so2_triton_grouped_linear_ops.py dptb/tests/test_so2_triton_grouped_linear_ops.py
PASS

conda run -n dptb pytest -q dptb/tests/test_so2_triton_grouped_linear_ops.py
52 passed, 9 skipped

conda run -n dptb pytest -q dptb/tests/test_mole_linear_indexed_ref.py
8 passed, 4 skipped

conda run -n dptb pytest -q dptb/tests/test_so2_streamed_lmax_bounds.py
45 passed, 9 skipped
```

The skips are expected on this local machine because CUDA is available but the
`triton` Python package is not installed in the `dptb` conda environment.

Required CUDA validation on natlan when shell access is available:

```bash
source /home/mingkang_nt/data/anaconda3/etc/profile.d/conda.sh
conda activate dptb_p2_wigner_cu12_py310
cd /home/mingkang_nt/codex/0422_tests/pr13_triton_lab/DeePTB

PYTHONPATH=$PWD DPTB_TRITON_LINEAR_REQUIRE=1 \
python -m pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q --tb=short --maxfail=1
```

Completed natlan CUDA validation:

```text
commit 0ada9ab:
  test_grouped_complex_exact_moe_linear_cuda_uses_triton_dw_reduce
  PASS, 2 passed, 1 warning in 16.64s

commit 16149fc:
  test_grouped_complex_moe_fused_linear_cuda_uses_triton_backward_reduce
  test_grouped_complex_moe_fused_linear_cuda_fp32_if_available
  test_grouped_complex_exact_moe_linear_cuda_uses_triton_dw_reduce
  test_grouped_complex_exact_moe_linear_cuda_fp32_if_available
  PASS, 4 passed, 1 warning in 5.64s
```

Production-like A/B should compare:

```text
baseline:
  so2_fusion_mode=streamed_m_major_cueq
  mole_linear_mode=cueq_indexed_linear
  so2_m_linear_mode=standard

route A:
  so2_fusion_mode=streamed_m_major_cueq
  mole_linear_m0_mode=triton_exact_grouped_linear
  so2_m_linear_mode=triton_complex_exact_grouped_linear
```

## Natlan Production A/B on 0422 Test Data

Repository: `/home/mingkang_nt/codex/0422_tests/pr13_exact_graph_mix/DeePTB`

Commit: `16149fc Use Triton reduce for complex fused MoE backward`

Dataset/input base: `/home/mingkang_nt/codex/0422_tests/pr13_exact_graph_mix/prod_inputs`

Each run used two L40S GPUs and stopped after the first `Epoch 1 summary`. The
reported step time is a short-run smoke metric and includes model startup and
first-use kernel compilation. It is useful for route comparison, not a stable
long-run throughput number.

| Batch size | Route | Allocator | Status | sec/iter | samples/s | peak allocated | peak reserved |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | baseline | default | valid epoch summary | 6.969 | 4.592 | 28.60 GiB | 41.21 GiB |
| 32 | exact graph mix | default | valid epoch summary | 16.148 | 1.982 | 28.23 GiB | 37.94 GiB |
| 48 | baseline | default | OOM | - | - | - | - |
| 48 | exact graph mix | default | OOM | - | - | - | - |
| 48 | baseline | `expandable_segments:True` | valid epoch summary | 10.452 | 4.592 | 41.23 GiB | 43.66 GiB |
| 48 | exact graph mix | `expandable_segments:True` | valid epoch summary | 24.223 | 1.982 | 40.70 GiB | 41.41 GiB |

Observed deltas:

```text
bs32 exact vs baseline:
  peak allocated: -0.37 GiB
  peak reserved:  -3.27 GiB
  throughput:     -56.8%

bs48 exact vs baseline with expandable allocator:
  peak allocated: -0.53 GiB
  peak reserved:  -2.25 GiB
  throughput:     -56.8%
```

The default allocator bs48 failures reported fragmentation symptoms: PyTorch
had roughly 9.7-11.0 GiB reserved but unallocated and failed a 2.63 GiB request.
Using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` made bs48 run to epoch
summary without changing the CUDA version or model math.

Conclusion: this Triton exact graph-mix route is useful as a memory-saving
exploration branch and proves bs48 can be made runnable with allocator tuning,
but it is not a production default. The current production default should remain
the cueq/compact path unless a later graph-persistent fusion removes the extra
weight-mix overhead.
