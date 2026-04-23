# Triton Exact Graph-Mix Route

This note records the next Triton experiment after the earlier row-tile fused
expert path.

## Route A: Exact Graph Mix + Grouped Apply/Reduce

New opt-in modes:

```text
mole_linear_mode=triton_exact_grouped_linear
so2_m_linear_mode=triton_complex_exact_grouped_linear
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

Limit: the scalar exact route now has Triton grouped reduce for `dWmix/dBmix`.
The complex exact route still uses the existing torch grouped complex reduce in
backward, so it is a correctness/perf candidate, not the final full fused
kernel.

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
python -m py_compile dptb\nn\so2_triton_grouped_linear_ops.py dptb\nn\tensor_product_moe_v3.py dptb\tests\test_so2_triton_grouped_linear_ops.py dptb\utils\argcheck.py
PASS

python -m pytest dptb\tests\test_so2_triton_grouped_linear_ops.py -q
1 skipped
```

The skip is expected on this local machine because torch is unavailable.

Required CUDA validation on natlan when shell access is available:

```bash
source /home/mingkang_nt/data/anaconda3/etc/profile.d/conda.sh
conda activate dptb_p2_wigner_cu12_py310
cd /home/mingkang_nt/codex/0422_tests/pr13_triton_lab/DeePTB

PYTHONPATH=$PWD DPTB_TRITON_LINEAR_REQUIRE=1 \
python -m pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q --tb=short --maxfail=1
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

