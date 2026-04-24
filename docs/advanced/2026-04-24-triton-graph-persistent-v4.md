# Triton graph-persistent exact MoE V4

Date: 2026-04-24
Branch target: `0422-triton-exact-graph-mix-complex` at latest reviewed tip
`a627c25 [codex] Isolate V3 CPU fallback tests`
Status: experimental, opt-in, memory-first, benchmark-gated

## Why V4 exists

The current branch already contains the V3 overlay.  The latest reviewed commit
`a627c25` is a V3 CPU fallback test isolation patch; it only clears
`DPTB_TRITON_EXACT_GP_V3_REQUIRE` inside selected CPU fallback tests and does
not change V3 kernels or runtime dispatch.  V4 copies that lesson by clearing
V4/V3/V2 REQUIRE flags in its CPU fallback tests.

V3 improved the backward reduce grid from V2:

```text
V2 reduce grid:
  (graph, expert, row_chunk, out_tile, in_tile)

V3 reduce grid:
  (graph, row_chunk, out_tile, in_tile)
  for expert in experts:
      atomic_add dW_expert
      atomic_add dCoeff_graph_expert
```

That is the right direction because `x` and `grad_y` are loaded once per tile
instead of once per expert.  The remaining hot spot is the scalar
`grad_coeff[g, e]` atomic: every row chunk, N tile, and K tile contributes to the
same coefficient-gradient scalar.  For large graphs or many N/K tiles, this
creates high atomic contention.

V4 keeps V3's no-`mixed_weights` / no-`grad_mixed_w` property, but changes only
coefficient-gradient accumulation:

```text
V4 first reduce kernel:
  grid = (graph, row_chunk, out_tile, in_tile)
  for expert in experts:
      atomic_add dW_expert
      atomic_add dBias / shared grads
      store dCoeff_partial[graph, expert, tile_id]

V4 second tiny reduce kernel:
  grid = (graph, expert)
  grad_coeff[graph, expert] = sum_tile dCoeff_partial[graph, expert, tile]
```

The expected gain is lower scalar atomic contention on `grad_coeff`.  The cost is
a bounded scratch buffer of size:

```text
num_graphs * num_experts * max_chunks_per_graph * tiles_n * tiles_k * sizeof(dtype)
```

This scratch is usually much smaller than explicit `mixed_weights` or
`grad_mixed_w`, but V4 still guards it with
`DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB`.

## Files

```text
dptb/nn/so2_triton_exact_gp_v4.py
dptb/tests/test_so2_triton_exact_gp_v4.py
tools/apply_triton_gp_v4_overlay.py
docs/advanced/2026-04-24-triton-graph-persistent-v4.md
```

## Apply and integrate

```bash
git apply triton_gp_v4_overlay_additive.patch
python3 tools/apply_triton_gp_v4_overlay.py
```

The overlay script is idempotent.  It inserts V4 guarded imports and dispatches
V4 before V3/V2, so V4 wins when several experimental env flags are enabled.

## Runtime switches

```bash
export DPTB_TRITON_EXACT_GP_V4=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V4=1
export DPTB_TRITON_EXACT_GP_V4_BWD=split_coeff
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Safe fallbacks:

```bash
# Reuse V3 expert-loop atomic reduce.
export DPTB_TRITON_EXACT_GP_V4_BWD=v3_atomic

# Reuse V2 atomic reduce.
export DPTB_TRITON_EXACT_GP_V4_BWD=v2_atomic

# Exact Torch backward for correctness A/B.
export DPTB_TRITON_EXACT_GP_V4_BWD=torch
```

Fail fast on a GPU node when V4 cannot actually launch:

```bash
export DPTB_TRITON_EXACT_GP_V4_REQUIRE=1
```

## Scratch guard

```bash
# Default: 128 MB.  Set to 0 to disable the guard.
export DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB=128
```

If the coefficient-partials buffer would exceed the limit, V4 falls back to V3's
expert-loop atomic reduce.  With `DPTB_TRITON_EXACT_GP_V4_REQUIRE=1`, it raises
instead so benchmark jobs do not silently measure the wrong route.

## Tile knobs

```bash
export DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_M=64
export DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_N=16
export DPTB_TRITON_EXACT_GP_V4_REDUCE_BLOCK_K=32
export DPTB_TRITON_EXACT_GP_V4_COEFF_REDUCE_BLOCK_T=1024
```

The first three knobs fall back to V3 names, then V2 names, so existing tuning
scripts can still be reused.

## Validation

CPU/static validation:

```bash
python3 -m py_compile \
  dptb/nn/so2_triton_exact_gp_v4.py \
  dptb/tests/test_so2_triton_exact_gp_v4.py \
  tools/apply_triton_gp_v4_overlay.py

PYTHONPATH=. python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v4.py -q
```

CUDA smoke validation:

```bash
export DPTB_TRITON_EXACT_GP_V4=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V4=1
export DPTB_TRITON_EXACT_GP_V4_BWD=split_coeff
export DPTB_TRITON_EXACT_GP_V4_REQUIRE=1
PYTHONPATH=. python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v4.py -q
```

## Benchmark matrix

```bash
# 1. Production baseline.
unset DPTB_TRITON_EXACT_GP_V2 DPTB_TRITON_COMPLEX_EXACT_GP_V2
unset DPTB_TRITON_EXACT_GP_V3 DPTB_TRITON_COMPLEX_EXACT_GP_V3
unset DPTB_TRITON_EXACT_GP_V4 DPTB_TRITON_COMPLEX_EXACT_GP_V4
python -m dptb ... 2>&1 | tee logs/baseline_cueq.log

# 2. Current branch V3.
export DPTB_TRITON_EXACT_GP_V3=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V3=1
export DPTB_TRITON_EXACT_GP_V3_BWD=expert_loop
python -m dptb ... 2>&1 | tee logs/gp_v3_expert_loop.log

# 3. New V4 split coefficient reduce.
unset DPTB_TRITON_EXACT_GP_V3 DPTB_TRITON_COMPLEX_EXACT_GP_V3
export DPTB_TRITON_EXACT_GP_V4=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V4=1
export DPTB_TRITON_EXACT_GP_V4_BWD=split_coeff
export DPTB_TRITON_EXACT_GP_V4_COEFF_SCRATCH_LIMIT_MB=128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m dptb ... 2>&1 | tee logs/gp_v4_split_coeff.log
```

Accept gate:

```text
forward / x_grad / coeff_grad / weight_grad / shared_grad parity pass
peak_allocated_gib <= V3
peak_reserved_gib <= V3, or clearly explained by bounded scratch
sec/iter <= V3 expert_loop on Natlan bs32/bs48
no silent fallback when REQUIRE=1
```

## Known risks

1. V4 adds one extra tiny kernel and a scratch tensor.  It is expected to help
   when `grad_coeff` atomic contention is large, but V3 may be faster for small
   graphs or very small N/K tile counts.
2. `dW` and shared-weight gradients still use atomic accumulation over row
   chunks.  Removing those atomics would require either a much larger scratch
   buffer or a more specialized two-stage reduce.
3. The atomic `dW` path remains non-bitwise-deterministic at the last few ulps.
