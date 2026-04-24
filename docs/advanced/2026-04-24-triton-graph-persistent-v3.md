# Triton graph-persistent exact MoE V3

Date: 2026-04-24
Branch target: `0422-triton-exact-graph-mix-complex` after `d039337`
Status: experimental, opt-in, memory-first, benchmark-gated

## What changed from V2

V2 made the exact graph-mix route much more aggressive by avoiding explicit
`mixed_weights` in forward / `dX` and by adding an atomic fused `dW/dCoeff`
backward.  The remaining performance risk is that the V2 atomic reduce launches
one program per `(graph, expert, row_chunk, output_tile, input_tile)`.  That is
memory-light, but every expert program reloads the same `x` and `grad_y` tile.

V3 keeps the same mathematical route but changes the reduce kernel to an
**expert-loop atomic reduce**:

```text
V2 reduce grid:
  (graph, expert, row_chunk, out_tile, in_tile)

V3 reduce grid:
  (graph, row_chunk, out_tile, in_tile)
  for expert in experts:
      accumulate dW_expert and dCoeff_graph_expert
```

The expected wins are:

1. `x` and `grad_y` are loaded once per tile instead of once per expert.
2. program count drops by roughly `num_experts` for the reduce side.
3. `shared_weight` and `shared_bias` atomics happen once per tile instead of
   relying on an `expert == 0` mask.
4. V3 still does not materialize `mixed_weights` or `grad_mixed_w` on the
   aggressive route.

The expected risks are:

1. More expert work inside a single Triton program can increase register
   pressure.  V3 defaults `REDUCE_BLOCK_N=16` instead of V2's larger tile.
2. Atomic accumulation is still non-deterministic at the last few ulps.
3. For very small expert counts or tiny graphs, V2's simpler atomic route may be
   faster.

## Files

```text
dptb/nn/so2_triton_exact_gp_v3.py
dptb/tests/test_so2_triton_exact_gp_v3.py
tools/apply_triton_gp_v3_overlay.py
docs/advanced/2026-04-24-triton-graph-persistent-v3.md
README_TRITON_GP_V3.md
VALIDATION_TRITON_GP_V3.md
```

## Apply and integrate

```bash
git apply triton_gp_v3_overlay_additive.patch
python3 tools/apply_triton_gp_v3_overlay.py
```

The overlay script is idempotent.  It inserts V3 guarded imports and dispatches
V3 before V2, so V3 wins when both env flags are enabled.

## Runtime switches

```bash
export DPTB_TRITON_EXACT_GP_V3=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V3=1
export DPTB_TRITON_EXACT_GP_V3_BWD=expert_loop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Safe fallbacks:

```bash
# Reuse V2's atomic reduce while keeping the V3 integration hook.
export DPTB_TRITON_EXACT_GP_V3_BWD=v2_atomic

# Use exact Torch dW/dCoeff backward for correctness A/B.
export DPTB_TRITON_EXACT_GP_V3_BWD=torch
```

Fail fast on a GPU node when V3 cannot actually launch:

```bash
export DPTB_TRITON_EXACT_GP_V3_REQUIRE=1
```

## Tile knobs

```bash
export DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_M=64
export DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_N=16
export DPTB_TRITON_EXACT_GP_V3_REDUCE_BLOCK_K=32
```

V3 falls back to the V2 env names when V3 names are unset, so existing tuning
scripts can be reused.

## Validation

CPU/static validation:

```bash
python3 -m py_compile \
  dptb/nn/so2_triton_exact_gp_v3.py \
  dptb/tests/test_so2_triton_exact_gp_v3.py \
  tools/apply_triton_gp_v3_overlay.py

PYTHONPATH=. python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v3.py -q
```

CUDA smoke validation:

```bash
export DPTB_TRITON_EXACT_GP_V3=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V3=1
export DPTB_TRITON_EXACT_GP_V3_BWD=expert_loop
export DPTB_TRITON_EXACT_GP_V3_REQUIRE=1
PYTHONPATH=. python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v3.py -q
```

Training A/B:

```bash
# Production baseline
unset DPTB_TRITON_EXACT_GP_V2 DPTB_TRITON_COMPLEX_EXACT_GP_V2
unset DPTB_TRITON_EXACT_GP_V3 DPTB_TRITON_COMPLEX_EXACT_GP_V3
python -m dptb ... 2>&1 | tee logs/baseline_cueq.log

# V2 branch route
export DPTB_TRITON_EXACT_GP_V2=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V2=1
export DPTB_TRITON_EXACT_GP_V2_BWD=atomic
python -m dptb ... 2>&1 | tee logs/gp_v2_atomic.log

# V3 expert-loop route
unset DPTB_TRITON_EXACT_GP_V2 DPTB_TRITON_COMPLEX_EXACT_GP_V2
export DPTB_TRITON_EXACT_GP_V3=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V3=1
export DPTB_TRITON_EXACT_GP_V3_BWD=expert_loop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m dptb ... 2>&1 | tee logs/gp_v3_expert_loop.log
```

Gate every run on:

```text
sec/iter
samples/s
peak_allocated_gib
peak_reserved_gib
first-epoch loss parity
forward / x_grad / coeff_grad / weight_grad parity on a small deterministic case
```

## Review notes for the current branch

1. The latest branch already carries V2 (`d039337`), including a direct env-gated
   hook in `so2_triton_grouped_linear_ops.py`.
2. V2 is a good correctness/prototyping step, but its reduce grid repeats the
   same activation and output-gradient loads across experts.
3. The previous exact graph-persistent PR/V2 route still needs benchmark gating;
   it should not become a default production backend until it beats or nearly
   matches the cuEq path on `sec/iter` while reducing both allocated and reserved
   memory.
4. Keep allocator and snapshot diagnostics in the benchmark script because the
   target problem is not just allocated memory but also reserved-memory pressure
   and fragmentation.
