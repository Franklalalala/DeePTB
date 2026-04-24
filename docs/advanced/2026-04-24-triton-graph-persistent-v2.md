# Triton graph-persistent exact MoE V2

Date: 2026-04-24
Branch target: `0422` / PR #14 follow-up
Status: experimental, opt-in, memory-first, benchmark-gated

## Why V2 exists

PR #14 is a good first step: it moves the exact graph-mix MoE forward and `dX`
onto a graph-persistent Triton path behind `DPTB_TRITON_EXACT_USE_GRAPH_PERSISTENT=1`.
However, the backward weight/coefficient side still forms per-graph mixed gradients and
then maps them back to expert gradients. That leaves the largest remaining activation
and reduce-side memory pressure untouched.

V2 makes the route more aggressive:

1. Real exact MoE forward avoids materialising `mixed_weights`.
2. Real exact MoE `dX` avoids materialising `mixed_weights`.
3. Real exact MoE `dW`, `dCoeff`, optional `dBias`, `dSharedW`, and `dSharedBias`
   are accumulated by a fused Triton atomic reduce; no `grad_mixed_w` tensor is
   returned to Python in the atomic path.
4. Complex exact MoE forward, `dX`, `dW`, `dCoeff`, and optional `dSharedW` are also
   implemented directly for the SO2_m complex layout `[real_weights, imag_weights]`.
5. All kernels are hidden behind environment flags and include an exact Torch fallback.

## Public switches

```bash
# Real-valued exact MoE V2
export DPTB_TRITON_EXACT_GP_V2=1

# Complex exact MoE V2. If unset, it inherits DPTB_TRITON_EXACT_GP_V2.
export DPTB_TRITON_COMPLEX_EXACT_GP_V2=1

# Backward reduce mode. atomic is the aggressive route; torch is safer for A/B.
export DPTB_TRITON_EXACT_GP_V2_BWD=atomic
# export DPTB_TRITON_EXACT_GP_V2_BWD=torch

# Fail fast if V2 is requested but CUDA/Triton/fp32 conditions are not met.
export DPTB_TRITON_EXACT_GP_V2_REQUIRE=1

# Batch-size-48 allocator guard recommended by previous A/B.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Optional tile knobs:

```bash
export DPTB_TRITON_EXACT_GP_V2_BLOCK_M=128
export DPTB_TRITON_EXACT_GP_V2_BLOCK_N=64
export DPTB_TRITON_EXACT_GP_V2_BLOCK_K=32
export DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_M=64
export DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_N=32
export DPTB_TRITON_EXACT_GP_V2_REDUCE_BLOCK_K=32
export DPTB_TRITON_EXACT_GP_V2_PERSISTENT_FACTOR=2
```

## Integration

The new module is additive:

```text
dptb/nn/so2_triton_exact_gp_v2.py
dptb/tests/test_so2_triton_exact_gp_v2.py
docs/advanced/2026-04-24-triton-graph-persistent-v2.md
tools/apply_triton_gp_v2_overlay.py
```

Run the overlay script from the DeePTB repository root to add the guarded imports and
route hooks into `dptb/nn/so2_triton_grouped_linear_ops.py`:

```bash
python3 tools/apply_triton_gp_v2_overlay.py
```

The script is idempotent. It inserts:

```python
from .so2_triton_exact_gp_v2 import exact_moe_linear_v2, complex_exact_moe_linear_v2
```

behind a broad import guard and dispatches from:

```python
grouped_exact_moe_linear(...)
grouped_complex_exact_moe_linear(...)
```

only when the V2 env flags are enabled.

## Validation commands

Basic static validation:

```bash
python3 -m py_compile dptb/nn/so2_triton_exact_gp_v2.py
python3 -m py_compile dptb/tests/test_so2_triton_exact_gp_v2.py
python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v2.py -q
```

CUDA smoke validation on a GPU node:

```bash
export DPTB_TRITON_EXACT_GP_V2=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V2=1
export DPTB_TRITON_EXACT_GP_V2_BWD=atomic
export DPTB_TRITON_EXACT_GP_V2_REQUIRE=1
python3 -m pytest dptb/tests/test_so2_triton_exact_gp_v2.py -q
```

Recommended training A/B:

```bash
# Baseline production route
unset DPTB_TRITON_EXACT_GP_V2
unset DPTB_TRITON_COMPLEX_EXACT_GP_V2
python -m dptb ... 2>&1 | tee logs/baseline_cueq.log

# PR #14 route, forward/dX only
export DPTB_TRITON_EXACT_USE_GRAPH_PERSISTENT=1
python -m dptb ... 2>&1 | tee logs/pr14_gp.log

# V2 aggressive route, direct forward/dX/dW/dCoeff
unset DPTB_TRITON_EXACT_USE_GRAPH_PERSISTENT
export DPTB_TRITON_EXACT_GP_V2=1
export DPTB_TRITON_COMPLEX_EXACT_GP_V2=1
export DPTB_TRITON_EXACT_GP_V2_BWD=atomic
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m dptb ... 2>&1 | tee logs/gp_v2_atomic.log

# V2 forward/dX + safe torch dW/dCoeff fallback
export DPTB_TRITON_EXACT_GP_V2_BWD=torch
python -m dptb ... 2>&1 | tee logs/gp_v2_torch_bwd.log
```

Gate every run on:

```text
sec/iter
samples/s
peak_allocated_gib
peak_reserved_gib
first-epoch convergence parity
forward / x_grad / coeff_grad / weight_grad parity on a small deterministic case
```

## Risks and expected failure modes

1. The atomic `dW/dCoeff` route is intentionally non-deterministic at the last few ulps.
   Treat it as speed/memory path, not bitwise path.
2. The Triton path currently accepts CUDA float32 tensors. Other dtypes fall back unless
   `DPTB_TRITON_EXACT_GP_V2_REQUIRE=1` is set.
3. The atomic reduce may be slower than PR #14 for tiny graphs or very small expert
   counts. Use `DPTB_TRITON_EXACT_GP_V2_BWD=torch` as a fallback.
4. If occupancy collapses, lower `BLOCK_M` or `BLOCK_N`; if atomics dominate, increase
   `REDUCE_BLOCK_M` cautiously.
5. Use PyTorch CUDA memory snapshots for unexplained reserved-memory jumps or OOM.
