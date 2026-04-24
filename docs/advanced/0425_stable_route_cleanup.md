# 0425 Stable Route Cleanup

## Production Route

The stable branch is intentionally narrow:

```text
so2_wigner_apply_mode = compact_blocks
so2_fusion_mode       = streamed_m_major_cueq
mole_linear_mode      = cueq_indexed_linear
onehot_tp_mode        = scalar_fast
E3ElementLinear       = block_view by default
```

This is the route used for production throughput/memory runs. It keeps the
compact Wigner representation, streams SO2 order-major data into cuEq indexed
linear, and keeps the lightweight scalar onehot tensor-product path.

## Removed From Stable Code

### `streamed_m_major_aggressive`

This mode had no independent implementation in the stable branch. It was only
an alias to the reference streamed implementation:

```text
streamed_m_major_aggressive -> _forward_streamed_m_major_ref
```

Keeping it made the mode matrix look larger than it really was and created
unnecessary test/config surface. It is removed from the stable branch. Use
`streamed_m_major_cueq` for production and `streamed_m_major_ref` only as a
correctness reference.

### Triton SO2/MoLE Route Flags

The Triton route remains an experiment branch, not a stable production route.
Stable accepts these legacy keys only for old input compatibility:

```text
so2_m_linear_mode   = null | standard
mole_linear_m0_mode = null | standard
```

Any non-standard Triton value should fail loudly and be run on the Triton
experiment branch instead.

## Retained Fallbacks

### `full_dense`

`full_dense` is retained as a correctness and regression fallback for Wigner
rotation tests. It is not the production memory path.

### `staged`

`staged` is retained as the baseline staged SO2 layout path for parity tests.
It is useful when checking that streamed routes preserve forward and gradient
semantics.

### `streamed_m_major_ref`

`streamed_m_major_ref` is retained as the PyTorch correctness reference for the
streamed SO2 dataflow. It is not intended to beat the cuEq route in production.

### `indexed_ref`

`indexed_ref` is retained as the PyTorch correctness oracle for MoLE indexed
linear. It materializes per-row weights and should not be used for performance.

## Abandoned Routes Documented Elsewhere

The following routes are documented as experiment history and should not be
reintroduced into stable without new benchmark evidence:

- direct cuEq `Rotation` replacement for Wigner apply
- Triton complex grouped linear
- Triton fused expert mixing inside row tiles
- Triton exact graph-mix grouped linear as a production speed path

The common lesson is that correctness alone was not enough: routes that kept
extra layout glue, repeated expert mixing at the wrong granularity, or reduced
memory at a large throughput cost were kept out of the stable production path.
