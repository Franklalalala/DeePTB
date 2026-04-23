# cuEq SO2_m Kernel Fusion Report

Date: 2026-04-23
Branch: `0422-cueq-complex-linear-fusion`

## Scope

This round pushed SO2_m fusion as far as cuEquivariance currently allows without Triton/custom CUDA. The target boundary was:

```text
MOLELinear -> real/imag SO2 postprocess
```

The Wigner path remains `compact_blocks` by default.

## Implemented Experimental Modes

- `cueq_complex_indexed_linear`: expands the real/imag operation into a 2x2 block matrix and calls `cuequivariance_torch.Linear(method="indexed_linear")` once.
- `cueq_segmented_complex_indexed_linear`: uses `cuequivariance_torch.SegmentedPolynomial(method="indexed_linear")` with two supported `uv,wu,wv` operations, one for `Wr * x` and one for `Wi * [-imag, real]`.

The segmented mode avoids explicit 2x2 block-weight construction, but cuEq does not turn it into one compact complex-linear kernel.

## cuEq Boundary

The desired one-descriptor complex operation requires coefficient mixing:

```text
uv,iu,jv+ij
```

Natlan cuEq rejects that for `indexed_linear`:

```text
NotImplementedError: Indexed_linear does not support the operation uv,iu,jv+ij.
```

So current cuEq can either use a block matrix or split the complex operation into two indexed-linear operations. It cannot express this SO2_m operation as one compact indexed-linear kernel.

Official references:

- https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SegmentedPolynomial.html
- https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.Linear.html
- https://docs.nvidia.com/cuda/cuequivariance/tutorials/stp.html

## Verification

Natlan CUDA env:

```text
conda env: dptb_p2_wigner_cu12_py310
checkout: /home/mingkang_nt/codex/0422_tests/cueq_so2_finish/DeePTB
```

Command:

```text
pytest dptb/tests/test_so2_streamed_lmax_bounds.py \
       dptb/tests/test_mole_linear_indexed_ref.py \
       dptb/tests/test_mole_router_full_expert_fast_path.py -q
```

Result:

```text
50 passed, 10 warnings in 7.06s
```

## Benchmark Summary

Setup: fp32, `B=32`, `num_experts=24`, `num_shared_experts=0`, irreps `32x0..32x6`, forward+backward, 3 warmups, 10 iterations.

| E | mode | mean ms | peak GB |
|---:|---|---:|---:|
| 4096 | `staged_split_loop` | 53.72 | 0.430 |
| 4096 | `staged_cueq_standard` | 39.06 | 0.487 |
| 4096 | `staged_cueq_complex` | 48.62 | 0.532 |
| 4096 | `staged_cueq_segmented` | 54.52 | 0.531 |
| 8192 | `staged_split_loop` | 69.94 | 0.778 |
| 8192 | `staged_cueq_standard` | 55.15 | 0.861 |
| 8192 | `staged_cueq_complex` | 56.10 | 0.905 |
| 8192 | `staged_cueq_segmented` | 61.93 | 0.948 |
| 16384 | `staged_split_loop` | 108.92 | 1.441 |
| 16384 | `streamed_cueq_standard` | 90.83 | 1.551 |
| 16384 | `streamed_cueq_complex` | 90.57 | 1.598 |
| 16384 | `streamed_cueq_segmented` | 95.81 | 1.718 |

## Recommendation

Do not make either complex mode default.

Recommended production-style setting:

```text
DPTB_MOLE_LINEAR_MODE=cueq_indexed_linear
DPTB_SO2_M_LINEAR_MODE=standard
DPTB_SO2_FUSION_MODE=staged
```

For very large edge counts, A/B this setting:

```text
DPTB_MOLE_LINEAR_MODE=cueq_indexed_linear
DPTB_SO2_M_LINEAR_MODE=standard
DPTB_SO2_FUSION_MODE=streamed_m_major_cueq
```

At `E=16384`, `streamed_cueq_standard` is about 16.6% faster than `staged_split_loop`, with about 0.11 GB more peak allocated memory in this module benchmark.

## Next Step

cuEq has reached the useful boundary for this SO2_m complex-linear subproblem. A true single-kernel route for:

```text
rotate-in pack -> SO2_m complex indexed linear -> rotate-out accumulate
```

needs a custom Triton/CUDA kernel with strict correctness tests against `compact_blocks + cueq_indexed_linear`.
