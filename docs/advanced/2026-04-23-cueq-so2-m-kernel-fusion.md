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

Result before final hardening:

```text
53 passed, 10 warnings in 5.67s
```

Final hardening result:

```text
58 passed, 10 warnings in 7.60s
```

The final hardening pass intentionally did not add fp16/bf16 promotion or AMP handling. It only added:

- per device/dtype/shape cuEq scalar weight-order cache;
- cached expanded graph indices for rank > 2 `MOLELinear` inputs;
- active-l-only Wigner block selection in the grouped SO2 route;
- single-segment no-`cat()` fast paths in grouped SO2 packing.

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

### Hybrid m0 / m>0 backend split

With the new `mole_linear_m0_mode` override, the scalar m=0 path and m>0 paths can use different MoLE backends.

Staged SO2 benchmark, fp32, `B=32`, `num_experts=24`:

| E | config | mean ms | peak GB |
|---:|---|---:|---:|
| 4096 | all split | 50.99 | 0.430 |
| 4096 | all cueq | 38.18 | 0.490 |
| 4096 | m0 split, m>0 cueq | 52.15 | 0.479 |
| 4096 | m0 cueq, m>0 split | 60.82 | 0.442 |
| 8192 | all split | 70.66 | 0.778 |
| 8192 | all cueq | 54.92 | 0.860 |
| 8192 | m0 split, m>0 cueq | 60.45 | 0.847 |
| 8192 | m0 cueq, m>0 split | 68.32 | 0.792 |
| 16384 | all split | 108.70 | 1.441 |
| 16384 | all cueq | 101.76 | 1.568 |
| 16384 | m0 split, m>0 cueq | 104.85 | 1.549 |
| 16384 | m0 cueq, m>0 split | 106.30 | 1.461 |

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

If memory is tighter and you want a milder cueq setting, the new hybrid knob is:

```text
DPTB_MOLE_LINEAR_MODE=split_loop
DPTB_MOLE_LINEAR_M0_MODE=cueq_indexed_linear
```

At `E=16384`, this hybrid staged setting is about 2.2% faster than all-split, while adding only about 0.02 GB peak allocated memory in the module benchmark.

## Final Hardening A/B

Setup: natlan CUDA, fp32, `B=32`, `num_experts=24`, `num_shared_experts=0`, irreps `32x0e + ... + 32x6e`, `radial_emb=False`, forward+backward, 3 warmups, 10 measured iterations. Values below are steady-state after cuEq graph-index and weight-order caches are warm.

| E | config | mean ms | peak allocated GB | peak reserved GB |
|---:|---|---:|---:|---:|
| 4096 | baseline staged split | 49.59 | 0.372 | 0.443 |
| 4096 | staged all cuEq | 37.75 | 0.386 | 0.449 |
| 4096 | streamed all cuEq | 52.90 | 0.373 | 0.465 |
| 4096 | m0 cuEq, m>0 split | 54.88 | 0.370 | 0.455 |
| 8192 | baseline staged split | 65.99 | 0.656 | 0.789 |
| 8192 | staged all cuEq | 51.54 | 0.677 | 0.779 |
| 8192 | streamed all cuEq | 59.37 | 0.663 | 0.791 |
| 8192 | m0 cuEq, m>0 split | 63.22 | 0.656 | 0.809 |
| 16384 | baseline staged split | 102.81 | 1.202 | 1.451 |
| 16384 | staged all cuEq | 96.53 | 1.232 | 1.477 |
| 16384 | streamed all cuEq | 82.92 | 1.232 | 1.432 |
| 16384 | m0 cuEq, m>0 split | 101.34 | 1.202 | 1.461 |

Final readout:

- `staged all cuEq` is the best default speed setting for smaller and medium edge counts: about 23.9% faster at `E=4096`, 21.9% faster at `E=8192`, and 6.1% faster at `E=16384`, with about 0.01-0.03 GB extra peak allocated memory.
- `streamed all cuEq` only becomes attractive at larger edge counts. At `E=16384`, it is about 19.3% faster than baseline while using about 0.03 GB more peak allocated memory and slightly less peak reserved memory in this module benchmark.
- `m0 cuEq, m>0 split` is memory-neutral but not worth making the speed default. It is mainly a conservative fallback knob when keeping allocated memory nearly identical matters more than throughput.

## Next Step

cuEq has reached the useful boundary for this SO2_m complex-linear subproblem. A true single-kernel route for:

```text
rotate-in pack -> SO2_m complex indexed linear -> rotate-out accumulate
```

needs a custom Triton/CUDA kernel with strict correctness tests against `compact_blocks + cueq_indexed_linear`.
