# Triton Grouped MoLE Linear A/B

Date: 2026-04-23

## Scope

Reviewed `pr10_triton_deeper_patch_package/pr10_triton_deeper_patch` and
integrated its useful parts directly, without the overlay monkey-patch:

```text
mole_linear_mode=triton_grouped_linear
```

This route moves `MOLELinear`'s per-graph linear apply into a Triton grouped GEMM
backend. It can be used by both `SO2_Linear.fc_m0` and `SO2_m_Linear.fc`.

The route is experimental and off by default.

## Engineering Notes

Adopted from the patch:

- `dptb/nn/so2_triton_grouped_linear_ops.py`
- `MOLELinear._apply_triton_grouped_linear`
- `SO2_m_Linear._forward_standard` prealloc combine, replacing `narrow -> cat`
- `DPTB_TRITON_LINEAR_DISABLE`
- `DPTB_TRITON_LINEAR_REQUIRE`
- `DPTB_TRITON_LINEAR_PERSISTENT_FACTOR`

Fixes added during integration:

- Restricted runtime Triton path to fp32 only.
- Replaced unsupported Triton `break` in the grouped tile selection loop.
- Set `tl.dot(..., input_precision="ieee")` to remove TF32 correctness drift.
- Avoided the misleading warning when `streamed_m_major_triton_fused` is paired
  with `mole_linear_mode=triton_grouped_linear`.
- Routed `MOLELinear.forward()` through `_mixed_weights_and_bias()` to avoid
  duplicating routed/shared/bias mixing logic.

## Correctness

Local Windows:

```text
python -m py_compile dptb/nn/so2_triton_grouped_linear_ops.py dptb/nn/tensor_product_moe_v3.py dptb/tests/test_so2_triton_grouped_linear_ops.py dptb/utils/argcheck.py
pass

pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q
1 skipped because local torch is unavailable
```

natlan CUDA:

```text
PYTHONPATH=$PWD python -m pytest dptb/tests/test_so2_triton_grouped_linear_ops.py -q
11 passed
```

Full `SO2_Linear` CUDA fp32 smoke, comparing staged split-loop against
`streamed_m_major_triton_fused + triton_grouped_linear`:

```text
forward max error:    1.192e-7
x grad max error:     7.276e-11
R grad max error:     6.985e-10
latents grad error:   1.164e-10
coeff grad max error: 1.281e-9
```

## Production-Like A/B

Host: natlan, 2 x NVIDIA L40S

Dataset:

```text
/home/mingkang_nt/data/0422_test
```

Common settings:

```text
so2_wigner_apply_mode=compact_blocks
so2_m_linear_mode=standard
mole_full_expert_fast_path=true
num_experts=24
top_k=24
num_shared_experts=0
```

Result file:

```text
/home/mingkang_nt/codex/0422_tests/pr10_deeper_linear/runs/pr10_deeper_linear_ab_results.json
```

| batch size | route | mole linear | result | wall time | peak allocated | peak reserved |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 32 | `streamed_m_major_cueq` | `cueq_indexed_linear` | pass | 53.557 s | 29,283.0 MB | 42,196.0 MB |
| 32 | `streamed_m_major_triton_fused` | `cueq_indexed_linear` | pass | 55.919 s | 32,658.1 MB | 41,606.0 MB |
| 32 | `streamed_m_major_triton_fused` | `triton_grouped_linear` | pass | 110.524 s | 32,659.0 MB | 41,582.0 MB |
| 32 | `streamed_m_major_triton_fused` | `triton_grouped_linear`, warm cache | pass | 72.915 s | 32,659.0 MB | 41,582.0 MB |
| 48 | `streamed_m_major_cueq` | `cueq_indexed_linear` | pass | 55.780 s | 42,313.3 MB | 44,850.0 MB |
| 48 | `streamed_m_major_triton_fused` | `cueq_indexed_linear` | OOM | failed | unavailable | unavailable |
| 48 | `streamed_m_major_triton_fused` | `triton_grouped_linear` | OOM | failed | unavailable | unavailable |

bs48 OOM details:

```text
outer Triton + cuEq: PyTorch allocated 43.66 GiB, only 3.94 MiB free, failed on 2 MiB allocation.
deeper Triton linear: PyTorch allocated 43.61 GiB, only 23.94 MiB free, failed on 32 MiB allocation.
```

## Conclusion

The deeper Triton grouped linear path is correct, but not useful for production on
this dataset:

- bs32 warm-cache runtime is 72.915 s vs 53.557 s for the current best cuEq route.
- bs32 peak allocated increases by about 3.38 GB.
- bs48 passes with the current best cuEq route but OOMs with the Triton fused route,
  whether the middle linear is cuEq or Triton grouped linear.

Keep `triton_grouped_linear` as an experimental backend only. Do not make it the
default and do not use it for batch-size expansion.

The current recommended production route remains:

```text
so2_wigner_apply_mode=compact_blocks
so2_fusion_mode=streamed_m_major_cueq
mole_linear_mode=cueq_indexed_linear
so2_m_linear_mode=standard
mole_full_expert_fast_path=true
top_k=24
num_experts=24
num_shared_experts=0
```

The main technical reason is that this route replaces only the grouped linear
apply. It still materializes mixed weights, saves `flat_x`/`mixed_weights` for
backward, and computes `grad_w/grad_b` with per-group torch matmul. It does not
fuse MoE weight mixing, SO2 real/imag combine, or rotate pack/scatter into one
training kernel.
