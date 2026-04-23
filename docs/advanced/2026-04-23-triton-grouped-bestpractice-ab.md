# Triton Grouped SO2 Route A/B

Date: 2026-04-23

## Scope

Reviewed `triton_grouped_bestpractice_patch_package/triton_grouped_bestpractice_patch`
and integrated the second-round route as an opt-in backend:

```text
so2_fusion_mode=streamed_m_major_triton_grouped
```

The route keeps the current best middle path unchanged:

```text
so2_wigner_apply_mode=compact_blocks
mole_linear_mode=cueq_indexed_linear
so2_m_linear_mode=standard
mole_full_expert_fast_path=true
top_k=24
num_experts=24
num_shared_experts=0
```

Compared with the previous `streamed_m_major_triton_fused` prototype, this route
uses one grouped Triton launch per `m` for active `l` groups rather than one launch
per `(m, l)` bridge op. The implementation is fp32-only for the Triton runtime.

## Correctness

Local Windows validation:

```text
python -m py_compile ...: pass
pytest selected SO2 tests: 66 skipped locally because torch/e3nn are unavailable
git diff --check: pass, CRLF warnings only
```

natlan CUDA validation:

```text
pytest dptb/tests/test_so2_triton_grouped_ops.py dptb/tests/test_so2_streamed_lmax_bounds.py -q
65 passed
```

Additional CUDA fp32 JIT smoke:

```text
grouped_pack_pair max error: 0.0
grouped_scatter_pair max error: 4.768e-7
```

Full `SO2_Linear` CUDA fp32 staged-vs-grouped smoke:

```text
forward max error: 5.960e-8
x grad max error: 5.821e-11
R grad max error: 5.384e-10
latents grad max error: 1.164e-10
```

## Production-Like A/B

Host: natlan, 2 x NVIDIA L40S

Dataset:

```text
/home/mingkang_nt/data/0422_test
```

Runner output:

```text
/home/mingkang_nt/codex/0422_tests/pr9_triton_fused/runs_bs32_bs48_grouped/bs32_bs48_triton_grouped_ab_results.json
```

### Results

| batch size | route | result | wall time | peak allocated | peak reserved |
| ---: | --- | --- | ---: | ---: | ---: |
| 32 | `streamed_m_major_cueq` | pass | 51.248 s | 29,278.1 MB | 42,192.0 MB |
| 32 | `streamed_m_major_triton_grouped` | pass | 296.949 s | 32,653.8 MB | 41,612.0 MB |
| 32 | `streamed_m_major_triton_grouped` warm cache | pass | 86.311 s | 32,653.8 MB | 41,612.0 MB |
| 48 | `streamed_m_major_cueq` | pass | 53.563 s | 42,311.2 MB | 44,850.0 MB |
| 48 | `streamed_m_major_triton_grouped` | OOM | failed | unavailable | unavailable |

The bs48 OOM happened with PyTorch already using 43.47 GiB on GPU 0 and only
23.94 MiB free, while trying to allocate another 82 MiB.

## Conclusion

`streamed_m_major_triton_grouped` is mathematically correct and its CUDA kernels
compile, but it is not a production improvement on this dataset:

- bs32 warm-cache runtime is 86.311 s vs 51.248 s for the current best cuEq route.
- bs32 peak allocated increases by 3,375.7 MB.
- bs48 passes on the current best cuEq route but OOMs on grouped Triton.

The route should remain experimental and off by default. The current recommended
production configuration remains:

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

The grouped route mainly reduces launch count for the outer pack/scatter bridge,
but it reintroduces larger grouped intermediate buffers and still leaves the real
heavy middle path unchanged. For the next optimization step, focus should move
inside `SO2_m_Linear`/MoLE or use a deeper fused kernel that avoids materializing
the grouped `y` buffers.
