# cuBLAS Grouped MoE Notes, 2026-05-21

## Scope

This branch adds a CUDA/C++ cuBLAS grouped GEMM backend for MoE linear dispatch and uses it from both:

- `lem_moe_v3_h0`: global/formula-routed MoE.
- `lem_moe_v3_edge_h0`: edge-wise MoE with unique bond-type routing.

The branch keeps the old `lem_moe_v3_h0` path registered and adds the edge-wise embedding separately. The cuBLAS path is selected with:

```json
"mole_linear_mode": "cublas_grouped"
```

Top-k routing now carries `topk_indices/topk_values` through `MOLEGlobals`, so sparse routing mixes only selected experts instead of reconstructing dense all-expert coefficients for parameter mixing.

## External References

- NVIDIA cuBLAS documentation for `cublasGemmGroupedBatchedEx`: it accepts per-group arrays for `m/n/k/lda/ldb/ldc`, keeps pointer arrays on device, and supports FP32/TF32 through `CUBLAS_COMPUTE_32F` or `CUBLAS_COMPUTE_32F_FAST_TF32`.
  https://docs.nvidia.com/cuda/cublas/index.html
- NVIDIA grouped GEMM blog: grouped GEMM is intended for one launch over different matrix sizes, transpositions, and scales, with MoE listed as a target use case.
  https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- CUTLASS grouped scheduler notes: grouped kernels schedule tiles across a list of GEMM problems, which explains why problem-size distribution and metadata overhead matter.
  https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html

## Liyue Short Bench

Environment:

- Host: Liyue, 1x NVIDIA L40S for the run.
- Conda: `/home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424`.
- Data: `/home/mingkang_nt/codex/0520_edge_moe_p0_20260521/feature_bs32_subset_first16/train`.
- Batch size: 32.
- Short test: one epoch, 25 iterations, each run under a 180 s timeout.
- FP32 mode: `DPTB_CUBLAS_GROUPED_FAST_TF32=0`; no TF32 fast math was enabled.
- Baseline for ratios: `global_all_cueq_baseline`, i.e. `lem_moe_v3_h0`, `top_k=24`, `mole_linear_mode=cueq_indexed_linear`.

| Case | Method | top_k | Dispatch | Wall s | Ratio vs baseline |
| --- | --- | ---: | --- | ---: | ---: |
| global_all_cueq_baseline | `lem_moe_v3_h0` | 24 | `cueq_indexed_linear` | 58.334 | 1.000 |
| global_all_cublas | `lem_moe_v3_h0` | 24 | `cublas_grouped` | 54.750 | 1.065 |
| global_top2_cublas | `lem_moe_v3_h0` | 2 | `cublas_grouped` | 55.235 | 1.056 |
| edge_all_cublas | `lem_moe_v3_edge_h0` | 24 | `cublas_grouped` | 57.403 | 1.016 |
| edge_top2_cublas | `lem_moe_v3_edge_h0` | 2 | `cublas_grouped` | 57.089 | 1.022 |

Module-level `MOLELinear` fwd+bwd microbench on the same host:

| Case | Reference dispatch | Reference ms | cuBLAS ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| global_all | `cueq_indexed_linear` | 5.783 | 3.153 | 1.83 |
| edge_all | `cueq_indexed_linear` | 5.879 | 4.295 | 1.37 |
| global_top2 | `cueq_indexed_linear` | 6.999 | 3.968 | 1.76 |
| edge_top2 | `cueq_indexed_linear` | 5.435 | 3.924 | 1.39 |

The module-level improvement is larger than end-to-end improvement because SO2 rotation, Wigner block handling, data loading, loss, and optimizer work remain outside this GEMM replacement.

## SO2 m Fusion Trial

`DPTB_SO2_FUSE_M_CUBLAS=1` enables an experimental SO2 path that fuses all `m=1..m_max` MoE raw linear calls in one multi-problem cuBLAS grouped GEMM call. `m=0` stays separate because it has a different scalar path and bias.

Short-test results:

| Case | Wall s | Notes |
| --- | ---: | --- |
| global_all_cublas, `DPTB_SO2_FUSE_M_CUBLAS=0` | 55.610 | Same code, m fusion disabled |
| global_all_cublas, `DPTB_SO2_FUSE_M_CUBLAS=1` | 54.596 | About 1.9% faster in this short run |
| edge_all_cublas, `DPTB_SO2_FUSE_M_CUBLAS=1` | 57.425 | Same as non-fused within noise |
| edge_top2_cublas, `DPTB_SO2_FUSE_M_CUBLAS=1` | 57.816 | Slower than non-fused 57.089 |

Conclusion: keep SO2 `m` fusion opt-in. The cuBLAS API supports this mixed-shape grouping, but the current model has only 6 `m>0` GEMM problems per SO2 block and all have different shapes, so metadata and scheduling overhead mostly cancel the launch reduction. The production default stays disabled.

## Artifacts

Liyue run root:

```text
/home/mingkang_nt/codex/0521_cublas_grouped_moe_e2e_20260521
```

Important files there:

```text
configs/*.json
runs/*/train.log
status.tsv
status_edge_retry.tsv
status_mfuse.tsv
status_mfuse_retry.tsv
status_mfuse_edge_retry.tsv
mole_linear_cublas_bench.txt
so2_m_shape_counts.tsv
```
