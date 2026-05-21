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

## SO2/MoE Fused P0 Prototype

`streamed_m_major_fused_p0` is an opt-in, trainable prototype added after the cuBLAS grouped GEMM branch. It keeps `m=0` on the existing path, but for `m>0` fuses these steps into one CUDA kernel per `m`:

1. load edge features and apply the Wigner input pair projection;
2. apply the existing mixed MoE route weight for that edge;
3. finish the complex SO2 pair output;
4. apply the Wigner output projection and accumulate into the full output buffer.

Safety gates:

- default `so2_fusion_mode` remains `streamed_m_major_cueq`;
- P0 requires CUDA fp32;
- Wigner/R are treated as constants. This is valid for Hamiltonian-only training here, but not for force or coordinate-gradient training;
- non-MoE `InterpolationBlock` SO2 layers fall back to the existing grouped path;
- CPU, non-fp32, or unsupported shapes fall back rather than changing production behavior.

Clean Liyue worktree:

```text
/home/mingkang_nt/codex/0521_fused_so2_moe_p0_clean_20260521/DeePTB
branch: 0521-fused-so2-moe-p0
base: 82cf1a75d726e7c7e72f807661fb51254c433c8f
env: /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424
torch: 2.8.0+cu128
GPU: NVIDIA L40S
TF32: disabled via DPTB_CUBLAS_GROUPED_FAST_TF32=0 and torch backend flags
```

Validation:

```text
pytest dptb/tests/test_so2_moe_fused_p0.py -q
3 passed, 1 warning in 38.49s

pytest dptb/tests/test_so2_moe_fused_p0.py \
  dptb/tests/test_mole_linear_indexed_ref.py::test_mole_linear_cublas_grouped_smoke_if_available \
  dptb/tests/test_mole_linear_indexed_ref.py::test_cublas_grouped_multi_smoke_if_available -q
4 passed, 1 warning in 3.69s
```

Smoke microbench, forward-only, dense Wigner, random SO2/MoE module:

| Case | N | Routes | top_k | Reference ms | Fused P0 ms | Speedup | max_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top2 smoke | 1024 | 32 | 2 | 3.726 | 0.721 | 5.17x | 4.77e-7 |
| top2 smoke | 4096 | 32 | 2 | 2.119 | 0.789 | 2.68x | 4.77e-7 |
| all-expert smoke | 4096 | 32 | 24 | 2.101 | 0.783 | 2.68x | 3.58e-7 |
| top2 smoke | 16384 | 32 | 2 | 3.494 | 2.664 | 1.31x | 4.77e-7 |

This P0 validates that the Wigner prologue and epilogue can be fused around the MoE linear operation. It is not yet the final production CUTLASS/CuTe implementation because the fused SO2 kernel mainloop is still a simple CUDA SIMT dot loop. Compact Wigner is consumed without materializing dense block-diagonal Wigner, but the per-l blocks are currently packed into a flat CUDA tensor before launch.

Trainable compact-Wigner update:

| Case | Scope | Route | Result |
| --- | --- | --- | ---: |
| `bench_so2_moe_fused_train.py --n 4096 --top-k 2` | module fwd+bwd | compact top2 | 17.994 ms -> 12.408 ms, 1.45x |
| `bench_so2_moe_fused_train.py --n 16384 --top-k 2` | module fwd+bwd | compact top2 | 17.595 ms -> 12.382 ms, 1.42x |

Production training smoke on the clean Liyue worktree, bs=32, FP32, TF32 off, precompiled extension:

| Case | Baseline comparator | rc | Iterations before 180s timeout | steady-state s/iter | Ratio vs comparator |
| --- | --- | ---: | ---: | ---: | ---: |
| `global_all_cueq_baseline` | baseline | 0 | 25 | 2.07 | 1.00 |
| `global_all_fused_p0` | `global_all_cueq_baseline` | 124 | 17 | 9.00 | 0.23 |
| `edge_top2_cublas` | edge comparator | 0 | 25 | 2.13 | 1.00 |
| `edge_top2_fused_p0` | `edge_top2_cublas` | 124 | 17 | 8.86 | 0.24 |

The fused P0 route did activate with compact Wigner (`compact_or_dense_wigner_mode=2, m_max=6`), and the final interpolation SO2 layer fell back as intended. The trainable P0 is therefore correct enough for experimentation, but not production-fast: the scalar/atomic backward overwhelms the forward-side fusion benefit at DeePTB production shapes. Keep it opt-in.

CUTLASS/CuTe attempt:

- CUTLASS was cloned on Liyue at `/home/mingkang_nt/codex/third_party/cutlass`, commit `546c3efa899ed1793d178a26fe764d63f93b49ea`.
- The extension can be built with `DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT=/home/mingkang_nt/codex/third_party/cutlass`.
- A CuTe dot-mainloop probe compiled and matched PyTorch with max abs `1.91e-6`.
- The production SO2 routed mainloop still needs a real grouped MMA tile scheduler plus a non-atomic backward; the current P0 deliberately does not replace the stable `cublas_grouped` fallback.

Follow-up after the negative training result:

- The original trainable P0 backward is slow because it launches one CUDA block per `(edge, out_channel)` and uses scalar `atomicAdd` into `grad_weight`, `grad_x`, and radial gradients. This looked acceptable in small module smoke tests but becomes the dominant cost at DeePTB production edge counts and SO2 channel sizes.
- A safer `DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE=cublas_segmented` mode was added. It reconstructs the pair tensors in PyTorch, then uses the existing cuBLAS grouped GEMM backend for raw-linear `grad_x_pair` and `grad_weight`. This removes the catastrophic atomics and finishes the 25-iteration production smoke, but it is still slower than the stable baseline because pair pack/scatter and per-m segmented launches remain outside one fused CUDA kernel.
- A `DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE=cutlass_segmented` backend was also added with a new `dptb.nn.cutlass_grouped_gemm` extension. It uses CUTLASS grouped GEMM for the same raw-linear backward subproblem. CUDA correctness passed on Liyue (`5 passed, 1 warning`) with `DPTB_CUTLASS_ROOT=/home/mingkang_nt/codex/third_party/cutlass`.
- CUTLASS official grouped kernels are persistent grouped kernels with device-side or host-precomputed scheduling. cuBLAS grouped GEMM is a useful drop-in for different shapes and transposes, but it cannot express the SO2 Wigner output rotation/scatter epilogue. On L40S/Ada SM89, the practical CUTLASS path is either a CUTLASS 2.x grouped kernel plus a hand-written threadblock epilogue, or a custom CuTe kernel. CUTLASS 3.x EVT-style custom epilogue is mainly a Hopper/SM90 TMA warp-specialized path, so it was not used as a production path on this L40S test.

Trainable fused P0 follow-up smoke, bs=32, FP32, TF32 off:

| Case | Backend | rc | Iterations | wall s | steady-state s/iter | Comparator | Ratio vs comparator |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `global_all_cueq_baseline` | stable baseline | 0 | 25 | 58.334 | 2.087 | baseline | 1.00 |
| `global_all_cublas` | stable cuBLAS grouped | 0 | 25 | 54.750 | 2.043 | baseline | 1.02 |
| `global_all_fused_p0` | atomic backward | 124 | 17 | timeout | 8.902 | baseline | 0.23 |
| `global_all_fused_p0_cublas_segmented` | cuBLAS segmented backward | 0 | 25 | 97.969 | 3.771 | baseline | 0.55 |
| `global_all_fused_p0_cutlass_segmented` | CUTLASS segmented backward | 0 | 25 | 97.097 | 3.765 | baseline | 0.55 |
| `edge_top2_cublas` | stable edge comparator | 0 | 25 | 57.089 | 2.134 | edge comparator | 1.00 |
| `edge_top2_fused_p0` | atomic backward | 124 | 17 | timeout | 8.855 | edge comparator | 0.24 |
| `edge_top2_fused_p0_cublas_segmented` | cuBLAS segmented backward | 0 | 25 | 103.212 | 3.987 | edge comparator | 0.54 |
| `edge_top2_fused_p0_cutlass_segmented` | CUTLASS segmented backward | 0 | 25 | 108.378 | 4.166 | edge comparator | 0.51 |

Conclusion: CUTLASS grouped GEMM is viable and correct as an opt-in raw-linear backend, but simply swapping cuBLAS grouped for CUTLASS grouped does not solve the production training bottleneck. The remaining performance problem is the still-fragmented SO2 backward dataflow: pair packing, output-rotation gradient, input-rotation scatter, and radial gradient are still separate PyTorch/CUDA work around each per-m GEMM. A production-speed fused SO2 training path needs a single custom CUDA/CuTe kernel family that owns both forward and backward prologue/epilogue, not only the raw GEMM mainloop.

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

Fused P0 smoke root:

```text
/home/mingkang_nt/codex/0521_fused_so2_moe_p0_clean_20260521/smoke
```
