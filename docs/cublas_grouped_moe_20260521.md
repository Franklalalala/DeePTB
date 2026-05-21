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

Aggressive follow-up, same day:

- Added `DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE=cuda_cublas_segmented` and made it the default for the opt-in fused P0 route. This keeps cuBLAS grouped GEMM for the raw linear backward, but moves backward input pair packing, output-pair gradient projection, and input-gradient scatter from Python/einsum into CUDA kernels.
- Liyue correctness smoke: `pytest dptb/tests/test_so2_moe_fused_p0.py::test_so2_fused_p0_compact_backward_matches_streamed_ref_if_available -q` passed for both `cublas_segmented` and `cuda_cublas_segmented` (`2 passed, 1 warning in 36.64s`).
- Module train smoke, `bench_so2_moe_fused_train.py --n 4096 --top-k 2 --iters 30 --warmup 10`: `streamed_grouped_train` 23.360 ms, fused P0 `cuda_cublas_segmented` 15.002 ms, speedup 1.56x, `x_grad_max_abs=2.27e-12`.
- Module train smoke, `--n 16384 --top-k 2 --iters 20 --warmup 5`: `streamed_grouped_train` 22.045 ms, fused P0 `cuda_cublas_segmented` 15.893 ms, speedup 1.39x, `x_grad_max_abs=6.82e-13`.
- Production smoke, bs=32, FP32, TF32 off, compact Wigner, 25 iterations:

| Case | Backend | wall time | Previous fused wall | Stable comparator | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `global_all_fused_p0_cuda_cublas` | CUDA epilogue + cuBLAS segmented | 90.099 s | 97.969 s | 58.334 s | fused branch +8.0%, still slower than stable |
| `edge_top2_fused_p0_cuda_cublas` | CUDA epilogue + cuBLAS segmented | 94.898 s | 103.212 s | 57.089 s | fused branch +8.1%, still slower than stable |

This confirms that moving the backward prologue/epilogue into CUDA gives a real fused-P0 improvement, but the current architecture still cannot beat the stable `streamed_m_major_cueq`/`cublas_grouped` production path. The likely remaining blockers are per-`m` custom-op launch count, mixed-weight materialization per route, m0 still on the old path, and the interpolation SO2 layer falling back because it uses `InterpolationBlock` rather than `MOLELinear`.

CuTe-tiled forward and input-radial scatter follow-up:

- Added CUTLASS-root/CuTe-indexed forward variants `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=cutlass_tiled{2,3,4,8}`. These keep the current scalar FP32 dot mainloop, but one CUDA block now computes a small tile of output channels and reuses the Wigner-rotated input pair across that tile. This is not a Tensor Core path; it is a dataflow/prologue-epilogue prototype for SM89 with TF32 off.
- Added a backward input epilogue kernel for the radial-on-input case. It fuses `grad_radial = sum(grad_x_pair_eff * x_pair_no_radial)`, radial scaling of `grad_x_pair_eff`, and Wigner input-gradient scatter into one CUDA kernel. It is controlled by `DPTB_SO2_MOE_FUSED_P0_FUSE_INPUT_RADIAL_SCATTER` and defaults on inside the opt-in fused P0 route.
- Liyue correctness smoke with CUTLASS root and compact Wigner: `10 passed, 1 warning in 43.24s`. This covers CUDA pair ops, tile2/3/4/8 forward vs scalar fused P0, and train backward vs streamed reference.

Module train tile search, `bench_so2_moe_fused_train.py --top-k 2`, FP32, TF32 off, `cuda_cublas_segmented` backward:

| N | forward mode | fused P0 train ms | Note |
| ---: | --- | ---: | --- |
| 4096 | scalar | 15.708 | baseline fused P0 |
| 4096 | tile2 | 16.877 | too little channel reuse |
| 4096 | tile3 | 15.141 | best module result |
| 4096 | tile4 | 15.880 | neutral |
| 4096 | tile8 | 15.909 | neutral in module test |
| 16384 | scalar | 15.871 | baseline fused P0 |
| 16384 | tile2 | 15.899 | neutral |
| 16384 | tile3 | 14.894 | best module result |
| 16384 | tile4 | 15.179 | small win |
| 16384 | tile8 | 15.238 | small win |

Production timestamp smoke, bs=32, 25 iterations, FP32, TF32 off, compact Wigner, same wrapper for all fused variants:

| Case | forward/backend | wall s | back-half s/iter | Ratio vs scalar fused | Ratio vs stable comparator |
| --- | --- | ---: | ---: | ---: | ---: |
| `global_all_fused_p0_scalar` | scalar forward + CUDA epilogue + cuBLAS segmented | 101.555 | 3.449 | 1.00 | 0.59 vs `global_all_cublas` |
| `global_all_fused_p0_tiled3` | tile3 forward + CUDA epilogue + cuBLAS segmented | 81.524 | 2.530 | 1.36 | 0.81 vs `global_all_cublas` |
| `global_all_fused_p0_tiled4` | tile4 forward + CUDA epilogue + cuBLAS segmented | 78.729 | 2.391 | 1.44 | 0.85 vs `global_all_cublas` |
| `global_all_fused_p0_tiled8` | tile8 forward + CUDA epilogue + cuBLAS segmented | 75.139 | 2.327 | 1.48 | 0.88 vs `global_all_cublas` |
| `global_all_fused_p0_tiled8_radial_scatter` | tile8 forward + fused input-radial scatter + cuBLAS segmented | 73.050 | 2.273 | 1.52 | 0.90 vs `global_all_cublas` |
| `edge_top2_fused_p0_scalar` | scalar forward + CUDA epilogue + cuBLAS segmented | 113.240 | 3.808 | 1.00 | 0.56 vs `edge_top2_cublas` |
| `edge_top2_fused_p0_tiled3` | tile3 forward + CUDA epilogue + cuBLAS segmented | 89.837 | 2.853 | 1.33 | 0.75 vs `edge_top2_cublas` |
| `edge_top2_fused_p0_tiled4` | tile4 forward + CUDA epilogue + cuBLAS segmented | 84.765 | 2.716 | 1.40 | 0.79 vs `edge_top2_cublas` |
| `edge_top2_fused_p0_tiled8` | tile8 forward + CUDA epilogue + cuBLAS segmented | 82.689 | 2.624 | 1.45 | 0.81 vs `edge_top2_cublas` |
| `edge_top2_fused_p0_tiled8_radial_scatter` | tile8 forward + fused input-radial scatter + cuBLAS segmented | 78.337 | 2.501 | 1.52 | 0.85 vs `edge_top2_cublas` |

This is the first fused P0 result that visibly improves production training, not just module microbench: best global wall time improved from the prior `global_all_fused_p0_cuda_cublas` 90.099 s to 73.050 s, and best edge wall time improved from 94.898 s to 78.337 s. The remaining negative signal is still important: even the best fused P0 remains slower than the stable `cublas_grouped` production comparator because the path still materializes per-route `mixed_weight`, launches per `m`, leaves `m=0` and interpolation SO2 outside the fused backend, and uses segmented cuBLAS for raw backward with intermediate `x_pair_eff`, `grad_raw`, and `grad_x_pair_eff` tensors.

Current recommendation:

- Keep `cutlass_tiled8` and fused input-radial scatter opt-in for the fused P0 branch.
- Do not make fused P0 the production default yet.
- The next meaningful step is not another GEMM wrapper. It is either direct expert/route reads to remove route-level `mixed_weight`, or a larger CuTe kernel family that owns the `m` loop plus backward projection/scatter around the raw GEMM tile.

Hardware/CUTLASS notes:

- Liyue reports `NVIDIA L40S`, compute capability 8.9, driver 535.104.05. This is the primary smoke platform used above.
- Vanda GPU nodes are A40-class; NVIDIA lists A40 as compute capability 8.6. CUTLASS 3.x/CuTe code can target Ampere, but Hopper-only features such as TMA/WGMMA/threadblock clusters are not available there. Prefer conservative SIMT/CUTLASS 2.x-style grouped GEMM or CuTe code that explicitly instantiates `Sm80`/`Sm86` paths.
- pro6000 reports `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, compute capability 12.0, driver 580.126.09, with PyTorch 2.8.0+cu128 available in the checked MoE envs. CUTLASS 3.x/Blackwell paths are a better architectural fit there, but extension builds still need an nvcc/toolkit that supports `sm_120`; nvcc was not on PATH in the probed conda envs.
- For this DeePTB SO2 workload, CUTLASS grouped + custom epilogue is conceptually closer than a standalone grouped GEMM because the useful work is input Wigner prologue, route/expert mainloop, output Wigner epilogue, and gradient projection. The current negative result for raw CUTLASS segmented GEMM is therefore not a rejection of CUTLASS overall; it rejects the narrower "swap only GEMM mainloop" change.

P1 review package comparison:

- The external P1 package under `E:\deeptb\codex\0521_graph_pyg\deeptb_so2_moe_fused_p1_20260521` targets the older `ad0cb5b` P0 state. Its main technical suggestion is to move segmented-backward pair pack, output-pair gradient, and input-gradient scatter into CUDA.
- That suggestion is already absorbed in this branch by the integrated `cuda_cublas_segmented` mode. The integrated implementation reuses the existing fused-P0 extension instead of adding a second pair-op extension, and avoids atomics in input-gradient scatter because the current SO2 pair maps are built from disjoint irrep slices for each `m`.
- The useful part that was still missing was direct helper-level test coverage. Added `test_so2_fused_p0_cuda_pair_ops_match_torch_helpers_if_available`, covering dense and compact Wigner representations for CUDA pair pack, output-pair gradient projection, and input-gradient scatter against the Torch helper path. Liyue result: `2 passed, 1 warning in 5.09s`.
- The separate P1 hook script is not applied: it expects the older `_segmented_pair_backward` marker and would duplicate the already-integrated code path. The P1 scatter kernel also uses atomics for overlap tolerance; that is safer for hypothetical overlapping maps but slower than the current disjoint-map implementation used by DeePTB's SO2 layout.

m0 Wigner fusion follow-up:

- Implemented an opt-in `DPTB_SO2_MOE_FUSED_P0_FUSE_M0=1` path. This moves the `m=0` Wigner input rotation, output rotation, and train backward projection/scatter into the fused P0 CUDA extension, including compact-Wigner input. It also adds CUDA helper kernels for `pack_m0`, `output_m0_grad`, `scatter_m0_grad`, and fused radial-input scatter.
- The default remains off because production smoke is negative. This is deliberately not hidden behind fallback: `DPTB_SO2_MOE_FUSED_P0_STRICT_M0=1` makes the test fail if the m0 fusion path declines.
- Liyue correctness smoke, FP32, TF32 off, compact Wigner where applicable: `pytest dptb/tests/test_so2_moe_fused_p0.py::test_so2_fused_p0_forward_matches_streamed_ref_if_available dptb/tests/test_so2_moe_fused_p0.py::test_so2_fused_p0_compact_backward_matches_streamed_ref_if_available -q` with `DPTB_SO2_MOE_FUSED_P0_FUSE_M0=1` and `DPTB_SO2_MOE_FUSED_P0_STRICT_M0=1` passed: `5 passed, 1 warning in 83.95s`.

Module train smoke, `cutlass_tiled8` forward, `cuda_cublas_segmented` backward:

| m0 fusion | N | streamed grouped train ms | fused P0 train ms | speedup vs streamed | max x-grad diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 4096 | 10.569 | 6.091 | 1.74x | 2.27e-12 |
| off | 16384 | 19.279 | 15.557 | 1.24x | 6.82e-13 |
| on | 4096 | 21.444 | 11.947 | 1.79x | 2.73e-12 |
| on | 16384 | 17.320 | 11.579 | 1.50x | 7.96e-13 |

Production smoke, bs=32, 25 iterations, FP32, TF32 off, `cutlass_tiled8`, fused input-radial scatter, and m0 fusion enabled:

| Case | wall s | back-half s/iter | Previous best fused P0 | Stable comparator |
| --- | ---: | ---: | ---: | ---: |
| `global_all_fused_p0_tiled8_m0_fused` | 178.325 | 3.311 | 73.050 s / 2.273 s/iter | `global_all_cublas` 54.750 s / 2.043 s/iter |
| `edge_top2_fused_p0_tiled8_m0_fused` | 105.025 | 3.558 | 78.337 s / 2.501 s/iter | `edge_top2_cublas` 57.089 s / 2.134 s/iter |

Negative-result analysis:

- The m0 path now does eat DeePTB's Wigner input and output rotations in CUDA, but it also replaces a strong cueq/cuBLAS-indexed scalar linear path with a custom scalar SIMT dot kernel. For m0, the linear work dominates more than the rotation work, so the weaker mainloop loses more than the fused prologue/epilogue saves.
- The m0 training path still needs intermediate `x_m0_eff`, `grad_linear`, and segmented raw-linear backward tensors. It reduces Python/einsum work, but does not yet remove enough launches or memory traffic.
- This is different from the m>0 pair path: m>0 benefited from tiled output-channel reuse and fused radial scatter because its old rotation/pack/scatter overhead was larger relative to the pair linear work.

Current decision:

- Keep m0 fusion as an explicit development switch for correctness and future CuTe epilogue work.
- Do not enable it by default inside fused P0.
- The production-relevant route remains m>0 `cutlass_tiled8` + fused input-radial scatter as the best fused-P0 prototype, while stable production remains `mole_linear_mode="cublas_grouped"` / `streamed_m_major_cueq`.
- The next deep CUTLASS/CuTe attempt should not put a weak SIMT dot in front of m0. It should either keep cueq/cublas for m0 and only fuse rotation around it, or build a larger route/m grouped kernel where Wigner input rotation, indexed linear, Wigner output epilogue, and backward projection/scatter are owned by the same schedule.

cueq-compatible indexed sandwich follow-up:

- Added `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich` and `cueq_sandwich`. These modes keep the existing indexed raw-linear backend instead of replacing it with the fused P0 SIMT dot mainloop. The new boundary is CUDA Wigner input pack -> configured `MOLELinear` indexed backend (`cublas_grouped` or `cueq_indexed_linear`) -> CUDA epilogue.
- Added `scatter_raw_pair_forward_fp32` / `raw_pair_output_grad_fp32`. For the common radial-on-input case this fuses the raw linear output's complex finish and Wigner output rotation into one CUDA epilogue, so the cueq/cublas raw output is no longer finished by PyTorch `narrow/sub/cat` before scatter.
- Liyue correctness smoke: `test_so2_fused_p0_indexed_sandwich_matches_streamed_ref_if_available` passed for both `cublas_grouped` and `cueq_indexed_linear` (`2 passed, 1 warning in 47.37s`). Helper coverage also checks dense/compact Wigner CUDA `scatter_pair_forward` against the Torch helper.

Module train smoke, compact Wigner, FP32, TF32 off:

| Backend | Forward mode | N | streamed train ms | fused train ms | Speedup | max x-grad diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cublas_grouped` | `indexed_sandwich` + raw CUDA epilogue | 4096 | 19.177 | 15.101 | 1.27x | 2.73e-12 |
| `cublas_grouped` | `indexed_sandwich` + raw CUDA epilogue | 16384 | 15.621 | 15.559 | 1.00x | 6.82e-13 |
| `cueq_indexed_linear` | `cueq_sandwich` + raw CUDA epilogue | 4096 | 25.601 | 25.659 | 1.00x | 1.82e-12 |
| `cueq_indexed_linear` | `cueq_sandwich` + raw CUDA epilogue | 16384 | 26.236 | 27.036 | 0.97x | 6.82e-13 |

Interpretation:

- This is a cleaner cueq-compatible experiment than replacing GEMM: it does eat DeePTB Wigner input rotation and output rotation around the indexed raw-linear call, while preserving cueq/cublas as the raw mainloop.
- It still does not beat the best fused P0 tiled SIMT path in production-shaped module tests. The reason is launch and materialization count: `pack_pair`, indexed linear, raw epilogue, and backward pack/scatter are still separate kernels/tensors per `m`.
- The useful next CUTLASS/CuTe direction is therefore not a cueq wrapper. It is a grouped route/m scheduler with a custom epilogue that owns raw output finish + Wigner output rotation, and a matching backward projection/scatter path. The indexed sandwich is kept as an opt-in correctness/dataflow reference for that kernel.

Production smoke for `indexed_sandwich` completed under `/home/mingkang_nt/codex/0521_fused_so2_moe_aggressive_20260521/prod_smoke_indexed_sandwich_raw_epi_20260521`:

| Case | wall s | back-half s/iter | stable comparator wall | stable comparator back-half |
| --- | ---: | ---: | ---: | ---: |
| `global_all_fused_p0_indexed_sandwich_raw_epi` | 107.079 | 1.875 | `global_all_cublas` 54.750 | 2.043 |
| `edge_top2_fused_p0_indexed_sandwich_raw_epi` | 66.450 | 1.985 | `edge_top2_cublas` 57.089 | 2.134 |

The wall time is worse because this smoke includes extension/runtime warmup and still has fragmented launch overhead. The steady-state back-half is nevertheless faster than the stable cublas comparator, which suggests that directly eating Wigner input/output rotation around a strong indexed backend can help once warmup is amortized.

m-loop indexed sandwich follow-up:

- Added `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich_multi` / `cublas_multi_sandwich` / `route_m_sandwich`.
- This path packs Wigner input rotation per `m`, but then runs all m>0 raw linears through one `grouped_gemm_multi` call and applies the raw CUDA epilogue per `m`. It is a minimal trainable route/m schedule prototype: m is now part of the raw-linear grouped schedule rather than three independent indexed calls.
- Liyue correctness smoke: indexed sandwich modes passed for cublas single-m, cublas multi-m, and cueq sandwich (`3 passed, 1 warning in 4.76s`).

Module train smoke, `cublas_grouped`, compact Wigner, FP32, TF32 off:

| Forward mode | N | streamed train ms | fused train ms | Speedup | max x-grad diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `indexed_sandwich` | 4096 | 17.569 | 15.972 | 1.10x | 2.73e-12 |
| `indexed_sandwich` | 16384 | 13.760 | 14.272 | 0.96x | 6.82e-13 |
| `indexed_sandwich_multi` | 4096 | 19.291 | 15.198 | 1.27x | 2.73e-12 |
| `indexed_sandwich_multi` | 16384 | 18.263 | 14.693 | 1.24x | 6.82e-13 |
| `cutlass_tiled8` | 4096 | 17.897 | 15.802 | 1.13x | 2.27e-12 |
| `cutlass_tiled8` | 16384 | 24.340 | 16.575 | 1.47x | 6.82e-13 |

`indexed_sandwich_multi` confirms that merging route/m raw-linear scheduling helps, but the custom tiled fused forward is still stronger at large N in the module benchmark. The remaining gap is now more specific: `indexed_sandwich_multi` still materializes per-m packed inputs and per-m raw outputs, then launches per-m epilogues. A deeper CUTLASS/CuTe kernel should combine the m scheduler with the raw-output epilogue and backward projection/scatter inside the same persistent grouped schedule.

Production smoke for `indexed_sandwich_multi`, with strict forward fallback disabled for the known interpolation SO2 layer:

| Case | wall s | back-half s/iter | stable comparator wall | stable comparator back-half |
| --- | ---: | ---: | ---: | ---: |
| `global_all_fused_p0_indexed_sandwich_multi` | 62.917 | 1.848 | `global_all_cublas` 54.750 | 2.043 |
| `edge_top2_fused_p0_indexed_sandwich_multi` | 66.065 | 1.965 | `edge_top2_cublas` 57.089 | 2.134 |

This is the best steady-state production signal in this branch so far: compared with the stable cublas grouped path, back-half iter time improves by about 9.5% for global all-expert and 7.9% for edge top2. Wall time is still slower because warmup/extension and remaining fallback work are not amortized in the 25-iteration smoke. Compared with the previous `indexed_sandwich` production smoke, multi-m scheduling improves global wall time from 107.079 s to 62.917 s while preserving the same rotation-sandwich semantics.

## Persistent Grouped SO2/MoE P1 Prototype

`streamed_m_major_persistent_grouped_p1` is an opt-in deeper fusion prototype. It was built after reviewing the local P1 manuscript, but the implementation in this branch makes several independent changes:

- the CUDA work queue is block-level, not thread-level, so one CTA cooperates on one route/m/output-tile problem;
- the forward kernel directly consumes compact Wigner blocks;
- `m=0` can be included as a special problem in the same route/m schedule via `DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0=1`;
- the path is trainable: backward reuses the existing segmented CUDA pair ops plus cuBLAS raw-linear reductions, while treating Wigner/R as constant because this branch targets Hamiltonian prediction without force/coordinate gradients;
- `DPTB_SO2_MOE_PERSISTENT_P1_STRICT=1` can be used during development to catch unexpected fallback.

Forward dataflow covered by P1:

1. persistent grouped schedule over route and `m`;
2. custom A-loader reads `x`, compact Wigner blocks, and per-`m` channel maps, then forms the Wigner-rotated input value inside the kernel;
3. scalar raw-linear accumulation uses the mixed route weight for that route/m problem;
4. epilogue performs m>0 complex finish, optional radial scaling, Wigner output rotation, and scatter/add into the full SO2 output.

Important limitation: this is not yet a production CUTLASS 3.x/CuTe MMA mainloop. It matches the CUTLASS grouped-kernel shape of a persistent scheduler with custom prologue/epilogue, but the dot product is still a custom FP32 SIMT loop. That is why it is useful as a dataflow prototype, not as the production fast path.

Correctness and module-train checks on Liyue, FP32 and TF32 off:

```text
pytest dptb/tests/test_so2_moe_persistent_grouped_p1.py -q
3 passed, 1 warning in 3.43s
```

Module train smoke, `cublas_grouped`, compact Wigner:

| Mode | N | streamed train ms | P1 train ms | Speedup | max x-grad diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| include m0 | 4096 | 23.866 | 13.809 | 1.73x | 2.27e-12 |
| include m0 | 16384 | 26.480 | 15.358 | 1.72x | 9.09e-13 |
| P0 comparison | 16384 | 22.074 | 15.627 | 1.41x | 6.82e-13 |

Production smoke, same Liyue short-run style, strict disabled only for the known `use_interpolation_out` SO2 layer whose m>0 path is an `InterpolationBlock` rather than MoE linear:

| Case | P1 setting | wall s | back-half s/iter | comparator |
| --- | --- | ---: | ---: | --- |
| `global_all_persistent_grouped_p1` | include m0 | 92.998 | 3.053 | worse than `indexed_sandwich_multi` 1.848 |
| `edge_top2_persistent_grouped_p1` | include m0 | 98.455 | 3.302 | worse than `indexed_sandwich_multi` 1.965 |
| `global_all_persistent_grouped_p1_nom0` | m0 fallback | 87.561 | 2.857 | still worse than `indexed_sandwich_multi` |
| `edge_top2_persistent_grouped_p1_nom0` | m0 fallback | 98.821 | 3.247 | still worse than `indexed_sandwich_multi` |

Warp-collective P1 follow-up:

- Added `DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP=warp_collective`. One warp owns one row of a route/m/output tile, lanes split the input-channel loop, and `__shfl_down_sync` reduces the dot product before the existing custom Wigner epilogue writes the result.
- Tile search showed that output tile 16 was best among the tested values, so the production smoke used `DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_N=16`.
- Correctness on Liyue: `pytest dptb/tests/test_so2_moe_persistent_grouped_p1.py -q` with `warp_collective` passed (`3 passed, 1 warning in 42.39s`).

Module train smoke:

| P1 mainloop | N | streamed train ms | P1 train ms | Speedup | max x-grad diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `warp_collective`, include m0 | 4096 | 22.714 | 12.899 | 1.76x | 2.73e-12 |
| `warp_collective`, m0 fallback | 4096 | 24.881 | 18.040 | 1.38x | 2.73e-12 |
| `warp_collective`, include m0 | 16384 | 24.224 | 15.764 | 1.54x | 9.09e-13 |

Production smoke:

| Case | P1 setting | wall s | back-half s/iter | comparator |
| --- | --- | ---: | ---: | --- |
| `global_all_persistent_grouped_p1_warp` | tile 8 | 108.337 | 2.041 | close to stable cublas, slower than `indexed_sandwich_multi` |
| `edge_top2_persistent_grouped_p1_warp` | tile 8 | 74.848 | 2.328 | slower than stable edge cublas |
| `global_all_persistent_grouped_p1_warp16` | tile 16 | 65.270 | 1.907 | faster than stable cublas steady-state, slower than `indexed_sandwich_multi` |
| `edge_top2_persistent_grouped_p1_warp16` | tile 16 | 74.444 | 2.157 | near stable edge cublas, slower than `indexed_sandwich_multi` |

Negative-result analysis:

- The P1 prologue/epilogue fusion is real: compact Wigner input rotation, complex finish, Wigner output rotation, and scatter happen inside the forward CUDA kernel for MoE SO2 blocks.
- The mainloop is the problem. Replacing the strong indexed/cuBLAS raw-linear path with scalar SIMT dot loops is too expensive at production channel sizes. The saved launches and intermediate tensors do not compensate for weaker math throughput.
- Warp-collective accumulation improves the scalar mainloop substantially versus the original CTA scalar loop, and tile 16 gives a visible global production gain. It still does not fully solve the problem because the raw-linear math is not a CUTLASS/CuTe MMA or cuBLAS-quality mainloop.
- Turning off fused `m=0` improved global all-expert from 3.053 to 2.857 s/iter, confirming that `m=0` should not use weak scalar dot by itself. It should be kept on the strong cueq/cuBLAS path or folded into a future true CUTLASS/CuTe grouped mainloop.
- Edge top2 stayed slow after the m0 adjustment, so the bottleneck is broader than scalar m0. The route/m persistent kernel still consumes route-mixed weights and uses scalar accumulation for all m>0.
- Production has a final interpolation SO2 layer. Strict mode correctly exposed that this layer is not covered by P1; the measured production runs allow that known fallback and keep the MoE SO2 warning visible in logs.

Current decision:

- Do not make P1 the production default.
- Keep P1 as an opt-in, trainable, compact-Wigner dataflow prototype for future custom-A-loader work.
- The best production signal remains `indexed_sandwich_multi`: it keeps the strong raw-linear backend and adds route/m grouped scheduling around the Wigner rotation sandwich.
- The next production-worthy step is not another SIMT fused dot. It is a real CUTLASS/CuTe grouped kernel where the A iterator computes Wigner-rotated input values, the mainloop uses a strong tiled GEMM path, and the epilogue owns top-k/radial scaling plus Wigner output scatter.

## Multi-m Grouped Pack and Output-major Epilogue Follow-up

The next experiment stays on the `indexed_sandwich_multi` line rather than the weak persistent SIMT-dot P1 line. The goal is to keep the strong grouped raw-linear backend while moving more of the SO2 rotation sandwich into CUDA:

- `DPTB_SO2_MOE_FUSED_P0_MULTI_PACK=1` packs all `m>0` Wigner input pairs in one CUDA launch into a flat `[N, 2, sum(Cin_m)]` tensor. Radial-on-input scaling remains a normal Torch multiply after slicing, so training gradients for radial MLP parameters remain intact.
- `DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE=1` replaces per-`m` raw CUDA epilogues with a grouped multi-`m` epilogue.
- `DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE_SCHEDULE=output_major` is the default. One CUDA thread owns one `(edge, output_feature)` and reduces all contributing `(m, channel, Wigner row)` entries before writing the output. This avoids the cross-`m` race that appears when each `(m, channel)` thread scatters into the same SO2 output coordinates.
- `DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE_SCHEDULE=atomic` exists only as a debug fallback. It is not recommended because it shows visible numeric drift in the module train smoke.
- `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich_multi_grouped` enables grouped pack plus output-major epilogue together.

This is closer to the requested custom epilogue direction than a GEMM wrapper: the raw output is not finished by PyTorch, and Wigner output rotation plus scatter are owned by a single CUDA schedule. It is still not a full CUTLASS 3.x `CollectiveMma + CollectiveEpilogue` kernel, because the raw GEMM itself remains the existing grouped cuBLAS call. A sub-agent review confirmed that the current CUTLASS 2.x wrapper only exposes a conventional `LinearCombination` epilogue and cannot directly express DeePTB's coordinate-aware Wigner scatter without a new custom grouped kernel.

Liyue correctness, compact Wigner, FP32 and TF32 off:

```text
export DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich_multi_grouped
export DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE_SCHEDULE=output_major
pytest dptb/tests/test_so2_moe_fused_p0.py -q -k "indexed_sandwich"

4 passed, 14 deselected, 1 warning in 48.60s
```

Module train smoke, `cublas_grouped`, compact Wigner, `cuda_cublas_segmented` backward:

| Forward mode | Extra schedule | N | fused train ms | max x-grad diff | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `indexed_sandwich_multi` | existing per-m pack/epilogue | 4096 | 15.094 | 2.73e-12 | internal comparator |
| `indexed_sandwich_multi_grouped` | grouped pack + output-major epilogue | 4096 | 15.140 | 2.73e-12 | correct, neutral/slower |
| `indexed_sandwich_multi_grouped` | grouped pack + atomic epilogue | 4096 | 15.351 | 6.07e-6 | numeric drift, reject |
| `indexed_sandwich_multi` | existing per-m pack/epilogue | 16384 | 15.072 | 6.82e-13 | internal comparator |
| `indexed_sandwich_multi_grouped` | grouped pack + output-major epilogue | 16384 | 16.400 | 6.82e-13 | correct, slower |
| `indexed_sandwich_multi_grouped` | grouped pack + atomic epilogue | 16384 | 16.184 | 2.11e-6 | numeric drift, reject |

Split ablation:

| Variant | N | fused train ms | max x-grad diff | Interpretation |
| --- | ---: | ---: | ---: | --- |
| grouped pack only | 4096 | 15.994 | 2.73e-12 | one launch saved, flat-pack/slice overhead loses |
| output-major epilogue only | 4096 | 15.165 | 2.73e-12 | close to baseline, not a clear win |
| grouped pack only | 16384 | 16.035 | 6.82e-13 | slower than old multi |
| output-major epilogue only | 16384 | 15.476 | 6.82e-13 | correct but still behind best old multi run |

Production smoke for `indexed_sandwich_multi_grouped`, Liyue L40S, bs=32, 25 iterations, FP32 and TF32 off. Strict forward mode is disabled only for the known interpolation SO2 layer fallback, matching the previous `indexed_sandwich_multi` production smoke:

| Case | wall s | back-half s/iter | previous `indexed_sandwich_multi` | stable comparator |
| --- | ---: | ---: | ---: | ---: |
| `global_all_fused_p0_indexed_sandwich_multi_grouped` | 65.191 | 1.944 | 62.917 / 1.848 | `global_all_cublas` 54.750 / 2.043 |
| `edge_top2_fused_p0_indexed_sandwich_multi_grouped` | 69.047 | 2.077 | 66.065 / 1.965 | `edge_top2_cublas` 57.089 / 2.134 |

The grouped output-major path is therefore a correct deeper-fusion prototype and still beats the stable cublas grouped path in steady-state iter time, but it regresses against the simpler `indexed_sandwich_multi` path by about 5.2% for global all-expert and 5.7% for edge top2 in back-half s/iter.

Negative-result analysis:

- The grouped input pack is semantically the right prologue step, but it produces one wide flat tensor and then slices it back per `m` before grouped GEMM. That reduces launch count but adds memory traffic and does not eliminate the per-`m` input tensors that the raw GEMM wrapper consumes.
- The output-major epilogue fixes the race without atomics and correctly eats Wigner output rotation, but each output feature now loops over a small entry list. At these channel shapes the saved per-`m` launches are not enough to beat the simpler per-`m` epilogue consistently.
- The atomic epilogue is rejected: it removes the race structurally but introduces unacceptable numeric drift in the training smoke.
- This experiment narrows the next target: pre/post fusion around an unchanged grouped GEMM is not sufficient. To get a production win, the grouped route/m scheduler must own at least the raw-output epilogue inside the GEMM tile, not after a materialized raw output tensor, and the backward projection/scatter has to be tiled the same way.

Current decision:

- Keep `indexed_sandwich_multi_grouped` as an opt-in development mode for the custom-epilogue dataflow.
- Do not replace `indexed_sandwich_multi` with it as the recommended fused-P0 steady-state path.
- Keep stable production default on `mole_linear_mode="cublas_grouped"` / `streamed_m_major_cueq`; this branch's best experimental steady-state path remains `indexed_sandwich_multi`, not grouped pre/post.
- The next real CUTLASS/CuTe step should be a custom grouped kernel whose accumulator epilogue directly performs raw complex finish and Wigner output scatter before writing to global memory. A post-GEMM epilogue kernel is useful for validating maps and semantics, but is still too late in the dataflow.

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

Aggressive fused P0/CuTe smoke root:

```text
/home/mingkang_nt/codex/0521_fused_so2_moe_aggressive_20260521
```

Important subdirectories:

```text
prod_smoke_scalar_timestamp_20260521
prod_smoke_tiled3_20260521
prod_smoke_tiled4_20260521
prod_smoke_tiled8_20260521
prod_smoke_tiled8_radial_scatter_20260521
prod_smoke_tiled8_m0_fused_20260521
prod_smoke_indexed_sandwich_raw_epi_20260521
prod_smoke_indexed_sandwich_multi_20260521
prod_smoke_indexed_sandwich_multi_grouped_20260522
prod_smoke_persistent_grouped_p1_20260521
prod_smoke_persistent_grouped_p1_nom0_20260521
prod_smoke_persistent_grouped_p1_warp_20260522
prod_smoke_persistent_grouped_p1_warp16_20260522
```
