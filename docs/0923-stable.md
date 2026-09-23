# 0923-stable

This branch continues `0921-stable` (7d004a5d). It moves the fixes that the 0923 SOC
production wave ran as worker patches into the library and repairs two defects in the
data and prior paths. Prior/target semantics, model architectures and checkpoint formats
are unchanged; configurations that ran on `0921-stable` run unmodified.

## Changes

### Record decoder: libzstd resolved once per process

`dptb/data/dataset/record_codec.py`. Without the `zstandard` package, every ZST1 record
went through the ctypes fallback, which called `ctypes.util.find_library("zstd")` (a fork
of `ldconfig` on Linux), loaded the library and set its argument types for each record,
and retried `import zstandard` each time. The library handle and the package probe are now
resolved once per process; forked DataLoader workers inherit them, and the decompression
calls and checks are unchanged.

On a Hopper node (L2 only2b stage, 120 records, warm page cache) the training
`__getitem__` went from 31.6 ms wall / 20.2 ms CPU plus 6.9 ms child CPU per record to
23.0 ms / 19.5 ms / 0, with identical decompressed bytes for all 120 records. In the only2b
stage of the L lane (bs 96, 9 loader workers, two jobs on one node) an update went from
0.592/0.596 to 0.580/0.571 s (onsite/hopping). The rest of that update is spent outside
record decoding: the trainer process, which collates each 862 MiB batch and hands it to the
GPU, runs near one full core. GNN stages are not loader-bound; a full test-set validation
pass reads about 26 s less.

### prior_activate on the SO2CUDA pack/scatter route

`dptb/nn/top1_so2_cuda.py`, `dptb/nn/tensor_product_moe_v3.py`. `prior_activate` routes
every active edge and mixes expert outputs in activation space. On the grouped streamed SO2
route (`streamed_m_major_cueq`) its Wigner rotation and m-packing now run on SO2CUDA's
pack/scatter autograd kernels, the same ones the independent (Switch) top-1 branch uses.
The per-edge expert mixing stays in `MOLELinear.forward`; no per-edge weights are built.
A `prior_activate` layer configured with `streamed_m_major_fused_p0` is declined by the
fused kernel and lands on this route.

The route is on by default and applies to CUDA float32 with fixed geometry and a supported
Wigner layout. `DPTB_SO2_ACTIVATION_CUDA=0` disables it. When `so2_cuda_ops` is not
importable, or after the first `RuntimeError` from the CUDA path (logged once), the process
keeps the streamed route. The first call prints `SO2_ACTIVATION_CUDA_ACTIVE`.

This is the qualified 20260912 activation CUDA adapter. Against the streamed route its
outputs agree to within 5.2e-7 and its gradients to within 2.8e-6. On the PA 24/2/1 hopping
arm (bs 32), 20 updates gave losses and gradient norms identical to 4 decimals, and an update
took 1.36 s instead of 1.66 s.

### prior_activate on the fused-P0 route

`dptb/nn/so2_activation_fused_p0.py`, `dptb/nn/tensor_product_moe_v3.py`,
`dptb/nn/embedding/lem_moe_v3_edge.py`, `dptb/utils/argcheck.py`. `streamed_m_major_fused_p0`
(`indexed_sandwich_multi`) mixes one weight per route token, which per-edge routing cannot
afford. For activation-space MoLE the same structure now runs with the grouped GEMM segmented
by expert id: SO2CUDA packs each m once; one cuBLAS grouped call per top-k slot covers m0 and
every m>0 block over that slot's rows sorted by expert, each expert with its own weight (the
shared expert folded in when the coefficients sum to one); the slot outputs of an edge are
summed with its routing coefficients; the raw GEMM output is scattered straight into the
rotated output. Interpolation m>0 blocks (the output layer) run their own linear and the
finished-output scatter, so all six SO2 layers of the production model take the route. No
per-edge weight is built and no new kernel is involved.

`prior_activate` now accepts `so2_fusion_mode: streamed_m_major_fused_p0` and uses it by
default. The route declines to the pack/scatter route, and then to the grouped streaming
route, off CUDA float32, with differentiable geometry, without per-row routing coefficients,
without `so2_cuda_ops`, with `DPTB_SO2_ACTIVATION_FUSED_P0=0`, or after its first CUDA error
(logged once as `SO2_ACTIVATION_FUSED_P0_FALLBACK`). The first call prints
`SO2_ACTIVATION_FUSED_P0_ACTIVE`. `DPTB_SO2_ACTIVATION_FUSED_P0_GEMM=expanded` puts the rows of
all slots into one grouped call instead.

One training batch of the S1 hopping configuration (32 structures, 72,274 edges, H200,
forward and backward, fixed weights, router buffers restored before every step, median of
30 steps):

| Model | SO2 route | ms per step | Peak GiB |
|---|---|---|---|
| PA 24/2/1 | grouped streaming (`streamed_m_major_cueq`, no SO2CUDA) | 1675 | 49.8 |
| PA 24/2/1 | pack/scatter activation route | 1321 | 49.0 |
| PA 24/2/1 | fused-P0, one grouped call per slot | **1124** | 48.9 |
| PA 24/2/1 | fused-P0, one grouped call for all slots | 1149 | 48.9 |
| 24/2/1 without PA | fused-P0 `indexed_sandwich_multi` | 1294 | 44.2 |
| 24/2/1 without PA | grouped streaming | 1395 | 43.8 |
| dense 1/1/0 | fused-P0 `indexed_sandwich_multi` | 1275 | 39.6 |
| dense 1/1/0 | grouped streaming | 1377 | 39.2 |

Against the grouped streaming route on the same step the fused-P0 route gives the same loss,
outputs within 1.4e-7 and gradients within 5.4e-6 (worst of 267 parameters, relative to the
parameter's largest gradient).

Switch top-1 routes (256/1/0, `dptb.nn.top1_prior`) take the same route as one slot without
folding: each edge's selected expert, bias included, scaled by its retained probability, as
`top1_prior.linear` computes it. The route declines for `top1_reference_so2` and for layers
with shared experts, which `top1_prior.linear` refuses. On one batch of the S9 hopping
configuration (256/1/0 Switch, P→H−P, 598.5M parameters, 72,274 edges, same protocol):

| SO2 route | ms per step | Peak GiB |
|---|---|---|
| grouped streaming, top-1 SO2CUDA branch off | 1395 | 50.5 |
| grouped streaming with the top-1 SO2CUDA branch | 1042 | 49.8 |
| fused-P0 | 994 | 49.7 |

The fused-P0 route gives the same loss, outputs within 1.9e-7 and gradients within 2.1e-5 of
the branch-off route (the top-1 branch: 1.0e-5, same parameter). The weight-space fused-P0 route of the dense and non-PA models
keeps the output layer's two interpolation SO2 layers on the grouped streaming route and m0 on
the torch path, which is why the PA model on the new route is faster than they are.

### `train_options.epoch_checkpoint`

`dptb/plugins/saver.py`, both training entrypoints. The Saver always had an extra
`(1, 'epoch')` trigger besides `save_freq` iterations. `epoch_checkpoint` (default `true`,
unchanged behaviour) selects the triggers through `checkpoint_intervals`; `false` keeps
only the `save_freq` interval. Epoch checkpoints remain the only exactly resumable ones: a
resume from an iteration checkpoint restores model, optimizer and scheduler state and
restarts the data order at the epoch boundary (`DPTB_ALLOW_INEXACT_RESUME=1`).

### `dynamic_batch.max_samples` warning

`dptb/data/dataloader.py`. `max_samples` defaults to `batch_size` only when it is absent.
An explicit value that differs from `batch_size` (for example a base configuration's
`max_samples: 96` kept after setting `batch_size: 32`) is kept and now logged with both
numbers, because batches then hold up to `max_samples` records and the cost budget is
calibrated for that size.

### Stable near-antiparallel rotation in the CPU P2/P23 table path

`dptb/data/interfaces/p2_table.py`. `rotation_z_to` evaluated `I + K + K²/(1+cos)`. Near
the south pole `1+cos` cancels: 2e-7 rad from −z the matrix was off orthogonality by 3.2e-3
and mapped z onto the bond direction with a 1.6e-3 error. For `cos < 0` the factor is now
the identical `(1−cos)/|z×n|²`, the form already used by `dptb.nacf.radial`. Edges with
`cos ≥ 0` and the exact-pole snaps rotate bit-for-bit as before; other `cos < 0` edges agree
to rounding. Datasets that store their priors are not affected; a P2 table evaluated on the
CPU for such edges is.

### From the 0811 line (6984214)

`SO2_Linear._materialize_output_l_groups` spells the flattened extent instead of
`reshape(n, -1)`, which raised for an empty l group (n = 0). `HamiltonianCFM.sample`
honours `prior_seed` on non-ODE routes by sampling in a forked, seeded RNG scope instead of
rejecting it.

## Operations outside the library

These stay in the production tooling (`nextham_budget_0923/bin` on Hopper and the
`nus-hopper-pbs-submit` skill scripts), because they depend on the cluster layout:

- the serial two-stage task runner (stage 1 only2b, then stage 2 from its checkpoint via
  `init_model`, one task holding one allocation);
- indexed `pread` from immutable LMDB sources on DPC `/scratch` and per-task staging of the
  validation set to `/dev/shm`;
- a dynamic-batch cost cache keyed by dataset identity, per-head pooled MAE/RMSE validation
  records and the calibration policy check (q = 0.95, 128 batches, `max_samples` = batch size).

For the only2b stage of a 12-core, 1-GPU allocation, `train_num_workers: 9` with one
BLAS/OpenMP thread per worker brought the L lane from 1.086 to 0.672 s per update; batch
composition and order do not depend on the worker count.

## Not included

- Fixes from the 0726-light / 0811-consolidated line that conflict with later refactors of
  `multi_train`, `nn/build`, `argcheck` and `data/build`: d031a5e (common_options merge on
  restart/init-model), c54a76f, e13c874, 26fd14b, 8d8c6f9, b1a8a35, 141e68e. 3d45775 and
  b582aa2 apply cleanly but assume the SK prior route and six schema keys were removed;
  this line still has them.
- Feature lines on separate branches: QEq, fixed-μ SCF, the transport bridge, orbital
  descriptors, committee heads and the target-derived tied-irrep latent layout.
- Upstream deepmodeling/DeePTB `main` after 2025-09-17 (38 commits in the fork's `main`).
- The NACF assembly default `ry_to_ev=13.605698` is kept: it equals the factor DeePTB uses
  to convert ABACUS Hamiltonian targets, and the SOC CLIs pass 13.605693122994 explicitly.

## Validation

On a Hopper H200 node (PyTorch 2.8.0+cu128), the focused tests pass: 284 passed and 2
skipped (the `zstandard` package is absent; no real P2 table root is configured). They cover
the record codec, the `prior_activate` route including its GPU equivalence test, the Saver
triggers, dynamic batching, the P2 table, streamed SO2 bounds, Switch top-1, prior-2b PA,
restart/resume, the plugin clock, the flow TE prior and the default smoke suite.

A GPU smoke on an H200 ran the Hopper release of this code with production workers that no
longer patch the library. Each comparison is against the same configuration on release
`20260916_top1_switch_noshared_v1`:

| Run | Result |
|---|---|
| PA 24/2/1 hopping, 20 updates | `SO2_ACTIVATION_CUDA_ACTIVE`; loss and gradient norm identical at every update to the old code run with the worker patch; peak 47.37 GB on both |
| 256/1/0 Switch P→Pres, 30 updates plus validation on 256 test records | losses identical; validation MAE/RMSE equal to 1e-9; top-1 CUDA branch used 1,716 times on both |
| dense two-stage, 30 only2b plus 30 GNN-stage updates | stage 2 initialised from the stage-1 checkpoint, GNN seeded, two-body branch unchanged |
| L2 only2b stage from step 0, 200 updates | loss, running mean and gradient norm identical at steps 100 and 200; 0.48 s per update with the job alone on the node |

With the fused-P0 `prior_activate` route (8af4d9d) the focused tests on an H200 give 250
passed, 8 skipped and 1 failed, with the production-only `DPTB_MOLE_LINEAR_MODE` and fused-P0
mode variables unset. The failure,
`test_so2_non_moe_cublas::test_non_moe_so2_indexed_sandwich_cuda_multi_block_complex_matches_standard`,
imports `indexed_sandwich_multi_block_direct_gemm`, which `dptb.nn.cuda_ops.grouped_gemm` does
not define; a503917 fails it identically. The route's GPU tests compare it with the grouped
streaming and pack/scatter routes in forward and backward (including the routing
coefficients) for front and non-front radial layers, interpolation m>0 blocks, layers without
a shared expert, both GEMM schedules and coefficients that do or do not sum to one, and, for
Switch top-1 routes (059507d), with the streamed route and the top-1 pack/scatter branch
including the gate gradient; those 44 tests pass on an H200.

A production worker carrying the same route code ran the prior_activate + TE-flow hopping arm
for 30 updates and one ODE validation on 256 test records: the validation MAE and RMSE differ
from the pack/scatter route by 1.2e-8 and 4.7e-8 (relative), the validation loss is identical,
and the validation pass took 147 s instead of 169 s.

## Deployment

Hopper release `dptb_ops/releases/20260923_stable_8af4d9d` holds the code up to the fused-P0
route (later commits change documentation only), with a `MANIFEST.json` of per-file SHA-256.
`20260923_stable_a503917` holds the code before it. `dptb_ops/current` is not changed. The
0923 production wave keeps running release `20260916_top1_switch_noshared_v1` with its worker
patches; its prior_activate tasks take this route through worker v9, whose route code is
`dptb/nn/so2_activation_fused_p0.py` of 8af4d9d.
