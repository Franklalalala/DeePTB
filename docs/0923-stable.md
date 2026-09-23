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
took 1.36 s instead of 1.66 s. A kernel that fuses `prior_activate` into fused-p0 itself
(segments by expert id with k-way weights) is not implemented.

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

## Deployment

Hopper release `dptb_ops/releases/20260923_stable_a503917` holds this code and a
`MANIFEST.json` with per-file SHA-256. The 0923 production wave keeps running release
`20260916_top1_switch_noshared_v1` with its worker patches until its tasks finish.
