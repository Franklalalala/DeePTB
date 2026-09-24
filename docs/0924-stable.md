# 0924-stable

This branch continues `0921-stable`; it carries the `0923-stable` work as focused commits.
It adds the SO2CUDA routes for activation-space edge-wise MoE (`prior_activate` and Switch
top-1), removes per-irrep slicing from the hot layers of the LEM-v3 embeddings, shortens
the Muon Newton–Schulz step, consolidates the test suite, and fixes defects in the record
decoder, dynamic batching, the P2 table rotation, edge-wise MoE row mapping, seeded prior
sampling, the full-expert router gate and the block-ODE H0 input. Model architectures,
prior/target semantics, parameters and checkpoint formats are unchanged, and
configurations that ran on `0921-stable` run unmodified. Results change in three places:
non-PA multi-expert edge-wise MoE models where they took the expanded dispatch or the
`split_loop` backend (*Edge-wise MoE row mapping*), routers that select every expert
(*Full-expert router gate*), and block-ODE flows with `h0_ao_cg` (*Block-ODE H0 input*).

## Activation-space edge-wise MoE on SO2CUDA

`dptb/nn/so2_activation_routes.py`. `prior_activate` routes every active edge to its
top-k experts and sums their outputs with the routing coefficients; a Switch top-1 route
(`dptb/nn/top1_prior.py`) scales the selected expert by its retained probability. The
weight-space SO2 routes build one mixed weight per route token, which per-edge routing
cannot afford, so `SO2_Linear` sends these layers to one of two routes:

- **fused-P0** (`so2_fusion_mode: streamed_m_major_fused_p0`, the default for
  `prior_activate`): SO2CUDA packs m0 and, in one multi-m pack, every m>0 block; per top-k
  slot one cuBLAS grouped GEMM, segmented by expert id, covers m0 and every MoLE m>0 block,
  each expert with its own weight (the shared expert is folded in when the coefficients
  sum to one; an expert bias is one more weight column against a column of ones); the
  slot outputs of an edge are summed with its coefficients and one output-major scatter
  writes the rotated output of all blocks from the raw GEMM outputs. m>0 blocks that are
  not MoLE linears (the interpolation blocks of an output layer) run their own linear. No
  per-edge weight is built.
- **pack/scatter** (`streamed_m_major_cueq`, and the fallback of fused-P0): the same
  SO2CUDA rotation, packing and scatter, with the expert linears in `MOLELinear.forward`.

The backward of the packing is the transposed rotation, written output-major like the
output scatter: each input-gradient element is summed by one thread, where a pack per m
added every block into a full-width gradient with atomics. A layer's output is written
once, where a scatter per m wrote a full-width output per block for autograd to add.

Both are first-order CUDA float32 routes for fixed geometry. A call they do not support
(CPU or float64 inputs, differentiable geometry, autocast, a `torch.func` transform,
routing without per-row coefficients, a layer whose SO2 blocks differ in expert count or
have m>0 biases, `top1_reference_so2`, or no `so2_cuda_ops`) is declined before any kernel
runs and takes the next route, ending with the grouped streaming route; an error inside a
route is raised, never retried on another route. `so2_activation_routes.STATS` counts the
calls each route returned and the declines per reason; the first call of each route prints
`SO2_ACTIVATION_FUSED_P0_ACTIVE` or `SO2_ACTIVATION_CUDA_ACTIVE`.

| Environment variable | Effect |
|---|---|
| `DPTB_SO2_ACTIVATION_FUSED_P0=0` | fused-P0 off; activation-space layers take pack/scatter |
| `DPTB_SO2_ACTIVATION_CUDA=0` | pack/scatter off (prior_activate and Switch) |
| `DPTB_SO2_ACTIVATION_FUSED_P0_GEMM=expanded` | one grouped call over the rows of all slots instead of one per slot |

The per-slot row order is the expert sort of `MOLEGlobals.expert_slot_layout`, shared
with `MOLELinear` and `top1_prior.linear`; the layout cache follows the storage and
version of the routing tensor. Against the grouped streaming route the routes give the
same outputs and gradients to float rounding (tests below).

## Layers without per-irrep slicing

The generated forwards of e3nn modules and several DeePTB layers took every irrep block
by `narrow` or wrote it in place; in the backward each view zero-fills a gradient of the
full feature width (`SliceBackward0`, `CopySlices`). These layers now split their input
once and concatenate their output once. Parameters, buffers and state_dict keys are
unchanged; results agree with the former code to float rounding.

- `ScalarOnehotTP` (`dptb/nn/embedding/lem_moe_v3.py`): uvu paths as before; uvw paths of
  one irrep and one scalar block that couple every input with every output irrep once
  (FullyConnectedTensorProduct) run as one matmul and one batched matmul. The adapter for
  an e3nn module (`_scalar_onehot_tp_fast`) caches a parameter-free layout built outside
  inference mode, not under a `torch.func` transform, and reads the module's live weight.
  An empty output representation returns an empty tensor.
- `dptb.nn.e3nn_fast.Linear` and `.Gate`: subclasses of `e3nn.o3.Linear` and
  `e3nn.nn.Gate` with the same construction and state, used by the LEM-v3 layers, the H0
  initial layer, the prior-2b heads and the output heads. Linear with explicit or
  per-sample weights runs e3nn's forward; Gate keeps e3nn's normalized activations and
  checks at construction that its product is the per-channel gating it computes.
- `E3ElementLinear` (`dptb/nn/rescale.py`): per-channel scales and 0e shifts per block
  (the `DPTB_E3_ELEMENT_LINEAR_MODE` switch is gone).
- `EquivariantMergedRMSNormFlat` (`dptb/nn/embedding/eqv3_grid_helpers.py`): reductions
  per irrep block instead of scatter/index over the full width; `affine=False` works.
- SO2 m blocks: `complex_pair_output` forms the complex product by unbind/stack; the
  SO2CUDA routes split `radial_emb(latents)` once.
- Expert sort and unsort (`permute_rows`, used by `MOLELinear` and fused-P0): the
  backward of a row permutation gathers with the inverse instead of `index_add`.

On one batch of an edge-wise MoE SOC embedding (prior_activate, 24 experts top-2 with one
shared expert, three layers, `32x0e+...+32x6e`, fused-P0; 22,104 edges; forward and
backward with fixed weights), L40S: 598.5 ms before, 428.5 ms after. Removing single
changes from the new code: Gate +74.6 ms, `E3ElementLinear` +38.0 ms, norm +31.6 ms,
Linear +25.6 ms, row permutation +3.8 ms.

## Speed

H200, float32 (TF32 off), fixed weights; an edge-wise MoE SOC embedding (prior_activate,
24 experts top-2 with one shared expert, three layers, lmax 6) with its Hamiltonian head.
Forward and backward: one batch of 32 structures, 72,274 edges, median of 15 steps.
Training update: six batches of 32 structures (43,778–72,274 edges), median of the steady
steps, Muon for the matrix parameters.

| route | measure | 0923-stable | 0924-stable |
|---|---|---|---|
| fused-P0 | forward + backward | 777 ms | 581 ms |
| fused-P0 | training update | 917 ms | 700 ms |
| pack/scatter | forward + backward | 971 ms | 778 ms |
| pack/scatter | training update | 990 ms | 774 ms |

Peak memory of the fused-P0 forward and backward: 48.7–49.4 GiB before, 51.2 GiB
after (the multi-m packing keeps the m>0 inputs in one buffer). Outputs and gradients
agree with the former code within its own run-to-run spread.

## Edge-wise MoE row mapping

`dptb/nn/embedding/lem_moe_v3_edge.py`, `dptb/nn/tensor_product_moe_v3.py`. Without
`prior_activate`, edge-wise MoE builds one coefficient row per active edge when
`edge_router_unique_types` is false and when unique-type routing has fewer active edges
than `edge_moe_compact_min_edges` (16384 by default). These rows carried neither sizes nor
a `graph_index`, so every `MOLELinear` backend applied the first edge's mix to all edges;
they now carry `graph_index = arange(edges)`. The `split_loop` backend split rows
contiguously by sizes and ignored `graph_index`, which put every edge of the compact
dispatch on the first bond type's mix; with routes given by `graph_index` alone it now
runs one linear per route, and a batch without rows keeps zero gradients to its inputs,
coefficients and parameters. Calls with sizes are unchanged; `prior_activate` and Switch
take other branches.

Up to this branch, predictions of multi-expert non-PA models on the expanded dispatch, and
with `mole_linear_mode: split_loop` on the compact dispatch, used one mix for the whole
batch. Training took the correct route only with `edge_router_unique_types: true`,
`edge_moe_compact_dispatch: true`, a backend that honours `graph_index` (`cublas_grouped`,
`cueq_indexed_linear`, `indexed_ref`) and batches at or above the threshold (always with
`edge_moe_compact_min_edges: 0`). Evaluations of such checkpoints made with `split_loop`
or on small batches change; checkpoints trained on the correct route need no retraining.

## Full-expert router gate

`dptb/nn/tensor_product_moe_v3.py`. `MOLERouterV3` with every expert selected (`top_k` at
least the expert count on the full-expert fast path, or `top_k` unset) normalised
`sigmoid(logits)` by its sum, while the top-k gate uses a softmax over the logits. Both
now take the softmax, so selecting all E experts equals top-k with k = E. Models trained
with such a router predict differently on this branch; single-expert and top-k routers
are unchanged.

## Block-ODE H0 input

`dptb/nnops/block_ode/route_adapters.py`, `dptb/nn/embedding/lem_moe_v3_h0_helpers.py`.
The block-ODE flows write their codec RME, already in coupled coordinates, into the H0
keys. With `h0_ao_cg: true` (the default) `H0InitLayer` converted that input from AO
products to coupled RME a second time, so the node input drifted under rotation (relative
2.7) and the outputs lost equivariance. The flows now declare their H0 as coupled
(`_keys.H0_COUPLED_RME_KEY`) and the layer applies only the irrep sort; the rotated input
agrees to 2e-15. Data that does not declare it is converted as before, bit for bit.
Block-ODE models trained with `h0_ao_cg: true` before this branch predict differently.

## Other fixes

- Record decoder (`dptb/data/dataset/record_codec.py`): without the `zstandard` package,
  every ZST1 record resolved libzstd with `ctypes.util.find_library` (a fork of
  `ldconfig`) and set up its bindings. The handle and the package probe are resolved once
  per process; a DataLoader worker forked while another thread held the initialisation
  lock gets a new lock and keeps loaded handles.
- `dynamic_batch.max_samples` (`dptb/data/dataloader.py`): an explicit value that differs
  from `batch_size` is kept and logged with both numbers.
- P2/P23 table rotation (`dptb/data/interfaces/p2_table.py`): `rotation_z_to` lost
  orthogonality near −z (3.2e-3 at 2e-7 rad); for `cos < 0` it now uses the identical
  `(1−cos)/|z×n|²`.
- `train_options.epoch_checkpoint` (default `true`): `false` keeps only the `save_freq`
  iteration checkpoints. Only epoch checkpoints resume exactly; a resume from an iteration
  checkpoint restarts the data order at the epoch boundary (`DPTB_ALLOW_INEXACT_RESUME=1`).
- Muon (`dptb/utils/dpa4_optim.py`): each Newton–Schulz iteration evaluates
  `a·X + (b·G + c·G·G)·X` with `G = X·Xᵀ`, the form of the reference implementation, in
  two `baddbmm` calls, where `a·X + b·G·X + c·(G·G)·X` took one [m, m]×[m, n] product more
  and separate scaling kernels. The polynomial is the same; float32 results differ at
  rounding level.
- `HamiltonianCFM.sample` honours `prior_seed` on non-ODE routes by sampling in a forked
  RNG scope that seeds the CPU and the CUDA devices holding the sampling state (other
  devices keep their generator state); an empty l group no longer fails to reshape.

## Tests

The suite in `dptb/tests` is regrouped by behaviour (see `TESTING.md`): shared builders
live in helper modules that tests import (`block_ode_fixtures`, `pair_helpers`,
`flow_helpers`, `model_helpers`, `nacf_support`, `p2_support`, `_trainer_probes`), frozen golden values,
self-comparisons and tests of private call order are gone, and every test that needs an
optional component (CUDA, several GPUs, SO2CUDA, the NACF native builds, `dftio`,
reference data) skips with a reason when it is missing. `conftest.py` fails a module that
changes the default dtype or leaks deterministic mode.

## Validation

Numerical checks compare against an independent reference: e3nn modules or explicit
per-channel formulas in float64 with all gradients for the rewritten layers, the grouped
streaming route for the SO2CUDA routes (prior_activate with and without coefficients
summing to one, shared experts, radial layers before and after the linear, interpolation
blocks, rotation switches, both GEMM schedules, Switch top-1 including the gate
gradient), and a per-edge reference for the row mapping. Regressions cover the adapter
after an inference warm-up and under `jvp`/`vmap`, empty outputs and empty rows. The
whole-embedding A/B above agrees with the former code within its own run-to-run spread
(outputs 2e-7, worst parameter gradient 2e-6 relative).

## Not included

- Fixes from the 0726-light / 0811-consolidated line that conflict with later refactors of
  `multi_train`, `nn/build`, `argcheck` and `data/build`: d031a5e, c54a76f, e13c874,
  26fd14b, 8d8c6f9, b1a8a35, 141e68e. 3d45775 and b582aa2 apply cleanly but assume the SK
  prior route and six schema keys were removed; this line still has them.
- Feature lines on separate branches: QEq, fixed-μ SCF, the transport bridge, orbital
  descriptors, committee heads and the target-derived tied-irrep latent layout.
- Upstream deepmodeling/DeePTB `main` after 2025-09-17.
- Deployment scripts for one cluster (the former `tools/hopper_serial`); they are kept
  outside the repository.
- The NACF assembly default `ry_to_ev=13.605698` is kept: it equals the factor DeePTB
  uses to convert ABACUS Hamiltonian targets.
