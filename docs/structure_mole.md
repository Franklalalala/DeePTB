# Structure MoLE (r14 A path)

This opt-in implementation targets `lem_moe_v3_edge_h0`, with the existing
pre-activation SO(2) parameter layout. Existing configurations, including an
explicit `structure_mole.enabled=false`, keep their legacy tensors, random
stream, router and execution path.

## Configuration and calibration

All new keys live in `model_options.embedding.structure_mole`:

```json
{
  "enabled": true,
  "route_scope": "structure",
  "prior_stats": true,
  "execution": "merged_core",
  "hidden": 64,
  "rbf_rmax": 10.0,
  "stats_path": "train_structure_stats.pt",
  "init_from": ""
}
```

The surrounding embedding must use `num_experts=top_k=4`,
`num_shared_experts=1`, `mole_expert_parameterization=shared_core`,
`mole_expert_rank=64`, `so2_expert_mixing_mode=pre_activation`,
`edge_router_prior_activate=false`, `edge_router_bias_speed=0`, and
`edge_router_route_drop_p=0`. The standard three-layer experiment keeps
`so2_moe_layers=all`. Existing interpolation blocks stay interpolation blocks.
Do not activate the old edge router alongside the structure router.

Before a fresh training build, run, in the configured DeePTB environment:

```bash
PYTHONPATH=. python tools/calibrate_structure_mole.py input.json train_structure_stats.pt --device cpu
```

The CLI reads only `data_options.train`, visits every record once and uses each
structure with equal weight. It ignores `stats_path` and `init_from` during
calibration. Pass the resulting file through `stats_path` for training. The API
`fit_structure_stats(model, iterable, split="train")` supports the same operation
on graph dictionaries or AtomicData. Calibration needs at least two structures;
recalibration of an already fitted instance and non-training split declarations
are rejected. Input provenance still depends on the caller identifying the
training split correctly. The bundle records source configuration SHA256,
training data options, basis/cutoff/descriptor contract, count, mean and scale.
No label statistics or fitted potential shifts are used.

Statistics are persistent buffers. A checkpoint reload restores the buffers,
projection and router, and never reopens the original statistics or dense
checkpoint paths. Fresh, uncalibrated structure routing raises an error. This
avoids silently using identity scales in a production run.

## Descriptor contract

For T elements the width is `T + T(T+1)/2 + 16 + 64`. The entries are:

1. Element fractions by atom count, including atoms with no neighbors.
2. Unordered element-pair fractions, using smooth cutoff weights.
3. Sixteen Gaussian radial means: centers uniformly span [0, rbf_rmax], width
   rbf_rmax/15; weights use the backbone's actual polynomial/cosine cutoff,
   including its element-dependent radii.
4. The existing `_raw_prior_source` mask/sort convention and `_gram_descriptor`
   construction (signed scalar channels and same-(l, parity) upper-triangle
   Gram, signed log). The new path first honors the H0 representation flag:
   AO-product priors pass through H0InitLayer's existing CG conversion; coupled
   RME priors only need sorting. Legacy edge routing is unchanged. AO-product
   inputs with `h0_ao_cg=false` are rejected because sorting alone is not an
   equivariant change of basis.
   Only explicit `edge_h0`/`edge_p2` keys are accepted. A CPU seed-0 Gaussian
   projection with entries N(0, 1/32) reduces the descriptor to 32 channels.
   The pooled mean and population standard deviation give 64 channels.

The validated periodic reverse mapping `(i,j,R) -> (j,i,-R)` symmetrizes the
projected descriptor, distance features and cutoff weight. One representative
of each pair is counted; a self-reverse edge counts once. Missing/duplicate
reverse edges and cross-structure edges raise errors. Per-structure means use
weights divided by their sum. Empty-neighbor structures have zero edge
statistics. Standard deviation uses a centered second moment, with exact zero
and a finite derivative at zero variance.

Training mean and population standard deviation normalize every channel.
Channels with training standard deviation <= 1e-6 use unit scale. No online
updates occur. Raw features and alpha are constructed once before the layer
loop. The batch index only selects a pooling segment; its numeric value never
enters the descriptor. Learned one-hot embeddings are not part of z.

`prior_stats=false` (A-R) replaces the last 64 **normalized** channels by exact
zeros, retaining the same width, router, all other features, and the A arm's
calibration file. This only ablates the routing statistics; the backbone still
uses its original prior.

## Execution and initialization

`alpha = softmax(Linear(SiLU(Linear(z))))` has four coefficients per structure.
Each legal SO(2) MoLE affine block owns P, Q and D_k independently, plus one
shared affine block. Alpha is reused in every layer.

`execution=reference` broadcasts alpha onto active edges and dispatches all four
slots through the existing activation-space route. With CUDA float32 and
`so2_fusion_mode=streamed_m_major_fused_p0`, this is the existing fused-P0
four-slot implementation; CPU uses its numerical fallback. Shared terms are
folded into slots only with normalized coefficients and before the unchanged
nonlinearity.

`execution=merged_core` first computes `D_s = sum_k alpha_sk D_k`. It groups
activations by structure once and, in each block, evaluates
`x W0^T + ((x Q) D_s^T) P^T`, including the corresponding shared/mixed biases.
Complex pair rows stay in their existing `[edge, 2, channel]` layout. Compact
Wigner operations use the existing streamed reference code, then the usual
single nonlinearity. No full expert bank or per-edge full weights are built in
this path. This first implementation uses PyTorch group loops, not a new CUDA
kernel; grouping, small core contractions and backward still cost time.

`route_scope=constant` (A-1) requires K=top_k=1 and merged execution. There is no
router or statistics module, no softmax, no top-k metadata, and no Switch gate.
The explicit branch applies W0 plus its single P D Q^T core. Rank and routed
layer placement stay the same.

For U-A, `init_from` is a new dense-checkpoint conversion entry in `build_model`.
It requires the LEM dense one-expert/zero-shared layout, the same backbone,
prediction and distance wrapper, and no fitted shift head. Each legal MoLE
parameter matrix is factorized in FP64 by truncated SVD, then cast back. At the
configured rank, `R = P D Q^T`, `W0 = W_dense - 0.5 R`, and all four cores are
`0.5 D`. Dense bias is copied to shared bias; routed bias is zero. The truncation
remainder stays in W0, preserving the dense function to floating-point rounding.
The m>0 stacked real/imaginary output matrix is factored intact, with its complex
pair assembly and no-bias rule unchanged. No AO matrix is factorized.

The randomly initialized router keeps nonzero feature weights, so distinct z
produce nonconstant coefficients initially. Identical cores make the output
independent of alpha at step zero, even with nonuniform coefficients. Core
updates differ because their gradients are weighted by different alpha. Router
gradients are mathematically zero at that first step, then become informative
when cores diverge. The optimizer is freshly initialized, not inherited from
dense. `init_from` remains in checkpoint configuration as provenance.

## Diagnostics and limits

Each training forward writes an INFO record and exposes detached
`structure_mole_alpha_mean`, `structure_mole_alpha_std`,
`structure_mole_covariance_effective_rank` and `structure_mole_n_eff` tensors.
These are per-local-batch, per-distance-head metrics, not a dataset-wide coverage
estimate or a cross-rank reduction. For structure routing, q_sk=alpha_sk and
`n_eff,k=(sum_s q_sk)^2/sum_s q_sk^2`. Covariance effective rank is
`exp(-sum_i p_i log p_i)` for normalized nonnegative covariance eigenvalues;
zero covariance yields 0. Its maximum is K-1 for normalized coefficients.
Population standard deviation is used. A single structure has zero covariance
rank and n_eff=1. Diagnostics synchronize device scalars and are included in the
benchmark step time.

Random rotations, atom/edge permutations, batch isolation and reverse pairing
are checked. Repeating a periodic primitive cell preserves the intensive raw
statistics and thus the same structure condition. Geometry/prior statistics
are recomputed for each forward; there is no cross-configuration cache.

Two different noninteracting fragments put into one structure share a pooled
condition: moving them farther apart after removing interfragment neighbors
does not remove this coupling. This is acceptable for the intended
compound-conditioned finite periodic sample, **not** for a model requiring
strict locality, fragment additivity or general size consistency across mixed
fragments. A-1 is the control for that global condition. No held-out accuracy,
energy-conservation or MD claim follows from these implementation tests.

The strict reverse mapper currently validates topology on CPU, and the merged
path has Python loops and small GEMMs. Production speed is unqualified until
measured. CPU timings are not L40S estimates. Run `tools/run_gpu_tests_r14a.sh`
from the delivery workspace for mandatory fused-P0 dispatch/gradient checks,
real-record smoke, and serial-residency peak-memory/step-time measurements.
