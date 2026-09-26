# Late atom routing

`lem_moe_v3_edge_h0` accepts `edge_router_scope="atom_after_layer0"`.
The default `"edge"` preserves the previous model, RNG consumption, parameter
layout and forward/backward. The new mode requires:

```json
{
  "edge_router_scope": "atom_after_layer0",
  "edge_router_prior_activate": true,
  "edge_router_input": "onehot_prior",
  "num_experts": 4,
  "top_k": 4,
  "num_shared_experts": 1,
  "so2_moe_layers": [1],
  "so2_expert_mixing_mode": "pre_activation",
  "edge_router_route_drop_p": 0.0,
  "edge_router_bias_speed": 0.0,
  "edge_router_type_support": 0
}
```

For three layers, layer 1 is the last hidden layer. Layer 0 must be shared only.
Other subsets of indices greater than zero are supported: the same router reads
updated node features immediately before each selected layer. Unselected layers
use the existing shared SO2 path. Other embedding variants with separate forward
loops are rejected. No extra router is constructed on the disabled path.

## Representation and gradients

Each node contributes signed 0e channels and per-copy irrep norms
`sqrt(sum_m h_m^2 + 1e-8) - 1e-4`, including scalar magnitudes. No cross-parity
contractions are used. These features are concatenated with a frozen-prior
neighborhood descriptor. The latter uses the existing signed-log/Gram descriptor,
with AO products first converted through the H0 init layer's AO-to-CG map. A batch
explicitly declaring coupled RME skips that conversion. Unmarked AO input with
`h0_ao_cg=false` is rejected. Target Hamiltonians are never a routing fallback.

The strict existing reverse-edge map pairs `(i,j,R)` with `(j,i,-R)` (including
periodic self images), rejects duplicate/missing/cross-graph partners, and checks
that active sets are paired. Reverse descriptors are averaged, then pooled over
outgoing neighbors with the average pair cutoff. Divide by total cutoff weight,
with a `1e-8` floor; zero-neighbor atoms get zero prior descriptors. The mean is
local to an atom and its periodic neighbors, never across structures.

The existing router MLP produces all four softmax coefficients, with its existing
logit kind and temperature. All input columns initialize normally in this mode,
so environment and prior can influence the initial route. There is **no stop
gradient**: an expert-specific task gradient should teach the first shared layer
what information is useful for subsequent specialization. No discrete top-k,
selection bias, noise, or route dropout participates in this mode.

The edge coefficient is `(alpha_i + alpha_j)/2`, so both directions are exactly
equal. Canonical expert slots 0..3 and these differentiable coefficients enter
`MOLEGlobals(activation_space=True, coefficients_sum_to_one=True)`. The existing
linear pre-activation mixing supports `full` and `shared_core` expert banks.
The latter still materializes **K expert matrices**, not an edge-sized weight
bank, and executes four slots. No speed improvement is claimed.

## Diagnostics and checkpoints

Every routed forward writes an INFO `atom_route` JSON record and detached
`embedding.last_atom_route_stats[layer]`. Values include coefficient means,
population standard deviations across atoms, element means (atomic-number keys),
soft atom/structure mass, and Kish effective counts. For an expert, atom ESS is
`(sum alpha)^2/sum(alpha^2)`; structure ESS uses each graph's **mean atom alpha**.
Uniform alpha therefore has maximal ESS even when the expert receives little
mass; read mass and ESS together. Statistics are per local batch/rank, not a
unique-structure count accumulated across training. Layer/mode/optimizer step
identify records; plain optimizers need to publish opt_step to the router if they
want that field incremented. Diagnostics contain no autograd graph or persistent
checkpoint state. Multiple selected layers sum their router z regularizers.

The new router changes parameter shape and therefore requires a new model/config;
there is no automatic old-checkpoint conversion. Normalization mean/std become
persistent buffers only in the new mode and restore with model state. If a config
names an external prior-statistics file it must still be accessible during model
construction. Old statistics on unconverted AO descriptors are not interchangeable
with these invariant descriptors. Model and optimizer roundtrip are tested.

## Validation and limits

`dptb/tests/test_r14c_atom_route.py` covers explicit FP64 expert sums and all
input/logit/parameter gradients for both parameterizations, proper rotations of a
complete embedding, AO/coupled input equivalence, node inversion invariance,
atom/edge permutation, route timing, selected layers, reverse consistency,
backbone gradient, pooling, logs, and config/checkpoint behavior. Its CUDA tests
compare staged and fused-p0 full-model outputs/all parameter gradients and require
observed fused dispatch **inside the routed layer**. They are skipped without
CUDA/SO2CUDA; a configured backend alone is not GPU evidence.

`test_r14c_real_training.py` accepts `R14C_MINI_ROOT`, `R14C_TEST_DEVICE` and
`R14C_METRICS_DIR`. It trains separate onsite/hopping heads on four real local LMDB
records for six steps, checks excluded-head gradients, logs every step and on GPU
requires routed-layer fused calls at every step.

Reverse-map validation currently transfers topology to CPU; INFO statistics also
synchronize for serialization. GPU throughput/memory must be measured before
production claims. This change has no held-out-accuracy or force/stress validation;
existing fused CUDA supports first-order fixed-geometry training. Small hidden
configs must cover output angular momentum (the real f-basis fixture needs l=6)
to reuse Wigner blocks, as in the production baseline.
