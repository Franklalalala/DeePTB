# Shared bases for routed SO2 experts

Opt in in the embedding config:

```json
{
  "mole_expert_parameterization": "shared_core",
  "mole_expert_rank": 64
}
```

The default `full` retains the existing parameter names, initialization, and
checkpoint format. `shared_core` represents each routed matrix as `P D[e] Q.T`.
Each MOLELinear owns shared `basis_left[out, rank]` and
`basis_right[in, rank]`, plus `core_experts[E, rank, rank]`. Rank is capped at
`min(in, out)`. Affine shared experts and routed biases keep their usual roles.
Bases are shared between experts of one SO2 block, not across different blocks.
The m>0 complex-pair layout and absence of bias are unchanged. Output
interpolation blocks are unchanged.

The readable `weight_experts` attribute returns the differentiable synthesized
bank. It is not a leaf parameter in shared-core mode: mutate the cores, not
`weight_experts.data`, and enumerate `named_parameters()` for optimizer state.
`scale_expert_weights_()` scales the represented routed bank in either mode.
The bank is expert-sized and goes through the existing grouped/fused-p0
interfaces; it is never replicated per edge on the activation-space path.
This implementation reduces trainable storage, not grouped-GEMM FLOPs, and
adds bank synthesis cost. It makes no training-accuracy or speed claim.

HybridMuon's default expert patterns include `core_experts`; the two basis
matrices follow ordinary shared-parameter rules. If an experiment explicitly
overrides `expert_name_patterns` or `adamw_name_patterns`, its core patterns
must be specified deliberately. Do not accidentally classify basis rows as
experts. No optimizer hyperparameter, router setting, gate, loss, or schedule
is changed by selecting this parameterization.

Random semi-orthogonal P/Q and random D match the full bank's expected squared
Frobenius norm at initialization. SO2 m>0 applies its existing `1/sqrt(2)`
scale to D. Factors initialize inside a CPU/CUDA RNG fork, followed by one
temporary full-bank uniform draw on the original device and dtype. This
preserves the full model's subsequent random-number sequence and all common
initial parameters (including routed biases, shared/radial blocks and router).
The temporary full bank costs memory during construction/reset only and is
discarded immediately; it is never a parameter or forward cache. The routed
matrices and hence the model's initial function still differ.
Full and shared-core checkpoints are intentionally
different and strict loading rejects a layout mismatch; use the same config
to resume a shared-core checkpoint. Existing external `dense_to_shared`
workers that only recognize `weight_experts` need an explicit converter
before they can initialize this mode. The initial candidate is from scratch.

For a zero-residual initialization, keep nonzero P/Q, zero D and routed bias,
and put the dense function in the shared branch. P/Q then have zero gradient
on the first step while D can learn; zeroing all three factors prevents
learning. This statement concerns the linear block; nonlinear branch
initializations need their own channel-level check.

Focused validation in a configured environment:

```bash
python tools/test.py dptb/tests/test_mole_shared_core.py
```

The CUDA case runs only with CUDA and SO2CUDA available and checks actual
fused-p0 dispatch plus output and gradients against staged execution. CPU
cases cover independent factorized arithmetic, an empty expert, m>0-style
paired rows, coefficient mass, default compatibility, serialization, builder
propagation, and expert-only optimizer scaling.

Full-soft per-edge routing is supported with `top_k = num_experts` (or
`top_k: null`). With E=4 the router exposes `[N, 4]` canonical expert indices
and the differentiable full softmax probabilities; empty inputs have the same
metadata contract. Activation-space dispatch sums all four experts, including
shared-core m>0 blocks, without a per-edge weight bank. No extra config key is
needed. Full selection never updates balancing bias. Opt-in training route
statistics now include soft loads and squared soft loads for this path; hard
load CV alone cannot diagnose a full-soft router. The optional legacy slow
router still updates its historical hard-load EMA when a finite K selects all
experts. The default fast route leaves that buffer alone. Sparse K<E routing
retains its previous arithmetic, selection, gradients and state updates.
