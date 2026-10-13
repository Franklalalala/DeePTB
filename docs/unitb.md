# UniTB

UniTB unifies dense and edge-routed prior-conditioned models in one embedding.
Select it with `model_options.embedding.method: unitb`; the public class is
`dptb.nn.embedding.unitb.UniTB`. The backbone performs geometry encoding, prior
initialization, routing, message passing, and Hamiltonian output. Initialization,
interaction layers, and prior projection each have one shared implementation.

## Defaults and UniTB-dense

The default configuration uses PDQ-MoE:

~~~json
{"method": "unitb"}
~~~

UniTB-dense selects one full expert, no shared experts, top-1 routing, uniform
channels, and type routing:

~~~json
{"method": "unitb", "num_experts": 1}
~~~

Explicit options override defaults. UniTB defaults to three layers,
`latent_dim=128`, an average of 80 neighbors, and hidden channels
`128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e`. UniTB-dense uses 32 channels per
angular-momentum order. Cutoffs may be scalar or per-element dictionaries;
preserve the original dictionary when loading a legacy model.
`use_interpolation_out` controls the final-layer interpolation MLP and defaults
to disabled. A UniTB-dense checkpoint trained with this option must retain its
original value.

See [examples/unitb](../examples/unitb/README.md) for data configuration.
The charge head and training noise are separate model/training options.

## PDQ-MoE

An H₀-routed shared-basis mixture of experts (PDQ-MoE) uses, in each routed
SO(2) linear layer,

\[
W(e)=W_s+\sum_{k\in\operatorname{top-2}(e)}g_k(e)P D_k Q^\mathsf{T}.
\]

`W_s` is a full-rank shared expert used on every edge. `P` and `Q` are rank-64
bases shared by the experts within that layer, and each `D_k` is an independent
64×64 core. The actual rank is capped by `min(in_features,out_features)`.
Bond-type one-hot features and rotation-invariant CG Gram features of the edge's
H₀ block select two of four experts. Selected gates are renormalized, and linear
outputs are mixed before activation.

`PDQMoE`, `PDQMoERouter`, and `PDQMoERouting` are defined in
`dptb.nn.pdq_moe`. Bases are shared only within one linear layer. Weights are
composed per expert; activation-space dispatch avoids expanding a full weight
matrix per edge. Fewer trainable parameters do not guarantee fewer grouped-GEMM
operations or shorter training time.

| Option | Default | Meaning |
| --- | --- | --- |
| `num_experts`, `num_shared_experts`, `top_k` | 4, 1, 2 | Routed experts, always-active shared experts, selected expert count |
| `expert_parameterization` | `pdq_moe` | Shared bases; `full` uses independent complete matrices |
| `expert_rank` | 64 | Shared-basis rank |
| `router_input` | `onehot_prior` | `onehot` omits the prior; `onehot_r` uses radial distance features |
| `router_gate` | `renorm` | `full_softmax` retains the selected probability mass from the global softmax |
| `expert_mixing` | `pre_activation` | `post_activation_shared` is available as a paper control |
| `edge_router_top1_mode` | `legacy` | `switch` uses the global-softmax top-1 probability with no shared expert |
| `so2_moe_layers` | `all` | Zero-based indices of layers using routed experts |
| `edge_router_route_drop_p` | 0 | Per-structure route-branch dropout during training; disabled in validation |

`h0_init_scope: none` disables only the H₀ input embedding. A
`router_input: onehot_prior` gate still reads the frozen edge prior named by
`h0_edge_key`, which defaults to `edge_h0`. Allowed keys are `edge_h0`,
`edge_p2`, and `edge_p23`. Data must provide the field (`get_H0: true` loads
`edge_h0`); missing priors raise without falling back to labels.

Set `top_k=num_experts` for soft gating over all experts. Selected
`full_softmax` gates need not sum to one; do not fold the shared expert into
each slot. `post_activation_shared` keeps a separate shared branch. Switch
requires `top_k=1`, `num_shared_experts=0`, and its own original gate settings.

The optional `structure_mole` paper control is selected at construction:
`route_scope=structure` uses a whole-graph descriptor for soft gating over four
experts; `route_scope=constant` uses one unconditional shared core. Both retain
a full-rank shared expert and are disabled by default.
`execution=merged_core` mixes low-rank cores first; structure mode also provides
`execution=reference`. Parameter names and training-statistics buffers retain
their original paths. Statistics weight training structures equally and are
preserved when loading checkpoints. The strategy is implemented in
`dptb/nn/embedding/unitb/structure.py`, keeping structure routing out of UniTB's main forward.
It is distinct from the historical graph-router API.

Layers excluded by `so2_moe_layers` contain shared parameters only, without
empty routing parameters. Radial conditioning, residuals, and output contracts
are preserved. Layer selection changes initialization RNG consumption;
changing it does not make arbitrary multi-expert checkpoints strictly loadable.

With shared cores, `weight_experts` is a differentiably composed tensor, not a
leaf parameter. Train or scale `core_experts`, or call
`scale_expert_weights_()`. Optimizers should classify `core_experts` as expert
parameters and `basis_left/right` as ordinary shared parameters.
`full` and `pdq_moe` have different parameter structures and cannot load one
another merely by changing the configuration.

## UniTB-SLEM

`layer_topology` defaults to `"lem"`: each layer updates edges, then updates
nodes from the new edge features. Node features can therefore depend on atoms
outside one cutoff sphere. Select `"layer_topology": "slem"` for UniTB-SLEM.
PDQ-MoE, prior routing, normalization, one-hot gains, time conditioning, and the
charge head retain their settings:

~~~json
{"method": "unitb", "layer_topology": "slem"}
~~~

UniTB-SLEM adds a hidden state \(x_{ij}\) on every edge, initialized to
\(x^{0}_{ij}=e^{0}_{ij}\), the initial edge features after prior initialization
and time conditioning. Each layer performs three SO(2) maps, all reading the
layer's input node features \(h\). \(\tilde{\cdot}\) denotes equivariant RMS
normalization:

\[
\begin{aligned}
x_{ij} &\leftarrow \mathrm{Res}(x_{ij}) + W_x(z_{ij})\,\mathrm{SO2}_x\big[\tilde h_i,\tilde x_{ij}\big],\\
e_{ij} &\leftarrow \mathrm{Res}(e_{ij}) + W_e(z_{ij})\,\mathrm{SO2}_e\big[\tilde h_i,\tilde x_{ij},\tilde h_j\big],\\
h_i &\leftarrow \mathrm{Res}(h_i) + \tfrac{1}{\sqrt{\bar N}}\textstyle\sum_j W_h(z_{ij})\,\mathrm{SO2}_h\big[\tilde h_i,\tilde x_{ij}\big].
\end{aligned}
\]

Hidden-state and edge updates receive bond-type one-hot gains; node updates
receive element one-hot gains. Residual coefficients match UniTB. The edge latent
\(z_{ij}\) is updated only in the hidden-state update, using its previous value,
scalar features of the new \(x_{ij}\), and bond-type one-hot features. The update
is multiplied by the cutoff function and mixed with the residual coefficient.
Subsequent edge and node updates use this new latent.

Edge features feed the next edge residual and the edge output head, without
feeding back into nodes or hidden states. Node features therefore depend only
on atoms within atom \(i\)'s cutoff sphere. The three maps use the same
SO2_Linear type, routing coefficients, and Wigner rotations.
`so2_moe_layers` selects all three maps at each specified layer. Expert mixing
matches UniTB. Every hidden-state output, including the last layer, uses
`irreps_hidden`; edge/node output irreps match UniTB.
`use_interpolation_out` affects only the last edge and node maps.

Both UniTB and UniTB-dense (`num_experts: 1`) support this option. Legacy methods
`lem_moe_v3_edge_h0` and `lem_moe_v3_edge` use the same forward and accept it.
Historical graph-routing interfaces, block-native output routing, and archived
options reject combination with `"slem"` at construction. Node features are
strictly local with edge routing or dense execution. `structure_mole` gates use
a whole-graph descriptor and thus introduce structure-wide dependence.

UniTB-SLEM adds parameters (about 40%–50% for the documented default UniTB and
UniTB-dense configurations, depending on basis and output heads) and one SO(2)
map per layer. Measure training memory and step time on the intended workload.
Its checkpoint layout adds `layers.*.hidden_update.*`;
`layers.*.edge_update` no longer contains the latent-update `ln` or
`latents_mlp_1/2`. The default `"lem"` topology retains its module layout,
parameters, initialization RNG consumption, and strict checkpoint loading.
See the [UniTB-SLEM example](../examples/unitb/slem/input.json).

## Priors and charge equilibration

Data provides `node_h0` and `edge_h0`. Alternatively, `h0_node_key` and
`h0_edge_key` can name other physical priors with compatible representations.
The shared `dptb.nn.conditioning.projection` module handles AO/CG representations and sorted
projection. Routers read explicit priors and reject missing inputs rather than
falling back to supervision labels. Ordinary supervised training and validation
use `t=0` time conditioning.

~~~json
{"shift_head": {"mode": "atom", "response": {"kind": "qeq"}}}
~~~

`ChargeHead` reads scalar features at the input of the final interaction layer
and solves charge equilibration with `Σq=0` per structure.
The default `qeq_local=true` outputs a local readout plus `κΓq`;
`qeq_local=false` retains the original electrostatic-response form.
The default output scale is 0.1. The local physical control
`kind=context, local_only=true` removes `κΓq`.

The head requires compact SOC uu-real output, direct e3tb conversion, and physical
overlap. Use `phys_node_overlap` and `phys_edge_overlap`, rather than the
embedding latents `node_overlap` and `edge_overlap`. Corrections are onsite
`v_i S_ii` and hopping `(v_i+v_j) S_ij/2`, with the existing active-edge and
distance-expert masks. Potentials are learnable correction coordinates, not
measured DFT potentials.

Sidecars must match the primary LMDB's shard names, keys, structure fingerprints,
and edge ordering. Set `overlap_sidecar_root` for each training/validation split.
The sidecar is not read when the head is disabled. QEq and local controls are
supported; legacy method names retain parameter paths for the same response
head. Additive potential shifts, sublattice response, and auxiliary-label
training belong to historical archived versions.

## Training-only prior noise

`train_options.prior_noise_augmentation=true` uses `PriorNoiseAugmentation`,
with structured noise sampled by `StructuredNoise`. Sampling settings remain in
the historical `flow_options` section with `flow_options.enabled=false`.
`te_prior_sigma` sets the noise scale: 0.5 for the onsite recipe and 5 for
hopping. `te_prior_scale_reference=target` uses the RMS of the current supervised
ΔH. The training schema defaults to `te_prior_scale_reference=residual` and
`te_prior_sigma=1`, so the UniTB examples specify both fields explicitly.
Also set `flow_options.t_max=0`; keep `t_min` and `t0_probability` at 0.
Historical disabled configurations remain readable; enabled flow training or
ODE integration raises explicitly.

Noise is applied only during training while the model is in training mode,
within the existing expert masks, and writes zero time conditioning. Labels
remain unchanged; validation consumes no sampling RNG. Sampling preserves
node→edge order, device, dtype, and RNG consumption. It creates neither a flow
loss nor an ODE solver.

## Checkpoints and other embeddings

Legacy `lem_moe_v3_edge_h0` configurations map to the UniTB compatibility entry
point while retaining historical defaults. Legacy `MOLELinear`, `shared_core`,
`mole_expert_*`, and `edge_router_*` remain accepted with INFO diagnostics.
Parameter paths are retained, including `basis_left`, `basis_right`,
`core_experts`, `weight_shared`, `layers.*`, and `shift_head.response_net.*`.
Loading uses `strict=True`; users do not need to edit checkpoints.

`lem`/`slem` do not accept priors. `lem_prior`/`slem_prior` accept priors but
do not use PDQ-MoE. Historical `LemMoEV3`/`LemMoEV3H0` graph routing retains its
own forward and layer interfaces while reusing shared layers. Its mathematics
differs from edge-routed UniTB.

## Optional CUDA backend

DeePTB can run without SO2CUDA. With SO2CUDA 0.3.0 installed, the default
`SO2_CUDA_BACKEND=auto` selects acceleration for supported CUDA/FP32 inputs.
`SO2_CUDA_BACKEND=off` selects PyTorch reference execution. See the
[SO2 backend guide](so2_backend.md) for switches and limitations and the
[SO2CUDA README](https://github.com/Franklalalala/SO2CUDA#readme) for
UniTB/UniTB-dense acceleration checks.

UniTB-dense uses `dense_pairs` through `so2_backend.dense_forward`.
UniTB uses activation-space fused-P0 and grouped GEMM. Historical interface
names remain for compatibility; names alone do not establish the executed route.
CPU and CUDA can select different active edges near cutoff boundaries, so
individual edge outputs can differ. Validate checkpoint equivalence on the
same device and report cross-device differences separately.
