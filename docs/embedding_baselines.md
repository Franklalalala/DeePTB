# LEM / SLEM baselines

`lem` and `slem` retain the upstream geometry-based implementations and do not
read prior inputs. `lem_prior` and `slem_prior` provide the shared H0/P input
interface. With no prior options, or `h0_init_scope: none`, they create no prior
parameters and strictly load the original geometry model's `state_dict`.

To enable priors, explicitly set `h0_init_scope: both`. Configured priors also
default to `both` when scope is omitted. Default inputs are `node_h0` and
`edge_h0` in the mapper's packed AO-product coordinates. For P priors, use
`h0_node_key: node_p23` and `h0_edge_key: edge_p2`.
`h0_init_scope` selects nodes, edges, or both; `h0_merge_mode` selects
replacement or addition; `h0_node_mode` selects direct node inputs or
zero-length self edges.

Source selection, merging, missing-input handling, and training guards follow
the production H0 interface. By default, sources can fall back first to
`node_hamiltonian` / `edge_hamiltonian` and then to the configured feature
fields. Training rejects fallback sources unless
`allow_target_fallback_in_training: true` is explicitly set.
`fallback_to_hamiltonian: false` disables fallback. If scope includes edges but
no edge source is available, both nodes and edges retain geometry initialization.
If edges are valid but node inputs are missing, only nodes retain geometry
initialization. `self_edge` selects node initialization according to whether
the whole graph contains valid self edges. The direct-node fallback always checks
source provenance, including training guards.

Legacy `method: lem` / `slem` configurations with prior options map to the
corresponding `_prior` method with a warning. When loading original geometry
checkpoints whose configuration contains prior options but whose weights lack
prior modules, the loader disables prior initialization with a warning to
preserve original behavior. Equivalent configurations produced by normal
configuration normalization are accepted; genuine overrides remain strictly
checked. New `_prior` checkpoints missing prior weights fail strict loading.

UniTB is a separate production embedding; see its [interface guide](unitb.md).
The [CPU examples](../examples/baselines/README.md) exercise prior inputs and
optimization without external data.

The SO2 layers of all four baselines select SO2CUDA by default when eligible.
The [SO2 backend guide](so2_backend.md#lem--slem) lists reference-route switches
and validation requirements. Numerical agreement and speed depend on the
selected device, precision, and workload.

Enabled priors add `prior_inputs.node_projector.{weight,bias}` and
`prior_inputs.edge_projector.{weight,bias}` after geometric parameters, retaining
geometry initialization order. e3nn Linear initializes weights from a standard
normal distribution and biases to zero. Weight counts sum the products of
matching input/output irrep multiplicities; biases cover output `0e` channels.
Version markers and output masks are buffers, not trainable parameters.
No prior options, or scope `none`, adds no such state. Original geometry weights
have not learned to use priors; enabling them adds a new capability.
