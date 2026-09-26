# Routed SO2 layer placement

`model_options.embedding.so2_moe_layers` defaults to `"all"`. Existing parameter
names, initialization and execution are unchanged. A nonempty list selects
zero-based LEM layer indices; `[1]` in a three-layer model selects the last hidden
layer, not the final output layer. The option requires per-edge prior-activate
routing and at least one shared expert when only a subset is selected.

Both node and edge updates in selected layers keep their configured routed
experts and mixing mode. Other layers own only `weight_shared` and `bias_shared`,
with no dormant or zero-sized routed parameters. They apply the shared SO2 path
and activation once. Radial conditioning, output interpolation, norms, residuals,
latent updates and heads retain their existing semantics. The inactive SO2 layer
constructs shared-only routing metadata, so embedding-level expert ids never
index an inactive layer. Fused-P0 retains its shared-path/packing implementation;
it does not dispatch routed grouped GEMMs for those layers.

## Dense checkpoint conversion

For a controlled comparison to all-layer dense upcycling, use exactly the same
finished dense checkpoint, data order, fresh optimizer and short-cycle schedule.
Change only `so2_moe_layers: "all" -> [1]`. Keep the prior router and, for an S19pc2
comparison, `post_activation_shared` plus `full_softmax`.

Fewer expert tensors consume fewer initialization random numbers. In particular,
the edge router is constructed after the layer stack, so the same common seed
alone does not guarantee an identical initial router. For a placement-only
comparison, copy the exact router module state (including balancing buffers) from
the all-layer arm's **iteration-zero initialization checkpoint**, after applying
the dense conversion below. Require exact key/shape equality for that module;
do not use its trained endpoint router. Keep the same prior-descriptor statistics
file/content as well: `_prior_mean`/`_prior_std` are non-persistent buffers and
must be reconstructed from the same configuration, not assumed present in the
checkpoint. Record both the dense source and reference-router checkpoint.

The existing worker-v20 `dense_to_shared` mapping applies without changing tensor
layout: copy each dense `weight_experts[0]`/`bias_experts[0]` into the matching
shared slot; zero every routed tensor that still exists; copy all other model
tensors by exact name and shape. Inactive layers have no routed tensors to zero.
Only explicitly identified router/descriptor state may remain newly initialized.
In a placement-only paired run, replace that new router state by the matched
iteration-zero router as described above.
Reject missing, unexpected or shape-mismatched non-router state. Mapping an
already-trained multi-expert checkpoint to this placement is not dense upcycling
and must not silently drop its routed parameters.

Do not turn inactive layers into one-routed-expert layers: optimizer patterns for
`*weight_experts*`/`*bias_experts*` would then apply routed learning-rate/decay
overrides to the dense backbone. Shared-only parameter names preserve the
all-layer upcycling optimizer contract. Record the actual optimizer groups.

Before training, compare full node/edge predictions and loss at iteration zero
against the dense checkpoint on representative real inputs. MAE equality alone
is insufficient. Check active experts and per-angular-channel gradients over the
first two updates: at zero routed outputs the router's initial task gradient is
expected to vanish, then become available after experts move. Preserve the final
layer's interpolation configuration in this check. Production CUDA dispatch and
representative forward/backward require a configured GPU environment.

Focused behavioral checks: `python tools/test.py dptb/tests/test_moe_layer_placement.py`.
