# TE-flow on a reused pairwise S1 checkpoint

Start from the matching two-stage baseline, not a historical single-stage PA
configuration. Strictly load a completed `only2b=true` S1 checkpoint into S2;
the pairwise branch stays frozen. Starting a new flow experiment does not reuse
the old S2 optimizer or schedule state.

Enable `use_flow_time_embedding`, `flow_time_condition_edges`, and set
`flow_time_allow_missing=false`. Enable `train_options.flow_options`, with
`prior=te` and `te_prior_mode=typewise`. The flow input keys must match the
serial model's selected prior:

| Model prior | Flow node input key | Flow edge input key | Dataset target |
|---|---|---|---|
| `h0` | `node_h0` | `edge_h0` | `h0res` |
| `na_cf` | `node_p23` | `edge_p2` | `nacfres` |

For this RME endpoint experiment, with selected prior P and target Y=H-P,
`x_t=(1-t)(P+TE)+tY`. The trainable GNN receives x_t and time; the frozen S1
branch receives the original P, preserved by both flow preparation and sampling.
This is clean-endpoint MSE, not direct velocity regression or a full-H endpoint.
The sinusoidal time conditioner adds no persistent checkpoint parameters.

Keep the baseline's PA-router choice, SOC/uu-real interpretation, loss masks,
optimizer, WSD successful-update clock, calibration quantile, batch coverage,
precision, and seeds unchanged. A smoke must exercise actual data, finite
forward/backward/optimizer updates, strict S1 reuse, frozen-input invariance,
time-only sensitivity, excluded gradients, and checkpoint writing. It does not
establish an improvement over the non-flow control.

## Retention and storage

Use a PBS parent independent of the training child. The supplied
`tools/hopper_serial/retained_workflow.pbs` waits for a deployment-ready marker,
then invokes a controller. On either controller success or failure it holds the
allocation without retrying. The existing allocation owner can run a `campaign`
request that performs smoke followed by production, with `stop_after=false`;
child failures then leave the owner and its staging lease alive. TERM/INT remain
explicit shutdown signals, and scheduler walltime/node failure still apply.

Prefer complete node-local data. If disk and job RAM cannot fit it, retain the
qualified indexed ordinary-pread loader: immutable source identity/offsets,
bounded short-read retries, no shared dataset mmap, no disk cache, no per-record
hashing. Keep the environment, CUDA extensions, and JIT caches on executable
node-local storage. Exclude foreign `__pycache__` and `.pyc` files when copying
source trees. Keep the original graph, labels, ordering, and batch calibration.
