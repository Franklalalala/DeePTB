# LEM / SLEM prior-aware baseline examples

From the repository root in an installed DeePTB environment, run:

~~~bash
python examples/baselines/run_smoke.py examples/baselines/lem_prior/input.json
python examples/baselines/run_smoke.py examples/baselines/slem_prior/input.json
~~~

Both configurations contain `common_options` and `model_options` accepted by
`build_model`. The example script reads `smoke_options`, builds a synthetic
three-carbon graph with H0 priors on CPU, performs 20 Adam updates, and prints
initial/final losses. The zero target checks model construction, prior inputs,
forward, and backward interfaces; a decreasing loss does not establish
predictive accuracy. No data download is needed. Use `--output <path>` to save
a JSON result.

Real data must provide `node_h0` and `edge_h0` aligned with graph node/edge
ordering, with shapes `[N, idp.reduced_matrix_element]` and
`[E, idp.reduced_matrix_element]`. Values use `OrbitalMapper` packed AO-product
coordinates. The default `h0_ao_cg: true` applies CG projection. For inputs
already in the same mapper's coupled RME coordinates, set the batch flag
`_h0_coupled_rme: true`. P priors use the same representation: set
`h0_node_key` / `h0_edge_key` to `node_p23` / `edge_p2` and provide those fields.
Priors must be independent of supervision targets; both examples disable target
fallback.

`h0_init_scope` accepts `none`, `node`, `edge`, or `both`.
`h0_merge_mode` accepts `replace` or `add`.
`h0_node_mode: direct` initializes from node priors; `self_edge` initializes
nodes from zero-length self edges, or node priors when the graph has no self
edges. Direct-node sources still undergo training fallback checks.

If scope includes edges but no edge source is available, both nodes and edges
retain geometry initialization. With valid edges but missing node inputs, only
nodes retain geometry initialization. The interface permits Hamiltonian/feature
fallback by default but rejects those sources during training unless explicitly
allowed. These examples disable fallback and provide complete independent priors.

No prior configuration options, or `h0_init_scope: none`, permits strict
restoration of the original geometry `state_dict`. Configure the scope or prior
fields explicitly to enable priors. See [the baseline guide](../../docs/embedding_baselines.md)
for checkpoint behavior.
