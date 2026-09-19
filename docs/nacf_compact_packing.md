# Shared feature maps and CUDA packing

`NACFFeaturePlan` accepts `mapping='compact'` and `packing_backend='cuda'`:

```python
assembly = bank.prepare(**geometry, topology='native')
features = NACFFeaturePlan(
    assembly, mapper, mapping='compact', packing_backend='cuda')
result = features()
```

The compact form stores one signed AO-to-feature template for each directed
species pair and one integer type per output block. It preserves original edge
order, orbital phases, padding, scalar/uu-real and full SOC feature conventions.
It reduces mapping metadata from O(blocks × features) to
O(pair_types × features + blocks). It does not change the physical operator,
geometry preparation, pair tables, overlap sources or output feature size.

The optional CUDA kernel gathers through the template directly, applies the
sign, selects the real/imaginary SOC component and casts to the output dtype in
one launch per node/edge array. Input and output support float32/64 and
complex64/128. Non-contiguous AO input is made contiguous. Differentiable AO
inputs use the ordinary Torch path, preserving gradients. Neither import nor
inference compiles code; explicitly rebuild with `python -m dptb.nacf.precompile`
using an architecture suitable for the target device before selecting CUDA.

The defaults remain `mapping='expanded', packing_backend='torch'`. Expanded maps
are now assembled by unique pair and vectorized indexing. This preserves the
existing per-block `node_indices`/`edge_indices` buffers for consumers accessing
them. Compact mode instead exposes `*_rows` and `*_template_*` buffers; its plan
state has a different layout and must not be loaded as an expanded plan. These
are geometry-bound prepared plans, not model parameter checkpoints.

Compact Torch packing expands the maps transiently, so compact resident storage
alone does not guarantee lower peak memory or faster execution. The direct CUDA
path removes that expansion. Benchmark preparation, packing and the surrounding
pipeline separately on the intended workload; do not multiply independently
measured speedups or call a warm-table pipeline timing production throughput.
