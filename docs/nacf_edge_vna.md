# Grid-free edge VNA

`NACFTableBank.prepare_edge_vna` and `prepare_edge_vna_batch` provide an explicit
third-centre local-potential correction to P2 hopping. Existing `prepare()` and
the default P23-onsite/P2-hopping recipe are unchanged. The correction is

`delta V(i,j,R) = sum_(k,t) Q(k,i,-t).T diag(epsilon[k]) Q(k,j,R-t)`.

The sum excludes `(k,t)=(i,0)` and `(j,R)` because P2 already contains endpoint
VNA. It retains periodic self images. Nonlocal pseudopotential terms already in
P2 must not be added again. Outputs are scalar AO blocks in eV in the original
directed graph order; SOC consumers must explicitly lift the scalar correction
onto both spin-diagonal blocks before their existing basis/target conversion.

## Build and use

Build on the target machine with a C++20 compiler. The standalone library has no
Torch ABI dependency and requires no installed Tonari package. Inference never
downloads sources or compiles a binary.

```bash
python tools/build_nacf_topology.py --output /your/build/libnacf_topology.so
export DPTB_NACF_TOPOLOGY_LIBRARY=/your/build/libnacf_topology.so
```

```python
plan = bank.prepare_edge_vna(
    symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift,
    pbc=pbc, chunk_bytes=32 * 1024 * 1024,
)
delta = plan()["edge_vna_ao_ev"]
```

For batches, pass dictionaries with the same argument names to
`bank.prepare_edge_vna_batch(geometries)`. `edge_ptr` partitions the returned
tensor and `plan.edge_slices` gives the same boundaries as host integers.
The output is padded to the largest AO width in the batch. Prefer batches with
similar basis widths/compositions; heterogeneous padding can erase the benefit.
Reuse the caller-owned table bank across plans. Rebuild a plan for each changed
geometry: positions, cell, periodicity, species and graph are bound at creation.

## Method and storage contract

Preparation and forward do not use structure-dependent density/potential grids
or real-space quadrature. The neighbor core may use spatial bins to enumerate
atoms; these are a search data structure, not an electronic density grid.
Offline atomic integration and one-dimensional radial knots are unchanged.

The edge correction reuses the existing ordered `(centre species, AO species)`
P23 factor tables. No species-triplet table, two-distance/angle tensor or new
radial bank is created. Missing composition coverage is an error even if the
bank's historical P2 fallback policy is enabled. Complete missing pair tables
in an immutable version before using such compositions.

Factorization rank, angular truncation, physical potential support and the
tabulated projector support remain the original table approximation. CPU/GPU
agreement tests implementation consistency, not equivalence to exact VNA or H0.
The new component contains no XC and is not guaranteed to improve every system.
Physical recipes must be compared on corrected geometry and a fixed validation
set; do not choose recipes per structure using target H correlations.

## Optimization and memory

The vendored neighbor core enumerates a full directed auxiliary center graph,
followed by the original directional AO/centre support filter. Our C++ integer
intersection selects one representative of each original reverse-edge pair.
Unique factor queries are grouped by species pair, and contractions are grouped
by species triple. Optional `prune_unused=True` removes unused queries by linear
marking; it is disabled by default because dense tested graphs reused nearly all
queries and compaction cost more than it saved. Only reverse
outputs are transposed. The label graph, its shifts and row order are retained.

Batching groups numerical work across graphs. The contraction uses diagonal
epsilon scaling and limits its estimated scratch with `chunk_bytes`; the
estimate excludes resident tables, factor cache, output, allocator overhead and
autograd-saved tensors. One indivisible term may exceed a smaller requested
budget. `max_terms` limits batch contraction-term storage and fails explicitly
on excess; it does not bound the preceding neighbor broad phase. These APIs
target inference/label generation. Run under `torch.no_grad()` when appropriate.

Benchmark preparation and synchronized forward separately, after warming the
shared bank. Report LMDB I/O, packing and writing separately until measured as
part of production throughput. Kernel-only timing does not establish that rate.

## Attribution and validation

See [the Tonari acknowledgement](../dptb/nacf/THIRD_PARTY.md) and the vendored MIT
license. Tonari's Python wrapper, Torch bindings and CUDA provider are not used.

```bash
python tools/test.py dptb/tests/test_nacf_edge_vna_native.py
```

The native tests compare integer topology with independent lattice enumeration
and values with an analytic NumPy oracle, including partial PBC, skew cells,
changed cell representatives, periodic self images, strict support boundaries,
batching, reversed/permuted edges, empty graphs and invalid-input rejection.
Real-data CPU/CUDA parity and throughput experiments are separate from this
small regression suite.
