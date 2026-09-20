# Explicit CUDA inference fusion

The default `NACFTableBank` and existing plans are unchanged. Opt in after
preparing a geometry, or a merged geometry batch:

```python
from dptb.nacf.fusion import enable_fusion
plan = bank.prepare(**geometry, topology="native")
enable_fusion(plan, radial=True, contraction=False)
blocks = plan()
```

`radial=False` and `contraction=False` independently retain the reference paths
on a fresh plan. To disable a previously enabled radial plan, create a fresh
geometry plan. SOC complex projector contractions retain the Torch path.
The CUDA contraction uses atomic addition; compare float64 outputs with absolute
tolerance 1e-12 eV, not bitwise equality. Fused radial results preserve each
table's existing spline and angular arithmetic. No numerical table is refitted.

Contraction is disabled by default in this opt-in helper: the 100-case audit
found FP64 AO agreement within 1e-12 eV, but two FP32 packed outputs crossed a
rounding midpoint (maximum 1.164e-10 eV). `contraction=True` is available for
explicit evaluation; it is not accepted under the strict end-to-end FP32 gate.
This is a uniform implementation switch, never a material-specific choice.

`RadialMultiPlan([(tables, query_count), ...])` accepts concatenated displacement
groups. Tables in one group share angular metadata and a single rotation.
Output views are ordered by group, then by table. P2 and overlap share their
query group. Pair-XC can use one heterogeneous multi plan for all species pairs
and directions; the caller preserves sign and transpose conventions.
`background_nodes` reserves the 2-D interface and currently raises
`NotImplementedError`. Rebuild plans after moving or replacing table buffers.

`density_topology` returns native atom and edge CSR lists in integer image
coordinates. Density queries use the neighbour's image sign, unlike the
projector/VNA AO-image sign. The query ball radius must enclose all probes for
every edge starting at the query atom. The search cutoff is query radius plus
that neighbour's species density support. Origin self is excluded, other self
images remain. Edge CSR excludes exactly the second endpoint image as well.

`onsite_density_neighbors` builds every atom's list together and explicitly
adds the origin density. `legacy_radius=27.` retains the accepted fixed-radius
truncation and species/atom/lexicographic-image order, including zero density
entries. Omitting this option uses physical support bounds; this is an explicit
caller decision, never a silent change to an accepted fixed-radius calculation.

`density_sum_cuda` evaluates the existing piecewise cubic valence/NLCC channels,
clamps each channel to nonnegative values, clips at support, and retains the
16-neighbour chunk order. Its inputs expose `knots` and a list of `[4,K-1]`
coefficient tensors. `probe_environment_density` is the Torch F4 reference:
`[E,P,3]` probes, CSR neighbour positions/species, density callables and optional
nonnegative normalized weights return `[E]`. Caller P supplies probes, weights,
endpoint exclusion and density conventions; this module chooses no physics.

Build explicitly with `python -m dptb.nacf.precompile --arch 8.9+PTX` and
`python tools/build_nacf_topology.py --output /isolated/libnacf_topology.so`.
Inference loads checksummed binaries and never invokes a compiler. The build
retains `--fmad=false`; no fast-math flags are introduced.
