# Shared radial tables and projector search

The shared store deduplicates identical S/T/Q curves across combinations.
It stores uniform cubic Hermite endpoints, tail coefficients, and sparse bit
corrections. Original grids, coefficients, species, complex SOC D matrices,
indices, and generator contracts remain intact. Not-a-knot AO splines retain
their original format instead of using the Hermite codec.

```bash
python -m dptb.priors.h0 compact-tables /path/to/original /path/to/new/shared
```

The source is read-only. A discoverable manifest is published only after every
decoded combination passes its original state checksum. Incomplete output is
retained for diagnosis; retry in a new directory. Runtime discovers complete
stores through `shared_radial.json` and expands original FP64 tensors. SQLite
is read-only, decoded coefficient bytes participate in hashing, and damaged
records or incompatible floating-point arithmetic fail verification.

```python
result = assemble_h0(
    ..., offline_table_dir='/path/to/new/shared',
    projector_reuse_max_mb=256,
    projector_search='anchor', spatial_backend='indexed',
)
```

The default search is `midpoint`. `anchor` uses atom `ci` and radius
`orbital_cutoff_i + max_projector_cutoff`: a contributing projector must overlap
AO i and therefore lies inside this candidate sphere. All outgoing edges of
atom i reuse sorted candidates; the original two species-distance tests and
native contraction decide actual contributions, including repeated periodic
images. Atom/translation order remains unchanged and only current-atom
candidates are cached.

Compaction applies to already prepared curves; unseen grids and combinations
still require explicit preparation. CUDA uses expanded four-coefficient
tables, so disk compression does not imply equal GPU memory savings. The Q
cache budget covers retained Q tensors, not the whole assembly working set.
`kernel_nonlocal_seconds` includes the wrapper through completed CPU results;
S/T call counts and timings are reported separately.
