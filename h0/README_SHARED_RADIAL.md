# Shared radial tables and anchored projector search

The shared store removes exact duplicate S/T/Q curves across compositions and
stores uniform cubic Hermite endpoints plus tail coefficients and sparse bit
corrections. It preserves the original numerical grid, coefficients, species,
complex SOC D matrices, index maps, and generator contract. Not-a-knot AO splines
are retained as originally stored, not converted with the Hermite codec.

Use the installed H0 environment and explicitly prepare a new directory:

```sh
python -B compact_tables.py /path/to/original/offline_tables /path/to/new/shared_tables
```

The original store is never modified. A runtime-discoverable manifest is published
only after every reconstructed composition passes its original state checksum.
An incomplete destination is retained for diagnosis; retry with a new directory.
The runtime detects a completed shared store via `shared_radial.json` and expands
the original FP64 tensors on load. It never generates missing tables or compiles.
SQLite is opened read-only. Decoded coefficient bytes are hashed, so incompatible
floating-point arithmetic or damaged records fail rather than silently diverge.

```python
result = assemble_h0(..., offline_table_dir='/path/to/new/shared_tables',
                     projector_reuse_max_mb=256,
                     projector_search='anchor', spatial_backend='indexed')
```

`projector_search='midpoint'` remains the default and provides the A/B reference.
The anchor route queries around `ci` with radius `orbital_cutoff_i + max_projector_cutoff`.
Every contributing projector must lie in this ball because it must overlap AO i.
All outgoing pairs of i reuse the sorted candidate superset. The original two
per-species distance tests and native contraction still determine actual
contributions, including repeated periodic images. Atom-major / translation order
is unchanged. Only one atom's candidate list is retained, so the search cache does
not grow with the full edge list. Reference spatial mode is unaffected.

Scope: this is shared storage of already generated curves, not a new universal
grid or a generator that skips all repeated pair integrals. New numerical grids or
unseen compositions still need explicit preparation. CUDA still holds expanded
four-coefficient tables; the disk reduction is not a claim of reduced GPU memory.
The Q cache budget still covers retained Q tensors only, not the complete assembly
working set. `kernel_nonlocal_seconds` on the reuse route now measures the wrapper
through the completed CPU result; S/T timing and call counts remain visible.

Validation and run-specific performance numbers belong in the dated evidence
bundle, not in this API contract. Native binaries are unchanged.
