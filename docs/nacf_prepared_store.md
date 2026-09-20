# Immutable prepared radial stores

`NACFTableBank(..., prepared_store=PreparedRadialStore(path))` loads a published
radial snapshot without runtime access to dense source table arrays. The source
P2, P23 and overlap manifest files remain required: the store binds their hashes
and the radial implementation hash before use. Missing tables, incomplete
manifests, corrupt payloads or an incompatible implementation fail explicitly.
The default bank behavior is unchanged.

Prepared stores and `prepared_cache_dir` (including `DPTB_NACF_PREPARED_DIR`) are
mutually exclusive. The latter is an opportunistic cache; the former is an
immutable artifact built and published separately. Existing evaluation adapters
no longer need to subclass `NACFTableBank` to inject the snapshot loader.

Version 2 uses either exact coefficients or compact nodal values. It selects
nodal encoding only when reconstructing coefficients reproduces the original
FP64 buffers bit for bit. Cold loading performs the same one-dimensional spline
reconstruction and verifies its coefficient hash; this is not spatial quadrature.
Version 1 coefficient snapshots remain readable. A SciPy arithmetic change can
require explicit regeneration even when the underlying physical tables match.

The implementation hash covers exact file bytes, including line endings. NACF
source files have an LF Git attribute so Windows and Linux checkouts agree.
Keep historical source/binary/manifest sets together; changing an old deployed
file's line endings invalidates its receipt. Do not bypass this check or edit an
immutable store's manifest to fit a different implementation.

Disk compression does not imply an equal reduction in resident GPU memory.
Report cold loading separately from warm table, fresh geometry inference.
Preparing a store does not validate a physical prior, a new dataset, or a model
checkpoint. The mixed prior and canonical graph contracts remain unchanged.
