# Offline stores

Prepare species and two-center radial tables explicitly with
`python -m dptb.priors.h0 prepare-tables RAW STORE`, or the
`dptb.priors.h0.offline` API. Installation and assembly are described in
[README.md](README.md).

Each store binds source contents, spin, radial grid, numerical generator,
dependencies, and native binary identity. Inference reads a matching prepared
store; it never repairs a missing table or compiles a backend. When a contract
changes, prepare a new directory instead of changing the old manifest.
`H0_OFFLINE_TABLE_DIR` selects the store for reference commands and defaults to
`H0_WORK_ROOT/offline_tables` when the work directory is explicitly configured.

The species codec is pickle-free and stores allowed dataclasses by their class
name and fields. Moving their Python module does not alter the file format.
Generator source edits still invalidate the strict numerical source contract.
See [shared radial stores](README_SHARED_RADIAL.md) for lossless compaction.
