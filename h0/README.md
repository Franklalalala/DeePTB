# H0 reconstruction

This is the maintained standalone H0 subsystem in `0917-stable`. It retains
the reviewed numerical implementation through `4a069aec`; NACF is maintained
separately in `../dptb/nacf`, without an overlay copy.

Activate a matching Linux Python/PyTorch/CUDA environment with pyabacus, NumPy,
SciPy and the selected field backend installed, then from the repository root:

```bash
source h0/env.sh
export ABACUS_SOURCE_DIR=/path/to/matching/abacus/source
# If headers such as OpenBLAS are outside the environment include directory:
export H0_EXTRA_INCLUDE_DIRS=/path/to/extra/include
bash h0/build_once.sh
python h0/precompile.py --check
```

The build discovers ModuleNAO libraries from the imported pyabacus and uses the
active Python environment's include/lib directories. CUDA_HOME and library
search paths must already select that matching environment. Native builds are
explicit installation work; normal inference never invokes a compiler. The
repository contains source, not installed binaries or prepared data. A new
installation must create its own manifests; never relabel an old manifest.

`h0/env.sh` selects this checkout and puts temporary files and caches under
`H0_WORK_ROOT` (default `h0/work`). It does not select a GPU or activate a conda
environment. Choose CUDA_VISIBLE_DEVICES for the host before execution.

Prepare species and two-center data with `prepare_tables.py RAW STORE` or the
APIs in `h0rebuild.offline`. Source, spin, radial-grid, dependency and binary
identities remain mandatory. A changed store identity requires explicit
preparation in a new namespace. `compact_tables.py` converts an existing store
to the lossless shared-curve representation; see README_SHARED_RADIAL.md.

The geometry API is `h0rebuild.assemble.assemble_h0`, with the physical cutoff,
FFT shape, spin and species explicitly supplied. `h0rebuild.deeptb` validates
AO-block provenance and packing conventions for DeePTB. `production_io.py`
reads reference ABACUS output for validation only; it is not an inference input
requirement. The assembly reports its actual backend and timing scope.

Use focused tests under `tests_h0fast` for a changed boundary. Dataset-dependent
verification scripts and the finite `acceptance.py` runner are opt-in and need
the original external reference inputs; historical Liyue fixture paths in those
tests are not bundled assets. `new100.py` provides the shared case worker and
compatibility CLI; `new100_v3.py` delegates to the same acceptance runner.

The previous full-cohort receipts apply to their frozen sources and stores.
The signed-density PBE correction and pair/projector optimizations are included.
The later radial-grid alignment experiment is not enabled by changing defaults;
it has only a limited-case qualification. See ../docs/0917-stable.md.
