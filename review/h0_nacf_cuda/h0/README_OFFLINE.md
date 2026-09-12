# Reviewed H0 offline runtime, revision 4; offline schema v2

The Liyue installation is `/home/mingkang_nt/codex/h0_review_fixes_20260913/h0`.
Source `env.sh` to select the pinned Python/Torch/CUDA environment and `H0_OFFLINE_TABLE_DIR`.
`python precompile.py --check` verifies existing binaries without importing a compiler helper.
The installed kernels contain sm80/86/89/90 cubins. Numerical verification is on L40S (sm89).
An ABI or unsupported architecture mismatch fails clearly; it never compiles during inference.

## Prepared data

`offline_tables/catalog.json` binds the original 100 structures to 59 distinct UPF/ORB inputs.
`offline_tables/species` contains parsed FP64 species arrays and exact AO cubic coefficients.
`offline_tables/two_center` contains exact S/T/projector cubic tables, Gaunt data, descriptors
and scalar/SOC projector matrices. The complete prepared store currently occupies about 8.9 GiB.
Tables are prepared for each composition/spin/radial-grid identity. A new composition may need
explicit table preparation once, even when its individual elements are already present.

`prepare_tables.py RAW_DIRECTORY STORE_DIRECTORY` explicitly prepares an immutable cohort.
Source files are read during that step. Runtime loads by immutable species IDs and never reads
UPF/ORB or invokes SBT/tabulation. Geometry, FFT-grid-dependent radial transforms, total atomic
charge, Hartree/PBE fields and AO spatial contractions remain online.

Generic APIs are `prepare_species(orb_path, upf_path, store)`, `load_species(store, identity)`
and `prepared_two_center(..., prepare=True)` for explicit installation/preparation. The full
runtime is `assemble_h0(..., cuda_precompiled=True, offline_table_dir=store)`. The environment
variable supplies the same directory by default. `production_io.load_case` is a validation
reader; its ABACUS log/CSR inputs are not required by the generic geometry-only assembly API.

The source contract binds numerical Python dependencies, the table key binds exact arrays,
spin, radial settings and native binary, and the prebuilt loader binds native source and ABI.
Changed numerical dependencies require an explicitly prepared new store. Existing snapshots
are immutable. Runtime does not silently regenerate or repair them.

## Failure corrections

* `PP_BETA.10 index="*"`: use validated contiguous numbered tags, as ABACUS does. Preserve the
  malformed attribute in provenance and preserve strict attribute rejection on explicit request.
* Read per-atom STRU `mag`, overriding the species moment. Apply collinear initial moments to
  the complete spinor H0. Reject unimplemented transverse/angle inputs explicitly.
* For local fields only, apply ABACUS's `pseudo_rcut`/odd `msh` rule and normalize each atomic
  valence density before superposition. Preserve the full projector/orbital two-center inputs.
  Keep the requested total-electron rescaling afterward.

The old campaigns are preserved as historical evidence. `acceptance.py` is the
current finite two-GPU runner; new100/new100_v3 main entries delegate to it.
Results live under acceptance_v4/<identity>/cases/<case>/<attempt>. Only an accepted
terminal receipt plus a matching zero child return code can be reused. Source,
input, binary or table identity changes select a new run. Paused workers are
excluded from active timeout accounting. Hmax < 5 meV and Smax < 1e-6 are unchanged.

The current focused verifier is tests_h0fast/test_review_boundaries.py. The older
check_offline.py is historical and asserts the superseded shared-object behavior.
Current cache hits deliberately return private buffers. Mn remains a numerical
failure; see the parent README for exact evidence and the running cohort status.
