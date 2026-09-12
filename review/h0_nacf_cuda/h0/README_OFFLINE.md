# Reviewed H0 offline runtime, revision 3

The Liyue installation is `/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate`.
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

The former 100-case CPU/direct versus hybrid campaign is stopped and preserved. `new100_v3.py`
is the current finite two-GPU, new-only campaign over the identical 100 structures. Its result
directory is `new100_v3`; each result records source identities, table identity, binary hashes,
input moments and cutoff. Acceptance remains Hmax < 5 meV and Smax < 1e-6. Full independent
FP64 parity is representative prior evidence, not a newly measured 100-case result.

`check_offline.py` verified exact fresh/disk coefficients for four scalar/SOC compositions,
reordered-species reuse, radial mutation rejection and a full H0 with UPF/ORB reads and
tabulation forbidden. Current acceptance still has unresolved failures; do not call this
100/100 scientific acceptance or complete SOC/non-SOC coverage before the campaign finishes.
