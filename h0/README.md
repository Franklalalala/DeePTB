# H0 reconstruction

This is the maintained standalone H0 subsystem in `0921-stable`. It retains
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

For complete H0 block reproduction, explicitly set
`pair_support='nonlocal_complete'`. The default `orbital_overlap` limits output
to overlapping orbital supports and can omit nonzero third-center nonlocal
blocks. `strict_reproduction=True` checks other numerical settings but does not
change this support selection; see [the hot-path example](README_HOT_PATH.md).

Use focused tests under `tests_h0fast` for a changed boundary. Dataset-dependent
verification scripts and the finite `acceptance.py` runner are opt-in and need
the original external reference inputs; historical Liyue fixture paths in those
tests are not bundled assets. `new100.py` provides the shared case worker and
compatibility CLI; `new100_v3.py` delegates to the same acceptance runner.

The previous full-cohort receipts apply to their frozen sources and stores.
The signed-density PBE correction and pair/projector optimizations are included.
The later radial-grid alignment experiment is not enabled by changing defaults;
it has only a limited-case qualification. See ../docs/0917-stable.md.

## Explicit SOC reference alignment

For the qualified `USE_NEW_TWO_CENTER` reference, import
`h0rebuild.soc_reference.align_soc_projectors` and `SOC_REFERENCE_DR_BOHR`.
The reference uses `cutoff=2*rmax` and `nr=int(rmax/0.01)+1`, giving about
0.02 Bohr spacing, together with ABACUS's odd projector sample-count rule.
This is a numerical reference convention, not a change to the H0 formula or
a general recommendation to coarsen integration grids.

`SOC_REFERENCE_DR_BOHR = 0.02` does not guarantee an identical grid for every
cutoff. The reference's actual spacing is `2*rmax/(nr-1)`, whereas the CUDA
table builder uses `nr=ceil(2*rmax/requested_dr)+1`. At `rmax=9` both give
901 points; at `rmax=5.005` the reference gives 501 points (0.02002 Bohr) and
the current 0.02 request gives 502 points (about 0.01998004 Bohr). Check the
actual grid and reference for each new cutoff before extending qualification.

The reference is ABACUS commit
[`ee99e3ca`](https://github.com/deepmodeling/abacus-develop/tree/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f).
[`setup_nonlocal.cpp:111–142`](https://github.com/deepmodeling/abacus-develop/blob/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f/source/module_cell/setup_nonlocal.cpp#L111-L142)
sets `cut_mesh = ir`, rounds an even value up, and copies `ir < cut_mesh`.
An odd last nonzero index is therefore excluded intentionally; the UPF reader's
inclusive support-count convention is different. The
[`USE_NEW_TWO_CENTER` entry](https://github.com/deepmodeling/abacus-develop/blob/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f/source/module_hamilt_lcao/hamilt_lcaodft/LCAO_init_basis.cpp#L56-L71)
passes those projectors to the
[`two-center grid`](https://github.com/deepmodeling/abacus-develop/blob/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f/source/module_basis/module_nao/two_center_bundle.cpp#L62-L91).

The helper retains the original `cutoff_radius` as a conservative support and
grid envelope; resetting it to the new endpoint could change the universal
grid as well as neighbor selection. After alignment it need not equal
`r[cutoff_index - 1]`. For an all-below-threshold projector, the sample count
is bounded by the available mesh, including an even mesh. This safe fallback
does not reproduce the reference's out-of-range count.

Apply the helper **once to original species**, after loading geometry and
physics options. Use the same corrected species and spacing in both stages:

```python
from h0rebuild.soc_reference import align_soc_projectors, SOC_REFERENCE_DR_BOHR
from h0rebuild.offline import prepared_two_center
from h0rebuild.assemble import assemble_h0

# structure, original_species, physics_options are already loaded; nspin=4.
assert physics_options['nspin'] == 4
species = align_soc_projectors(original_species)
table = prepared_two_center(
    species, store=new_store, dr_bohr=SOC_REFERENCE_DR_BOHR,
    nspin=4, device='cuda:0', prepare=True,
)
print(table.metadata)  # actual radial spacing and table identity
del table  # preparation is outside assembly timing
result = assemble_h0(
    structure, species, **physics_options,
    two_center_backend='pyabacus', two_center_dr_bohr=SOC_REFERENCE_DR_BOHR,
    offline_table_dir=str(new_store), **runtime_options,
)
```

Keep the existing CUDA/FFT runtime options and use a **new store**; old table
keys describe different projector samples and spacing. Defaults, the generic
`prepare_tables.py` CLI and existing frozen stores retain their legacy route.
The input species are not mutated. Keep raw inputs for future preparations;
do not repeatedly apply the cutoff transformation to corrected species.

The original worst SOC case, `SOC_mp-754958`, was rechecked with four separately
prepared stores in the same installed CUDA reconstruction runtime. Against the
original ABACUS initial-H0/S CSR, the maximum matrix errors were:

| Two-center spacing | Projector alignment | H0 max error (meV) | S max error |
|---|---|---:|---:|
| 0.01 Bohr | Original | 0.272390 | 1.25393e-8 |
| 0.02 Bohr | Original | 0.012519 | 4.99835e-10 |
| 0.01 Bohr | ABACUS sample rule | 0.028457 | 1.25393e-8 |
| 0.02 Bohr | ABACUS sample rule | 0.012519 | 4.99835e-10 |

For this case, spacing alone reaches the reported combined maximum error;
the combined number does not establish an additional gain from alignment at
0.02 Bohr. Alignment also reduces the error at 0.01 Bohr and reproduces the
specified reference's projector rule. These are single-case implementation
checks, not full-cohort precision acceptance, quadrature convergence, speedup,
or the stored-H0 prior's MAE against the final converged Hamiltonian. Convert
H from Ry to eV before reporting errors.
