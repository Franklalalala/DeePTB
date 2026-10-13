# Offline physical H0 prior

`dptb.priors.h0` is the authoritative engine for the atomic-charge operator
`H[rho0] = T + Vnl + Vion + VH[rho0] + Vxc[rho0]` and overlap S. It is an
offline physical prior subsystem, independent of neural model embeddings.
Compare it with the initial operator built from identical physical inputs.
NACF remains in `dptb.nacf`.

## Commands and installation

The engine ships in the DeePTB wheel, including native CUDA source resources.
The CUDA binaries are built explicitly in a compatible Linux Python/PyTorch/
CUDA/pyabacus environment; importing H0 never invokes a compiler.

Numerical reconstruction requires Python 3.10 or newer and SciPy 1.15 or newer
for `scipy.special.sph_harm_y`. This optional offline-engine requirement does
not change the core DeePTB Python 3.9 support. Command help, package resources,
and precomputed-prior model inference do not import the reconstruction engine.

```bash
export H0_WORK_ROOT=/path/to/external/h0-work
export H0_OFFLINE_TABLE_DIR=/path/to/prepared/offline_tables
export ABACUS_SOURCE_DIR=/path/to/matching/abacus/source
export H0_EXTRA_INCLUDE_DIRS=/path/to/extra/include
python -m dptb.priors.h0 precompile
python -m dptb.priors.h0 precompile --check
python -m dptb.priors.h0 prepare-tables /path/to/raw /path/to/new/store
python -m dptb.priors.h0 reconstruct config.json -o H0_blocks.npz
```

The main DeePTB CLI exposes the same commands as `dptb h0 <command>`.
`h0rebuild config.json` remains a reconstruction command. The Python API is:

```python
from dptb.priors.h0.api import reconstruct_from_files
from dptb.priors.h0.assemble import assemble_h0
from dptb.priors.h0.offline import prepared_two_center
```

The optional `env.sh` configures temporary directories and caches beneath an
explicit `H0_WORK_ROOT`; it does not activate an environment or select a GPU.
It no longer modifies `PYTHONPATH`. Build discovery uses the currently imported
pyabacus ModuleNAO, the current Python include/lib directories, and the explicit
ABACUS source tree. Configure `CUDA_HOME` and dynamic-library search paths for
that same environment. See [native installation](README_PREBUILT.md).

## Package migration

The former root `h0/h0rebuild` engine is now `dptb/priors/h0`. Offline commands
are importable modules under `dptb.priors.h0.cli`; the explicit native builder is
`dptb.priors.h0.precompile`. Tests moved to `dptb/tests/h0`.

Legacy `h0rebuild.<engine-module>` imports lazily resolve to the canonical
module object, preserving class identity. `h0rebuild.cli` resolves to the
reconstruction command through a thin forwarding module, including
`python -m h0rebuild.cli config.json`. The shim contains no physical implementation.
Filesystem script paths such as `python h0/precompile.py` and
`python h0/prepare_tables.py` migrate to the module commands above. The finite
reference acceptance command is `python -m dptb.priors.h0 accept`.

Artifact schemas, feature keys, units, spin packing, and numerical defaults
retain their existing meanings. Native manifests and prepared-table contracts
hash their actual source files; source edits invalidate old claims. Explicitly
build a new installation and prepare a new store when verification rejects a
changed identity. Do not rename or restamp old manifests.

## Physical and artifact contracts

`assemble_h0` receives structure, species, physical cutoffs, the exact FFT grid,
spin, and initial magnetic moments. `dptb.priors.h0.deeptb` validates provenance,
energy units, structure identity, graph coverage, and real-space Hermiticity
before DeePTB packs complex AO blocks. The bridge itself imports neither
PyTorch nor neural model code. `production_io.load_case` reads ABACUS physical
inputs and a separate post-construction CSR reference.

Use `pair_support='nonlocal_complete'` for complete nonlocal block support.
The default `orbital_overlap` can omit a nonzero third-center nonlocal
contribution. `strict_reproduction=True` does not change that support choice.
See [hot-path behavior](README_HOT_PATH.md) and [offline stores](README_OFFLINE.md).

`output_atom_cell_shifts` specifies integer lattice shifts. If
`r_out_i = r_in_i + q_i @ cell`, H0/S/component keys transform as
`R_out = R_in + q_i - q_j`; serialized coordinates and their fingerprint change
with them. Matrix values and periodic fields retain their physical meaning.
The default retains input coordinates. `cell_gauge` checks integer shifts and
rebases results; reference input rejects true displacements or reordered atoms.

## Explicit SOC reference alignment

`soc_reference.align_soc_projectors` and `SOC_REFERENCE_DR_BOHR` implement the
ABACUS `USE_NEW_TWO_CENTER` reference contract. The reference grid uses
`cutoff=2*rmax`, `nr=int(rmax/0.01)+1`, and odd projector sample counts.
The reference is [ABACUS ee99e3ca](https://github.com/deepmodeling/abacus-develop/tree/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f).
Its projector truncation rounds even `cut_mesh` upward and copies only samples
`ir < cut_mesh`, unlike the UPF reader's inclusive support count.

A nominal 0.02 Bohr request does not guarantee identical grids for every
cutoff: the reference uses `2*rmax/(nr-1)`, whereas CUDA preparation uses
`nr=ceil(2*rmax/requested_dr)+1`. At `rmax=9`, both have 901 points; at
`rmax=5.005`, the reference has 501 points and a 0.02 request has 502. Check
the actual grid for each new cutoff.

Apply alignment once to original species, and use that species and spacing in
both preparation and assembly:

```python
from dptb.priors.h0.soc_reference import align_soc_projectors, SOC_REFERENCE_DR_BOHR
species = align_soc_projectors(original_species)
table = prepared_two_center(
    species, store=new_store, dr_bohr=SOC_REFERENCE_DR_BOHR,
    nspin=4, device='cuda:0', prepare=True,
)
result = assemble_h0(
    structure, species, **physics_options,
    two_center_backend='pyabacus', two_center_dr_bohr=SOC_REFERENCE_DR_BOHR,
    offline_table_dir=str(new_store), **runtime_options,
)
```

The original projector `cutoff_radius` remains a conservative support/grid
bound and can differ from `r[cutoff_index-1]`. An all-below-threshold projector
uses the available mesh to bound sample count. Preserve original species to
avoid double alignment. Existing frozen stores and generic preparation do not
automatically opt into this reference-specific transformation.

## Validation

Run `python -m pytest dptb/tests/h0` for the affected boundary. CUDA tests need
verified matching binaries; real reference tests additionally need
`H0_REFERENCE_ROOT` and pyabacus. Reference assets remain external.

Compare H0, S, and physical components, including keys, shapes, dtype, and cell
gauge. Convert Ry energies to eV before reporting errors. Reference alignment
alone establishes neither full-dataset accuracy, integration convergence,
training prior accuracy, nor end-to-end model acceleration.
