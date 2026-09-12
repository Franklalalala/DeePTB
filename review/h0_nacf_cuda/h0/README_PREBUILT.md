# CUDA H0 precompiled candidate

This candidate combines the corrected atom-cell-gauge core with the reviewed two-center and local-grid CUDA extensions. It supports a complete single-structure H0 build through the existing API. Cross-structure true batch runtime is not included: Gemini revision 789f950 still has open review findings.

On Liyue, use the existing Python 3.10 / Torch 2.8.0+cu128 environment:

```bash
cd /home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate
source env.sh
python precompile.py --check
```

`--check` only validates and prints REUSED. Normal API calls never invoke `nvcc`, `ninja`, setuptools, or torch's JIT extension loader. Both `.so` files are already installed in `h0rebuild/`; their source/runtime/dependency manifests are in `prebuilt/`. Changing species, coordinates, batch composition or starting another Python process does not change the native binaries. Species radial preparation is runtime data work, not CUDA compilation.

```python
from h0rebuild.assemble import assemble_h0
result = assemble_h0(
    structure, species_data,
    ecutrho_ry=production_cutoff, fft_shape=production_fft_shape,
    xc="PBE", nspin=1,  # use nspin=4 for spinor SOC
    two_center_backend="pyabacus", local_integration="fft_grid_periodic",
    compute_device="cuda:0", field_backend="torch", cuda_precompiled=True,
    structure_factor_backend="cufinufft", strict_reproduction=False,
    pair_support="nonlocal_complete", spatial_backend="indexed",
)
```

Pass the real physical cutoff/grid and species inputs. This uses the existing controlled cuFINUFFT approximation at the default 1e-12 tolerance, with strict numerical comparison against the CPU direct reference; the exact production grid is preserved. `strict_reproduction=False` permits that explicit backend, not a reduced acceptance threshold. Generic geometry inference does not use oracle H/S or reference logs. `production_io.load_case` is only the validation reader for existing ABACUS cases. The call returns all H0 components, S and core cell-gauge metadata. The measured backend is recorded as `cuda_precompiled`.

The fatbins contain sm_80, sm_86, sm_89 and sm_90. Numerical validation in this task is on Liyue L40S (sm_89); this is not numerical qualification of other hardware/environments. Two-center accepts orbital and projector l<=4. Unsupported channels raise explicitly. The pyabacus private-layout adapter is tied to the hashed installed libnaopack and Python extension; this package is not a portable arbitrary-ABI wheel.

Only a native source/ABI/dependency change requires an explicit installation-time build:

```bash
bash build_once.sh
```

Unchanged installation calls reuse the binaries without entering the compiler. Builds are serialized by a dedicated precompile lock. GPU verification uses the existing shared GPU1 lock; GPU0 and the original 100-structure benchmark are preserved.

Run the existing verification entrypoint when reviewing a changed numerical boundary:

```bash
bash verify_once.sh
```

The fresh process denies compiler execution, tests multiple species/spin contracts, strict two-center error limits and local-grid operations on a non-default stream. `work/full_verification.json` records separate complete H0/S checks against the original CPU implementation and raw ABACUS. Single-pass diagnostic runtimes are not steady-state acceleration benchmarks.
