# NACF acceleration transferred from H0

This isolated candidate starts from the current SOC-integration source, including configurable
Ry/eV units and manifest-bound missing-P23 policy. Running label/training snapshots were not replaced.

Source `env.sh` on Liyue. `python -m dptb.nacf.precompile --check` returns REUSED for the installed
sm80/86/89/90 binary. `python -m dptb.nacf.precompile` is an explicit installation build if needed.
Native inference never invokes NVCC, Ninja or torch.cpp_extension.load. ABI/source/binary/GPU
checks reject incompatible packages. The current binary was compiled against Torch 2.8.0+cu128
and CUDA 12.8, with `-O3 --fmad=false`; only L40S has been numerically verified in this update.

`DPTB_NACF_PREPARED_DIR` (set by env.sh) or `NACFTableBank(..., prepared_cache_dir=...)` persists
GPU-ready radial buffers, exact spline coefficients, rotation bases and contraction packing.
Each source table still passes the normal source-store loading/checksum and P23 coverage gate.
Repeated preparation then restores those buffers without QR, spline/rotation metadata rebuilding
or Python orbital-packing loops. Warm numerical forward remains entirely on existing GPU tensors.
The cache is FP64 master data and supports explicit caller FP32 conversion. A changed source
table/cubic polynomial, support, shells or radial implementation selects a different identity;
corrupt compiled cache files raise. No approximate refitting is introduced.

The physical recipe remains node=P23 onsite, edge=P2 hopping, H=prior+residual. Complex SOC
projector matrices, canonical directed periodic edges, near-south-pole rotations, units and
missing/corrupt P23 behavior are preserved. H0's FFT/Hartree/PBE field calculation is not added
to the NACF prior.

Verification: 30 relevant tests passed (prebuilt/cache, graph/periodic assembly, P23 policy,
SOC and spinor inference). After enabling the default cache environment, the three new cache
tests passed again. They include nondefault CUDA stream, exact cold/disk forward equality,
native-versus-Torch p/d rotations, source-change selection and corruption rejection. This update
does not claim a measured production throughput multiplier or a new full physical label audit.
