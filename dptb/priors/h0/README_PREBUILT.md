# Native installation

Run `python -m dptb.priors.h0 precompile` explicitly in the compatible Linux
environment, followed by `python -m dptb.priors.h0 precompile --check`.
The convenience `build_once.sh` uses the same module entrypoint under a lock.
Set `H0_WORK_ROOT` to an external writable task directory for build products
and caches. Environment, headers, and library discovery are described in
[README.md](README.md).

Native sources are wheel resources in `dptb.priors.h0/csrc`. The installation
builder places the verified `_cuda_two_center.so` and `_cuda_local_grid.so`
alongside the engine and records their manifests in `prebuilt/`. The binary
and manifests must match the actual Python/PyTorch/CUDA ABI, source contents,
dependencies, and target device. Package installation must permit those
explicit installation-time writes.

Ordinary imports and inference never launch a compiler. An existing manifest
cannot be made current by renaming its paths or changing its stored hashes.
Keep immutable old installations intact and create a fresh qualified
installation when rebuilding for a new identity.
