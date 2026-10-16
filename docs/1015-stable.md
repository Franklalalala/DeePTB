# `1015-stable` release

This release moves the two physical priors and inference onto the GPU. Starting
from geometry (atomic numbers, positions and cell in Å), a structure's prior,
model input, model output and final H, S (plus the UniTB residual) stay in
device memory end to end. Nothing is written to disk and nothing is copied back
to the host. The prior table generators run on the GPU behind one entry point.
Configuration names, checkpoint formats, table formats and numerical behaviour
are unchanged: the priors use the same grids, cutoffs, tables and precision
(FP64), and the models still run in FP32 with TF32 off.

## Component layout

| Location | Purpose |
| --- | --- |
| `dptb/prior/common/` | Operators shared by both priors: periodic pair enumeration (`neighbors`), block tables and the ABACUS→DeePTB gauge (`blocks`), real spherical harmonics and rotations (`harmonics`), radial splines and bond-frame tables (`radial`), two-centre integrals (`twocentre`), spherical Bessel transforms (`sbt`), real Gaunt coefficients (`gaunt`), Kleinman–Bylander and SOC matrices (`kb`), ORB/UPF readers (`pporb`), PBE and PZ81 (`xc`), and the write-once table store (`store`) |
| `dptb/prior/gf/` | GF prior of the Flash route (`prior.GFPrior`), with its tables (`tables`), topology, assembly, onsite XC and environment XC; GPU builders `build_{p2,overlap,soc,p23,pairxc,envxc}` |
| `dptb/prior/h0/` | H0 prior of the UniTB route (`prior.H0Prior`): pair bank (`bank`), local potential on FFT-grid tiles (`field`, `local`), nonlocal projectors (`projector`), prepared records (`records`); GPU pair-bank builder `build`; reader of stored `h0rebuild` artifacts (`deeptb`, `io`) |
| `dptb/prior/build.py` | `dptb prior build`: every table family behind one command |
| `dptb/inference/` | `InferenceEngine`: graph, prior, model input, the two heads, Δ symmetrisation, H = prior + Δ, S and the UniTB residual, as device block tables (`engine`, `graph`, `features`, `heads`) |

## Module migration

| `1013-stable` | `1015-stable` |
| --- | --- |
| `dptb/nacf/*` (CUDA extension, Tonari topology, candidate plans, prepared stores) | `dptb/prior/gf/` (Triton and batched torch on the device; no compiled extension) |
| `dptb/priors/h0/*` (per-pair assembly, support caches, pylibxc, pyabacus tables) | `dptb/prior/h0/` (tile integration, per-centre nonlocal, GPU PBE, GPU tables) |
| `dptb h0 ...` | `dptb prior build h0 ...` |
| `tools/build_nonsoc_p2_tables.py`, `build_nonsoc_p23_tables.py`, `build_overlap_sidecar.py`, `build_nacf_envxc_tables.py` | `dptb prior build p2 / p23 / overlap / envxc` |
| `examples/nacf_gpu`, `docs/nacf_*.md` | this document |

Stored names are converted only where they are read:
- checkpoints go through `dptb.nn.checkpoint.load_checkpoint`;
- the legacy schema ids `nacf-envxc/v1` and `deeptb.nacf.pair_xc/v1` of deployed table manifests go through `dptb.prior.gf.tables.LEGACY_SCHEMAS`;
- the training-record fields `node_delta_nacf` / `edge_delta_nacf` and the target kind `nacfres` keep their stored names in `dptb/data`.

Stored data and tables are never rewritten.

## Inference

```python
from dptb.prior.gf.prior import GFPrior
from dptb.prior.h0.prior import H0Prior
from dptb.inference.engine import InferenceEngine

prior = H0Prior(bank_dir)            # UniTB; Flash: GFPrior(tables_dir, soc=..., pairxc=..., moments=..., pporb=...)
engine = InferenceEngine('unitb', prior, on, hp)    # on, hp: the onsite and hopping heads (dptb.inference.heads)
out = engine.run(numbers, positions, cell, prefetch=next_numbers)   # out.H, out.S, out.residual: device block tables
```

`prefetch` reads the tables of the next structure's composition in a background
thread, so a cold composition costs no table I/O on the critical path.

Periodic QEq uses the Ewald direct solver by default in inference and training.
Set `DPTB_QEQ_SOLVER=dense` before importing the model to use the reciprocal
reference solver. Ewald preserves the Gaussian kernel, neutral charge constraint,
FP64 charge solve and gradients through learned electronegativity and hardness.
Small cells and kernel cutoffs unsuitable for splitting use the reciprocal sum.
Geometry derivatives apply within a fixed mode and periodic-image set.

Paired-head inference shares a periodic QEq matrix by default. Set
`DPTB_INFERENCE_SHARE_QEQ=0` before constructing the engine to disable sharing.
Sharing requires identical geometry values, precision, device and kernel options;
different inputs build separate matrices. Each head solves for its own charges
with its own hardness and electronegativity. The shared matrix belongs to one
`run` call, uses CUDA events to order readers across streams and is discarded
before the next structure. Sharing requires inference mode and never retains an
autograd graph. Independent training processes build their own kernels on each
forward; they do not share matrices across batches or trainers. Checkpoint
layouts and model parameters are unchanged.

## Device execution

Every step from geometry to the block tables stays on the GPU; nothing returns
to the host or is written to disk. The main fused paths:

| Step | Implementation |
| --- | --- |
| Radial tables (both priors) | `RadialTable.into`: knot search, spline, both rotations and the write into the padded block in one Triton launch per species pair; row-tiled for tables taller than 32 rows |
| Rotation matrices | `rotation_pack_of`: D^0..D^lmax of all displacements in one Triton launch |
| GF onsite XC | Triton density kernel (neighbours sorted by reach), Gram contraction as two GEMMs; runs in a worker thread on a low-priority stream beside the assembly |
| GF three-centre terms | Triton row kernel: each block accumulated in registers and written once |
| H0 local potential | FFT-grid tiles: AO values in a Triton kernel, one batched GEMM per tile; one cufinufft plan per grid shape, kept across structures up to 256^3 points |
| Heads | the edge head runs in a worker thread on its own stream beside the node head |
| Heads' SO2 layers | the route each checkpoint was trained with (`streamed_m_major_fused_p0`, `cublas_grouped` expert linears): SO2CUDA's activation-space kernels and grouped GEMM when the package is installed, otherwise the grouped PyTorch reference |

Prior mathematics is FP64; the heads run in FP32 with TF32 disabled.

### SO2CUDA

The heads use the optional SO2CUDA package (`so2_cuda_ops`, 0.3 series, entry
point `so2_cuda_ops.deeptb`) through `dptb.nn.so2_backend`. Install it from its
source tree (`pip install --no-deps <SO2CUDA>` or its `src` directory on
`PYTHONPATH`). Its two CUDA extensions are compiled on first use:

| Setting | Value |
| --- | --- |
| CUDA toolkit | `nvcc` with the CUDA major version of the PyTorch build (`CUDA_HOME`), host compiler `CXX` |
| `TORCH_CUDA_ARCH_LIST` | target architecture, `9.0` for H100/H200 |
| `SO2_CUDA_PACK_SCATTER_BUILD_DIR`, `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR` | persistent build directories, so later processes load the compiled modules |
| `SO2_CUDA_FAST_TF32=0` | strict FP32 GEMMs |

Leave `DPTB_SO2_ACTIVATION_FUSED_P0` unset (`0` disables the fused route) and
use `SO2_CUDA_BACKEND=off` to force the reference. `engine.run(...).info['so2']`
reports the backend and the dispatch counts of the structure.

## Building prior tables

```bash
dptb prior build h0 p2 overlap soc p23 pairxc envxc moments \
    --out EXT --reference TABLES_ROOT --h0-reference H0_BANK \
    --element X --pporb PPORB_DIR [--species-files X=X.orb,X.upf] [--missing-only]
```

Each family is written to `EXT/<family>` with its own manifest: shard SHA256,
generator identity, and the physical contract copied from the reference. A store
never replaces an existing file, and deployed tables are never written.
`--pairs 'A|B,...'` builds explicit keys; `--element X` builds every table
between `X` and the reference species.

Both priors can also build missing tables during inference. This is off by
default:

```python
H0Prior(bank_dir, extension=EXT, build_missing=True, pporb=[PPORB_DIR])
GFPrior(tables_dir, ..., extension=EXT, build_missing=True)
```

An extension root is stacked after the deployed tables, and the first layer that
holds a key wins. With `build_missing=True`, keys of a composition that no layer
holds are built into the extension before the structure runs. The extension is
reused afterwards with the switch off.

## Timings

One H200, one process, warm, median of three runs, outputs kept in device
memory, model forward on SO2CUDA. "1013" is the previous production pipeline
on the same card without writing to disk. TePb are rock-salt supercells;
GST225 and GSS4T1 are amorphous cells.

| Structure | Atoms | Flash 1013 (s) | Flash 1015 (s) | UniTB 1013 (s) | UniTB 1015 (s) | GF prior (s) | H0 prior (s) | Peak GiB Flash / UniTB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TePb 2³×2 | 16 | 0.74 | 0.166 | 5.8 | 0.188 | 0.036 | 0.054 | 0.8 / 3.3 |
| TePb 3³×2 | 54 | 1.64 | 0.183 | 19.9 | 0.240 | 0.052 | 0.104 | 1.5 / 3.6 |
| TePb 4³×2 | 128 | 3.31 | 0.244 | 47.4 | 0.366 | 0.104 | 0.222 | 3.0 / 5.2 |
| GST225 | 216 | 6.28 | 0.333 | 85.6 | 0.566 | 0.176 | 0.404 | 4.8 / 7.8 |
| TePb 5³×2 | 250 | 5.91 | 0.362 | 88.5 | 0.663 | 0.201 | 0.497 | 5.4 / 8.9 |
| GSS4T1 | 360 | 13.55 | 0.412 | 148.9 | 1.011 | 0.218 | 0.809 | 8.2 / 14.1 |
| TePb 6³×2 | 432 | 10.09 | 0.554 | 151.3 | 0.973 | 0.343 | 0.755 | 9.0 / 14.0 |
| TePb 7³×2 | 686 | 15.28 | 0.843 | 243.9 | 1.616 | 0.559 | 1.314 | 12.9 / 21.6 |
| TePb 8³×2 | 1024 | 22.07 | 1.253 | 366.9 | 2.282 | 0.863 | 1.870 | 19.1 / 32.1 |
| TePb 9³×2 | 1458 | — | 1.824 | — | 3.282 | 1.284 | 2.713 | 27.0 / 43.6 |
| TePb 10³×2 | 2000 | — | 2.644 | — | 4.862 | 1.882 | 4.078 | 36.6 / 61.4 |

The heads take 0.12 s at 16 atoms and 0.34 s at 1,024 atoms on either route.
Throughput with four processes under MPS on one card: Flash 15,735 and UniTB
8,930 structures per hour on a 16–432-atom mix; 9,492 and 4,287 per hour on
40 distinct compositions (28–288 atoms), each seen for the first time.

Table builds on one H200: per table 0.013–0.034 s for H0, overlap, p2, P23,
SOC and moments, 0.16 s for envxc and 0.28 s for pair-XC; all 1,220 tables
between a new element and 68 species take 128 s.
