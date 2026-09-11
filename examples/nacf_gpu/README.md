# Geometry-only NACF inference

`NACFGeometryPredictor` builds NACF and overlap from species, Cartesian positions,
cell and periodic flags. Its input requires no DFT calculation, H0, structure
prior file or training labels. It supports `lem_moe_v3_prior` and
`lem_moe_v3_prior_2b` models whose **output is Full H minus NACF**. Full SOC
requires an additional source-bound SOC projector sidecar and a checkpoint
trained for the full spinor target.

This path reconstructs `H = residual + NACF`. NACF means P23 onsite and P2
hopping; the model's learned 2b contribution is already part of its residual
output. Do not add it again, and do not add physical H0.

## What runs where

All NACF-specific implementation lives in `dptb/nacf`: `radial.py` and
`csrc/radial.cu` evaluate tables; `assembly.py` builds P2/P23/S and packs RME;
`overlap.py` reads the S sidecar; `inference.py` loads and runs the checkpoint.
CLI, offline S construction and benchmarking are optional modules in this same
package. The three scripts under `tools/` are five-line compatibility launchers.
The pre-existing generic P2/P23 readers and qualified SBT builder are reused.

1. Once per table bank, load and checksum required P2/P23/S shards, copy the
   original cubic spline coefficients to device, and build harmonic transforms.
   Tables are lazy, shared across structures, and kept as PyTorch module buffers.
2. For each geometry, CPU code creates the directed graph and all periodic
   projector queries. Preparation also binds AO order and compiles RME gathers.
3. GPU code evaluates the original spline, rotates real AO blocks, contracts
   nonlocal projectors and crystal-field factors, forms S, packs features, runs
   the model, and adds NACF back. No table files, SciPy evaluation, NumPy work or
   device-to-host transfers occur in the numerical prior forward.
4. Batches merge queries by table. Separate atom, query, output-block and cell
   offsets prevent cross-structure interactions.

The native CUDA kernel fuses distance evaluation, interval lookup, Horner
evaluation, real harmonics and sparse AO rotation in one launch per table batch.
Projector contractions and RME packing remain PyTorch GPU operations. Neighbour
construction remains CPU work. Default prior precision is float64 and the
verified model precision is float32.

### Native CUDA backend

`backend='auto'` selects the fused CUDA inference kernel on CUDA tensors and
uses the torch reference on CPU or for autograd inputs. `backend='cuda'` is
strictly CUDA inference; `backend='torch'` explicitly selects the reference.
The optional native extension is compiled lazily at first use, then cached.
That first compilation performs source-file I/O and is excluded from warm
timings. Compilation failures are reported; they do not silently switch to the
torch backend. No kernel is compiled merely by importing `dptb.nacf`.

A CUDA toolkit, C++ compiler and Ninja are required. Set `CUDA_HOME` and place
the environment's `bin` plus `$CUDA_HOME/bin` on PATH. Keep compiler/cache paths
on the intended workspace filesystem using `TORCH_EXTENSIONS_DIR`, `TEMP`,
`TMP`, and `TMPDIR`. Limit build parallelism with `MAX_JOBS=2` when sharing a host.
Use `TORCH_CUDA_ARCH_LIST` appropriate to the installed compiler and device.
The verified PRO6000 host has CUDA 12.4 nvcc and a CUDA 12.8 PyTorch runtime;
`TORCH_CUDA_ARCH_LIST='8.9+PTX'` supplies forward-compatible PTX for Blackwell.
This kernel uses no architecture-specific tensor-core instructions.

One CUDA block owns each displacement query. It locates the nonuniform interval
once, computes real harmonics and small rotation matrices in shared memory, and
evaluates only structurally nonzero canonical channels while rotating to AO
output. The source cubic coefficients and support mask remain unchanged.
There is no dense canonical intermediate or Python shell loop in this path.
The generic fused kernel allows up to 48 KiB shared-memory angular metadata;
larger bases raise an explicit error and can use the torch backend.

## Python interface

```python
from ase.build import bulk
from dptb.data.interfaces.p2_table import P2TableStore
from dptb.data.interfaces.p23_table import P23VNAFactorTableStore
from dptb.nacf import OverlapTableStore, NACFTableBank, NACFGeometryPredictor

# model and model_options must come from the same verified checkpoint.
bank = NACFTableBank(P2TableStore(p2_dir), P23VNAFactorTableStore(p23_dir),
                     overlap_store=OverlapTableStore(overlap_dir), device='cuda')
predict = NACFGeometryPredictor(
    model, bank, model_options, target='full_h_minus_nacf',
    expected_p2_source_fingerprint=training_p2_manifest_sha256)
atoms = bulk('Si', 'diamond', a=5.43)
result = predict(atoms)
batch = predict([atoms, atoms.copy()])
```

The output is a dictionary with `node_features`, `edge_features` (absolute H in
eV), `node_overlap`, `edge_overlap` (dimensionless S), geometry and graph keys.
Features use the checkpoint's RME layout, not a dense global Hamiltonian:
triangular non-SOC, explicitly reduced uu-real, or full SOC as configured.
`ptr` and `batch` separate structures; edge rows follow `edge_index`
and `edge_cell_shift`. Use the existing DeePTB reconstruction tools for AO blocks
or H(k)/S(k). No band eigensolve occurs in this interface.

`prepared = predict.prepare(atoms)` and repeated `prepared()` calls are valid
only for the **identical geometry**. They recompute the numerical prior on GPU.
After moving atoms, changing cell/PBC/species, or requesting coordinate forces,
do not reuse this prepared object. The public inference interface is not an
autograd force API. Its snapshot is independent of later ASE edits.

## CLI

Run in the DeePTB environment with the checkout on `PYTHONPATH`:

```bash
python tools/infer_nacf_geometry.py \
  --checkpoint /path/to/nnenv.iter100000.pth \
  --p2 /path/to/p2 --p23 /path/to/p23 --overlap /path/to/overlap \
  --expected-p2-sha256 TRAINING_P2_MANIFEST_SHA256 \
  --target full_h_minus_nacf \
  --geometry structures.extxyz --batch-size 8 --output predictions.pt
```

The equivalent package entry is `python -m dptb.nacf.cli`. Add `--backend cuda`
to require the native path, or `--backend torch` for the previous GPU reference.

The CLI uses `streamed_m_major_ref` and `split_loop` inference implementations,
scans model weights for nonfinite values, and writes CPU tensor batches plus a
JSON provenance record. Model loading and file writes are outside reported
geometry-to-output timings. The first batch can include cold table/kernel costs.
Use a trusted locally produced checkpoint. The CLI's explicit target and P2 hash
must be recovered from that checkpoint's training contract: not every checkpoint
embeds its original `data_options`.

The current verified recipe is checkpoint `p2b_s2_600773/ckpt/nnenv.iter100000.pth`:

- Checkpoint SHA256: `6e3f4f9ad9ce5c5a4e43d5e05c19dbab6de7d7b0c21730c3f716a76e943bff53`.
- P2: `abc214afcfe5b44334f2d923770d3a204baf939e03bcdab67472b07b84ff384b`.
- P23: `1dd587ed014175b1d3d84b87231a4f558fa759a5ed3cbd805ee92371a074efda`.
- Non-SOC, 425 RME, 68-element model basis, `prior_kind=na_cf`,
  `only2b=false`, `two_b_seed_gnn=true`.

The full original manifests do not imply that all shards have been deployed.
Every requested pair must exist. Missing P23 or S is an error; there is no
P2-only or identity-overlap fallback.

## Overlap sidecar

The recovered production P2 table contains no S arrays. Construct an immutable
S-only sidecar from the **exact ORB files recorded by that P2 manifest**:

```bash
python tools/build_overlap_sidecar.py \
  --p2-manifest /path/to/p2/manifest.json --orb-root /path/to/orbs \
  --species C Si --output /new/path/overlap_csi
```

The builder uses the existing SBT quadrature settings and verifies ORB SHA256 and
shell order. `h0rebuild` is an offline builder dependency, not an inference
dependency. `complete=true` means construction finished, not that a new basis or
quadrature has passed an independent numerical qualification. Validate a new
sidecar against direct AO integration or an independently sourced ABACUS S.
Original P2 and P23 tables are never modified.
Offline construction requires the source checkout's existing
`tools.build_nonsoc_p2_tables` SBT implementation. The inference package does not
import this builder or `h0rebuild`.

## Tests and matched timing

```bash
python -m pytest dptb/tests/test_p2_gpu.py dptb/tests/test_nacf_gpu.py dptb/tests/test_nacf_soc.py -q
python tools/benchmark_nacf_gpu.py \
  --checkpoint /path/to/checkpoint --p2 /path/to/p2 --p23 /path/to/p23 \
  --overlap /path/to/overlap --expected-p2-sha256 TRAINING_P2_MANIFEST_SHA256 \
  --geometry periodic_cases.extxyz --repeats 5 --output benchmark.json
```

The benchmark compares independently assembled CPU P2/P23/S with GPU tables,
uses the same model and graph on both routes, and alternates execution order.
It includes synchronization, warmup, sample timings and numerical errors. Its
legacy CPU oracle supports fully periodic cells only. GPU prepared timings
exclude topology compilation; `gpu_geometry_to_output` includes it. Neither
route includes checkpoint loading or output serialization. Shared GPU/CPU load
can materially change timings, so retain individual samples and hardware state.

Add `--compare-native` (or run `python -m dptb.nacf.benchmark`) to compare CUDA
and torch table backends on identical cached tables, geometry and checkpoint.
Each repetition alternates backend order. Both primitive table times and full
prior/model timings are reported. The verified C/Si run with 600773 measured:

| Geometry | Torch GPU NACF+S+packing | Native CUDA NACF+S+packing | Prepared full inference, torch → native |
|---|---:|---:|---:|
| Si, 2 atoms / 172 edges | 12.52 ms | 1.19 ms | 285.5 → 273.8 ms |
| SiC, 2 atoms / 316 edges | 51.80 ms | 3.58 ms | 355.1 → 304.4 ms |

Native-vs-torch Full-H RME differences were at most 3.82 micro-eV. Individual
radial blocks differed by at most 1.68e-15 in their source units. These are
implementation-agreement measurements on C/Si, not DFT prediction accuracy or
an all-element qualification. Tests additionally cover float32, CUDA Graph
replay, nonuniform grids, support boundaries, zero channels, and autograd routing.

For single-structure versus multi-structure AI forward timing, pass a geometry
file containing at least 16 distinct structures and add:

```bash
python -m dptb.nacf.benchmark \
  --checkpoint /path/to/checkpoint --p2 /path/to/p2 --p23 /path/to/p23 \
  --overlap /path/to/overlap --expected-p2-sha256 TRAINING_P2_MANIFEST_SHA256 \
  --geometry distinct_structures.extxyz --forward-batches 1 2 4 8 16 \
  --warmup 3 --repeats 7 --output forward.json
```

Each batch uses an ordered prefix of the input file. `ai_forward` measures only
`model(inputs)` with graph and NACF/S already on GPU. Every call gets fresh input
clones, completed and synchronized before its timer; table lookup, input copying
and NACF add-back are excluded. Synchronized wall time includes Python dispatch
as well as GPU execution. `prepared_full` recomputes priors and includes packing,
copies, model and add-back; `geometry_to_full` also includes CPU graph/topology
preparation and transfers. All three exclude initial loading/compilation and disk
output. JSON contains raw samples, medians per batch and per structure, throughput
and residual/Full-H/S agreement against independently prepared singleton runs.
It rejects cross-structure neighbours and nonfinite or inconsistent predictions.

## DPA4C relationship and scope

The reference is deepmd-kit commit `28b7d068801716765ab8119257f814596e49a10c`:
`deepmd/pt_expt/kernels/dpa4c/graph_compress.py` builds quintic radial tables;
`source/op/pt/dpa4c/graph_compress.cuh` performs GPU interval lookup and Horner
evaluation; `graph_compress_kernel.cuh` fuses neighbourhood work. This code borrows
GPU-resident coefficients and batched evaluation, while retaining the qualified
DeePTB **cubic** interpolant and cutoff semantics. It is not a port of the DPA4C
descriptor or a claim to match DPA4C kernel performance.

## Full SOC lookup

The scalar radial functions remain real and use the same fused CUDA kernel.
`soc.py` builds complex spinor projector D matrices from the exact source UPFs,
including both j channels and matching off-diagonal radial couplings. Assembly
contracts all four spin blocks (uu, ud, du, dd); scalar base, VNA and overlap
contribute to spin diagonal blocks. Hermitian projection uses conjugate transpose.

Build a new immutable sidecar, bound to the original P2 manifest and UPF hashes:

```bash
python -m dptb.nacf.build_soc \
  --p2 /path/to/p2 --upf-root /path/to/PP_ORB \
  --gate1-script /path/to/qualified_gate1.py --species C Si \
  --output /new/path/soc
```

Pass `soc_store=SOCProjectorStore(soc_dir, p2_store)` to `NACFTableBank`, or
`soc=soc_dir` to `load_predictor`, or `--soc /path/to/soc` to the inference CLI.
Full real-valued SOC features are ordered per orbital pair as
`[Re(uu,ud,du,dd), Im(uu,ud,du,dd)]`. A 27-spatial-AO canvas has 5832 features,
whereas its uu-real reduction has 729. Complex features are also supported.
The ABACUS padded AO layout is spin-major, with spin offset equal to the largest
AO width in the structure/batch. Copying scalar blocks or padding 729 features
does not recover spin-flip or imaginary channels.

`prepare_geometry(bank, idp, structures, cutoffs)` returns a feature plan and
geometry graph without loading a learned model. This supports independent full
SOC lookup validation. A uu-real checkpoint remains a reduced predictor even
when supplied a full SOC sidecar; it cannot produce full spinor H. The verified
600773 checkpoint above is non-SOC. Full-SOC lookup results alone are not evidence
of a trained full-SOC model's speed or prediction accuracy.

## Production timing and memory

```bash
python -m dptb.nacf.production_benchmark \
  --checkpoint /path/to/checkpoint --p2 /path/to/p2 --p23 /path/to/p23 \
  --overlap /path/to/overlap --expected-p2-sha256 TRAINING_P2_MANIFEST_SHA256 \
  --geometry production100.extxyz --batch-sizes 2 4 8 16 --output /new/report
python -m dptb.nacf.soc_benchmark \
  --config /path/to/train_config.json --p2 /path/to/p2 --p23 /path/to/p23 \
  --overlap /path/to/overlap --soc /path/to/soc \
  --geometry production100.extxyz --output /new/soc_report
python -m dptb.nacf.memory_probe --p2 /path/to/p2 --species Si --output /new/memory.json
```

The production benchmark compares original CPU/SciPy assembly, torch GPU tables,
and fused native CUDA. The CPU reference retains the production assembler's
internal query batching and original CPU graph; it does not copy the graph back
from GPU. Mixed CPU batches call that assembler per structure and concatenate
features. GPU batches group radial queries across structures by table.

`ai_forward` includes only the model on resident graph/prior inputs, with fresh
clones and synchronization outside its timer. `*_prior` includes numerical NACF,
S, packing and any transfer on a prepared geometry; `*_fresh_prior` additionally
includes new geometry preparation. `cuda_geometry_to_full` includes the entire
geometry-to-H/S path and NACF add-back. Cold loading, hashes, JIT and output file
writes are excluded. No DFT labels enter any forward call.

Ratios are **prior time / pure AI forward time**, not prior fractions of total
inference latency. Report both the ratio of summed per-structure median times
and the distribution of per-structure ratios. Batch sizes use ordered prefixes;
each is validated against independent singleton H/S predictions and graph
ownership. Failures are retained. Shared-GPU rows are excluded from the primary
aggregate; `--allow-shared-gpu` permits a separately labelled diagnostic aggregate.
Presence snapshots cannot prove that no transient interference occurred.

The separate fresh-process memory probe records CUDA context, extension import,
table/input allocation, and first/second native launch after output deallocation.
PyTorch allocated bytes measure live tensors, reserved bytes include its cache,
and NVIDIA process MiB additionally includes CUDA runtime/module memory with
driver rounding. Table buffers, model tensors, geometry-plan tensors, outputs and
operation peaks are different quantities; do not call their sum kernel size.
