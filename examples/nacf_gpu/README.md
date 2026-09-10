# Geometry-only NACF inference

`NACFGeometryPredictor` builds NACF and overlap from species, Cartesian positions,
cell and periodic flags. Its input requires no DFT calculation, H0, structure
prior file or training labels. It supports non-SOC `lem_moe_v3_prior` and
`lem_moe_v3_prior_2b` models whose **output is Full H minus NACF**.

This path reconstructs `H = residual + NACF`. NACF means P23 onsite and P2
hopping; the model's learned 2b contribution is already part of its residual
output. Do not add it again, and do not add physical H0.

## What runs where

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

This is a PyTorch implementation with GPU-resident tables. It does **not** yet
implement GPU neighbour construction or a custom fused CUDA kernel. Default
prior precision is float64 and the verified model precision is float32.

## Python interface

```python
from ase.build import bulk
from dptb.data.interfaces.p2_table import P2TableStore
from dptb.data.interfaces.p23_table import P23VNAFactorTableStore
from dptb.data.interfaces.overlap_table import OverlapTableStore
from dptb.data.interfaces.nacf_gpu import NACFTableBank
from dptb.postprocess.nacf_geometry import NACFGeometryPredictor

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
Features use the checkpoint's triangular non-SOC RME layout, not a dense global
Hamiltonian. `ptr` and `batch` separate structures; edge rows follow `edge_index`
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

## Tests and matched timing

```bash
python -m pytest dptb/tests/test_p2_gpu.py dptb/tests/test_nacf_gpu.py -q
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

## DPA4C relationship and scope

The reference is deepmd-kit commit `28b7d068801716765ab8119257f814596e49a10c`:
`deepmd/pt_expt/kernels/dpa4c/graph_compress.py` builds quintic radial tables;
`source/op/pt/dpa4c/graph_compress.cuh` performs GPU interval lookup and Horner
evaluation; `graph_compress_kernel.cuh` fuses neighbourhood work. This code borrows
GPU-resident coefficients and batched evaluation, while retaining the qualified
DeePTB **cubic** interpolant and cutoff semantics. It is not a port of the DPA4C
descriptor or a claim to match DPA4C kernel performance.

SOC requires its own spinor nonlocal assembly and output packing contract.
Scalar P2/P23 radial assets are reusable ingredients, but copying scalar blocks
or padding 425 features to the SOC 729 layout does not implement SOC. This API
rejects SOC mappers. Existing SOC materialization and production data are kept
separate from this non-SOC model deployment.
