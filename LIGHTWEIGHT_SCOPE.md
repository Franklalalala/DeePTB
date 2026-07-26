# DeePTB `0726-light` scope

`0726-light` is a deliberately incompatible, development-focused derivative
of:

```text
origin/0721-stable@af4b62fa7518f30aee2b515fad40276486d0d0e7
```

The selection window was 2026-03-26 through 2026-07-25. Commit frequency was
used as a first-pass hotspot signal, then checked against the current import
graph, registries, strict configuration schema, configs, and tests. A recent
touch alone was not enough to retain an unreachable historical copy.

## Hotspot evidence

The most frequently touched maintained files in the four-month window were:

| Commit touches | File |
| ---: | --- |
| 95 | `dptb/utils/argcheck.py` |
| 50 | `dptb/nnops/flow.py` |
| 42 | `dptb/nnops/trainer.py` |
| 41 | `dptb/nn/embedding/lem_moe_v3.py` |
| 39 | `dptb/nnops/multi_trainer.py` |
| 24 | `dptb/nn/tensor_product_moe_v3.py` |
| 22 | `dptb/nn/embedding/lem_moe_v3_h0.py` |
| 22 | `dptb/data/dataset/lmdb_dataset.py` |
| 16 | `dptb/nn/so2_moe_fused_p0.py` |
| 15 | `dptb/nn/tensor_product.py` |
| 13 | `dptb/plugins/saver.py` |

This evidence keeps the current LEM/H0/pair model line, E3/block-native
outputs, LMDB and record materialization, flow/trainer/restart semantics, and
the active SO(2) acceleration work.

## Retained public model surface

The strict schema and runtime registry expose exactly these 18 embedding
methods:

```text
emoles
emoles_openequi
emoles_openequi_eqv3
emoles_openequi_eqv3_ffn
emoles_openequi_nodeffn
emoles_openequi_norm
emoles_openequi_norm_v2
lem_in_frame
lem_in_frame_openequi
lem_moe_openequi
lem_moe_v3
lem_moe_v3_edge
lem_moe_v3_edge_h0
lem_moe_v3_h0
lem_moe_v3_prior
lem_non_linear
lem_non_linear_h0
lem_pair
```

The retained prediction methods are:

```text
e3tb
block_native
```

Supporting modules for AO/RME projection, late block expansion, output-route
contracts, flow-time encoding, pair refinement, and soft edge memory are
implementation dependencies rather than additional public embedding aliases.

Also retained:

- Hamiltonian and SOC construction, HR-to-HK conversion, band inference, and
  AO-block output;
- LMDB datasets, record decoding, dynamic batching, and durable
  materialization;
- CFM/MeanFlow, physical and precomputed priors, block-space ODE, H-B0,
  uu-real, and residual-H routes;
- trainer, multi-trainer, distributed restart, saver, monitor, and current
  train/test/run entrypoints;
- standard PyTorch/cuEquivariance SO(2) routes and in-repo CUDA extension
  loaders. The separately packaged `so2-cuda-ops` backend is an optional
  `so2` installation extra.

## Removed surface

- NNSK, SKTB, DFTB-SK, MIX, empirical-SK, NRL conversion, and their parameter
  databases or pretrained blobs;
- NEGF transport and TBtrans/TBPLAS/pybinding integrations;
- DOS and Fermi-surface integrations;
- 52 historical embedding Python implementations, including duplicate or
  unreachable old `lem` registrations and dated experimental variants;
- non-LMDB ASE/NPZ/HDF5/ABACUS/DeePH/default dataset classes and the old
  in-memory dataset implementation;
- old data conversion, config-template, bond, checkpoint-to-JSON, and SKF
  collection CLI commands;
- the vendored `_xitorch` copy and utilities used only by removed routes;
- obsolete SKTB/NEGF/interface tutorials, notebooks, generated schema
  sections, and their images.

Precomputed external-prior labels such as `dftb`, `xtb`, `sk`, and `nnsk`
remain recognized inside the preserved flow contract so old data keys can be
diagnosed or rejected deterministically. This branch does not ship an
on-the-fly SK/DFTB implementation.

## Intentional incompatibilities

- Data configs must use `type: LMDBDataset`.
- Model configs must define a retained embedding plus `e3tb` or
  `block_native`.
- Removed embedding aliases, model families, CLI commands, and postprocessing
  integrations fail during schema validation, model construction, or command
  dispatch.
- Checkpoints that depend on removed Python classes or model-option sections
  are not load-compatible with this branch.
- Install `.[so2]` to exercise the external fused/scheduled SO(2) backend.
  Its CUDA tests also require a working nvcc/ninja toolchain.

## Size reduction

The following figures compare the Git archive of the baseline with the
current tracked/untracked source state (ignored caches excluded):

| Measure | `0721-stable` | `0726-light` | Reduction |
| --- | ---: | ---: | ---: |
| Files | 662 | 458 | 204 (30.82%) |
| Bytes | 31,969,324 | 7,908,988 | 24,060,336 (75.26%) |
| Python files | 502 | 346 | 156 (31.08%) |
| Python lines | 215,233 | 149,610 | 65,623 (30.49%) |

The source diff currently removes nearly 85,000 lines. Generated
documentation changes account for part of that total.

## `flow.py` freeze

Per the branch requirement, `dptb/nnops/flow.py` is not part of this cleanup.
Its baseline and working-tree Git blob are both:

```text
597ec6ae9e7c658923c8ca73dd7b584a312c0748
```

## Validation contract

Before publication, this branch is checked with:

- Git diff whitespace validation;
- Python 3.9-compatible AST parsing and internal-import resolution;
- package compile/import and CLI-help smoke tests;
- runtime embedding-registry and strict-config checks;
- sharded repository tests plus full test collection;
- deleted-module and documentation-route residual scans;
- a final `flow.py` blob comparison.

Latest local evidence:

- 313 package/test Python files parsed with the Python 3.9 grammar, with zero
  parse failures or missing internal modules;
- 188 production modules imported, with zero failures in the default
  installation environment;
- six complete repository configs passed the real strict normalizer, and the
  runtime embedding registry contained exactly 18 methods;
- 1,771 pytest nodes collected across 124 test files;
- all test files were exercised in memory-safe shards; non-skipped assertions
  passed. The materialization durability shard reported an inner pytest return
  code of zero before the low-memory Windows host reclaimed its already
  completed outer process with code 137;
- the optional external SO(2) kernels were not executed because this host has
  neither `so2-cuda-ops` nor an nvcc/ninja build toolchain. Their tests skipped
  through explicit capability gates.
