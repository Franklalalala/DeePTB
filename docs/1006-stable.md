# `1006-stable` maintenance release

This release provides UniTB, including its single-expert UniTB-dense configuration,
and upstream embeddings through shared model-building, training, evaluation, and
electronic-structure postprocessing interfaces.

## Component layout

| Location | Purpose |
| --- | --- |
| `dptb/nn/embedding/unitb/` | UniTB and UniTB-dense; `layer_topology` selects optional SLEM interactions (UniTB-SLEM); compatibility interfaces resolve historical names |
| `dptb/nn/embedding/lem.py`, `slem.py` | Upstream baselines without priors |
| `dptb/nn/embedding/lem_prior.py`, `slem_prior.py` | Prior-aware baselines |
| `dptb/data` | Graphs, LMDB records, sidecars, AO blocks, and feature conversion |
| `dptb/nnops`, `dptb/entrypoints` | Single-model and `multi_train` training, validation, and inference |
| `dptb/nacf`, `dptb/priors/h0` | NACF and physical H0 prior construction and reconstruction |
| `dptb/postprocess` | Bands, density of states, operator export, and other postprocessing |

New configurations use `model_options.embedding.method = "unitb"`.
UniTB and UniTB-dense are configurations of the same implementation. Options
select expert counts, routing, hidden irreps, priors, and charge response.
Loading an existing checkpoint must preserve its effective configuration,
parameter layout, and prior semantics.

Upstream methods such as `baseline`, `deeph-e3`, `e3baseline_6`,
`e3baseline_nonlocal`, `none`, `mpnn`, `se2`, and `trinity` remain
available through their original interfaces.

See the [architecture guide](architecture.md) for canonical ownership, import
migration, embedding discovery, and extension registration. H0 offline commands
use `dptb h0`; its [guide](../dptb/priors/h0/README.md) documents the external
work/table directories and preparation lifecycle.

## Components and dependencies

[SO2CUDA](https://github.com/Franklalalala/SO2CUDA) provides SO2 tensor products,
MoE GEMM, and pack/scatter CUDA kernels through its DeePTB interface. DeePTB's
PyTorch reference routes support environments without the required CUDA
components. See the SO2CUDA repository for installation and benchmarks.

[LoopSCF](https://github.com/Franklalalala/loopscf) is maintained separately.
It depends on this release's model building, graph data, AO-block conversion,
LMDB, optimizers, and distributed helpers. EMolStudio uses `build_model`,
`AtomicData`, `AtomicDataDict`, `DataLoader`, `build_dataset`, and the
`HR2HK` interfaces.

## Data and training

Data must define the meaning of full-operator, residual, and prior fields.
H0, NACF, overlap, and other sidecars must align with the original graph's atoms,
edges, periodic shifts, and basis ordering. Add priors back exactly once in the
corresponding reconstruction path.

Training supports distance-expert masks, LMDB, dynamic batches, HybridMuon/WSD,
prior noise, spectral clipping, and checkpoint restoration. Matching loss names
do not establish comparability between different configurations: also check
targets, units, reduced spaces, and masks.

HybridMuon defaults to cached routing, same-shape matrix buckets, and multi-tensor
updates. CUDA lazily captures pure tensor-bucket operations with CUDA Graphs.
Newton–Schulz iteration counts and coefficients, MAGMA, clipping clocks, learning
rates, expert scaling, and checkpoint fields retain their semantics. Capture adds
first-use overhead and retains graph memory; report warm-up and steady-state
timing and memory separately. Direct optimizer users may set
`optimizer.execution_mode = "legacy"` for sequential execution or
`optimizer.use_cuda_graph = False` to disable graph caching. These are runtime
attributes, not training configuration fields. Loading optimizer state clears
execution caches without editing old checkpoints.

SO2CUDA block-layout kernel updates retain the DeePTB call interface and the
UniTB/UniTB-SLEM parameter layouts. Checkpoint loading and numerical behavior
must still be validated for affected execution routes.

- [Prior inputs and checkpoints](advanced/prior_inputs.md)
- [NACF candidate priors](nacf_candidate_prior.md)
- [NACF compact mapping](nacf_compact_packing.md)
- [NACF prepared stores](nacf_prepared_store.md)
- [Dynamic-batch OOM behavior](advanced/dynamic_batch_oom_fallback.md)

See [TESTING.md](../TESTING.md) for the testing strategy. Release validation
covers interfaces, CPU behavior, affected optional CUDA routes, existing
checkpoint inference, and deterministic short-training comparisons separately.
