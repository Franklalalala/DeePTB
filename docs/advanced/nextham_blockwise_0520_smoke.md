# NexTHam Blockwise Loss Smoke Record

Date: 2026-05-20

Branch prepared for review:

```text
0520-soc-h0-train-eval-clean
```

## Purpose

This branch adds a correctness-first blockwise Hamiltonian training path for the
current non-SOC NexTHam feature workflow.  It allows converted LMDB samples to
store AO-block Delta-H targets, train with an AO-block loss, and still emit
feature-compatible onsite/hopping metrics for comparison with the existing
feature-level `hamil_abs` reports.

The branch deliberately does not change the legacy `hamil_abs` reduction.

## Data Contract

Source feature LMDB samples are expected to contain:

```text
node_features / edge_features   Delta-H feature labels
node_h0 / edge_h0               H0 feature inputs
edge_index / edge_cell_shift    graph topology and lattice shifts
atomic_numbers or atom_types    atom identity
```

The converter adds:

```text
node_delta_hamil_blocks / edge_delta_hamil_blocks
node_delta_hamil_block_shape / edge_delta_hamil_block_shape
node_h0_blocks / edge_h0_blocks
node_h0_block_shape / edge_h0_block_shape
```

By default the converter performs a strict feature-to-block-to-feature
roundtrip check before dropping or shadowing feature tensors.

## Config Switches

Training config switches:

```text
model_options.prediction.blockwise_hamiltonian = true
train_options.loss_options.train.method = hamil_blockwise_nextham
```

Prediction options:

```text
complete_edges
strict_complete_edges
symmetrize_onsite
add_h0
node_pad_shape / edge_pad_shape
full_output_node_field / full_output_edge_field
```

Loss options:

```text
optimization
block_reduction
complex_reduction
log_feature_compatible
distributed_log_reduce
expose_component_sums
```

## Validation

Natlan smoke workdir:

```text
/home/mingkang_nt/codex/train_eval_clean_smoke_20260520_1779271285/DeePTB
```

Baseline:

```text
origin/0506-stable @ 9770876
```

Environment:

```text
/home/mingkang_nt/data/anaconda3/envs/moe_soc_20260419_natlan/bin/python
```

Commands:

```bash
export PYTHONPATH=$REPO

$PY -m py_compile \
  dptb/data/_keys.py \
  dptb/data/AtomicData.py \
  dptb/data/dataset/lmdb_dataset.py \
  dptb/data/interfaces/blockwise_tensor.py \
  dptb/nn/blockwise_hamiltonian.py \
  dptb/nnops/blockwise_nextham_loss.py \
  dptb/nnops/loss.py \
  tools/convert_feature_lmdb_to_blockwise.py

$PY tools/convert_feature_lmdb_to_blockwise.py --help
$PY -m pytest dptb/tests/test_blockwise_clean_integration_static.py dptb/tests/test_blockwise_clean.py -q
```

Result:

```text
18 passed, 1 warning in 3.64s
```

The warning is from `torch_geometric.distributed` deprecation in the installed
environment and is unrelated to this patch.

## Prior 50-Frame Training Smoke

Earlier validation on the same code path used a 50-frame non-SOC NexTHam feature
sample.  The blockwise run used `BlockwiseE3Hamiltonian` and
`hamil_blockwise_nextham`, completed 25 iterations, and produced:

```text
epoch train_loss: 0.1827
epoch train_feature_compat_loss: 1.1589
epoch train_onsite_loss: 1.9654
epoch train_hopping_loss: 0.3524
epoch train_block_loss: 0.1827
epoch train_block_element_mae: 0.1827
epoch train_block_onsite_loss: 1.1085
epoch train_block_hopping_loss: 0.1732
```

This smoke confirms the blockwise path is runnable and logs both AO-block and
feature-compatible metrics.  It is not expected to be speed-positive yet:
`BlockwiseE3Hamiltonian` still decodes through the existing `E3Hamiltonian`
feature head and then materializes AO blocks for the loss.
