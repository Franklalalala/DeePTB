# NexTHam blockwise loss smoke record

Date: 2026-05-20

Branch:

```text
0520-clean-test
```

Local worktree:

```text
E:\deeptb\codex\0520_blk_loss\DeePTB_0520_clean_test
```

Liyue worktree:

```text
/home/mingkang_nt/codex/nextham_blockwise_clean_0520/DeePTB_0520_clean_test
```

## Purpose

Test whether a NexTHam-style blockwise Hamiltonian target path can run in DPTB
without storing target feature tensors, while still logging feature-compatible
onsite and hopping losses for comparison with the existing feature-level loss.

## Cueq environment

The production smoke keeps:

```text
model_options.embedding.mole_linear_mode = cueq_indexed_linear
```

The existing liyue conda environment that supports this path is:

```text
/home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424
python = /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python
dptb   = /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/dptb
torch  = 2.8.0+cu128
cuequivariance = 0.9.1
cuequivariance_torch = 0.9.1
cuequivariance_ops_torch = 0.9.1
```

Use `PYTHONPATH` to point this environment at the test worktree:

```bash
export PYTHONPATH=/home/mingkang_nt/codex/nextham_blockwise_clean_0520/DeePTB_0520_clean_test
```

Avoid `set -u` when activating this environment because the activation script
references `ADDR2LINE` before it is bound.

## Dataset

Production non-SOC featureized dataset on liyue:

```text
/home/mingkang_nt/data/nextham_feature_align_20260516/nextham_my_split_feature_20260516
```

Observed split size:

```text
train: 240 lmdb shards, 12000 rows
valid: 40 lmdb shards, 2000 rows
test:  60 lmdb shards, 3000 rows
```

Smoke source shard:

```text
/home/mingkang_nt/data/nextham_feature_align_20260516/nextham_my_split_feature_20260516/train/data.0000.lmdb
```

The first train entry contains feature and H0 feature tensors:

```text
node_features: [1, 425]
edge_features: [14, 425]
node_h0:       [1, 425]
edge_h0:       [14, 425]
```

It does not contain raw `hamiltonian` or `hamiltonian_0`.

## Conversion

Smoke root:

```text
/home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke
```

Feature sample:

```text
/home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke/feature_sample/train/data.0000.lmdb
```

Converted blockwise sample:

```text
/home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke/blockwise_drop_h0keep_sample/data.0000.lmdb
```

Conversion command:

```bash
PYTHONPATH=$REPO \
/home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python \
tools/convert_feature_lmdb_to_blockwise.py \
  --input-root /home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke/feature_sample \
  --output-root /home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke/blockwise_drop_h0keep_sample \
  --split train \
  --input-config /home/mingkang_nt/data/nextham_order_align_20260516/input_bs96_2x1_rawh0_0505_stable_prod.json \
  --target-feature-policy drop \
  --h0-feature-policy keep \
  --edge-complete-policy hermitian \
  --strict-roundtrip \
  --max-entries 50 \
  --map-size 17179869184 \
  --report /home/mingkang_nt/data/nextham_blockwise_clean_0520_smoke/convert_drop_h0keep_train0000_report.json \
  --overwrite
```

Conversion result:

```text
wrote entries: 50
delta max:     0.000e+00
h0 max:        0.000e+00
elapsed:       11.2 s
```

For this dataset, target features can be dropped, but H0 features must be kept
unless a block-native H0 initialization path is added.  The blockwise smoke
therefore uses:

```text
data_options.train.get_Hamiltonian = false
data_options.train.get_H0 = true
data_options.train.prefer_precomputed_h0 = true
model_options.prediction.blockwise_hamiltonian = true
train_options.loss_options.train.method = hamil_blockwise_nextham
```

## Validation

Local validation:

```text
python -m pytest dptb\tests\test_blockwise_clean_integration_static.py -q
6 passed

python -m py_compile dptb\entrypoints\train.py dptb\entrypoints\multi_train.py dptb\nnops\trainer.py dptb\nnops\multi_trainer.py dptb\nnops\blockwise_nextham_loss.py
passed
```

Liyue validation:

```text
PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m pytest dptb/tests/test_blockwise_clean_integration_static.py -q
6 passed

PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m pytest dptb/tests/test_blockwise_clean.py -q
5 passed, 1 warning

PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m py_compile dptb/entrypoints/train.py dptb/entrypoints/multi_train.py
passed
```

## Smoke runs

Feature baseline run:

```text
/home/mingkang_nt/codex/nextham_blockwise_clean_0520/runs/feature_cueq_metrics_20260520_153742
model: E3Hamiltonian
iters: 25
entry wall: 15.221 s
process wall: 27.67 s
iter 1 to last: 0.5160 s/iter
back half: 0.3036 s/iter
MAX_RSS: 3089.7 MB
epoch train_loss: 1.0596
epoch train_onsite_loss: 1.7392
epoch train_hopping_loss: 0.3800
```

Blockwise run:

```text
/home/mingkang_nt/codex/nextham_blockwise_clean_0520/runs/blockwise_drop_h0keep_cueq_metrics_20260520_153809
model: BlockwiseE3Hamiltonian
iters: 25
entry wall: 20.085 s
process wall: 30.91 s
iter 1 to last: 0.7119 s/iter
back half: 0.5246 s/iter
MAX_RSS: 3239.3 MB
epoch train_loss: 0.1827
epoch train_feature_compat_loss: 1.1589
epoch train_onsite_loss: 1.9654
epoch train_hopping_loss: 0.3524
epoch train_block_loss: 0.1827
epoch train_block_element_mae: 0.1827
epoch train_block_onsite_loss: 1.1085
epoch train_block_hopping_loss: 0.1732
```

After both runs, `nvidia-smi` reported both L40S GPUs idle at 13 MiB.

## Interpretation

The cleaned blockwise path is runnable and supports a feature-dropped target
LMDB while preserving feature-compatible loss logs.  It is not speed-positive in
this 50-frame smoke.  The current `BlockwiseE3Hamiltonian` still calls the
existing `E3Hamiltonian` head and then materializes AO blocks, so it removes
target feature storage but does not remove model-side feature computation.

The next speed-relevant design step is a block-native prediction head or a
training path that avoids differentiable feature-to-block materialization on the
forward path.

## Review Update

Date: 2026-05-20

Review package:

```text
E:\deeptb\codex\0520_blk_loss\blockwise_nextham_review_update_pkg\blockwise_nextham_review_update_pkg
```

The follow-up patch tightens the correctness-first smoke path in three places:

```text
1. strict Hermitian edge completion
2. raw abs/square/count component exposure for exact logging reducers
3. forwarding prediction.* blockwise options into BlockwiseE3Hamiltonian
```

New config options accepted by `prediction` include:

```text
complete_edges
strict_complete_edges
symmetrize_onsite
add_h0
node_pad_shape / edge_pad_shape
full_output_node_field / full_output_edge_field
```

New blockwise loss options include:

```text
distributed_log_reduce
expose_component_sums
```

The clean liyue review worktree used for this validation was:

```text
/home/mingkang_nt/codex/nextham_blockwise_review_0520/DeePTB_0520_review_smoke
```

Validation commands and results:

```text
PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m pytest dptb/tests/test_blockwise_clean_integration_static.py -q
9 passed

PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m pytest dptb/tests/test_blockwise_clean.py -q
7 passed, 1 warning
```

Strict conversion command added `--strict-edge-completion` to the earlier
50-frame conversion smoke:

```text
converted sample:
/home/mingkang_nt/data/nextham_blockwise_review_0520_smoke/blockwise_drop_h0keep_strict_sample/data.0000.lmdb

wrote entries: 50
delta max:     0.000e+00
h0 max:        0.000e+00
elapsed:       15.3 s
```

Strict blockwise smoke config:

```text
/home/mingkang_nt/data/nextham_blockwise_review_0520_smoke/input_blockwise_strict_review_cueq_smoke.json
```

Strict blockwise training run:

```text
/home/mingkang_nt/codex/nextham_blockwise_review_0520/runs/blockwise_strict_review_cueq_20260520_162341
model: BlockwiseE3Hamiltonian
iters: 25
entry wall: 22.440 s
process wall: 33.65 s
MAX_RSS: 3215.4 MB
epoch train_loss: 0.1827
epoch train_feature_compat_loss: 1.1589
epoch train_onsite_loss: 1.9654
epoch train_hopping_loss: 0.3524
epoch train_block_loss: 0.1827
epoch train_block_element_mae: 0.1827
epoch train_block_onsite_loss: 1.1085
epoch train_block_hopping_loss: 0.1732
```

After the run, both L40S GPUs were idle at 13 MiB.
