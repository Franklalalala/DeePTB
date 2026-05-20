# NexTHam direct blockwise decoder smoke

Date: 2026-05-20

Branch:

```text
0520-block-native
```

Base:

```text
0520-clean-test @ 42092fcf3b1bfd718338f252f6b2cfba7d29cc2e
```

## Reference-code comparison

NexTHAM does not directly emit AO dense blocks from its network head.  The model
emits per-edge equivariant/RME-like `net_out` coefficients, then calls
`e3TensorDecomp.get_H()` to materialize OpenMX/AO-order flattened Hamiltonian
blocks before loss.  Its labels and loss are AO-block based, but there is still
a net-output to AO-block materialization step.

QHNet's active training path is closer to a direct block target route.  It
generates node/pair representations, expands them through an irreps/Wigner head
into padded AO diagonal and non-diagonal blocks, and its loss compares those
blocks directly against block labels with masks.  Searching the inspected
`learn_qh9` path did not show an RME/reduced-element target stage.

## Implemented route

The first blockwise wrapper on `0520-clean-test` used:

```text
E3Hamiltonian -> node_features/edge_features -> feature_tensors_to_block_tensors -> block loss
```

This update keeps the existing `E3Hamiltonian` stage, because that is not the
dominant cost in the current smoke.  It replaces the deterministic
feature-to-block materializer with a learned direct AO block decoder:

```text
E3Hamiltonian -> node_features/edge_features -> DirectAOBlockDecoder -> block loss
```

New config switch:

```json
"prediction": {
  "blockwise_hamiltonian": true,
  "direct_blockwise_hamiltonian": true
}
```

New modules:

```text
dptb.nn.blockwise_hamiltonian.DirectAOBlockDecoder
dptb.nn.blockwise_hamiltonian.DirectBlockwiseE3Hamiltonian
```

The direct decoder predicts padded `node_hamil_blocks` and `edge_hamil_blocks`
from the E3 Hamiltonian feature tensors with small linear heads.  It preserves
the existing blockwise loss and feature-compatible log path.

## Review-fix update

The review package in
`E:\deeptb\codex\0520_blk_loss\deeptb_0520_block_native_review_fixes\deeptb_0520_block_native_review_fixes`
raised one correctness issue and two implementation hygiene issues.  This
revision applies the requested fixes:

- direct decoded AO blocks are zero-masked outside each atom/pair valid block
  shape before the prediction tensors are attached or H0 is added;
- reverse-edge Hermitian averaging keeps the control-flow checks on CPU and
  only moves the reverse index and boolean mask to the model device;
- the decoder now checks E3 feature width against the direct-head input width
  and raises a clear error when they diverge;
- a regression test covers larger padded AO blocks and verifies invalid padded
  rows/columns are zero.

## Local validation

Local Windows Python does not have `torch`, so only static and compile checks
were run locally:

```text
python -m pytest dptb\tests\test_blockwise_clean_integration_static.py -q
11 passed

python -m py_compile dptb\nn\blockwise_hamiltonian.py dptb\nn\deeptb.py dptb\utils\argcheck.py dptb\tests\test_blockwise_clean.py dptb\tests\test_blockwise_clean_integration_static.py
passed
```

The RED step was checked first: the new static tests failed before implementation
because `DirectBlockwiseE3Hamiltonian`, `direct_blockwise_hamiltonian`, and
`DirectAOBlockDecoder` were absent.

## Liyue validation

Liyue worktree:

```text
/home/mingkang_nt/codex/nextham_blockwise_direct_0520/DeePTB_0520_direct_smoke
```

Liyue tests:

```text
PYTHONPATH=$REPO /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424/bin/python -m pytest \
  dptb/tests/test_blockwise_clean_integration_static.py \
  dptb/tests/test_blockwise_clean.py -q

20 passed, 1 warning
```

Smoke data:

```text
/home/mingkang_nt/data/nextham_blockwise_report_0520_smoke
```

The converted 50-frame strict block sample is reused from the prior report:

```text
/home/mingkang_nt/data/nextham_blockwise_report_0520_smoke/blockwise_drop_h0keep_strict_sample
```

## Liyue training smoke

Environment:

```text
/home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424
torch 2.8.0+cu128
cuequivariance 0.9.1
```

Runs:

```text
feature:
/home/mingkang_nt/codex/nextham_blockwise_direct_0520/runs/feature_compare_20260520_200442

materialized blockwise:
/home/mingkang_nt/codex/nextham_blockwise_direct_0520/runs/blockwise_materialized_compare_20260520_200512

direct blockwise:
/home/mingkang_nt/codex/nextham_blockwise_direct_0520/runs/blockwise_direct_compare_20260520_200543
```

Speed:

| route | model | iter 1-25 | iter 13-25 | DPTB wall | process wall | max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| feature | `E3Hamiltonian` | 0.5173 s/iter | 0.3026 s/iter | 15.409 s | 29.87 s | 3221.9 MB |
| materialized blockwise | `BlockwiseE3Hamiltonian` | 0.7174 s/iter | 0.5310 s/iter | 20.181 s | 31.02 s | 3212.4 MB |
| direct blockwise | `DirectBlockwiseE3Hamiltonian` | 0.5486 s/iter | 0.3327 s/iter | 16.045 s | 26.98 s | 3209.9 MB |

Against materialized blockwise, direct blockwise improves:

```text
steady iter: 0.5310 -> 0.3327 s/iter, wall-time -37.4%
DPTB wall:   20.181 -> 16.045 s,      wall-time -20.5%
process:     31.02  -> 26.98 s,       wall-time -13.0%
```

Against feature baseline, direct blockwise is still slightly slower on the
steady window:

```text
0.3026 -> 0.3327 s/iter, +9.9%
```

The direct head adds about 621k parameters:

```text
DirectBlockwiseE3Hamiltonian: 621,108 params
node_decoder: 310,554
edge_decoder: 310,554
```

Epoch metrics:

| metric | feature | materialized blockwise | direct blockwise |
| --- | ---: | ---: | ---: |
| train_loss | 1.0596 | 0.1827 | 0.1206 |
| train_feature_compat_loss | N/A | 1.1589 | 1.5720 |
| train_onsite_loss | 1.7392 | 1.9654 | 2.9878 |
| train_hopping_loss | 0.3800 | 0.3524 | 0.1563 |
| train_block_loss | N/A | 0.1827 | 0.1206 |
| train_block_element_mae | N/A | 0.1827 | 0.1206 |

The loss numbers are not an accuracy comparison because the direct decoder adds
a new randomly initialized block head and the smoke is only 25 iterations.  The
important result is that the training path runs, logs feature-compatible metrics,
passes the AO padding regression, and removes most of the materialized blockwise
overhead.

After smoke, both L40S GPUs were idle:

```text
0, 13, 46068
1, 13, 46068
```

## Interpretation

This direct decoder is a better experimental direction than the first
materialized wrapper if the goal is training speed.  It keeps the existing E3
Hamiltonian stage but removes the deterministic feature-to-block materializer
from the block loss path.

The current direct head is deliberately minimal.  It is not yet a full QHNet-like
irreps expansion head and does not guarantee the same physics constraints beyond
onsite symmetrization and optional reverse-edge averaging.  The next useful
step is a longer same-seed run and then a more structured block head if the
short-run convergence remains acceptable.
