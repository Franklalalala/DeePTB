# NexTHAM-style structured blockwise decoder smoke

Date: 2026-05-20

Branch:

```text
0520-block-native
```

## Design

The dense direct decoder was useful as a speed upper-bound experiment, but it
does not match NexTHAM's final Hamiltonian construction.  NexTHAM keeps the
learnable prediction upstream and uses a parameter-free angular-momentum
mapping to construct AO/openmx Hamiltonian blocks from `net_out`.

This update adds the same design choice to the DeePTB blockwise route:

```text
E3Hamiltonian -> structured Hamiltonian features -> NexTHamAOBlockDecoder -> AO block loss
```

The new `NexTHamAOBlockDecoder` has no final `nn.Linear` or `nn.Parameter`.
It uses the existing DeePTB E3 Hamiltonian stage for the CG/Wigner constrained
RME-to-H feature transform, then scatters those structured features into padded
AO blocks through precomputed orbital slice indices.  This keeps the explicit
angular-momentum structure and removes the old Python-heavy
`feature_tensors_to_block_tensors` materializer from the training path.

New config switch:

```json
"prediction": {
  "blockwise_hamiltonian": true,
  "nextham_blockwise_hamiltonian": true
}
```

Selection priority in `NNENV` is:

```text
nextham_blockwise_hamiltonian -> direct_blockwise_hamiltonian -> blockwise_hamiltonian -> E3Hamiltonian
```

The previous dense `DirectAOBlockDecoder` remains only as an ablation path.

## Local validation

Local Windows Python does not have `torch`, so behavior tests are run on natlan.
Local static checks and compile checks were used before remote smoke.

```text
python -m pytest dptb\tests\test_blockwise_clean_integration_static.py -q
13 passed

python -m py_compile dptb\nn\blockwise_hamiltonian.py dptb\nn\deeptb.py dptb\utils\argcheck.py tools\convert_feature_lmdb_to_blockwise.py
passed
```

## Natlan validation

Worktree:

```text
/home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/DeePTB
```

Environment:

```text
/home/mingkang_nt/data/anaconda3/envs/dptb_p2_wigner_cu12_py310
torch 2.8.0+cu128
```

Unit/static suite:

```text
PYTHONPATH=$REPO python -m pytest \
  dptb/tests/test_blockwise_clean_integration_static.py \
  dptb/tests/test_blockwise_clean.py -q

23 passed, 1 warning
```

The converter was also hardened for this natlan sample format by tensorizing
NumPy-backed LMDB feature and metadata arrays before calling `feature_to_block`
or `block_to_feature`.

Converted blockwise smoke data:

```text
source: /home/mingkang_nt/codex/stored_edge_feature_fastpath_20260506/small_stored_feature_dataset_n30
output: /home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/data/blockwise_n30
report: /home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/data/blockwise_n30_report.json
```

Conversion summary:

```text
30 entries
strict feature->block->feature roundtrip
delta max_abs = 0.000e+00
h0 max_abs = 0.000e+00
```

## Natlan training smoke

Runs:

```text
feature:
/home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/runs/feature_20260520_204518

materialized blockwise:
/home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/runs/blockwise_materialized_20260520_204756

NexTHAM-style structured blockwise:
/home/mingkang_nt/codex/0520_block_native_nextham_structured_20260520/runs/blockwise_nextham_structured_20260520_204827
```

All runs used the same 30-sample smoke line, 1 epoch, 8 iterations.

Speed:

| route | model path | iterations | logged DPTB wall | process wall | max RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| feature | `E3Hamiltonian` | 8 | 11.287 s | 23.97 s | 3228.3 MB |
| materialized blockwise | `BlockwiseE3Hamiltonian` | 8 | 19.974 s | 30.68 s | 3516.8 MB |
| structured blockwise | `NexTHamBlockwiseE3Hamiltonian` | 8 | 11.615 s | 22.25 s | 3513.7 MB |

Against materialized blockwise, the NexTHAM-style structured route improves:

```text
logged DPTB wall: 19.974 -> 11.615 s, wall-time -41.9%
process wall:     30.68  -> 22.25 s,  wall-time -27.5%
```

Against the feature baseline, structured blockwise is close on this short run:

```text
logged DPTB wall: 11.287 -> 11.615 s, +2.9%
```

This feature-vs-structured gap should be treated as smoke-level only because
the run has only 8 iterations and fixed startup/setup costs are visible.

Epoch metrics:

| metric | feature | materialized blockwise | structured blockwise |
| --- | ---: | ---: | ---: |
| train_loss | 2.1280 | 0.2220 | 0.2220 |
| train_feature_compat_loss | N/A | 2.2704 | 2.2704 |
| train_onsite_loss | 3.8056 | 4.1072 | 4.1072 |
| train_hopping_loss | 0.4503 | 0.4336 | 0.4336 |
| train_block_loss | N/A | 0.2220 | 0.2220 |
| train_block_element_mae | N/A | 0.2220 | 0.2220 |
| train_block_onsite_loss | N/A | 1.8650 | 1.8650 |
| train_block_hopping_loss | N/A | 0.2028 | 0.2028 |

The structured route exactly matches the materialized blockwise metrics in this
smoke, which is the expected result for a parameter-free replacement of the
materializer.

## Interpretation

This is the better NexTHAM-aligned direction compared with the dense direct
decoder.  It preserves the explicit angular-momentum constrained feature path,
does not add learnable AO-entry parameters at the final step, and still removes
most of the old materialized blockwise overhead.

The remaining performance cost is the AO block loss itself, reverse-edge block
completion, valid-shape/padding handling, and blockwise metric logging.  The
next performance step should focus on grouping/scattering efficiency and on a
longer same-seed convergence run, not on a free dense AO head.
