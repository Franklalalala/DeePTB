# 0425 FlashTP-Style SO2 Route

The 0425 stable production route remains:

```text
so2_wigner_apply_mode = compact_blocks
so2_fusion_mode       = streamed_m_major_cueq
mole_linear_mode      = cueq_indexed_linear
onehot_tp_mode        = scalar_fast
```

This branch adds a tunable SO2 path-aggregation mode:

```text
DPTB_SO2_FLASH_AGGREGATE=input   # default: rotate input l-groups once
DPTB_SO2_FLASH_AGGREGATE=output  # aggregate local outputs, rotate each l once
DPTB_SO2_FLASH_AGGREGATE=hybrid  # input aggregation plus high-l output aggregation
DPTB_SO2_FLASH_AGGREGATE=1       # full input+output aggregate
DPTB_SO2_FLASH_AGGREGATE=0       # direct-output fallback
```

FlashTP's public kernel targets `e3nn.o3.TensorProduct` channelwise `uvu`
paths with edge scatter/reduce. DeePTB's current hot path is different:
`SO2_Linear` decomposes irreps by SO(2) order `m`, applies MoE linear maps, and
rotates with Wigner blocks. A direct FlashTP call does not match this interface
without rewriting the layer into a standard CG tensor product.

The transferable FlashTP idea is path aggregation:

1. rotate each input `l` group once into the local frame;
2. serve all `m` paths from local-frame slices;
3. optionally accumulate local outputs by `l`;
4. optionally rotate each output `l` group once back to the global frame.

The default is input-side aggregation only, so the route still writes `m`
contributions directly into the final output tensor and avoids the per-`l`
output group buffer. Output aggregation is opt-in because it can reduce repeated
small rotation work but reintroduces the grouped output buffer. Hybrid mode is
the intermediate benchmark target: it keeps direct output for low `l` channels
and only aggregates output groups with `l >= DPTB_SO2_FLASH_HYBRID_L_MIN`
(default `2`).

## Validation

The SO2 pytest coverage compares aggregate modes against the direct fallback for
forward values and gradients, including mixed `rotate_in`/`rotate_out` settings.
End-to-end CUDA speed and peak memory should still be decided by Liyue A/B runs
because the change mainly affects kernel launch and tensor materialization
patterns.

## Liyue Short A/B Results, 2026-04-25

Environment:

```text
machine: liyue, 2 x L40S
branch: origin/0425-flash at 4f5394a
python env: /home/mingkang_nt/anaconda3/envs/dptb_triton_gp_0424
run root: /home/mingkang_nt/codex/so2_flash_queue_20260425_214335
per-case limit: timeout -k 10s 590s
```

Static validation passed before the training queue:

```text
43 passed, 28 warnings in 32.26s
```

All training cases exited with code `124`, which is the expected timeout exit
for the 590 s short-run limit. CUDA memory rows were logged every 10 iterations.
The summary median uses adjacent logged intervals after dropping only the first
`10 -> 20` interval; later-window medians below are included to reduce warmup
and dynamic compilation bias.

Full short-run summary:

| mode | last iter | median s/iter | wall-time vs `direct_0` | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_0` | 290 | 1.8019 | baseline | 36.33 GiB | 38.66 GiB |
| `input` | 270 | 1.9468 | -8.04% | 35.85 GiB (-0.47) | 38.89 GiB (+0.23) |
| `hybrid` | 270 | 1.9572 | -8.62% | 35.29 GiB (-1.04) | 37.85 GiB (-0.81) |
| `output` | 280 | 1.8630 | -3.39% | 35.75 GiB (-0.57) | 38.74 GiB (+0.08) |
| `full_1` | 260 | 1.9428 | -7.82% | 35.28 GiB (-1.05) | 38.57 GiB (-0.09) |

Later-window median `s/iter`:

| window | `direct_0` | `input` | `hybrid` | `output` | `full_1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `>=100` | 1.8130 | 1.9399 (-7.00%) | 1.9974 (-10.17%) | 1.8152 (-0.12%) | 1.9213 (-5.97%) |
| `>=150` | 1.7384 | 1.8679 (-7.45%) | 1.9244 (-10.70%) | 1.7550 (-0.95%) | 1.9013 (-9.37%) |
| `>=200` | 1.7394 | 1.8626 (-7.08%) | 1.9344 (-11.21%) | 1.7444 (-0.29%) | 1.8668 (-7.33%) |
| `last60` | 1.7762 | 1.8302 (-3.04%) | 1.8888 (-6.34%) | 1.7444 (+1.79%) | 1.8668 (-5.10%) |

Interpretation:

- `input`, `hybrid`, and `full_1` remain slower after excluding the earlier
  windows, so input-side aggregation is not a production-default candidate for
  this workload.
- `hybrid` is the cleanest short-run memory reduction: peak allocated drops by
  about 1.04 GiB and peak reserved drops by about 0.81 GiB, but it costs about
  6-11% steady-state iteration time depending on the window.
- `output` is the only speed candidate. It is roughly neutral after iteration
  100 and slightly faster in the last-60-iteration window, but that window has
  too few samples for a production claim. It should be validated by a longer
  paired `direct_0` vs `output` run before changing any default behavior.
