# 0422 cuEq Fastest Route A/B

This note records the final short production A/B on Natlan L40S for the
`0422-cueq-fastest` route. The production route keeps the cuEq SO2 path as the
default:

- `so2_wigner_apply_mode=compact_blocks`
- `so2_fusion_mode=streamed_m_major_cueq`
- `mole_linear_mode=cueq_indexed_linear`
- `mole_full_expert_fast_path=true`
- `onehot_tp_mode=scalar_fast`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The final fastest branch also defaults `E3ElementLinear` to the block-view scale
path. Set `DPTB_E3_ELEMENT_LINEAR_MODE=indexed_gather` to recover the previous
indexed-gather implementation.

## Production Short A/B

All rows used `/home/mingkang_nt/data/0422_test`, 2x L40S, 8 epochs, and CUDA
memory monitoring. The reported speed is `wall_time_s / last_iteration`.

| Variant | Batch size | sec/iter | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Fastest before E3 block-view | 32 | 2.7698 | 32472.1 MB | 34398.0 MB |
| Fastest + E3 block-view | 32 | 2.7325 | 31490.7 MB | 33446.0 MB |
| Fastest before E3 block-view | 48 | 4.0526 | 43262.1 MB | 44606.0 MB |
| Fastest + E3 block-view | 48 | 3.9674 | 41948.5 MB | 44776.0 MB |

Relative to the pre-block-view fastest route:

| Batch size | Speed change | Allocated change | Reserved change |
| ---: | ---: | ---: | ---: |
| 32 | +1.35% | -3.02% | -2.77% |
| 48 | +2.10% | -3.04% | +0.38% |

The bs48 reserved value rises by about 170 MB, but allocated memory drops by
about 1.31 GB and the short production run remains OOM-free.

## Microbench Notes

`E3ElementLinear` block-view was slower in an isolated forward microbench, but it
reduced temporary allocation:

| Path | Forward ms | Peak allocated |
| --- | ---: | ---: |
| indexed gather | 5.753 ms | 652.1 MB |
| block view | 5.929 ms | 468.1 MB |

The production A/B is therefore the deciding signal for this workload: the
smaller temporary scale tensor reduces memory pressure enough to improve
end-to-end short-run throughput.
