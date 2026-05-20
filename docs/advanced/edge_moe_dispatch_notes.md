# Edge MoE Dispatch Notes

This note records the dispatch options tested for `lem_moe_v3_edge_h0`.

## Selected Path

The compact edge-wise MoE path uses `pyg_lib.ops.segment_matmul`:

```text
x_sorted: [sum_edges * inner, in_features]
ptr:      [num_unique_bond_types + 1]
weight:   [num_unique_bond_types, in_features, out_features]

out[start:end] = x_sorted[start:end] @ weight[group]
```

This avoids both Python per-group loops and padding groups to the largest
group. It matches the fast path used by typed/relation-specific GNN operators
such as R-GCN.

Current validated environments:

| Host | GPU | PyTorch | pyg-lib | Result |
| --- | --- | --- | --- | --- |
| liyue | L40S, sm89 | 2.8.0+cu128 | 0.5.0+pt28cu128 | float32 forward/backward OK |
| pro6000 | RTX PRO 6000 Blackwell, sm120 | 2.8.0+cu128 | 0.5.0+pt28cu128 | float32 forward/backward OK |

The current pyg-lib wheel only supports `float32` for this operator in the
tested environments.

## Rejected Paths

### Padding Batched BMM

Padding every bond-type group to the largest group is simple, but it is a poor
fit for production batches. On the 0516 feature dataset with batch size 32, the
observed padding ratio was:

| Metric | Padding ratio |
| --- | ---: |
| min | 6.32 |
| median | 19.60 |
| p90 | 61.58 |
| max | 95.03 |

Forcing the padding path on L40S caused large memory spikes and OOM in
end-to-end smoke. Example microbenchmarks:

| Case | Ratio | Loop ms | Padding ms | Padding peak |
| --- | ---: | ---: | ---: | ---: |
| balanced 16k | 1.0 | 2.71 | 0.95 | 466 MB |
| skew75 16k | 72.0 | 2.87 | 32.38 | 16.3 GB |
| skew98 16k | 94.5 | 2.61 | 42.34 | 21.4 GB |

The padding implementation was removed to keep the production path simple and
avoid silent slow or memory-heavy fallback behavior.

### PyTorch grouped_mm

PyTorch 2.8 exposes the internal `torch._grouped_mm`, but the tested wheel only
allows compute capability 9.0. It failed on both L40S sm89 and Blackwell sm120
with:

```text
torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0
```

Future PyTorch releases may expose a broader `torch.nn.functional.grouped_mm`,
but this was not usable in the current training environments.

### Triton Jagged Prototype

A small Triton jagged grouped GEMM prototype confirmed that a no-padding grouped
kernel is viable on L40S. It was not integrated because forward-only Triton code
would require a maintained custom backward path. `pyg_lib.ops.segment_matmul`
already provides forward and backward for the current float32 training path.

## References

- PyG R-GCN uses `pyg_lib.ops.segment_matmul` for sorted relation segments.
- DGL `TypedLinear` uses the same gather/segment matmul design for relation
  specific linear projections.
- NVIDIA cuBLAS and CUTLASS grouped GEMM remain possible lower-level extension
  routes if pyg-lib becomes insufficient.
