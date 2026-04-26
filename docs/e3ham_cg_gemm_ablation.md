# E3Hamiltonian CG Contraction Ablation

This branch keeps the `so2_w_out_opt` production route and replaces the
E3Hamiltonian CG contraction with a GEMM-backed implementation:

```text
[N, R, C] -> [N*C, R]
[I, J, R] -> [I*J, R]
F.linear([N*C, R], [I*J, R]) -> [N, C, I, J]
```

The operation is mathematically equivalent to both the previous einsum
implementation and the original broadcast reference. The CG basis remains a
fixed tensor, so this does not introduce new trainable parameters.

Liyue CUDA microbench results from 2026-04-27:

| case | einsum | GEMM | tiled broadcast |
| --- | ---: | ---: | ---: |
| d-d, N=32768, C=4 | 0.510 ms | 0.429 ms, 1.19x | 2.371 ms, 0.215x |
| g-g, N=8192, C=4 | 0.804 ms | 0.678 ms, 1.185x | 7.084 ms, 0.114x |

The tiled-broadcast route is intentionally not included in production code. It
was slower in the tested shapes and increased peak memory versus the einsum and
GEMM routes.
