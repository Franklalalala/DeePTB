# SO2 Wigner representations

`so2_wigner_apply_mode` selects rotation-matrix storage without changing the mathematical definition of the Wigner rotation.

- `compact_blocks` is the default. It stores separate matrices for each angular-momentum order, with shapes `[E,1,1]`, `[E,3,3]`, …, `[E,2*lmax+1,2*lmax+1]`.
- `full_dense` stores the complete block-diagonal matrix, with shape `[E,(lmax+1)^2,(lmax+1)^2]`, for compatibility and reference calculations.

`E` is the number of edges. Set the option in `model_options.embedding`:

```json
{
  "so2_wigner_apply_mode": "compact_blocks"
}
```

The optional SO2CUDA package provides CUDA acceleration. See the [installation guide](../quick_start/easy_install.md) and [testing guide](../../TESTING.md). Memory use and speed depend on the model, graph size, and device; measure the intended configuration.
