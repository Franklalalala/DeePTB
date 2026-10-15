# SO2 backend

SO2 parameters, routing, and PyTorch reference computation remain in DeePTB.
The optional SO2CUDA package provides acceleration through
`dptb.nn.so2_backend`; DeePTB does not compile SO2 CUDA extensions.

With SO2CUDA installed, layers meeting the CUDA FP32, layout, and routing
requirements call `so2_cuda_ops.deeptb` automatically. CPU, other dtypes,
autocast, geometry gradients, nonlinear interpolation blocks, `torch.func`,
or an unavailable optional package select the reference implementation, with a
fallback reason logged once. Native execution errors propagate so configuration
and toolchain problems remain visible.

UniTB-dense calls SO2CUDA's `dense_pairs` through
`so2_backend.dense_forward`, preserving the m=0 and per-m accumulation order.
UniTB (PDQ-MoE) uses activation-space fusion and grouped GEMM, preserving each
top-k slot's nonlinear and weighting order. True-dense layers use
`true_dense_pairs`; older packages lacking this interface fall back.

## Configuration

| Setting | Effect |
| --- | --- |
| `SO2_CUDA_BACKEND=auto` | Default automatic selection; `off` forces reference execution |
| `so2_fusion_mode=streamed_m_major_fused_p0` | UniTB default; uses grouped PyTorch without acceleration; `staged` and `streamed_m_major_ref` are alternative PyTorch routes |
| `mole_linear_mode=cublas_grouped` | Expert linear layers call grouped GEMM through the same optional backend |
| `so2_m_linear_mode=indexed_sandwich_cuda_multi` | Default route for extended true-dense layers; `standard` uses PyTorch |

These optional environment variables select routes without editing configuration:

| Variable | Effect |
| --- | --- |
| `DPTB_SO2_FUSION_MODE` | Selects the MoE SO2 route when no explicit model option is supplied |
| `DPTB_MOLE_LINEAR_MODE` | Selects the expert-linear route when no explicit model option is supplied |
| `DPTB_SO2_M_LINEAR_MODE` | Selects the true-dense route when no explicit argument is supplied |
| `DPTB_SO2_ACTIVATION_FUSED_P0` | `0` disables activation-space fusion |
| `DPTB_SO2_ACTIVATION_FUSED_P0_GEMM` | SO2CUDA activation-space scheduling; defaults to `per_slot` |
| `DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES` / `MAX_EDGES` | True-dense acceleration edge-count limits; override `SO2_CUDA_MIN_EDGES` / `MAX_EDGES`; `0` means no limit |

SO2CUDA manages its build-directory and precision variables, including
`SO2_CUDA_PACK_SCATTER_BUILD_DIR`, `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`, and
`SO2_CUDA_FAST_TF32`. Set `SO2_CUDA_FAST_TF32=0` for strict FP32.

## LEM / SLEM

`lem`, `slem`, `lem_prior`, and `slem_prior` use
`tensor_product.SO2_Linear`. With SO2CUDA installed, eligible layers default
to the true-dense route (`so2_m_linear_mode=indexed_sandwich_cuda_multi`).
It replaces only the paired m>0 operations; parameter layouts, Wigner rotations,
and m=0 computation retain the reference contract.

Select the reference with either setting:

| Setting | Scope |
| --- | --- |
| `DPTB_SO2_M_LINEAR_MODE=standard` | Set before model construction; true-dense SO2 layers use PyTorch |
| `SO2_CUDA_BACKEND=off` | All SO2CUDA calls use reference execution |

These settings change execution routes without changing parameter names or
checkpoint layouts. Floating-point differences depend on device, precision,
input, and dispatch. Validate outputs, gradients, optimizer steps, and rotation
equivariance for the intended workload. Record actual dispatch, warm-up,
synchronized forward/backward timings, and memory use before reporting speedups.
Backend selection alone does not guarantee unchanged accuracy or training loss.

## Tensor-product calls

Upstream LEM/SLEM use a fixed tensor return type:

~~~python
from dptb.nn.tensor_product import SO2_Linear
layer = SO2_Linear(irreps_in, irreps_out)
features = layer(features, edge_vectors, latents)
~~~

Use the separate class for rotation caching, interpolation, or backend options:

~~~python
from dptb.nn.tensor_product import SO2LinearCached
layer = SO2LinearCached(irreps_in, irreps_out)
features, rotation = layer(features, edge_vectors, latents)
features, rotation = layer(features, edge_vectors, latents, rotation)
~~~

Neither class infers its return type from arguments. The existing tuple contract
of `tensor_product_moe_v3.SO2_Linear` is retained. The backend adds no model
parameters or buffers; existing checkpoints load under their original names.

Crystal-symmetry projection and the independent H0/GF prior kernels (`dptb.prior`) are
outside this backend.
