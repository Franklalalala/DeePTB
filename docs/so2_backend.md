# SO2 后端

SO2 参数、路由和纯 PyTorch 参考计算保留在 DeePTB。可选的 SO2CUDA 包通过 `dptb.nn.so2_backend` 提供加速；DeePTB 不编译 SO2 CUDA 扩展。

安装 SO2CUDA 后，符合 CUDA FP32、布局与路由条件的层自动调用 `so2_cuda_ops.deeptb`。CPU、其他 dtype、autocast、几何求导、非线性插值块、`torch.func` 或缺少可选包时使用参考实现，并按原因记录一次回退日志。原生执行错误会直接抛出，便于发现配置或工具链问题。

UniTB-dense 通过 `so2_backend.dense_forward` 调用 SO2CUDA 的 `dense_pairs`，保持 m=0 与逐 m 的累加次序；UniTB-X1 使用激活空间融合和分组 GEMM，保留每个 top-k 槽的非线性与加权顺序。true-dense 层使用 SO2CUDA 的 `true_dense_pairs` 接口；没有该接口的旧包会回退。

## 配置

| 设置 | 作用 |
|---|---|
| `SO2_CUDA_BACKEND=auto` | 默认自动选择；设为 `off` 强制参考实现。旧值 `none`、`torch` 等同 `off`。 |
| `so2_fusion_mode=streamed_m_major_fused_p0` | UniTB 的生产配置；保留 `staged`、`streamed_m_major_ref` 和 grouped 参考路线。 |
| `mole_linear_mode=cublas_grouped` | 专家线性层通过同一可选后端调用分组 GEMM。 |
| `so2_m_linear_mode=indexed_sandwich_cuda_multi` | 扩展 true-dense 层的默认路线；`standard` 使用 PyTorch。 |

兼容现有启动脚本的变量如下；无需全部设置：

| 旧变量 | 处理 |
|---|---|
| `DPTB_SO2_FUSION_MODE` | 在未显式给出模型选项时选择 MoE SO2 路线。 |
| `DPTB_MOLE_LINEAR_MODE` | 在未显式给出模型选项时选择专家线性路线。 |
| `DPTB_SO2_M_LINEAR_MODE` | 在未显式给出参数时选择 true-dense 路线；`cuda_pack_scatter_multi` 映射为默认路线。 |
| `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE` | 由 SO2CUDA 解析；生产值为 `scalar` 或 `indexed_sandwich_multi`。 |
| `DPTB_SO2_ACTIVATION_FUSED_P0` | 值为 `0` 时关闭激活空间融合。 |
| `DPTB_SO2_ACTIVATION_FUSED_P0_GEMM` | SO2CUDA 激活空间调度，生产默认 `per_slot`。 |
| `DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES` / `MAX_EDGES` | true-dense 加速边数范围；优先于 `SO2_CUDA_MIN_EDGES` / `MAX_EDGES`，`0` 不设限。 |

`DPTB_SO2_FUSE_M_CUBLAS`、`DPTB_SO2_SORTED_EDGE_VIEW`、persistent、scheduled、materialized 和 CUTLASS 实验开关不再选择 DeePTB 路线。SO2CUDA 的构建目录和精度变量由该包维护，例如 `SO2_CUDA_PACK_SCATTER_BUILD_DIR`、`SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`、`SO2_CUDA_FAST_TF32`。严格 FP32 使用 `SO2_CUDA_FAST_TF32=0`。

## 张量积调用

上游 LEM / SLEM 使用固定的张量返回值：

```python
from dptb.nn.tensor_product import SO2_Linear
layer = SO2_Linear(irreps_in, irreps_out)
features = layer(features, edge_vectors, latents)
```

需要旋转缓存、插值或后端参数时使用独立类：

```python
from dptb.nn.tensor_product import SO2LinearCached
layer = SO2LinearCached(irreps_in, irreps_out)
features, rotation = layer(features, edge_vectors, latents)
features, rotation = layer(features, edge_vectors, latents, rotation)
```

两者不按参数猜测返回类型。MoE 模块 `tensor_product_moe_v3.SO2_Linear` 的原有元组返回合同保持不变。后端不添加模型参数或 buffer，既有检查点仍按原参数名加载。

晶体对称投影和 H0 / NACF 的独立数值核不属于 SO2 后端。
