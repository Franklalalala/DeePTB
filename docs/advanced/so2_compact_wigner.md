# SO2 Wigner 表示

`so2_wigner_apply_mode` 控制旋转矩阵的存储方式，不改变 Wigner 旋转的数学定义。

- `compact_blocks`：按角动量阶保存独立的小矩阵，形状依次为 `[E,1,1]`、`[E,3,3]`、…、`[E,2*lmax+1,2*lmax+1]`，是默认选项。
- `full_dense`：保存完整分块对角矩阵，形状为 `[E,(lmax+1)^2,(lmax+1)^2]`，供兼容与参考计算使用。

这里 `E` 是边数。配置写在 `model_options.embedding`：

```json
{
  "so2_wigner_apply_mode": "compact_blocks"
}
```

CUDA 加速由可选 SO2CUDA 提供。安装与测试入口见[安装说明](../quick_start/easy_install.md)和[测试说明](../../TESTING.md)。显存与速度取决于模型、图规模和设备，应以实际配置的测量为准。
