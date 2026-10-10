# 测试

使用已配置的 DeePTB 环境，按修改影响的行为选择测试。一次有效的针对性检查即可；文档修改无需重跑数值实验。

```bash
python tools/test.py dptb/tests/test_record_codec.py
```

`python tools/test.py` 和 `bash ut.sh` 使用同一组短测试，覆盖谱损失、记录解码、先验模型和专家掩码。它们适合检查安装和通用接口，不覆盖全部功能。

需要完整检查时显式指定：

```bash
python tools/test.py dptb/tests
python -m pytest h0/tests_h0fast
```

## 目录与可选组件

`dptb/tests` 按模块家族和行为组织；公共构造器放在辅助模块中，测试文件之间不互相导入。`h0/tests_h0fast` 单独运行。LoopSCF 的测试在独立仓库运行，并使用已安装的 DeePTB。

可选依赖的测试在依赖缺失时说明原因并跳过。`dptb/tests/_requires.py` 提供 `requires_cuda`、`requires_multi_gpu`、`requires_so2_cuda`、`requires_module(name)`，NACF 原生依赖由 `conftest.py` 延迟检查。真实参考数据测试由各测试说明的环境变量启用；数据和生成结果放在仓库外。

安装 SO2CUDA 0.3.0 后，可以运行受影响的 CUDA 测试：

```bash
python -m pytest dptb/tests/test_so2_kernels_cuda.py
```

未安装 SO2CUDA 时，应验证对应的 PyTorch 参考路径；也可通过 `SO2_CUDA_BACKEND=off` 明确禁用加速。后端选择、设备、精度和实际运行路径需要写入验证记录；不能仅凭配置开关认定 CUDA 已执行。

`conftest.py` 检查测试导入和执行后的默认 dtype 及确定性设置，并在 CUDA 初始化前设置 `CUBLAS_WORKSPACE_CONFIG`。CPU 并行度和 GPU 资源由运行环境明确配置。

## 验证原则

保留能验证可观察行为的测试：小型独立数值参考、数据与图的对齐、检查点恢复、有限且正确路由的梯度，以及损坏输入的拒绝。复现缺陷通常只需一个最小回归用例；边界情况优先参数化。

不测试源码拼写、注释、私有调用顺序、实现的冻结副本或偶然的错误文本。配置测试应验证真实的接受或拒绝行为。

UniTB 与 UniTB-dense 的重构应使用相同输入、检查点、严格 FP32 设置和确定性模式，比较推理输出以及四步短训练的 loss 和参数。报告差异量级、运行环境和检查点加载结果，不以全量测试通过代替数值等价性。

只有受影响的内核、设备路径、数据契约或未解决的数值问题需要新的 GPU 或真实数据检查。硬件基准和预测精度研究使用独立资产；不开展种子重复或同配置重复训练。

提交可复用代码、当前功能文档和针对性的行为回归测试。运行日志、作业清单、基准快照、源码副本及一次性诊断产物保存在仓库外，变更说明概括验证范围与结果。

数值对照使用严格 FP32：设置 `NVIDIA_TF32_OVERRIDE=0`、`SO2_CUDA_FAST_TF32=0`、`DPTB_CUBLAS_GROUPED_FAST_TF32=0`，并关闭 PyTorch matmul/cuDNN 的 TF32。CPU 与 CUDA 在 cutoff 边界附近可能选择不同活动边，因此同设备检查点回归与跨设备比较应分别报告。缺少旧设备参考的可运行项只证明可运行，不声明数值等价。

独立 LoopSCF 的包测试和 checkpoint 回归应安装本分支后在 loopscf 仓库执行。性能测试使用 SO2CUDA 提供的 UniTB / UniTB-dense 示例，记录实际后端、预热与计时次数、前向/反向中位数和四分位、显存以及数值差；随机模型的合成结构基准不代表物理精度。
