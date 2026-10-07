# `1006-stable` 维护版本

本版本提供 UniTB-dense、UniTB-X1 和上游标准 embedding，统一保留模型构建、训练、评测及电子结构后处理接口。

## 模块结构

| 位置 | 功能 |
| --- | --- |
| `dptb/nn/embedding/unitb.py` | UniTB-dense 与 UniTB-X1；既有生产名称由兼容接口解析 |
| `dptb/nn/embedding/lem.py`、`slem.py` | 上游无先验基线 |
| `dptb/nn/embedding/lem_prior.py`、`slem_prior.py` | 带先验输入的基线 |
| `dptb/data` | 图、LMDB 记录、侧车、AO block 与特征转换 |
| `dptb/nnops`、`dptb/entrypoints` | 单模型及 `multi_train` 训练、验证和推理入口 |
| `dptb/nacf`、`h0` | NACF 与物理 H0 先验构建及重建 |
| `dptb/postprocess` | 能带、态密度、算符导出及其他后处理 |

新配置采用 `model_options.embedding.method = "unitb"`。dense 与 X1 是同一实现的不同配置；专家数、路由、隐藏 irreps、先验和电荷响应设置由配置决定。加载已有检查点应保留它的有效配置、参数布局及先验语义。

上游 `baseline`、`deephe3`、`e3baseline_local6`、`e3baseline_nonlocal`、`identity`、`mpnn`、`se2`、`trinity` 等标准 embedding 继续通过原接口使用。

## 组件与依赖

[SO2CUDA](https://github.com/Franklalalala/SO2CUDA) 的 DeePTB 接口提供 SO2 张量积、MoE GEMM 和 pack/scatter CUDA 加速。DeePTB 的 PyTorch 参考路径用于不具备相应 CUDA 组件的环境；基准与安装方法见 SO2CUDA 仓库。

[LoopSCF](https://github.com/Franklalalala/loopscf) 使用独立仓库，依赖本版本的模型构建、图数据、AO block 转换、LMDB、优化器及分布式辅助接口。EMolStudio 通过 `build_model`、`AtomicData`、`AtomicDataDict`、`DataLoader`、`build_dataset` 和 `HR2HK` 系列接口调用 DeePTB。

## 数据与训练

数据需要明确完整算符、残差和先验字段的含义。H0、NACF、重叠矩阵及其他侧车必须与原图的原子、边、周期位移和基组排序对齐；先验只在相应重建路径中加回一次。

训练保留距离专家掩码、LMDB 加载、动态批次、HybridMuon/WSD、先验噪声、谱裁剪及检查点恢复。配置不同的实验不能仅凭同名 loss 判断数值可比较性；必须同时核对目标、单位、约化空间和掩码。

- [先验输入与检查点](advanced/prior_inputs.md)
- [NACF 候选先验](nacf_candidate_prior.md)
- [NACF 紧凑映射](nacf_compact_packing.md)
- [NACF prepared store](nacf_prepared_store.md)
- [动态批次 OOM 行为](advanced/dynamic_batch_oom_fallback.md)

测试策略见仓库根目录的 `TESTING.md`。发布验证分别覆盖接口、CPU 行为、可选 CUDA 路径、既有检查点推理和确定性短训练等价性。
