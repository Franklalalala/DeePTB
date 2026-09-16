# 0917-stable：统一维护入口

后续功能与修复统一提交到 `0917-stable`。旧分支和已部署的不可变运行版本保留作
追溯与 checkpoint 兼容性参考。更新仓库不代表运行中的 Python 进程已切换代码。

## 已整合的功能

| 功能 | 代码入口 | 来源 |
|---|---|---|
| LoopSCF、能带损失、动态 batch、按成功更新计数的 WSD | `dptb/nnops`、`dptb/nnops/loopscf` | `0910-stable`，`657f85b3` |
| H0/先验双阶段 Edge-MoE、onsite/hopping mask、非有限 batch 跳过、AO→CG 初始化 | `dptb/nn/embedding`、`dptb/nnops` | Hopper serial 分支及稳定运行基线 |
| 保持 S1 物理输入不变的 S2 TE-flow | `dptb/nnops/flow.py`、`docs/serial_s2_teflow.md` | `7be01e38` |
| GPU NACF/S、完整 spinor SOC、几何推理、预编译径向算子 | `dptb/nacf` | `896a5db0` 加最新审查修复 |
| H0 离线表、共享 Hermite 曲线、projector 复用、批量 pair 装配、带符号 PBE 密度修复 | `h0/h0rebuild` | 本地开发链至 `4a069aec` |
| NACF 的 VNA 仅分配 onsite 块 | `dptb/nacf/assembly.py` | `4a069aec` |
| Switch 式可学习 top-1，复用 grouped cuBLAS / SO2CUDA | `dptb/nn/top1_prior.py`、`top1_so2_cuda.py` | 已验证的 `20260916_top1_switch_noshared_v1` 发布包 |
| 三条先验/目标路线的双阶段配置生成、数据与 checkpoint 检查 | `tools/general_model` | 2026-09-16 通用模型交接包 |

H0 审查分支基于另一条 upstream 历史，因此整合其数值子系统，而不是将该分支的
全部 upstream 改动覆盖训练基线。NACF 审查 overlay 已归入 `dptb/nacf`，不再复制
一份 `review/.../nacf_overlay`。Edge-MoE 两种 schema 共用一套路由参数定义。
NACF 推理入口同时接受双阶段 Edge-MoE 模型。

## Switch top-1

以下是显式选择的模型变体，旧配置默认保留 `legacy`：

```json
{
  "method": "lem_moe_v3_edge_prior_2b",
  "num_experts": 256,
  "top_k": 1,
  "num_shared_experts": 0,
  "edge_router_prior_activate": true,
  "edge_router_top1_mode": "switch",
  "mole_linear_mode": "cublas_grouped",
  "so2_fusion_mode": "streamed_m_major_cueq"
}
```

路由使用全部专家 logits 的 float32 softmax，选择最大概率专家并保留其概率作为
输出权重。不对单个选中概率再次归一化。没有共享专家、选择专用平衡 bias、STE、
概率下限、新增辅助均衡损失或容量丢弃。旧 top-2 / shared-expert checkpoint
不能仅通过更改配置就转换成此结构。该变体的历史证据是双臂各 20 步，不代表长期收敛。

## 三条训练路线

```bash
python tools/general_model/make_configs.py --data /path/to/data --output /path/to/configs
python tools/general_model/check_dataset.py --input /path/to/configs/nacf_h0res.onsite.s1.json --output /path/to/loader.json
python tools/general_model/run_stage.py --input /path/to/configs/nacf_h0res.onsite.s1.json --output /path/to/s1
python tools/general_model/run_stage.py --input /path/to/configs/nacf_h0res.onsite.s2.json --s1-checkpoint /path/to/s1.pth --output /path/to/s2
```

配置由一份已有实验基线和 onsite/hopping 的两个差异字段生成，避免维护重复配置：

| 路线名 | 模型输入 | 训练目标 |
|---|---|---|
| `h0_h0res` | H0 | H − H0 |
| `nacf_nacfres` | NACF（P23 onsite / P2 hopping） | H − NACF |
| `nacf_h0res` | NACF | H − H0 |

这些是 SOC uu-real、256/1/0 的具体配方，不是所有模型的通用默认值。保留 q=.95、
128 个校准 batch、1..96 样本上限、无固定 max_cost 覆盖和 `oom_fallback=false`。
历史 Hopper 短测峰值约 98 GiB；本分支不声称默认配置适配 80GB A100。
波尔生产与数据转换的完整独立交接包仍是单独交付物，其第三方源码副本不重复导入仓库。

## H0 与 NACF

H0 是可从本 checkout 使用的独立子系统，见 [H0 使用说明](../h0/README.md)。
`source h0/env.sh` 只设置当前 checkout 路径与任务缓存，不再跳转历史机器目录。
预制表与 CUDA 二进制不放进 Git；在目标环境显式准备并核对 source/ABI 身份。
H0、NACF 原有数值内核沿用已验证实现；本次整合没有放宽误差门限。

H0 的 ABACUS 对齐径向网格诊断曾在最差样本上改善精度，但新网格尚未完成百例验证；
本次没有静默改变默认径向网格，也不把该诊断冒充新全队列结果。

## 验证范围

遵循 [TESTING.md](../TESTING.md)，只运行受整合影响的行为检查。CPU 定向检查、
配置生成校验与小规模 CUDA 前后向对照用于验证此分支的接口和计算行为。
完整 H0/NACF 百例、原有训练收敛和硬件容量结论沿用各自历史版本的证据边界，
本次没有重新启动生产训练或长时间验收任务。

2026-09-17 整合检查：定向测试覆盖 Switch 路由概率与梯度、H0/NACF 双阶段 S1
冻结、TE-flow、非有限更新处理、NACF 推理入口、共享径向表与 PBE。依赖外部数据
或可选库的项目保留显式 skip。12 份配置通过严格 schema，去重前后逐字段完全一致。
Liyue L40S / PyTorch 2.8.0+cu128 的独立小样本对照通过，观察到 8 次 grouped CUDA
和 2 次 SO2CUDA 调用，前后向最大相对 L2 差约 1.53e-7。
H0 Python 数值模块与导入来源的可执行 AST 一致（忽略 docstring 格式）；没有重跑整个 H0 原生构建或百例验收。
