# UniTB

UniTB 把 dense 与逐边先验路由模型放在同一个 embedding 中。配置入口是
`model_options.embedding.method: unitb`，实现入口是
`dptb.nn.embedding.unitb.UniTB`。主干依次执行几何编码、先验初始化、路由、
消息传递与 Hamiltonian 输出。初始化层、交互层和先验投影只有一份实现。

## dense 与 X1

X1 embedding 使用默认配置：

```json
{"method": "unitb"}
```

dense 自动选择一个 full 专家、无共享专家、top-1、均匀通道和类型路由：

```json
{"method": "unitb", "num_experts": 1}
```

显式配置覆盖默认值。X1 默认为 3 层、latent_dim=128、平均邻居数 80，
隐藏通道为 `128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e`；
dense 为每阶 32 通道。截断半径可以是标量或逐元素字典；加载旧模型时保留其
原始字典。`use_interpolation_out` 控制末层插值 MLP，默认关闭；训练过该选项
的 dense 检查点必须继续使用原值。

完整数据入口例子见 [examples/unitb](../examples/unitb/README.md)。
电荷头和训练加噪属于模型/训练配置，不因选择 embedding 而自动启用。

## PDQ-MoE

H₀-routed shared-basis mixture of experts (PDQ-MoE)，即 H₀ 先验路由的共享基底专家混合，在每个被路由的 SO(2) 线性层使用

\[
W(e)=W_s+\sum_{k\in\operatorname{top-2}(e)}g_k(e)P D_k Q^\mathsf{T}.
\]

`W_s` 是每条边都使用的满秩共享专家；`P`、`Q` 是本层专家共用的秩 64 基底；
每个 `D_k` 是独立的 64×64 核心。实际秩上限为 `min(in_features,out_features)`。
门控输入是键型 one-hot 编码与该边 H₀ 块的 CG Gram 旋转不变量，4 选 2，
选中权重重新归一化，在激活之前混合。

代码类名为 `PDQMoE`、`PDQMoERouter`、`PDQMoERouting`，模块为
`dptb.nn.pdq_moe`。基底只在同一个线性层内部共享。实现按专家合成权重，
激活空间分派不会为每条边展开一份完整权重。该参数化减少可训练参数，
并不意味着分组 GEMM 浮点运算量或训练时间必然减少。

| 配置 | 默认 | 含义 |
|---|---|---|
| `num_experts`, `num_shared_experts`, `top_k` | 4, 1, 2 | 路由专家、常开共享专家、选中数量 |
| `expert_parameterization` | `pdq_moe` | 共享基底，或 `full` 独立完整矩阵 |
| `expert_rank` | 64 | 共享基底秩 |
| `router_input` | `onehot_prior` | 可选 `onehot`、`onehot_r`，分别去先验或改为距离基函数 |
| `router_gate` | `renorm` | `full_softmax` 保留全局 softmax 的选中概率质量 |
| `expert_mixing` | `pre_activation` | 论文对照可用 `post_activation_shared` |
| `edge_router_top1_mode` | `legacy` | `switch` 用全局 softmax 的 top-1 概率，无共享专家 |
| `so2_moe_layers` | `all` | 指定使用路由专家的零起始层索引 |
| `edge_router_route_drop_p` | 0 | 训练时按结构丢弃路由分支，验证关闭 |

全专家软门控可设 `top_k=num_experts`。`full_softmax` 的选中权重和不保证为 1，
不能把共享专家重复折进每个槽；`post_activation_shared` 保持单独共享分支。
Switch 需要 `top_k=1`、`num_shared_experts=0`，并使用其自己的原始门控设置。

论文 MoE 对照还有独立的 `structure_mole` 策略，由构建时选择：
`route_scope=structure` 使用整图描述子的四专家软门控，
`route_scope=constant` 使用无条件的单一共享核心。两者均保留一个满秩共享专家。
默认关闭；`execution=merged_core` 先混合低秩核心，结构模式另有
`execution=reference` 对照路线。参数与训练统计缓冲区沿用原名，
统计按训练集中的结构等权拟合；加载检查点保留其已有统计。
策略实现位于 `unitb_structure.py`，UniTB 主 forward 不增加结构路由分支。
它与冻结的历史 graph-router API 是两个独立接口。

选择部分 `so2_moe_layers` 时，未选中的层只有共享参数，不创建空的路由参数。
它们保持径向条件、残差和输出契约。层选择改变初始化随机数消耗，因此改变
层配置不等于可以严格加载任意已有多专家检查点。

共享核心模式中，`weight_experts` 是可微合成结果，不是叶参数；训练或缩放应
操作 `core_experts`，或调用 `scale_expert_weights_()`。优化器模式应把
`core_experts` 视为专家参数，`basis_left/right` 视为普通共享参数。
`full` 与 `pdq_moe` 的参数结构不同，不能通过改配置强行互载。

## 先验与电荷平衡头

数据提供 `node_h0`、`edge_h0`；也可以用 `h0_node_key`、`h0_edge_key`
指向兼容表示的其他物理先验。公共模块 `prior_common` 负责 AO/CG 表示与
排序投影。路由器直接读取显式先验，缺失时报错，不回退到监督标签。
X1 的时间条件在普通监督训练/验证中均使用 t=0。

```json
{"shift_head": {"mode": "atom", "response": {"kind": "qeq"}}}
```

`ChargeHead` 读取最后交互层输入的标量特征，求解满足每个结构
`Σq=0` 的电荷平衡。默认 `qeq_local=true` 输出局部读出加 `κΓq`，
`qeq_local=false` 保留原来的静电响应形式。默认输出尺度为 0.1。
纯局部物理对照使用 `kind=context, local_only=true`，去掉 `κΓq`。

该头需要 compact SOC uu-real 输出、直接 e3tb 变换及物理 overlap。
物理 overlap 字段是 `phys_node_overlap`、`phys_edge_overlap`，不能用 embedding
的 `node_overlap/edge_overlap` latent 代替。修正为 onsite `v_i S_ii` 和
hopping `(v_i+v_j) S_ij/2`，遵循原 active-edge 与距离专家掩码。
势是可学习修正坐标，不是实测 DFT 电势。

数据 sidecar 必须与主 LMDB 的 shard 名称、key、结构指纹和边顺序一致；
通过 `overlap_sidecar_root` 为训练/验证分别指定路径。头关闭时不读 sidecar。
支持 QEq 与纯局部对照；旧方法名加载同类响应头时保持原参数路径。
加性势移、子晶格响应及辅助标签训练由历史归档版本提供。

## 纯先验加噪

`train_options.prior_noise_augmentation=true` 使用独立
`PriorNoiseAugmentation`，结构噪声采样由 `StructuredNoise` 实现。
保留历史 `flow_options` 中的采样配置；`flow_options.enabled=false`。
`te_prior_sigma` 是噪声尺度：onsite 配方为 0.5，hopping 配方为 5。
`te_prior_scale_reference=target` 使用当前监督 ΔH 的 RMS。
标准训练 schema 的 `te_prior_scale_reference` 默认是 `residual`、
`te_prior_sigma` 默认是 1，因此 X1 示例显式设置这两个字段。
同时设置 `flow_options.t_max=0`；`t_min` 与 `t0_probability` 保持为 0。
历史配置中的关闭状态仍可读取；启用 flow 训练或 ODE 会明确报错。

仅在训练且模型处于 training 模式时，对已有专家掩码内的先验字段加噪，
并写入零时间条件；标签不变，验证不耗随机数。采样保留原 node→edge 顺序、
设备、dtype 和 RNG 消耗。它不构造 flow 损失或 ODE 求解器。

## 检查点与其他 embedding

旧 `lem_moe_v3_edge_h0` 方法在构建时翻译为 UniTB 的兼容入口，保留历史默认值。
旧 `MOLELinear`、`shared_core`、`mole_expert_*`、`edge_router_*` 仍能读，
构建时给出 INFO 提示。所有参数路径保持原样，包括 `basis_left`、`basis_right`、
`core_experts`、`weight_shared`、`layers.*` 与 `shift_head.response_net.*`。
加载使用 `strict=True`，用户无需编辑检查点。

`lem`/`slem` 是不接受先验的基线；`lem_prior`/`slem_prior` 接受先验但不使用
PDQ-MoE。历史图路由 `LemMoEV3`/`LemMoEV3H0` 保留自己的 forward 与层接口，
并复用共同层；它们与逐边 UniTB 的数学不同。

## 可选 CUDA 后端

DeePTB 不依赖 SO2CUDA 才能运行。安装 SO2CUDA 0.2.0 后，默认 `SO2_CUDA_BACKEND=auto` 对受支持的 CUDA/FP32 输入自动使用加速；`SO2_CUDA_BACKEND=off` 选择 PyTorch 参考路径。完整开关和限制见 [SO2 后端](so2_backend.md)，dense/X1 的三行加速测试见 [SO2CUDA README](https://github.com/Franklalalala/SO2CUDA#dense-与-x1-加速测试)。

单专家 dense 通过 `so2_backend.dense_forward` 使用 `dense_pairs`；X1 使用 activation fused-P0 与分组 GEMM。旧接口名保留用于兼容，不能据名称推断已执行的路线。CPU 与 CUDA 在 cutoff 边界附近的活动边选择可能不同，个别边输出不等价；检查点等价性按同设备验证。

## English

UniTB unifies dense and X1 prior-conditioned embeddings. X1 uses an
H₀-routed shared-basis mixture of experts (PDQ-MoE):
`W(e) = W_s + Σ_{k∈top-2(e)} g_k(e) P D_k Qᵀ`.
Each routed SO(2) linear has one full-rank shared expert, shared rank-64 bases,
and four independent cores. Bond-type and H₀ CG Gram features select two experts;
renormalized gates mix linear outputs before activation.

Use `{"method":"unitb"}` for X1 defaults or add `"num_experts":1` for dense.
Explicit options override defaults. Charge equilibration and training-only prior
noise are configured separately. The charge solve is neutral per structure;
`qeq_local` controls local readout plus electrostatic response, and the
`context/local_only` control removes that response. Physical overlap must be
aligned with the exact graph and compact AO representation.

Legacy configuration names are translated with INFO diagnostics. Parameter names,
checkpoint shapes and strict restoration are retained. Shared-core and full
expert banks remain distinct mathematical parameterizations. Historical graph
routing retains its separate API while sharing the same layers. CPU/reference
availability and CUDA acceleration depend on the selected SO2 backend; selecting
a backend never authorizes changing activation order or gate normalization.

The optional `structure_mole` paper control selects structure-wide soft routing
or a constant shared core at construction. Its parameter paths and frozen
training-statistics buffers remain checkpoint-compatible. The training examples
are minimal configuration templates with placeholder data paths, not complete
production training recipes. For training-only prior noise, explicitly set
`flow_options.t_max=0` along with the intended TE mode and noise scale.
