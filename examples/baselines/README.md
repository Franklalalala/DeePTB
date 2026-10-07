# LEM / SLEM 先验基线示例

在安装好 DeePTB 的环境中，从仓库根目录运行：

```bash
python examples/baselines/run_smoke.py examples/baselines/lem_prior/input.json
python examples/baselines/run_smoke.py examples/baselines/slem_prior/input.json
```

两个 `input.json` 包含可直接交给 `build_model` 的 `common_options` 和
`model_options`。`smoke_options` 由示例脚本读取。脚本在 CPU 上创建一个含三个
碳原子的图，提供合成 H0 先验，并执行 20 次 Adam 更新；输出初始和最终 loss。
零目标仅用于验证建模、先验输入、前向与反向的接口，loss 下降不表示预测精度。
运行不需要下载数据。可用 `--output <path>` 保存 JSON 回执。

真实数据须提供和图的节点、边顺序一致的 `node_h0` / `edge_h0`，形状分别为
`[N, idp.reduced_matrix_element]` / `[E, idp.reduced_matrix_element]`，内容采用
`OrbitalMapper` 的 packed AO-product 坐标。默认 `h0_ao_cg: true` 进行 CG 投影。
输入已经是同一 mapper 的 coupled RME 时，批次设置 `_h0_coupled_rme: true`。
P 先验使用相同坐标契约，将 `h0_node_key` / `h0_edge_key` 设置为 `node_p23` /
`edge_p2`，并提供这些字段。先验必须独立于监督目标；两个示例关闭了目标回退。

`h0_init_scope` 可选 `none`、`node`、`edge`、`both`；`h0_merge_mode` 可选
`replace`、`add`。`h0_node_mode: direct` 从节点先验初始化；`self_edge` 从零长度
自边初始化节点，图没有自边时使用节点先验；直接节点来源仍执行训练回退检查。
scope 包含边时，找不到可用边来源会使节点、边都保持几何初始化；边有效而节点
缺失时仅节点保持几何初始化。接口默认允许 Hamiltonian / feature 回退，训练时
拒绝使用这类来源，除非显式允许；本示例关闭回退并提供完整的独立先验。
不提供任何先验配置选项或设置 `h0_init_scope: none` 可严格恢复旧几何 state_dict；
需要使用先验时，请显式配置 scope 或先验字段。

The examples build the complete DeePTB model and run 20 CPU optimizer updates on
a synthetic three-atom graph. They require no external dataset and establish
the prior input and gradient interfaces only. Packed AO-product priors must
follow the mapper and graph ordering; coupled RME inputs require the batch flag
`_h0_coupled_rme`. P priors use the same contract through configurable field names.
