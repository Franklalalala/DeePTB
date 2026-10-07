# UniTB 训练入口示例

`input.json`（UniTB）与 `dense/input.json`（UniTB-dense）通过仓库的标准训练配置入口读取。
embedding 只列出与默认不同的选项。示例是 H/O、compact uu-real、H−H0 监督，
请把 basis 改成数据的真实轨道基组，保留其 SOC 与表示约定；不要只替换元素名称。

这里的一轮训练、小批次和 Adam 设置用于检查入口，不是完整生产训练配方。

将 `DATA_ROOT/train`、`DATA_ROOT/validation` 换为 LMDB 数据目录。
主记录须包含保存的边图、Hamiltonian 监督目标和 `hamiltonian_0` 物理先验，
加载后得到 `node_h0/edge_h0`。示例 target_kind=h0res，监督对象是 ΔH。
逐元素截断半径须与数据配方相符，可在 embedding.r_max 中提供字典。

UniTB 示例还需将 `OVERLAP_ROOT` 换为物理 overlap sidecar 根目录。每个 split 的
shard/key、结构身份和边顺序须与主数据对应；侧车提供物理 `S`，不是网络 latent。
两份示例通过 distance_ranges 只监督 onsite。UniTB 示例启用 QEq 局部电荷头和
onsite 尺度的训练加噪。hopping 配方把
`te_prior_sigma` 改为 5，并使用主训练入口已有的距离/标签掩码；不要靠加噪配置
代替距离专家的监督掩码。完整训练超参数与批次设置由实验配方决定。

纯先验加噪固定在 `t=0`：例子显式设置 `flow_options.t_max=0`，
`t_min` 与 `t0_probability` 沿用默认的 0。仅关闭 flow 不会把其默认
`t_max=0.999` 改成 0；加噪采样器会检查这三个值。

可用标准 CLI 运行 `dptb train input.json -o output`。CPU 参考检查需选择
`common_options.device=cpu` 与可用的参考 SO2 配置；CUDA 使用可选 SO2CUDA。
这些数据路径是占位符，仓库不附带训练数据或已训练权重。

结构、兼容名称与选项见 [docs/unitb.md](../../docs/unitb.md)。
