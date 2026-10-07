# LEM / SLEM 基线

`lem` / `slem` 使用上游实现，通过几何与原子类型生成特征，不读取先验输入。
`lem_prior` / `slem_prior` 是具有共享 H0 / P 输入接口的基线。不传任何先验选项，
或设置 `h0_init_scope: none` 时，不创建先验参数，并可 strict 加载原几何模型的
state_dict。启用先验可显式设置 `h0_init_scope: both`；已配置先验但未指定 scope
时，默认也是 `both`。启用后默认从
`node_h0` / `edge_h0` 读取 mapper 的 packed AO-product 坐标，P 先验可通过
`h0_node_key: node_p23`、`h0_edge_key: edge_p2` 使用；`h0_init_scope` 选择节点、
边或两者，`h0_merge_mode` 选择替换或相加，`h0_node_mode` 选择直接节点输入或
零长度自边。读取、合并、缺失字段与训练防护遵循生产 H0 接口：默认允许依次
回退到 `node_hamiltonian` / `edge_hamiltonian` 和配置的 feature 字段；训练中
使用回退来源会报错，除非显式设置 `allow_target_fallback_in_training: true`。
可用 `fallback_to_hamiltonian: false` 关闭回退。scope 包含边时，若找不到可用
边来源，节点和边都保持几何初始化；边有效但节点缺失时，仅节点保持几何初始化。
`self_edge` 按全图是否存在有效自边选择节点初始化，直接节点回退路径始终执行
来源检查，包括训练防护。旧配置的 `method: lem` / `slem` 若带
先验选项，模型构建会给出 warning 并映射到对应 `_prior` 名称；新配置应明确
使用 `_prior` 名称。加载原几何检查点时，若旧配置含先验选项而权重中没有先验
模块，加载器自动关闭先验初始化以保持原输出，并给出 warning；经过正常归一化
产生的等价配置也适用，真正改变配置仍严格检查。新 `_prior`
检查点缺少先验权重时严格报错。UniTB 是独立的生产 embedding，与这两类基线区分；其接口
说明随 UniTB 模块提供。最小 CPU 示例见 `examples/baselines/README.md`。

`lem` and `slem` retain the upstream geometry-based implementations and do not
read prior inputs. With no prior options, or scope `none`, `lem_prior` and
`slem_prior` create no adapter state and strictly load their original geometry
state dictionaries. Set scope `both` to enable the shared H0/P input
contract: mapper-packed AO-product features, configurable node/edge field names,
node/edge initialization scope, and replacement or additive merging. Configured
priors default to scope `both`. The production input contract enables Hamiltonian
then feature fallback by default, guarded against use during training unless
explicitly permitted. If an enabled edge source is missing, both initial states
stay geometric; otherwise a missing node source preserves only the node state.
Self-edge mode also evaluates the guarded direct-node fallback before selection.
Legacy `lem`/`slem` configurations containing prior options are mapped to the
corresponding `_prior` method with a warning. Geometry-only legacy checkpoints
automatically disable prior initialization for equivalent raw or normalized
options to preserve their original outputs; genuine overrides remain strict;
new prior checkpoints with missing prior weights fail strict restoration.
UniTB is the separate production
embedding, whose interface is documented with that module. The CPU examples in
`examples/baselines` demonstrate the baseline input and optimization interfaces.

启用先验新增 `prior_inputs.node_projector.{weight,bias}` 与
`prior_inputs.edge_projector.{weight,bias}`。e3nn `Linear` 的 weight 为标准正态
初始化，bias 为零；先验模块在几何参数之后构造，保留几何初始化顺序。每个投影
的 weight 数量由 mapper 输入与初始化输出中相同 irrep 的通道乘积之和决定，
bias 只用于输出 `0e` 通道。版本标记和 output mask 是 buffer，不是可训练参数。
无先验选项或 scope `none` 不增加这些状态。旧几何权重从未学过使用先验，启用
先验是新增能力。

Enabled priors add node and edge equivariant projector weights and biases after
the geometric parameters. e3nn Linear initializes weights from a standard
normal distribution and biases to zero. Weight counts sum the products of
matching input/output irrep multiplicities; biases cover output `0e` channels.
Version markers and output masks are buffers. Disabled priors add no state;
legacy geometry weights have never learned to use priors.
