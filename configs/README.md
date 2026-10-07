# 可复用配置片段

本目录提供需要合并到完整输入配置的模型或数据片段。实际使用时明确基组、数据目录、先验来源、目标与训练设置，使用配置校验后再加载数据。

- `physical_h0_flow_overlay.yaml`：物理 H0 flow 字段与模型先验输入键必须一致。
- `route_*.yaml`：已注册输出路线的接口示例，包括 RME、late-CG 和 AO projector；模型、预测头及输出空间需要匹配。
- `h_b0_block_ode_*.yaml`：block-ODE 的结构、目标空间、先验及约化配置。
- `n*_snippet.yaml`：输出头片段，需要合并有效的公共模型和训练配置。
- `p2_prior_non_soc_full_h_smoke.yaml`：CPU 小模型的 P2/Full-H 契约测试片段；其中未绑定来源的开发选项仅用于 smoke，正式输入需要固定实际先验来源 fingerprint。

这些片段不是完整数据集或已训练检查点，不承诺指定结构上的预测精度。通用 table、cache、sidecar 构建入口见 `tools/`，H0 安装和表准备见 `h0/README.md`。
