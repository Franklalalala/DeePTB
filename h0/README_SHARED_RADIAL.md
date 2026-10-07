# 共享径向表与投影子搜索

共享 store 去除不同组合中完全相同的 S/T/Q 曲线，以 uniform cubic Hermite endpoints、尾部系数及稀疏位修正保存。原网格、系数、物种、SOC 复数 D 矩阵、索引和生成器契约保持一致；not-a-knot AO spline 按原格式保存，不转换到 Hermite codec。

在已配置的 H0 环境中显式准备新目录：

```bash
python -B h0/compact_tables.py /path/to/original/offline_tables /path/to/new/shared_tables
```

原 store 不变。每个组合的解码结果通过原状态 checksum 后才发布可发现的 manifest；不完整输出保留用于诊断，重试使用新目录。运行时通过 `shared_radial.json` 发现完整 store，加载后展开原 FP64 张量。SQLite 只读，解码系数字节参与 hash 检查，损坏记录或不相容浮点算术会报错。

```python
result = assemble_h0(
    ..., offline_table_dir='/path/to/new/shared_tables',
    projector_reuse_max_mb=256,
    projector_search='anchor', spatial_backend='indexed',
)
```

默认搜索为 `projector_search='midpoint'`。`anchor` 以原子 `ci` 为中心、`orbital_cutoff_i + max_projector_cutoff` 为半径；贡献投影子必须与 AO i 重叠，因此处于此候选球内。原子 i 的所有出边复用排序后的候选集，原两次物种距离判断和原生收缩确定实际贡献，包括重复周期镜像。原子与 translation 顺序保持一致，仅缓存当前原子的候选集。

共享格式只压缩已生成曲线；新网格与未见组合仍需显式准备。CUDA 使用展开后的四系数表，磁盘压缩不代表 GPU 内存等比例减少。Q cache 预算只覆盖保留的 Q 张量，不是完整组装工作集；`kernel_nonlocal_seconds` 包括 wrapper 到已完成 CPU 结果的范围，S/T 调用数和计时单独报告。
