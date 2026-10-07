# 双阶段先验模型工具

本目录保留 H0/NACF 先验与 residual 目标的双阶段配置、数据检查和训练入口。生成器使用 `configs/base.json` 的固定模型容量与训练控制；修改容量或实验变量时应明确记录，不能把默认值当作所有模型的通用设置。

```bash
python tools/general_model/make_configs.py --data /path/to/data --output /path/to/configs
python tools/general_model/check_dataset.py --input /path/to/configs/h0_h0res.onsite.s1.json --output /path/to/loader.json
python tools/general_model/run_stage.py --input /path/to/configs/h0_h0res.onsite.s1.json --output /path/to/s1
python tools/general_model/run_stage.py --input /path/to/configs/h0_h0res.onsite.s2.json --s1-checkpoint /path/to/s1.pth --output /path/to/s2
```

三条数据路线分别是 `h0_h0res`、`nacf_nacfres` 和 `nacf_h0res`。先验选择与目标选择独立：数据字段需要按相应路线保存，原子、边和周期位移保持一致；不能对不匹配的 residual 重复加回先验。

S1 为 pairwise 分支，S2 加载匹配且完成的 S1；同阶段恢复使用 `--restart`。`--smoke-steps` 保留原调度器设置，有限运行只验证入口、梯度和检查点行为。

`verify_checkpoint.py CHECKPOINT --steps N --output RECEIPT` 检查有效训练步、参数有限性及该工具的固定专家结构。它不证明预测精度或数据游标的精确恢复。配置 manifest 中的生成状态也不表示 GPU 训练已验证。

正式运行保留有效输入、模型、数据身份及验证记录；生成配置、运行结果与检查点存放在仓库外。
