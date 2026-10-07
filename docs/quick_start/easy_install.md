# 安装

使用独立 Python 环境，支持的 Python 范围与依赖见仓库根目录的 `pyproject.toml`。先按设备安装 [PyTorch](https://pytorch.org/get-started/locally)，然后安装匹配的 `torch-scatter` 和 DeePTB：

```bash
git clone --branch 1006-stable https://github.com/Franklalalala/DeePTB.git
cd DeePTB
python docs/auto_install_torch_scatter.py
python -m pip install -e .
```

SO2CUDA 是可选 CUDA 加速组件，需要与 PyTorch 相容的 CUDA 工具链：

```bash
python -m pip install -e '.[so2]'
```

使用 `dptb --help` 查看命令入口。安装后可执行短测试：

```bash
python tools/test.py
```

完整测试策略见根目录的 `TESTING.md`；真实数据、硬件基准和 LoopSCF 使用各自的资产与测试入口。版本结构见 [维护版本说明](../1006-stable.md)。
