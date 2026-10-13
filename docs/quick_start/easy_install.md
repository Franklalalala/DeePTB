# Installation

Use a separate Python environment. The supported Python range and dependencies are specified in the root `pyproject.toml`. Install [PyTorch](https://pytorch.org/get-started/locally) for your device, followed by a matching `torch-scatter` and DeePTB:

```bash
git clone --branch 1006-stable https://github.com/Franklalalala/DeePTB.git
cd DeePTB
python docs/auto_install_torch_scatter.py
python -m pip install -e .
```

SO2CUDA is an optional acceleration component and requires a CUDA toolchain compatible with PyTorch:

```bash
python -m pip install -e '.[so2]'
```

Use `dptb --help` to inspect the commands. After installation, run the short suite:

```bash
python tools/test.py
```

See [TESTING.md](../../TESTING.md) for the full testing strategy. Real-data checks, hardware benchmarks, and LoopSCF use their own assets and test entry points. The [maintenance release guide](../1006-stable.md) describes the component layout.
