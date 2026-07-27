<p align="center">
  <img src="docs/deeptb-logo.png" alt="DeePTB Logo" width="720" />
</p>

# DeePTB 0726-light

`0726-light` is the focused DeePTB development line for the actively
maintained E3-equivariant Hamiltonian and quantum-operator workflows. It is
derived from `0721-stable` and intentionally removes historical model,
transport, conversion, and dataset implementations that are outside the
current development path.

## Maintained scope

- LEM/MoE-v3, H0, pair, nonlinear, in-frame, and EMol embedding endpoints.
- `e3tb` and `block_native` prediction heads.
- Hamiltonian/SOC construction and band or AO-block output.
- LMDB-backed datasets and record/materialization pipelines.
- Conditional flow matching, MeanFlow, block ODE, physical priors, and
  current trainer/restart/distributed workflows.
- Current SO(2), grouped-GEMM, and optional `so2-cuda-ops` acceleration.

The large, recently maintained `dptb/nnops/flow.py` implementation is kept
unchanged from `0721-stable`.

## Intentional incompatibilities

This branch does not provide NNSK/SKTB/DFTB/MIX models, NEGF transport,
DOS/Fermi-surface integrations, legacy non-LMDB dataset classes, historical
embedding aliases, or the old conversion/template CLI commands. Unsupported
configuration and checkpoint routes fail early with a clear error.

See [`LIGHTWEIGHT_SCOPE.md`](LIGHTWEIGHT_SCOPE.md) for the exact retained and
removed surface.

## Installation

Use Python 3.9-3.12 and install PyTorch for the intended CPU/CUDA platform
first:

```bash
conda create -n dptb python=3.10
conda activate dptb
pip install "torch>=2.0"
python docs/auto_install_torch_scatter.py
pip install .
```

`torch-scatter` is a required runtime dependency of the retained LEM and loss
paths. It is installed separately because its wheel must match the installed
PyTorch CPU/CUDA build.

The optional optimized SO(2) kernels are supplied by
[`so2-cuda-ops`](https://github.com/Franklalalala/SO2CUDA). CPU imports and
standard fallback routes remain usable when that extension is unavailable.
Install them together with DeePTB using `pip install ".[so2]"`.

The public `*_openequi*` embedding methods additionally require Python >=3.10,
PyTorch >=2.4, and a Linux NVIDIA/AMD GPU toolchain. Install OpenEquivariance
with `pip install ".[openequi]"`.

## Commands

```bash
dptb train input.yaml -o output
dptb test test.yaml --init-model checkpoint.pth --output output
dptb run run.yaml --init-model checkpoint.pth --output output
```

Use the strict schema reference under `docs/input_params/` for current
configuration fields.

## Validation

```bash
python -m pytest dptb/tests
```

CUDA-only kernel tests skip automatically when the required device or
extension is unavailable.

## Citation

For the maintained E3 model family, cite:

> Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, “Learning Local
> Equivariant Representations for Quantum Operators,” ICLR 2025 Spotlight.
