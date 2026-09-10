# LoopSCF

LoopSCF adds charge feedback to a residual-H0 Hamiltonian model. It currently
supports non-SOC, spin-degenerate, single-GPU training with the non-flow model.

## Training

Run from this repository in a configured DeePTB / PyTorch / SO2 environment:

```bash
python examples/loopscf/train.py --input /path/to/train.json --output /path/to/run --init-model /path/to/base.pth --mode head --K 2
```

`head` trains output heads and feedback adapters. `moe` also enables embedding
layers and routers. `--n-k-train` controls the occupation sampling count (default:
5). Use `--restart` to restore training state, or `--init-model` to initialize a
new run. When either checkpoint contains saved feedback adapters, provide its
matching base checkpoint with `--base-model` (or `LOOPSCF_BASE_CKPT`). Adapter
checkpoints are loaded strictly after constructing their architecture.

Each graph must supply explicit `nelec`, matching H0 and overlap, and aligned
atom/edge order. Derive electron counts from the actual pseudopotentials and
total charge. Hamiltonian-only training can use records without spectral fields.

## Package interface

`dptb.nnops.loopscf` provides feedback adapters, occupation solving, sparse
k-space assembly, and stepwise training. Use
`install_working_memory_true_diag(..., collect_diagnostics=True)` to collect
per-step populations in `_loop_q` during evaluation.

The launcher installs process-local Trainer and FW10 compatibility hooks.
The Trainer uses an explicit backward-completion flag for stepwise losses;
ordinary training requires no LoopSCF hook installation.

Evaluation integrates charge on deterministic Sobol Brillouin-zone samples,
separately from the band plotting path. Check k-point convergence for the target
systems. Changed occupation semantics make previous adapter checkpoints warm
starts; their previous numerical trajectories are not reproduced exactly.

## Validation

```bash
python tools/test.py
```

The short suite covers numerical, data, model, and training contracts.
Predictive gains from charge feedback require matched held-out experiments.
