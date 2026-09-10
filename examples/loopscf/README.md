# LoopSCF

LoopSCF adds Mulliken electron-population feedback to a residual-H0 Hamiltonian model. It currently
supports non-SOC, spin-degenerate, single-GPU training with the non-flow model.

The alternative [shared equivariant latent corrector](LATENT.md) encodes once,
retains full node/edge irreps, and compares BPTT, detached-state, and reset-state
updates without occupation feedback. Select it with `--architecture latent`.

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
checkpoints are loaded strictly after constructing their architecture. The
current AO trace feedback (v3) rejects old adapters even if weight shapes match;
start from the original base checkpoint and retrain the adapters. A change in
population or feedback representation changes the learned input distribution.

Add `--no-feedback` to train an otherwise identical repeated-forward control.
Keep the same initial base, trainable heads, seed, batch order, K, loss weights
and number of optimizer steps. This control still computes the physical state,
so its wall time is not a direct-model latency baseline. Reuse the same flags on
restart; a timestamped protocol sidecar records each invocation. Turning off
feedback only at evaluation is a sensitivity test, not this trained control.

Each graph must supply explicit `nelec`, matching H0 and overlap, and aligned
atom/edge order. Derive electron counts from the actual pseudopotentials and
total charge. Hamiltonian-only training can use records without spectral fields.
The loss target must be residual AO `dH`; spectral loss reconstructs `H0 + dH`
on both sides. `residual_hamiltonian=false` only disables loader subtraction:
historical LMDB records may already contain residual targets. Verify their
provenance and that reconstructed label bands match the stored DFT spectrum.

## Package interface

`dptb.nnops.loopscf` provides feedback adapters, occupation solving, sparse
k-space assembly, and stepwise training. Use
`install_working_memory_true_diag(..., collect_diagnostics=True)` to collect
per-step populations in `_loop_q` during evaluation and per-graph/per-k overlap
diagnostics in `_loop_overlap` (eigenvalue range, condition number, retained rank,
discarded modes and spin-degenerate capacity). `overlap_cutoff` defaults to the
fixed absolute value `1e-5`. Conditioning is checked even if Cholesky succeeds.
Projection removes basis directions and changes the eigenproblem; always report
its incidence and failed-structure denominator, and preselect cutoff sensitivity
checks rather than tuning the cutoff against the desired spectral error.

The representation chain is explicit: equivariant hidden irreps → output RME →
`E3Hamiltonian` → packed AO blocks. The model must use `transform=True`. Feedback
extracts `trace(H_ll)/sqrt(2l+1)` from equal-angular-momentum shell pairs, then
injects these invariants only into hidden `0e` channels. A coordinate selected
from a flattened p/d/f AO block is not an invariant. Active edge indices are
applied in order, including when the mapping is a full-length permutation.

Physical `node_h0`/`edge_h0` remain AO blocks for assembly and losses. The legacy
H0 embedding consumes RME irreps, so LoopSCF computes a separate H0 feature copy
using the existing inverse CG transform, caches it for the batch, and restores
the physical AO fields in each prediction. Head injection uses the final layer's
irreps; MoE injection uses the initialization layer's irreps. Their scalar counts
need not match (17 versus 10 in the inspected production configuration).

This H0 correction changes even K1 relative to the old executable. A base
checkpoint remains a weight initialization, not a promise of identical base
predictions. Record the original executable baseline and the corrected-input
K1 baseline separately; re-establish H-only training before testing WM gains.
Use this LoopSCF launcher with `--K 1 --no-feedback` for that corrected H-only
control; the generic historical training entrypoint does not add this H0 conversion.

The launcher installs process-local Trainer and FW10 compatibility hooks.
The Trainer uses an explicit backward-completion flag for stepwise losses;
ordinary training requires no LoopSCF hook installation.

Evaluation integrates charge on deterministic Sobol Brillouin-zone samples,
separately from the band plotting path. Check k-point convergence for the target
systems. `q` is an electron population, not signed ionic charge; H0 is the physical
prior, not necessarily the frozen neural base prediction.

Stepwise backward optimizes the detached sum of per-step losses. It is not full
BPTT. Single-GPU non-flow training is supported; the asynchronous self-consistency
loss is rejected rather than silently skipped. AMP/DDP/gradient-accumulation
extensions require a different backward integration.

## Spectral evaluation

For a complete development split, use the chunked evaluator:

```bash
python examples/loopscf/evaluate.py --input evaluation.json --checkpoint base.pth --compare-original --K 1 --output baseline.json
python examples/loopscf/evaluate.py --input evaluation.json --checkpoint trained.pth --base-model base.pth --label wm --K 1 2 4 --rotation-indices 0 11 --output wm.json
```

The default split is `validation`; indices always refer to that configured
dataset. Use `--no-feedback` for the matched control. Every output records
matrix MAE, complete-path FW10, label closure, overlap projection and cumulative
forward time per structure. `--k-chunk` bounds path-assembly memory without
changing the global path alignment. The requested rotations transform geometry,
H0, target H and S together, and check relative AO output error below 1e-3.
Timing includes compilation for new shapes and is diagnostic on a shared GPU.

Closure mismatches above 1 meV are retained with `label_closure_pass=false` and
listed in `closure_mismatches`. They are not a successful physical-label audit.
Report the full set and the explicitly identified closure-consistent subset.
Execution failures are recorded separately and cause a nonzero exit status.

```bash
python examples/loopscf/summarize_experiment.py --inputs baseline.json control.json wm.json --output summary.json
python examples/loopscf/summarize_experiment.py --inputs baseline.json control.json wm.json --output closure_summary.json --closure-only
```

The report separates equal-structure means from valid-element-weighted MAE,
retains per-structure adverse tails, and refuses paired statistics for unequal
index sets. Input files must describe the same dataset. Closure filtering is
explicit, requires closure evidence, and preserves excluded IDs and failures.

For bounded matched training, set a finite `train_options.num_epoch` and use
the existing epoch checkpoints. For example, one epoch of 69 structures with
batch size one is 69 optimizer updates. Both arms start with `--init-model`
from the identical checkpoint, reset optimizer state, and share one config:

```bash
python examples/loopscf/train.py --input pilot.json --output control --init-model baseline.pth --base-model original.pth --mode head --K 2 --no-feedback --trace-batches
python examples/loopscf/train.py --input pilot.json --output wm --init-model baseline.pth --base-model original.pth --mode head --K 2 --trace-batches
```

The launcher uses one worker per loader by default (`--loader-workers`), records
trainable parameter names/shapes, and optionally hashes the actual graph order
and random k draws. Check these traces before attributing differences to WM.
The K1 difference from the starting checkpoint measures fine-tuning; K2 minus
K1 within one checkpoint measures iteration; WM versus the matched control
measures the feedback increment. These answer different questions.

For a bounded same-checkpoint comparison of original and corrected H0 inputs:

```bash
python examples/loopscf/compare_h0.py --input /path/to/train.json --checkpoint /path/to/base.pth --output /path/to/paired.json --indices 0 11
```

This CUDA diagnostic uses residual labels, checks their reconstructed spectrum
against stored DFT bands, and reports separate onsite/hopping MAEs plus legacy
FW10. It runs no training and uses the configured training dataset; its results
do not establish held-out accuracy. The checkpoint must be an original base
without WM adapters.

The existing `fw10` loss remains **legacy VBM-aligned FW10**. Its fixed occupied
band index is not a general metallic Fermi-level definition. The evaluation-only
helpers in `dptb.nnops.loopscf.metrics` offer an additional protocol:

```python
mu_p = fermi_level(predicted_bz, nelec, k_weights=weights, smearing=0.05)
mu_r = fermi_level(reference_bz, nelec, k_weights=weights, smearing=0.05)
error, count = mu_aligned_band_error(predicted_path, reference_path,
                                    mu_pred=mu_p, mu_ref=mu_r, window=10.0)
```

Import both functions from `dptb.nnops.loopscf.metrics`. Energies, smearing and
window are in eV. BZ integration points and plotting paths are separate inputs;
the same quadrature, electron count and smearing apply to prediction/reference.
At zero smearing, partially filled shells define mu; an insulating gap uses its
midpoint. Empty/full finite bases have no unique finite mu and are rejected.
Do not pass synthetic padding eigenvalues from projected spectra. Validate
label H/S against DFT files before attributing remaining discrepancies to the
model. Report both metrics, the window count and metal/insulator strata.

## Validation

```bash
python tools/test.py
```

The short suite covers numerical, data, model, and training contracts.
Predictive gains from charge feedback require matched held-out experiments.

For these feedback changes, run the focused behavioral checks:

```bash
python tools/test.py dptb/tests/test_loopscf_feedback.py dptb/tests/test_loopscf_numerics.py dptb/tests/test_band_losses.py dptb/tests/test_loopscf_evaluation.py dptb/tests/test_loopscf_report.py
```

When a configured GPU and real data are available:

```bash
python examples/loopscf/validate_model.py --input /path/to/train.json --checkpoint /path/to/base.pth --output /path/to/check --indices 0 11
```

This uses two diagnostic structures, deliberately nonzero random v3 adapters,
K1–4 rotations and one optimizer update per arm. It writes an evidence JSON;
it does not train a production checkpoint or establish predictive improvement.

## Pretrained whole-stack route

The whole-stack recurrent training interface is documented in
[STACK.md](STACK.md), with a separate `anneal_stack.py` entrypoint. It trains
the existing pretrained LEM stack and an invariant exit gate.
