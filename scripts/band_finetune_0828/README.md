# Band-structure fine-tuning (0828–0829)

Fine-tuning a converged Hamiltonian model on the downstream band task, following
[NextHAM](https://arxiv.org/abs/2509.19877). Band labels are expensive, so the
first goal is a pipeline that is verifiably correct rather than one that scores well.

## What landed in `dptb/`

| File | Change |
|---|---|
| `nnops/loss.py` | `eig_ham_h0res`, `hamil_abs_gauged`, `nextham_kspace` |
| `data/dataset/record_pipeline.py` | `_decode_band_targets` — reads `node_overlap` / `edge_overlap` / `kpoint` / `eigenvalue` from LMDB records |
| `utils/argcheck.py` | argument groups for the three losses |
| `plugins/monitor.py` | iteration-level validation now honours `valid_fast` |

### `hamil_abs_gauged`

$H \to H + \mu S$ leaves the physics untouched, so a model that predicts the
residual $\Delta H$ should not be penalised for a constant orbital-energy offset.
$\mu$ is solved in closed form against the overlap, detached, clipped to $\pm 1$ eV,
and the shift is applied to the **target**:

$$\mu=\frac{\langle \Delta H_{\text{pred}}-\Delta H_{\text{ref}},\,S\rangle}{\langle S,S\rangle},
\qquad \mathcal{L}=\operatorname{hamil\_abs}\!\left(\Delta H_{\text{pred}},\ \Delta H_{\text{ref}}+\mu S\right)$$

Measured on 8 held-out structures: $\lvert\mu\rvert$ median 18.6 meV, gauge gain
median 1.0171, and feeding the label as the prediction gives $5\times10^{-7}$.

> `S` must be read from `ref_data`. The model's forward pass overwrites
> `edge_overlap` with a 128-wide internal tensor; the loss asserts the RME width
> so this fails loudly instead of silently supervising the wrong thing.

### `nextham_kspace`

NextHAM's k-space projection loss (`train_val.py:228-413`). Per step one random
k-point; $U$ comes from diagonalising the **label** $H(k)$ under `no_grad`; the
prediction is projected onto occupied ($P$), virtual ($Q$), and cross ($PQ$) blocks.

Deviations from the reference implementation, each deliberate:

- **k-points** — NextHAM samples from a stored $4\times4\times4$ grid; here
  $H(R)\to H(k)$ is analytic, so any k is available and none is cached.
- **$U$** — computed online rather than read from `wfc.pth`.
- **P/Q windows** — NextHAM fixes `band_cut_index` once at $\Gamma$ and takes
  $Q$ as everything above. Our spectra reach $+700$ eV, where $Q$ would swallow
  near-singular states, so P and Q are set per-k around $E_F$.
- **$E_F$** — derived from `nelec`, which the records carry; they do not carry $E_F$.

Checks: $\max\lvert U^\dagger S U-I\rvert=9.2\times10^{-6}$; label-as-prediction
drives P/Q/PQ to 0; the k-terms take a 0.25% share of the weighted total.

> Add $H_0$ back before diagonalising. Without it the residual spectrum spans
> only ~5 eV, the Q window comes out empty, and all three terms are identically
> zero while the logs still report the loss as active.

## Scripts

Written against pro6000 (`/data/wgh/`); paths are hardcoded and are kept as a
record of how the numbers above were produced.

| Script | Purpose |
|---|---|
| `csr_to_rme.py` | ABACUS CSR → RME blocks without reading `running_scf.log`; orbital metadata is rebuilt from `atomic_numbers` + basis. Bit-identical to `_abacus_parse` on 15 structures. |
| `build_train_with_S.py` | Merges overlap blocks into the training LMDB |
| `build_band_ds.py`, `split_ds.py` | Band-label dataset construction |
| `verify_chain.py` | Label-side round trip: $H_0+\Delta H$ + analytic $S$ → generalised eigenproblem → cached DFT bands. `fw_10` median $4.0\times10^{-5}$ eV. |
| `ghost_diag.py` | Per-k `fw_10` spread, for locating isolated k-point blow-ups |
| `test_arm_a_loss.py`, `test_kspace.py` | Unit checks for the two losses |
| `compare_arms.py`, `baseline28.py` | Per-structure paired comparison against the base model |
| `run_finetune.py` | Freezes the backbone, leaving the output heads trainable |

Reporting metric `fw_10`: prediction and reference are each shifted by their own
VBM, the mask is taken on the reference side, and the MAE is equal-weighted over
$\pm 10$ eV. Note the distribution is heavy-tailed — mean and median differ by
three orders of magnitude — so paired per-structure comparison is the only safe
read.
