# Chemical onsite readout

Set `model_options.embedding.node_readout="chemical_core"` for the legacy RME
output route of `lem_moe_v3_edge_h0` (the r14 B arm). Omit the key, or use
`"shared"`, for the original model. The base LEM readout and both edge-routing
paths share this implementation. AO-block/nonlinear output heads are rejected.
The optional edge readout is **not implemented**.

Only `out_node`'s same `(l, parity)` channel maps receive a residual. The existing
`out_node.weight`, `out_node.bias`, output one-hot tensor product, backbone,
Hamiltonian assembly and loss masks keep their roles. For each channel block:

```
R_g = P @ D[g] @ Q.T
Delta_g = c * R_g / sqrt(c*c + sum(R_g*R_g))
W_g = W_0 + n_g/(n_g+100) * Delta_g
c = 0.25 * ||W_0_initial||_F
```

Rank is **16**, support scale **100**, cap fraction **0.25**. These are fixed
constants, not configuration or optimizer options. P/Q are shared across elements
within a channel block; D has shape `[number_of_basis_elements,16,16]` per block.
Rank stays 16 even when a channel dimension is smaller. `W_0` denotes the actual
channel map, including e3nn instruction path weights, with magnetic components
factored out. Repeated/nonadjacent identical irreps form one block, not separate
chemical groups. Bias has no chemical residual. A zero initial W block has c=0
and remains without a chemical residual; its shared weights can still train.

D starts at zero; P/Q are random with scale `1/sqrt(16)`. Construction restores
RNG states around the new parameters. Exact addition preserves signed zero while
keeping the D gradient live. Default-off creates no new parameters or persistent
buffers. GPU and CPU GEMM implementations can have different rounding; exact
identity is evaluated against the same device/backend.

Each batch obtains the present element IDs once, gathers rows by element, builds
one channel matrix per present element/block, applies it to all magnetic
components, and scatters results back. It never creates `[atoms,out,in]` weights.
This implementation uses grouped PyTorch GEMMs, not a new custom fused kernel.
Temporary matrix storage is independent of the number of atoms; gathered feature
and output tensors still scale with atoms. It does not claim dense throughput.

## Support and checkpoints

`Trainer.__init__` (also used by MultiTrainer) calls
`initialize_chemical_readouts(model, train_dataset)` before optimizer creation.
It scans the actual training dataset, counting each element at most once per
structure; minibatch size, repeated atoms and training epochs do not inflate
counts. A dataset subset is counted as that subset. Atomic numbers are recovered
from the dataset mapper if its transform removed them. No labels enter the
statistics. Generic direct `build_model` users must call this initializer before
forward, or explicitly `head.set_counts(counts)` with counts in mapper order.
Uninitialized support raises an error instead of silently running an inert head.

`embedding.chemical_core` stores `atomic_numbers`, integer `n_g`, `counts_ready`
and each `blocks.<b>.c` as persistent buffers. Once initialized, ordinary trainer
construction/resume skips the dataset entirely. Strict checkpoint loading rejects
missing buffers. Unseen **training** elements in the configured basis have rho=0
and exactly use the shared head. Elements outside the model's basis remain
subject to the backbone's existing unknown-element rejection.

For dense initialization, set
`model_options.embedding.node_readout_init_from` to the same-topology dense
checkpoint and make a fresh build (do not also use CLI `--init-model` or
`--restart`). All shared keys/shapes must match exactly; only new chemical keys
may be absent. After copying the dense parameters, caps are captured from the
loaded shared matrices. D is still zero, and initial predictions match the dense
checkpoint. This starts the new arm's optimizer/scheduler. For later continuation
use the normal chemical checkpoint/`--restart`; its original source path need
not exist, and counts/caps are restored without rescanning or recapturing.

For r14 B, preserve the experiment's original HybridMuon optimizer and schedule;
no custom decay or learning-rate groups are introduced. Optimizer routing follows
the existing HybridMuon rules for each parameter's shape/name. Parameterization
therefore changes optimization geometry even with the same optimizer settings.

## Focused tests

`test_r14b_chemical_readout.py` covers grouping with an independent per-atom
reference, repeated irreps, cap/shrinkage, unseen support, nonzero-D rotations
and inversion, permutation, signed-zero/zero-cap behavior, exact initialization,
strict dense loading, Saver/checkpoint/optimizer continuation and HybridMuon.
`test_r14b_real_training.py` is an opt-in four-record onsite LMDB benchmark. Set
`R14B_MINI_ROOT`, `R14B_TEST_DEVICE` and optionally `R14B_METRICS_DIR`. GPU metrics
synchronize updates, exclude two warmup updates from the reported mean, and
record peak allocated/reserved bytes separately. Its Adam smoke recipe is not a
production optimization ablation.
