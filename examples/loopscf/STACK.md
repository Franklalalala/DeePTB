# Pretrained whole-stack recurrence

This route fine-tunes the pretrained LEM stack, H0 initializer/router and
shared readout. The alternative frozen-base corrector is described in
[LATENT.md](LATENT.md).

Each expert encodes the physical AO H0 through inverse CG once. The complete
pretrained stack is then recurrently applied to persistent node, edge, and
scalar edge-latent states. There is no eigensolver or physical SCF transition
inside this recurrence. The same readout is supervised at every step.

The first and last layers have different irreps. Learned equivariant linear
bridges close this interface by adding a correction to the previous input
state. Components absent from the output representation survive unchanged in
the residual state. Scalar latents use a bounded channelwise mixing coefficient.
These bridges start at zero: K1 is the corrected-entry pretrained computation,
and every depth starts with the same prediction. This is a deliberate adapter
for a pretrained LEM stack, rather than an assertion that LEM has exactly the
Transformer architecture in Ouro.

## Training and halting

For each graph, a shared invariant gate predicts a logit from pooled norms of
the final hidden irreps. The exit hazard includes a fixed remaining-depth offset
so zero learned logits initialize a uniform exit distribution. First-exit
probabilities use survival products, and the final step receives all remaining
mass. `predict_until_exit` actually stops computation at a CDF threshold for
one graph; batched variable-depth compaction is not implemented.

Stage I jointly trains the stack, bridge, head, and gate with full BPTT:

`0.8 * sum_t p_t L_t + 0.2 * mean_t L_t - beta * entropy(p)`.

The 20% uniform auxiliary term is our stabilization choice: every depth continues
to receive task supervision while the exit distribution learns. Ouro Eq. (4)
does not include this term. Entropy is an exploration regularizer, not a penalty
for compute. No monotonic per-step error guarantee is claimed.

Stage II freezes the body and readout, calibrating the shared gate on a reserved
slice of the training split. It uses the retrospective positive improvement
`max(0, L_(t-1)-L_t)` at intermediate steps, following Ouro Eq. (5)'s indexing.
The terminal hazard has no free probability mass to learn. The slope is
5,000/eV and the gain threshold 0.0002 eV; these are adapted to matrix-loss
units, not copied from language cross-entropy. Both pre-calibration and
post-calibration results are retained.

## Usage

Use the dynamic runner for edge-budget batches and optional WSD scheduling;
see [DYNAMIC.md](DYNAMIC.md). The fixed-batch runner is also available:

```bash
python examples/loopscf/anneal_stack.py \
  --input train.json --base-model PRETRAINED.pth --output RUN \
  --K 3 --steps 6000 --batch-size 4 --hours 8

python examples/loopscf/evaluate.py \
  --input explicit_electron_validation.json --checkpoint RUN/joint_final.pth \
  --base-model PRETRAINED.pth --architecture stack --K 1 2 3 \
  --output RUN/bands.json
```

Matrix supervision does not require inferred electron counts. Spectral evaluation
requires explicit electron counts and aligned H0, H and S records. The loss is
an equal-graph mean of onsite/hopping masked L1 and RMSE components; it differs
from a global valid-element MAE. A reserved training subset calibrates the gate.

The corrected H0 entry can change predictions from an existing checkpoint;
input adaptation and loop learning must be separated when interpreting gains.
K1/K3 comparisons should report samples, stack applications and elapsed time.

The runner writes checkpoints and diagnostics under the requested output path.
Fixed-batch checkpoints support inference and weight recovery; use the dynamic
runner for saved optimizer and committed-sampler state. Keep generated run
artifacts outside the source tree.

Focused stack tests cover bridge equivariance and gradients, alias-free state
reuse, per-graph losses, exit mass and actual early stopping.

Architecture reference: Ouro, arXiv:2510.25741v5, Fig. 3 and sections 3.1–3.4.
This adaptation does not establish physical SCF convergence.
