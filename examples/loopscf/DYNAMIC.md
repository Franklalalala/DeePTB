# Edge-budget training and WSD

`anneal_dynamic.py` trains the [shared pretrained stack](STACK.md) with full
cross-step BPTT. It packs consecutive graphs by edge cost and sends an accepted
batch directly to the GPU. `--max-graphs` caps graph count; `--batch-size`
controls CPU prefetch chunks. `--initial-graphs` initializes the edge budget.
The controller adjusts that budget from observed CUDA peaks toward
`--memory-target-gib`; actual graph count varies with structure size.

Only a CUDA forward/backward OOM activates split replay of the same graphs.
Replay clears gradients, restores RNG and buffers, and weights each partial
loss by its share of graphs. Non-OOM errors, singleton OOM and optimizer OOM
terminate without silently skipping data or retrying partial optimizer state.

```bash
python examples/loopscf/anneal_dynamic.py \
  --input train.json --base-model PRETRAINED.pth --output RUN \
  --K 3 --steps 6000 --batch-size 8 --initial-graphs 4 --max-graphs 8 \
  --memory-target-gib 68 --schedule wsd \
  --lr 1e-2 --bridge-lr 1e-2 --warmup 120 \
  --warmup-lr 1e-6 --min-lr 1e-6 --decay-ratio 0.65 --hours 8
```

These are example settings, not accuracy or stability guarantees. A K1 control
can use `--K 1 --batch-size 24 --initial-graphs 12 --max-graphs 24`; compare
actual samples and time rather than raw update counts. Large prefetched graph
batches share many tensor file descriptors. Ensure the launcher's soft file
limit is sufficient (for example, `ulimit -Sn 65536` within the hard limit).

## Learning rate and restart

The optimizer is AdamW. `--lr` sets the backbone/readout/gate peak and
`--bridge-lr` sets the bridge peak. The default `cosine` schedule warms over
`--warmup` updates, then decays to 3% of peak. `--schedule wsd` uses the existing
`WarmupStableDecayLR`: linear warmup, constant peak until `--decay-ratio`, then
cosine decay to `--min-lr`. WSD progress uses the greater of committed updates
and elapsed training time mapped to the step budget. Its clock starts after
initial validation. `--deadline` can replace the relative `--hours` budget.

Fresh runs load only the base weights and initialize adapters, optimizer and
sampler. `--resume` restores joint-stage weights and AdamW state at matching
depth, seed and data configuration. Dynamic checkpoints also restore the
committed sampler cursor, random states and cost controller, and require the
same CPU chunk size. Each invocation starts a new declared LR phase; legacy
checkpoints without sampler state start a new seeded sampling phase.

Use a fresh output directory for each phase. Protocols, histories, checkpoints,
validation results and calibration outputs belong there, outside the source
tree. `direct_batch=true` and one microbatch entry indicate direct execution;
multiple entries identify OOM replay. The last three periodic checkpoints are
retained. Completion of the LR schedule is not proof of convergence.
