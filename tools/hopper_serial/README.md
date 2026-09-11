# H0 SOC serial training on Hopper

This deployment targets retained `sheng.lei` allocations 612077 (onsite) and
612078 (hopping), under `/scratch/sheng.lei/0912_h0_serial_edgemoe`.
The account, allocation IDs and source paths in these operational scripts are
specific to this deployment; inspect them before reuse.

Each allocation runs S1 pairwise fitting for 100,000 optimizer updates, then
initializes S2 from the committed S1 checkpoint and runs another 100,000 updates.
S2 freezes the pairwise modules and their shared learned edge-type embedding.
Both stages use the same edge-router shapes: 24 routed experts, top-2, one shared
expert, H0 projectors, and `streamed_m_major_fused_p0` SO2 CUDA. The learning-rate
peak is 0.01 with 5,000 warmup updates and update-count WSD.

The data contract is SOC **uu_real**, 729 spatial RME components, with named
H-minus-H0 labels. It is not a full complex spinor prediction. Onsite uses
`[[0, 1e-6]]` with clipping; hopping uses `[[1e-6, 10]]` without final clipping,
matching Frank's existing split runs.

`smoke_contract.py smoke` first resolves full local dataset staging or indexed
pread fallback, then generates reusable batch-cost metadata with 12 CPU workers.
The cached shard/index ordering is checked before use. The smoke runs the actual
`multi-train` CLI with the production architecture and batch budget for three
updates per stage, verifies H0/target loading, masks, finite gradients, frozen
parameters, dispatch, final checkpoint step and normal termination.
`smoke_contract.py production` refuses to run without `SMOKE_OK.json`.

`allocation_owner.py` owns process groups and a lease spanning staging, smoke,
and both production stages. It keeps the allocation after child completion or
failure, allowing operator recovery through a new `request.json` tag. There is
no log-grep or 100k signal watchdog. `MultiTrainer.max_steps` commits its final
checkpoint and exits normally. Unknown local consumers cause cleanup to retain
the stage. Stopped original PBS parents must remain stopped.

The relocated Python invokes the public CLI parser directly so batch metadata
can be installed before training. No checkpoint weights or optimizer state are
carried from smoke into production. Run outputs retain resolved configs, logs,
smoke evidence and checkpoints outside the repository.
