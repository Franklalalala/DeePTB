# Pixel MeanFlow: validation alignment with no-CFM/CFM + opt-in JVP du/dt backend

Base: `0703-Flow` at `d57c009` (fix(flow): reduce meanflow boundary graph memory).
This change absorbs and extends the external patch package
`deeptb_pmf_validation_jvp_patch` (same base commit): its one-step renamespacing,
validation return-value alignment, canary scalars, and benchmark tool were kept;
its non-sticky jvp retry, still-registrable `train_compatible_*` fields, and
untouched entry-point registration were replaced by the stricter variants below.

## Part A — validation semantics aligned with no-CFM/CFM

Semantics after this change, for any `flow_options.objective` in
{`pixel_meanflow`, `pixel_mean_flow`, `pmf`, `meanflow`, `mean_flow`}:

- `meanflow.log_validation_compatible_loss` defaults to **true** (code layer,
  no production config change needed). Validation euler-samples to the
  endpoint (`validation_ode_steps`, default `(1, 3)`) and scores the plain
  blockwise validation criterion on the sampled endpoint.
- The euler-1 result is written to the legacy keys
  `validation_loss` / `validation_onsite_loss` / `validation_hopping_loss`,
  so pMF curves are directly comparable with no-CFM and CFM runs.
  Explicit opt-outs are honored: `meanflow.log_validation_compatible_loss=false`
  or `meanflow.compatible_loss_to_legacy_keys=false`.
- `Trainer.validation()` now **returns** that legacy `validation_loss` when it
  exists (previously it returned the accumulated random-time flow objective).
  This matches `MultiTrainer.validation` and what `Validationer`/saver/LR
  scheduler already consumed.
- The MeanFlow objective stays observable, namespaced away from legacy keys:
  - random-time objective: `validation_flow_random_t_loss` (+ per-component
    `validation_flow_onsite_*` / `validation_flow_hopping_*`),
  - one-step (r=0, t=1) objective: `validation_flow_one_step_*` — these were
    previously emitted as `validation_one_step_flow_*`, which matched neither
    the TensorBoard prefix scan (`validation_flow_*` / `validation_compatible_*`)
    nor anything else, i.e. they were computed but never plotted.
- No fabricated zeros:
  - Scalar log fields are now registered from the **resolved flow object**
    (`dptb.nnops.flow.resolve_flow_log_fields`) instead of raw top-level
    `flow_options` keys. pMF never computes the raw-batch `train_compatible_*`
    metrics, so those fields are no longer registered (previously they were
    seeded 0.0 at registration and printed as a constant, perfect-looking 0).
    Same for CFM-only keys pMF never emits (`validation_flow_t0_loss`,
    `validation_flow_euler_N_loss`).
  - Legacy `validation_onsite_loss`/`validation_hopping_loss` monitors are
    only registered when the run will actually produce them.
  - A scalar-only validation criterion (no onsite/hopping side effects) yields
    `validation_loss` but **no** legacy component keys, instead of fake zeros.
  - `pixel_meanflow.log_train_compatible_loss=true` is rejected with a warning
    (forced off): the model-in-loss objective has no code path that computes
    it, so accepting the flag could only produce lying zeros.
- Wasted work removed: with compatible validation explicitly off, the pMF
  validation branch no longer euler-samples at all (the samples had no other
  consumer).

Trainer/MultiTrainer both key off the same flow attributes, so multi-GPU
(hanhai `multi_train`) validation is aligned by the same defaults.

## Part B — `du_dt_backend = "jvp"` (opt-in, finite_difference remains default)

Paper recap (Lu, Geng, He et al., arXiv:2601.22158, pMF Alg. 1):

```
v = u_fn(z, t, t)                       # boundary forward, r=t
u, dudt = jvp(u_fn, (z, r, t), (v, 0, 1))
V = u + (t - r) * stopgrad(dudt)
loss = metric(V, e - x)
```

i.e. **two** model calls per step; `u` and `du/dt` come out of one
forward-mode call. DeePTB's `finite_difference` backend is an engineering
approximation of that call: a third, no-grad forward at
`(z + eps*v, r, t + eps)` and a one-sided difference quotient
(`O(eps)` truncation plus float32 cancellation in `(u_eps - u)/eps`).

The new backend:

```json
"flow_options": {
  "enabled": true,
  "objective": "pixel_meanflow",
  "meanflow": {
    "du_dt_backend": "jvp",          // default: "finite_difference"
    "jvp_fallback": true              // alias: jvp_fallback_to_finite_difference
  }
}
```

- `torch.func.jvp` over the x-prediction with tangents
  `(dz/dt, dr/dt, dt/dt) = (tangent, 0, 1)`, `tangent` being the boundary
  velocity `(z - x_boundary)/t` (paper form, default) or the path velocity
  `prior - clean` (`jvp_tangent="path"`). du/dt is then assembled exactly:
  `du/dt = (tangent - dx/dt)/t - u/t`, and detached (paper stop-gradient).
- Forward-mode tangents on `t` reach the model's time conditioning because the
  jvp path writes flow-time keys **without** `.detach()`
  (`_write_times(..., detach=False)`); the fd path is unchanged.
- The jvp primal keeps the reverse-mode graph, so it replaces both the main
  grad forward and the fd_eps forward: boundary + jvp = **2 calls**
  (`jvp_tangent="path"` + `aux_boundary_v_weight=0`: **1 call**).
- Boundary grad semantics preserved: `aux_boundary_v_weight=0` keeps the
  boundary forward under `torch.no_grad()`; `>0` keeps it in the graph.
- Fallback: any exception from the jvp call (forward-AD-unsupported op, DDP or
  wrapper interference) logs one warning and **stickily** disables jvp for the
  rest of the run (no per-step retry cost); `jvp_fallback=false` makes it
  fatal for debugging.
- Canary scalars in every step state, also under TensorBoard:
  - `{train,validation}_flow_du_dt_backend_jvp` — 1.0 while jvp is live,
    0.0 after a fallback (watch this; a silent fallback is otherwise invisible),
  - `{train,validation}_flow_explicit_model_calls` — 3 fd / 2 jvp
    (boundary tangent), 2 fd / 1 jvp (path tangent).

### Paper-vs-implementation gaps (unchanged by this work, for the record)

- The velocity loss core matches Eq. (12) with the u=(z-x)/t re-parameterization
  of Eq. (11) applied to the *residual* state (base = physical H0); `norm_p`
  reproduces iMF's adaptive weighting.
- `aux_endpoint_weight` (default 0.05) and `aux_boundary_v_weight` are
  engineering extras not present in the paper loss; the paper supervises the
  boundary through its `r=t` time-sampling mass (DeePTB keeps that too via
  `data_proportion=0.5`).
- `finite_difference` vs the paper's exact jvp is the main numerical gap; the
  jvp backend closes it where forward-mode AD is available.

## Measurements (RTX 4060 Laptop 8GB, torch 2.5.1+cu124, commit d57c009+this)

`tools/bench_pixel_meanflow_du_dt_backend.py`, synthetic time-conditioned MLP
endpoint, forward+backward per step, 30 timed steps after 10 warmup,
`batch_size=96` graphs. `dynamic_batch` is not exercised (synthetic batch).

| model | tangent | backend | calls/step | median ms | peak MB |
|---|---|---|---:|---:|---:|
| 512×4, dims 64  | boundary | finite_difference | 3 | 11.70 | 133 |
| 512×4, dims 64  | boundary | jvp               | 2 | 15.45 (+32%) | 326 (2.4×) |
| 512×4, dims 64  | path     | finite_difference | 2 |  9.96 | 132 |
| 512×4, dims 64  | path     | jvp               | 1 | 13.92 (+40%) | 321 (2.4×) |
| 1024×6, dims 128 | boundary | finite_difference | 3 | 102.67 | 628 |
| 1024×6, dims 128 | boundary | jvp               | 2 | 135.42 (+32%) | 1847 (2.9×) |

`jvp_used_fraction = 1.0` in all jvp rows (no silent fallback on plain
PyTorch ops). The external patch package's CPU-only smoke showed the same
shape (2 vs 3 calls, jvp ~2.7× slower on a toy MLP).

Accuracy (same batch, float32, boundary tangent, untrained 512×4 model;
velocity-loss relative difference fd vs exact jvp):

| fd_eps | onsite rel diff | hopping rel diff |
|---:|---:|---:|
| 1e-3 | 6.7e-8 | 6.7e-8 |
| 5e-4 | 3.4e-7 | 1.3e-7 |
| 1e-4 | 1.2e-6 | 4.7e-7 |
| 1e-5 | 9.1e-6 | 7.4e-7 |

Note the trend: *smaller* eps is *worse* in float32 — cancellation dominates
truncation on a smooth model. The regime where jvp's exactness genuinely
matters is high-frequency time conditioning (the known
`fd_eps × flow_time_max_positions` phase-step pathology that NaN'd the
historical default; the stable recipe max_positions≈200 sidesteps it, jvp is
immune to it by construction).

### Why jvp costs what it costs

`torch.func.jvp` composed over reverse-mode runs every op in dual mode
(~2× flops on the augmented call) *and* keeps the primal autograd graph plus
live dual buffers — hence ≈3 forward-equivalents of compute (same as fd's
3 calls) with extra per-op dispatch overhead, and ~2.4–2.9× peak activation
memory. Fewer model calls ≠ less work here.

### Production stack blockers (real LEM/SO2 models)

Static scan of the production model path finds forward-AD-incompatible ops —
on the real stack the jvp call will raise on first contact and stickily fall
back to finite_difference (one warning, `*_flow_du_dt_backend_jvp=0`):

- `torch_scatter.scatter_add/scatter_max/scatter_mean` in
  `dptb/nn/embedding/lem_moe_v3.py` (message passing) — no forward-AD formulas.
- `_GroupedGemmFunction` (`dptb/nn/cublas_grouped_gemm.py`,
  `dptb/nn/cutlass_grouped_gemm.py`) and `_PersistentGroupedP1Function`
  (`dptb/nn/so2_moe_persistent_grouped.py`) — custom `autograd.Function`s
  without a `jvp` staticmethod.
- cuEquivariance segmented ops when enabled — same category.

If jvp ever becomes worth enabling there, the migration is mechanical:
replace `torch_scatter` calls with native `torch.index_add`/`scatter_add`
(forward-AD-supported in core) and add `jvp` staticmethods to the two grouped
GEMM Functions (jvp of a matmul is a matmul). Not recommended now given the
memory numbers.

## Recommendation for Hanhai bs96 pMF

**Do not enable jvp for the bs96 production task.** Three independent reasons:

1. Peak memory ×2.4–2.9 on the grad-carrying step — bs96 already runs with
   strict OOM-fallback submission on A100s; the jvp backend would burn the
   headroom that motivated the recent boundary no-grad optimization.
2. Wall-clock is 30–40% *worse* per step on plain PyTorch ops despite one
   fewer model call.
3. The real stack (torch_scatter + SO2/grouped-GEMM custom ops) does not
   support forward-mode AD today, so production jvp would silently run as
   finite_difference anyway (correct, but pointless).

Where jvp *is* useful now: an accuracy oracle/canary on small batches (verify
fd du/dt against the exact derivative when changing time-embedding
frequencies, fd_eps, or tangent mode), and CI-side exactness tests. Keep
`finite_difference` as the deployed backend; the validation-semantics fix in
Part A is the production-relevant half of this change.

## Verification

```bash
# unit tests (60 in the flow file, all green; plus te_prior/time_embedding 35)
python -m pytest dptb/tests/test_hamiltonian_flow.py -q
python -m pytest dptb/tests/test_flow_te_prior.py dptb/tests/test_flow_time_embedding.py -q

# focused selections
python -m pytest dptb/tests/test_hamiltonian_flow.py -q \
  -k "jvp or resolve_flow or forces_off or skips_sampling or legacy_endpoint or fabricate"

# GPU smoke (records commit/torch/gpu/batch in the JSON output)
python tools/bench_pixel_meanflow_du_dt_backend.py --device cuda --batch-size 96 --backend both
```

Pre-existing failures at clean `d57c009` (verified via `git stash`; unrelated
to this change): `test_dynamic_cost_batch.py` (2, missing
`log_single_model_compatible_loss` attr in partially-stubbed MultiTrainer
tests), `test_expert_data_parallel_layout.py` (1, same),
`test_trainer_reference_batches.py` (1, same),
`test_gpu_residency_hot_paths.py` (1, source-scan on `lem_moe_v3.py`
`batch.max().item()`).
