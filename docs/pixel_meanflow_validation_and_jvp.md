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

- Validation always samples an Euler-1 endpoint baseline and scores the plain
  validation criterion on it. `validation_ode_steps` controls only additional
  endpoint diagnostics; it cannot remove step 1.
- The euler-1 result is written to the legacy keys
  `validation_loss` / `validation_onsite_loss` / `validation_hopping_loss`,
  so pMF curves are directly comparable with no-CFM and CFM runs in the same
  metric space. Block-native criteria use the same L1+RMSE endpoint family in
  AO-block space unless their loss explicitly opts into the more expensive
  old-RME-compatible reduction.
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
- No fabricated zeros: endpoint all/onsite/hopping is one fail-closed contract.
  A criterion that cannot produce the complete triplet raises immediately.
  TensorBoard registers fields from the resolved flow object and suppresses the
  duplicate `train_compatible_*` and Euler-1 compatible aliases; extra Euler-N
  diagnostics remain visible.
- `validation_flow_metrics` is the single selector for optional random-time,
  one-step, and trajectory flow diagnostics. It does not control the endpoint
  triplet.

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
    "jvp_fallback": true
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

### Production-like crystal measurement (natlan, 2×L40S 46GB, torch 2.8.0+cu128)

`~/codex/0704_jvp_crystal_bench` on natlan: real `lem_moe_v3_h0` crystal stack
(0516_2range nextham dataset, full-periodic-table basis, `batch_size=32`,
`dynamic_batch.max_samples=16` = the hanhai production cap; first packed batch
16 graphs / 244 nodes / 27.7k edges). Module-level loss_with_model+backward:

| knobs | backend | s/step (median) | peak MB | jvp live? |
|---|---|---:|---:|---|
| cueq (production) | finite_difference | 4.45 | 28 787 | — |
| cueq (production) | jvp | 4.32 (= fd after fallback) | 28 787 | **no — falls back** |
| pure-torch ref | finite_difference | 7.29 | 26 292 | — |
| pure-torch ref | jvp | (= fd after fallback) | — | **no — falls back** |

**On the as-shipped stack the jvp call falls back immediately.** Fail-fast
(`jvp_fallback=false`) pins the first blocker to
`e3nn/o3/_spherical_harmonics.py:98` (entered from `lem_moe_v3_h0.py:99`):
e3nn's SphericalHarmonics is TorchScript-compiled, and under `torch.func.jvp`
*every* tensor in the transformed region is a storageless functorch wrapper —
even ones that carry no tangent, like the edge direction vectors — so the JIT
graph executor dies on `data_ptr()`. This is knob-independent (cueq and
pure-torch ref fail identically). The sticky fallback behaves exactly as
designed: one warning, `*_flow_du_dt_backend_jvp = 0`, and the run proceeds at
finite_difference speed — a 5-minute real `dptb train` pair confirmed a
misconfigured jvp production job degrades gracefully instead of crashing.

### jvp CAN run on the crystal stack — feasibility ladder (proved on natlan)

Climbing the walls one by one makes jvp fully operational on the real
`lem_moe_v3_h0` crystal model (no JAX / e3nn-jax port needed):

1. **e3nn eager mode** — `e3nn.set_optimization_defaults(jit_mode="eager")`
   before model construction (e3nn ≥ 0.5.8 keeps SphericalHarmonics a plain
   Python function and CodeGenMixin an eager FX module). This repo now does it
   automatically in both train entrypoints when `du_dt_backend=jvp`
   (`dptb.nnops.flow.configure_jvp_friendly_backends`).
2. **softmax stabilizer detach** — `scatter_max(logits.detach(), ...)` in the
   edge-attention softmax: shift-invariance makes it gradient-free by
   construction, and torch_scatter's custom kernel never sees dual tensors.
3. **no `data_ptr()` aliasing checks** — `y is x` replaces
   `y.data_ptr() == x.data_ptr()` in `eqv3_grid_helpers` (also
   FakeTensor/compile-friendly).
4. **routing metadata as plain host values** — split sizes/indices are
   non-differentiable metadata. `_tensor_to_split_tuple` gets a storage-read
   fast path (host reads of *plain* tensors are legal inside torch.func
   regions; running detach/reshape first would lift them into storageless
   wrappers), remaining `.item()` sites go through a `_functorch_plain`
   unwrap. The multi_train production path already precomputes
   `LEM_ACTIVE_EDGE_SPLIT_SIZES` on CPU at batch prep, which is exactly the
   jvp-safe carrier; the single-Trainer bench replicates that prep.

With those in place (pure-torch ref knobs, `max_samples=4`, 4 graphs / 14.5k
edges, L40S):

| backend | s/step | peak MB | calls | canary |
|---|---:|---:|---:|---|
| finite_difference | 3.75 | 13 921 | 3 | 0 |
| jvp (live) | 4.44 (+18.5%) | 30 810 (**2.21×**) | 2 | **1** |

Numeric parity at fixed (r, t, seed): velocity-loss rel. diff **2.4e-7**
(onsite) / **0.0** (hopping) vs fd at eps=5e-4 — the jvp implementation is
exact on the real stack, and conversely fd's truncation error at the
production eps is float32-noise-level.

At `max_samples=8` the dual forward OOMs a 46GB L40S (fd peak 21.2 GB →
dual needs >44 GB), consistent with the 2.2–2.9× activation-memory law.

### jvp now runs on the production SO2 CUDA fast line (fused_p0 + cublas_grouped)

The production line (`so2_fusion_mode=streamed_m_major_fused_p0` +
`mole_linear_mode=cublas_grouped`, the 998933/1027855 knobs) now runs the jvp
backend end-to-end. What it took, in order:

1. **e3nn eager** (as above) — `configure_jvp_friendly_backends`.
2. **native `torch.autograd.forward_ad`, not `torch.func.jvp`.** functorch
   wraps every tensor in a storageless interpreter wrapper, so a custom
   Function's `jvp` staticmethod cannot call a CUDA kernel that reads
   `data_ptr()`. Native dual tensors carry real storage, so the fused-pair and
   grouped-GEMM kernels run inside their jvp rules. (`_jvp_du_dt` in
   `dptb/nnops/flow.py`.)
3. **forward-AD rules on the two custom autograd.Functions**, both linear in
   the feature input so `jvp = same op applied to the tangent`:
   - `_GroupedGemmFunction` (`dptb/nn/cublas_grouped_gemm.py`) — bilinear in
     (x, weight); `setup_context` + `jvp`, in-repo.
   - `_FusedPairFunction` (external `so2_cuda_ops`, SO2 tensor product) —
     trilinear in (x, mixed_weight, radial). Applied by
     `tools/patch_so2_cuda_ops_fusedpair_jvp.py` (idempotent, anchored,
     backs up the original). The so2_cuda_ops source is editable-installed;
     re-run the patcher after any so2_cuda_ops update.
4. **build fix** for the natlan dev box (nvcc 12.2 rejects gcc 12.3):
   `CUDAHOSTCXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda-12.2`. Production
   (hanhai) already has a working so2_cuda_ops build.

### Memory-efficient split vs fused, and the production measurement

Two jvp execution modes (`meanflow.jvp_memory_efficient`, default **true**):

- **split (default)**: primal in a normal grad forward + detached du/dt tangent
  in a *separate no_grad* forward-mode pass. Forward-mode tangents free
  layer-by-layer under no_grad, so peak memory stays ~1× (like fd). One extra
  model call.
- **fused**: one grad-tracking dual forward yields primal + tangent together
  (one fewer call), but every stored activation is a primal+tangent dual → ~2.2×
  peak.

natlan L40S 46GB, real production `fused_p0 + cublas_grouped` kernels
(`my_oeq` env), boundary tangent, per step forward+backward, median of 3 after
warmup. `max_samples` = dynamic-batch cap (production is bs96/`max_samples=48`):

| max_samples | fd s/step · peak | jvp **split** s/step · peak | jvp fused s/step · peak |
|---:|---|---|---|
| 4  | 5.85 · 13.3 GB | 10.18 (+74%) · **14.4 GB (1.08×)** | 8.74 (+49%) · 29.5 GB (2.21×) |
| 8  | 9.03 · 20.3 GB | 15.96 (+77%) · **21.9 GB (1.08×)** | OOM |
| 16 | 11.24 · 25.2 GB | 19.85 (+77%) · **27.2 GB (1.08×)** | OOM |

Numeric parity vs fd (fixed r,t,seed): velocity-loss rel. diff **1.2–2.4e-7**
at every batch size — the jvp is exact on the production kernels, and fd's
truncation error at the production `fd_eps=5e-4` is at that same float-noise
level.

Key results:
- **split keeps jvp at fd-level memory (1.08×) on the measured `max_samples=
  {4,8,16}` points**, so it fits wherever fd fits there. fused OOMs a 46 GB card
  by `max_samples=8` (measured). Extrapolating the fd curve (~1 GB/sample) to
  production bs96/ms48 gives fd ≈ 57 GB, split ≈ 61 GB, fused ≈ 125 GB — these
  ms48 numbers are **extrapolated, not measured** (ms48 OOMs the 46 GB L40S; it
  needs an 80 GB A100 to measure). Treat "split fits 80 GB" as a hypothesis to
  validate, not a benchmark. **split is what plausibly makes jvp usable at
  production batch; confirm on a real A100 job.**
- Time is the forward-mode tax: split +74–77%, fused +49% over fd. Optional
  `jvp_tangent="path"` drops the boundary forward (split ms4: 10.18 → 8.61 s,
  memory unchanged, still exact) but changes the tangent from the paper's
  instantaneous velocity to the straight-line velocity.

### Backend coverage summary

- **fused_p0 + cublas_grouped** (production, 998933/1027855): **jvp works** with
  the split mode at fd-level memory (this section). Requires the two
  forward-AD rules above + the so2_cuda_ops patcher.
- **ref** (`streamed_m_major_ref + split_loop`, pure torch): jvp works
  natively (no custom-op rules needed); ~1.6× slower than the fused kernels.
- **cueq** (`streamed_m_major_cueq + cueq_indexed_linear`): **still blocked** —
  jvp dies at `torch.ops.cuequivariance.indexed_linear_B`
  (`Cannot access storage of TensorWrapper`); that CUDA kernel is third-party
  (cuEquivariance) with no forward-AD rule and can't be patched from DeePTB.
  This is not the production path, so it doesn't gate the goal; a DeePTB-side
  autograd.Function wrapper around the cueq call (jvp = same call on the
  tangent) would fix it if ever needed.

finite_difference remains the *default* backend (`du_dt_backend` default
`finite_difference`): jvp is opt-in. jvp's value is exactness — it removes fd's
`O(fd_eps)` truncation / float32-cancellation error and matches the paper's
`jvp(u_fn, (z,r,t), (v,0,1))` definition — at the cost of +74–77% wall clock.
With the split mode its memory is now essentially free (1.08×), so enabling it
on the production line no longer forces a batch reduction. Whether to pay the
time for the exactness is a training-quality call; the capability is now there
and verified on the real kernels.

Memory-cap note: with `max_samples=32` the same batch (32 graphs / 48.2k
edges) OOMs a 46GB L40S under pMF's 3-forward step — consistent with hanhai
history (bs32/ms16 needed 43–44GB on A100 *pre* no-grad patch; post-patch
bs96/ms48 fits 80GB). pMF peak = grad-forward graph + no-grad forward
transients stacked on top, so per-sample headroom must be sized ~1.3–1.5× a
supervised step, not 1×.

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

This section supersedes an earlier draft that said jvp could not run on the
production stack — the "SO2/grouped-GEMM custom ops don't support forward AD"
blocker has since been removed (native forward_ad + the two forward-AD rules +
the so2_cuda_ops patcher; see the production-line section above).

Current state and recommendation:

- **`finite_difference` stays the default backend.** jvp is opt-in
  (`du_dt_backend=jvp`). fd's truncation error at the production `fd_eps=5e-4`
  is float-noise-level (~1e-7 vs the exact jvp), so fd is a fine production
  default; jvp buys exactness, not accuracy you can currently measure a
  training difference from.
- **jvp no longer forces a batch reduction.** With the split mode (default)
  its peak memory is ~1.08× fd on the measured L40S `max_samples={4,8,16}`
  points, so it fits where fd fits. The bs96/`max_samples=48` A100 figures
  (fd ≈ 57 GB, split ≈ 61 GB, fused ≈ 125 GB) are **extrapolated from a
  ~1 GB/sample fit, not measured** — validate on a real A100 job before
  relying on "fits 80 GB". The fused mode's ~2.2× memory does OOM a 46 GB card
  by `max_samples=8` (measured).
- **Cost of enabling jvp:** +74–77% wall clock (measured on the real
  fused_p0+cublas kernels). Whether that's worth exactness is a training-quality
  call, not a hard blocker.
- **If you do a first production jvp canary run, set `jvp_fallback=false`** so a
  fallback is a hard crash rather than a silent degrade to fd (see the
  observability caveat below), and confirm `train_flow_du_dt_backend_jvp==1`.

Observability caveat (review finding 3): a jvp→fd fallback always emits a
one-time `WARNING ... falling back to finite_difference` on every trainer,
including the distributed `MultiTrainer`. The per-step canary *scalar*
(`train_flow_du_dt_backend_jvp`) is surfaced to TensorBoard by the single-GPU
`Trainer` but **not yet threaded through `MultiTrainer`'s packed
all-reduce/display path** — so on hanhai bs96 the live-vs-fallback signal is the
warning log, not a TB scalar. `jvp_fallback=false` makes this moot for a canary
run. Threading the canary through the distributed reducer is deferred (it
touches two reduction paths and is not worth the regression risk on the
production trainer for a diagnostic).

The validation-semantics fix in Part A remains the production-relevant half of
this change regardless of backend choice.

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

The 0714 endpoint contract is always enabled; partially stubbed trainer tests
therefore need only set `endpoint_loss_mode` when they exercise the packed
`reduce` versus `full_forward` implementation choice.
