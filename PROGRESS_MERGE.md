# BLOCKED — Stage 4 exposed a real baseline H0 raw/sorted-RME equivariance bug; Stage 5 was not entered.

## Merge Stage 4 — l>0 flow AO-block equivariance

- Timestamp: 2026-07-24 10:08:30 +08:00
- Status: BLOCKED (hard gate)
- Added `test_block_ode_flow_highl_equivariance.py`. It uses a genuine C
  `1s1p` state with nonzero p-shell rows, rotates geometry, and rotates all
  node/edge H0 and delta-H AO blocks by `D B D^T` before flow preparation.
- Reproduction:
  `pytest dptb/tests/test_block_ode_flow_highl_equivariance.py -q -s`
- Result: **2 failed, 4 warnings** in 30.53 s.
  - two-stage off: node `max|Δ|=8.1091233946705366e-01`, edge
    `1.9660634910835956e-01`;
  - two-stage on: node `max|Δ|=8.1091233946705366e-01`, edge
    `2.8586735079904640e-01`.
- Localization:
  - flow-prepared spatial residual AO blocks remain covariant:
    node `1.67e-16`, edge `5.55e-17`;
  - `DirectSpatialResidualBlockProjector._contract`: node `9.99e-16`,
    edge `5.55e-17`;
  - its equivariant linears: node `2.66e-15`, edge `1.11e-16`;
  - flow-produced H0 RME interpreted in the current order: node `2.287`,
    edge `0.905`;
  - applying the already-defined projector `sort_index` to those H0 RME
    rows restores covariance: node `1.78e-15`, edge `1.11e-16`;
  - current `H0InitLayer` output therefore first loses covariance at the H0
    projection boundary: node `1.792`, edge `1.127`.
- Root cause: `BlockStateCodec.blocks_to_rme()` supplies coupled RME in raw
  orbpair order, while `H0InitLayer.h0_irreps` and its node/edge `Linear`
  modules declare sorted irreps. The non-SOC spatial path explicitly omits the
  sort applied by the uu-real path. This is shared baseline code and explains
  why the failure is identical with two-stage off/on.
- Full finding: `F:\claude\0724_pair_iter2\results\highl_equivariance_finding.md`.
- Per the Stage-4 hard gate, no fix was implemented and Stage 5
  cross-tree/full regression was not started.

## Merge flag matrix

| Flag | Default | Owner | Semantics | Lane |
|---|---:|---|---|---|
| `mp_cutoff` | `None` | `LemPair` | Enables the construction-time private MP topology; `None` keeps legacy topology. | A |
| `mp_avg_num_neighbors` | `None` | `LemPair` | Optional MP-subgraph aggregation normalization. | A |
| `res_update_additive` | `false` | `LemPair` | Uses unscaled additive residual updates in the pair backbone. | A |
| `latents_layernorm` | `true` | `LemPair` | Preserves legacy latent LayerNorm; set false for the norm-free ablation. | A |
| `pair_refine_enable` | `false` | `LemPair` | Constructs and applies the post-backbone SO(3) pair refinement. | A/C |
| `pair_refine_rank` | `16` | `LemPair` | Conditioner bottleneck width. | A/C |
| `pair_refine_condition` | `scalar_0e` | `LemPair` | Invariant conditioner source. | A/C |
| `pair_refine_internal_weights` | `true` | `LemPair` | Enables learned static TP weights. | A/C |
| `pair_refine_init` | `0.0` | `LemPair` | Dynamic conditioner initialization scale. | A/C |
| `pair_refine_weight_mode` | `full` | `LemPair` | Selects legacy full external weights or low-cost `per_path` gates. | C |
| `pair_refine_max_weight_numel` | `None` | `LemPair` | Optional fail-closed guard on full TP weight count. | C |
| `pair_refine_identity_init` | `false` | `LemPair` | Zeroes static/dynamic refinement for exact identity initialization. | C |
| `hb0_hermitian_average` | `false` | `LemMoEV3H0` | Opt-in reverse-edge transpose averaging at the H-B0 boundary. | B |
| `condition_source` | `edge_0e` | `LemMoEV3H0` | Selects legacy edge scalar or endpoint scalar head conditioning. | B |
| `log_head_input_rms` | `false` | `LemMoEV3H0` | Attaches detached per-irrep node/edge head-input RMS telemetry. | B |
| `two_stage_pair_enable` | `false` | `LemMoEV3H0` | Constructs late pair rebuilding plus norm-free refinement tail. | D |
| `two_stage_pair_refine_layers` | `2` | `LemMoEV3H0` | Number of two-stage refinement tail layers. | D |
| `two_stage_pair_tail_gate` | `false` | `LemMoEV3H0` | Enables the optional tail output gate. | D |
| `two_stage_pair_refine_rank` | `16` | `LemMoEV3H0` | Dynamic per-path conditioner rank. | D |
| `two_stage_pair_refine_condition` | `scalar_0e` | `LemMoEV3H0` | Two-stage tail conditioner source. | D |
| `two_stage_pair_refine_radial_dim` | `4` | `LemMoEV3H0` | Radial feature width used by the two-stage tail. | D |
| `two_stage_pair_refine_edge_chunk_size` | `64` | `LemMoEV3H0` | Edge chunk size for bounded tail memory. | D |

## Merge commits (`ad907ba..HEAD`)

- `HEAD test(merge): expose high-l H0 flow equivariance blocker`
- `fe763ff feat(merge): integrate lane D two-stage pair`
- `2ea5d4e feat(merge): integrate lane B H-B0 heads`
- `0f1f97f feat(merge): integrate lane C refine controls`

## Deferred item

- BRIEF D4 remains deferred: the private MP topology still uses a hard boolean
  cutoff. A smooth force/position-gradient envelope is out of scope for this
  fixed-structure Hamiltonian round.

## Merge Stage 3 — Lane D two-stage pair stream

- Timestamp: 2026-07-24 09:58:11 +08:00
- Status: PASS
- Imported `TwoStagePairStream` and D tests from `e8b1c2c`.
- Added the seven `two_stage_pair_*` arguments to `LemMoEV3H0` and
  `slem_h0()`. `two_stage_pair_enable` defaults to `false`; disabled models
  retain `two_stage_pair=None` and construct no module.
- Forward ordering is exactly: backbone → optional two-stage replacement of
  active-edge features → optional C refine envelope → B endpoint/Hermitian/RMS
  head dispatch.
- D targeted result: **13 passed, 2 warnings** in 47.48 s.
- D+B+A merge gate: **55 passed, 6 warnings** in 158.58 s.
- Disabled bit-exact gate: post-construction RNG/state/output all
  `torch.equal`, no `two_stage_pair.*` module/state.
- Enabled end-to-end equivariance: `max|Δ|=7.2164496600635175e-16`.
- `LemPair + two_stage_pair + condition_source=endpoints` combination:
  `max|Δ|=1.9984014443252818e-15`.
- Gate: PASS.

## Merge Stage 2 — Lane B endpoint/Hermitian/RMS heads

- Timestamp: 2026-07-24 09:51:35 +08:00
- Status: PASS
- Imported B-exclusive head implementation and tests from `8baec4e`.
- Reconciled `LemMoEV3H0`: `condition_source` defaults to `edge_0e`;
  `hb0_hermitian_average` and `log_head_input_rms` default to `false`.
  Their H-B0-only validation is retained.
- The head call is executed once. C's optional full-cutoff envelope is passed
  first, then B's optional RMS 3-tuple is unpacked. This preserves the required
  `LemPair._apply` → `LemMoEV3._apply` MRO.
- B targeted result including the added
  `pair_refine_enable=true + log_head_input_rms=true` MRO smoke:
  **13 passed** in 58.16 s.
- Complete B + all `test_lem_pair_*` + output-route/argcheck gate:
  **44 passed, 5 warnings** in 136.29 s.
- Default gates are proven bit-exact by B's tests: implicit vs explicit
  `condition_source=edge_0e`, `hb0_hermitian_average=false`, and
  `log_head_input_rms=false` all preserve RNG/state/output exactly.
- Gate: PASS.

## Merge Stage 1 — Lane C refine low-rank/identity/envelope

- Timestamp: 2026-07-24 09:45:57 +08:00
- Status: PASS
- Imported Lane C implementation from `49cdb74`: `pair_so3_refine.py`,
  `pair_refine_cost.py`, and `test_pair_refine_lowrank.py`.
- Reconciled A-owned `LemPair`: added `pair_refine_weight_mode` (default
  `full`), `pair_refine_max_weight_numel` (default `None`), and
  `pair_refine_identity_init` (default `false`). Full-r_max cutoff
  coefficients are selected on active edges and passed as the explicit
  refinement envelope.
- Added a merge-level wiring test proving per-path identity initialization
  is bit-exact with refinement disabled and receives a non-empty positive
  envelope.
- Targeted C + legacy refine result: **25 passed** in 41.95 s.
- Final C + all eight `test_lem_pair_*` files: **52 passed, 4 warnings** in
  108.91 s.
- Explicit Lane-A-vs-merge default-full golden: post-construction RNG,
  7 state tensors, and output all `torch.equal`; `max|Δ|=0`.
- Gate: PASS.

## Merge Stage 0 — A baseline and overlap audit

- Timestamp: 2026-07-24 09:37:00 +08:00
- Status: PASS
- Worktree/branch: `E:\deeptb\wt_0724_merge`, `feat/0724-merge-all`, starting
  merge commit `ad907ba`; worktree clean.
- Baseline command: `pytest dptb/tests/test_lem_pair_common.py
  dptb/tests/test_lem_pair_dual_cutoff.py
  dptb/tests/test_lem_pair_flow_contract.py
  dptb/tests/test_lem_pair_hard_gates.py
  dptb/tests/test_lem_pair_lifecycle.py
  dptb/tests/test_lem_pair_contract_validation.py
  dptb/tests/test_configuration_canonicalization.py -q`
- Result: **92 passed, 4 warnings** in 96.31 s.
- B/C/D `dcacda5..HEAD --stat` and name-status were reviewed. They match the
  supplied overlap map: C owns only refine implementation/cost/test files;
  B and D meet in `lem_moe_v3_h0.py` and `argcheck.py`; B additionally owns
  the endpoint head implementation; D owns the new two-stage module/tests.
- Gate: PASS.

## Stage 0 — scope / design

- Timestamp: 2026-07-23 17:41:24 +08:00
- Status: SUCCESS
- Merge worktree: `E:\deeptb\wt_0723_merge`, branch `0721-stable`,
  starting commit `a6f152e97e3c6b40eab0e89dfef2db21c9a0951e`; worktree was clean and
  matched `origin/0721-stable`.
- Read completely:
  - `E:\deeptb\wt_0723_dualcut\PROGRESS_A.md` and the A implementation/test
    diff from `c7d097f`;
  - `E:\deeptb\wt_0723_norm\PROGRESS_B.md` and the B implementation/test diff
    from `c7d097f`;
  - `E:\deeptb\wt_0723_b1\PROGRESS_C.md` and the C implementation/test diff
    from `c7d097f`;
  - `a6f152e` commit diff and the current block-ODE topology helpers, route
    adapters, subclass-dispatch characterization tests, config contract, model
    registry, and argcheck dispatch.

### Structure decision

- New public embedding method and class: `method: lem_pair`,
  `LemPair(LemMoEV3H0)`, implemented in
  `dptb/nn/embedding/lem_pair.py`.
- `LemPair.forward` is the sole owner of the A dual-cutoff topology and the C
  post-backbone pair-refinement wiring. The legacy
  `lem_moe_v3_h0.py` forward remains untouched.
- A keeps the accepted QHFlow2 topology:
  - `r_max`/precomputed metadata continues to define the ordered full head pair
    set and `active_edges`;
  - `mp_cutoff` defines only the private backbone edge subset;
  - a real split runs the backbone on the MP subset, then reconstructs every
    ordered head edge from mature endpoint nodes through one shared
    SO(2)-equivariant `UpdateEdge` pair readout;
  - `mp_cutoff=None` and an all-true MP mask retain the exact legacy forward
    branch.
- C remains a separate reusable module,
  `dptb/nn/embedding/pair_so3_refine.py`, imported only by `lem_pair.py`.
  It is constructed only when `pair_refine_enable=true` and is applied after
  all backbone layers and before the existing H-B0 head.

### B switch decision

- Use new-file subclasses `PairLayer`, `PairUpdateNode`, and
  `PairUpdateEdge`.
- Add only narrow protected construction/coefficient/normalization hooks to
  `lem_moe_v3.py` where needed:
  - legacy factories still instantiate the exact existing `Layer`,
    `UpdateNode`, and `UpdateEdge` classes in the same order;
  - legacy residual hooks execute the original sigmoid/rsqrt operations in the
    original order;
  - pair subclasses override the coefficient hook to return `(1.0, 1.0)` when
    `res_update_additive=true`;
  - `PairLayer` replaces only the new model's latent `LayerNorm` with
    `Identity` when `latents_layernorm=false`.
- This avoids copying the large update forwards and keeps old constructor RNG,
  parameter names, and arithmetic unchanged. Stage 5 will prove cross-tree
  tensor/state/output bit identity against the untouched `a6f152e` source.

### RF1 decision

- `a6f152e` RF1 dispatch is for `HamiltonianCFM` subclass overrides of
  block-topology helper leaves (`_block_primary_topology_keys`,
  `_block_topology_keys`, snapshot/restore/match); it does not dispatch the
  string-valued `embedding.method` config validator.
- `LemPair` naturally inherits `LemMoEV3H0` runtime full-edge coverage
  behavior, but `block_ode_contract.py` still explicitly rejects every method
  except `lem_moe_v3_h0`. Therefore Stage 3 will make the minimal validator
  change to allow exactly `{"lem_moe_v3_h0", "lem_pair"}`. No route-adapter or
  topology-helper change is needed.

### Reproduction commands

```powershell
git status --short --branch
git show a6f152e --stat
git -C E:\deeptb\wt_0723_dualcut diff c7d097f --stat
git -C E:\deeptb\wt_0723_norm diff c7d097f --stat
git -C E:\deeptb\wt_0723_b1 diff c7d097f --stat
rg -n "lem_moe_v3_h0|_block_topology_keys|dispatch" `
  dptb/utils/block_ode_contract.py dptb/nnops/block_ode `
  dptb/tests/test_block_ode_subclass_dispatch.py
```

## Stage 1 — new model + dual cutoff + pair refinement

- Timestamp: 2026-07-23 17:55:00 +08:00
- Status: SUCCESS
- Added registered `LemPair(LemMoEV3H0)` in `lem_pair.py`.
- The legacy H0 forward is reused as the control skeleton and dispatches into
  pair-specific `PairInitLayer` / `PairLayer` subclasses:
  - no split calls the original init/layer methods directly;
  - a real split aggregates initial nodes and every backbone layer only over
    `mp_mask`;
  - the final pair layer runs one fresh full-edge `UpdateEdge` readout from
    mature nodes, with full ordered `active_edges` unchanged.
- Kept `PairSO3RefineTP` in its own module. `LemPair` inserts it by overriding
  the existing block-native head entry point, exactly after the final backbone
  output and before H-B0 decoding.
- `lem_moe_v3_h0.py` and `lem_moe_v3_h0_helpers.py` are untouched.
- Added two protected type factories to `lem_moe_v3.py`; legacy factories
  still return the exact old classes and construction order.
- Validation:
  - py_compile: PASS;
  - real-split H-B0 smoke: 2 MP edges / 12 ordered head edges, finite
    `(12, 4, 4)` blocks;
  - default-off `LemPair` versus equivalent `LemMoEV3H0`: constructor RNG,
    all 80 state tensors, node blocks, edge blocks, and overlap latents were
    bit-exact;
  - standalone pair refinement: 3 passed;
  - legacy H-B0/registry focused regression: 22 passed.

Reproduction:

```powershell
$env:PYTHONPATH='E:\deeptb\wt_0723_merge'
C:\Users\16608\.conda\envs\dptb\python.exe -m py_compile `
  dptb/nn/embedding/lem_moe_v3.py `
  dptb/nn/embedding/lem_pair.py `
  dptb/nn/embedding/pair_so3_refine.py
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest `
  E:\deeptb\wt_0723_b1\dptb\tests\test_pair_so3_refine.py -q
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest `
  dptb/tests/test_hb0_active_edge_contract.py `
  dptb/tests/test_output_route_registry.py -q
```

## Stage 2 — pair-backbone norm switches

- Timestamp: 2026-07-23 18:01:00 +08:00
- Status: SUCCESS
- Added `PairUpdateNode` / `PairUpdateEdge` subclasses and made `PairLayer`
  construct only those subclasses through protected legacy-default factories.
- `res_update_additive=true` overrides the new model's residual coefficient
  hook with exact Python scalars `(1.0, 1.0)` for node, edge, and latent
  streams. The default delegates to the original sigmoid/rsqrt hook.
- `latents_layernorm=false` replaces only `PairUpdateEdge.ln` (including the
  optional full-pair readout) with `torch.nn.Identity`.
- Equivalent default `LemPair` versus `LemMoEV3H0`: constructor RNG, state-dict
  keys, and all 128 state tensors were bit-exact in the two-layer fp64 probe.
- Enabled structure probe: every layer/update had the pair subclass, every
  latent norm was `Identity`, and both residual hooks returned exactly
  `(1.0, 1.0)`.
- Focused legacy regression: 15 passed.

Reproduction:

```powershell
$env:PYTHONPATH='E:\deeptb\wt_0723_merge'
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest `
  dptb/tests/test_hb0_active_edge_contract.py `
  dptb/tests/test_equivariant_norm_precision.py -q
```

## Stage 3 — registry / contract / argcheck

- Timestamp: 2026-07-23 18:06:00 +08:00
- Status: SUCCESS
- Added `slem_pair()` as `slem_h0()` plus the pair-only controls:
  `mp_avg_num_neighbors`, `res_update_additive`, `latents_layernorm`, and all
  five `pair_refine_*` fields. `mp_cutoff` is inherited from `slem()`.
- Added `lem_pair` to the embedding Variant and cutoff extraction method set.
- Changed the block-ODE method gate to the exact allowlist
  `{"lem_moe_v3_h0", "lem_pair"}` while retaining an error string compatible
  with existing red-team assertions.
- Added `lem_pair` to legacy SwiGLU checkpoint compatibility routing.
- Strict dargs normalization/check of an H-B0 config with every new field:
  PASS; normalized method `lem_pair`, pair refinement enabled.
- Real block-ODE water overlay with `method: lem_pair`: contract validation
  PASS.
- Focused config/contract regression: 46 passed, 13 baseline deprecation
  warnings.

Reproduction:

```powershell
$env:PYTHONPATH='E:\deeptb\wt_0723_merge'
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest `
  dptb/tests/test_output_route_config_argcheck.py `
  dptb/tests/test_block_ode_graph_contract.py `
  dptb/tests/test_block_ode_redteam.py -q
```

## Stage 4 — merged `test_lem_pair_*` regression suite

- Timestamp: 2026-07-23 18:16:00 +08:00
- Status: SUCCESS
- Added shared fp64 fixtures plus four focused suites:
  `test_lem_pair_dual_cutoff.py`, `test_lem_pair_norm_switches.py`,
  `test_lem_pair_refine.py`, and `test_lem_pair_flow_contract.py`.
- Result: **7 passed, 2 baseline deprecation warnings** in 36.72 s.
- Hard-gate evidence:
  - `mp_cutoff=None` versus all-active cutoff: node blocks, edge blocks, and
    overlap latents bit-exact;
  - real dual split preserved full ordered head rows and produced fp64 AO-block
    equivariance max drift `5.8286708792820718e-16`;
  - additive node/edge/latent residuals were bit-exact to `2 + 1 = 3`;
  - every disabled latent norm was `Identity`;
  - pair-refinement invariant-weight drift
    `5.5511151231257827e-17`, AO-block drift
    `2.1163626406917047e-16`, and nontrivial minimum refinement amplitude
    `3.8065839747235534e-02`;
  - enabled residual block-ODE flow passed ordered full-edge coverage,
    strict certification, edge graph ownership, and finite-output checks;
  - default-all-off `LemPair` versus equivalent `LemMoEV3H0`: constructor RNG,
    complete state dict, node blocks, edge blocks, and overlap latents were
    bit-exact.

Reproduction:

```powershell
$env:PYTHONPATH='E:\deeptb\wt_0723_merge'
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest `
  dptb/tests/test_lem_pair_dual_cutoff.py `
  dptb/tests/test_lem_pair_norm_switches.py `
  dptb/tests/test_lem_pair_refine.py `
  dptb/tests/test_lem_pair_flow_contract.py -q
```
