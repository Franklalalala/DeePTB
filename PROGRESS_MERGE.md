# IN PROGRESS — Stage 0 scope complete; implementation not yet started.

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
