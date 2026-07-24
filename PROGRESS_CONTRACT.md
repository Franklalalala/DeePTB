# Lane A — dual cutoff contract repair

Overall status: IN PROGRESS

## Stage 0 — baseline

- Timestamp: 2026-07-24 08:19:03 +08:00
- Status: PASS
- Worktree: `E:\deeptb\wt_0724_contract`
- Branch/base: `feat/0724-pair-contract` / `dcacda50876a1bd9aae82d26cd0457ef7a4dbb92`
- Required source and tests read completely.
- Baseline result: 7 passed, 2 warnings in 38.88 s (wall clock 42.839 s).
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_common.py dptb/tests/test_lem_pair_dual_cutoff.py dptb/tests/test_lem_pair_flow_contract.py dptb/tests/test_lem_pair_norm_switches.py dptb/tests/test_lem_pair_refine.py -q`

## Stage 1 — reviewer patches

- Timestamp: 2026-07-24 08:21:52 +08:00
- Status: PASS
- Applied as two independent commits with reviewer authorship:
  - `00c027c fix(lem-pair): harden cutoff and compatibility contracts`
  - `94aca71 fix(lem-pair): preserve full-edge context in dual readout`
- The patches were generated against CRLF working-tree blobs while this repository
  normalizes the index to LF. A literal first attempt failed two context matches;
  `git am --keep-cr --ignore-whitespace` applied every hunk and retained provenance.
- Result: 11 passed, 2 warnings in 46.58 s (wall clock 50.223 s).
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_common.py dptb/tests/test_lem_pair_dual_cutoff.py dptb/tests/test_lem_pair_flow_contract.py dptb/tests/test_lem_pair_norm_switches.py dptb/tests/test_lem_pair_refine.py dptb/tests/test_lem_pair_contract_validation.py -q`

## Stage 3 — lifecycle-safe pair topology

- Timestamp: 2026-07-24 08:42:19 +08:00
- Status: PASS
- Chosen design: dual projection/readout ownership moved into the final
  `PairLayer`; the initial full-edge context is carried explicitly as a
  layer-to-layer value. This avoids changing the duplicated H0 forward loop
  while removing all owner references and pair forward-time context state.
- Every pair layer stores a pure cutoff copy and a non-persistent type-pair
  cutoff lookup buffer. No pair module attribute or buffer changes in forward.
- Exact state_dict migration map (`L = n_layers - 1`):
  - `dual_cutoff_readout_normalization` ->
    `layers.L.dual_cutoff_readout_normalization`
  - `dual_cutoff_pair_readout.*` ->
    `layers.L.dual_cutoff_pair_readout.*`
  - `dual_cutoff_edge_context_projection.*` ->
    `layers.L.dual_cutoff_edge_context_projection.*`
- State key count remains 225 in the two-layer dual fixture; only the above
  prefixes and traversal order changed.
- e3nn's `SphericalHarmonics.sph_func` ScriptFunction cache was the remaining
  deepcopy/pickle blocker. LemPair now uses the exact underlying Python
  spherical-harmonics function; golden outputs remain bit-identical.
- Lifecycle tests: 3 passed in 31.65 s. Deepcopy object/module/parameter graphs
  are independent, strict state_dict round-trip is exact, and whole-object
  `torch.save`/`torch.load(weights_only=False)` is exact.
- Full stage result: 19 passed, 2 warnings in 63.37 s.
- S2-to-S3 golden: dual `torch.equal=True`, legacy `torch.equal=True`, both
  `max|delta|=0`. Temporary golden tensors were removed after comparison.
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_common.py dptb/tests/test_lem_pair_dual_cutoff.py dptb/tests/test_lem_pair_flow_contract.py dptb/tests/test_lem_pair_norm_switches.py dptb/tests/test_lem_pair_refine.py dptb/tests/test_lem_pair_contract_validation.py dptb/tests/test_lem_pair_lifecycle.py -q`

## Stage 4 — schema ownership, fail-closed validation, migration

- Timestamp: 2026-07-24 08:49:48 +08:00
- Status: PASS
- Consumer audit: besides LemPair, `dptb/nn/embedding/lem_cutoff.py`
  independently implements `mp_cutoff`. The field was removed from public
  `slem()` and retained only by `slem_pair()` plus a dedicated
  `slem_cutoff()` schema. Strict argcheck rejects `lem_moe_v3_h0 +
  mp_cutoff` and accepts both real consumers.
- Constructor red-team now rejects scalar/dict NaN, infinity, non-positive
  values, booleans, missing basis species, unknown species, and invalid
  `mp_avg_num_neighbors`.
- Runtime dict masking now raises on malformed bond labels, missing element
  keys, and unknown bond-type rows instead of silently excluding them.
- Added `load_lem_h0_backbone(model, state_dict, *,
  allowed_missing_prefixes)`. It rejects uncovered missing keys, every
  unexpected key, and unused allowlist prefixes. Public top-level
  `dual_cutoff_*` allowlist prefixes map explicitly to the Stage-3 final-layer
  ownership path.
- Legacy `LemMoEV3H0` -> dual+refine LemPair migration passes with the exact
  allowlist; a single renamed source key fails closed. The existing all-off
  strict-load/bit-exact test remains green.
- Gate result: 82 passed, 1 warning in 51.03 s (wall clock 54.791 s).
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_contract_validation.py dptb/tests/test_configuration_canonicalization.py dptb/tests/test_output_route_config_argcheck.py -q`

## Stage 5 — six hard-gate groups

- Timestamp: 2026-07-24 09:08:41 +08:00
- Status: PASS
- G1 batch partition invariance: A standalone, A+B, and B+A are bit-exact
  across node Hamiltonian, edge Hamiltonian, and overlap rows;
  `max|delta|=0`.
- G2 dual + block-ODE:
  - non-MP H0 row perturbation output `max|delta|=4.9686738103199395e-04`;
  - non-MP residual row perturbation output
    `max|delta|=6.8409908952647008e-04`;
  - H0 input-row gradient norm `2.4423842928700354e-02`;
  - residual input-row gradient norm `2.0258515789310576e-02`;
  - nonzero residual-state SO(3) block drift
    `7.6327832942979512e-17`.
  - The rotation subtest holds mapper-order H0 at its valid zero tensor while
    rotating the nonzero residual AO state. Nonzero H0 sensitivity is tested
    independently in the same test; mapper/codec H0 covariance remains covered
    by `test_residual_ao_block_ode.py`.
- G3: consecutive all-active and real-split backward passes give non-None
  gradients for every `requires_grad` parameter, including dual projection and
  readout parameters.
- Dead parameter policy tightened for DDP: unreachable base environment weights
  under H0 `replace+both`, unused identity-residual linears, and discarded fresh
  readout latent parameters are frozen. Forward tensors and state_dict entries
  are unchanged.
- G4: split sizes `(0,)`, `(0, 6)`, and `(0, 0)` observed for single-zero,
  mixed, and all-zero MP cases; all outputs/gradients finite.
- G5 references the Stage-4 12-test red-team file; G6 references Stage-3
  lifecycle plus Stage-4 migration and is completed by Stage 6 cross-tree
  golden.
- New hard-gate file: 4 passed, 2 warnings in 29.90 s.
- Full Stage-5 gate (all `test_lem_pair_*` plus five required block-ODE
  suites): 208 passed, 136 warnings in 83.08 s.
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_*.py dptb/tests/test_hb0_active_edge_contract.py dptb/tests/test_block_ode_flow.py dptb/tests/test_residual_ao_block_ode.py dptb/tests/test_block_ode_graph_contract.py dptb/tests/test_block_ode_redteam.py -q`

## Stage 6 — base-vs-head legacy golden

- Timestamp: 2026-07-24 09:13:10 +08:00
- Status: PASS
- Added `scripts/crosstree_golden_lem_h0.py`. Each child imports and constructs
  `LemMoEV3H0` directly from its own PYTHONPATH; LemPair is not imported.
- Base tree verified at `a6f152e`; children run with
  `PYTHONDONTWRITEBYTECODE=1` to preserve the base tree as read-only.
- fp32: PASS, 184 ordered state tensors, all tensors and node/edge/overlap
  outputs `torch.equal`, `max|delta|=0`.
- fp64: PASS, 184 ordered state tensors, all tensors and node/edge/overlap
  outputs `torch.equal`, `max|delta|=0`.
- Wall clock: 99.620 s.
- Report:
  `F:\claude\0724_pair_iter2\results\crosstree_golden_A.md`
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe scripts/crosstree_golden_lem_h0.py`

## Stage 2 — configuration-determined dual architecture

- Timestamp: 2026-07-24 08:27:49 +08:00
- Status: PASS
- Added scalar/dict pairwise cutoff canonicalization. `mp_cutoff` is reduced to
  `None` only when every represented element pair is provably at least its
  corresponding `r_max`; otherwise dual mode remains configured.
- Removed batch-data-dependent all-active fallback, `_pair_run_dual`, and
  `mp_mask.all().item()`. Detached last masks remain diagnostics only.
- Fixed MP neighbor normalization as a construction-time constant on every
  dual pair layer; removed forward swap/restore.
- Tightened patch 1 semantics intentionally per REVIEW_A P1-1: a configured,
  non-redundant cutoff runs dual math even when all active rows are inside it.
- Fresh-readout dead-config probe: toggling only `res_update_additive` and
  replacing its latent LN with `Identity` gave `torch.equal=True`,
  `max|delta|=0`; assignments were removed because `res_update=False` and the
  returned readout latent is discarded.
- Test result: 16 passed, 2 warnings in 46.56 s (wall clock 49.961 s).
- S1-to-S2 `mp_cutoff=None` golden: `torch.equal=True`, `max|delta|=0`.
- D4 deferral: the private MP cutoff remains a hard topology mask in this
  energy/Hamiltonian-only scope. A smooth envelope and position-gradient
  contract are explicitly deferred to a force/stress/MD follow-up.
- Reproduce:
  `C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_lem_pair_common.py dptb/tests/test_lem_pair_dual_cutoff.py dptb/tests/test_lem_pair_flow_contract.py dptb/tests/test_lem_pair_norm_switches.py dptb/tests/test_lem_pair_refine.py dptb/tests/test_lem_pair_contract_validation.py -q`
