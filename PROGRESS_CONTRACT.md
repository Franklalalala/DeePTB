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
