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

