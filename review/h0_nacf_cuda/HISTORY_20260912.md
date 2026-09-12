# H0 / NACF CUDA: independent review requested

This is a frozen review snapshot, not a production merge. Numerical jobs are paused by user request.
Do not resume, submit new work, recompile, or retest a legacy version as part of the initial review.

## Current result

Same 100-structure cohort (50 SOC + 50 non-SOC, seed 20260912): 20 completed, 19 PASS,
1 NUMERICAL_FAIL. Thresholds remain full Hmax < 5 meV and Smax < 1e-6. The incomplete
80 include two suspended workers, not 80 finished failures. SOC mp-561353 remains at
13.24518 meV (Smax 1.90217e-8). Neither all-SOC completion nor 100/100 acceptance is established.

Original six failure cases after current full-matrix fixes:

| Case | Current Hmax meV | Status |
|---|---:|---|
| nonSOC 10868 | 0.603987 | PASS |
| nonSOC 1773 | 1.479808 | PASS |
| nonSOC 2251 | 0.010038 | PASS |
| nonSOC 9278 | 0.032339 | PASS |
| SOC mp-510294 | 0.048992 | PASS |
| SOC mp-561353 | 13.245180 | FAIL |

## Implemented

1. Corrected Gemini's host/device harmonic constants; fixed CUDA current-stream/device guards;
   reject orbital/projector l>4. Rejected runtime-batch lane 789f950 remains unmerged.
2. Strict installation-time H0 and NACF binary loading. ABI/source/dependency/binary/device
   identity checks; no implicit NVCC/Ninja/cpp_extension.load in inference.
3. H0 pickle-free species and prepared S/T/projector/Gaunt/D tables, exact AO cubic coefficients,
   content identities and bounded process reuse. 100 compositions prepared from 59 distinct
   UPF/ORB inputs (~8.9 GiB installed). Geometry/FFT-dependent quantities remain online.
4. Fixed validated numbered UPF tags with legacy index="*". Fixed omitted per-atom STRU mag.
   Added per-species valence normalization and ABACUS pseudo_rcut/odd-msh field preparation.
   Kept projector two-center inputs separate from the field cutoff.
5. NACF now has a precompiled extension and persistent exact GPU-buffer/rotation/packing cache.
   Normal source validation still occurs before cache selection; cache does not bypass missing
   or corrupt P23 checks. Node=P23, edge=P2, H=prior+residual, complex SOC D and stored graph remain.

## Review priorities (do not assume these are all resolved)

* P1: Diagnose the residual SOC Mn error independently, without H/S fitting or threshold relaxation.
  Distinguish polarized PBE/magnetization treatment, local interpolation, radial and projector errors.
* P1: Suspension is NOT yet a safe restart protocol. new100_v3.py uses child.wait(timeout=1200);
  its monotonic deadline advances while SIGSTOPped. Blind SIGCONT of the dispatcher after a long
  pause may kill the resumed workers. Its fresh supervisor also resets state and reruns every ID.
  Design a provenance-aware resume before any user-authorized continuation. Keep current jobs stopped.
* P1/P2: Audit offline cache identity/immutability. H0 fingerprint includes source paths and full
  SpeciesData; dependency enforcement is at assemble_h0's table_contract boundary, while direct
  prepared_two_center callers bypass that extra guard. Resident objects are mutable. Check corruption,
  source drift, changed settings/spin, concurrent preparation and device identity behavior.
* P2: H0 source_contract.json was added after initial table preparation, using unchanged numerical
  dependencies. This is a retrospective receipt, not proof that an earlier manifest existed.
  prepare_tables.py only verifies a pre-existing contract: review unmanifested-store handling.
* P2: NACF cache .sha256 and .pt publication is not one atomic transaction and has no per-key lock.
  Concurrent preparation is untested; review fail-closed races and runtime repair policy.
* P2: Cache speed is not yet benchmarked end-to-end. H0 hashes/transfers serialized GPU state on disk
  read; NACF reads source tables before cache lookup and hashes the source arrays. Measure those
  costs, element switching, resident memory and composition-level duplication before speedup claims.
* P2: Generic geometry APIs and production_io are different contracts. production_io reads logged
  FFT/cell-gauge information for validation. Do not make generic inference depend on oracle H/S/logs.
* P2: Verify exact ABACUS radial cutoff, origin, normalization, per-atom moments, units and cell gauge.
  Transverse/angle magnetization is explicitly unsupported. Two-center adapter uses private pyabacus
  C++ layout (offset 8); it is ABI-pinned, not a portable library API.

## Evidence limits

H0 prior component tests: T max 1.81e-14 eV, sampled Vnl 1.36e-14 eV; nondefault-stream local-grid
tests; four unique full structures / five element-switch calls. These precede the final field fix.
Offline tests checked bitwise fresh/disk tables for four scalar/SOC compositions, reordered species,
mutation rejection, and a full H0 with UPF/ORB reads/tabulation blocked. Not a whole-code sandbox:
the constructor monkeypatch cannot prove that no native implementation could ever read a source file.
NACF: 30 relevant tests passed, then 3 final cache tests passed. Final field helper: 1 focused test.
No fresh independent FP64 parity for all 100, no measured production speedup, no new NACF physical
label audit. Only L40S sm89 is numerically tested; sm80/86/89/90 cubins exist.

The 20 v3 records contain two assembly hashes. Their EXACT difference is the later two-line
table_contract import/check, verified by reconstructing the earlier bytes and hash. Do not flatten
this into a claim that all runs used one immutable whole-tree commit. Per-case records are retained.

## Read order

1. h0/production_io.py, h0/h0rebuild/assemble.py, h0/h0rebuild/field_inputs.py.
2. h0/h0rebuild/offline.py, table_contract.py, precompiled.py and h0/prepare_tables.py.
3. h0/csrc and cuda_two_center.py/cuda_local_grid.py; patches/*_vs_gemini.patch.
4. nacf_overlay/dptb/nacf/precompiled.py, precompile.py, prepared.py, assembly.py; focused patch.
5. h0/new100_v3.py and new100.py; tests and evidence contracts.

Return prioritized findings with file/line, trigger, impact and smallest justified fix. Separate
confirmed defects from hypotheses, and report scientific correctness, runtime portability, cache
correctness and performance evidence separately. Do not perform a broad refactor before review.
