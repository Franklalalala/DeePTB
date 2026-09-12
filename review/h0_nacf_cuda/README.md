# H0 / NACF CUDA review fixes

The 2026-09-13 continuation fixes the cache, validation and native-boundary defects
reported against commit 72f3cb6. This remains a review branch, not a production merge.
The parent repository's installed model and training recipes are unchanged.

The old 100-case campaign was paused at 20/100 (19 PASS, one Mn NUMERICAL_FAIL),
not completed. Its six verified stopped processes were retired without resuming
their expired supervisors. Their results remain historical evidence. A new finite
acceptance run uses the original 50 SOC + 50 non-SOC cohort with Hmax < 5 meV and
Smax < 1e-6. Completion is distinct from every case passing.

## Implemented and checked

- F02: field-only UPF view applies the reference reader's even-to-odd mesh rule
  before the radial cutoff; full projector input is kept separate.
- F03: post-nonlinearity PW projection for spin gradients and flux divergence;
  the same divergence operator is used by scalar CPU/CUDA PBE. LibXC receives
  the explicit 1e-6 density threshold used by the reference implementation.
- F04: native S/T, projector and nonlocal entries validate devices, dtype, rank,
  contiguous layout, descriptor/index bounds and table shapes, set CUDAGuard,
  and check kernel launches. Python normalizes strided displacements and validates N.
- F05/F06: lowest-level H0 loaders enforce the numerical-source contract. Every
  table key and record binds generator dependencies. Nonempty unmanifested stores
  cannot be stamped current. Private resident masters return independent buffers;
  an RLock and CUDA readiness event protect publication/reuse. Disk validation
  occurs before GPU upload. Catalog publication is atomic.
- F07: unique attempt directories, immutable source/binary/table/input identities,
  child return-code + terminal-receipt checks, version-aware reuse, and real
  stopped-worker accounting. Queue/lock time is outside the active budget.
  Unobserved supervisor suspension gaps extend the budget rather than triggering
  immediate termination. Existing live orphan attempts block duplicate launches.
- F08/F09: mag and magmom are equivalent; unsupported charge/XC inputs fail
  explicitly. UPF indices require decimal integers; numbered '*' compatibility
  is restricted to overflow slots >=10 and strict mode rejects it.
- F10/F11: NACF keys include rotation bases/directions and builder source identity;
  cold and disk hits share validation. One self-checksummed atomic artifact replaces
  the checksum/data publication race. Corruption still raises.
- F12: local-grid consumes and validates saved AO polynomials without refitting.

## Evidence and open numerical issue

25 focused tests passed (23 original focused/cell-gauge checks plus two additional
tests; the extended GPU check was also rerun). They include real L40S current-device
and nondefault-stream execution, current PyAbacus component parity, private-buffer
mutation isolation, real SIGSTOP/SIGCONT budget behavior, and synthetic NACF cache
publication/rotation/corruption cases. A further 30 NACF integration tests passed
in a separate copy of the original SOC parent, reusing native binary c1c84fc8.
These synthetic NACF cases are not a new physical label validation campaign.

100/100 compositions were prepared from 59 unique UPF/ORB inputs, with zero
preparation errors, in a new v2 namespace. That is table preparation, not 100/100
scientific acceptance. The two-center extension was explicitly rebuilt once for
sm80/86/89/90; local-grid reused its unchanged binary. Runtime forbids compilation,
UPF/ORB reads and two-center tabulation. Only L40S/sm89 was numerically exercised.

Four initial full-structure checks completed on the fixed source:

| Case | Hmax (meV) | Result |
|---|---:|---|
| SOC mp-561353 (Mn) | 13.244664 | NUMERICAL_FAIL |
| SOC mp-510294 | 0.027804 | PASS |
| nonSOC 10868 | 0.010696 | PASS |
| nonSOC 1773 | 0.016418 | PASS |

F01 remains OPEN. Mn has nearly unchanged error concentrated in the s-orbital
block, with a dominant common-spin contribution. An independent onsite-only
experiment implementing the fixed ABACUS reference's local AO interpolation
(0.001 bohr uniform grid, Uni_RadialF and Hermite interpolation) still gave
13.236165 meV. It was diagnostic only and was not merged into the runtime.
Neither this experiment nor component bookkeeping identifies the reference
T/Vlocal/Vnl error. The next decisive evidence is component-resolved output from
the same reference build, including its fields near atom/grid coincidences.
No matrix fitting, potential offset, threshold relaxation or case substitution
was used. Archived v3 and current v4 records are never combined as one version.

## Run and deployment contract

Use `h0/acceptance.py` (new100/new100_v3 main entries delegate to it). Running it
again selects the same immutable identity, reuses only accepted terminal attempts,
and retries errors. A changed numerical version starts a separate run namespace.
Do not edit the selected runtime while a campaign is running. A dispatcher restart
must reconcile any living orphan worker before continuing.

Installation is explicit: select the pinned Python/Torch/CUDA/PyAbacus environment,
run `precompile.py` once when native sources change, prepare a new table namespace,
then use `precompile.py --check` and compiler-free runtime. Existing manifests retain
strict ABI/dependency/source checks. This is still an environment-specific build:
relative dependency packaging, other-GPU numerical validation, true cross-structure
GPU batching and end-to-end speedup qualification remain open.

`SOURCE_MANIFEST.json` identifies this tree. `HISTORY_20260912.md` and the patches
subdirectory describe the parent snapshot and are not current deployment commands.
`check_offline.py` is a historical verifier with the superseded shared-object identity
expectation; use `tests_h0fast/test_review_boundaries.py` for the current cache API.
No credentials, raw H/S datasets, large installed tables or binaries are in Git.

Additional bounded input audit: the 59 prepared species have at most nine radial
channels and lmax=3. The legacy local-grid kernel has a fixed 16-channel stack
array and insufficient arbitrary-input validation. This does not affect this
cohort, but general-purpose local-grid native API hardening remains open; do not
advertise support for >16-channel bases or unvalidated raw native tensors.

## Latest paused snapshot (2026-09-13)

At user request, the v4 dispatcher and two workers are paused after 76/100 completed: 73 PASS, 2 NUMERICAL_FAIL, 1 budget ERROR; 22 cases have not started. Source identity was reverified unchanged. Mn Hmax is 13.2446639 meV and Ca (SOC_mp-19824 atom 0) is 7.5399653 meV. SOC_mp-23435 requires an estimated 5143.45 MiB versus the configured 4096 MiB; it was not retried. The queued correlation-mask diagnostic was cancelled without producing results. See ONLINE_REVIEW_BRIEF_CN.md and SCALE68_REVIEW.md for the online review and full68 expansion scope.
