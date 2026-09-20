# 0920-stable integration

Historical integration snapshot. Follow-up interface fixes and module organization
are maintained in `0921-stable`; see `0921-stable.md`. The measurements below refer
to the stated earlier runs, not to final testing of every commit on this snapshot.

This branch continues `0917-stable` without rewriting its history. Historical
branches and deployed snapshots remain available for checkpoint compatibility.
NACF's maintained implementation is `dptb/nacf`; generated binaries, tables,
benchmark outputs and review bundles belong outside Git.

## Recent changes carried forward

| Commit | Behavior and scope |
| --- | --- |
| `286290a1` | Previous integration: H0/NACF, frozen-prior S2 flow and Switch top-1 routing. |
| `0ed9744f` | Grid-free third-center edge VNA with Tonari's standalone C++ neighbor core. |
| `23fd3c9c` | Verified forward-compatible PTX loading and GPU edge packing. |
| `2f47f7be` | Native topology for P23 onsite and P2 hopping, preserving canonical directed graph rows. |
| `80aa6a4a` | Shared feature templates and CUDA compact packing. |
| `a039e41b` | Clear failure when an old binary lacks native packing symbols. |
| `b3b7b9ad` | Explicit radial fusion and native density neighborhoods; contraction fusion remains opt-in. |
| `ed566a17` | Grid-free environment XC alternatives: D2 background, GSN moment and McWEDA. |
| `74ccb62f` | Shell-covariant envelope bounds for finite-rank moment stabilization. |
| `2fa65b6f` | Batched FP64 CUDA onsite density evaluation with the accepted local quadrature. |
| `d3e34350` | Maintained immutable prepared-store API, replacing an external evaluation-only subclass. |
| `ddc69100` | Atomic snapshot publication refuses to overwrite a concurrent result and preserves another writer's pending file. |

The previous contraction-fusion experiment did not pass the strict packed FP32
output gate for two real cases. Its existence is not a reason to enable it by
default. New numerical or backend defaults require explicit measured acceptance.

## Numerical and compatibility boundaries

The historical mixed prior is P23 onsite plus P2 hopping. The newer explicit
candidate additionally combines onsite XC, edge VNA, direct pair XC, one chosen
environment correction and the analytic overlap-weighted zero-point term.
These are distinct recipes. Existing model checkpoints must keep the recipe on
which they were trained; the generic legacy predictor does not silently switch.

Hopping evaluation is grid-free. Onsite XC still uses local quadrature with
neutral valence plus unscaled NLCC density and unpolarized LDA-PZ81. This is not
PBE, self-consistent charge, a full complex spinor Hamiltonian, or a force model.
Stored directed edges, image shifts, AO phases and units remain part of the
interface. Missing overlap or table provenance fails explicitly.

Tonari supplies the vendored C++ neighbor core under MIT, pinned in
`dptb/nacf/THIRD_PARTY.md`. NACF supplies the operator definitions, periodic-image
intersection rules and numerical kernels. The integration does not use Tonari's
Python package or CUDA provider.

## Verification and timing

Use the focused behavioral policy in `TESTING.md`. Build the topology library
explicitly and provide `DPTB_NACF_TOPOLOGY_LIBRARY`; install the verified CUDA
extension separately. Inference must not compile a new binary implicitly.

Keep backend parity separate from a change to stabilization. Compare the same
geometry, graph, tables, precision and quadrature policy. Report warm table,
fresh geometry preparation separately from cold loading, compilation and output
writing. A fixed development cohort is a regression set, not held-out physical
validation or completion of the 29,303-structure production dataset.

The integration baseline passed 86 NACF behavioral tests on an RTX PRO 6000,
including verified PTX execution, and the maintained general DeePTB smoke suite
passed 63 tests. The prepared-store concurrent-publication fix passed six CPU
tests; the unchanged CUDA reader had already passed in the 86-test run.

Independent fixed100 re-evaluation of the old recipe completed all 100 cases.
With resident tables and species quadratures, fresh geometry each call, FP64
arithmetic and three repeats, its mean/median/P95 times were
0.181114/0.138026/0.468441 seconds per structure. The maximum difference from
archived accepted features was 1.82e-12 eV. Pooled onsite/hopping matrix MAEs
were 95.698208/13.973641 meV; these are not band errors.

The separately versioned McWEDA v2 stabilization completed the same 100 cases:
onsite/hopping pooled MAEs were 95.698208/13.972847 meV. Raw onsite blocks were
unchanged. The largest per-structure hopping-MAE regression was 0.011616 meV;
the maximum individual changed hopping feature was 0.066191 eV. This is a
formula correction and must not be described as bitwise backend equivalence.
Legacy v1 remains available for existing recipe/checkpoint lineages.

NACF source byte identities include line endings. The scoped LF Git attributes
make new checkouts reproducible across platforms. Historical mixed-line-ending
prebuilt snapshots remain immutable and must retain their matching manifests;
new binaries are built explicitly from the release's LF source.
