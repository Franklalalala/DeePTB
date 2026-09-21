# Full NACF candidate prior: `dptb.nacf.candidate.CandidatePriorPlan`

The fixed100 evaluation harness assembled the full candidate prior by hand from the
accepted research scripts. `CandidatePriorPlan` is the maintained entry for exactly that
composition, with an explicit, hashed recipe identity and fail-closed provenance checks:

```
node = P23 + onsite XC (accepted atom-centred local quadrature, fixed 27 Bohr neighbourhood) + c S
edge = P2 + edge VNA3c + direct pair XC + c S + environment XC (one arm: d2 | d2_moment | mcweda)
c    = 4 pi (sum_a M2_a) Ry_to_eV / (3 Omega)        fully periodic cells only
```

Everything is injected; nothing is discovered from directories, mpids, H, H0 or labels:

```python
from dptb.nacf.candidate import (CandidatePriorPlan, CandidateRecipe, ConvergenceOrderPolicy,
                                 FixedOrderPolicy, PairXCTables, AtomicMoments, XC_KEY)
recipe = CandidateRecipe(envxc_arm="mcweda", stabilization="v2", order_policy=ConvergenceOrderPolicy())
plan = CandidatePriorPlan(recipe, table_bank=bank, envxc_bank=xbank, onsite=evaluator,
                          onsite_identity={"species_sources": {...}, "xc_functional": XC_KEY,
                                           "potential": "<implementation label>", "density_definition": "..."},
                          pair_xc=PairXCTables.from_manifest(...), atomic_moments=AtomicMoments.from_json(...),
                          topology_library="/path/libnacf_topology.so")
prepared = plan.prepare(geometry)          # symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift, pbc
out = prepared()                           # node_ao_ev, edge_ao_ev, overlaps, components, diagnostics
features = prepared.feature_plan(idp).pack(out["node_ao_ev"], out["edge_ao_ev"])
```

## Module map

| Module | Owns | Torch import |
|---|---|---|
| `dptb/nacf/candidate_policy.py` | declarations: frozen `CandidateRecipe`, canonical XC vocabulary (`XC_KEY`, `XC_FUNCTIONAL`), immutable `FusionSettings`, frozen `FixedOrderPolicy` / `ConvergenceOrderPolicy`, the `order_check` row schema | lazy, inside methods |
| `dptb/nacf/candidate_checks.py` | shared input/provider checks: `validated_geometry` (raw graph arrays), `pair_shell_problems` (pair-XC headers vs P2), `species_source_problems`, `declared_xc`, `CandidateInputError` | none (NumPy) |
| `dptb/nacf/candidate.py` | injected numerical providers (`PairXCTables`, `AtomicMoments`), binding and identity (`CandidatePriorPlan`), numerical prepare/forward (`PreparedCandidate`); re-exports every public name of the 0920 entry | yes |
| `onsite.py`, `envxc.py`, `edge_vna.py`, `assembly.py`, `fusion.py` | the engines the plan composes; contracts unchanged (see `dptb/nacf/FUSION.md`) | yes |

## Identity rules at construction

All raise `CandidateIdentityError`: unsupported arm/stabilization, fused contraction or unknown
fusion keys in the recipe, SOC banks, D2 arms without background layers, an onsite evaluator
whose radius differs from the recipe, missing `onsite_identity` fields, a density definition
that differs between the recipe, the onsite provider and the EnvXC tables, AO shells or orbital
cutoffs that differ between P2 and EnvXC tables, any species whose declared UPF/ORB/source hash
differs between two families or is missing in one (disjoint source keys do not establish
identity), and any pair-XC table whose ordered `left_shells`/`right_shells` header differs from
the P2 shells of its two species. The shell check reads the real table headers: two s shells plus
one p shell and one d shell are both five AOs and every matrix shape agrees, so neither source
hashes nor output dimensions prove the gauge. Table `support_bohr` is not compared.

Precompiled pair-XC tables retain their own buffers; the `PairXCTables` constructor
does not migrate them. Every buffer must already be on the table bank's device,
and every floating buffer must match its dtype. Binding rejects mismatches with
the table and buffer name rather than changing shared providers implicitly.

The onsite provider must declare the recipe's functional. `onsite_identity["xc_functional"]` is
the contract (`XC_KEY = "lda_pz81_unpolarized"`, or the recipe label); `onsite_identity["potential"]`
stays a free implementation label kept for provenance. A label that recognizably names the
functional (`pz81` token, e.g. `"in-repo lda_pz81_v_dv_torch"`) is accepted without the key; a
label that recognizably contradicts it (`PBE`, `PW92`, `zero potential`, ...) is refused even with
the key; any other label needs the key. What the injected callable actually computes is not
inferred; that remains the caller's responsibility, as stated in the module docstring.

`plan.identity` holds the recipe identity, the family manifests, the validated species and, new
in 0921, `families.onsite.xc_functional`; `plan.identity_sha256` hashes it and every forward
reports the hash. The recipe identity itself (`identity["recipe"]`) is unchanged from 0920, so the
same recipe yields the same recipe dict, while the plan hash differs from 0920 receipts because of
the added onsite field.

## Structure input at prepare

`validated_geometry` runs before any cast, provider or plan sees the caller's arrays: finite
`[n, 3]` positions and a nondegenerate `[3, 3]` cell, three boolean `pbc` flags, `edge_index`
`[2, E]` inside the structure and `edge_cell_shift` `[E, 3]`. Integer arrays pass; floating arrays
are accepted only when finite and exactly integral, so a legal integer-valued float graph works
and fractional shifts such as `+0.25 / -0.25` raise `CandidateInputError` (a `ValueError`) instead
of being truncated to another graph. The supported magnitude is that of the native topology core.
Edge rows keep the caller's order; the plan owns copies, so later caller edits cannot mix new
onsite coordinates with old edge plans. Then every species must be covered by every family,
every species pair used by the directed edges must have a pair-XC table, and partially periodic
cells are refused because of the cS zero point.

After the order policy selects one order per atom, preparation calls `qgrid` once
per selected `(species, order)` to verify nonempty `[points, 3]` coordinates and
exactly `[points, P2_species_AO_count]` basis values. Global AO padding remains
legal but cannot hide an incorrect species basis width. Providers should cache
their quadrature by species/order and remain immutable while bound. These shape
checks do not establish shell ordering, phases, weights or provenance.

## Order policy and diagnostics

Order policy is geometry-only: `FixedOrderPolicy(order)` or `ConvergenceOrderPolicy` (the
accepted rule re-run on the structure: medium vs fine, fine vs finer, optional 23-Bohr tail
check). Each `order_checks` row carries `selected_order`, `convergence_checked` and `converged`.
A fixed order records `convergence_checked=False, converged=None`: it is a choice, not a
convergence proof, and no `passed` flag is emitted. The convergence policy records
`convergence_checked=True`, the measured `medium_fine_eV` / `fine_finer_eV` /
`environment_tail_eV` values, the `tolerance_eV` and the verdict in `converged`; `strict=True`
raises on failure, otherwise a failed verdict is only recorded and is not a release acceptance.
Forward diagnostics summarize the rows as `convergence_checked` and `converged` (`None` unless
every atom was compared); `prepared.convergence_summary()` returns the same pair.

## Immutability of bound settings

`CandidateRecipe`, `FusionSettings` and the built-in order policies are frozen dataclasses;
assigning to their attributes raises. `onsite_identity` is copied at binding. For custom
`OrderPolicy` subclasses the plan snapshots `identity()` at construction and re-checks it before
every `prepare`; a changed identity raises `CandidateIdentityError` instead of evaluating under a
stale `identity_sha256`. Injected providers and their tensors must remain immutable for the plan
lifetime; construct new providers and a new plan after changing tables.

## Boundaries

The onsite evaluator objects (quadrature, density splines, potential) carry no provenance, so the
adapter passes `onsite_identity`; the plan checks it against the recipe and the other families
but cannot derive it. `NACFGeometryPredictor` (P23/P2 prior of trained checkpoints) is untouched;
a checkpoint trained on the old prior must not be evaluated with this plan as if it were the same
prior. Fused contraction is never enabled by the recipe. The `v1`/`v2` stabilization choice is
part of the identity; `v1` reproduces the fixed100 receipts, `v2` is the audited composed-residual
recipe. Only the implemented unpolarized LDA-PZ81, neutral-valence-plus-NLCC and cS conventions
are accepted; arbitrary XC or zero-point labels are rejected. Pair-XC providers without a manifest
hash bind their compiled contents once at construction. The actual atomic M2 values are always
included. Onsite packed-density and grid caches key on tensor object, storage, strides, offset and
version; replacing a knot or coefficient tensor by another object or view is detected, while
in-place writes through inference tensors, `.data` or NumPy views still require the explicit
`invalidate()` / `invalidate_grid_radius(quadrature)` calls.
