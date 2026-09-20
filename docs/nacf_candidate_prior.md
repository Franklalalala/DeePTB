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
                                 FixedOrderPolicy, PairXCTables, AtomicMoments)
recipe = CandidateRecipe(envxc_arm="mcweda", stabilization="v2", order_policy=ConvergenceOrderPolicy())
plan = CandidatePriorPlan(recipe, table_bank=bank, envxc_bank=xbank, onsite=evaluator,
                          onsite_identity={"species_sources": {...}, "potential": "...", "density_definition": "..."},
                          pair_xc=PairXCTables.from_manifest(...), atomic_moments=AtomicMoments.from_json(...),
                          topology_library="/path/libnacf_topology.so")
prepared = plan.prepare(geometry)          # symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift, pbc
out = prepared()                           # node_ao_ev, edge_ao_ev, overlaps, components, diagnostics
features = prepared.feature_plan(idp).pack(out["node_ao_ev"], out["edge_ao_ev"])
```

Identity rules (all raise `CandidateIdentityError`): unsupported arm/stabilization, fused
contraction in the recipe, SOC banks, D2 arms without background layers, an onsite
evaluator whose radius differs from the recipe, missing `onsite_identity` fields, a
density definition that differs between the recipe, the onsite provider and the EnvXC
tables, AO shells or orbital cutoffs that differ between P2 and EnvXC tables, and any
species whose declared UPF/ORB/source hash differs between two families or is missing in
one. Disjoint source keys do not establish identity. Per structure, every species must
be covered by every family and every species pair used by its directed edges
must have a pair-XC table; partially periodic cells are refused because of the cS zero
point. The recipe identity, the family manifests and the validated species are in
`plan.identity` and hashed in `plan.identity_sha256`, which every forward reports.

Order policy is geometry-only: `FixedOrderPolicy(order)` or `ConvergenceOrderPolicy`
(the accepted rule re-run on the structure: medium vs fine, fine vs finer, optional
23-Bohr tail check; `strict=True` raises on failure, otherwise the checks are recorded).

Boundaries: the onsite evaluator objects (quadrature, density splines, potential) carry
no provenance, so the adapter passes `onsite_identity`; the plan checks it against the
other families but cannot derive it. `NACFGeometryPredictor` (P23/P2 prior of trained
checkpoints) is untouched; a checkpoint trained on the old prior must not be evaluated
with this plan as if it were the same prior. Fused contraction is never enabled by the
recipe. The `v1`/`v2` stabilization choice is part of the identity; `v1` reproduces the
fixed100 receipts, `v2` is the audited composed-residual recipe.

Only the implemented unpolarized LDA-PZ81, neutral-valence-plus-NLCC and cS conventions
are accepted; arbitrary XC or zero-point labels are rejected. Pair-XC providers without
a manifest hash bind their compiled contents once at construction. The actual atomic
M2 values are always included. Injected providers and their tensors must remain immutable
for the plan lifetime; construct new providers and a new plan after changing tables.
Prepared geometries own copies of caller arrays, so later caller edits cannot mix new
onsite coordinates with old edge plans. Onsite grid cache mutations use the separate
`invalidate_grid_radius(quadrature)` contract for inference tensors or NumPy/.data edits.
