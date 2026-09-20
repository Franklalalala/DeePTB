# `dptb.nacf` module map

One authoritative implementation per role. Inference never compiles; offline builders and
benchmarks are not imported by `dptb.nacf.__init__`.

## Geometry-only inference core

- `assembly.py`: `NACFTableBank` (P2/P23/overlap tables on one device), `NACFAssemblyPlan`, batch plan and
  `NACFFeaturePlan` (AO blocks to checkpoint RME features). `assembly_topology.py`: native preparation of the
  P23-onsite/P2-edge recipe. `topology.py`: ctypes loader of the vendored Tonari cell-list core
  (`csrc/topology.cpp`, integer-image queries, onsite neighbour rows).
- `radial.py`: `TorchRadialBlockTable`, device-resident spline plus real-harmonic rotation of one table.
  `prepared.py` / `prepared_store.py`: checksummed compact radial snapshots (see `docs/nacf_prepared_store.md`).
- `_cuda.py`, `precompiled.py`: load the verified CUDA binary; `fusion.py`: opt-in fused radial/contraction
  plans (`FUSION.md`). `density.py`: native density topology and the Torch probe reference.
- `overlap.py`, `soc.py`, `spinor_inference.py`, `spinor_completion.py`, `soc_cpu_reference.py`: overlap sidecar,
  spinor projectors and full SOC completion. `inference.py`, `cli.py`: `NACFGeometryPredictor` and the
  geometry-only command line of trained checkpoints (P23/P2 prior; untouched by the candidate entry).

## Full candidate prior (`docs/nacf_candidate_prior.md`)

- `candidate_policy.py`: declarations (frozen `CandidateRecipe`, canonical XC key, `FusionSettings`, frozen
  order policies). `candidate_checks.py`: raw structure validation and provider declaration checks (NumPy only).
  `candidate.py`: providers, `CandidatePriorPlan` binding/identity, `PreparedCandidate` numerics; public entry.
- Engines composed by the plan: `edge_vna.py` (grid-free third-centre VNA), `envxc.py` (D2 / D2+moment /
  McWEDA environment XC, `v1`/`v2` stabilization), `onsite.py` (fused or reference onsite XC on the accepted
  local quadrature, packed density bank with content identity, native or NumPy 27 Bohr neighbourhoods).

## Offline builders and diagnostics

- `envxc_tables.py`: environment-XC table builder (schema nacf-envxc/v1). `build_overlap.py`, `build_soc.py`:
  S and SOC sidecars from the exact bound ORB/UPF files. `precompile.py`: explicit CUDA build and `--check`.
- `benchmark.py`, `production_benchmark.py`, `soc_benchmark.py`, `memory_probe.py`: opt-in timing and memory
  studies. `numerical_identity.py`: strict dependency identities for table receipts.
- `THIRD_PARTY.md`: vendored Tonari licence and provenance.
