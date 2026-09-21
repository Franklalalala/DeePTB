# 0921-stable

This branch continues the preserved `0920-stable` snapshot. It finishes the
candidate interface repairs and organizes its existing code without changing the
NACF operator formulas or the prior used by existing checkpoints.

The changes reject fractional cell shifts before conversion, check the actual AO
shell sequences of pair-XC tables, and refresh packed density caches when tensor
views are replaced. Candidate recipe settings and onsite XC declarations are
bound explicitly; selecting a fixed quadrature no longer reports convergence.

The historical P23-onsite/P2-hopping entry, explicit full candidate, McWEDA v1/v2,
prepared radial stores, Tonari topology, and CUDA kernels remain available. Use
`docs/nacf_candidate_prior.md` for the candidate API and its module map. Reusable
implementation stays under `dptb/nacf`; generated tables, binaries and run logs
stay outside Git.

## Build and test

Use the configured DeePTB environment, with its `bin` directory and the CUDA
toolkit `bin` directory on `PATH`. Set `CUDA_HOME` and `TORCH_EXTENSIONS_DIR`
before building. The interrupted 0920 release build had failed because Ninja was
absent from the launched process's `PATH`; Ninja was already installed.

```bash
python -m dptb.nacf.precompile --arch 8.9+PTX
python tools/build_nacf_topology.py --output /path/libnacf_topology.so
export DPTB_NACF_TOPOLOGY_LIBRARY=/path/libnacf_topology.so
python tools/test.py dptb/tests/test_nacf_candidate.py dptb/tests/test_nacf_onsite_fused.py
```

On RTX PRO 6000 with PyTorch 2.8.0 / CUDA 12.8, the CUDA and native topology builds
succeeded. The four focused candidate, onsite, grid-cache and stabilization test
modules passed all 37 tests. Support/knot boundaries, 16-neighbour segments and
non-default CUDA streams matched the scalar reference exactly; the requested
architecture mismatch correctly returned a nonzero exit code.

Five existing geometries (4–49 atoms, 172–5,766 directed edges) passed comparisons
between `CandidatePriorPlan` and the previous manual assembly, including each
component and final RME packing. Maximum component difference was 5.69e-14 eV;
maximum packed difference was 2.85e-14 eV. The check used v2 on all five and also
v1 on the Ca/Sr case, comparing within each recipe. Fixed orders were 128/24/48.

These are implementation regression results. They do not establish new physical
accuracy, quadrature convergence, full production coverage or an end-to-end
speedup. See `0920-stable.md` for the separately recorded historical benchmarks.

The SOC50 native-topology/radial-fusion speed result applies to the explicitly
configured table-bank assembly benchmark. The public geometry predictors share
`prepare_geometry`, which currently calls `bank.prepare` without selecting
native topology or radial fusion. Their default path therefore does not inherit
that measured speedup. Check the actual preparation settings before quoting a
predictor timing; the checkpoint's prior recipe remains unchanged.

## Provider boundary follow-up

Candidate binding now rejects precompiled pair-XC buffers on a different device
or with a different floating dtype. After selecting onsite orders, preparation
checks each species' exact P2 AO count and its quadrature point alignment before
global AO padding can hide a mismatched basis. Raw graph range checks compare
Python scalars so float32 cannot round the integer upper bound into an accepted
out-of-range value. These guards do not change the candidate formula or add the
optional onsite neighbor cache.

The focused candidate, provider-contract, onsite, grid-cache and H0 SOC-reference
tests passed 53 cases on the configured RTX PRO 6000 environment, with one
inapplicable negative-uint64 case skipped. The original guard tests reproduced
7 failures before the fix. Twelve synthetic geometry/recipe combinations yielded
120 output arrays identical bit for bit before and after the guard changes.
Two existing real geometries also passed CUDA candidate/manual composition and
RME packing: maximum component difference 5.69e-14 eV and packed difference
7.11e-15 eV. These are implementation checks, with fixed quadrature orders.

The H0 SOC helper bounds the all-below-threshold fallback by the available mesh.
The reference's nonzero projector rule is retained and linked to the fixed
ABACUS source. See [the H0 reference documentation](../h0/README.md#explicit-soc-reference-alignment)
for the separate spacing/projector ablation and the retained cutoff-radius
semantics; the combined single-case gain must not be attributed to both changes
individually.

## Local-grid batch follow-up

`CudaPeriodicFFTGridAOCache.contract_pairs_batch` now passes an empty spin-z
anchor list when no spin-z field is supplied, as required by the native API.
Previously a nonempty batch failed with an anchor-list length error. The main
`assemble_h0` path uses `contract_chunk` and does not call this batch wrapper.

On an L40S with PyTorch 2.8.0 / CUDA 12.8, the new non-spin regression failed
before the fix; after it, all eight focused batch/chunk cases passed. Batch
blocks with and without a spin-z field matched independent CPU contractions,
including periodic images and empty inputs. The current wrapper and unchanged
native sources were checked against the verified installed binary manifest.
This validates the interface repair, not full-cohort H0 precision.
