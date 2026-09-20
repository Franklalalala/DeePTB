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
