# Testing

Use a configured DeePTB environment and select tests for the behavior affected by your change. One effective targeted check is sufficient for an isolated change; documentation edits do not require numerical experiments.

```bash
python tools/test.py dptb/tests/test_record_codec.py
```

`python tools/test.py` and `bash ut.sh` run the same short suite, covering public APIs, embedding discovery and compatibility, spectral losses, record decoding, prior-aware models, and expert masks. This suite checks installation and common interfaces. It does not cover every feature.

Request the full suites explicitly when needed:

```bash
python tools/test.py dptb/tests
python -m pytest dptb/tests/h0
```

## Test layout and optional components

`dptb/tests` is organized by module family and behavior; H0 tests are in `dptb/tests/h0`. Shared constructors belong in helper modules; test files must not import one another. LoopSCF tests run in its own repository against installed DeePTB.

Tests requiring missing optional dependencies skip with an explanation. `dptb/tests/_requires.py` provides `requires_cuda`, `requires_multi_gpu`, `requires_so2_cuda`, and `requires_module(name)`; `conftest.py` checks native NACF dependencies lazily. Each real-reference-data test documents the environment variables that enable it. Keep data and generated results outside the repository.

With SO2CUDA 0.3.0 installed, run the affected CUDA tests, for example:

```bash
python -m pytest dptb/tests/test_so2_kernels_cuda.py
```

When SO2CUDA is unavailable, validate the corresponding PyTorch reference path. `SO2_CUDA_BACKEND=off` explicitly disables acceleration. Record the backend selection, device, precision, and actual execution route; a configuration flag alone does not establish that CUDA kernels ran.

`conftest.py` checks the default dtype and deterministic settings after test import and execution, and sets `CUBLAS_WORKSPACE_CONFIG` before CUDA initialization. Configure CPU parallelism and GPU resources explicitly in the execution environment.

## Validation principles

Keep tests of observable behavior: small independent numerical references, data/graph alignment, checkpoint restoration, finite and correctly routed gradients, and rejection of corrupt inputs. A bug usually needs one minimal regression case; parameterize related boundary cases.

Avoid assertions on source spelling, comments, private call order, frozen implementation copies, or incidental error wording. Configuration tests should exercise actual acceptance or rejection.

For UniTB and UniTB-dense refactors, use identical inputs, checkpoints, strict FP32 settings, and deterministic execution. Compare inference outputs and the losses and parameters from four training steps. Report difference magnitudes, environment details, and checkpoint-loading results. Passing the full suite does not establish numerical equivalence.

New GPU or real-data checks are needed for affected kernels, device paths, data contracts, or unresolved numerical issues. Hardware benchmarks and predictive-accuracy studies use separate assets. Routine maintenance does not require repeated seeds or duplicate training with the same configuration.

Commit reusable code, current feature documentation, and targeted behavioral regression tests. Keep run logs, job inventories, benchmark snapshots, source copies, and one-off diagnostics outside the repository. Describe validation scope and results in the change summary.

Use strict FP32 for numerical comparisons: set `NVIDIA_TF32_OVERRIDE=0`, `SO2_CUDA_FAST_TF32=0`, and `DPTB_CUBLAS_GROUPED_FAST_TF32=0`, and disable PyTorch matmul/cuDNN TF32. CPU and CUDA can select different active edges near cutoff boundaries, so report same-device checkpoint regressions separately from cross-device comparisons. A runnable case without an old-device reference establishes execution only.

Install this branch before running the standalone LoopSCF package tests and checkpoint regressions in the LoopSCF repository. For performance checks, use the UniTB/UniTB-dense examples provided by SO2CUDA. Record the actual backend, warm-up and timed iteration counts, forward/backward medians and quartiles, device memory, and numerical differences. Synthetic structures with random models do not establish physical accuracy.
