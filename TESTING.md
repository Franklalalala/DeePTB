# Testing

Use a configured DeePTB environment. For a code change, run the behavioral tests
for the affected module. One suitable check is enough; there is no sequence of
mandatory per-feature gates.

```bash
python tools/test.py dptb/tests/test_record_codec.py
```

With no arguments, `python tools/test.py` runs a short smoke suite for charge
conservation, spectral losses, record decoding, prior routing, and ensemble
metadata. `bash ut.sh` uses the same entry point. This is a quick signal, not
coverage of every DeePTB feature.

For a deliberately broad review, request the full suite explicitly:

```bash
python tools/test.py dptb/tests
```

## Layout and optional components

`dptb/tests` groups tests by behaviour, one file per module family. Builders shared by
several files live in helper modules that tests import (`block_ode_fixtures`,
`pair_helpers`, `flow_helpers`, `model_helpers`, `nacf_support`, `p2_support`, `_trainer_probes`); a test
module never imports another test module. `h0/tests_h0fast` runs on its own:
`python -m pytest h0/tests_h0fast`.

A test that needs an optional component skips with a reason when it is missing
(`dptb/tests/_requires.py`): `requires_cuda`, `requires_multi_gpu`, `requires_so2_cuda`
(the external `so2_cuda_ops` package), `requires_module(name)` (e.g. `dftio`), and the
NACF native markers `requires_nacf_prebuilt` / `requires_nacf_topology`, resolved lazily in
`conftest.py`. Tests on real reference data are opt-in through the environment variable
the test names. Run the GPU paths on a machine that has them, e.g. with SO2CUDA on the
path:

```bash
PYTHONPATH=$PWD:/path/to/SO2CUDA/src python -m pytest dptb/tests/test_so2_kernels_cuda.py
```

`conftest.py` fails a module that changes torch's default dtype at import or leaks it, or
that leaves `torch.use_deterministic_algorithms` switched on, and sets
`CUBLAS_WORKSPACE_CONFIG` before CUDA starts.

Keep tests that establish observable facts: numerical results against a small
independent reference, data/graph alignment, valid checkpoint restoration,
finite and correctly routed gradients, or rejection of corrupt input. A
reproduced bug usually needs one minimal regression case. Prefer parameterized
boundary cases to copied tests.

Do not test source-code spelling, comments, private call order, a frozen copy
of the implementation, or exact incidental error wording. Keep configuration
tests when they exercise actual accepted/rejected behavior. Hardware benchmarks
and real-checkpoint studies are opt-in experiments with their own assets, not
routine development gates. Documentation-only changes need no numerical rerun.

Run a new GPU/real-data check when a changed kernel, device path, data contract,
or unresolved numerical issue requires it. Reuse prior evidence when its
relevant code, data, and environment are unchanged. Code validation and claims
about held-out prediction quality require different evidence.

Commit reusable code, feature documentation and focused behavioral regressions.
Keep job manifests, run logs, benchmark snapshots, copied source trees and
one-off diagnostic outputs outside the repository. Summarize validation in
the change description; do not upload an evidence directory for each run.
