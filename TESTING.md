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
