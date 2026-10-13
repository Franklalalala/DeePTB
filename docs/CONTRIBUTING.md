# Contributing to DeePTB

Use `1013-stable` for maintenance changes to this repository. Read
[AGENTS.md](../AGENTS.md), [the architecture guide](architecture.md), and
[TESTING.md](../TESTING.md) before changing interfaces or numerical code.

## Issues and discussions

Search [existing issues](https://github.com/Franklalalala/DeePTB/issues) before
reporting a bug. Include a minimal reproducer, input/configuration details, the
software environment, and expected versus observed behavior. General upstream
discussions are available in [deepmodeling/DeePTB](https://github.com/deepmodeling/DeePTB/discussions).

## Implementation and documentation

Keep one authoritative implementation per component and use shared construction
interfaces. Preserve public contracts and checkpoint parameter paths unless a
change includes an explicit migration strategy. Describe representations, units,
targets, and prior reconstruction semantics for scientific behavior changes.

Use English for documentation, comments, and commit messages. Prefer comments
explaining mathematical choices or non-obvious constraints. Follow
[PEP 8](https://www.python.org/dev/peps/pep-0008/) and the surrounding code style.
Keep generated data, checkpoints, run logs, and one-off scripts outside the
repository.

## Tests

Select behavioral tests using [TESTING.md](../TESTING.md). Add a minimal
regression case for a reproducible bug and parameterize related boundaries.
Shared fixtures belong in helper modules; avoid importing test modules from
one another.

~~~bash
python tools/test.py
python tools/test.py dptb/tests/test_record_codec.py
~~~

Run the full suite, affected optional-backend checks, and numerical checkpoint
comparisons when the change requires them. Documentation changes do not need
numerical reruns.

## Pull requests

1. Create a branch from `1013-stable` in this repository or a fork.
2. Make the change with current documentation and relevant behavioral tests.
3. Run required validation and describe its scope and results.
4. Push the branch and open a PR targeting `Franklalalala/DeePTB:1013-stable`.

Use a draft PR while implementation or required validation is incomplete.
Explain the concrete problem, resulting behavior, compatibility implications,
and validation. Keep operational logs and internal deployment details out of
public PRs. Preserve historical branches and immutable deployments.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org) with a concise
English description, for example:

~~~text
fix(data): reject misaligned prior sidecars

Validate graph ordering before using a prior sidecar.

Fixes #123
~~~

Common types include `feat`, `fix`, `docs`, `refactor`, `perf`, `test`,
`build`, `ci`, and `revert`. Add a scope when it clarifies the affected module.
The body may explain motivation or compatibility; the footer may reference
related issues.
