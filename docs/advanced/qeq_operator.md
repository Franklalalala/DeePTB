# Dense QEq Operator

`dptb.nnops.qeq_operator` provides a small NumPy reference implementation of
the dense charge-equilibration problem

```text
E(q) = chi.T q + 0.5 q.T J q,    sum(q) = total_charge.
```

The API is intentionally independent from `fixed_mu_operator`. QEq solves a
fixed-total-charge finite-site charge vector; fixed-mu postprocessing solves
Fermi occupations and density matrices from dense Hamiltonian/overlap matrices.

```python
from dptb.nnops.qeq_operator import solve_qeq

result = solve_qeq(
    electronegativity=[1.0, 3.0],
    hardness_kernel=[[5.0, 1.0], [1.0, 7.0]],
    total_charge=0.0,
)
print(result.charges)
```

The units are explicit in `result.units`: charges are `e`, energies are `eV`,
electronegativity is `eV/e`, and the hardness kernel is `eV/e^2`.
These are canonical, unscaled units; relabeling them without numerical
conversion is rejected.

The solver fails closed for non-finite, complex, non-square, non-symmetric,
singular, ill-conditioned, or non-convex constrained kernels. Returned
diagnostics include the fixed-charge residual, KKT stationarity residual,
energy-identity residual, constrained-kernel eigenvalues, constrained condition
number, and KKT condition number.

## Validator Semantics

QEq and fixed-mu result dataclasses own defensive read-only copies of their
array fields.  A frozen dataclass alone does not protect nested NumPy arrays, so
in-place mutation of returned arrays is intentionally rejected.

`validate_qeq_result` does not trust cached diagnostics.  It recomputes charge
conservation, KKT stationarity, total/linear/quadratic/identity energies,
constrained-kernel diagnostics, and KKT conditioning from the current
`QEqResult` arrays, then compares those recomputed values against the stored
fields and diagnostics before applying the physical residual tolerances.

`validate_conservation` for fixed-mu results is intentionally separate from
QEq.  Solver-produced `FixedMuResult` objects carry private read-only
Hamiltonian and overlap snapshots so the validator can recompute density
aggregation, Fermi occupations from eigenvalues, all band/entropy/free/grand
energy ledger terms, `free_energy = band_energy + entropy_term`,
`grand_energy = free_energy - mu * N`, `Tr(D S)`, `Tr(D H)`, residual ledgers,
and density hermiticity from the current fixed-mu arrays.  These checks are
deliberately dense-reference checks over the returned arrays and snapshots; the
validator is for fail-closed postprocess validation, not a cheap training-loop
assertion.  Externally constructed fixed-mu results that lack those snapshots
fail closed because their ledgers cannot be checked against the original H/S
matrices.
