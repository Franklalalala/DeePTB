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
