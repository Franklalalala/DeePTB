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

## Units

Nothing in either module converts units. `result.units` is a **declarative
label**: it records the eV/e convention DeePTB feeds this reference (charges in
`e`, energies in `eV`, electronegativity in `eV/e`, hardness kernel in
`eV/e^2`), and relabeling it without a numerical conversion is rejected — but
the numbers are never checked against it. Feed Hartree-valued `chi` and `J` and
you get Hartree-valued energies still labelled `eV`. Convert at the boundary if
you need eV.

The same holds for fixed-mu: `mu`, `kT` and `h` must share one caller-chosen
energy unit, and every energy that comes back — eigenvalues, the whole band
ledger, `Tr(D H)` — is in that unit. There is no unit enum and no conversion
machinery.

## Solve Semantics

The solve is performed **on the sum-zero tangent space**. With `Z` an
orthonormal basis of `{v : sum(v) = 0}`, the physical problem is
`(Z.T J Z) c = -Z.T (chi + J q0)` for `q = q0 + Z c`, `q0 = (Q/n) 1`. The
Lagrange multiplier is then recovered as `lambda = -mean(chi + J q)`.

This matters because `J -> J + alpha * 11.T` is a gauge freedom: it leaves the
fixed-charge minimizer completely unchanged (`Z.T 11.T Z == 0`) while driving
the condition number of the raw augmented KKT matrix `[[J, 1], [1.T, 0]]` up as
`alpha^2`. The raw KKT condition number is therefore reported as
**telemetry only** (`diagnostics.kkt_condition`); it is not gated. Safety is
gated on the scale-invariant tangent spectrum instead:
`constrained_min_eig`, `constrained_max_eig`, `constrained_condition`.

The solver still fails closed for non-finite, complex, non-square,
non-symmetric, zero-site, zero-length-batch, singular, ill-conditioned, or
non-convex constrained kernels, and for torch tensors at every boundary
(including the result and validator boundaries — nothing is silently
`detach().cpu()`-ed).

## Diagnostics

| field | meaning |
| --- | --- |
| `charge_residual` | `sum(q) - Q` |
| `stationarity_max_abs`, `stationarity_l2` | norms of the raw KKT vector `chi + J q + lambda 1` (gauge-dependent) |
| `stationarity_tangent_max_abs` | `max abs Z.T (chi + J q)` — the gauge-invariant equation the solve enforces |
| `multiplier_residual` | `mean(chi + J q) + lambda` — the uniform component that defines `lambda` |
| `energy_identity_residual` | derived telemetry, see below |
| `input_kernel_symmetry_error` | asymmetry of `J` **as supplied**, measured before symmetrization |
| `kernel_symmetry_error` | asymmetry of the stored, already-symmetrized `J`; identically zero for any accepted input |
| `constrained_min_eig`, `constrained_max_eig`, `constrained_condition` | tangent spectrum of `Z.T J Z` |
| `kkt_condition` | raw augmented-KKT condition number, telemetry only |

`energy_identity_residual` is `energy - energy_identity` with
`energy_identity = 0.5 chi.q - 0.5 lambda Q`. This is algebraically
`0.5 q.stationarity - 0.5 lambda (sum(q) - Q)`, i.e. it is implied by the charge
and stationarity residuals and is **not** an independent certificate of the
energy. It carries no terminal gate of its own; it is recorded and checked
against its own recomputation.

## Safety Policy

`QEqResult` persists the policy of the solve that produced it —
`symmetry_tol`, `constrained_eig_floor`, `max_condition`, `residual_tol` — so a
downstream consumer can re-enforce it.

That policy is *self-declared data*, so it is clamped against module-level
ceilings that a payload can only tighten, never loosen:

```python
qeq_operator.MODULE_MAX_CONDITION    # 1e12
qeq_operator.MODULE_MAX_RESIDUAL_TOL # 1e-6
qeq_operator.MODULE_MAX_SYMMETRY_TOL # 1e-6
```

`solve_qeq` refuses arguments above the same ceilings, so a policy that solves
is always a policy the validator will also certify. A caller who does not trust
a payload's self-declaration can override it:

```python
validate_qeq_result(
    untrusted,
    expected_max_condition=1e6,
    expected_constrained_eig_floor=1e-9,
    expected_symmetry_tol=1e-12,
)
```

## Validator Semantics

QEq and fixed-mu result dataclasses own defensive read-only copies of their
array fields. A frozen dataclass alone does not protect nested NumPy arrays, so
in-place mutation of returned arrays is intentionally rejected — and the
read-only flag is re-applied in `__setstate__`, so it survives a pickle round
trip.

`validate_qeq_result` does not trust cached diagnostics. It recomputes charge
conservation, KKT stationarity, total/linear/quadratic/identity energies,
constrained-kernel diagnostics, and KKT conditioning from the current
`QEqResult` arrays, then compares those recomputed values against the stored
fields and diagnostics before applying the physical residual tolerances.

`atol` defaults to `None`, meaning "adopt the `residual_tol` this result
recorded" — a recorded policy is re-enforced as recorded, whether it is looser
or tighter than the shipped default. An explicit `atol` always wins.

Residual gates are absolute in the caller's energy unit **plus an arithmetic
floor**: no float64 evaluation of `chi + J q + lambda` can resolve below the
rounding scale of its own intermediates, so each gate admits
`atol + 64 * eps * (max|chi| + max|J| * sum|q| + |lambda|)`. This is what makes
the gates invariant under the `alpha * 11.T` gauge — a shift of `alpha = 1e9`
leaves the charges and every gate verdict intact — while a forged charge vector
is still rejected by many orders of magnitude. Magnitude-carrying quantities
(condition numbers, eigenvalues, energies) are compared to their recomputation
with a relative tolerance, because one ULP of a condition number at the `1e12`
ceiling is `1.2e-4`, far above any absolute `atol`.

When the stationarity residual does fail and the tangent condition number is
large enough to explain it — roughly `eps * cond * scale >= tolerance` — the
error raised is `QEqKernelConditionError` naming the attainable residual, not
the generic contract-violation `QEqOperatorError`.

`input_kernel_symmetry_error` cannot be recomputed, because the stored kernel is
already symmetrized. The validator re-enforces the recorded `symmetry_tol`
against it instead.

## Fixed-mu Validator

`validate_conservation` for fixed-mu results is intentionally separate from
QEq. Solver-produced `FixedMuResult` objects carry a non-field, non-serialized
validation context with immutable Hamiltonian and overlap snapshots. The
context is excluded from `dataclasses.asdict`, `FixedMuResult.to_dict()`, and
pickle state, so deserialized or externally constructed results must call
`validate_conservation(result, h=..., s=...)` explicitly. The result stores the
scalar provenance needed to rerun validation, including `normalize_k_weights`,
`eig_floor`, `max_condition`, and `hermitian_tol`.

The validator recomputes k-axis canonicalization, finite/nonnegative
stored-weight checks, normalized-weight sums when normalization provenance is
enabled, H/S Hermiticity, S positive-definiteness and condition, `C^H S C = I`,
`H C = S C eps`, Fermi occupations from eigenvalues, density aggregation from
the eigenvectors, the band energy ledger and its closures, `Tr(D S)`,
`Tr(D H)`, residual ledgers, and density hermiticity from the current fixed-mu
arrays.

Before any of that it requires the eigenbasis to be **complete**:
`eigvecs.shape[-2:] == (n, n)` and `eigvals.shape[-1] == n` against the H/S
dimension `n`. Without that check the `C^H S C = I` identity is sized from the
stored eigenvector column count, and a result carrying only `m < n` eigenpairs
certifies itself while `N(mu)` and `D(mu)` are silently wrong.

The energy ledger names are explicit about what the terms are — they are
single-particle band-structure quantities, with no double-counting correction
and no ion-ion term:

| field | definition |
| --- | --- |
| `band_energy` | `g * sum_k w_k * sum_i eps_i f_i` |
| `minus_t_s` | `-T*S`, i.e. `g * sum_k w_k * kT * [f log f + (1-f) log(1-f)]`, always `<= 0` |
| `band_free_energy` | `band_energy + minus_t_s` |
| `band_grand_energy` | `band_free_energy - mu * N` |

`ConservationLedger` also records `input_h_hermiticity_error` /
`input_s_hermiticity_error`, the raw `max |X - X^H|` of the matrices as
supplied. Like the QEq kernel asymmetry these are measured before
symmetrization and cannot be recomputed from the stored snapshot; the validator
re-enforces the recorded `hermitian_tol` against them.

Two policies are conditioning-aware. The acceptance gates admit `cond(S)` up to
`max_condition = 1e12`, but the `C^H S C = I` and `H C = S C eps` residuals of a
backward-stable generalized solve grow like `eps * cond(S)` — measured on a
`6x6` stack, `~0.09 * eps * cond` and `~0.26 * eps * cond * |H|`. Those checks,
and the `Tr(D S)` / `Tr(D H)` gates that route through the same eigenvectors,
therefore use `max(atol, 16 * eps * cond(S) * scale)`. At the `cond(S) ~ 1e2` of
realistic NAO overlaps that floor is `~1e-12` and the caller's `atol` is
unchanged; at `cond(S) = 1e10` it is what lets an untampered result validate
against itself at all.

Finally, every conservation identity is homogeneous of degree one in
`spin_degeneracy`, so a result computed at the wrong SOC/non-SOC convention
validates perfectly against itself. `spin_degeneracy` is constrained to `{1, 2}`
at construction, and the caller can bind a result to the request that was
actually made:

```python
validate_conservation(
    result,
    expected_mu=mu,
    expected_kT=kT,
    expected_spin_degeneracy=2.0,
    expected_k_weights=weights,
)
```

These checks are deliberately dense-reference checks over the returned arrays
and validation H/S; the validator is for fail-closed postprocess validation, not
a cheap training-loop assertion.
