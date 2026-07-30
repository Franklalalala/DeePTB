# Fixed-μ Electrostatic SCF Reference

`dptb.nnops.fixed_mu_scf_operator` provides a small NumPy/SciPy CPU reference
for adding a molecular electrostatic response loop around the dense
`fixed_mu_observables` operator.

## Approximation level

This operator is **frozen-H⁰ + self-consistent Mulliken electrostatics**. It
captures only the electrostatic response through an overlap-consistent onsite
potential. It does not include occupation-dependent exchange-correlation
response or orbital rehybridization. Its approximation level lies above
frozen-H fixed-μ postprocessing and below grand-canonical DFT (GC-DFT).

The network prediction `H0` is never retrained or reevaluated. The SCF loop
only adds the charge-dependent electrostatic correction described below. This
is a reference operator for finite clusters and molecules, not a total-energy
method, force model, molecular-dynamics engine, or constant-potential DFT
implementation.

## Physical definition

For one dense Hamiltonian/overlap pair, each iteration evaluates

```text
D = fixed_mu_observables(H, S, mu, kT, ...).density

p_i = sum_{mu in atom i} Re[(D S)_{mu,mu}]
q_i = reference_populations_i - p_i
phi = coulomb_kernel @ q

H[m,n] = H0[m,n]
         - 0.5 * S[m,n] * (phi[atom(m)] + phi[atom(n)])
```

The caller supplies:

- the AO-to-atom assignment;
- the neutral/reference Mulliken populations;
- a finite, real, symmetric `n_atom × n_atom` Coulomb kernel;
- one consistent unit convention for `H0`, `mu`, `kT`, charge, and the kernel.

The module performs no unit conversion and records no vacuum, work-function,
SHE, or Li/Li⁺ reference. A numerically identical `mu` is not physically
comparable across structures unless the caller has already aligned their
energy references.

The overlap-weighted coupling is intentional. For a spatially uniform
potential `phi_i=c`,

```text
H = H0 - c S.
```

This is exactly equivalent to `mu -> mu-c`, so it preserves the
`H+cS, mu+c` gauge covariance of `fixed_mu_observables`. When `S=I`, the
correction reduces to a pure onsite diagonal shift.

## API

```python
import numpy as np

from dptb.nnops import fixed_mu_electrostatic_scf

h0 = np.array([[-0.3, 0.05], [0.05, 0.2]])
s = np.array([[1.0, 0.1], [0.1, 1.0]])

result = fixed_mu_electrostatic_scf(
    h0,
    s,
    mu=0.0,
    kT=0.025,
    ao_atom_index=np.array([0, 1]),
    reference_populations=np.array([1.0, 1.0]),
    coulomb_kernel=np.array([[0.5, 0.2], [0.2, 0.5]]),
    mixing="pdiis",
    mixing_step=0.2,
    n_history=6,
    mixing_period=3,
    max_iter=100,
    charge_tol=1e-8,
)

print(result.q)
print(result.phi)
print(result.fixed_mu_result.electron_count)
```

The public entry point is:

```text
fixed_mu_electrostatic_scf(
    h0, s, *,
    mu,
    kT=0.0,
    ao_atom_index,
    reference_populations,
    coulomb_kernel,
    mixing="pdiis",
    mixing_step=0.2,
    n_history=6,
    mixing_period=3,
    max_iter=100,
    charge_tol=1e-8,
    divergence_tol=1e6,
    spin_degeneracy=2.0,
    k_weights=None,
    k_axis=None,
    normalize_k_weights=True,
    eig_floor=1e-10,
    max_condition=1e12,
    hermitian_tol=1e-8,
)
```

`spin_degeneracy`, `normalize_k_weights`, `eig_floor`, `max_condition`, and
`hermitian_tol` are passed through to `fixed_mu_observables`. v1 accepts
`k_weights` and `k_axis` only as `None`; the arguments are present so an
unsupported k-point request fails with an explicit v1 error instead of being
silently interpreted as a molecule.

## Result contract

`FixedMuSCFResult` stores:

- `q`: terminal Mulliken net charges;
- `phi`: `coulomb_kernel @ q`;
- `iterations` and the maximum-norm `residual_history`;
- `fixed_mu_result`: the terminal `FixedMuResult`;
- every SCF strategy field (`mixing`, step, history, period, iteration and
  tolerance policies);
- every forwarded fixed-μ strategy field.

All arrays are defensive copies backed by immutable byte buffers. Their write
flag cannot be re-enabled, and a pickle round trip restores the same read-only
guarantee. The nested `FixedMuResult` follows its own serialization contract:
after deserialization its private H/S validation context is absent, so a later
standalone call to `validate_conservation` must again receive explicit `h=` and
`s=`.

## Convergence and mixing

The fixed-point residual is the unmixed charge difference

```text
r_n = F(q_n) - q_n
residual_n = max(abs(r_n)).
```

Convergence requires `residual_n <= charge_tol`. The terminal charge reported
in `result.q` is measured from the terminal density matrix. The Hamiltonian
used for that terminal diagonalization was formed from the preceding SCF input
charge; their maximum difference is the final recorded residual and is
therefore bounded by `charge_tol`.

`mixing="linear"` applies

```text
q_next = q + mixing_step * r.
```

`mixing="pdiis"` uses the same linear step between Pulay iterations. Every
`mixing_period` iterations it applies the periodic Pulay correction built from
up to `n_history` residual-difference and charge-step vectors. This is the
NumPy port of the historical DeePTB `dptb/negf/scf_method.py` PDIIS structure.
Only `q` is mixed; neither `H`, `D`, nor `phi` has an independent mixing
history.

After convergence, the operator evaluates the terminal `FixedMuResult` and
runs `validate_conservation` with explicit `h=H_final` and `s=S`, binding the
chemical potential, temperature, spin convention, overlap policy, and
Hermiticity policy.

## Fail-closed behavior

`FixedMuSCFError` rejects:

- k-stacked or leading-batch H/S arrays, and any nonempty k-weight request;
- non-square, non-finite, non-Hermitian H/S or unsafe overlap matrices;
- AO assignments with the wrong length, non-integer values, or atom indices
  outside the reference-population range;
- empty, non-real, or non-finite reference populations;
- a Coulomb kernel with the wrong shape, complex values, NaN/Inf, or
  asymmetry;
- unknown mixing policies or invalid step/history/period/iteration/tolerance
  values.

`FixedMuSCFConvergenceError` is raised immediately for a non-finite iterate or
a residual above `divergence_tol`, and after `max_iter` if the fixed point has
not converged. It carries:

```text
error.iterations
error.residual_history
```

The diagnostic history is also included in the exception message and is a
bytes-backed read-only array. There is no best-effort unconverged result.

## Scope relative to neighboring methods

- **DP-QEq** (Hu et al., *Nature Communications* 2025, 16, 7379) supplies a
  fixed-total-charge QEq/long-range electrostatic route that can be integrated
  with short-range energies, forces, periodic electrostatics, and
  constant-potential MD; this operator instead closes a finite-cluster
  AO-Mulliken charge loop at fixed electronic `mu`.
- **DP-Ne** (Sun et al., *Nature Communications* 2025, 16, 3600) learns
  `E(R, N_e)` and supports grand-canonical electron-number sampling; this
  operator has neither a many-body potential-energy surface nor electron-number
  Monte Carlo.
- **DeePTB-NEGF** (Zou et al., *npj Computational Materials* 2025, 11, 375)
  couples predicted H/S to open-boundary Green functions and a grid
  NEGF-Poisson loop; this operator borrows the `charge -> potential -> H ->`
  re-solve idea but has no leads, self-energies, transport, grid Poisson
  solver, or electrode boundary conditions.

Periodic Ewald/PME, slab corrections, externally calibrated electrode
potentials, total-energy double-counting corrections, forces, and
occupation-dependent XC/orbital response are intentionally deferred.
