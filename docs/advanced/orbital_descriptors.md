# CIET orbital descriptors from dense H/S/D

`dptb.nnops.orbital_descriptors` is a fail-closed NumPy reference module for
orbital descriptors that can be formed from a frozen dense single-particle
Hamiltonian, overlap, eigensystem, and density matrix. It uses exactly the
generalized-eigenvector convention of `fixed_mu_operator`:

\[
HC=SC\varepsilon,\qquad C^\dagger SC=I,\qquad
D=C\,\mathrm{diag}(f)\,C^\dagger .
\]

All energies share the caller's energy unit. The code does not infer or convert
units.

## Public entry points

- `mulliken_lowdin_populations`: atom and optional `(atom,l)` populations.
- `fragment_pdos`: state-resolved fragment weights and spectral band centers.
- `bond_energy_partition`: once-counted atom-pair one-electron energy terms.
- `fragment_charge_response`: population projection of a supplied `dD/dmu`.
- `orbital_descriptors`: combined evaluation with a consistency check between
  an optional supplied `D` and the density rebuilt from `C/f`.
- `fixed_mu_scan_fragment_populations`: population and response curves from a
  `FixedMuScanResult` without rediagonalization.

Every returned array is a defensive, bytes-backed read-only copy. Pickle
restoration freezes arrays again.

## Definitions and conventions

### Mulliken and Löwdin populations

For AO indices belonging to atom \(A\),

\[
p_A^\mathrm{M}=\sum_{\mu\in A}\mathrm{Re}\,(DS)_{\mu\mu}.
\]

The symmetric-Löwdin population uses the unique positive spectral square root:

\[
p_A^\mathrm{L}=\sum_{\mu\in A}
\left(S^{1/2}DS^{1/2}\right)_{\mu\mu}.
\]

Both partitions close to
\(\sum_Ap_A=\mathrm{Re}\,\mathrm{Tr}(DS)\). If `ao_l_index` is present, the
same AO contributions are additionally reduced by sorted `(atom,l)` keys.
These definitions follow Mulliken's population analysis
([J. Chem. Phys. 23, 1833 (1955)](https://doi.org/10.1063/1.1740588)) and
Löwdin symmetric orthogonalization
([J. Chem. Phys. 18, 365 (1950)](https://doi.org/10.1063/1.1747632)).

### Fragment pDOS weights and band centers

For state \(i\) and fragment \(F\),

\[
w_{iF}=\mathrm{Re}\sum_{\mu\in F,\nu}
C^*_{\mu i}S_{\mu\nu}C_{\nu i}.
\]

The atom mapping must cover every AO, hence
\(\sum_F w_{iF}=1\) for every complete normalized state. Individual Mulliken
fragment weights need not be positive in a nonorthogonal basis.

The band center is the first projected spectral moment

\[
\bar{\varepsilon}_F =
\frac{\sum_i w_{iF}\omega_i\varepsilon_i}
     {\sum_i w_{iF}\omega_i}.
\]

Supported and persisted window policies are:

| `window` | \(\omega_i\) | Extra input |
| --- | --- | --- |
| `all` | 1 | none |
| `occupied` | supplied/generated \(f_i\) | occupations or `(mu,kT,g)` |
| `energy` | \(1_{\varepsilon_\min\leq\varepsilon_i\leq\varepsilon_\max}\) | inclusive `energy_window` |

A zero fragment denominator is undefined and raises an error rather than
returning NaN.

### Once-counted ICOHP-type energy partition

Define the directed atom block

\[
Q_{AB}=\sum_{\mu\in A,\nu\in B}
\mathrm{Re}\,[D_{\nu\mu}H_{\mu\nu}].
\]

`directed_energy[A,B]` stores \(Q_{AB}\). `onsite_energy[A]` stores \(Q_{AA}\).
For \(A<B\), `pair_energy[A,B]` stores \(Q_{AB}+Q_{BA}\), while its diagonal and
lower triangle are zero. Therefore every off-diagonal AO contribution is
counted exactly once:

\[
\sum_A E_{AA}+\sum_{A<B}E_{AB}
=\mathrm{Re}\,\mathrm{Tr}(DH).
\]

This is an integrated Hamilton-population-style one-electron energy
partition, following the COHP construction of Dronskowski and Blöchl
([J. Phys. Chem. 97, 8617 (1993)](https://doi.org/10.1021/j100135a014)).
It is not a full LOBSTER-compatible energy-resolved COHP implementation and
does not adopt a separate plotting sign convention such as `-COHP`.

### Fragment charge response

For a supplied fixed-spectrum density response \(D_\mu=dD/d\mu\),

\[
\frac{dp_A^\mathrm{M}}{d\mu}
=\sum_{\alpha\in A}\mathrm{Re}\,(D_\mu S)_{\alpha\alpha},
\]

with the analogous \(S^{1/2}D_\mu S^{1/2}\) Löwdin projection. This is the
derivative of the frozen-spectrum population ledger only. It excludes changes
of \(H\), \(S\), eigenvectors, ions, solvent, and double layer, so it is not a
complete electrochemical capacitance.

## Energy-zero gauge properties

Under a scalar energy-zero shift \(H'=H+cS\),

\[
\varepsilon'_i=\varepsilon_i+c,\qquad C'=C.
\]

An `all` band center therefore shifts by exactly \(c\). An `occupied` center
has the same covariance when \(\mu\) is shifted by \(c\), leaving occupations
unchanged. An `energy` center has it only when both energy-window bounds are
also shifted by \(c\).

The ICOHP-type partition defined above is **not gauge invariant**:

\[
Q'_{AB}=Q_{AB}+cM_{AB},\qquad
M_{AB}=\sum_{\mu\in A,\nu\in B}
\mathrm{Re}\,[D_{\nu\mu}S_{\mu\nu}].
\]

Thus the total changes by \(c\,\mathrm{Tr}(DS)=cN\); an unordered pair changes
by \(c(M_{AB}+M_{BA})\), and an on-site term by \(cM_{AA}\). This dependence is
an unavoidable consequence of partitioning the one-particle energy itself.
The analytic tests assert this transformation law and deliberately do not
claim false invariance.

## Relation to the Stenlid descriptor list

Stenlid and Žguns,
[ACS Energy Lett. 9, 3608–3617 (2024)](https://doi.org/10.1021/acsenergylett.4c01375),
discuss electrolyte-dependent Li-ion charge-transfer kinetics and identify
electronic-structure observables useful for alternative descriptor models.
The correspondence in this reference module is:

| Descriptor family | Module output | Native functional |
| --- | --- | --- |
| pDOS / band center | `PDOSResult.weights`, `band_centers` | \(C,S,\varepsilon\), plus window |
| Charge population | `PopulationResult.mulliken`, `lowdin` | \(D,S\) |
| COHP-like bonding | `BondEnergyResult` | \(D,H\) |
| Chemical-potential population response | `ChargeResponseResult` and scan curves | \(dD/d\mu,S\) or \(C,S,df/d\mu\) |

## Honest scope boundary

These outputs describe a single frozen one-particle spectrum. They are useful
features and diagnostics for D3/CIET-oriented datasets, but they are not
\(\Delta G^\mathrm{IT}\), the nuclear/solvent reorganization energy
\(\lambda\), or the donor–acceptor coupling \(H_{DA}\). They do not turn the
fixed-\(\mu\) postprocessor into grand-canonical DFT or a constant-potential
electronic-structure solver. Any model connecting these descriptors to charge
transfer kinetics requires separate labels, assumptions, and validation.
