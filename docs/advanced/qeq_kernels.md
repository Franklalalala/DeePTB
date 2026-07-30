# Finite-cluster QEq geometry kernels

`dptb.nnops.qeq_kernels` is a NumPy reference layer that constructs the
electronegativity vector and dense hardness kernel consumed by
[`solve_qeq`](qeq_operator.md). Coordinates are Cartesian Angstroms; charges
are in units of the elementary charge, and the constructed matrix is in
eV/e².

This layer is for isolated finite clusters only. It has no periodic minimum
image, Ewald/PME, constant-potential constraint, dipole correction, force, or
differentiable backend.

## Kernels

Write \(k_e = 14.3996\ {\rm eV\,Å}/e^2\), \(r_{ij}=|\mathbf R_i-\mathbf
R_j|\), and let \(J_i\) be the tabulated on-site hardness.

### Bare point-charge kernel

\[
J_{ii}=J_i,\qquad J_{ij}=\frac{k_e}{r_{ij}}\quad(i\ne j).
\]

This is the unscreened Coulomb limiting form of the QEq electrostatic matrix;
the QEq construction is from Rappé and Goddard, *J. Phys. Chem.* **95**, 3358
(1991), [doi:10.1021/j100161a070](https://doi.org/10.1021/j100161a070).
The off-diagonal term diverges at zero separation. The public builder rejects
overlapping sites before evaluating it.

Bare point interactions can also make the fixed-charge tangent block
non-positive at ordinary bond lengths when paired with a small empirical
on-site hardness. `solve_qeq_from_geometry` does not hide that problem: the
existing `solve_qeq` conditioning gate rejects such a matrix.

### Ohno/Klopman regularization

This implementation chooses the explicit convention

\[
\gamma_{ij}=\frac{2k_e}{J_i+J_j},\qquad
J_{ii}=J_i,\qquad
J_{ij}=\frac{k_e}{\sqrt{r_{ij}^2+\gamma_{ij}^2}}\quad(i\ne j).
\]

Thus the zero-distance pair limit is \((J_i+J_j)/2\), while the long-distance
limit is \(k_e/r_{ij}\). This screened two-center form traces to K. Ohno,
“Some Remarks on the Pariser–Parr–Pople Method,” *Theor. Chim. Acta* **2**,
219–227 (1964), and G. Klopman, *J. Am. Chem. Soc.* **86**, 1463–1469
(1964), [doi:10.1021/ja01062a001](https://doi.org/10.1021/ja01062a001).

The pair limit is finite, but an exact atom overlap is still invalid input and
is rejected.

### Isolated Gaussian-charge kernel

For element widths \(\sigma_i\),

\[
\sigma_{ij}=\sqrt{\sigma_i^2+\sigma_j^2},
\]

\[
J_{ij}=k_e\,
\frac{\operatorname{erf}\!\left[r_{ij}/(\sqrt{2}\sigma_{ij})\right]}
{r_{ij}}\quad(i\ne j),
\]

\[
J_{ii}=J_i+\frac{k_e}{\sqrt{\pi}\sigma_i}.
\]

The final term is the analytic Gaussian self interaction. The off-diagonal
zero-distance limit is \(k_e\sqrt{2/\pi}/\sigma_{ij}\), and its
long-distance limit is again \(k_e/r_{ij}\).

The equations follow the Gaussian-charge decomposition in Hu et al.,
*Nature Communications* **16**, 7379 (2025),
[doi:10.1038/s41467-025-62824-5](https://doi.org/10.1038/s41467-025-62824-5),
Supplementary Eqs. S6–S7. This module keeps only the isolated analytic
Gaussian interaction. It does not implement the paper's periodic PME or
dipole correction.

## Default element parameters

The default \(\chi\) and \(J\) values are Hu et al. Supplementary Table S1.
The Gaussian widths are the covalent radii used as `sigma` in the authors'
accompanying `sxu39/DP-QEq` implementation.

| Element | χ (eV/e) | J (eV/e²) | σ (Å) |
|---|---:|---:|---:|
| Li | -3.0000 | 10.0241 | 1.28 |
| C | 5.8678 | 7.0000 | 0.76 |
| H | 5.3200 | 7.4366 | 0.31 |
| O | 8.5000 | 8.9989 | 0.66 |
| P | 1.8000 | 7.0946 | 1.07 |
| F | 9.0000 | 8.0000 | 0.57 |

The table carries structured source, DOI, unit, and applicability provenance.
Its numeric entries and provenance have a deterministic SHA-256 digest.
Callers may pass a `QEqParameterTable` as a complete table, or pass a mapping
whose entries override or extend the default. Missing elements are rejected;
there is no fallback based on periodic-table trends.

The original Hu method contains Gaussian charges, periodic PME, and a dipole
correction. Using these constants with the bare finite-cluster kernel is not a
reproduction of Hu et al. The Ohno kernel is another reference regularization,
not a fit reported by Hu et al.

## Geometry solve and provenance

```python
from dptb.nnops import solve_qeq_from_geometry

result = solve_qeq_from_geometry(
    positions=[[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]],
    symbols=["H", "F"],
    total_charge=0.0,
    kernel="gaussian",
)

print(result.charges)
print(result.provenance.geometry_sha256)
print(result.provenance.kernel_sha256)
print(result.provenance.parameter_table_sha256)
```

The function constructs \(\chi\) and \(J\), calls the existing `solve_qeq`
without modifying or bypassing it, and returns a `QEqGeometryResult`.
`QEqGeometryResult` is also a `QEqResult`; it adds bytes-backed read-only
coordinates, canonical symbols, the resolved parameter table, and SHA-256
bindings for geometry, kernel, and table.

## Honest boundaries

- Cluster only: no cell, periodic images, Ewald, PME, or production Poisson
  solver.
- Fixed total charge only: this layer does not implement Hu-style ConstP,
  electrode masks, or applied potentials.
- No forces or coordinate derivatives. It is a CPU NumPy reference, not a
  training backend.
- The QEq Lagrange multiplier \(\lambda\) enforces total charge. It is not an
  electrode voltage.
- A finite pair formula does not legalize duplicated atoms; exact or numerical
  overlaps fail closed.
- A constructed symmetric matrix is still subject to `solve_qeq`'s constrained
  positive-definiteness and conditioning checks.
