# C1 operator conformance and adversarial dataset

This benchmark is a deterministic reference-oracle, regression, and fuzz
dataset for the NumPy/SciPy fixed-μ, QEq, and finite-cluster
Mulliken-Hartree SCF operators.  It is not a materials dataset.

## Reproduce

Use the project test interpreter from the repository root:

```powershell
$python = 'C:/Users/16608/.conda/envs/dptb/python.exe'
$out = 'F:/claude/0730_next_stage/results/K'

& $python -m benchmarks.conformance.runner `
  --n-cases 20000 --seed 730 --output-dir $out --shard-size 500

& $python -m benchmarks.conformance.figures `
  --cases "$out/cases.csv" --output-dir "$out/figs"

& $python -m pytest dptb/tests/test_conformance_smoke.py -q
```

`--case-id c1-000123` reruns one case from the generated prefix.  Set
`--n-cases` high enough to include the requested numeric ID.  The generator
derives an independent seed from `(seed, case_index)`, so increasing
`--n-cases` does not change the existing prefix.

## Controlled families

- fixed-μ: real, complex, and explicit SOC-style Hermitian `H`; SPD `S`;
  `n_orb=1..32`; `cond(S)` log-uniform through `1e12` plus reject cases through
  `1e14`; `kT={0,0.02585,0.1}`, `g={1,2}`, and 1–8 k points.
- QEq: a controlled SPD spectrum in the `sum(q)=0` tangent space, plus
  `alpha 11^T` with `alpha=1e-6..1e16`; `n_site=1..64`; and
  `Q in [-2,2]`.
- SCF: analytic one-level, symmetric/asymmetric dimers, and random small
  matrices; zero, diagonal Hubbard, and controlled SPD kernels; linear/PDIIS
  mixing.
- Mutations: wrong shape, NaN/Inf, non-Hermitian input, non-SPD input,
  truncated eigenbasis, wrong request binding, rewritten serialized fields,
  and self-reported loose tolerances.

Every successful result is passed through `validate_conservation` or
`validate_qeq_result`.  The SCF result's terminal fixed-μ payload is also
validated.  Both accepted and rejected cases are retained.

## On-disk schema

`cases.csv` contains scalar requests, diagnostics, provenance, verdicts, and
an NPZ lookup.  `matrices-*.npz` contains full arrays under keys of the form
`<matrix_prefix>__H`, `<matrix_prefix>__S`, `<matrix_prefix>__J`, and so on.

| Field group | CSV fields | NPZ examples |
|---|---|---|
| Identity/provenance | `case_id`, `seed`, `operator`, `family`, `generator_version`, `code_commit`, `dtype`, `expected_status` | — |
| Request | `mu`, `kT`, `spin_degeneracy`, `total_charge`, `alpha`, mixing controls, dimensions | `H`, `S`, `mu_grid`, `k_weights`, `chi`, `J`, `H0`, `K`, `ao_atom_index`, `n_ref` |
| fixed-μ labels | `electron_count`, `dos_like_response`, `band_*`, eigen/trace/Hermiticity/condition diagnostics | `eps`, `C`, `f`, `df_dmu`, `D`, `dD_dmu` |
| QEq labels | `qeq_energy`, `qeq_lambda`, charge/tangent/condition/gauge diagnostics | `q`, `q_independent`, `J_base` |
| SCF labels | `scf_iterations`, `scf_final_residual`, analytic parity | `q`, `phi`, `residual_history`, terminal `D` |
| Certification | `actual_status`, `validator_pass`, `verdict_match`, `exception_class`, `reject_reason`, `analytic_reference`, `max_abs_error` | exact rejected payload |
| Array lookup | `matrix_path`, `matrix_prefix` | `<matrix_prefix>__<field>` |

If an expected-reject case is accepted, the runner writes a minimal
dimension-two payload and metadata under `bug_reproducers/<case_id>/`.

## Seven C1.2 figures

The figure command writes:

1. `01_condition_residual.png`: `cond(S)` versus generalized-eigen and
   `Tr(DS)-N` residuals.
2. `02_acceptance_domain_heatmap.png`: condition/tolerance acceptance domain.
3. `03_fixed_mu_gauge_curve.png`: `H+cS, mu+c` covariance.
4. `04_qeq_alpha_gauge_scan.png`: QEq uniform-gauge drift with the documented
   float64 loss region shaded.
5. `05_scan_point_parity.png`: scan versus point `N`, `dN/dmu`, and band-ledger
   parity.
6. `06_analytic_root_parity.png`: one-level SCF and independent QEq parity.
7. `07_mutation_confusion_matrix.png`: expected versus validator verdict.

For the production run, the example PNGs and `figures.json` are at
`F:/claude/0730_next_stage/results/K/figs/`.

## C1.3 诚实边界（原文）

- 本集证明的是**数值与契约一致性**，不证明 H、S、χ、J 对任何材料是真实的。
- 不能用它声称电位响应、界面电容、氧化还原或恒电位 MD 能力。
- QEq 超大均匀规范项在 float64 形成输入时会吞掉切空间小量；审稿乙实测
  `α=10¹²–10¹⁶` 已出现从微电子到百分之一电子量级的漂移
  （审稿乙 `:136-147`）。图中必须把这是数值适用域而非材料效应写清。

In particular, fixed-μ here is frozen-H/S grand-canonical single-particle
occupation post-processing, `band_grand_energy` is a single-particle ledger,
the SCF operator is a finite-cluster Mulliken-Hartree reference, QEq is a
fixed-total-charge reference, and `dN/dmu` is not a complete electrochemical
capacitance.
