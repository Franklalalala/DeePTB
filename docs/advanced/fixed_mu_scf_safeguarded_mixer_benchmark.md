# Fixed-mu SCF safeguarded-mixer benchmark

The periodic Pulay safeguards in `fixed_mu_scf_operator.py` were calibrated
against a deterministic 7200-case benchmark rather than a single clipped
failure. The frozen artifacts are under:

```text
F:\claude\0730_next_stage\results\M\benchmark
```

The grid covers one-level, two-level, diatomic, and seeded random small-matrix
fixed-point models. It varies chemical potential relative to the levels,
`kT`, feedback strength, mixing step, history, and period. The independent
harness exactly reproduces the reviewer's 3150-case counts
(`2100 / 14 / 284 / 752`) and matches the production operator's convergence
classification and iteration count on all 51 sampled arm/case checks.

## Calibrated defaults

| parameter | default |
|---|---:|
| `pdiis_gram_condition_threshold` | `1e10` |
| `pdiis_step_ratio_threshold` | `40` |
| `pdiis_residual_growth_threshold` | `1.0` |

The Gram condition is measured on the numerical range retained by the legacy
`pinv(..., rcond=1e-12)`. Structural null directions are excluded because a
mature one-level history is necessarily rank deficient when represented as a
history-by-history Gram matrix. The step-ratio threshold is above the
approximately `1 / beta = 33.3` ratios observed for valid accelerated cases at
`mixing_step=0.03`. The residual-growth threshold requires an accepted Pulay
candidate not to worsen the independently evaluated next residual.

## Three-arm acceptance

| metric | linear | legacy PDIIS | safeguarded |
|---|---:|---:|---:|
| converged / 7200 | 3813 | 5196 | 5391 |
| mean iterations, failures capped at 200 | 114.616 | 69.116 | 64.102 |
| mean iterations, baseline-both subset | 37.880 | 12.019 | 11.575 |
| mean iterations, baseline-linear-only subset | 71.176 | 200.000 | 33.137 |
| mean iterations, retained-PDIIS-only subset | 200.000 | 33.716 | 31.263 |

- Linear retention: `3813 / 3813 = 100%` (zero losses).
- PDIIS-only retention: `1439 / 1485 = 96.90%`.
- Nonconverged cases decrease from 2004 for legacy PDIIS to 1809.

The production change preserves the independent merit evaluator introduced by
`64db014`, replaces its hard-coded comparison with the explicit
residual-growth threshold, and layers Gram conditioning, step-ratio rejection,
history reset, persistence fields, and a legacy-disable switch on top.

## Reproduction

```powershell
Set-Location 'F:\claude\0730_next_stage\results\M\benchmark'
& 'C:/Users/16608/.conda/envs/dptb/python.exe' benchmark_suite.py
& 'C:/Users/16608/.conda/envs/dptb/python.exe' plot_baseline.py
& 'C:/Users/16608/.conda/envs/dptb/python.exe' tune_safeguards.py `
  --thresholds 1e10,40,1.0 `
  --output three_arm/safeguarded_results.csv
& 'C:/Users/16608/.conda/envs/dptb/python.exe' summarize_three_arm.py
& 'C:/Users/16608/.conda/envs/dptb/python.exe' validate_production_oracle.py
```
