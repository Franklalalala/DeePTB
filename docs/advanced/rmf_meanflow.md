# PyTorch Riemannian MeanFlow for Hamiltonian CFM

This patch adds an opt-in RMF path for DeePTB Hamiltonian/SOC `uu_real` residual
training.  It does **not** change default CFM behavior: the normal builder still
returns `HamiltonianCFM` unless `flow_options.type` or `flow_options.objective` is
set to `rmf`.

## Entry point

RMF is installed through the existing trainer-side flow builder:

```json
"flow_options": {
  "enabled": true,
  "type": "rmf",
  "objective": "rmf",
  "mode": "residual",
  "prior": "zero",
  "manifold": "euclidean"
}
```

`mode = "residual"` keeps the current DeePTB/NextHAM semantics: the dataset H0
is the base Hamiltonian, RMF works on the residual coordinate `z`, the model sees
`H0 + z_t`, and the endpoint target remains the clean real-H residual/full
Hamiltonian output expected by the existing loss stack.  The model is not changed
into a direct full-H residual-free predictor.

## Manifold interface

`dptb.nnops.rmf` defines a small PyTorch manifold API:

- `project(x, v)`
- `expmap(x, v)`
- `logmap(x, y)`
- `geodesic_interpolate(x0, x1, t)`
- `tangent_velocity(x0, x1, t)`

The first implementation is `EuclideanManifold`, which is the correct fallback
for unconstrained Hamiltonian residual tensors and introduces no JAX or external
geometry dependency.  Non-Euclidean names raise a clear `NotImplementedError`
rather than silently importing optional dependencies.

## Math convention

The RMF paper defines average velocity with the manifold logarithm and recovers a
flow map with the exponential map.  DeePTB keeps the existing denoising convention
used by pixel MeanFlow: clean residual is at `t = 0`, prior residual is at
`t = 1`.  Therefore the conditional path is

```text
z_t = geodesic_interpolate(z_clean, z_prior, t)
```

and the endpoint/x1-prediction average noise-time velocity used in the loss is

```text
u_theta(z_t, t) = -logmap(z_t, z_theta_clean) / t.
```

In Euclidean space this reduces to `(z_t - z_theta_clean) / t`, matching the
existing pixel MeanFlow semantics.  The target velocity uses the same expression
with the clean residual endpoint.  A finite-difference `du/dt` target is used so
training remains PyTorch-only and avoids higher-order derivative dependencies.

## TensorBoard and expert routing

RMF does not alter expert construction or distance-range routing.  The model and
trainer still own `distance_ranges`, `mean_max_prob`, `expert_load_cv`, and expert
learning-rate reporting.  RMF exposes train-time legacy-compatible source keys
(`train_onsite_loss`, `train_hopping_loss`, `mean_max_prob`, `expert_load_cv`) so
the existing TensorBoard tags remain available:

- `train_hopping_loss_iter`
- `train_onsite_loss_iter`
- `validation_loss_mean`
- `mean_max_prob_iter`
- `expert_load_cv_iter`
- `Expert_LR_Iter`

## Minimal smoke command

Adjust the dataset/checkpoint paths in the smoke JSON to the local Hanhai small
SOC uu_real data, then run:

```bash
dptb train examples/hanhai_soc_uureal_rmf_smoke.json -o runs/hanhai_soc_uureal_rmf_smoke
```

For a faster unit-only check:

```bash
pytest -q dptb/tests/test_rmf_flow.py
```
