# Pixel MeanFlow smoke configuration

Pixel MeanFlow is an opt-in Hamiltonian endpoint objective. The DeePTB path
uses endpoint prediction and supports the original local MeanFlow objective
with either finite-difference or JVP `du/dt`. It also exposes a KAIST-style
semigroup consistency objective for ablations.

Minimal paper-conservative flow fragment:

```json
{
  "flow_options": {
    "enabled": true,
    "objective": "pixel_meanflow",
    "mode": "residual",
    "prior": "zero",
    "overwrite_feature_keys": true,
    "validation_ode_steps": [1, 3],
    "apply_to_reference": false,
    "meanflow": {
      "profile": "conservative",
      "objective": "finite_difference",
      "jvp_tangent": "boundary",
      "time_sampling": "logit_normal",
      "p_mean": -0.4,
      "p_std": 1.0,
      "data_proportion": 0.5,
      "tr_uniform_prob": 0.1,
      "min_t": 0.05,
      "aux_endpoint_weight": 0.05,
      "aux_boundary_v_weight": 0.0,
      "norm_p": 0.0,
      "du_dt_backend": "finite_difference"
    }
  }
}
```

KAIST semigroup ablation:

```json
{
  "flow_options": {
    "enabled": true,
    "objective": "pixel_meanflow",
    "mode": "residual",
    "prior": "zero",
    "overwrite_feature_keys": true,
    "validation_ode_steps": [1, 3],
    "meanflow": {
      "profile": "conservative",
      "objective": "semigroup",
      "semigroup_weight": 1.0,
      "semigroup_endpoint_weight": 1.0,
      "time_sampling": "logit_normal",
      "p_mean": -0.4,
      "p_std": 1.0,
      "data_proportion": 0.5,
      "tr_uniform_prob": 0.1,
      "min_t": 0.05
    }
  }
}
```

`meanflow.objective: "hybrid"` keeps the original local MeanFlow loss and adds
the semigroup term as an auxiliary loss.

The model must receive the two-time conditioning keys:

```json
{
  "embedding": {
    "use_flow_time_embedding": true,
    "flow_time_keys": ["flow_time_t", "flow_time_r", "flow_time_h"]
  }
}
```

Place this embedding block under your active `model_options.embedding` or
`lem_moe_v3_h0` embedding config. It is not a top-level config section.
For DeePTB residual correction smokes, prefer `prior: "zero"`; Gaussian
residual priors should be separate ablations.

Use the aggressive profile only as a separate ablation. It enables extra
stabilizers such as adaptive normalization and boundary-velocity auxiliary
loss; those are not required for the paper-conservative path.

## Physical prior and jitter

For no-H0 or weak-H0 PMF experiments, prefer a low-cost physical prior over a
plain dense Gaussian. The current non-NN on-the-fly path uses DeePTB's DFTB-SK
module with Slater-Koster parameters:

```json
{
  "flow_options": {
    "enabled": true,
    "objective": "pixel_meanflow",
    "mode": "residual",
    "prior": "dftbsk",
    "prior_skdata": "/path/to/slater_koster_files_or_skparams.pth",
    "dftb_prior_overlap": false,
    "dftb_prior_require_geometry": true,
    "missing_h0_policy": "warn_zero",
    "physical_prior_jitter_sigma": 0.02,
    "physical_prior_jitter_reference_scale": true,
    "physical_prior_jitter_edge_decay": 3.0
  }
}
```

This path runs `DFTBSK(..., transform=True)` under `torch.no_grad()`, aligns the
result to the active node/edge feature layout, converts the absolute guess to a
residual prior, and then writes the interpolated state back to `node_h0` /
`edge_h0` before the model initial layer sees the batch. It does not use NNSK.

For user-facing reports, describe the endpoint metric as
`label_delta_H - pred_delta_H`. The code still contains internal
MeanFlow-derived correction keys, but the model output remains the clean
Hamiltonian endpoint/residual endpoint.

`physical_prior_jitter_sigma` adds perturbations around the physical prior. With
`physical_prior_jitter_reference_scale=true`, the perturbation is scaled by the
row RMS of the active target/residual blocks. With
`physical_prior_jitter_edge_decay > 0`, edge jitter is multiplied by
`exp(-edge_length / physical_prior_jitter_edge_decay)`, so long-range hopping
entries receive smaller perturbations.

## Comparable loss keys

For non-CFM, block-native/P2, CFM, and Pixel MeanFlow runs, `train_loss`,
`train_onsite_loss`, `train_hopping_loss`, `validation_loss`,
`validation_onsite_loss`, and `validation_hopping_loss` are reserved for the
route's clean endpoint metric. Feature/RME non-CFM, CFM, and MeanFlow use the
same `hamil_abs` reduction and can be plotted together.

Block-native heads use AO-block endpoint reductions by default. They keep the
same six tag names, so a block-flow run can be compared with its block-native
non-CFM baseline, but those values must not be presented as RME-space values.
Setting `log_feature_compatible=true` on a blockwise loss opts into the exact old
RME-compatible slice reduction. That slice walk does not materialize a full RME
tensor, but it launches many small GPU reductions; prefer enabling it only on a
separate validation loss when cross-representation comparison is essential.

Pixel MeanFlow is a model-in-loss route: its training endpoint statistics are
reduced directly from `flow_options.node_target_key` and `edge_target_key`.
Consequently, a block-native training criterion (`endpoint_metric_space=block`)
must use explicit block-space MeanFlow target keys that the selected model route
also emits under those same names, with the criterion prediction/target route
aligned. The default `node_features` / `edge_features` targets are RME-space and
are rejected with a configuration-time error when paired with a block endpoint
criterion. The Trainer intentionally does not add an online block-to-RME or
RME-to-block conversion. If a model route cannot expose its endpoint under a
shared block-space key, use the matching RME feature route/criterion instead.

The actual block/flow/semigroup optimization objective remains visible under
`train_loss_opt`. Single-Trainer flow runs additionally expose their detailed
`train_flow_*` / `validation_flow_*` diagnostics; MultiTrainer does not invent
a flow-only scalar when it cannot aggregate that objective unambiguously.
TensorBoard hides the duplicate Euler-1 `compatible` aliases; extra Euler-step
endpoint diagnostics remain available.

## Time-embedding vs finite-difference scale

When the model uses a sinusoidal `FlowTimeConditioner`, keep the finite-difference
time step small relative to the embedding scale. With the default
`flow_time_max_positions=2000`, `fd_eps=0.01` moves the fastest sinusoidal phase by
about 20 radians between the main and finite-difference forward passes. That
can make the finite-difference `du/dt` term measure time-embedding oscillation
instead of the intended path derivative. Start with one of:

- `meanflow.fd_eps <= 5e-4` with the default `flow_time_max_positions=2000`;
- or an explicit `flow_time_max_positions` ablation such as 100-200;
- validation always emits endpoint-compatible legacy loss keys; keep at least one
  `validation_flow_*` diagnostic enabled only if you also need to inspect the
  internal flow objective.
