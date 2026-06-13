# Pixel MeanFlow smoke configuration

Pixel MeanFlow is an opt-in Hamiltonian endpoint objective. The DeePTB path
uses endpoint prediction with a finite-difference `du/dt` backend; exact JVP is
not implemented in this repo yet.

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
    "meanflow": {
      "profile": "conservative",
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

The model must receive the two-time conditioning keys:

```json
{
  "embedding": {
    "use_flow_time_embedding": true,
    "flow_time_keys": ["flow_time_t", "flow_time_r", "flow_time_h"]
  }
}
```

Use the aggressive profile only as a separate ablation. It enables extra
stabilizers such as adaptive normalization and boundary-velocity auxiliary
loss; those are not required for the paper-conservative path.
