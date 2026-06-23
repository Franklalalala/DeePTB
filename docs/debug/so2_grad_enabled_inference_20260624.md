# SO2 Grad-Enabled Inference Debug Note

Date: 2026-06-24

Scope: Liyue downstream water evaluation for
`liyue__N1_single_h0_blockwise_rop_cfm_te1_lmax4_qhflow64_0621cfm_20260623_160330`
on branch `0621-CFM`.

## Symptom

The downstream water evaluator finished the first batch, then stalled on the
second model forward under `torch.no_grad()`. The same second-forward stall was
reproduced with direct `model(data)` calls, so the failure was not isolated to
the CFM sampler wrapper. An older N0 no-CFM run showed the same pattern in the
same evaluator shape.

## Checks

- The evaluated checkpoint was the latest `nnenv.ep250.pth` snapshot from the
  Liyue N1 production run.
- The run config reported `embedding.output_route = None` and
  `prediction.blockwise_hamiltonian = True`; `Output Head: route=legacy_rme`
  is expected for this Liyue N1 family.
- The water dataset was verified as the water test split, with 3999 samples and
  24x24 Hamiltonian/overlap matrices.
- Early `Sc ~= 0.361` came from an older `iter10000` probe and was not the final
  latest-checkpoint result.
- Switching SO2 settings to staged or split-loop fallbacks did not remove the
  second-forward stall while the evaluator stayed inside `torch.no_grad()`.
- Keeping autograd enabled for prediction avoided the stall. Metrics were still
  detached for aggregation.

## Practical Workaround

Downstream evaluators that hit this SO2 second-forward stall should wrap the
model or flow prediction section in `dptb.utils.inference.grad_enabled_inference`.
Use the explicit flag when reproducing this Liyue evaluator:

```python
from dptb.utils.inference import grad_enabled_inference

with grad_enabled_inference(model, enable_grad=True):
    pred = model(data)
```

The helper also supports the environment override
`DPTB_ENABLE_GRAD_INFERENCE=1`. It does not change model `forward` semantics or
training behavior.

## Full Water Result

The final full water evaluation completed all 3999 samples with grad-enabled
prediction. The latest N1 checkpoint was still much worse than the previous D3
downstream baseline:

- Hamiltonian MAE: `0.023148583237992605`
- epsilon occupied MAE: `0.37118323037199136`
- HOMO MAE: `0.08684083235375624`
- LUMO MAE: `0.20504036354847135`
- gap MAE: `0.13189678810816746`
- Sc fraction: `0.5436195281591482`
- onsite L1/RMSE: `0.03981095948733192`
- hopping L1/RMSE: `0.04218727894411538`

The retrieved local bundle is
`E:\deeptb\codex\0621_cfm_fix\retrieved\liyue_N1_water_downstream_20260624_0103`.
