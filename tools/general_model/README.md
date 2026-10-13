# Two-stage prior model tools

Configuration, dataset checks and the training entry point for two-stage models
that take an H0 or NACF prior and learn a residual target. `make_configs.py`
uses the fixed model capacity and training controls in `configs/base.json`;
record any change to capacity or to an experimental variable explicitly instead
of treating these defaults as general settings.

```bash
python tools/general_model/make_configs.py --data /path/to/data --output /path/to/configs
python tools/general_model/check_dataset.py --input /path/to/configs/h0_h0res.onsite.s1.json --output /path/to/loader.json
python tools/general_model/run_stage.py --input /path/to/configs/h0_h0res.onsite.s1.json --output /path/to/s1
python tools/general_model/run_stage.py --input /path/to/configs/h0_h0res.onsite.s2.json --s1-checkpoint /path/to/s1.pth --output /path/to/s2
```

The three data routes are `h0_h0res`, `nacf_nacfres` and `nacf_h0res`. Prior
and target are chosen independently: store the data fields for the chosen route
with consistent atoms, edges and periodic shifts, and never add a prior back to
a residual that does not match it.

S1 is the pairwise branch; S2 loads a matching, completed S1. Resume within a
stage with `--restart`. `--smoke-steps` keeps the original scheduler and runs a
finite number of updates to check the entry point, gradients and checkpointing.

`verify_checkpoint.py CHECKPOINT --steps N --output RECEIPT` checks completed
steps, finite parameters and the tool's fixed expert structure. It does not
establish prediction accuracy or exact data-cursor restoration.

Keep generated configurations, run outputs and checkpoints outside the repository.
