# UniTB training configurations

`input.json` (UniTB), `dense/input.json` (UniTB-dense), and `slem/input.json`
(UniTB-SLEM) use the standard training configuration. The SLEM example differs
from `input.json` only by `"layer_topology": "slem"`.
Embedding sections list options differing from defaults. The examples use H/O,
compact uu-real output, and H−H0 supervision. Replace the basis with the data's
actual atomic orbitals while preserving its SOC and representation conventions;
changing element names alone is insufficient.

The single epoch, small batch, and Adam settings exercise the training entry
point. Full training hyperparameters and batching depend on the experiment.

Replace `DATA_ROOT/train` and `DATA_ROOT/validation` with LMDB directories.
Primary records must include the saved edge graph, Hamiltonian targets, and the
physical `hamiltonian_0` prior, loaded as `node_h0/edge_h0`.
`target_kind=h0res` supervises ΔH. Per-element cutoffs must match the data recipe;
`embedding.r_max` accepts a dictionary.

For UniTB and UniTB-SLEM, replace `OVERLAP_ROOT` with the physical overlap
sidecar root. Each split's shards, keys, structure identities, and edge ordering
must match the primary data. Sidecars contain physical `S`, not network latents.

All three examples supervise onsite terms through `distance_ranges`.
UniTB and UniTB-SLEM enable the local QEq charge head and onsite-scale prior
noise. For hopping, set `te_prior_sigma` to 5 and use the standard training
entry point's distance/label masks. Noise configuration does not replace
distance-expert supervision masks.

Training-only prior noise uses `t=0`. The examples explicitly set
`flow_options.t_max=0`; `t_min` and `t0_probability` retain their defaults of 0.
Disabling flow alone does not change its default `t_max=0.999`.
The noise sampler checks all three values.

Run from this example directory, after setting the data paths:

~~~bash
dptb train input.json -o output
dptb train dense/input.json -o dense-output
dptb train slem/input.json -o slem-output
~~~

For CPU reference checks, select `common_options.device=cpu` and a supported
reference SO2 configuration. CUDA acceleration uses optional SO2CUDA.
Data paths are placeholders; training data and trained weights are not included.

See [the UniTB guide](../../docs/unitb.md) for structure, compatibility names,
and options, and [the architecture guide](../../docs/architecture.md) for shared
construction APIs.
