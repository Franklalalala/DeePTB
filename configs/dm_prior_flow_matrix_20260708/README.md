# DeePTB ordinary-CFM snippets for the DM prior matrix

These snippets cover groups C/D/E of the six-run comparison.  They should be
merged into the same ordinary DeePTB CFM base config used for the latest aligned
CFM smoke.  Groups A/B/F live in the EMolFlow companion config directory.

## Dataset roots

- C/E use:
  `/home/mingkang_nt/data/1118/cluster_1123_charge/updated_lmdb_overlap_huckel_haar_k8`
- D uses:
  `/home/mingkang_nt/data/1118/cluster_1123_charge/updated_lmdb_overlap_huckel_haar_k8_dm_rme_k1`

The base k8 dataset contains precomputed `node_features`, `edge_features`,
`node_overlap`, `edge_overlap`, `node_h0`, `edge_h0`, and `haar_u0`.
Group D additionally requires `haar_node_features` and `haar_edge_features`.

## Groups

| Group | File | Optimized route |
|---|---|---|
| C | `C_cfm_te_zero_base.yaml` | `base=0`, `prior=te`, typewise feature-space CFM |
| D | `D_cfm_haar_dm_prior.yaml` | `base=0`, `prior=haar_dm`, precomputed `RME(D_haar)` |
| E | `E_cfm_overlap_huckel_te.yaml` | `base=node_h0/edge_h0`, `prior=te` on the residual |

## Loss and log semantics

For all three groups, the optimized scalar is DeePTB `train_flow_loss`.  The
`loss_options` block is still present because the trainer uses it for
endpoint-compatible report metrics.  It is not added to the optimized CFM loss.

Current CFM code forces endpoint-compatible train/validation report fields when
`flow_options.enabled=true`; read them as report-only metrics.  The flow
objective itself is logged under `train_flow_*` and `validation_flow_*`.

## Group formulas

C:

```text
base = 0
residual = ref
te_prior = structured_noise_like(ref)

t=0:   pure TE prior
t=0.5: TE prior / ref each half
t=1:   pure ref
```

D:

```text
x0 = RME(D_haar)
x1 = ref

t=0:   pure Haar DM prior
t=0.5: Haar DM prior / ref each half
t=1:   pure ref
```

The offline construction targets:

```text
D_haar S D_haar ~= spin_factor * D_haar
Tr(D_haar S) ~= nelec
D_haar = D_haar.T
```

If `haar_node_features` or `haar_edge_features` is absent, `prior=haar_dm`
raises a `KeyError`.  This is intentional; group D must not silently degrade to
a nonphysical random-matrix prior.

E:

```text
base = overlap_huckel
residual = ref - overlap_huckel
te_prior = structured_noise_like(ref - overlap_huckel)

t=0:   overlap_huckel + TE residual prior
t=0.5: overlap_huckel + TE prior / ref each half
t=1:   pure ref
```

The base is read from `node_h0/edge_h0` only because that is the existing loader
key convention for a precomputed state.  Do not interpret this as a learned or
DFT H0 input.

## Precompute command for D

Run this in the EMolFlow checkout to create the extra group-D dataset:

```bash
python tools/preprocess_overlap_huckel_haar.py \
  --input-root /home/mingkang_nt/data/1118/cluster_1123_charge/updated_lmdb \
  --output-root /home/mingkang_nt/data/1118/cluster_1123_charge/updated_lmdb_overlap_huckel_haar_k8_dm_rme_k1 \
  --splits train valid test \
  --haar-k 8 \
  --haar-dm-rme-k 1 \
  --spin-factor 2.0 \
  --overwrite
```

## Smoke commands

After merging one snippet into the full base config:

```bash
dptb train merged_C_cfm_te_zero_base.json -o smoke_C_cfm_te_zero_base
dptb train merged_D_cfm_haar_dm_prior.json -o smoke_D_cfm_haar_dm_prior
dptb train merged_E_cfm_overlap_huckel_te.json -o smoke_E_cfm_overlap_huckel_te
```

