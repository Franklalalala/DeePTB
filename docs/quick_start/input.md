# Inputs and commands

`0726-light` supports the maintained E3/LEM/EMol model family. Training
configuration is validated strictly: unknown embedding, prediction, loss, or
task keys are rejected before a run starts.

## Data

Every sample contains atomic positions, cell/PBC information, atomic numbers,
and the labels required by the selected objective. Hamiltonian, density-matrix,
overlap, H0, P2/P23, and block-native fields are stored in LMDB shards
produced by the maintained conversion tools.

The graph cutoff comes from `model_options.embedding.r_max`. Dataset-specific
options live under `data_options.train`, `data_options.validation`, and
`data_options.reference`.

## Training configuration

A training input has four top-level sections:

```json
{
  "common_options": {
    "basis": {"Si": ["3s", "3p"]},
    "device": "cuda",
    "dtype": "float32",
    "overlap": false
  },
  "model_options": {
    "embedding": {
      "method": "lem_moe_v3",
      "irreps_hidden": "64x0e + 32x1o + 16x2e",
      "avg_num_neighbors": 16.0,
      "r_max": 5.0,
      "n_layers": 4
    },
    "prediction": {
      "method": "e3tb"
    }
  },
  "data_options": {
    "train": {
      "type": "LMDBDataset",
      "root": "/path/to/train",
      "prefix": "data",
      "get_Hamiltonian": true
    }
  },
  "train_options": {
    "num_epoch": 100,
    "optimizer": {"type": "Adam", "lr": 0.001},
    "loss_options": {
      "train": {"method": "hamil_abs"}
    }
  }
}
```

Use the generated reference under [Input Parameters](../input_params/index.rst)
for the complete schema, including current flow, block-ODE, H0, pair, and
nonlinear options.

Start or resume training with:

```bash
dptb train input.json -o output
dptb train input.json --restart output/checkpoint/nnenv.latest.pth -o output
```

## Inference tasks

The maintained `dptb run` tasks are:

- `band`
- `write_block`

Other historical SK conversion, transport, DOS, and Fermi-surface entrypoints
are intentionally not part of `0726-light`.
