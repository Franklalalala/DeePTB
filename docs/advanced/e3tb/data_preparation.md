# LMDB data preparation

`0726-light` keeps only the LMDB dataset backend used by the maintained
training and flow pipelines. Each shard directory must match the configured
`prefix` and contain an `.mdb` file.

```json
{
  "data_options": {
    "train": {
      "type": "LMDBDataset",
      "root": "./data",
      "prefix": "set",
      "get_Hamiltonian": true
    }
  }
}
```

The same contract is available from Python:

```python
from dptb.data import build_dataset

dataset = build_dataset(
    root="./data",
    type="LMDBDataset",
    prefix="set",
    get_Hamiltonian=True,
    basis={"Si": "2s2p1d"},
    r_max=6.0,
)
```

Legacy in-memory, HDF5, DeePH, ASE, NPZ, and ABACUS dataset classes are not
part of this branch. Convert those sources to LMDB before training.
