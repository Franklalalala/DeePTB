# Basic API

## Build a model

```python
from dptb.nn import build_model

model = build_model(checkpoint="/path/to/checkpoint.pth")
```

For construction from scratch, pass the same `model_options` and
`common_options` dictionaries accepted by the strict training schema:

```python
model = build_model(
    model_options=model_options,
    common_options=common_options,
)
```

`0726-light` accepts the maintained `e3tb` and `block_native` prediction
routes. Retired NNSK, DFTB-SK, MIX, and SKTB prediction checkpoints fail
explicitly.

## Build graph data

```python
from dptb.data import AtomicData

atomic_data = AtomicData.from_ase(
    atoms=atoms,
    r_max=5.0,
)
data = AtomicData.to_AtomicDataDict(atomic_data)
```

Datasets can be built through `dptb.data.build_dataset`; the cutoff must match
`model_options.embedding.r_max`.

## Predict Hamiltonian features

```python
prediction = model(data)
```

For E3 prediction, `node_features` and `edge_features` contain onsite and
hopping Hamiltonian features. Convert them to real-space blocks with:

```python
from dptb.data import feature_to_block

h_blocks = feature_to_block(data=prediction, idp=model.idp)
```

## Electronic structure

```python
from dptb.nn import HR2HK, Eigenvalues
from dptb.postprocess import Band
```

`HR2HK` converts real-space blocks to k-space, `Eigenvalues` solves the
eigenproblem, and `Band` evaluates and plots a band path.
