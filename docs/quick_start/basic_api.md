# Basic API

The Python API provides the same model construction used by the CLI. See the
[architecture guide](../architecture.md) for component ownership and embedding
registration.

## Build a model

Restore a checkpoint, including one written by an earlier release:

~~~python
from dptb.nn import build_model, load_checkpoint

model = build_model(checkpoint="checkpoint.pth")
config = load_checkpoint("checkpoint.pth")["config"]
~~~

Read checkpoint files with `load_checkpoint` rather than `torch.load`; it maps
tensors to CPU and resolves objects saved under former module paths.

Or construct a complete model from a configuration:

~~~python
import json
from dptb.nn import build_model

with open("input.json", encoding="utf-8") as handle:
    config = json.load(handle)

model = build_model(
    model_options=config["model_options"],
    common_options=config["common_options"],
)
~~~

Construct a single embedding with `Embedding(method, **options)`. Direct
embedding construction does not include the prediction head or dataset pipeline.

~~~python
from dptb.nn.embedding import Embedding

embedding = Embedding("unitb", **embedding_options)
~~~

## Prepare graph data

`AtomicData` represents an atomic structure and its graph. Build it from ASE
atoms or explicit positions:

~~~python
from ase import Atoms
from dptb.data import AtomicData

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
graph = AtomicData.from_ase(atoms=atoms, r_max=2.0)
~~~

The graph cutoff, atom types, basis, dtype, and device must match the model's
data contract. SK models may also require `er_max` and `oer_max`.
`AtomicData.from_points` accepts explicit positions and atomic numbers.
For stored datasets, use `dptb.data.build_dataset` with the data options and
basis from the training configuration; index the resulting dataset to get a
graph.

Prior-aware models require explicit, correctly aligned prior fields. Charge
heads also require physical overlap. Supply those fields before calling the
model; a geometry-only graph does not satisfy every model's input contract.
See [prior inputs](../advanced/prior_inputs.md) and [UniTB](../unitb.md).

## Predict operators

Convert a prepared graph to the tensor dictionary consumed by models:

~~~python
import torch
from dptb.data import AtomicData

model.eval()
with torch.no_grad():
    prediction = model(AtomicData.to_AtomicDataDict(graph))
~~~

The prediction representation depends on the configured model family and output
route. SK outputs contain Slater–Koster parameters; use `SKHamiltonian` when an
explicit conversion to AO features is needed. E3 models produce node/edge
operator features in the configured mapper representation. Overlap outputs are
available only when configured and must not be confused with latent overlap
features. For residual models, follow the configured prior reconstruction path
and add the prior back exactly once.

For compatible AO-feature output, convert to `(i,j,R) -> H_ij(R)` blocks:

~~~python
from dptb.data import feature_to_block

h_blocks = feature_to_block(data=prediction, idp=model.idp)
# Only when prediction contains physical overlap features:
s_blocks = feature_to_block(data=prediction, idp=model.idp, overlap=True)
~~~

These functions return dictionaries of real-space blocks.

## Electronic properties

`dptb.nn.HR2HK` converts real-space operators to reciprocal-space matrices.
`dptb.nn.Eigenvalues` computes spectra, and `dptb.postprocess.Band` evaluates
and plots band structures. See the [band guide](../advanced/elec_properties/band.md)
for model, structure, and k-path configuration.
