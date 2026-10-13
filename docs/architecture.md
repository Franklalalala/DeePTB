# Architecture

DeePTB has one construction API for complete models and one registry for
embeddings. Components own their numerical implementations; compatibility
imports resolve to those implementations without maintaining source copies.

## Ownership

| Location | Responsibility |
| --- | --- |
| `dptb/nn/build.py` | `build_model`: model-family selection, configuration, and checkpoint restoration |
| `dptb/nn/ensemble.py` | Distance-expert composition and deterministic expert initialization |
| `dptb/nn/embedding/registry.py` | Lazy built-in catalogue, discovery, construction, and extension registration |
| `dptb/nn/embedding/unitb/` | UniTB: `model.py` public class, backbone, layers, routing, and options |
| `dptb/nn/embedding/legacy/` | Historical MoE methods and their distinct mathematical contracts |
| `dptb/nn/embedding/{lem,slem,lem_prior,slem_prior}.py` | Upstream and prior-aware baselines |
| `dptb/nn/conditioning/` | Prior projection/input adapters, time conditioning, and edge memory |
| `dptb/nn/heads/` | Output-route selection and AO-block/RME output heads |
| `dptb/nn/projectors/` | Shared AO, Cartesian, and ICT projector implementations |
| `dptb/nn/pdq_moe.py` | Shared expert banks and routing metadata |
| `dptb/nn/so2_backend.py` | Optional SO2CUDA dispatch and PyTorch fallback |
| `dptb/data/` | Graphs, LMDB, sidecars, AO ordering, and feature conversion |
| `dptb/nacf/` | NACF tables, preparation, packing, and prior inference |
| `dptb/priors/h0/` | Physical H0 reconstruction and explicit offline preparation |
| `dptb/nnops/`, `dptb/entrypoints/` | Training, evaluation, and CLI orchestration |
| `dptb/postprocess/` | Electronic-structure postprocessing |

SO2CUDA and LoopSCF are separate repositories. DeePTB owns the backend interface
and model/data contracts they consume.

## Discover and construct

List embedding method names without constructing models:

~~~bash
dptb models
dptb models --all --json
~~~

The default list excludes historical MoE names; `--all` includes them.
Python provides the same catalogue:

~~~python
from dptb.nn import build_embedding, build_model, list_embeddings

methods = list_embeddings()
all_methods = list_embeddings(include_legacy=True)
~~~

Use `build_model(model_options=..., common_options=...)` for a complete model,
or `build_model(checkpoint=...)` to restore one. The existing
`dptb.nn.build.build_model` import remains supported.
`build_embedding(method, **options)` creates only an embedding and forwards
constructor options; it does not supply prediction heads or a data pipeline.

Complete-model construction selects the SK, DFTB-SK, mixed, or embedding-based
family. Embedding-based models normalize options, construct an `OrbitalMapper`,
resolve the selected embedding lazily, attach prediction/response heads, and
apply the configured operator conversion. Distance-expert composition is handled
by the ensemble module. Checkpoint loading restores saved configurations and
parameter state rather than treating package moves as new model variants.

The normal training CLI applies across model families:

~~~bash
dptb train input.json -o output
dptb multi-train input.json -o output
~~~

Input configurations select the family and training recipe. See the
[configuration guide](quick_start/input.md) and [UniTB examples](../examples/unitb/README.md).

H0's offline lifecycle uses the same CLI namespace:

~~~bash
dptb h0 --help
dptb h0 reconstruct --help
dptb h0 precompile --help
dptb h0 prepare-tables --help
dptb h0 compact-tables --help
dptb h0 accept --help
~~~

Use `H0_WORK_ROOT` for external work assets; `H0_OFFLINE_TABLE_DIR` optionally
selects the offline table directory. Preparation and compilation are explicit.
See [the H0 guide](../dptb/priors/h0/README.md) for configuration and commands.

## Import migration

Prefer canonical public APIs for new code. Historical imports remain
compatibility entry points, implemented by exact module aliases or retained
exports.

| Historical location | Canonical owner |
| --- | --- |
| `dptb.nn.embedding.emb` | `dptb.nn.embedding.registry` |
| `dptb.nn.embedding.unitb` public class | `dptb.nn.embedding.unitb.model`; public `UniTB` export retained |
| `dptb.nn.embedding.unitb_*` internals | Matching modules under `dptb.nn.embedding.unitb` |
| `dptb.nn.embedding.lem_moe_v3*` | Matching modules under `dptb.nn.embedding.legacy` |
| `dptb.nn.embedding.prior_common` | `dptb.nn.conditioning.projection` |
| `dptb.nn.embedding.prior_inputs` | `dptb.nn.conditioning.inputs` |
| `dptb.nn.embedding.flow_time` | `dptb.nn.conditioning.time` |
| `dptb.nn.embedding.soft_edge_memory` | `dptb.nn.conditioning.memory` |
| `dptb.nn.embedding.output_routes` | `dptb.nn.heads.routes` |
| Former embedding output-head modules | `dptb.nn.heads` |
| Former embedding AO/Cartesian projector modules | `dptb.nn.projectors` |
| `dptb.nn.build.DistanceEnsembleWrapper` | `dptb.nn.ensemble.DistanceEnsembleWrapper` |
| `h0rebuild` | `dptb.priors.h0` |
| `h0/tests_h0fast` | `dptb/tests/h0` |

The module move preserves configuration method names and `state_dict` paths.
Different topologies and expert parameterizations remain distinct.
Validate strict loading, inference, and deterministic short training for affected
production models as described in [TESTING.md](../TESTING.md).

## Register an extension

Import the extension in each process before construction, then register its
constructor under a unique method name:

~~~python
from dptb.nn import build_embedding, list_embeddings
from dptb.nn.embedding import Embedding
from my_package import MyEmbedding

Embedding.register("my_embedding")(MyEmbedding)
embedding = build_embedding("my_embedding", **options)
assert "my_embedding" in list_embeddings()
~~~

The decorator form `@Embedding.register("my_embedding")` is equivalent.
Registration forwards options to the constructor; it does not provide automatic
package discovery. A complete-model extension must implement the graph,
feature-dimension/irrep, output-route, device, and dtype contracts consumed by
its prediction path. For CLI use, add options to the configuration schema and
import the registration before configuration/model construction. Include
targeted interface and checkpoint tests.
