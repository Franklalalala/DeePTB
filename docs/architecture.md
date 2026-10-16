# Architecture

## Layout

| Location | Responsibility |
| --- | --- |
| `dptb/nn/build.py` | `build_model`: model-family selection, configuration, and checkpoint restoration |
| `dptb/nn/checkpoint.py` | `load_checkpoint` and the adapters that keep checkpoints from earlier releases loading |
| `dptb/nn/ensemble.py` | Distance-expert composition and deterministic expert initialization |
| `dptb/nn/embedding/registry.py` | `Embedding`: method-name registry and factory |
| `dptb/nn/embedding/unitb/` | UniTB: `model.py` public class, backbone, layers, routing, and options |
| `dptb/nn/embedding/legacy/` | Historical `lem_moe_v3*` methods and their distinct mathematical contracts |
| `dptb/nn/embedding/{lem,slem,lem_prior,slem_prior}.py` | Upstream and prior-aware baselines |
| `dptb/nn/conditioning/` | Prior projection/input adapters, time conditioning, and edge memory |
| `dptb/nn/heads/` | Output-route selection, AO-block/RME output heads, and charge/potential shift heads |
| `dptb/nn/projectors/` | Shared AO, Cartesian, and ICT projector implementations |
| `dptb/nn/pdq_moe.py` | Shared expert banks and routing metadata |
| `dptb/nn/so2_backend.py` | Optional SO2CUDA dispatch and PyTorch fallback |
| `dptb/data/` | Graphs, LMDB, sidecars, AO ordering, and feature conversion |
| `dptb/prior/common/` | Operators shared by both priors ([prior pipelines](prior_pipeline.md)): neighbour enumeration, block keys and gauge, real harmonics, radial splines and the fused table kernel, two-centre integrals, three-centre sums (the nonlocal step) on FP64 tensor cores, the finish kernel, step timing and checks, units, spherical Bessel transforms, Gaunt, KB, ORB/UPF reading, XC, table store |
| `dptb/prior/gf/` | Flash GF prior on the device (topology, assembly, environment XC, onsite XC) and its table builders |
| `dptb/prior/h0/` | UniTB H0 prior on the device (pair bank, field, local and nonlocal parts), its pair-bank builder, and the h0rebuild data readers ([guide](h0.md)) |
| `dptb/prior/build.py` | `dptb prior build`: one entry point for every table family |
| `dptb/inference/` | Fused inference engine: graph, prior, features, both heads, H = prior + delta, S, residual on the device ([1015-stable](1015-stable.md)) |
| `dptb/nnops/`, `dptb/entrypoints/` | Training, evaluation, and CLI orchestration |
| `dptb/postprocess/` | Electronic-structure postprocessing |

SO2CUDA and LoopSCF are separate repositories. DeePTB owns the backend interface
and model/data contracts they consume.

## Construct models

`build_model(model_options=..., common_options=...)` builds a complete model;
`build_model(checkpoint=...)` restores one. It selects the SK, DFTB-SK, mixed,
or embedding-based family. Embedding-based models normalize options, construct
an `OrbitalMapper`, build the embedding named by `model_options.embedding.method`
through `Embedding(method, **options)`, attach prediction/response heads, and
apply the configured operator conversion. Checkpoint loading restores the saved
configuration and parameter state.

~~~bash
dptb train input.json -o output
dptb multi-train input.json -o output
dptb prior --help
~~~

Input configurations select the family and training recipe. See the
[configuration guide](quick_start/input.md) and [UniTB examples](../examples/unitb/README.md).

## Register an embedding

Importing `dptb.nn.embedding` registers every built-in method. An extension
registers its constructor under a unique method name before the model is built:

~~~python
import torch
from dptb.nn.embedding import Embedding

@Embedding.register("my_embedding")
class MyEmbedding(torch.nn.Module):
    ...
~~~

`Embedding("my_embedding", **options)` forwards the options to the constructor.
A complete-model extension must implement the graph, feature-dimension/irrep,
output-route, device, and dtype contracts consumed by its prediction path. For
CLI use, add its options to the configuration schema.

## Module moves in `1013-stable`

Configuration method names and `state_dict` keys are unchanged; only Python
import paths moved. Old paths are not kept as import aliases. Checkpoints from
earlier releases load unchanged: every checkpoint read goes through
`dptb.nn.checkpoint.load_checkpoint`, which also resolves objects pickled under
a former module path (`MOVED_MODULES`).

| Former location | Current location |
| --- | --- |
| `dptb.nn.embedding.emb` | `dptb.nn.embedding.registry` |
| `dptb.nn.embedding.unitb_*` | Matching modules under `dptb.nn.embedding.unitb` |
| `dptb.nn.embedding.eqv3_grid_helpers` | `dptb.nn.embedding.unitb.normalization` |
| `dptb.nn.embedding.lem_moe_v3*` | `dptb.nn.embedding.legacy.{graph,edge,h0,prior,prior_2b}` |
| `dptb.nn.embedding.prior_common`, `prior_inputs` | `dptb.nn.conditioning.projection`, `inputs` |
| `dptb.nn.embedding.flow_time`, `soft_edge_memory` | `dptb.nn.conditioning.time`, `memory` |
| `dptb.nn.embedding.output_routes` | `dptb.nn.heads.routes` |
| `dptb.nn.embedding.{late_*,ao_angular_projector,block_native_head,rme_nocg_fusion_head}` | `dptb.nn.heads.*` |
| `dptb.nn.embedding.{ao_projector_bank,cartesian_projector,cartesian_ict_bank}` | `dptb.nn.projectors.{ao,cartesian,ict}` |
| `dptb.nn.{charge_head,charge_response,response_shift_head,shift_head,shift_overlap}` | `dptb.nn.heads.{charge,charge_response,response_shift,shift,shift_overlap}` |
| `dptb.nn.so2_activation_routes` | `dptb.nn.so2_backend` |
| `h0/h0rebuild`, `h0/*.py` scripts | `dptb.prior.h0`, `dptb prior <command>` |
| `dptb.nacf.*` (1013-stable) | `dptb.prior.gf.*` ([module table](1015-stable.md)) |
| `dptb.priors.h0.*` (1013-stable) | `dptb.prior.h0.*` |
| `configs/route_*.yaml` | `examples/output_routes/` |
| `scripts/export_*_projector_bank.py` | `tools/` |
