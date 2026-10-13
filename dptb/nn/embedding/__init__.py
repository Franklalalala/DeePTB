"""Lazy public embedding interfaces and exact legacy module aliases."""
from importlib import import_module

from dptb._compat import install_module_aliases
from .registry import EMBEDDING_CATALOGUE, Embedding, build_embedding, list_embeddings

# Retired imports resolve to their canonical module objects, preserving patches
# and checkpoint class identities without importing any model at discovery time.
_LEGACY_MODULES = {
    'dptb.nn.embedding.emb': 'dptb.nn.embedding.registry',
    'dptb.nn.embedding.unitb_backbone': 'dptb.nn.embedding.unitb.backbone',
    'dptb.nn.embedding.unitb_layers': 'dptb.nn.embedding.unitb.layers',
    'dptb.nn.embedding.unitb_ops': 'dptb.nn.embedding.unitb.ops',
    'dptb.nn.embedding.unitb_router': 'dptb.nn.embedding.unitb.router',
    'dptb.nn.embedding.unitb_router_inputs': 'dptb.nn.embedding.unitb.router_inputs',
    'dptb.nn.embedding.unitb_prior': 'dptb.nn.embedding.unitb.prior',
    'dptb.nn.embedding.unitb_options': 'dptb.nn.embedding.unitb.options',
    'dptb.nn.embedding.unitb_structure': 'dptb.nn.embedding.unitb.structure',
    'dptb.nn.embedding.unitb_edge': 'dptb.nn.embedding.unitb.edge',
    'dptb.nn.embedding.unitb_activations': 'dptb.nn.embedding.unitb.activations',
    'dptb.nn.embedding.eqv3_grid_helpers': 'dptb.nn.embedding.unitb.normalization',
    'dptb.nn.embedding.lem_moe_v3': 'dptb.nn.embedding.legacy.graph',
    'dptb.nn.embedding.lem_moe_v3_edge': 'dptb.nn.embedding.legacy.edge',
    'dptb.nn.embedding.lem_moe_v3_h0': 'dptb.nn.embedding.legacy.h0',
    'dptb.nn.embedding.lem_moe_v3_prior': 'dptb.nn.embedding.legacy.prior',
    'dptb.nn.embedding.lem_moe_v3_prior_2b': 'dptb.nn.embedding.legacy.prior_2b',
    'dptb.nn.embedding.output_routes': 'dptb.nn.heads.routes',
    'dptb.nn.embedding.block_native_head': 'dptb.nn.heads.block_native',
    'dptb.nn.embedding.ao_angular_projector': 'dptb.nn.heads.angular',
    'dptb.nn.embedding.late_block_expansion_cg': 'dptb.nn.heads.block_expansion',
    'dptb.nn.embedding.late_block_cartesian_projector': 'dptb.nn.heads.block_cartesian',
    'dptb.nn.embedding.late_rme_expansion_nocg': 'dptb.nn.heads.rme_expansion',
    'dptb.nn.embedding.late_rme_cartesian_hybrid': 'dptb.nn.heads.rme_cartesian',
    'dptb.nn.embedding.rme_nocg_fusion_head': 'dptb.nn.heads.rme_fusion',
    'dptb.nn.embedding.ao_projector_bank': 'dptb.nn.projectors.ao',
    'dptb.nn.embedding.cartesian_projector': 'dptb.nn.projectors.cartesian',
    'dptb.nn.embedding.cartesian_ict_bank': 'dptb.nn.projectors.ict',
    'dptb.nn.embedding.prior_common': 'dptb.nn.conditioning.projection',
    'dptb.nn.embedding.prior_inputs': 'dptb.nn.conditioning.inputs',
    'dptb.nn.embedding.flow_time': 'dptb.nn.conditioning.time',
    'dptb.nn.embedding.soft_edge_memory': 'dptb.nn.conditioning.memory',
}
install_module_aliases(_LEGACY_MODULES)

_EXPORTS = {
    spec.constructor: (spec.module, spec.constructor)
    for spec in EMBEDDING_CATALOGUE.values()
    if spec.constructor != "build_prior_2b"
}
_EXPORTS["LemMoEV3Prior2b"] = (
    "dptb.nn.embedding.legacy.prior_2b", "LemMoEV3Prior2b"
)

__all__ = [
    "Embedding", "build_embedding", "list_embeddings",
    "UniTB", "Identity", "SE2Descriptor", "BASELINE", "MPNN", "E3DeePH",
    "E3BaseLineModel6", "Slem", "Lem", "LemPrior", "SlemPrior", "LemMoEV3",
    "LemMoEV3Edge", "LemMoEV3EdgeH0", "Trinity", "E3BaseLineModelNonLocal",
    "LemMoEV3H0", "LemMoEV3Prior", "LemMoEV3Prior2b",
]


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
    elif __name__ + "." + name in _LEGACY_MODULES:
        value = import_module(__name__ + "." + name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
