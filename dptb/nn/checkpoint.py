"""Read DeePTB checkpoints, including those written by earlier releases.

Every checkpoint read goes through :func:`load_checkpoint`. A checkpoint holds
``config`` and ``model_state_dict``; an object that an older tree pickled under
a module path that has since moved resolves to the current module. The option
adapters below keep historical models loading with their original behavior;
:func:`dptb.nn.build.build_model` applies them.
"""
import copy
import logging
import pickle
import types
import warnings

import torch

log = logging.getLogger(__name__)

# Module path in an earlier release -> current module path.
MOVED_MODULES = {
    "dptb.nn.embedding.emb": "dptb.nn.embedding.registry",
    "dptb.nn.embedding.lem_moe_v3": "dptb.nn.embedding.legacy.graph",
    "dptb.nn.embedding.lem_moe_v3_edge": "dptb.nn.embedding.legacy.edge",
    "dptb.nn.embedding.lem_moe_v3_h0": "dptb.nn.embedding.legacy.h0",
    "dptb.nn.embedding.lem_moe_v3_prior": "dptb.nn.embedding.legacy.prior",
    "dptb.nn.embedding.lem_moe_v3_prior_2b": "dptb.nn.embedding.legacy.prior_2b",
    "dptb.nn.embedding.unitb_activations": "dptb.nn.embedding.unitb.activations",
    "dptb.nn.embedding.unitb_backbone": "dptb.nn.embedding.unitb.backbone",
    "dptb.nn.embedding.unitb_edge": "dptb.nn.embedding.unitb.edge",
    "dptb.nn.embedding.unitb_layers": "dptb.nn.embedding.unitb.layers",
    "dptb.nn.embedding.eqv3_grid_helpers": "dptb.nn.embedding.unitb.normalization",
    "dptb.nn.embedding.unitb_ops": "dptb.nn.embedding.unitb.ops",
    "dptb.nn.embedding.unitb_options": "dptb.nn.embedding.unitb.options",
    "dptb.nn.embedding.unitb_prior": "dptb.nn.embedding.unitb.prior",
    "dptb.nn.embedding.unitb_router": "dptb.nn.embedding.unitb.router",
    "dptb.nn.embedding.unitb_router_inputs": "dptb.nn.embedding.unitb.router_inputs",
    "dptb.nn.embedding.unitb_structure": "dptb.nn.embedding.unitb.structure",
    "dptb.nn.embedding.prior_common": "dptb.nn.conditioning.projection",
    "dptb.nn.embedding.prior_inputs": "dptb.nn.conditioning.inputs",
    "dptb.nn.embedding.flow_time": "dptb.nn.conditioning.time",
    "dptb.nn.embedding.soft_edge_memory": "dptb.nn.conditioning.memory",
    "dptb.nn.embedding.output_routes": "dptb.nn.heads.routes",
    "dptb.nn.embedding.ao_angular_projector": "dptb.nn.heads.angular",
    "dptb.nn.embedding.late_block_cartesian_projector": "dptb.nn.heads.block_cartesian",
    "dptb.nn.embedding.late_block_expansion_cg": "dptb.nn.heads.block_expansion",
    "dptb.nn.embedding.block_native_head": "dptb.nn.heads.block_native",
    "dptb.nn.embedding.late_rme_cartesian_hybrid": "dptb.nn.heads.rme_cartesian",
    "dptb.nn.embedding.late_rme_expansion_nocg": "dptb.nn.heads.rme_expansion",
    "dptb.nn.embedding.rme_nocg_fusion_head": "dptb.nn.heads.rme_fusion",
    "dptb.nn.embedding.ao_projector_bank": "dptb.nn.projectors.ao",
    "dptb.nn.embedding.cartesian_projector": "dptb.nn.projectors.cartesian",
    "dptb.nn.embedding.cartesian_ict_bank": "dptb.nn.projectors.ict",
    "dptb.nn.charge_head": "dptb.nn.heads.charge",
    "dptb.nn.charge_response": "dptb.nn.heads.charge_response",
    "dptb.nn.response_shift_head": "dptb.nn.heads.response_shift",
    "dptb.nn.shift_head": "dptb.nn.heads.shift",
    "dptb.nn.shift_overlap": "dptb.nn.heads.shift_overlap",
    "dptb.nn.so2_activation_routes": "dptb.nn.so2_backend",
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(MOVED_MODULES.get(module, module), name)


_pickle = types.ModuleType("dptb_checkpoint_pickle")
_pickle.Unpickler = _Unpickler
_pickle.load = lambda file, **kwargs: _Unpickler(file, **kwargs).load()


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint (or any torch-saved training artifact) onto ``map_location``."""
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_pickle)


def _has_legacy_swiglu_s2_state(state_dict: dict) -> bool:
    if not state_dict:
        return False
    return any(".activation.mul." in key for key in state_dict.keys())


def _baseline_checkpoint_options_equal(checkpoint_options: dict, model_options: dict):
    """Recognize schema defaults and aliases without dropping caller overrides."""
    if model_options == checkpoint_options:
        return True

    # Use the same preprocessing and model schema as the training entrypoint.
    # Unknown options remain in the comparison rather than being trimmed away.
    from dptb.configuration import canonicalize_training_config
    from dptb.utils.argcheck import _resolve_legacy_baseline_methods, model_options as model_schema

    schema = model_schema()

    def normalized(options):
        config = _resolve_legacy_baseline_methods({"model_options": copy.deepcopy(options)})
        config = canonicalize_training_config(config, warn_deprecated=False)
        return schema.normalize_value(config["model_options"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return normalized(model_options) == normalized(checkpoint_options)


def _maybe_preserve_legacy_baseline_checkpoint(
        model_options: dict, state_dict: dict, *, checkpoint_options: dict = None):
    """Preserve geometry-only legacy baselines whose prior options were ignored.

    Apply only to unchanged checkpoint options, including equivalent aliases
    and schema defaults. The saved method determines legacy provenance; a new
    prior checkpoint with missing adapter weights retains strict state loading.
    """
    if state_dict is None or not model_options:
        return model_options
    saved_options = model_options if checkpoint_options is None else checkpoint_options
    embedding = saved_options.get("embedding")
    if not isinstance(embedding, dict) or embedding.get("method") not in {"lem", "slem"}:
        return model_options
    from dptb.nn.conditioning.inputs import PRIOR_INPUT_KEYS

    if not PRIOR_INPUT_KEYS.intersection(embedding):
        return model_options
    if any(".prior_inputs." in key or key.startswith("prior_inputs.") for key in state_dict):
        return model_options
    if not _baseline_checkpoint_options_equal(saved_options, model_options):
        return model_options

    # Equivalent normalized options can still alter incidental state shapes
    # (for example an injected scalar residual-update ratio). Reconstruct from
    # the saved options so historical constructor defaults remain intact.
    patched = copy.deepcopy(saved_options)
    options = patched["embedding"]
    method = embedding["method"]
    options["method"] = method + "_prior"
    options["h0_init_scope"] = "none"
    for key in ("use_h0_init", "use_h0_node_init", "use_h0_edge_init"):
        options.pop(key, None)
    warnings.warn(
        f"Legacy {method!r} checkpoint contains no prior adapter weights: the "
        "original embedding was geometry-only and historically ignored prior "
        f"options. Loading as {options['method']!r} with h0_init_scope='none' "
        "to preserve its learned behavior. Changed embedding overrides retain "
        "strict checkpoint loading.",
        FutureWarning,
        stacklevel=3,
    )
    return patched


def _maybe_enable_legacy_swiglu_s2_compat(model_options: dict, state_dict: dict):
    if not state_dict or not model_options:
        return model_options

    embedding = model_options.get("embedding", None)
    if not isinstance(embedding, dict):
        return model_options
    if embedding.get("method") not in {
        "lem_moe_v3",
        "lem_moe_v3_h0",
        "lem_moe_v3_prior",
        "lem_moe_v3_prior_2b",
    }:
        return model_options
    if embedding.get("swiglu_s2_compat_mode", "modern") != "modern":
        return model_options
    if not _has_legacy_swiglu_s2_state(state_dict):
        return model_options

    patched = copy.deepcopy(model_options)
    patched.setdefault("embedding", {})
    patched["embedding"]["swiglu_s2_compat_mode"] = "legacy_uniform_only"
    log.warning(
        "Detected legacy flat SwiGLU-S2 checkpoint layout; forcing "
        "embedding.swiglu_s2_compat_mode='legacy_uniform_only' for compatibility."
    )
    return patched
