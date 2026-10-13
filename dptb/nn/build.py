"""Build any DeePTB model family from configuration or a checkpoint."""

import copy
import logging
import warnings

import torch

from dptb.configuration import migrate_legacy_checkpoint_model_options
from dptb.nn.deeptb import NNENV, MIX
from dptb.nn.nnsk import NNSK
from dptb.nn.dftbsk import DFTBSK
# Retained exports for downstream callers of the historical build module.
from dptb.nn.ensemble import DeterministicExpertSeed, DistanceEnsembleWrapper
from dptb.nnops.distance_expert_mask import clip_last_expert_range_from_options
from dptb.utils.tools import j_must_have, j_loader

log = logging.getLogger(__name__)


def _infer_num_xgrid_from_state_dict(state_dict: dict):
    if state_dict is None: return None
    for k, v in state_dict.items():
        if k.endswith("distance_param") and torch.is_tensor(v) and v.ndim >= 1: return int(v.shape[0])
    return None


def _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                            ref_state_dict=None):
    extra_kwargs = {}
    num_xgrid = _infer_num_xgrid_from_state_dict(ref_state_dict)
    if num_xgrid is not None:
        if init_dftbsk:
            extra_kwargs["num_xgrid"] = num_xgrid
        elif init_mixed and model_options.get("dftbsk") is not None:
            extra_kwargs["num_xgrid"] = num_xgrid

    if init_nnenv:
        return NNENV(**model_options, **common_options)
    elif init_nnsk:
        return NNSK(**model_options["nnsk"], **common_options)
    elif init_mixed:
        return MIX(**model_options, **common_options, **extra_kwargs)
    elif init_dftbsk:
        return DFTBSK(**model_options["dftbsk"], **common_options, **extra_kwargs)
    else:
        raise ValueError("Failed to determine model type.")


def _construct_single_model_from_reference(checkpoint, init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options,
                                           common_options):
    if init_nnenv:
        return NNENV.from_reference(checkpoint, **model_options, **common_options)
    elif init_nnsk:
        return NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
    elif init_mixed:
        return MIX.from_reference(checkpoint, **model_options, **common_options)
    elif init_dftbsk:
        return DFTBSK.from_reference(checkpoint, **model_options["dftbsk"], **common_options)
    else:
        raise ValueError("Failed to determine model type.")


def _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk, init_mixed, init_dftbsk,
                                     model_options, common_options, clip_last_expert_range=False, seed_offset=0):
    proto_state = prototype_model.state_dict()
    experts = [prototype_model]
    for i in range(1, len(distance_ranges)):
        with DeterministicExpertSeed(i + 1 + seed_offset):
            m = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                                        ref_state_dict=proto_state)
        m.load_state_dict(proto_state, strict=True)
        experts.append(m)
    return DistanceEnsembleWrapper(
        experts=experts,
        distance_ranges=distance_ranges,
        clip_last_expert_range=clip_last_expert_range,
    )


def _count_experts_in_state_dict(state_dict: dict):
    if state_dict is None: return 0
    ids = {int(k.split(".")[1]) for k in state_dict.keys() if
           k.startswith("experts.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()}
    return len(ids)


def _is_multi_expert_state_dict(state_dict: dict):
    return _count_experts_in_state_dict(state_dict) > 0


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


def _build_ensemble_from_wrapper_state(wrapper_state_dict, distance_ranges, init_nnenv, init_nnsk, init_mixed,
                                       init_dftbsk, model_options, common_options, clip_last_expert_range=False,
                                       seed_offset=0):
    ckpt_num_experts = _count_experts_in_state_dict(wrapper_state_dict)
    if ckpt_num_experts != len(distance_ranges):
        raise ValueError(f"Checkpoint has {ckpt_num_experts} experts, but requires {len(distance_ranges)}.")
    experts = []
    for i in range(len(distance_ranges)):
        with DeterministicExpertSeed(i + 1 + seed_offset):
            m = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                                        ref_state_dict=wrapper_state_dict)
        experts.append(m)
    model = DistanceEnsembleWrapper(
        experts=experts,
        distance_ranges=distance_ranges,
        clip_last_expert_range=clip_last_expert_range,
    )
    model.load_state_dict(wrapper_state_dict, strict=True)
    return model


# ======================================================================
# Model factory
# ======================================================================
def build_model(
        checkpoint: str = None,
        model_options: dict = {},
        common_options: dict = {},
        train_options: dict = {},
        no_check: bool = False,
        device: str = None,
):
    if checkpoint is not None:
        from_scratch = False
    else:
        from_scratch = True
        if not all((model_options, common_options)):
            logging.error(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")
            raise ValueError(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")

    init_nnenv = False
    init_nnsk = False
    init_mixed = False
    init_dftbsk = False
    ckpt_state_dict = None
    moe_init_metadata = None

    if not from_scratch:
        if checkpoint.split(".")[-1] == "json":
            ckptconfig = j_loader(checkpoint)
        else:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckptconfig = f['config']
            ckpt_state_dict = f.get("model_state_dict", None)
            moe_init_metadata = copy.deepcopy(f.get("moe_init"))
            del f

        checkpoint_model_options = migrate_legacy_checkpoint_model_options(
            ckptconfig["model_options"]
        )
        if len(model_options) == 0 or model_options == ckptconfig["model_options"]:
            model_options = checkpoint_model_options
        model_options = _maybe_preserve_legacy_baseline_checkpoint(
            model_options, ckpt_state_dict, checkpoint_options=checkpoint_model_options
        )

        if len(common_options) == 0:
            common_options = ckptconfig["common_options"]

        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})

        del ckptconfig

    model_options = _maybe_enable_legacy_swiglu_s2_compat(model_options, ckpt_state_dict)

    if model_options.get("dftbsk"):
        assert not model_options.get("nnsk"), "There should only be one of the dftbsk and nnsk in model_options."
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                log.error("The prediction method must be sktb for mix mode.")
                raise ValueError("The prediction method must be sktb for mix mode.")

            if not model_options['embedding']['method'] in ['se2']:
                log.error("The embedding method must be se2 for mix mode.")
                raise ValueError("The embedding method must be se2 for mix mode.")
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_dftbsk = True
        else:
            raise ValueError("Model_options are not set correctly!")

    elif model_options.get("nnsk"):
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                log.error("The prediction method must be sktb for mix mode.")
                raise ValueError("The prediction method must be sktb for mix mode.")

            if not model_options['embedding']['method'] in ['se2']:
                log.error("The embedding method must be se2 for mix mode.")
                raise ValueError("The embedding method must be se2 for mix mode.")

        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnsk = True
        else:
            raise ValueError("Model_options are not set correctly!")
    else:
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnenv = True
            if model_options["prediction"]['method'] == 'sktb':
                if not model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method must be se2 for sktb prediction in nnenv mode.")
                    raise ValueError("The embedding method must be se2 for sktb prediction in deeptb mode.")

            if model_options["prediction"]['method'] in {'e3tb', 'block_native'}:
                if model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method can not be se2 for e3tb/block_native prediction in deeptb mode.")
                    raise ValueError("The embedding method can not be se2 for e3tb/block_native prediction in deeptb mode.")
        else:
            raise ValueError("Model_options are not set correctly!")

        assert int(init_dftbsk) + int(init_mixed) + int(init_nnenv) + int(
            init_nnsk) == 1, "IF not sk, you can only choose one of the mixed, nnenv, dftb and nnsk options."

    if device:
        common_options.update({"device": device})

    distance_ranges = train_options.get("distance_ranges", None)
    use_distance_ensemble = distance_ranges is not None

    # Independent initialisation seeds (train_options.init_seed, default 0 = the historical fixed seeds 1, 2, ...):
    # every expert seed is shifted by 1000 * init_seed.  common_options.seed only controls data order.
    seed_offset = 1000 * int((train_options or {}).get("init_seed", 0) or 0)
    if use_distance_ensemble:
        log.info(f"Wrapping model with DistanceEnsembleWrapper ({len(distance_ranges)} experts)")
        clip_last = clip_last_expert_range_from_options(train_options)
        if from_scratch:
            with DeterministicExpertSeed(1 + seed_offset):
                prototype_model = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options,
                                                          common_options, ref_state_dict=None)
            model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                     init_mixed, init_dftbsk, model_options, common_options,
                                                     clip_last_expert_range=clip_last, seed_offset=seed_offset)
        else:
            if ckpt_state_dict is None:
                with DeterministicExpertSeed(1 + seed_offset):
                    prototype_model = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk,
                                                              model_options, common_options, ref_state_dict=None)
                model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                         init_mixed, init_dftbsk, model_options, common_options,
                                                         clip_last_expert_range=clip_last, seed_offset=seed_offset)
            elif _is_multi_expert_state_dict(ckpt_state_dict):
                model = _build_ensemble_from_wrapper_state(ckpt_state_dict, distance_ranges, init_nnenv, init_nnsk,
                                                           init_mixed, init_dftbsk, model_options, common_options,
                                                           clip_last_expert_range=clip_last, seed_offset=seed_offset)
            else:
                with DeterministicExpertSeed(1 + seed_offset):
                    prototype_model = _construct_single_model_from_reference(checkpoint, init_nnenv, init_nnsk,
                                                                             init_mixed, init_dftbsk, model_options,
                                                                             common_options)
                model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                         init_mixed, init_dftbsk, model_options, common_options,
                                                         clip_last_expert_range=clip_last, seed_offset=seed_offset)

    else:
        with DeterministicExpertSeed(1 + seed_offset):
            if from_scratch:
                if init_nnenv:
                    model = NNENV(**model_options, **common_options)
                elif init_nnsk:
                    model = NNSK(**model_options["nnsk"], **common_options)
                elif init_mixed:
                    model = MIX(**model_options, **common_options)
                elif init_dftbsk:
                    model = DFTBSK(**model_options["dftbsk"], **common_options)
                else:
                    model = None
            else:
                if init_nnenv:
                    model = NNENV.from_reference(checkpoint, **model_options, **common_options)
                elif init_nnsk:
                    model = NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
                elif init_mixed:
                    model = MIX.from_reference(checkpoint, **model_options, **common_options)
                elif init_dftbsk:
                    model = DFTBSK.from_reference(checkpoint, **model_options["dftbsk"], **common_options)
                else:
                    model = None

    structure_options = (model_options.get("embedding") or {}).get("structure_mole") or {}
    if from_scratch and structure_options.get("enabled", False):
        from dptb.nn.embedding.unitb.structure import initialize_fresh
        initialize_fresh(model, structure_options)



    if not no_check:
        for k, v in model.model_options.items():
            if k not in model_options:
                log.warning(f"The model options {k} is not defined in input model_options, set to {v}.")
            else:
                deep_dict_difference(k, v, model_options)

    model.to(model.device)
    if moe_init_metadata is not None:
        # Provenance only: loading or resuming never repeats dense upcycling.
        model.moe_init_metadata = moe_init_metadata

    return model


def deep_dict_difference(base_key, expected_value, model_options):
    target_dict = copy.deepcopy(model_options)
    if isinstance(expected_value, dict):
        for subk, subv in expected_value.items():
            if subk not in target_dict.get(base_key, {}):
                log.warning(
                    f"The model option {subk} in {base_key} is not defined in input model_options, set to {subv}.")
            else:
                target2 = copy.deepcopy(target_dict[base_key])
                deep_dict_difference(f"{subk}", subv, target2)
    else:
        if expected_value != target_dict[base_key]:
            log.warning(
                f"The model option {base_key} is set to {expected_value}, but in input it is {target_dict[base_key]}, make sure it it correct!")
