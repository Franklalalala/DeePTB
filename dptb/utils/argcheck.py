from typing import List, Callable, Dict, Any, Union
from dargs import dargs, Argument, Variant, ArgumentEncoder
import logging
import math
import re
from numbers import Integral, Number


log = logging.getLogger(__name__)

nnsk_model_config_checklist = ['unit','skfunction-skformula']
nnsk_model_config_updatelist = ['sknetwork-sk_hop_nhidden', 'sknetwork-sk_onsite_nhidden', 'sknetwork-sk_soc_nhidden']
dptb_model_config_checklist = ['dptb-if_batch_normalized', 'dptb-hopping_net_type', 'dptb-soc_net_type', 'dptb-env_net_type', 'dptb-onsite_net_type', 'dptb-hopping_net_activation', 'dptb-soc_net_activation', 'dptb-env_net_activation', 'dptb-onsite_net_activation',
                        'dptb-hopping_net_neuron', 'dptb-env_net_neuron', 'dptb-soc_net_neuron', 'dptb-onsite_net_neuron', 'dptb-axis_neuron', 'skfunction-skformula', 'sknetwork-sk_onsite_nhidden',
                        'sknetwork-sk_hop_nhidden']

# set default values in case of plateau schedulers & update lr per step
def chk_avg_per_iter(jdata):
    if jdata["train_options"]["lr_scheduler"]["type"] in {"rop", "warmup_rop"} and jdata["train_options"]["update_lr_per_iter"]:
        avg_per_iter = True
    else:
        avg_per_iter = False

    return avg_per_iter

def gen_doc_train(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    co = common_options()
    tr = train_options()
    da = data_options()
    mo = model_options()
    ptr = []
    ptr.append(co.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(tr.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(da.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(mo.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    # ptr.insert(0, make_index(key_words))

    return "\n\n".join(ptr)


def gen_doc_run(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    rop = run_options()

    ptr = []
    ptr.append(rop.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    # ptr.insert(0, make_index(key_words))

    return "\n\n".join(ptr)


def gen_doc_setinfo(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sio = set_info_options()
    ptr = []
    ptr.append(sio.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    # ptr.insert(0, make_index(key_words))

    return "\n\n".join(ptr)


def common_options():
    doc_device = "The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`"
    doc_dtype = """The digital number's precison, choose among: 
                    Default: `float32`
                        - `float32`: indicating torch.float32
                        - `float64`: indicating torch.float64
                """

    doc_seed = "The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`"
    doc_basis = "The atomic orbitals used to construct the basis. e.p. {'A':['2s','2p','s*'],'B':'[3s','3p']}"
    doc_overlap = "Whether to calculate the overlap matrix. Default: False"
    doc_train_w_charge = "Whether to train with charge info. Default: False"
    doc_has_soc = "Whether to train with SOC. Default: False"
    doc_nextham_uureal_mask = (
        "Whether to expose the NextHAM SOC uu.real mask to dataset and loss "
        "helpers. Default: False"
    )
    doc_full_soc_prediction = (
        "Whether to predict the full SOC target space. When True, this "
        "overrides nextham_uureal_mask and restores all spin and real/imag "
        "SOC channels. Default: False"
    )
    doc_train_dip = "Whether to train the dipole moment tensor. Default: False"
    doc_train_polar = "Whether to train the polarizaty tensor. Default: False"
    doc_wave_align = "Whether to align the wavefunctions. Default: False"

    args = [
        Argument("basis", dict, optional=False, doc=doc_basis),
        Argument("overlap", bool, optional=True, default=False, doc=doc_overlap),
        Argument("train_polar", bool, optional=True, default=False, doc=doc_train_polar),
        Argument("wave_align", bool, optional=True, default=False, doc=doc_wave_align),
        Argument("train_dip", bool, optional=True, default=False, doc=doc_train_dip),
        Argument("train_w_charge", bool, optional=True, default=False, doc=doc_train_w_charge),
        Argument("has_soc", bool, optional=True, default=False, doc=doc_has_soc),
        Argument("nextham_uureal_mask", bool, optional=True, default=False, doc=doc_nextham_uureal_mask),
        Argument("full_soc_prediction", bool, optional=True, default=False, doc=doc_full_soc_prediction),
        Argument("device", str, optional = True, default="cpu", doc = doc_device),
        Argument("dtype", str, optional = True, default="float32", doc = doc_dtype),
        Argument("seed", int, optional=True, default=3982377700, doc=doc_seed),
    ]

    doc_common_options = ""

    return Argument("common_options", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_common_options)


def dynamic_batch_options():
    doc = (
        "Dynamic DeePTB block/edge batching. When enabled, batch_size remains "
        "the maximum number of samples per batch, while max_cost caps the total "
        "sample cost. The default mode is block, which uses raw Hamiltonian/H0 "
        "offsite block-key counts when available and falls back to edge counts. "
        "The edge mode uses edge counts and falls back to block counts. If "
        "max_cost is omitted and calibrate is true, DeePTB derives max_cost from "
        "the requested calibration quantile over fixed-size batches."
    )
    args = [
        Argument("enabled", bool, optional=True, default=False),
        Argument("mode", str, optional=True, default="block"),
        Argument("max_cost", [int, float, None], optional=True, default=None),
        Argument("max_edge", [int, float, None], optional=True, default=None),
        Argument("max_samples", [int, None], optional=True, default=None),
        Argument("min_samples", int, optional=True, default=2),
        Argument("calibrate", bool, optional=True, default=False),
        Argument("calibration_batches", int, optional=True, default=1000),
        Argument("calibration_quantile", (int, float), optional=True, default=0.95),
        Argument("bucket_size", int, optional=True, default=0),
        Argument("packing_strategy", str, optional=True, default="random_evict"),
        Argument("drop_last", bool, optional=True, default=False),
        Argument("drop_oversized", bool, optional=True, default=False),
        Argument("seed", [int, None], optional=True, default=None),
        Argument("num_steps", [int, None], optional=True, default=None),
        Argument("use_global_dist", bool, optional=True, default=False),
        Argument("oom_fallback", bool, optional=True, default=False),
        Argument("oom_shrink_factor", (int, float), optional=True, default=0.8),
    ]
    return Argument(
        "dynamic_batch",
        dict,
        optional=True,
        default={"enabled": False},
        sub_fields=args,
        sub_variants=[],
        doc=doc,
    )


def flow_options():
    doc = (
        "Trainer-side conditional flow matching for Hamiltonian prediction. "
        "When enabled, DeePTB replaces node_h0/edge_h0 by an interpolated "
        "Hamiltonian state H_t and trains the existing model to predict the "
        "clean target Hamiltonian, following a QHFlow2-style residual CFM path."
    )
    args = [
        Argument("enabled", bool, optional=True, default=False),
        Argument("objective", str, optional=True, default="cfm"),
        Argument("mode", str, optional=True, default="residual"),
        Argument("prior", str, optional=True, default="zero"),
        Argument("node_h0_key", str, optional=True, default="node_h0"),
        Argument("edge_h0_key", str, optional=True, default="edge_h0"),
        Argument("node_target_key", str, optional=True, default="node_features"),
        Argument("edge_target_key", str, optional=True, default="edge_features"),
        Argument("output_space", str, optional=True, default="rme"),
        Argument("state_space", str, optional=True, default=""),
        Argument("target_semantics", str, optional=True, default=""),
        Argument("block_input_adapter", str, optional=True, default=""),
        Argument("h0_condition_space", str, optional=True, default=""),
        Argument("block_export_final_full_h", bool, optional=True, default=False),
        Argument("block_ode", bool, optional=True, default=False),
        Argument("prediction_add_h0", bool, optional=True, default=False),
        Argument("time_conditioning_required", bool, optional=True, default=False),
        Argument("block_inverse_mode", str, optional=True, default="strict"),
        Argument("block_inverse_atol", [int, float, None], optional=True, default=None),
        Argument("strict_certification", str, optional=True, default="always"),
        Argument("node_output_key", str, optional=True, default="node_hamil_blocks"),
        Argument("edge_output_key", str, optional=True, default="edge_hamil_blocks"),
        Argument("node_block_target_key", str, optional=True, default="node_delta_hamil_blocks"),
        Argument("edge_block_target_key", str, optional=True, default="edge_delta_hamil_blocks"),
        Argument("node_block_shape_key", str, optional=True, default="node_delta_hamil_block_shape"),
        Argument("edge_block_shape_key", str, optional=True, default="edge_delta_hamil_block_shape"),
        Argument("flow_time_key", str, optional=True, default="flow_time"),
        Argument("flow_time_r_key", str, optional=True, default="flow_time_r"),
        Argument("flow_time_t_key", str, optional=True, default="flow_time_t"),
        Argument("flow_time_h_key", str, optional=True, default="flow_time_h"),
        Argument("meanflow", dict, optional=True, default={}),
        Argument("time_sampling", str, optional=True, default="uniform"),
        Argument("t_min", (int, float), optional=True, default=0.0),
        Argument("t_max", (int, float), optional=True, default=0.999),
        Argument("t0_probability", (int, float), optional=True, default=0.0),
        Argument("t_eps", (int, float), optional=True, default=1.0e-3),
        Argument("time_logit_mean", (int, float), optional=True, default=-0.4),
        Argument("time_logit_std", (int, float), optional=True, default=1.0),
        Argument("node_sigma", (int, float), optional=True, default=1.0),
        Argument("edge_sigma", (int, float), optional=True, default=1.0),
        Argument("residual_sigma_floor", (int, float), optional=True, default=1.0e-6),
        Argument("te_prior_sigma", (int, float), optional=True, default=1.0),
        Argument("te_prior_mode", str, optional=True, default="irrep"),
        Argument("te_prior_per_graph", bool, optional=True, default=True),
        Argument("te_prior_validation_seed", int, optional=True, default=None),
        Argument("prior_node_key", str, optional=True, default=""),
        Argument("prior_edge_key", str, optional=True, default=""),
        Argument("prior_key_prefixes", list, optional=True, default=[]),
        Argument("external_prior_strict", bool, optional=True, default=True),
        Argument("allow_complex_prior_real_projection", bool, optional=True, default=False),
        Argument("prior_skdata", str, optional=True, default=""),
        Argument("dftb_skdata", str, optional=True, default=""),
        Argument("dftb_prior_overlap", bool, optional=True, default=False),
        Argument("dftb_prior_strict", bool, optional=True, default=True),
        Argument("dftb_prior_require_geometry", bool, optional=True, default=True),
        Argument("physical_prior_fallback", str, optional=True, default="basis_onsite"),
        Argument("basis_onsite_scale", (int, float), optional=True, default=1.0),
        Argument("basis_onsite_missing_value", (int, float), optional=True, default=0.0),
        Argument("basis_onsite_edge_value", (int, float), optional=True, default=0.0),
        Argument("huckel_k", (int, float), optional=True, default=1.75),
        Argument("overlap_huckel_k", (int, float), optional=True, default=1.75),
        Argument("huckel_node_overlap_key", str, optional=True, default="node_overlap"),
        Argument("huckel_edge_overlap_key", str, optional=True, default="edge_overlap"),
        Argument("huckel_strict_overlap", bool, optional=True, default=True),
        Argument("huckel_strict_basis", bool, optional=True, default=True),
        Argument("huckel_edge_energy_fallback", (int, float), optional=True, default=0.0),
        Argument("huckel_edge_length_decay", (int, float), optional=True, default=0.0),
        # Hueckel v2: orbital-pair endpoint energies and offline scale calibration.
        Argument("huckel_energy_mode", str, optional=True, default="type_mean",
                 doc="'type_mean' (legacy: one scalar per edge) or 'orbital_pair' "
                     "(Wolfsberg-Helmholz 0.5*(eps_mu+eps_nu) per orbpair slice, "
                     "indexed by edge_type)."),
        Argument("huckel_scale_mode", str, optional=True, default="none",
                 doc="'none' | 'global' (multiply huckel_scale_global) | 'pair_block' "
                     "(per bond-type x orbpair-slice signed scales from prior_calibration)."),
        Argument("huckel_scale_global", (int, float), optional=True, default=1.0),
        Argument("huckel_edge_channel_scale", (str, list, int, float, type(None)),
                 optional=True, default=None,
                 doc="Manual scalar or per-edge-channel overlap-Hueckel scale. "
                     "Non-scalar vectors must be constant within each orbpair slice; "
                     "prefer huckel_scale_mode='pair_block' with prior_calibration for "
                     "fingerprinted train-fit calibration."),
        Argument("overlap_huckel_edge_channel_scale", (str, list, int, float, type(None)),
                 optional=True, default=None,
                 doc="Alias for huckel_edge_channel_scale."),
        Argument("prior_calibration", str, optional=True, default="",
                 doc="Path to a calibration artifact from tools/calibrate_huckel_scales.py "
                     "(edge_scale + node_table). Verified fail-closed against the idp basis."),
        Argument("basis_onsite_mode", str, optional=True, default="table",
                 doc="'table' (free-atom onsite DB diagonal) or 'calibrated' "
                     "(per-type onsite rows from prior_calibration's node_table)."),
        Argument("prior_node", str, optional=True, default="",
                 doc="Split prior: family for the node/onsite prior (basis_onsite / "
                     "overlap_huckel / external / dftbsk). Must be set together with prior_edge."),
        Argument("prior_edge", str, optional=True, default="",
                 doc="Split prior: family for the edge/hopping prior (e.g. 'external' with "
                     "prior_edge_key=edge_h0 for the hybrid H0-hopping oracle)."),
        Argument("haar_node_key", str, optional=True, default="haar_node_features"),
        Argument("haar_edge_key", str, optional=True, default="haar_edge_features"),
        Argument("haar_candidate_index", int, optional=True, default=-1),
        Argument("haar_dm_strict", bool, optional=True, default=True),
        Argument("physical_prior_jitter_sigma", (int, float), optional=True, default=0.0),
        Argument("physical_prior_jitter_reference_scale", bool, optional=True, default=True),
        Argument("physical_prior_jitter_edge_decay", (int, float), optional=True, default=0.0),
        Argument("prior_jitter_sigma", (int, float), optional=True, default=0.0),
        Argument("loss_type", str, optional=True, default="mse"),
        Argument("node_weight", (int, float), optional=True, default=1.0,
                 doc="Finite non-negative node loss multiplier. CFM global_elements "
                     "requires 1.0; use equal_components for node/edge multipliers."),
        Argument("edge_weight", (int, float), optional=True, default=1.0,
                 doc="Finite non-negative edge loss multiplier. CFM global_elements "
                     "requires 1.0; use equal_components for node/edge multipliers."),
        Argument("z_loss_coef", (int, float), optional=True, default=0.0),
        Argument("omit_time_scaling", bool, optional=True, default=True),
        Argument("endpoint_weight_power", (int, float), optional=True, default=0.0),
        Argument("endpoint_weight_cap", (int, float), optional=True, default=100.0),
        Argument("component_reduction", str, optional=True, default="global_elements",
                 doc="global_elements performs one true reduction over all valid elements "
                     "and is unit-weight-only for CFM; equal_components sums independently "
                     "reduced node/edge losses and applies node_weight/edge_weight."),
        Argument("validation_ode_steps", list, optional=True, default=[1, 3]),
        Argument("apply_to_reference", bool, optional=True, default=False),
        Argument("log_compatible_loss", bool, optional=True, default=True),
        Argument("log_validation_random_t_loss", bool, optional=True, default=True),
        Argument("log_validation_t0_loss", bool, optional=True, default=True),
        Argument("log_validation_flow_euler_loss", bool, optional=True, default=True),
        # Backward-compatible accepted keys. Flow logging always maps endpoint-compatible
        # metrics to legacy train/validation loss keys when flow is enabled.
        Argument("log_train_compatible_loss", bool, optional=True, default=True),
        Argument("log_validation_compatible_loss", bool, optional=True, default=True),
        Argument("compatible_loss_to_legacy_keys", bool, optional=True, default=True),
        Argument("overwrite_feature_keys", bool, optional=True, default=True),
        Argument("detach_interpolated_h0", bool, optional=True, default=True),
        Argument("warn_missing_h0", bool, optional=True, default=True),
        Argument("strict_h0", bool, optional=True, default=True),
    ]
    return Argument(
        "flow_options",
        dict,
        optional=True,
        default={"enabled": False},
        sub_fields=args,
        sub_variants=[],
        doc=doc,
    )


def validate_flow_loss_contract(data):
    """Fail early on ambiguous or non-finite flow component weighting."""
    train = dict(data.get("train_options", {}) or {})
    flow = dict(train.get("flow_options", {}) or {})
    node_weight = float(flow.get("node_weight", 1.0))
    edge_weight = float(flow.get("edge_weight", 1.0))
    for name, value in (("node_weight", node_weight), ("edge_weight", edge_weight)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"flow_options.{name} must be finite and non-negative, got {value!r}"
            )
    if node_weight == 0.0 and edge_weight == 0.0:
        raise ValueError("flow_options.node_weight and edge_weight may not both be zero")

    objective = str(flow.get("objective", flow.get("type", "cfm"))).lower().replace(
        "-", "_"
    )
    pixel_meanflow = objective in {
        "pixel_meanflow",
        "pixel_mean_flow",
        "pmf",
        "meanflow",
        "mean_flow",
    }
    reduction = str(flow.get("component_reduction", "global_elements")).lower()
    if (
        reduction == "global_elements"
        and not pixel_meanflow
        and (node_weight != 1.0 or edge_weight != 1.0)
    ):
        raise ValueError(
            "flow_options.component_reduction='global_elements' requires "
            "node_weight=edge_weight=1 for CFM; use equal_components for "
            "node/edge loss multipliers"
        )


def validate_block_ode_contract(data):
    """Cross-check the full model/data contract that a field schema cannot see."""
    train = dict(data.get("train_options", {}) or {})
    flow = dict(train.get("flow_options", {}) or {})
    output_space = str(flow.get("output_space", "rme")).lower().replace("-", "_")
    requested = bool(flow.get("block_ode", False)) or output_space in {
        "ao_block_ode",
        "block_ode",
        "ao_blocks_ode",
        "uureal_block_ode",
        "spatial_uureal_residual_block_ode",
    }
    if not requested:
        return
    if not bool(flow.get("enabled", False)):
        raise ValueError("block_ode requires train_options.flow_options.enabled=true")
    uureal_mode = output_space in {
        "uureal_block_ode", "spatial_uureal_residual_block_ode"
    }
    expected_output_space = "uureal_block_ode" if uureal_mode else "ao_block_ode"
    if output_space != expected_output_space or not bool(flow.get("block_ode", False)):
        raise ValueError(
            "block_ode is a distinct mode: set block_ode=true and "
            "output_space='ao_block_ode'; do not reuse the frozen ao_block adapter"
        )
    if str(flow.get("mode", "")).lower() != "residual":
        raise ValueError("block_ode v1 requires flow_options.mode='residual'")
    prior = str(flow.get("prior", "")).lower().replace("-", "_")
    if uureal_mode and prior != "zero":
        raise ValueError("uureal_block_ode requires prior='zero'")
    if prior not in {"zero", "projected_te"}:
        raise ValueError(
            "block_ode supports only prior='zero' or explicit prior='projected_te'"
        )
    projected_scales = None
    if prior == "projected_te":
        mode = str(flow.get("te_prior_mode", "irrep")).lower().replace("-", "_")
        if mode != "irrep":
            raise ValueError(
                "projected_te block_ode requires te_prior_mode='irrep'; typewise "
                "mode reads target residual scales"
            )
        raw_scales = {
            "node_sigma": flow.get("node_sigma", 1.0),
            "edge_sigma": flow.get("edge_sigma", 1.0),
            "te_prior_sigma": flow.get("te_prior_sigma", 1.0),
        }
        invalid = [
            name
            for name, value in raw_scales.items()
            if isinstance(value, bool)
            or not isinstance(value, Number)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "projected_te block_ode requires finite positive scales; "
                f"invalid options={invalid}"
            )
        projected_scales = {
            name: float(value) for name, value in raw_scales.items()
        }
        validation_seed = flow.get("te_prior_validation_seed", None)
        if (
            isinstance(validation_seed, bool)
            or not isinstance(validation_seed, int)
            or validation_seed < 0
            or validation_seed > (1 << 64) - 1
        ):
            raise ValueError(
                "projected_te block_ode requires an explicit integer "
                f"te_prior_validation_seed in [0, {(1 << 64) - 1}]"
            )
    semantics = str(flow.get("target_semantics", "")).lower().replace("-", "_")
    if semantics not in {"absolute_full_h", "residual_dh"}:
        raise ValueError("block_ode requires explicit absolute_full_h/residual_dh semantics")
    if uureal_mode and semantics != "residual_dh":
        raise ValueError("uureal_block_ode requires target_semantics='residual_dh'")
    if uureal_mode:
        # These three options are declarative contract markers (validated only;
        # runtime behavior is driven solely by output_space=uureal_block_ode).
        # Accept both the F4 canonical names and the authoritative V2 dataset
        # contract aliases so a config written to either vocabulary validates:
        #   state_space:       residual_ao_block   <-> nextham_uureal_delta_block
        #   h0_condition_space: compact_uureal_rme <-> nextham_uureal_rme
        #   block_input_adapter: direct_cg (shared)
        accepted_mode_options = {
            "state_space": ("residual_ao_block", "nextham_uureal_delta_block"),
            "block_input_adapter": ("direct_cg",),
            "h0_condition_space": ("compact_uureal_rme", "nextham_uureal_rme"),
        }
        for option, accepted in accepted_mode_options.items():
            if str(flow.get(option, "")).lower().replace("-", "_") not in accepted:
                raise ValueError(
                    f"uureal_block_ode requires flow_options.{option} in {accepted!r}"
                )
        if bool(flow.get("block_export_final_full_h", False)):
            raise ValueError(
                "uureal_block_ode keeps full-H export outside the ODE hot path"
            )
    target_fields = {
        "absolute_full_h": {
            "node_block_target_key": "node_full_hamil_target_blocks",
            "edge_block_target_key": "edge_full_hamil_target_blocks",
            "node_block_shape_key": "node_full_hamil_target_block_shape",
            "edge_block_shape_key": "edge_full_hamil_target_block_shape",
        },
        "residual_dh": {
            "node_block_target_key": "node_delta_hamil_blocks",
            "edge_block_target_key": "edge_delta_hamil_blocks",
            "node_block_shape_key": "node_delta_hamil_block_shape",
            "edge_block_shape_key": "edge_delta_hamil_block_shape",
        },
    }[semantics]
    for option, expected in target_fields.items():
        actual = flow.get(option)
        if actual != expected:
            raise ValueError(
                f"block_ode target_semantics={semantics!r} requires "
                f"flow_options.{option}={expected!r}; got {actual!r}"
            )
    if bool(flow.get("prediction_add_h0", False)):
        raise ValueError("block_ode requires flow_options.prediction_add_h0=false")
    if not bool(flow.get("time_conditioning_required", False)):
        raise ValueError("block_ode requires flow_options.time_conditioning_required=true")
    if not bool(flow.get("strict_h0", True)):
        raise ValueError(
            "block_ode requires flow_options.strict_h0=true; physical H0 may not "
            "fall back to a zero base"
        )
    if str(flow.get("block_inverse_mode", "strict")).lower() != "strict":
        raise ValueError("block_ode requires block_inverse_mode='strict'")
    strict_certification = str(
        flow.get("strict_certification", "always")
    ).strip().lower()
    if strict_certification not in {"always", "first_batch"} and re.fullmatch(
        r"every_n\(([1-9][0-9]*)\)", strict_certification
    ) is None:
        raise ValueError(
            "block_ode strict_certification must be 'always', 'first_batch', "
            "or 'every_n(N)' with N >= 1"
        )
    common = dict(data.get("common_options", {}) or {})
    configured_dtype = str(common.get("dtype", "float32")).lower().replace(
        "torch.", ""
    )
    inverse_atol_caps = {"float32": 2.0e-5, "float64": 1.0e-10}
    if configured_dtype not in inverse_atol_caps:
        raise ValueError(
            "block_ode requires common_options.dtype='float32' or 'float64'; "
            f"got {configured_dtype!r}"
        )
    if projected_scales is not None:
        representable = {
            "float32": (2.0 ** -149, (2.0 - 2.0 ** -23) * 2.0 ** 127),
            "float64": (2.0 ** -1074, (2.0 - 2.0 ** -52) * 2.0 ** 1023),
        }
        minimum, maximum = representable[configured_dtype]
        effective_scales = {
            "node_sigma*te_prior_sigma": projected_scales["node_sigma"]
            * projected_scales["te_prior_sigma"],
            "edge_sigma*te_prior_sigma": projected_scales["edge_sigma"]
            * projected_scales["te_prior_sigma"],
        }
        invalid_effective = [
            name
            for name, value in effective_scales.items()
            if not math.isfinite(value) or not minimum <= abs(value) <= maximum
        ]
        if invalid_effective:
            raise ValueError(
                "projected_te block_ode effective scales must be finite and "
                f"non-zero in {configured_dtype}; invalid products={invalid_effective}"
            )
    maximum_inverse_atol = inverse_atol_caps[configured_dtype]
    raw_steps = flow.get("validation_ode_steps", [])
    if not raw_steps or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw_steps
    ):
        raise ValueError("block_ode validation_ode_steps must contain integers drawn from [1, 3]")
    steps = {int(value) for value in raw_steps}
    if not steps or not steps.issubset({1, 3}):
        raise ValueError("block_ode validation_ode_steps must be a non-empty subset of [1, 3]")
    configured_atol = flow.get("block_inverse_atol", None)
    if configured_atol is not None:
        configured_atol = float(configured_atol)
        if not math.isfinite(configured_atol) or configured_atol < 0:
            raise ValueError("block_ode block_inverse_atol must be finite and non-negative")
        if configured_atol > maximum_inverse_atol:
            raise ValueError(
                "block_ode block_inverse_atol exceeds the certified "
                f"{configured_dtype} maximum {maximum_inverse_atol:.6g}; "
                f"got {configured_atol:.6g}"
            )

    if uureal_mode:
        if not bool(common.get("has_soc", False)):
            raise ValueError("uureal_block_ode requires common_options.has_soc=true")
        if not bool(common.get("nextham_uureal_mask", False)):
            raise ValueError("uureal_block_ode requires common_options.nextham_uureal_mask=true")
        if bool(common.get("full_soc_prediction", False)):
            raise ValueError("uureal_block_ode requires full_soc_prediction=false")
    elif bool(common.get("has_soc", False)):
        raise ValueError("block_ode v1 is non-SOC only; set common_options.has_soc=false")

    distance_ranges = train.get("distance_ranges", None)
    if distance_ranges is not None:
        valid_full_graph_range = (
            isinstance(distance_ranges, list)
            and len(distance_ranges) == 1
            and isinstance(distance_ranges[0], (list, tuple))
            and len(distance_ranges[0]) == 2
            and not isinstance(distance_ranges[0][0], bool)
            and isinstance(distance_ranges[0][0], Number)
            and math.isfinite(float(distance_ranges[0][0]))
            and float(distance_ranges[0][0]) <= 0.0
        )
        if not valid_full_graph_range:
            raise ValueError(
                "block_ode v1 cannot use distance-partitioned experts because "
                "H-B0 must predict every graph edge; omit distance_ranges or use "
                "one full-graph range whose lower bound is <= 0"
            )

    model = dict(data.get("model_options", {}) or {})
    prediction = dict(model.get("prediction", {}) or {})
    embedding = dict(model.get("embedding", {}) or {})
    if str(embedding.get("method", "")).lower() != "lem_moe_v3_h0":
        raise ValueError("block_ode requires embedding.method='lem_moe_v3_h0'")
    if str(embedding.get("output_route", "")).lower() != "h_b0":
        raise ValueError("block_ode requires embedding.output_route='h_b0'")
    if not bool(embedding.get("require_full_block_edge_coverage", False)):
        raise ValueError(
            "block_ode requires embedding.require_full_block_edge_coverage=true"
        )
    if str(prediction.get("method", "")).lower() != "block_native":
        raise ValueError("block_ode requires prediction.method='block_native'")
    if str(prediction.get("block_decoder", "")).lower() != "expansion_cg":
        raise ValueError("block_ode requires prediction.block_decoder='expansion_cg'")
    if not bool(prediction.get("blockwise_hamiltonian", False)):
        raise ValueError("block_ode requires prediction.blockwise_hamiltonian=true")
    if bool(prediction.get("add_h0", False)):
        raise ValueError(
            "block_ode requires model_options.prediction.add_h0=false to prevent double add"
        )
    if not bool(embedding.get("use_flow_time_embedding", False)):
        raise ValueError("block_ode requires embedding.use_flow_time_embedding=true")
    if not bool(embedding.get("flow_time_condition_edges", False)):
        raise ValueError("block_ode requires flow-time conditioning on both nodes and edges")
    if bool(embedding.get("flow_time_allow_missing", True)):
        raise ValueError("block_ode requires embedding.flow_time_allow_missing=false")
    if str(embedding.get("flow_time_key", "flow_time")) != str(
        flow.get("flow_time_key", "flow_time")
    ):
        raise ValueError("block_ode flow and embedding flow_time_key values must match")
    if str(embedding.get("h0_merge_mode", "replace")).lower() != "replace":
        raise ValueError("block_ode requires embedding.h0_merge_mode='replace'")
    if not bool(embedding.get("use_h0_node_init", True)) or not bool(
        embedding.get("use_h0_edge_init", True)
    ):
        raise ValueError("block_ode requires both node and edge H0 initialization")
    if bool(embedding.get("use_uureal_residual_block_input", False)) != uureal_mode:
        raise ValueError(
            "uureal_block_ode and embedding.use_uureal_residual_block_input must be enabled together"
        )

    data_options_value = data.get("data_options", {}) or {}
    if not isinstance(data_options_value.get("train"), dict):
        raise ValueError("block_ode requires a configured data_options.train split")
    configured_splits = {
        split: data_options_value[split]
        for split in ("train", "validation", "reference", "test")
        if isinstance(data_options_value.get(split), dict)
    }
    expected_residual = semantics == "residual_dh" and not uureal_mode
    expected_full_h_target = semantics == "absolute_full_h"
    for split, split_options in configured_splits.items():
        path = f"data_options.{split}"
        dataset_type = str(split_options.get("type", "DefaultDataset"))
        if dataset_type != "LMDBDataset":
            raise ValueError(
                f"{path}.type must be 'LMDBDataset' for block_ode; "
                "other dataset backends do not implement the physical-H0 and "
                "versioned AO-block target contract"
            )
        if not bool(split_options.get("get_Hamiltonian", False)):
            raise ValueError(f"{path}.get_Hamiltonian must be true for block_ode")
        if not bool(split_options.get("get_H0", False)):
            raise ValueError(
                f"{path}.get_H0 must be true for block_ode physical-H0 initialization"
            )
        actual_residual = bool(split_options.get("residual_hamiltonian", False))
        if actual_residual != expected_residual:
            raise ValueError(
                f"{path}.residual_hamiltonian={actual_residual} conflicts with "
                f"block_ode target_semantics={semantics!r}"
            )
        actual_full_h_target = bool(
            split_options.get("require_full_h_target", False)
        )
        if actual_full_h_target != expected_full_h_target:
            raise ValueError(
                f"{path}.require_full_h_target must be "
                f"{str(expected_full_h_target).lower()} for block_ode "
                f"target_semantics={semantics!r}"
            )
        actual_residual_h_target = bool(
            split_options.get("require_residual_h_target", False)
        )
        if actual_residual_h_target != expected_residual:
            raise ValueError(
                f"{path}.require_residual_h_target must be "
                f"{str(expected_residual).lower()} for block_ode "
                f"target_semantics={semantics!r}"
            )
        actual_uureal = bool(split_options.get("require_uureal_block_ode", False))
        if actual_uureal != uureal_mode:
            raise ValueError(
                f"{path}.require_uureal_block_ode must be {str(uureal_mode).lower()}"
            )


def self_consistency_options():
    doc = (
        "WS4-C training-period self-consistency loss (see F:\\claude\\0702_nextham_dm_plan and "
        "dptb/nnops/self_consistency.py). Every `every_n_steps` steps, a `sample_frac` slice of the "
        "batch's predicted Hamiltonians is sent to an ABACUS restart_dh hrebuild endpoint "
        "(dptb.postprocess.hrebuild_server) for one-shot repair; the (stop-gradient) repaired H is used "
        "as a self-consistency target L_sc = ||H_pred - stopgrad(R(H_pred))||^2_masked "
        "(Zhang et al., ICML 2024). Off by default. Gap-threshold guard reuses hrebuild's red-line #3 "
        "refusal for near-metallic systems."
    )
    args = [
        Argument("enabled", bool, optional=True, default=False),
        Argument("endpoint", str, optional=True, default=""),
        Argument("sample_mode", str, optional=True, default="feature_tensors",
                 doc="`feature_tensors` submits node/edge tensors separately; `payload`/`atomic_data` submits the full post-forward AtomicDataDict so one repair can use structure, overlap, node, and edge context together."),
        Argument("tensor_keys", list, optional=True, default=["node_features", "edge_features"],
                 doc="Feature tensor keys used when assembling the self-consistency loss."),
        Argument("every_n_steps", int, optional=True, default=100),
        Argument("sample_frac", (int, float), optional=True, default=0.1),
        Argument("weight", (int, float), optional=True, default=0.1),
        Argument("warmup_epochs", int, optional=True, default=0),
        Argument("gap_threshold_ev", (int, float), optional=True, default=0.5),
        Argument("staleness_steps", int, optional=True, default=1,
                 doc="Double-buffering delay Delta: results requested at step k are consumed at step k+Delta."),
        Argument("consume_timeout", (int, float), optional=True, default=0.0,
                 doc="Seconds to wait for a due repair result before requeue/drop policy is applied."),
        Argument("max_workers", int, optional=True, default=2),
        Argument("retry_unfinished", bool, optional=True, default=True),
        Argument("mode", str, optional=True, default="one_shot"),
        Argument("unlabeled_pool_weight", (int, float), optional=True, default=0.0,
                 doc="Weight for the semi-supervised L_sc branch on unlabeled structures (0 disables it)."),
    ]
    return Argument(
        "self_consistency",
        dict,
        optional=True,
        default={"enabled": False},
        sub_fields=args,
        sub_variants=[],
        doc=doc,
    )


def activation_recompute_options():
    doc = (
        "Train-time activation recomputation/checkpointing for memory hot paths. "
        "Supported targets are lem_moe_v3_tp and lem_non_linear_expert_block. "
        "The nonlinear target checkpoints gather/cat, full expert TP, expert activation, "
        "and 0e post-activation mixing without changing state_dict keys."
    )
    args = [
        Argument("enabled", bool, optional=True, default=False),
        Argument(
            "targets",
            list,
            optional=True,
            default=["lem_moe_v3_tp", "lem_non_linear_expert_block"],
        ),
        Argument("checkpoint_node_tp", bool, optional=True, default=True),
        Argument("checkpoint_edge_tp", bool, optional=True, default=True),
        Argument("use_reentrant", bool, optional=True, default=False),
        Argument("preserve_rng_state", bool, optional=True, default=False),
    ]
    return Argument(
        "activation_recompute",
        dict,
        optional=True,
        default={"enabled": False},
        sub_fields=args,
        sub_variants=[],
        doc=doc,
    )


def train_options():
    doc_num_epoch = "Total number of training epochs. It is worth noted, if the model is reloaded with `-r` or `--restart` option, epoch which have been trained will counted from the time that the checkpoint is saved."
    doc_save_freq = "Frequency, or every how many iteration to saved the current model into checkpoints, The name of checkpoint is formulated as `latest|best_dptb|nnsk_b<bond_cutoff>_c<sk_cutoff>_w<sk_decay_w>`. Default: `10`"
    doc_validation_freq = "Frequency or every how many iteration to do model validation on validation datasets. Set 0 to disable iteration validation. Default: `10`"
    doc_validation_epoch_freq = "Frequency or every how many epochs to do model validation on validation datasets. Set 0 to disable epoch validation. Default: `1`"
    doc_display_freq = "Frequency, or every how many iteration to display the training log to screem. Default: `1`"
    doc_use_tensorboard = (
        "Set true to use tensorboard. It will record iteration error once every `25` iterations, "
        "epoch error once per epoch. There are tree types of error will be recorded. "
        "`train_loss_iter` is iteration loss, `train_loss_last` is the error of the last iteration in an epoch, "
        "`train_loss_mean` is the mean error of all iterations in an epoch. "
        "Learning rates are tracked as well. A folder named `tensorboard_logs` will be created in the working directory. "
        "Use `tensorboard --logdir=tensorboard_logs` to view the logs. Default: `False`"
    )

    doc_update_lr_per_iter = "Set true to update learning rate per-step. Default: `False`."
    doc_sliding_win_size = "Sliding window size for the average of the latest iterations' loss. Used for the reduce on plateau learning rate scheduler in case of the pairing of large dataset and small batch size. Default: `50`"
    doc_monitor_param_dynamics = (
        "Set true to enable lightweight parameter dynamics monitoring without forward/backward hooks. "
        "The monitor records sampled parameter update and gradient-flow metrics for key module groups."
    )
    doc_monitor_param_dynamics_freq = (
        "Parameter dynamics sampling interval in iterations. Use 0 to follow display_freq. Default: `0`."
    )
    doc_monitor_gated_edge_attention = (
        "Set true to record Fig.2-style diagnostics for gated edge aggregation: gate statistics, "
        "pre/post-gate sparsity, activation maxima, and top inbound-edge contribution share."
    )
    doc_monitor_gated_edge_attention_freq = (
        "Gated edge aggregation monitor sampling interval in iterations. Use 0 to follow display_freq. Default: `0`."
    )
    doc_monitor_gated_edge_attention_heatmap = (
        "Set true to save Fig.2-like query-key heatmap PNG/NPZ snapshots for gated edge aggregation. "
        "Rows are target/query nodes, columns are source/key nodes, and colors are normalized edge-message contribution mass."
    )
    doc_monitor_gated_edge_attention_heatmap_size = (
        "Maximum number of query and key nodes shown in gated edge aggregation heatmaps. Default: `64`."
    )
    doc_expert_lrs = (
        "Optional per-expert initial learning rates. "
        "If provided, it must be a list of floats with length == num_experts (len(distance_ranges)). "
        "expert_lrs[i] will override optimizer.lr when building optimizer for expert i. "
        "Default: [] (disabled, use optimizer.lr for all experts)."
    )
    doc_expert_optimizer_overrides = (
        "Optional per-expert optimizer override dictionaries. "
        "If provided, it should be a list with length == num_experts (len(distance_ranges)); "
        "a single item is broadcast to all experts, and identical legacy entries collapse for a single expert. "
        "Each element is merged into the shared `optimizer` config for the corresponding expert. "
        "Use `null` / `{}` to keep the shared optimizer config for one expert."
    )
    doc_expert_lr_scheduler_overrides = (
        "Optional per-expert learning-rate scheduler override dictionaries. "
        "If provided, it should be a list with length == num_experts (len(distance_ranges)); "
        "a single item is broadcast to all experts, and identical legacy entries collapse for a single expert. "
        "Each element is merged into the shared `lr_scheduler` config for the corresponding expert. "
        "This allows per-expert `patience`, `factor`, `min_lr`, or even scheduler `type`."
    )
    doc_optimizer = "\
        The optimizer setting for selecting the gradient optimizer of model training. Optimizer supported includes `Adam`, `AdamW`, `SGD` and `LBFGS` \n\n\
        For more information about these optmization algorithm, we refer to:\n\n\
        - `Adam`: [Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980)\n\n\
        - `AdamW`: [AdamW: Decoupled Weight Decay Regularization.](https://arxiv.org/abs/1711.05101)\n\n\
        - `SGD`: [Stochastic Gradient Descent.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)\n\n\
        - `LBFGS`: [On the limited memory BFGS method for large scale optimization.](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf) \n\n\
    "
    doc_lr_scheduler = "The learning rate scheduler tools settings, the lr scheduler is used to scales down the learning rate during the training process. Proper setting can make the training more stable and efficient. The supported lr schedular includes: `Exponential Decaying (exp)`, `Linear multiplication (linear)`, `Reduce on pleatau (rop)`, `Cyclic learning rate (cyclic)`. See more documentation on Pytorch. "

    doc_batch_size = (
        "The training batch size. In expert data parallel mode the default semantics are same-expert "
        "global batch, so the per-rank DataLoader batch is batch_size / expert_data_parallel_size. "
        "Default: `1`"
    )
    doc_ref_batch_size = (
        "The reference-data batch size. In expert data parallel mode the default semantics are local/per-rank "
        "so the common default value `1` remains valid when expert_data_parallel_size > 1. Default: `1`"
    )
    doc_val_batch_size = (
        "The validation batch size. In expert data parallel mode the default semantics are local/per-rank "
        "so the common default value `1` remains valid when expert_data_parallel_size > 1. Default: `1`"
    )
    doc_max_ckpt = "The maximum number of saved checkpoints, Default: `4`"
    doc_distance_ranges = "The ranges split for distance-based MoE / expert parallelism. Default: `[[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]`"

    # ================= 分布式 / DDP / expert-parallel =================
    doc_use_ddp = (
        "Set true to enable distributed expert-parallel training across multiple GPUs. "
        "When `distance_ranges` contains multiple experts, each rank will host one expert. "
        "Default: `False`"
    )
    doc_ddp_backend = "The backend used for distributed training. Usually `nccl` for GPUs and `gloo` for CPUs. Default: `nccl`"
    doc_ddp_master_addr = "Master node address for distributed communication. Default: `127.0.0.1`"
    doc_ddp_master_port = "Master node port for distributed communication. Default: `29501`"
    doc_ddp_timeout_sec = "Timeout in seconds for distributed process group operations. Default: `1800`"
    doc_expert_data_parallel_size = (
        "Number of same-expert replicas in distributed expert-parallel training. "
        "With two `distance_ranges` and `expert_data_parallel_size=2`, ranks 0/1 train "
        "expert 0 and ranks 2/3 train expert 1, synchronizing gradients only inside each "
        "same-expert group. Default: `1`"
    )

    doc_parallel_multi = (
        "Set true to start parallel training on CUDA streams in single-process multi-expert mode. "
        "This option is automatically disabled when `use_ddp=True`."
    )

    # ================= stitched / compatible loss & scheduler =================
    doc_log_single_model_compatible_loss = (
        "Deprecated compatibility switch. Split/flow trainers now always reconstruct and log "
        "endpoint-compatible legacy loss keys when the required packed stats are available. "
        "This is mainly for fair metric comparison between the split-expert model and the unsplit single model. "
        "Default: `True`"
    )
    doc_log_single_model_compatible_loss_mode = (
        "Reduction mode for reconstructing the compatible stitched loss. "
        "Currently the recommended mode is `reduce`. Default: `reduce`"
    )

    # ================= 轻量 debug tag =================
    doc_debug_tags = (
        "Set true to print stage-level timing logs for iteration, batch preparation, forward, backward, communication, "
        "scheduler and plugin stages. Useful for bottleneck diagnosis. Default: `False`"
    )
    doc_debug_tag_freq = "Print debug timing tags once every N iterations. Default: `1`"
    doc_debug_tag_cuda_mem = "Set true to record CUDA allocated/reserved/peak memory in debug stage logs. Default: `True`"
    doc_debug_tag_cuda_sync = (
        "Set true to call `torch.cuda.synchronize()` before measuring each stage. "
        "This makes timing more accurate but will slow training, so use it only for debugging. Default: `False`"
    )
    doc_debug_tag_reset_peak = (
        "Set true to reset CUDA peak counters at every debug tag boundary. "
        "When `monitor_cuda_memory=True`, the default is False so regular window-level peak memory remains valid. "
        "When `monitor_cuda_memory=False`, the default is True to preserve historical debug-tag behavior."
    )
    doc_debug_oom_dump = "Set true to dump detailed CUDA memory summary on OOM. Default: `True`"
    doc_monitor_cuda_memory = (
        "Set true to record CUDA allocated/reserved and peak allocated/reserved memory in regular "
        "iteration/epoch logs and TensorBoard. In distributed expert mode, per-rank values are gathered "
        "as expert_i_cuda_*_mb fields and global cuda_*_mb fields use the maximum across ranks. Default: `True`"
    )
    doc_monitor_cuda_cache_memory = (
        "Set true to log lightweight before/after CUDA memory deltas on persistent cache misses, "
        "including Wigner static tensors and cuEquivariance indexed_linear modules. "
        "This helps attribute stepwise memory jumps without enabling hook-heavy module tracing. "
        "Default: unset, which follows the DPTB_CUDA_CACHE_MEMORY_DIAG environment variable."
    )
    doc_monitor_cuda_cache_memory_sync = (
        "Set true to synchronize CUDA before cache-memory snapshots. More accurate but slower; "
        "default unset follows DPTB_CUDA_CACHE_MEMORY_SYNC."
    )
    doc_monitor_cuda_cache_memory_min_delta_mb = (
        "Only log cache-memory rows whose absolute allocated/reserved/peak/free delta is at least this many MiB. "
        "Default: `0`, log every probed cache miss."
    )
    doc_monitor_cuda_cache_events = (
        "Set true to log pure-Python persistent cache hit/miss events, including cuEq indexed_linear "
        "num_graphs keys. This does not query CUDA memory or synchronize. Default: unset, follows "
        "DPTB_CUDA_CACHE_EVENT_DIAG."
    )
    doc_monitor_cuda_cache_event_summary_interval = (
        "When cache event monitoring is enabled, log hit summaries every N events per cache key. "
        "Set 0 to log only misses. Default: `0`."
    )
    doc_monitor_cuda_module_memory = (
        "Set true to record CUDA memory snapshots around selected module forward/backward hooks. "
        "This is independent from monitor_flag, and currently targets SO2_Linear, MOLELinear, S2/FFN helpers, "
        "and non-TorchScript TensorProduct wrappers. Default: unset, follows monitor_flag for compatibility."
    )
    doc_monitor_cuda_module_memory_sync = (
        "Set true to synchronize CUDA before module-memory snapshots. More accurate but slower. Default: `False`"
    )
    doc_monitor_cuda_module_memory_min_delta_mb = (
        "Only write module-memory rows whose allocated/reserved/current peak delta is at least this many MiB. "
        "Use a positive threshold for long production runs to avoid very large CSV files. Default: `0`"
    )
    doc_sync_expert_dp_buffers = (
        "Set true to synchronize same-expert buffers after each expert data-parallel optimizer step. "
        "Disable only for throughput A/B when buffers are known not to affect training state. Default: `True`"
    )
    doc_expert_dp_grad_sync_mode = (
        "Same-expert data-parallel gradient synchronization implementation. "
        "`coalesced` uses torch.distributed.all_reduce_coalesced when available and falls back to flat buckets. "
        "`flat` always uses explicit flat buckets. Default: `coalesced`"
    )
    doc_expert_dp_backend = (
        "Same-expert data-parallel backend. `manual` uses DeePTB's explicit post-backward gradient sync; "
        "`ddp` wraps the local expert in torch.nn.parallel.DistributedDataParallel with the same-expert "
        "process group so gradient all-reduce can overlap with backward. Default: `manual`"
    )
    doc_expert_dp_use_ddp = (
        "Shortcut for setting expert_dp_backend to `ddp`. When true, expert_dp_backend is ignored. "
        "Default: `False`"
    )
    doc_expert_dp_batch_size_semantics = (
        "Legacy default for training batch size interpretation when expert_data_parallel_size > 1. "
        "`global` means same-expert global batch and automatically divides local DataLoader batch by "
        "expert_data_parallel_size; `local` preserves per-rank semantics. Default: `global`"
    )
    doc_expert_dp_train_batch_size_semantics = (
        "How batch_size is interpreted when expert_data_parallel_size > 1. Defaults to "
        "expert_dp_batch_size_semantics, normally `global`, to preserve fixed same-expert global batch."
    )
    doc_expert_dp_ref_batch_size_semantics = (
        "How ref_batch_size is interpreted when expert_data_parallel_size > 1. Default: `local`, so "
        "reference loaders keep their configured per-rank batch unless explicitly changed to `global`."
    )
    doc_expert_dp_val_batch_size_semantics = (
        "How val_batch_size is interpreted when expert_data_parallel_size > 1. Default: `local`, so "
        "validation loaders keep their configured per-rank batch unless explicitly changed to `global`."
    )
    doc_expert_dp_sampler_drop_last = (
        "Set true to make the corresponding same-expert DistributedSampler drop tail samples instead of "
        "padding duplicate indices. Default: `False` to preserve PyTorch DistributedSampler behavior."
    )
    doc_expert_dp_ddp_static_graph = (
        "Set true when expert DDP graphs are static, enabling DDP static_graph optimization. Default: `False`"
    )
    doc_expert_dp_ddp_gradient_as_bucket_view = (
        "Set true to let expert DDP gradients view all-reduce buckets and avoid extra bucket copies. Default: `False`"
    )
    doc_expert_dp_ddp_find_unused_parameters = (
        "Set true if DDP-wrapped expert forward can leave trainable parameters unused. Default: `True`"
    )
    doc_expert_dp_ddp_broadcast_buffers = (
        "Set true to let DDP broadcast expert buffers at forward start. Default: `False`; DeePTB keeps its "
        "post-step expert buffer sync path for manual parity. When true with `expert_dp_backend=ddp`, "
        "DDP owns buffer synchronization and DeePTB skips the post-step expert buffer sync."
    )
    doc_expert_dp_grad_check_mode = (
        "Same-expert data-parallel missing-gradient check mode. `auto` performs a safe tiny collective before "
        "the dense bucket reductions; `assume_dense` skips that check for static dense expert graphs. "
        "Use `assume_dense` only for throughput A/B after confirming every trainable expert parameter receives "
        "a dense gradient every iteration. Default: `auto`"
    )
    doc_expert_dp_grad_bucket_mb = (
        "Target same-expert data-parallel gradient bucket size in MiB. Default: `64`"
    )
    doc_expert_dp_ddp_bucket_cap_mb = (
        "DDP bucket_cap_mb for same-expert DDP backend. Leave unset to use PyTorch's default bucket size."
    )
    doc_expert_dp_buffer_sync_mode = (
        "Same-expert data-parallel buffer synchronization implementation. "
        "`coalesced` uses coalesced float-buffer all-reduce when available. Default: `coalesced`"
    )
    doc_expert_dp_buffer_bucket_mb = (
        "Target same-expert data-parallel float-buffer bucket size in MiB. Default: `64`"
    )

    # ================= profiler =================
    doc_debug_profile = (
        "Set true to enable PyTorch profiler for a selected iteration range and export Chrome trace json files. "
        "Useful for detailed CPU/CUDA/kernel timeline analysis. Default: `False`"
    )
    doc_debug_profile_start_iter = "The first iteration index to profile when `debug_profile=True`. Default: `5`"
    doc_debug_profile_end_iter = (
        "The last iteration index to profile when `debug_profile=True`. "
        "If equal to `debug_profile_start_iter`, only one iteration is profiled. Default: same as `debug_profile_start_iter`"
    )
    doc_debug_profile_dir = (
        "Output directory for profiler Chrome trace json files. "
        "If not set, a default local profile directory will be used."
    )

    # ================= 分布式 debug / NCCL debug =================
    doc_ddp_debug_detail = (
        "Set true to enable `TORCH_DISTRIBUTED_DEBUG=DETAIL`, which prints more detailed distributed runtime diagnostics. "
        "Default: `False`"
    )
    doc_nccl_debug = "Set true to enable `NCCL_DEBUG`. Default: `False`"
    doc_nccl_debug_level = "Debug level for NCCL when `nccl_debug=True`, e.g. `INFO` or `WARN`. Default: `INFO`"
    doc_cuda_launch_blocking = (
        "Set true to enable `CUDA_LAUNCH_BLOCKING=1` for easier debugging of asynchronous CUDA errors. "
        "This will significantly slow training and should NOT be used for performance benchmarking. Default: `False`"
    )
    doc_nccl_async_error_handling = "Set true to enable `NCCL_ASYNC_ERROR_HANDLING=1`. Recommended for distributed runs. Default: `True`"

    # ================= 运行时性能开关 =================
    doc_cudnn_benchmark = (
        "Set true to enable `torch.backends.cudnn.benchmark`, which may improve performance when input shapes are stable. "
        "Default: `False`"
    )
    doc_allow_tf32 = (
        "Set true to allow TF32 on supported NVIDIA GPUs for faster matrix operations with possible tiny numerical differences. "
        "Default: `True`"
    )
    doc_float32_matmul_precision = (
        "Precision policy for float32 matmul, passed to `torch.set_float32_matmul_precision`. "
        "Typical values are `highest`, `high`, `medium`. Empty string means keeping framework default."
    )
    doc_precompute_lem_active_edges = (
        "Precompute LEM/MoE-v3 active edge indices on the CPU batch before moving tensors to GPU. "
        "This avoids the CUDA nonzero used by InitLayer when the cutoff configuration is fixed."
    )
    doc_precompute_lem_cutoff_coeffs = (
        "Also precompute LEM/MoE-v3 cutoff coefficients before model forward. "
        "Default true for fixed-geometry Hamiltonian training; set false for force/stress/virial "
        "or other geometry-gradient training."
    )

    args = [
        Argument("num_epoch", int, optional=False, doc=doc_num_epoch),

        # expert / MoE split
        Argument("distance_ranges", list, optional=True, doc=doc_distance_ranges),
        Argument("parallel_multi", bool, optional=True, default=False, doc=doc_parallel_multi),

        # data / batch
        Argument("batch_size", int, optional=True, default=1, doc=doc_batch_size),
        dynamic_batch_options(),
        activation_recompute_options(),
        Argument("ref_batch_size", int, optional=True, default=1, doc=doc_ref_batch_size),
        Argument("val_batch_size", int, optional=True, default=1, doc=doc_val_batch_size),

        # training misc
        Argument("monitor_flag", bool, optional=True, default=False, doc='Set true to start monitor.'),
        Argument("monitor_param_dynamics", bool, optional=True, default=False, doc=doc_monitor_param_dynamics),
        Argument("monitor_param_dynamics_freq", int, optional=True, default=0, doc=doc_monitor_param_dynamics_freq),
        Argument("monitor_param_dynamics_tensorboard", bool, optional=True, default=None, doc="Write parameter dynamics curves to TensorBoard when the monitor is enabled. Default follows use_tensorboard."),
        Argument("monitor_param_dynamics_dead_patience", int, optional=True, default=3, doc="Number of consecutive no-gradient samples before marking a group as DEAD."),
        Argument("monitor_param_dynamics_delta_eps", float, optional=True, default=0.0, doc="Absolute element-change threshold used for delta_nonzero_fraction."),
        Argument("monitor_param_dynamics_grad_eps", float, optional=True, default=0.0, doc="Absolute gradient threshold used for grad_nonzero_fraction."),
        Argument("monitor_param_dynamics_delta_norm_dead_threshold", float, optional=True, default=1.0e-12, doc="Deprecated compatibility option. DEAD detection is gradient-norm based; delta metrics are diagnostic only."),
        Argument("monitor_param_dynamics_grad_norm_dead_threshold", float, optional=True, default=1.0e-12, doc="Gradient norm threshold used by parameter dynamics DEAD detection; groups below this value count as no-gradient."),
        Argument("monitor_gated_edge_attention", bool, optional=True, default=False, doc=doc_monitor_gated_edge_attention),
        Argument("monitor_gated_edge_attention_freq", int, optional=True, default=0, doc=doc_monitor_gated_edge_attention_freq),
        Argument("monitor_gated_edge_attention_tensorboard", bool, optional=True, default=None, doc="Write gated edge aggregation diagnostics to TensorBoard when the monitor is enabled. Default follows use_tensorboard."),
        Argument("monitor_gated_edge_attention_heatmap", bool, optional=True, default=False, doc=doc_monitor_gated_edge_attention_heatmap),
        Argument("monitor_gated_edge_attention_heatmap_size", int, optional=True, default=64, doc=doc_monitor_gated_edge_attention_heatmap_size),
        Argument("clip_grad", float, optional=True, default=1, doc='Gradient clipping max norm.'),
        Argument("valid_fast", bool, optional=True, default=True, doc="Set True to valid on the first batch of validation dataset, set False to valid the whole dataset. Default: `True`"),

        # optimizer / lr scheduler
        Argument("optimizer", dict, sub_fields=[], optional=True, default={}, sub_variants=[optimizer()], doc=doc_optimizer),
        Argument("lr_scheduler", dict, sub_fields=[], optional=True, default={}, sub_variants=[lr_scheduler()], doc=doc_lr_scheduler),
        Argument("update_lr_per_iter", bool, optional=True, default=False, doc=doc_update_lr_per_iter),
        Argument("sliding_win_size", int, optional=True, default=50, doc=doc_sliding_win_size),
        Argument("expert_lrs", list, optional=True, default=[], doc=doc_expert_lrs),
        Argument("expert_optimizer_overrides", list, optional=True, default=[], doc=doc_expert_optimizer_overrides),
        Argument("expert_lr_scheduler_overrides", list, optional=True, default=[], doc=doc_expert_lr_scheduler_overrides),
        # save / log
        Argument("save_freq", int, optional=True, default=10, doc=doc_save_freq),
        Argument("validation_freq", int, optional=True, default=10, doc=doc_validation_freq),
        Argument("validation_epoch_freq", int, optional=True, default=1, doc=doc_validation_epoch_freq),
        Argument("display_freq", int, optional=True, default=1, doc=doc_display_freq),
        Argument("use_tensorboard", bool, optional=True, default=False, doc=doc_use_tensorboard),
        Argument("max_ckpt", int, optional=True, default=4, doc=doc_max_ckpt),

        # distributed / DDP
        Argument("use_ddp", bool, optional=True, default=False, doc=doc_use_ddp),
        Argument("ddp_backend", str, optional=True, default="nccl", doc=doc_ddp_backend),
        Argument("ddp_master_addr", str, optional=True, default="127.0.0.1", doc=doc_ddp_master_addr),
        Argument("ddp_master_port", (str, int), optional=True, default=29501, doc=doc_ddp_master_port),
        Argument("ddp_timeout_sec", int, optional=True, default=1800, doc=doc_ddp_timeout_sec),
        Argument("expert_data_parallel_size", int, optional=True, default=1, doc=doc_expert_data_parallel_size),
        Argument("expert_dp_size", int, optional=True, default=1, doc=doc_expert_data_parallel_size),
        Argument("train_num_workers", int, optional=True, default=0,
                 doc="Number of DataLoader workers for train loader (implemented in MultiTrainer)."),
        Argument("ref_num_workers", int, optional=True, default=0,
                 doc="Number of DataLoader workers for reference loader (implemented in MultiTrainer)."),
        Argument("val_num_workers", int, optional=True, default=0,
                 doc="Number of DataLoader workers for validation loader (implemented in MultiTrainer)."),
        Argument("data_pin_memory", bool, optional=True, default=True,
                 doc="Enable pin_memory when rebuilding loaders in MultiTrainer."),
        Argument("data_persistent_workers", bool, optional=True, default=True,
                 doc="Enable persistent_workers when rebuilding loaders in MultiTrainer."),
        Argument("data_prefetch_factor", int, optional=True, default=2,
                 doc="Prefetch factor when rebuilding loaders in MultiTrainer."),
        Argument("distributed_rank0_prepare_batch", bool, optional=True, default=False,
                 doc="In distributed expert mode, only rank0 loads batch, performs CPU preprocessing + H2D + with_edge_vectors, then broadcasts packed GPU tensor groups to other ranks."),
        Argument("precompute_lem_active_edges", bool, optional=True, default=True, doc=doc_precompute_lem_active_edges),
        Argument("precompute_lem_cutoff_coeffs", bool, optional=True, default=True, doc=doc_precompute_lem_cutoff_coeffs),


        # stitched loss / scheduler behavior
        Argument("log_single_model_compatible_loss", bool, optional=True, default=True, doc=doc_log_single_model_compatible_loss),
        Argument("log_single_model_compatible_loss_mode", str, optional=True, default="reduce", doc=doc_log_single_model_compatible_loss_mode),

        # lightweight stage debug
        Argument("debug_tags", bool, optional=True, default=False, doc=doc_debug_tags),
        Argument("debug_tag_freq", int, optional=True, default=1, doc=doc_debug_tag_freq),
        Argument("debug_tag_cuda_mem", bool, optional=True, default=True, doc=doc_debug_tag_cuda_mem),
        Argument("debug_tag_cuda_sync", bool, optional=True, default=False, doc=doc_debug_tag_cuda_sync),
        Argument("debug_tag_reset_peak", bool, optional=True, default=None, doc=doc_debug_tag_reset_peak),
        Argument("debug_oom_dump", bool, optional=True, default=True, doc=doc_debug_oom_dump),
        Argument("monitor_cuda_memory", bool, optional=True, default=True, doc=doc_monitor_cuda_memory),
        Argument("monitor_cuda_cache_memory", bool, optional=True, default=None, doc=doc_monitor_cuda_cache_memory),
        Argument("monitor_cuda_cache_memory_sync", bool, optional=True, default=None, doc=doc_monitor_cuda_cache_memory_sync),
        Argument("monitor_cuda_cache_memory_min_delta_mb", (int, float), optional=True, default=0.0, doc=doc_monitor_cuda_cache_memory_min_delta_mb),
        Argument("monitor_cuda_cache_events", bool, optional=True, default=None, doc=doc_monitor_cuda_cache_events),
        Argument("monitor_cuda_cache_event_summary_interval", int, optional=True, default=0, doc=doc_monitor_cuda_cache_event_summary_interval),
        Argument("monitor_cuda_module_memory", bool, optional=True, default=None, doc=doc_monitor_cuda_module_memory),
        Argument("monitor_cuda_module_memory_sync", bool, optional=True, default=False, doc=doc_monitor_cuda_module_memory_sync),
        Argument("monitor_cuda_module_memory_min_delta_mb", (int, float), optional=True, default=0.0, doc=doc_monitor_cuda_module_memory_min_delta_mb),
        Argument("sync_expert_dp_buffers", bool, optional=True, default=True, doc=doc_sync_expert_dp_buffers),
        Argument("expert_dp_backend", str, optional=True, default="manual", doc=doc_expert_dp_backend),
        Argument("expert_dp_use_ddp", bool, optional=True, default=False, doc=doc_expert_dp_use_ddp),
        Argument("expert_dp_batch_size_semantics", str, optional=True, default="global", doc=doc_expert_dp_batch_size_semantics),
        Argument("expert_dp_train_batch_size_semantics", str, optional=True, default=None, doc=doc_expert_dp_train_batch_size_semantics),
        Argument("expert_dp_ref_batch_size_semantics", str, optional=True, default="local", doc=doc_expert_dp_ref_batch_size_semantics),
        Argument("expert_dp_val_batch_size_semantics", str, optional=True, default="local", doc=doc_expert_dp_val_batch_size_semantics),
        Argument("expert_dp_train_sampler_drop_last", bool, optional=True, default=False, doc=doc_expert_dp_sampler_drop_last),
        Argument("expert_dp_ref_sampler_drop_last", bool, optional=True, default=False, doc=doc_expert_dp_sampler_drop_last),
        Argument("expert_dp_val_sampler_drop_last", bool, optional=True, default=False, doc=doc_expert_dp_sampler_drop_last),
        Argument("expert_dp_ddp_static_graph", bool, optional=True, default=False, doc=doc_expert_dp_ddp_static_graph),
        Argument("expert_dp_ddp_gradient_as_bucket_view", bool, optional=True, default=False, doc=doc_expert_dp_ddp_gradient_as_bucket_view),
        Argument("expert_dp_ddp_find_unused_parameters", bool, optional=True, default=True, doc=doc_expert_dp_ddp_find_unused_parameters),
        Argument("expert_dp_ddp_broadcast_buffers", bool, optional=True, default=False, doc=doc_expert_dp_ddp_broadcast_buffers),
        Argument("expert_dp_ddp_bucket_cap_mb", (int, float), optional=True, default=None, doc=doc_expert_dp_ddp_bucket_cap_mb),
        Argument("expert_dp_grad_sync_mode", str, optional=True, default="coalesced", doc=doc_expert_dp_grad_sync_mode),
        Argument("expert_dp_grad_check_mode", str, optional=True, default="auto", doc=doc_expert_dp_grad_check_mode),
        Argument("expert_dp_grad_bucket_mb", (int, float), optional=True, default=64, doc=doc_expert_dp_grad_bucket_mb),
        Argument("expert_dp_buffer_sync_mode", str, optional=True, default="coalesced", doc=doc_expert_dp_buffer_sync_mode),
        Argument("expert_dp_buffer_bucket_mb", (int, float), optional=True, default=64, doc=doc_expert_dp_buffer_bucket_mb),

        # profiler
        Argument("debug_profile", bool, optional=True, default=False, doc=doc_debug_profile),
        Argument("debug_profile_start_iter", int, optional=True, default=5, doc=doc_debug_profile_start_iter),
        Argument("debug_profile_end_iter", int, optional=True, default=5, doc=doc_debug_profile_end_iter),
        Argument("debug_profile_dir", str, optional=True, default="", doc=doc_debug_profile_dir),

        # distributed debug env
        Argument("ddp_debug_detail", bool, optional=True, default=False, doc=doc_ddp_debug_detail),
        Argument("nccl_debug", bool, optional=True, default=False, doc=doc_nccl_debug),
        Argument("nccl_debug_level", str, optional=True, default="INFO", doc=doc_nccl_debug_level),
        Argument("cuda_launch_blocking", bool, optional=True, default=False, doc=doc_cuda_launch_blocking),
        Argument("nccl_async_error_handling", bool, optional=True, default=True, doc=doc_nccl_async_error_handling),

        # runtime performance tuning
        Argument("cudnn_benchmark", bool, optional=True, default=False, doc=doc_cudnn_benchmark),
        Argument("allow_tf32", bool, optional=True, default=True, doc=doc_allow_tf32),
        Argument("float32_matmul_precision", str, optional=True, default="", doc=doc_float32_matmul_precision),

        flow_options(),
        loss_options(),
        self_consistency_options(),
    ]

    doc_train_options = "Options that define the training behaviour of DeePTB, including optimizer/scheduler, expert split, distributed expert-parallel execution, debugging and profiling."

    return Argument("train_options", dict, sub_fields=args, sub_variants=[], optional=True, doc=doc_train_options)


def test_options():
    doc_display_freq = "Frequency, or every how many iteration to display the training log to screem. Default: `1`"
    doc_batch_size = "The batch size used in testing, Default: 1"
    doc_use_tensorboard = "Set true to write test loss and component loss scalars to TensorBoard. Default: `False`"

    args = [
        Argument("batch_size", int, optional=True, default=1, doc=doc_batch_size),
        Argument("display_freq", int, optional=True, default=1, doc=doc_display_freq),
        Argument("use_tensorboard", bool, optional=True, default=False, doc=doc_use_tensorboard),
        loss_options()
    ]

    doc_test_options = "Options that defines the testing behaviour of DeePTB."

    return Argument("test_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_test_options)


def Adam():
    doc_lr = "learning rate. Default: 1e-3"
    doc_betas = "coefficients used for computing running averages of gradient and its square Default: (0.9, 0.999)"
    doc_eps = "term added to the denominator to improve numerical stability, Default: 1e-8"
    doc_weight_decay = "weight decay (L2 penalty), Default: 0"
    doc_amsgrad = "whether to use the AMSGrad variant of this algorithm from the paper On the [Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) ,Default: False"

    return [
        Argument("lr", float, optional=True, default=1e-3, doc=doc_lr),
        Argument("betas", list, optional=True, default=[0.9, 0.999], doc=doc_betas),
        Argument("eps", float, optional=True, default=1e-8, doc=doc_eps),
        Argument("weight_decay", float, optional=True, default=0, doc=doc_weight_decay),
        Argument("amsgrad", bool, optional=True, default=False, doc=doc_amsgrad)
    ]

def HybridMuon():
    doc_lr = "learning rate. Default: 1e-3"
    doc_weight_decay = "decoupled weight decay for Muon and AdamW fallback parameters. Default: 1e-3"
    doc_muon_beta = "momentum coefficient for Muon-routed matrix parameters. Default: 0.95"
    doc_muon_scale = "DPA4 update-RMS matching scale gamma. Default: 0.18"
    doc_adam_betas = "Adam-family beta coefficients for vector/scalar parameters. Default: (0.9, 0.999)"
    doc_adam_eps = "Adam-family epsilon for vector/scalar parameters. Default: 1e-20"
    doc_matrix_min_dim = "Minimum trailing matrix dimension for Muon routing. Default: 2"
    doc_magma_lite = "Set true to enable DPA4 Magma-lite momentum-alignment damping for Muon blocks. Default: True"
    doc_magma_temperature = "Temperature for Magma-lite alignment sigmoid. DPA4 uses 2.0."
    doc_magma_ema_beta = "EMA coefficient for Magma-lite damping scores. DPA4 uses 0.9."
    doc_magma_min_scale = "Lower bound for Magma-lite damping scale. DPA4 uses 0.1."
    doc_muon_1d_route_mode = "Flattened 1D weight routing mode: auto/force/off. Default: auto"
    doc_muon_1d_include = "Name patterns eligible for automatic flattened 1D Muon routing."
    doc_muon_1d_exclude = "Name patterns excluded from flattened 1D Muon routing."
    doc_muon_1d_min_numel = "Minimum flattened 1D weight size for Muon routing. Default: 16"
    doc_muon_1d_max_aspect = "Maximum factorized matrix aspect ratio. Default: 64"
    doc_muon_1d_allow_degenerate = "Allow unfactorable 1D weights to use a 1 x N matrix. Default: False"
    doc_muon_force_patterns = "Optional name patterns that force eligible 1D tensors through Muon."
    doc_muon_clip = "Enable Muon update clipping. Default: True"
    doc_muon_clip_mode = "Muon clip mode: auto/fixed/rms/off. Default: auto"
    doc_muon_clip_rms = "Hard cap for scaled Muon update RMS. Default: 0.6"
    doc_muon_clip_auto_beta = "EMA beta for automatic relative-step clipping. Default: 0.98"
    doc_muon_clip_auto_mult = "EMA multiplier for automatic relative-step clipping. Default: 3.0"
    doc_muon_clip_auto_std_mult = "Standard-deviation multiplier for automatic clipping. Default: 2.0"
    doc_muon_clip_min_ratio = "Minimum relative-step cap. Default: 0.01"
    doc_muon_clip_max_ratio = "Maximum relative-step cap. Default: 0.25"
    doc_muon_clip_param_rms_floor = "Parameter RMS floor for relative-step clipping. Default: 1e-3"
    doc_muon_clip_warmup_steps = "Per-parameter steps before automatic clipping activates. Default: 5"

    return [
        Argument("lr", float, optional=True, default=1e-3, doc=doc_lr),
        Argument("weight_decay", float, optional=True, default=1e-3, doc=doc_weight_decay),
        Argument("muon_beta", float, optional=True, default=0.95, doc=doc_muon_beta),
        Argument("muon_scale", float, optional=True, default=0.18, doc=doc_muon_scale),
        Argument("adam_betas", list, optional=True, default=[0.9, 0.999], doc=doc_adam_betas),
        Argument("adam_eps", float, optional=True, default=1e-20, doc=doc_adam_eps),
        Argument("matrix_min_dim", int, optional=True, default=2, doc=doc_matrix_min_dim),
        Argument("magma_lite", bool, optional=True, default=True, doc=doc_magma_lite),
        Argument("magma_temperature", float, optional=True, default=2.0, doc=doc_magma_temperature),
        Argument("magma_ema_beta", float, optional=True, default=0.9, doc=doc_magma_ema_beta),
        Argument("magma_min_scale", float, optional=True, default=0.1, doc=doc_magma_min_scale),
        Argument("muon_1d_route_mode", str, optional=True, default="auto", doc=doc_muon_1d_route_mode),
        Argument("muon_1d_include_name_patterns", list, optional=True, default=["*weight*", "*tensor_product*"], doc=doc_muon_1d_include),
        Argument("muon_1d_exclude_name_patterns", list, optional=True, default=["*bias*", "*norm*", "*scale*", "*shift*", "*offset*", "*res_update*", "*bessel*", "*cutoff*", "*temperature*", "*freq*", "*router*"], doc=doc_muon_1d_exclude),
        Argument("muon_1d_min_numel", int, optional=True, default=16, doc=doc_muon_1d_min_numel),
        Argument("muon_1d_max_aspect_ratio", float, optional=True, default=64.0, doc=doc_muon_1d_max_aspect),
        Argument("muon_1d_allow_degenerate_matrix", bool, optional=True, default=False, doc=doc_muon_1d_allow_degenerate),
        Argument("muon_force_name_patterns", list, optional=True, default=[], doc=doc_muon_force_patterns),
        Argument("muon_clip", bool, optional=True, default=True, doc=doc_muon_clip),
        Argument("muon_clip_mode", str, optional=True, default="auto", doc=doc_muon_clip_mode),
        Argument("muon_clip_rms", float, optional=True, default=0.6, doc=doc_muon_clip_rms),
        Argument("muon_clip_auto_beta", float, optional=True, default=0.98, doc=doc_muon_clip_auto_beta),
        Argument("muon_clip_auto_mult", float, optional=True, default=3.0, doc=doc_muon_clip_auto_mult),
        Argument("muon_clip_auto_std_mult", float, optional=True, default=2.0, doc=doc_muon_clip_auto_std_mult),
        Argument("muon_clip_min_ratio", float, optional=True, default=0.01, doc=doc_muon_clip_min_ratio),
        Argument("muon_clip_max_ratio", float, optional=True, default=0.25, doc=doc_muon_clip_max_ratio),
        Argument("muon_clip_param_rms_floor", float, optional=True, default=1e-3, doc=doc_muon_clip_param_rms_floor),
        Argument("muon_clip_warmup_steps", int, optional=True, default=5, doc=doc_muon_clip_warmup_steps),
    ]

def SGD():
    doc_lr = "learning rate. Default: 1e-3"
    doc_weight_decay = "weight decay (L2 penalty), Default: 0"
    doc_momentum = "momentum factor Default: 0"
    doc_dampening = "dampening for momentum, Default: 0"
    doc_nesterov = "enables Nesterov momentum, Default: False"

    return [
        Argument("lr", float, optional=True, default=1e-3, doc=doc_lr),
        Argument("momentum", float, optional=True, default=0., doc=doc_momentum),
        Argument("weight_decay", float, optional=True, default=0., doc=doc_weight_decay),
        Argument("dampening", float, optional=True, default=0., doc=doc_dampening),
        Argument("nesterov", bool, optional=True, default=False, doc=doc_nesterov)
    ]


def RMSprop():
    doc_lr = "learning rate. Default: 1e-2"
    doc_alpha = "smoothing constant, Default: 0.99"
    doc_eps = "term added to the denominator to improve numerical stability, Default: 1e-8"
    doc_weight_decay = "weight decay (L2 penalty), Default: 0"
    doc_momentum = "momentum factor, Default: 0"
    doc_centered = "if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance, Default: False"

    return [
        Argument("lr", float, optional=True, default=1e-2, doc=doc_lr),
        Argument("alpha", float, optional=True, default=0.99, doc=doc_alpha),
        Argument("eps", float, optional=True, default=1e-8, doc=doc_eps),
        Argument("weight_decay", float, optional=True, default=0, doc=doc_weight_decay),
        Argument("momentum", float, optional=True, default=0, doc=doc_momentum),
        Argument("centered", bool, optional=True, default=False, doc=doc_centered)
    ]


def LBFGS():
    doc_lr = "learning rate. Default: 1"
    doc_max_iter = "maximal number of iterations per optimization step. Default: 20"
    doc_max_eval = "maximal number of function evaluations per optimization step. Default: None -> max_iter*1.25"
    # doc_tolerance_grad = "termination tolerance on first order optimality (default: 1e-7)."
    # doc_line_search_fn = "either 'strong_wolfe' or None (default: None)."
    # doc_history_size = "update history size. Default: 100"
    # doc_tolerance_change = "termination tolerance on function value/parameter changes (default: 1e-9)."

    return [
        Argument("lr", float, optional=True, default=1, doc=doc_lr),
        Argument("max_iter", int, optional=True, default=20, doc=doc_max_iter),
        Argument("max_eval", int, optional=True, default=None, doc=doc_max_eval)
    ]

def optimizer():
    doc_type = "select type of optimizer, support type includes: `Adam`, `AdamW`, `HybridMuon`, `SGD` and `LBFGS`. Default: `Adam`"

    return Variant("type", [
            Argument("Adam", dict, Adam()),
            Argument("AdamW", dict, Adam()),
            Argument("HybridMuon", dict, HybridMuon()),
            Argument("SGD", dict, SGD()),
            Argument("RMSprop", dict, RMSprop()),
            Argument("LBFGS", dict, LBFGS()),
        ],optional=True, default_tag="Adam", doc=doc_type)

def ExponentialLR():
    doc_gamma = "Multiplicative factor of learning rate decay."

    return [
        Argument("gamma", float, optional=True, default=0.999, doc=doc_gamma)
    ]

def LinearLR():
    doc_start_factor = "The number we multiply learning rate in the first epoch. \
        The multiplication factor changes towards end_factor in the following epochs. Default: 1./3."
    doc_end_factor = "The multiplication factor changes towards end_factor in the following epochs. Default: 1./3."
    doc_total_iters = "The number of iterations that multiplicative factor reaches to 1. Default: 5."

    return [
        Argument("start_factor", float, optional=True, default=0.3333333, doc=doc_start_factor),
        Argument("end_factor", float, optional=True, default=0.3333333, doc=doc_end_factor),
        Argument("total_iters", int, optional=True, default=5, doc=doc_total_iters)
    ]

def ReduceOnPlateau():
    doc_mode = "One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; \
        in max mode it will be reduced when the quantity monitored has stopped increasing. Default: 'min'."
    doc_factor = "Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1."
    doc_patience = "Number of epochs with no improvement after which learning rate will be reduced. For example, \
        if patience = 2, then we will ignore the first 2 epochs with no improvement, \
        and will only decrease the LR after the 3rd epoch if the loss still hasn't improved then. Default: 10."
    doc_threshold = "Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4."
    doc_threshold_mode = "One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or \
        best * ( 1 - threshold ) in min mode. In abs mode, \
        dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: 'rel'."
    doc_cooldown = "Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0."
    doc_min_lr = "A scalar or a list of scalars. \
        A lower bound on the learning rate of all param groups or each group respectively. Default: 0."
    doc_eps = "Minimal decay applied to lr. \
        If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8."

    return [
        Argument("mode", str, optional=True, default="min", doc=doc_mode),
        Argument("factor", float, optional=True, default=0.1, doc=doc_factor),
        Argument("patience", int, optional=True, default=10, doc=doc_patience),
        Argument("threshold", float, optional=True, default=1e-4, doc=doc_threshold),
        Argument("threshold_mode", str, optional=True, default="rel", doc=doc_threshold_mode),
        Argument("cooldown", int, optional=True, default=0, doc=doc_cooldown),
        Argument("min_lr", [float, list], optional=True, default=0, doc=doc_min_lr),
        Argument("eps", float, optional=True, default=1e-8, doc=doc_eps),
    ]

def CyclicLR():
    doc_base_lr = "Initial learning rate which is the lower boundary in the cycle for each parameter group."
    doc_max_lr = "Upper learning rate boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_lr - base_lr). The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore max_lr may not actually be reached depending on scaling function."
    doc_step_size_up = "Number of training iterations in the increasing half of a cycle. Default: 2000"
    doc_step_size_down = "Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is set to step_size_up. Default: None"
    doc_mode = "One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above. If scale_fn is not None, this argument is ignored. Default: 'triangular'"
    doc_gamma = "Constant in 'exp_range' scaling function: gamma**(cycle iterations) Default: 1.0"
    doc_scale_fn = "Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then 'mode' is ignored. Default: None"
    doc_scale_mode = "{'cycle', 'iterations'}. Defines whether scale_fn is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Default: 'cycle'"
    doc_cycle_momentum = "If True, momentum is cycled inversely to learning rate between 'base_momentum' and 'max_momentum'. Default: True"
    doc_base_momentum = "Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is 'max_momentum' and learning rate is 'base_lr'. Default: 0.8"
    doc_max_momentum = "Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum). The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude; therefore base_momentum may not actually be reached depending on scaling function. Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is 'max_momentum' and learning rate is 'base_lr'. Default: 0.9"
    doc_last_epoch = "The index of the last batch. This parameter is used when resuming a training job. Since step() should be invoked after each batch instead of after each epoch, this number represents the total number of batches computed, not the total number of epochs computed. When last_epoch=-1, the schedule is started from the beginning. Default: -1"
    doc_verbose = "If True, prints a message to stdout for each update. Default: False."

    return [
        Argument("base_lr", [float, list], optional=False, doc=doc_base_lr),
        Argument("max_lr", [float, list], optional=False, doc=doc_max_lr),
        Argument("step_size_up", int, optional=True, default=10, doc=doc_step_size_up),
        Argument("step_size_down", int, optional=True, default=40, doc=doc_step_size_down),
        Argument("mode", str, optional=True, default="exp_range", doc=doc_mode),
        Argument("gamma", float, optional=True, default=1.0, doc=doc_gamma),
        Argument("scale_fn", object, optional=True, default=None, doc=doc_scale_fn),
        Argument("scale_mode", str, optional=True, default="cycle", doc=doc_scale_mode),
        Argument("cycle_momentum", bool, optional=True, default=False, doc=doc_cycle_momentum),
        Argument("base_momentum", [float, list], optional=True, default=0.8, doc=doc_base_momentum),
        Argument("max_momentum", [float, list], optional=True, default=0.9, doc=doc_max_momentum),
        Argument("last_epoch", int, optional=True, default=-1, doc=doc_last_epoch),
        Argument("verbose", [bool, str], optional=True, default="deprecated", doc=doc_verbose)
    ]


def CosineAnnealingLR():
    doc_T_max = "Maximum number of iterations. Default: 100."
    doc_eta_min = "Minimum learning rate. Default: 0."

    return [
        Argument("T_max", int, optional=True, default=100, doc=doc_T_max),
        Argument("eta_min", float, optional=True, default=0, doc=doc_eta_min),
    ]

def QHFlowPolynomialLR():
    doc_warmup = "Number of linear warmup steps before polynomial decay. Default: 1000."
    doc_total = "Total number of scheduler steps. Default: 200000."
    doc_end_lr = "Final learning rate reached at num_training_steps. Default: 1e-9."
    doc_power = "Polynomial decay power. QHFlow2 water config uses 1.0. Default: 1.0."

    return [
        Argument("warmup_step", int, optional=True, default=1000, doc=doc_warmup),
        Argument("num_training_steps", int, optional=True, default=200000, doc=doc_total),
        Argument("end_lr", float, optional=True, default=1.0e-9, doc=doc_end_lr),
        Argument("scheduler_power", float, optional=True, default=1.0, doc=doc_power),
        Argument("last_epoch", int, optional=True, default=-1),
    ]

def WarmupStableDecayLR():
    doc_total_steps = "Total number of optimizer steps for DPA4 warmup-stable-decay scheduling."
    doc_warmup_steps = "Number of linear warmup steps. DPA4 tables use 5000."
    doc_decay_ratio = "Training-progress fraction at which final decay starts. DPA4 tables use 0.65."
    doc_min_lr = "Final minimum learning rate. DPA4 tables use 1e-6."
    doc_warmup_lr = "Initial warmup learning rate. Default: 0."
    doc_decay_steps = "Optional explicit decay length. If set, overrides decay_ratio-derived decay length."
    doc_decay_type = "Decay type. DPA4 uses cosine."
    doc_last_epoch = "Last scheduler step index for resume. Default: -1."

    return [
        Argument("total_steps", int, optional=False, doc=doc_total_steps),
        Argument("warmup_steps", int, optional=True, default=5000, doc=doc_warmup_steps),
        Argument("decay_ratio", float, optional=True, default=0.65, doc=doc_decay_ratio),
        Argument("min_lr", [float, list], optional=True, default=1e-6, doc=doc_min_lr),
        Argument("warmup_lr", [float, list], optional=True, default=0.0, doc=doc_warmup_lr),
        Argument("decay_steps", int, optional=True, default=None, doc=doc_decay_steps),
        Argument("decay_type", str, optional=True, default="cosine", doc=doc_decay_type),
        Argument("last_epoch", int, optional=True, default=-1, doc=doc_last_epoch),
    ]

def WarmupReduceOnPlateau():
    doc_warmup_steps = "Number of linear warmup scheduler steps before ReduceLROnPlateau starts. Default: 5000."
    doc_warmup_lr = "Initial warmup learning rate. Can be a scalar or one value per optimizer parameter group. Default: 0."
    doc_last_epoch = "Last scheduler step index for resume. Default: -1."

    return [
        Argument("warmup_steps", int, optional=True, default=5000, doc=doc_warmup_steps),
        Argument("warmup_lr", [float, list], optional=True, default=0.0, doc=doc_warmup_lr),
        *ReduceOnPlateau(),
        Argument("last_epoch", int, optional=True, default=-1, doc=doc_last_epoch),
    ]

def lr_scheduler():
    doc_type = "select type of lr_scheduler, support type includes `exp`, `linear`, `rop`, `warmup_rop`, `cos`, `wsd`, `cyclic`, and `qhflow_poly`"

    return Variant("type", [
            Argument("exp", dict, ExponentialLR()),
            Argument("linear", dict, LinearLR()),
            Argument("rop", dict, ReduceOnPlateau(), doc="rop: reduce on plateau"),
            Argument("warmup_rop", dict, WarmupReduceOnPlateau(), doc="warmup_rop: linear warmup followed by reduce on plateau"),
            Argument("cos", dict, CosineAnnealingLR(), doc="cos: cosine annealing"),
            Argument("wsd", dict, WarmupStableDecayLR(), doc="wsd: DPA4 warmup-stable-decay"),
            Argument("cyclic", dict, CyclicLR(), doc="Cyclic learning rate"),
            Argument("qhflow_poly", dict, QHFlowPolynomialLR(), doc="QHFlow2-style warmup plus polynomial decay")
        ],optional=True, default_tag="exp", doc=doc_type)


def train_data_sub():
    doc_root = "This is where the dataset stores data files."
    doc_prefix = "The prefix of the folders under root, which will be loaded in dataset."
    doc_ham = "Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset."
    doc_h0 = "Choose whether to load H0 initialization data. If `hamiltonian_0` is present it will be converted online; if precomputed `node_h0/edge_h0` are present they will be used directly."
    doc_h0_key = "The raw LMDB key used for H0 Hamiltonian blocks. Default: `hamiltonian_0`."
    doc_precomputed_h0 = "Prefer precomputed `node_h0/edge_h0` over online conversion from `hamiltonian_0` when both exist. Default: `True`."
    doc_eig = "Choose whether the eigenvalues and k-points are loaded when building dataset."
    doc_vlp = "Choose whether the overlap blocks are loaded when building dataset."
    doc_DM = "Choose whether the density matrix is loaded when building dataset."
    doc_separator = "the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'"

    args = [
        Argument("type", str, optional=True, default="DefaultDataset", doc="The type of dataset."),
        Argument("root", str, optional=False, doc=doc_root),
        Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
        Argument("separator", str, optional=True, default='.', doc=doc_separator),
        Argument("get_Hamiltonian", bool, optional=True, default=False, doc=doc_ham),
        Argument("get_H0", bool, optional=True, default=False, doc=doc_h0),
        Argument("get_P2", bool, optional=True, default=False, doc="Backward-compatible enable switch for the selected first-class non-SOC P2/P23 physical prior."),
        Argument("prior_kind", str, optional=True, default="p2", doc="Selected first-class non-SOC physical prior: p2 or p23. P23 requires p2_key=hamiltonian_p23 and the dual-prior sample schema."),
        Argument("residual_hamiltonian", bool, optional=True, default=False, doc="If true (with get_Hamiltonian), subtract H0 (raw LMDB key h0_key, default hamiltonian_0) from the Hamiltonian target so the block-native loss regresses the residual dH = H - H0. The MAE stays on the same error scale as the absolute-H target."),
        Argument("h0_key", str, optional=True, default="hamiltonian_0", doc=doc_h0_key),
        Argument("prefer_precomputed_h0", bool, optional=True, default=True, doc=doc_precomputed_h0),
        Argument("p2_key", str, optional=True, default="hamiltonian_p2", doc="Raw LMDB AO-block dictionary key for the P2 physical prior."),
        Argument("prefer_precomputed_p2", bool, optional=True, default=True, doc="Prefer precomputed node_p2/edge_p2 RME features while retaining P2 AO blocks for Full-H reconstruction."),
        Argument("require_full_h_target", bool, optional=True, default=False, doc="Require versioned absolute Full-H target fields/metadata; never infer Full H from historical delta-named targets."),
        Argument("require_residual_h_target", bool, optional=True, default=False, doc="Require a versioned raw-H/raw-H0 residual target declaration; never infer H-H0 provenance from field names."),
        Argument("require_uureal_block_ode", bool, optional=True, default=False, doc="Require the fail-closed compact uu_real already-delta block contract."),
        Argument("expected_p2_source_fingerprint", str, optional=True, default="", doc="Optional SHA256 lock for the P2 table/source provenance."),
        Argument("expected_physical_h0_source_fingerprint", str, optional=True, default="", doc="Externally trusted SHA256 lock for a dedicated physical-H0 source manifest."),
        Argument("allow_unbound_prior_source_fingerprint", bool, optional=True, default=False, doc="Development-only escape hatch for synthetic prior-conditioned Full-H configs that intentionally omit expected_p2_source_fingerprint. Production configs must keep this false."),
        Argument("audit_p2_representations", bool, optional=True, default=False, doc="Reconstruct P2 AO blocks from stored RME and compare at dataset ingest (audit/smoke only)."),
        Argument("require_p2_blocks", bool, optional=True, default=False, doc="Require P2 AO block/shape fields for prior-plus-correction reconstruction."),
        Argument("get_overlap", bool, optional=True, default=False, doc=doc_vlp),
        Argument("get_DM", bool, optional=True, default=False, doc=doc_DM),
        Argument("get_eigenvalues", bool, optional=True, default=False, doc=doc_eig)
    ]

    doc_train = "The dataset settings for training."

    return Argument("train", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_train)

def validation_data_sub():
    doc_root = "This is where the dataset stores data files."
    doc_prefix = "The prefix of the folders under root, which will be loaded in dataset."
    doc_ham = "Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset."
    doc_h0 = "Choose whether to load H0 initialization data. If `hamiltonian_0` is present it will be converted online; if precomputed `node_h0/edge_h0` are present they will be used directly."
    doc_h0_key = "The raw LMDB key used for H0 Hamiltonian blocks. Default: `hamiltonian_0`."
    doc_precomputed_h0 = "Prefer precomputed `node_h0/edge_h0` over online conversion from `hamiltonian_0` when both exist. Default: `True`."
    doc_eig = "Choose whether the eigenvalues and k-points are loaded when building dataset."
    doc_vlp = "Choose whether the overlap blocks are loaded when building dataset."
    doc_DM = "Choose whether the density matrix is loaded when building dataset."
    doc_separator = "the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'"

    args = [
        Argument("type", str, optional=True, default="DefaultDataset", doc="The type of dataset."),
        Argument("root", str, optional=False, doc=doc_root),
        Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
        Argument("separator", str, optional=True, default='.', doc=doc_separator),
        Argument("get_Hamiltonian", bool, optional=True, default=False, doc=doc_ham),
        Argument("get_H0", bool, optional=True, default=False, doc=doc_h0),
        Argument("get_P2", bool, optional=True, default=False, doc="Backward-compatible enable switch for the selected first-class non-SOC P2/P23 physical prior."),
        Argument("prior_kind", str, optional=True, default="p2", doc="Selected first-class non-SOC physical prior: p2 or p23. P23 requires p2_key=hamiltonian_p23 and the dual-prior sample schema."),
        Argument("residual_hamiltonian", bool, optional=True, default=False, doc="If true (with get_Hamiltonian), subtract H0 (raw LMDB key h0_key, default hamiltonian_0) from the Hamiltonian target so the block-native loss regresses the residual dH = H - H0. The MAE stays on the same error scale as the absolute-H target."),
        Argument("h0_key", str, optional=True, default="hamiltonian_0", doc=doc_h0_key),
        Argument("prefer_precomputed_h0", bool, optional=True, default=True, doc=doc_precomputed_h0),
        Argument("p2_key", str, optional=True, default="hamiltonian_p2", doc="Raw LMDB AO-block dictionary key for the P2 physical prior."),
        Argument("prefer_precomputed_p2", bool, optional=True, default=True, doc="Prefer precomputed node_p2/edge_p2 RME features while retaining P2 AO blocks for Full-H reconstruction."),
        Argument("require_full_h_target", bool, optional=True, default=False, doc="Require versioned absolute Full-H target fields/metadata; never infer Full H from historical delta-named targets."),
        Argument("require_residual_h_target", bool, optional=True, default=False, doc="Require a versioned raw-H/raw-H0 residual target declaration; never infer H-H0 provenance from field names."),
        Argument("require_uureal_block_ode", bool, optional=True, default=False, doc="Require the fail-closed compact uu_real already-delta block contract."),
        Argument("expected_p2_source_fingerprint", str, optional=True, default="", doc="Optional SHA256 lock for the P2 table/source provenance."),
        Argument("expected_physical_h0_source_fingerprint", str, optional=True, default="", doc="Externally trusted SHA256 lock for a dedicated physical-H0 source manifest."),
        Argument("allow_unbound_prior_source_fingerprint", bool, optional=True, default=False, doc="Development-only escape hatch for synthetic prior-conditioned Full-H configs that intentionally omit expected_p2_source_fingerprint. Production configs must keep this false."),
        Argument("audit_p2_representations", bool, optional=True, default=False, doc="Reconstruct P2 AO blocks from stored RME and compare at dataset ingest (audit/smoke only)."),
        Argument("require_p2_blocks", bool, optional=True, default=False, doc="Require P2 AO block/shape fields for prior-plus-correction reconstruction."),
        Argument("get_overlap", bool, optional=True, default=False, doc=doc_vlp),
        Argument("get_DM", bool, optional=True, default=False, doc=doc_DM),
        Argument("get_eigenvalues", bool, optional=True, default=False, doc=doc_eig)
    ]

    doc_validation = "The dataset settings for validation."

    return Argument("validation", dict, optional=True, sub_fields=args, sub_variants=[], doc=doc_validation)

def reference_data_sub():
    doc_root = "This is where the dataset stores data files."
    doc_prefix = "The prefix of the folders under root, which will be loaded in dataset."
    doc_ham = "Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset."
    doc_h0 = "Choose whether to load H0 initialization data. If `hamiltonian_0` is present it will be converted online; if precomputed `node_h0/edge_h0` are present they will be used directly."
    doc_h0_key = "The raw LMDB key used for H0 Hamiltonian blocks. Default: `hamiltonian_0`."
    doc_precomputed_h0 = "Prefer precomputed `node_h0/edge_h0` over online conversion from `hamiltonian_0` when both exist. Default: `True`."
    doc_eig = "Choose whether the eigenvalues and k-points are loaded when building dataset."
    doc_vlp = "Choose whether the overlap blocks are loaded when building dataset."
    doc_DM = "Choose whether the density matrix is loaded when building dataset."
    doc_separator = "the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'"

    args = [
        Argument("type", str, optional=True, default="DefaultDataset", doc="The type of dataset."),
        Argument("root", str, optional=False, doc=doc_root),
        Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
        Argument("separator", str, optional=True, default='.', doc=doc_separator),
        Argument("get_Hamiltonian", bool, optional=True, default=False, doc=doc_ham),
        Argument("get_H0", bool, optional=True, default=False, doc=doc_h0),
        Argument("get_P2", bool, optional=True, default=False, doc="Backward-compatible enable switch for the selected first-class non-SOC P2/P23 physical prior."),
        Argument("prior_kind", str, optional=True, default="p2", doc="Selected first-class non-SOC physical prior: p2 or p23. P23 requires p2_key=hamiltonian_p23 and the dual-prior sample schema."),
        Argument("residual_hamiltonian", bool, optional=True, default=False, doc="If true (with get_Hamiltonian), subtract H0 (raw LMDB key h0_key, default hamiltonian_0) from the Hamiltonian target so the block-native loss regresses the residual dH = H - H0. The MAE stays on the same error scale as the absolute-H target."),
        Argument("h0_key", str, optional=True, default="hamiltonian_0", doc=doc_h0_key),
        Argument("prefer_precomputed_h0", bool, optional=True, default=True, doc=doc_precomputed_h0),
        Argument("p2_key", str, optional=True, default="hamiltonian_p2", doc="Raw LMDB AO-block dictionary key for the P2 physical prior."),
        Argument("prefer_precomputed_p2", bool, optional=True, default=True, doc="Prefer precomputed node_p2/edge_p2 RME features while retaining P2 AO blocks for Full-H reconstruction."),
        Argument("require_full_h_target", bool, optional=True, default=False, doc="Require versioned absolute Full-H target fields/metadata; never infer Full H from historical delta-named targets."),
        Argument("require_residual_h_target", bool, optional=True, default=False, doc="Require a versioned raw-H/raw-H0 residual target declaration; never infer H-H0 provenance from field names."),
        Argument("require_uureal_block_ode", bool, optional=True, default=False, doc="Require the fail-closed compact uu_real already-delta block contract."),
        Argument("expected_p2_source_fingerprint", str, optional=True, default="", doc="Optional SHA256 lock for the P2 table/source provenance."),
        Argument("expected_physical_h0_source_fingerprint", str, optional=True, default="", doc="Externally trusted SHA256 lock for a dedicated physical-H0 source manifest."),
        Argument("allow_unbound_prior_source_fingerprint", bool, optional=True, default=False, doc="Development-only escape hatch for synthetic prior-conditioned Full-H configs that intentionally omit expected_p2_source_fingerprint. Production configs must keep this false."),
        Argument("audit_p2_representations", bool, optional=True, default=False, doc="Reconstruct P2 AO blocks from stored RME and compare at dataset ingest (audit/smoke only)."),
        Argument("require_p2_blocks", bool, optional=True, default=False, doc="Require P2 AO block/shape fields for prior-plus-correction reconstruction."),
        Argument("get_overlap", bool, optional=True, default=False, doc=doc_vlp),
        Argument("get_DM", bool, optional=True, default=False, doc=doc_DM),
        Argument("get_eigenvalues", bool, optional=True, default=False, doc=doc_eig)
    ]

    doc_reference = "The dataset settings for reference."

    return Argument("reference", dict, optional=True, sub_fields=args, sub_variants=[], doc=doc_reference)

def test_data_sub():
    doc_root = "This is where the dataset stores data files."
    doc_prefix = "The prefix of the folders under root, which will be loaded in dataset."
    doc_ham = "Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset."
    doc_h0 = "Choose whether to load H0 initialization data. If `hamiltonian_0` is present it will be converted online; if precomputed `node_h0/edge_h0` are present they will be used directly."
    doc_h0_key = "The raw LMDB key used for H0 Hamiltonian blocks. Default: `hamiltonian_0`."
    doc_precomputed_h0 = "Prefer precomputed `node_h0/edge_h0` over online conversion from `hamiltonian_0` when both exist. Default: `True`."
    doc_eig = "Choose whether the eigenvalues and k-points are loaded when building dataset."
    doc_vlp = "Choose whether the overlap blocks are loaded when building dataset."
    doc_DM = "Choose whether the density matrix is loaded when building dataset."
    doc_separator = "the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'"

    args = [
        Argument("type", str, optional=True, default="DefaultDataset", doc="The type of dataset."),
        Argument("root", str, optional=False, doc=doc_root),
        Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
        Argument("get_Hamiltonian", bool, optional=True, default=False, doc=doc_ham),
        Argument("get_H0", bool, optional=True, default=False, doc=doc_h0),
        Argument("get_P2", bool, optional=True, default=False, doc="Backward-compatible enable switch for the selected first-class non-SOC P2/P23 physical prior."),
        Argument("prior_kind", str, optional=True, default="p2", doc="Selected first-class non-SOC physical prior: p2 or p23. P23 requires p2_key=hamiltonian_p23 and the dual-prior sample schema."),
        Argument("residual_hamiltonian", bool, optional=True, default=False, doc="If true (with get_Hamiltonian), subtract H0 (raw LMDB key h0_key, default hamiltonian_0) from the Hamiltonian target so the block-native loss regresses the residual dH = H - H0. The MAE stays on the same error scale as the absolute-H target."),
        Argument("h0_key", str, optional=True, default="hamiltonian_0", doc=doc_h0_key),
        Argument("prefer_precomputed_h0", bool, optional=True, default=True, doc=doc_precomputed_h0),
        Argument("p2_key", str, optional=True, default="hamiltonian_p2", doc="Raw LMDB AO-block dictionary key for the P2 physical prior."),
        Argument("prefer_precomputed_p2", bool, optional=True, default=True, doc="Prefer precomputed node_p2/edge_p2 RME features while retaining P2 AO blocks for Full-H reconstruction."),
        Argument("require_full_h_target", bool, optional=True, default=False, doc="Require versioned absolute Full-H target fields/metadata; never infer Full H from historical delta-named targets."),
        Argument("require_residual_h_target", bool, optional=True, default=False, doc="Require a versioned raw-H/raw-H0 residual target declaration; never infer H-H0 provenance from field names."),
        Argument("require_uureal_block_ode", bool, optional=True, default=False, doc="Require the fail-closed compact uu_real already-delta block contract."),
        Argument("expected_p2_source_fingerprint", str, optional=True, default="", doc="Optional SHA256 lock for the P2 table/source provenance."),
        Argument("expected_physical_h0_source_fingerprint", str, optional=True, default="", doc="Externally trusted SHA256 lock for a dedicated physical-H0 source manifest."),
        Argument("allow_unbound_prior_source_fingerprint", bool, optional=True, default=False, doc="Development-only escape hatch for synthetic prior-conditioned Full-H configs that intentionally omit expected_p2_source_fingerprint. Production configs must keep this false."),
        Argument("audit_p2_representations", bool, optional=True, default=False, doc="Reconstruct P2 AO blocks from stored RME and compare at dataset ingest (audit/smoke only)."),
        Argument("require_p2_blocks", bool, optional=True, default=False, doc="Require P2 AO block/shape fields for prior-plus-correction reconstruction."),
        Argument("get_eigenvalues", bool, optional=True, default=False, doc=doc_eig),
        Argument("get_overlap", bool, optional=True, default=False, doc=doc_vlp),
        Argument("get_DM", bool, optional=True, default=False, doc=doc_DM),
        Argument("separator", str, optional=True, default='.', doc=doc_separator)
    ]

    doc_test = "The dataset settings for testing."

    return Argument("test", dict, optional=False, sub_fields=args, default={}, sub_variants=[], doc=doc_test)


def data_options():
    args = [
            Argument("r_max", [float,int,None], optional=True, default=None, doc="r_max"),
            Argument("oer_max", [float,int,None], optional=True, default=None, doc="oer_max"),
            Argument("er_max", [float,int,None], optional=True, default=None, doc="er_max"),
            train_data_sub(),
            validation_data_sub(),
            reference_data_sub()
            ]

    doc_data_options = "The options for dataset settings in training."

    return Argument("data_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_data_options)

def test_data_options():

    args = [
        Argument("r_max", [float,int,None], optional=True, default=None, doc="r_max"),
        Argument("oer_max", [float,int,None], optional=True, default=None, doc="oer_max"),
        Argument("er_max", [float,int,None], optional=True, default=None, doc="er_max"),
        test_data_sub()
    ]

    doc_test_data_options = "The options for dataset settings in testing"

    return Argument("data_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_test_data_options)


def embedding():
    doc_method = "The parameters to define the embedding model."
    doc_only2b = "Whether to train the model with 2b interaction as a model initialization."

    return Variant("method", [
            Argument("se2", dict, se2()),
            Argument("deeph-e3", dict, deephe3()),
            Argument("slem", dict, slem()),
            Argument("lem_high_order", dict, slem()),
            Argument("lem", dict, slem()),
            Argument("lem_full_tp_oeq", dict, slem()),
            Argument("lem_frame", dict, slem()),
            Argument("emoles", dict, slem()),
            Argument("emoles_openequi", dict, slem()),
            Argument("emoles_openequi_norm", dict, slem()),
            Argument("emoles_openequi_norm_v2", dict, slem()),
            Argument("emoles_openequi_eqv3", dict, slem()),
            Argument("emoles_openequi_eqv3_ffn", dict, slem()),
            Argument("emoles_openequi_nodeffn", dict, slem()),
            Argument("lem_light", dict, slem()),
            Argument("lem_light_v2", dict, slem()),
            Argument("lem_charge", dict, slem()),
            Argument("lem_cutoff", dict, slem()),
            Argument("lem_moe_openequi", dict, slem()),
            Argument("lem_in_frame_moe", dict, slem()),
            Argument("lem_full_tp", dict, slem()),
            Argument("lem_in_frame_e3nn", dict, slem()),
            Argument("lem_wo_ln", dict, slem()),
            Argument("lem_in_frame", dict, slem()),
            Argument("lem_in_frame_openequi", dict, slem()),
            Argument("lem_in_frame_heavy", dict, slem()),
            Argument("lem_moe_charge", dict, slem()),
            Argument("lem_moe_topk", dict, slem()),
            Argument("lem_moe_v3", dict, slem()),
            Argument("lem_moe_v3_edge", dict, slem_edge()),
            Argument("lem_moe_v3_h0", dict, slem_h0()),
            Argument("lem_moe_v3_prior", dict, slem_prior()),
            Argument("lem_moe_v3_edge_h0", dict, slem_edge_h0()),
            Argument("lem_non_linear", dict, slem()),
            Argument("lem_non_linear_h0", dict, slem_h0()),
            Argument("lem_moe", dict, slem()),
            Argument("lem_so2", dict, slem()),
            Argument("lem_so2_local", dict, slem()),
            Argument("lem_local", dict, slem()),
            Argument("lem_global", dict, slem()),
            Argument("lem_so2_global", dict, slem()),
            Argument("trinity", dict, slem()+[Argument("only2b", bool, optional=True, default=False, doc=doc_only2b)],),
        ],optional=True, default_tag="se2", doc=doc_method)

def se2():
    doc_rs = "The soft cutoff where the smooth function starts."
    doc_rc = "The hard cutoff where the smooth function value ~0.0"
    doc_n_axis = "the out axis shape of the deepmd-se2 descriptor."
    doc_radial_net = "network to build the descriptors."

    doc_neurons = "the size of nn for descriptor"
    doc_activation = "activation"
    doc_if_batch_normalized = "whether to turn on the batch normalization."

    radial_net = [
        Argument("neurons", list, optional=False, doc=doc_neurons),
        Argument("activation", str, optional=True, default="tanh", doc=doc_activation),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized),
    ]

    return [
        Argument("rs", [float, int], optional=False, doc=doc_rs),
        Argument("rc", [float, int], optional=False, doc=doc_rc),
        Argument("radial_net", dict, sub_fields=radial_net, optional=False, doc=doc_radial_net),
        Argument("n_axis", [int, None], optional=True, default=None, doc=doc_n_axis),
    ]


def baseline():

    doc_rs = ""
    doc_rc = ""
    doc_n_axis = ""
    doc_radial_embedding = ""

    doc_neurons = ""
    doc_activation = ""
    doc_if_batch_normalized = ""

    radial_embedding = [
        Argument("neurons", list, optional=False, doc=doc_neurons),
        Argument("activation", str, optional=True, default="tanh", doc=doc_activation),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized),
    ]

    return [
        Argument("p", [float, int], optional=False, doc=doc_rs),
        Argument("rc", [float, int], optional=False, doc=doc_rc),
        Argument("n_basis", int, optional=False, doc=doc_rc),
        Argument("n_radial", int, optional=False, doc=doc_rc),
        Argument("n_sqrt_radial", int, optional=False, doc=doc_rc),
        Argument("n_layer", int, optional=False, doc=doc_rc),
        Argument("radial_net", dict, sub_fields=radial_embedding, optional=False, doc=doc_radial_embedding),
        Argument("hidden_net", dict, sub_fields=radial_embedding, optional=False, doc=doc_radial_embedding),
        Argument("n_axis", [int, None], optional=True, default=None, doc=doc_n_axis),
    ]

def deephe3():
    doc_irreps_embed = ""
    doc_irreps_mid = ""
    doc_lmax = ""
    doc_n_basis = ""
    doc_rc = ""
    doc_n_layer = ""

    return [
            Argument("irreps_embed", str, optional=True, default="64x0e", doc=doc_irreps_embed),
            Argument("irreps_mid", str, optional=True, default="64x0e+32x1o+16x2e+8x3o+8x4e+4x5o", doc=doc_irreps_mid),
            Argument("lmax", int, optional=True, default=3, doc=doc_lmax),
            Argument("n_basis", int, optional=True, default=128, doc=doc_n_basis),
            Argument("rc", float, optional=False, doc=doc_rc),
            Argument("n_layer", int, optional=True, default=3, doc=doc_n_layer),
        ]

def e3baseline():
    doc_irreps_hidden = ""
    doc_lmax = ""
    doc_avg_num_neighbors = ""
    doc_n_radial_basis = ""
    doc_r_max = ""
    doc_n_layers = ""
    doc_env_embed_multiplicity = ""
    doc_linear_after_env_embed = ""
    doc_latent_resnet_update_ratios_learnable = ""
    doc_latent_kwargs = ""

    return [
            Argument("irreps_hidden", str, optional=True, default="64x0e+32x1o+16x2e+8x3o+8x4e+4x5o", doc=doc_irreps_hidden),
            Argument("lmax", int, optional=True, default=3, doc=doc_lmax),
            Argument("avg_num_neighbors", [int, float], optional=True, default=50, doc=doc_avg_num_neighbors),
            Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max),
            Argument("n_layers", int, optional=True, default=3, doc=doc_n_layers),
            Argument("n_radial_basis", int, optional=True, default=3, doc=doc_n_radial_basis),
            Argument("PolynomialCutoff_p", int, optional=True, default=6, doc="The order of polynomial cutoff function. Default: 6"),
            Argument(
                "latent_kwargs", dict,
                optional={
                "mlp_latent_dimensions": [128, 128, 256],
                "mlp_nonlinearity": "silu",
                "mlp_initialization": "uniform"
            },
            default=None,
            doc=doc_latent_kwargs
            ),
            Argument("env_embed_multiplicity", int, optional=True, default=1, doc=doc_env_embed_multiplicity),
            Argument("linear_after_env_embed", bool, optional=True, default=False, doc=doc_linear_after_env_embed),
            Argument("latent_resnet_update_ratios_learnable", bool, optional=True, default=False, doc=doc_latent_resnet_update_ratios_learnable)
        ]

def e3baselinev5():
    doc_irreps_hidden = ""
    doc_lmax = ""
    doc_avg_num_neighbors = ""
    doc_n_radial_basis = ""
    doc_r_max = ""
    doc_n_layers = ""
    doc_env_embed_multiplicity = ""

    return [
            Argument("irreps_hidden", str, optional=False, doc=doc_irreps_hidden),
            Argument("lmax", int, optional=False, doc=doc_lmax),
            Argument("avg_num_neighbors", [int, float], optional=False, doc=doc_avg_num_neighbors),
            Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max),
            Argument("n_layers", int, optional=False, doc=doc_n_layers),
            Argument("n_radial_basis", int, optional=True, default=10, doc=doc_n_radial_basis),
            Argument("PolynomialCutoff_p", int, optional=True, default=6, doc="The order of polynomial cutoff function. Default: 6"),
            Argument("cutoff_type", str, optional=True, default="polynomial", doc="The type of cutoff function. Default: polynomial"),
            Argument("env_embed_multiplicity", int, optional=True, default=1, doc=doc_env_embed_multiplicity),
            Argument("tp_radial_emb", bool, optional=True, default=False, doc="Whether to use tensor product radial embedding."),
            Argument("tp_radial_channels", list, optional=True, default=[128, 128], doc="The number of channels in tensor product radial embedding."),
            Argument("latent_channels", list, optional=True, default=[128, 128], doc="The number of channels in latent embedding."),
            Argument("latent_dim", int, optional=True, default=256, doc="The dimension of latent embedding."),
            Argument("res_update", bool, optional=True, default=True, doc="Whether to use residual update."),
            Argument("res_update_ratios", float, optional=True, default=0.5, doc="The ratios of residual update, should in (0,1)."),
            Argument("res_update_ratios_learnable", bool, optional=True, default=False, doc="Whether to make the ratios of residual update learnable."),
        ]

def slem():
    doc_irreps_hidden = ""
    doc_avg_num_neighbors = ""
    doc_n_radial_basis = ""
    doc_r_max = ""
    doc_n_layers = ""
    doc_env_embed_multiplicity = ""
    doc_universal = "Set true to activate universal model related features. Currently, this will create a broader onehot embedding for the transfer learning into unseen elements. Other features are on the way. Default: `False`"
    doc_use_interpolation_out = "Set true to activate SO2 interpolation layer in the final output layer. Default: `False`"
    doc_so2_attn_aggressive = "Set true to activate SO2 attention radical mode. Default: `False`"

    doc_norm_build_node_condition_branch = "Whether to build the conditioned branch for node layer norm. Default: `True`"
    doc_norm_use_node_onehot = "Whether to use node one-hot as conditioning in node layer norm. Default: `True`"
    doc_norm_build_edge_condition_branch = "Whether to build the conditioned branch for edge layer norm. Default: `True`"
    doc_norm_use_edge_onehot = "Whether to use edge one-hot embedding as conditioning in edge layer norm. Default: `True`"
    doc_equivariant_norm_type = "Equivariant normalization on the flat irreps path. Supported: `none`, `merged_rms`."
    doc_hidden_edge_activation_type = "Activation used for hidden UpdateEdge blocks. Supported: `gate`, `swiglu_s2`."
    doc_hidden_node_activation_type = "Activation used for hidden UpdateNode blocks. Supported: `gate`, `swiglu_s2`."
    doc_swiglu_s2_grid_resolution = "Grid resolution `[lat, long]` for the flat SwiGLU-S2 adapter."
    doc_swiglu_s2_compat_mode = "Compatibility mode for hidden `swiglu_s2`. `modern` uses the new flexible layout; `legacy_uniform_only` preserves the old behavior that falls back to Gate when irreps multiplicities are not uniform across degrees."
    doc_ffn_hidden_factor = "Expansion factor for the optional node-wise equivariant FFN. Values `<= 1.0` disable it."
    doc_ffn_apply_to_last = "Whether to also attach the node-wise FFN to the final layer. Default: `False`."
    doc_so2_wigner_apply_mode = "Wigner rotation application mode for SO2 TP. Supported: `compact_blocks`, `full_dense`. Default uses compact per-l Wigner blocks to reduce peak memory; set `full_dense` to restore the previous dense Wigner path."
    doc_mole_full_expert_fast_path = "When `top_k >= num_experts`, skip top-k/one-hot/scatter router work and directly use dense normalized expert weights. This is mathematically equivalent to selecting all routed experts. Default: `True`."
    doc_so2_fusion_mode = "SO2_Linear fusion mode. Supported: `staged`, `streamed_m_major_ref`, `streamed_m_major_cueq`, `streamed_m_major_fused_p0`. The 0425-stable branch defaults to `streamed_m_major_cueq`; `streamed_m_major_fused_p0` is an opt-in trainable prototype that treats Wigner/R as constants and falls back on unsupported shapes."
    doc_mole_linear_mode = "MoLELinear backend. Supported: `split_loop`, `indexed_ref`, `cueq_indexed_linear`, `cublas_grouped`. The 0422-cueq-fastest branch defaults to `cueq_indexed_linear`."
    doc_so2_m_linear_mode = "SO2 m-linear backend for non-MoE SO2 TP. Supported values are `standard`, `indexed_sandwich_multi`, or null; `cublas_grouped` is accepted only as a legacy alias. Triton experiment modes remain unsupported."
    doc_so2_expert_mixing_mode = "Expert mixing placement for SO2 MoE TP. `pre_activation` keeps the existing fused-weight path; `post_activation` evaluates raw expert TP outputs, applies equivariant activation, routes from 0e output scalars, and mixes activated outputs."
    doc_so2_expert_route_chunk_size = "Maximum original SO2 rows processed per post-activation expert-mixing chunk. Null or non-positive means process all rows in one chunk."
    doc_so2_expert_route_checkpoint = "Whether to activation-checkpoint each post-activation expert-route chunk. This recomputes TP/activation/router during backward to reduce saved route activations."
    doc_so2_output_router_hidden_dim = "Hidden size for the 0e router used by `so2_expert_mixing_mode=post_activation`."
    doc_mole_linear_m0_mode = "Legacy Triton route compatibility key. The 0425-stable branch accepts only `standard` or null; non-standard Triton values belong on the Triton experiment branch."
    doc_onehot_tp_mode = "Backend for scalar onehot tensor products. The 0422-cueq-fastest branch supports only `scalar_fast`, storing a lightweight scalar-onehot module and applying TP as direct per-irrep scaling/mixing."
    doc_output_route = "Canonical output route. Official matrix: `h_a0`, `h_a1`, `h_b0`, `h_b1`, `p_b0`, `p_b1_ict`. Controls: `legacy_rme`, `rme_fusion`, `p_b1_reference`, `debug_block_linear`."
    doc_rme_head_mode = "Deprecated output-route alias retained for old configs/checkpoints. Prefer `output_route`."
    doc_rme_fusion_rank = "Low-rank scalar-conditioning width for output heads. Default: 16."
    doc_rme_fusion_init = "Stddev of dynamic output-head projections. 0.0 disables dynamic residual/path weights at initialization."
    doc_rme_fusion_condition = "Condition source for output heads. Currently only `scalar_0e`."
    doc_rme_cartesian_scope = "ICT/Cartesian product scope for `late_rme_cartesian_hybrid` and `late_block_cartesian_projector`: `missing_only` or `all`."
    doc_ao_projector_channels = "Direct AO-pair decoder multiplicity. `0` builds the complete ordered AO-pair representation with dimension max_norb^2; positive values are compressed ablations."
    doc_ao_projector_normalization = "AO-pair projector normalization. Currently `e3hamiltonian`."
    doc_ao_projector_basis = "AO-pair projector basis convention. Currently `deeptb_real_ao`."
    doc_ao_projector_backend = "AO-pair projector source: `reference_wigner` or convention-checked `precomputed` bank."
    doc_ao_projector_bank_path = "Path to projector bank JSON when ao_projector_backend=`precomputed`."
    doc_node_message_aggregation = "Node message aggregation mode. Supported: `scatter` for the legacy sum, `single_head_0e` for DPA4-style envelope-gated scalar attention."
    doc_num_focus = "Number of post-activation 0e focus gates. Values larger than 1 enable DPA4-style channel focus routing."
    doc_focus_attention_dim = "Hidden dimension of the single-head 0e attention query/key projections."
    doc_edge_aggregation_gated_attention = "Apply query-dependent sigmoid gating after edge-to-node aggregation, following the SDPA-output gated-attention pattern while preserving equivariant irrep groups. Default: `False`."
    doc_edge_attention_key_source = "Key source for single-head edge attention. Currently supported: `message`, using post-activation edge message 0e scalars as keys. Default: `message`."
    doc_edge_attention_envelope_power = "Power applied to cutoff coefficients in single-head edge attention numerator. `1.0` preserves the legacy implementation; `2.0` uses cutoff^2. Default: `1.0`."
    doc_edge_attention_use_latent_bias = "Whether to add latent-conditioned bias to single-head edge attention logits. Default: `True`, preserving the legacy implementation."
    doc_edge_attention_key_layer_norm = "Apply LayerNorm only to message 0e scalars before the single-head edge-attention key projection. Default: `False`."
    doc_edge_attention_query_layer_norm = "Apply LayerNorm only to destination node 0e scalars before the single-head edge-attention query projection. Default: `False`."
    doc_edge_attention_qk_layer_norm = "Shortcut that applies LayerNorm to both query and key 0e scalar inputs before the single-head edge-attention projections. Default: `False`."
    doc_edge_message_env_weight = "Whether to apply the legacy latent-conditioned env value weighting to node-update edge messages before aggregation. Default: `True`, preserving the legacy implementation."
    doc_edge_message_value_gate = "Apply a query-dependent sigmoid value gate to edge messages before node aggregation. The gate is generated from destination node 0e scalars and message 0e scalars, then applied per equivariant irrep group. Default: `False`."
    doc_edge_message_value_gate_hidden_dim = "Optional hidden dimension for edge_message_value_gate. Use 0 for a single linear sigmoid gate. Default: `0`."

    return [
        Argument("irreps_hidden", str, optional=False, doc=doc_irreps_hidden),
        Argument("avg_num_neighbors", [int, float], optional=False, doc=doc_avg_num_neighbors),
        Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max),
        Argument("n_layers", int, optional=False, doc=doc_n_layers),
        Argument("mp_cutoff", [float, int, dict], optional=True),

        Argument("self_mix_mode", str, optional=True, default="full"),
        Argument("self_mix_type", str, optional=True, default="all"),
        Argument("self_mix_flag", bool, optional=True, default=False),
        Argument("optimized_in_frame", bool, optional=True, default=True),
        Argument("self_mix_iter", int, optional=True, default=2),

        Argument("n_radial_basis", int, optional=True, default=128, doc=doc_n_radial_basis),
        Argument("top_k", int, optional=True, default=4, doc="The number of experts to be used in MoE. Default: 1"),
        Argument("num_experts", int, optional=True, default=24, doc="The number of experts for MoE. Default: 8"),
        Argument("num_shared_experts", int, optional=True, default=4, doc="The number of experts for MoE. Default: 8"),
        Argument("mole_full_expert_fast_path", bool, optional=True, default=True, doc=doc_mole_full_expert_fast_path),
        Argument("PolynomialCutoff_p", int, optional=True, default=6, doc="The order of polynomial cutoff function. Default: 6"),
        Argument("cutoff_type", str, optional=True, default="polynomial", doc="The type of cutoff function. Default: polynomial"),
        Argument("color_mode", str, optional=True, default="tp", doc="The type of color mode. Default: tp"),
        Argument("onehot_mode", str, optional=True, default="FullTP", doc="The type of onehot mode. Default: FullTP"),
        Argument("env_embed_multiplicity", int, optional=True, default=64, doc=doc_env_embed_multiplicity),
        Argument("tp_radial_emb", bool, optional=True, default=False, doc="Whether to use tensor product radial embedding."),
        Argument("tp_radial_channels", list, optional=True, default=[32], doc="The number of channels in tensor product radial embedding."),
        Argument("latent_channels", list, optional=True, default=[32], doc="The number of channels in latent embedding."),
        Argument("latent_dim", int, optional=True, default=64, doc="The dimension of latent embedding."),
        Argument("edge_one_hot_dim", int, optional=True, default=128, doc="The dimension of edge_one_hot."),
        Argument("use_out_onehot_tp", bool, optional=True, default=True, doc="Whether to use out_onehot_tp."),
        Argument("use_layer_onehot_tp", bool, optional=True, default=True, doc="Whether to use layer_onehot_tp."),
        Argument("output_route", [str, None], optional=True, default=None, doc=doc_output_route),
        Argument("rme_head_mode", [str, None], optional=True, default=None, doc=doc_rme_head_mode),
        Argument("rme_fusion_rank", int, optional=True, default=16, doc=doc_rme_fusion_rank),
        Argument("rme_fusion_init", [float, int], optional=True, default=0.0, doc=doc_rme_fusion_init),
        Argument("rme_fusion_condition", str, optional=True, default="scalar_0e", doc=doc_rme_fusion_condition),
        Argument("rme_cartesian_scope", [str, None], optional=True, default=None, doc=doc_rme_cartesian_scope),
        Argument("rme_ict_scope", [str, None], optional=True, default=None, doc=doc_rme_cartesian_scope),
        Argument("ao_projector_channels", int, optional=True, default=0, doc=doc_ao_projector_channels),
        Argument("ao_projector_normalization", str, optional=True, default="e3hamiltonian", doc=doc_ao_projector_normalization),
        Argument("ao_projector_basis_convention", str, optional=True, default="deeptb_real_ao", doc=doc_ao_projector_basis),
        Argument("ao_projector_backend", str, optional=True, default="reference_wigner", doc=doc_ao_projector_backend),
        Argument("ao_projector_bank_path", [str, None], optional=True, default=None, doc=doc_ao_projector_bank_path),
        Argument("res_update", bool, optional=True, default=True, doc="Whether to use residual update."),
        Argument("res_update_ratios", float, optional=True, default=0.5, doc="The ratios of residual update, should in (0,1)."),
        Argument("norm_bottleneck_ratio", float, optional=True, default=0.1, doc="The ratios of norm bottle neck gate."),
        Argument("res_update_ratios_learnable", bool, optional=True, default=False, doc="Whether to make the ratios of residual update learnable."),
        Argument("use_interpolation_out", bool, optional=True, default=False, doc=doc_use_interpolation_out),
        Argument("so2_attn_aggressive", bool, optional=True, default=False, doc=doc_so2_attn_aggressive),
        Argument("universal", bool, optional=True, default=False, doc=doc_universal),
        Argument("in_frame_flag", bool, optional=True, default=True),
        Argument("ln_flag", bool, optional=True, default=True),
        Argument("use_angle", bool, optional=True, default=False, doc="Whether to use angle."),
        Argument("norm_eps", float, optional=True, default=1e-8, doc="eps in SeperableLayerNorm."),
        Argument("equivariant_norm_type", str, optional=True, default="none", doc=doc_equivariant_norm_type),
        Argument("hidden_edge_activation_type", str, optional=True, default="gate", doc=doc_hidden_edge_activation_type),
        Argument("hidden_node_activation_type", str, optional=True, default="gate", doc=doc_hidden_node_activation_type),
        Argument("swiglu_s2_grid_resolution", list, optional=True, default=[14, 14], doc=doc_swiglu_s2_grid_resolution),
        Argument("swiglu_s2_compat_mode", str, optional=True, default="modern", doc=doc_swiglu_s2_compat_mode),
        Argument("ffn_hidden_factor", float, optional=True, default=0.0, doc=doc_ffn_hidden_factor),
        Argument("ffn_apply_to_last", bool, optional=True, default=False, doc=doc_ffn_apply_to_last),
        Argument("so2_wigner_apply_mode", str, optional=True, default="compact_blocks", doc=doc_so2_wigner_apply_mode),
        Argument("so2_fusion_mode", str, optional=True, default="streamed_m_major_cueq", doc=doc_so2_fusion_mode),
        Argument("mole_linear_mode", [str, None], optional=True, default="cueq_indexed_linear", doc=doc_mole_linear_mode),
        Argument("so2_m_linear_mode", [str, None], optional=True, default=None, doc=doc_so2_m_linear_mode),
        Argument("so2_expert_mixing_mode", str, optional=True, default="pre_activation", doc=doc_so2_expert_mixing_mode),
        Argument("so2_expert_route_chunk_size", [int, None], optional=True, default=None, doc=doc_so2_expert_route_chunk_size),
        Argument("so2_expert_route_checkpoint", bool, optional=True, default=False, doc=doc_so2_expert_route_checkpoint),
        Argument("so2_output_router_hidden_dim", int, optional=True, default=32, doc=doc_so2_output_router_hidden_dim),
        Argument("mole_linear_m0_mode", [str, None], optional=True, default=None, doc=doc_mole_linear_m0_mode),
        Argument("onehot_tp_mode", [str, None], optional=True, default=None, doc=doc_onehot_tp_mode),
        Argument("node_message_aggregation", str, optional=True, default="scatter", doc=doc_node_message_aggregation),
        Argument("num_focus", int, optional=True, default=1, doc=doc_num_focus),
        Argument("focus_attention_dim", int, optional=True, default=32, doc=doc_focus_attention_dim),
        Argument("edge_aggregation_gated_attention", bool, optional=True, default=False, doc=doc_edge_aggregation_gated_attention),
        Argument("edge_attention_key_source", str, optional=True, default="message", doc=doc_edge_attention_key_source),
        Argument("edge_attention_envelope_power", float, optional=True, default=1.0, doc=doc_edge_attention_envelope_power),
        Argument("edge_attention_use_latent_bias", bool, optional=True, default=True, doc=doc_edge_attention_use_latent_bias),
        Argument("edge_attention_key_layer_norm", bool, optional=True, default=False, doc=doc_edge_attention_key_layer_norm),
        Argument("edge_attention_query_layer_norm", bool, optional=True, default=False, doc=doc_edge_attention_query_layer_norm),
        Argument("edge_attention_qk_layer_norm", bool, optional=True, default=False, doc=doc_edge_attention_qk_layer_norm),
        Argument("edge_message_env_weight", bool, optional=True, default=True, doc=doc_edge_message_env_weight),
        Argument("edge_message_value_gate", bool, optional=True, default=False, doc=doc_edge_message_value_gate),
        Argument("edge_message_value_gate_hidden_dim", int, optional=True, default=0, doc=doc_edge_message_value_gate_hidden_dim),

        # ---- New norm conditioning flags ----
        Argument("norm_build_node_condition_branch", bool, optional=True, default=True, doc=doc_norm_build_node_condition_branch),
        Argument("norm_use_node_onehot", bool, optional=True, default=True, doc=doc_norm_use_node_onehot),
        Argument("norm_build_edge_condition_branch", bool, optional=True, default=True, doc=doc_norm_build_edge_condition_branch),
        Argument("norm_use_edge_onehot", bool, optional=True, default=True, doc=doc_norm_use_edge_onehot),
    ]


def slem_h0():
    doc_use_h0_init = "Whether to replace the geometry init layer with the H0 init plugin. Default: `True`."
    doc_h0_node_key = "Node-wise H0 key. Defaults to `node_h0`. When absent and fallback is enabled, the plugin checks `node_hamiltonian` first and then the configured fallback feature key."
    doc_h0_edge_key = "Edge-wise H0 key. Defaults to `edge_h0`. When absent and fallback is enabled, the plugin checks `edge_hamiltonian` first and then the configured fallback feature key."
    doc_use_h0_node_init = "Whether H0 replaces/adds to the native node InitLayer output. Default: `True`."
    doc_use_h0_edge_init = "Whether H0 replaces/adds to the native edge InitLayer output. Default: `True`."
    doc_h0_node_mode = "How to build node init from H0. Supported: `direct`, `self_edge`. Default: `direct`."
    doc_fallback_to_hamiltonian = "Whether to fall back to the LMDB Hamiltonian-derived node/edge features when explicit H0 keys are absent. Default: `True`."
    doc_fallback_node_key = "Fallback node key used when explicit H0 is absent. Default: `node_features`."
    doc_fallback_edge_key = "Fallback edge key used when explicit H0 is absent. Default: `edge_features`."
    doc_h0_merge_mode = "How to combine H0-projected features with the base init output. Supported: `replace`, `add`. Default: `replace`."
    doc_h0_self_edge_tol = "Tolerance used to detect self-edges in `self_edge` node mode. Default: `1e-8`."
    doc_use_flow_time_embedding = "Whether to inject graph-level flow time into scalar channels before message passing. Default: `False`."
    doc_flow_time_condition_edges = "Whether to also inject graph-level flow time into active edge scalar channels when flow-time embedding is enabled. Default: `True`."
    doc_flow_time_key = "Graph-level flow time key written by train_options.flow_options. Default: `flow_time`."
    doc_flow_time_keys = "Optional list of graph-level time keys to embed and sum, e.g. [`flow_time_t`, `flow_time_r`, `flow_time_h`] for Pixel MeanFlow."
    doc_flow_time_max_positions = "Scale used by the sinusoidal flow-time embedding. Default: `2000`."
    doc_flow_time_allow_missing = "Whether missing flow time may fall back to flow_time_missing_value. Default: `True`; block-ODE requires `False`."
    doc_flow_time_missing_value = "Fallback normalized time when flow_time is absent. Default: `0.0`."
    doc_require_full_block_edge_coverage = "Fail before the H-B0 head unless its actual active rows are the ordered full graph-edge range with finite positive cutoff coefficients. Default: `False`; block-ODE requires `True`."

    return slem() + [
        Argument("use_h0_init", bool, optional=True, default=True, doc=doc_use_h0_init),
        Argument("h0_node_key", str, optional=True, default="node_h0", doc=doc_h0_node_key),
        Argument("h0_edge_key", str, optional=True, default="edge_h0", doc=doc_h0_edge_key),
        Argument("use_h0_node_init", bool, optional=True, default=True, doc=doc_use_h0_node_init),
        Argument("use_h0_edge_init", bool, optional=True, default=True, doc=doc_use_h0_edge_init),
        Argument("h0_node_mode", str, optional=True, default="direct", doc=doc_h0_node_mode),
        Argument("fallback_to_hamiltonian", bool, optional=True, default=True, doc=doc_fallback_to_hamiltonian),
        Argument("h0_fallback_to_hamiltonian", bool, optional=True, default=True, doc=doc_fallback_to_hamiltonian),
        Argument("fallback_node_key", str, optional=True, default="node_features", doc=doc_fallback_node_key),
        Argument("fallback_edge_key", str, optional=True, default="edge_features", doc=doc_fallback_edge_key),
        Argument("allow_target_fallback_in_training", bool, optional=True, default=False,
                 doc="Permit the H0 input fallback to resolve to the target Hamiltonian/"
                     "features while the module is in training mode. Off by default: "
                     "that fallback is a label leak during training (it is a deliberate "
                      "surrogate only at inference)."),
        Argument("use_uureal_residual_block_input", bool, optional=True, default=False,
                 doc="Enable the mapper-derived bias-free residual AO-block projector."),
        Argument("h0_merge_mode", str, optional=True, default="replace", doc=doc_h0_merge_mode),
        Argument("h0_self_edge_tol", float, optional=True, default=1e-8, doc=doc_h0_self_edge_tol),
        Argument("use_flow_time_embedding", bool, optional=True, default=False, doc=doc_use_flow_time_embedding),
        Argument("flow_time_condition_edges", bool, optional=True, default=True, doc=doc_flow_time_condition_edges),
        Argument("flow_time_key", str, optional=True, default="flow_time", doc=doc_flow_time_key),
        Argument("flow_time_keys", list, optional=True, default=[], doc=doc_flow_time_keys),
        Argument("flow_time_max_positions", int, optional=True, default=2000, doc=doc_flow_time_max_positions),
        Argument("flow_time_allow_missing", bool, optional=True, default=True, doc=doc_flow_time_allow_missing),
        Argument("flow_time_missing_value", (int, float), optional=True, default=0.0, doc=doc_flow_time_missing_value),
        Argument("require_full_block_edge_coverage", bool, optional=True, default=False,
                 doc=doc_require_full_block_edge_coverage),
    ]


def slem_edge():
    doc_edge_router_in_features = "Input dimension for the edge-wise MoE router. Defaults to `edge_one_hot_dim`."
    doc_edge_router_unique_types = "For edge-wise MoE, route unique active bond types once and map them back to active edges. Default: `True`."
    doc_edge_moe_compact_dispatch = "For edge-wise MoE with unique-type routing, enable grouped compact dispatch for large-edge batches. Default: `True`."
    doc_edge_moe_compact_min_edges = "Minimum active-edge count before grouped compact dispatch is used. Default: `16384`."

    return slem() + [
        Argument("edge_router_in_features", [int, None], optional=True, default=None, doc=doc_edge_router_in_features),
        Argument("edge_router_unique_types", bool, optional=True, default=True, doc=doc_edge_router_unique_types),
        Argument("edge_moe_compact_dispatch", bool, optional=True, default=True, doc=doc_edge_moe_compact_dispatch),
        Argument("edge_moe_compact_min_edges", int, optional=True, default=16384, doc=doc_edge_moe_compact_min_edges),
    ]


def slem_edge_h0():
    doc_edge_router_in_features = "Input dimension for the edge-wise MoE router. Defaults to `edge_one_hot_dim`."
    doc_edge_router_unique_types = "For edge-wise MoE, route unique active bond types once and map them back to active edges. Default: `True`."
    doc_edge_moe_compact_dispatch = "For edge-wise MoE with unique-type routing, enable grouped compact dispatch for large-edge batches. Default: `True`."
    doc_edge_moe_compact_min_edges = "Minimum active-edge count before grouped compact dispatch is used. Default: `16384`."

    return slem_h0() + [
        Argument("edge_router_in_features", [int, None], optional=True, default=None, doc=doc_edge_router_in_features),
        Argument("edge_router_unique_types", bool, optional=True, default=True, doc=doc_edge_router_unique_types),
        Argument("edge_moe_compact_dispatch", bool, optional=True, default=True, doc=doc_edge_moe_compact_dispatch),
        Argument("edge_moe_compact_min_edges", int, optional=True, default=16384, doc=doc_edge_moe_compact_min_edges),
    ]


def prediction():
    doc_method = "The options to indicate the prediction model. Can be sktb, e3tb, or block_native."
    doc_nn = "neural network options for prediction model."

    return Variant("method", [
            Argument("sktb", dict, sktb_prediction(), doc=doc_nn),
            Argument("e3tb", dict, e3tb_prediction(), doc=doc_nn),
            Argument("block_native", dict, block_native_prediction(), doc=doc_nn),
        ], optional=False, doc=doc_method)

def sktb_prediction():
    doc_neurons = "neurons in the neural network."
    doc_activation = "activation function."
    doc_if_batch_normalized = "if to turn on batch normalization"

    nn = [
        Argument("neurons", list, optional=False, doc=doc_neurons),
        Argument("activation", str, optional=True, default="tanh", doc=doc_activation),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized),
    ]

    return nn


def e3tb_prediction():
    doc_scales_trainable = "The scale parameter is from the statistics. Whether to train this parameter."
    doc_shifts_trainable = "The scale parameter is from the statistics. Whether to train this parameter."
    doc_neurons = "neurons in the neural network."
    doc_activation = "activation function."
    doc_if_batch_normalized = "if to turn on batch normalization"
    doc_scale_type = ("Which scale method to use. Can be no_scale, "
                      "scale_wo_back_grad (the scale parameter will not engage the back grad computation graph), "
                      "scale_w_back_grad (the scale parameter will engage the back grad computation graph)")
    doc_blockwise_hamiltonian = (
        "If true, materialize E3 Hamiltonian feature predictions into AO block tensors "
        "for block-wise loss. This is non-SOC AO/block supervision, not a block-native head."
    )

    nn = [
        Argument("scales_trainable", bool, optional=True, default=False, doc=doc_scales_trainable),
        Argument("shifts_trainable", bool, optional=True, default=False, doc=doc_shifts_trainable),
        Argument("neurons", list, optional=True, default=None, doc=doc_neurons),
        Argument("activation", str, optional=True, default="tanh", doc=doc_activation),
        Argument("scale_type", str, optional=True, default="scale_w_back_grad", doc=doc_scale_type),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized),
        Argument("blockwise_hamiltonian", bool, optional=True, default=False, doc=doc_blockwise_hamiltonian),
        Argument("node_pad_shape", [list, None], optional=True, default=None, doc="Padded node AO block shape for blockwise Hamiltonian output."),
        Argument("edge_pad_shape", [list, None], optional=True, default=None, doc="Padded edge AO block shape for blockwise Hamiltonian output."),
        Argument("symmetrize_onsite", bool, optional=True, default=True, doc="Hermitian-complete onsite AO blocks in blockwise output."),
        Argument("complete_edges", bool, optional=True, default=True, doc="Fill missing edge AO entries from reverse directed edges in blockwise output."),
        Argument("strict_complete_edges", bool, optional=True, default=False, doc="Fail if reverse-edge completion leaves unresolved valid AO entries."),
        Argument("add_h0", bool, optional=True, default=False, doc="Also expose full H block tensors by adding converted H0 blocks to delta predictions."),
        Argument("add_prior", bool, optional=True, default=False, doc="Expose Full-H blocks as a physical prior plus learned correction. Mutually exclusive with add_h0."),
        Argument("prior_node_block_field", str, optional=True, default="node_p2_blocks", doc="Node AO-block field used for physical-prior Full-H reconstruction."),
        Argument("prior_edge_block_field", str, optional=True, default="edge_p2_blocks", doc="Edge AO-block field used for physical-prior Full-H reconstruction."),
        Argument("prior_label", str, optional=True, default="P2"),
        Argument("validate_prior_blocks", bool, optional=True, default=False, doc="Debug-only finite checks for prior AO blocks during every model forward. Production caches are validated at dataset ingest."),
        Argument("full_output_node_field", str, optional=True, default="node_full_hamil_blocks", doc="Output key for reconstructed Full-H node blocks when add_h0 or add_prior is true."),
        Argument("full_output_edge_field", str, optional=True, default="edge_full_hamil_blocks", doc="Output key for reconstructed Full-H edge blocks when add_h0 or add_prior is true."),
    ]

    return nn


def block_native_prediction():
    doc_scale_type = "Block-native decoder bypasses RME scale/shift and E3Hamiltonian; use no_scale."
    doc_block_decoder = "Block-native decoder backend: `linear`, `expansion_cg`, `cartesian_projector`, or `ao_projector`."
    doc_blockwise_hamiltonian = "Whether the downstream consumer expects explicit AO Hamiltonian blocks."

    return [
        Argument("scale_type", str, optional=True, default="no_scale", doc=doc_scale_type),
        Argument("block_decoder", str, optional=True, default="linear", doc=doc_block_decoder),
        Argument("blockwise_hamiltonian", bool, optional=True, default=True, doc=doc_blockwise_hamiltonian),
        Argument("add_h0", bool, optional=True, default=False, doc="Also expose full-H AO blocks as H0 plus residual block-native predictions. Requires get_H0=true for every dataset split."),
        Argument("add_prior", bool, optional=True, default=False, doc="Expose Full-H AO blocks as a physical prior plus learned correction. Mutually exclusive with add_h0."),
        Argument("prior_node_block_field", str, optional=True, default="node_p2_blocks"),
        Argument("prior_edge_block_field", str, optional=True, default="edge_p2_blocks"),
        Argument("prior_label", str, optional=True, default="P2"),
        Argument("validate_prior_blocks", bool, optional=True, default=False, doc="Debug-only finite checks for prior AO blocks during every model forward. Production caches are validated at dataset ingest."),
        Argument("full_output_node_field", str, optional=True, default="node_full_hamil_blocks", doc="Output key for reconstructed Full-H node blocks when add_h0 or add_prior is true."),
        Argument("full_output_edge_field", str, optional=True, default="edge_full_hamil_blocks", doc="Output key for reconstructed Full-H edge blocks when add_h0 or add_prior is true."),
    ]


def slem_prior():
    """Strict first-class non-SOC P2/P23 physical-prior schema."""
    return slem() + [
        Argument("prior_kind", str, optional=True, default="p2", doc="Physical prior family: p2 or p23."),
        Argument("use_prior_init", bool, optional=True, default=True, doc="Project the selected prior RME into the equivariant node/edge hidden state."),
        Argument("prior_node_key", str, optional=True, default="node_p2", doc="Node-wise selected-prior RME field; P23 must explicitly use node_p23."),
        Argument("prior_edge_key", str, optional=True, default="edge_p2", doc="Edge-wise selected-prior RME field; P23 must explicitly use edge_p23."),
        Argument("use_prior_node_init", bool, optional=True, default=True),
        Argument("use_prior_edge_init", bool, optional=True, default=True),
        Argument("prior_node_mode", str, optional=True, default="direct", doc="Supported: direct or self_edge."),
        Argument("prior_merge_mode", str, optional=True, default="replace", doc="Supported: replace or add."),
        Argument("prior_self_edge_tol", float, optional=True, default=1e-8),
        Argument("use_soft_edge_memory", bool, optional=True, default=True, doc="Enable scalar-only multi-head external edge KV memory."),
        Argument("soft_edge_memory_num_slots", int, optional=True, default=64),
        Argument("soft_edge_memory_num_heads", int, optional=True, default=4),
        Argument("soft_edge_memory_head_dim", int, optional=True, default=16),
        Argument("soft_edge_memory_temperature", float, optional=True, default=1.0),
        Argument("soft_edge_memory_dropout", float, optional=True, default=0.0),
        Argument("soft_edge_memory_gate_mode", str, optional=True, default="deepseek", doc="Memory gate: deepseek normalized-similarity gate or linear scalar gate."),
        Argument("soft_edge_memory_gate_bias", float, optional=True, default=0.0),
        Argument("soft_edge_memory_gate_eps", float, optional=True, default=1e-6),
        Argument("soft_edge_memory_zero_init_output", bool, optional=True, default=True),
        Argument("soft_edge_memory_input_norm", bool, optional=True, default=True),
        Argument("prior_validate_inputs", bool, optional=True, default=False, doc="Debug-only finite checks for P2 tensors during every forward; production LMDBs are validated at ingest."),
        Argument("soft_edge_memory_diagnostics_mode", str, optional=True, default="off", doc="Attention diagnostics: off, sampled, or full."),
        Argument("soft_edge_memory_diagnostics_sample_size", int, optional=True, default=1024),
    ]



def model_options():

    doc_model_options = "The parameters to define the `nnsk`,`mix` and `dptb` model."
    doc_embedding = "The parameters to define the embedding model."
    doc_prediction = "The parameters to define the prediction model"

    return Argument("model_options", dict, sub_fields=[
        Argument("embedding", dict, optional=True, sub_fields=[], sub_variants=[embedding()], doc=doc_embedding),
        Argument("prediction", dict, optional=True, sub_fields=[], sub_variants=[prediction()], doc=doc_prediction),
        nnsk(),
        dftbsk(),
        ], sub_variants=[], optional=True, doc=doc_model_options)

def dftbsk():
    doc_dftbsk = "The parameters to define the dftb sk model."

    return Argument("dftbsk", dict, sub_fields=[
                Argument("skdata", str, optional=False, doc="The path to the skfile or sk database."),
                Argument("r_max", float, optional=False, doc="the cutoff values to use sk files."),
                ], sub_variants=[], optional=True, doc=doc_dftbsk)

def nnsk():
    doc_nnsk = "The parameters to define the nnsk model."
    doc_onsite = "The onsite options to define the onsite of nnsk model."
    doc_hopping = "The hopping options to define the hopping of nnsk model."
    doc_soc = """The soc options to define the soc of nnsk model,
                Default: {} # empty dict\n
                - {'method':'none'} : use database soc value. 
                - {'method':uniform} : set lambda_il; assign a soc lambda value for each orbital -l on each atomtype i; l=0,1,2 for s p d."""
    doc_freeze = """The parameters to define the freeze of nnsk model can be bool and string and list.\n
                    Default: False\n
                     - True: freeze all the nnsk parameters\n
                     - False: train all the nnsk parameters\n 
                     - 'hopping','onsite','overlap' and 'soc' to freeze the corresponding parameters.
                     - list of the strings e.g. ['overlap','soc'] to freeze both overlap and soc parameters."""
    doc_std = "The std value to initialize the nnsk parameters. Default: 0.01"
    doc_atomic_radius = "The atomic radius to use for the nnsk model. Default: v1, can be v1 or cov"

    # overlap = Argument("overlap", bool, optional=True, default=False, doc="The parameters to define the overlap correction of nnsk model.")

    return Argument("nnsk", dict, sub_fields=[
            Argument("onsite", dict, optional=False, sub_fields=[], sub_variants=[onsite()], doc=doc_onsite),
            Argument("hopping", dict, optional=False, sub_fields=[], sub_variants=[hopping()], doc=doc_hopping),
            Argument("soc", dict, optional=True, default={}, doc=doc_soc),
            Argument("freeze", [bool,str,list], optional=True, default=False, doc=doc_freeze),
            Argument("std", float, optional=True, default=0.01, doc=doc_std),
            Argument("atomic_radius", str, optional=True, default='v1', doc=doc_atomic_radius),
            push(),
        ], sub_variants=[], optional=True, doc=doc_nnsk)

def push():
    doc_rs_thr = "The step size for cutoff value for smooth function in the nnsk anlytical formula."
    doc_rc_thr = "The step size for cutoff value for smooth function in the nnsk anlytical formula."
    doc_w_thr = "The step size for decay factor w."
    doc_ovp_thr = "The step size for overlap reduction"
    doc_period = "the interval of iterations to modify the rs w values."

    return Argument("push", [bool,dict], sub_fields=[
        Argument("rs_thr", [int,float], optional=True, default=0., doc=doc_rs_thr),
        Argument("rc_thr", [int,float], optional=True, default=0., doc=doc_rc_thr),
        Argument("w_thr",  [int,float], optional=True,  default=0., doc=doc_w_thr),
        Argument("ovp_thr", [int,float], optional=True, default=0., doc=doc_ovp_thr),
        Argument("period", int, optional=True, default=100, doc=doc_period),
    ], sub_variants=[], optional=True, default=False, doc="The parameters to define the push the soft cutoff of nnsk model.")

def onsite():
    doc_method = r"""The onsite correction mode, the onsite energy is expressed as the energy of isolated atoms plus the model correction, the correction mode are:
                    Default: `none`: use the database onsite energy value.
                    - `strain`: The strain mode correct the onsite matrix densly by $$H_{i,i}^{lm,l^\prime m^\prime} = \epsilon_l^0 \delta_{ll^\prime}\delta_{mm^\prime} + \sum_p \sum_{\zeta} \Big[ \mathcal{U}_{\zeta}(\hat{\br}_{ip}) \ \epsilon_{ll^\prime \zeta} \Big]_{mm^\prime}$$ which is also parameterized as a set of Slater-Koster like integrals.\n\n\
                    - `uniform`: The correction is a energy shift respect of orbital of each atom. Which is formally written as: 
                                $$H_{i,i}^{lm,l^\prime m^\prime} = (\epsilon_l^0+\epsilon_l^\prime) \delta_{ll^\prime}\delta_{mm^\prime}$$ Where $\epsilon_l^0$ is the isolated energy level from the DeePTB onsite database, and $\epsilon_l^\prime$ is the parameters to fit.
                    - `NRL`: use the NRL-TB formula.
                """

    doc_rs = "The smooth cutoff `fc` for strain model. rs is where fc = 0.5"
    doc_w = "The decay factor of `fc` for strain and nrl model."
    doc_rc = "The smooth cutoff of `fc` for nrl model, rc is where fc ~ 0.0"
    doc_lda = "The lambda type encoding value in nrl model. now only support elementary substance"

    strain = [
        Argument("rs", float, optional=True, default=6.0, doc=doc_rs),
        Argument("w", float, optional=True, default=0.1, doc=doc_w),
    ]

    NRL = [
        Argument("rs", float, optional=True, default=6.0, doc=doc_rc),
        Argument("w", float, optional=True, default=0.1, doc=doc_w),
        Argument("lda", float, optional=True, default=1.0, doc=doc_lda)
    ]

    return Variant("method", [
                    Argument("strain", dict, strain),
                    Argument("uniform", dict, []),
                    Argument("uniform_noref", dict, []),
                    Argument("NRL", dict, NRL),
                    Argument("none", dict, []),
                ],optional=False, doc=doc_method)

def hopping():
    doc_method = """The hopping formula. 
                    -  `powerlaw`: the powerlaw formula for bond length dependence for sk integrals.
                    -  `varTang96`: a variational formula based on Tang96 formula.
                    -  `NRL0`: the old version of NRL formula for overlap, we set overlap and hopping share same options.
                    -  `NRL1`: the new version of NRL formula for overlap. 
                    """
    doc_rs_soft = "The cut-off for smooth function fc for powerlaw and varTang96, fc(rs)=0.5"
    doc_w = " The decay w in fc"
    doc_rs_hard = "The cut-off for smooth function fc, fc(rs) = 0."

    powerlaw = [
        Argument("rs", [float,dict], optional=True, default=6.0, doc=doc_rs_soft),
        Argument("w", float, optional=True, default=0.1, doc=doc_w),
    ]
    varTang96 = [
        Argument("rs",  [float,dict], optional=True, default=6.0, doc=doc_rs_soft),
        Argument("w", float, optional=True, default=0.1, doc=doc_w),
    ]
    common_params = [
        Argument("rs",  [float,dict], optional=True, default=6.0, doc=doc_rs_hard),
        Argument("w", float, optional=True, default=0.1, doc=doc_w),
    ]

    formulas = [
        'poly1pow',
        'poly2pow',
        'poly3pow',
        'poly4pow',
        'poly2exp',
        'poly3exp',
        'poly4exp',
        'NRL0',
        "NRL1"]

    args = [
        Argument("powerlaw", dict, powerlaw),
        Argument("varTang96", dict, varTang96),
        Argument("custom", dict, [])
    ]

    for ii in formulas:
        args.append(Argument(ii, dict, common_params))

    return Variant("method", args,optional=False, doc=doc_method)


def loss_options():
    doc_method = """The loss function type, defined by a string like `<fitting target>_<loss type>`, Default: `eigs_l2dsf`. supported loss functions includes:\n\n\
                    - `eigvals`: The mse loss predicted and labeled eigenvalues and Delta eigenvalues between different k.
                    - `hamil`: 
                    - `hamil_abs`:
                    - `hamil_abs_element_avg`:
                    - `hamil_blas`:
                """
    doc_train = "Loss options for training."
    doc_validation = "Loss options for validation."
    doc_reference = "Loss options for reference data in training."
    doc_test = "Loss options for testing."
    doc_model_basis_name = "The basis used by the model for the calculation of fock matrix. Default: def2svp"
    doc_on_the_fly_ovp_flag = "Calculate overlap matrices on the fly. Default: True"
    doc_on_the_fly_solve_eigen = "Get eigen values on the fly. Default: True"
    doc_add_ham_flag = "Add huber loss of hamiltonian element to the waloss. Default: True"
    doc_use_energy_weighting = "Use gaussian smearing for energy weighting. Default: True"
    doc_dataset_basis_name = "The basis used in the dataset. Default: def2svp"

    hamil = [
        Argument(
            "onsite_shift",
            bool,
            optional=True,
            default=False,
            doc="Whether to apply a global onsite shift (μ) between prediction and reference Hamiltonians. "
                "Implemented by shifting ref_data using the overlap matrix. Default: False.",
        ),
        Argument(
            "debug_flag",
            bool,
            optional=True,
            default=False,
            doc="Whether to print additional debug information inside the loss (e.g. norms, masks). "
                "Default: False.",
        ),
        Argument(
            "nextham_uureal_mask",
            bool,
            optional=True,
            default=False,
            doc="Whether to use NextHAM-style uu.real masking on SOC features. "
                "When True, only the uu.real block of each SOC slice is supervised in the loss; "
                "other spin/im parts are ignored. Default: False.",
        ),
        # 以下是我们为 HamilLossAbsMAE 新增的可选参数（如果你启用了 onsite_boost 机制）
        Argument(
            "onsite_boost",
            bool,
            optional=True,
            default=False,
            doc="Whether to up-weight onsite matrix-element errors in the early stage of training. "
                "If True, the onsite part of the loss is multiplied by a time-decaying factor "
                "that starts from `onsite_boost_max` and decays to 1.0 over `onsite_boost_steps` iterations. "
                "Default: False.",
        ),
        Argument(
            "onsite_boost_steps",
            int,
            optional=True,
            default=50000,
            doc="Number of iterations over which the onsite loss weight decays linearly from "
                "`onsite_boost_max` to 1.0. Only used when `onsite_boost=True`. Default: 20000.",
        ),
        Argument(
            "onsite_boost_max",
            float,
            optional=True,
            default=200.0,
            doc="Initial multiplicative factor for onsite loss when `onsite_boost=True`. "
                "At iteration 0 the onsite loss is multiplied by this value, then linearly decays "
                "to 1.0 at `onsite_boost_steps`. Default: 100.0.",
        ),
        Argument(
            "z_loss_coef",
            float,
            optional=True,
            default=0,
            doc="Coefficient used to punish the unbalance of expert workload",
        ),

    ]

    property_aux = [
        Argument("model_basis_name", str, optional=True, default='def2svp', doc=doc_model_basis_name),
        Argument("on_the_fly_ovp_flag", bool, optional=True, default=True, doc=doc_on_the_fly_ovp_flag),
        Argument("dataset_basis_name", str, optional=True, default='def2svp', doc=doc_dataset_basis_name),
        Argument("num_e_loss_weight", float, optional=True, default=0.01)
    ]

    wa_loss_aux = [
        Argument("model_basis_name", str, optional=True, default='def2svp', doc=doc_model_basis_name),
        Argument("use_energy_weighting", bool, optional=True, default=True, doc=doc_use_energy_weighting),
        Argument("add_ham_flag", bool, optional=True, default=True, doc=doc_add_ham_flag),
        Argument("on_the_fly_solve_eigen", bool, optional=True, default=True, doc=doc_on_the_fly_solve_eigen),
        Argument("dataset_basis_name", str, optional=True, default='def2svp', doc=doc_dataset_basis_name)
    ]

    wt = [
        Argument("onsite_weight", [int, float, dict], optional=True, default=1., doc="Whether to use onsite shift in loss function. Default: False"),
        Argument("hopping_weight", [int, float, dict], optional=True, default=1., doc="Whether to use onsite shift in loss function. Default: False"),
    ]

    eigvals = [
        Argument("diff_on", bool, optional=True, default=False, doc="Whether to use random differences in loss function. Default: False"),
        Argument("eout_weight", float, optional=True, default=0.001, doc="The weight of eigenvalue out of range. Default: 0.01"),
        Argument("diff_weight", float, optional=True, default=0.1, doc="The weight of eigenvalue difference. Default: 0.01"),
        Argument("diff_valence", [dict,None], optional=True, default=None, doc="set the difference of the number of valence electrons in DFT and TB. eg {'A':6,'B':7}, Default: None, which means no difference"),
        Argument("spin_deg", int, optional=True, default=2, doc="The spin degeneracy of band structure. Default: 2"),
    ]

    eig_ham = [
        Argument("coeff_ham", float, optional=True, default=1., doc="The coefficient of the hamiltonian penalty. Default: 1"),
        Argument("coeff_ovp", float, optional=True, default=1., doc="The coefficient of the hamiltonian penalty. Default: 1"),
    ]

    skints = [
        Argument("skdata", str, optional=False, doc="The path to the skfile or sk database."),
    ]

    hamil_blockwise = [
        Argument("pred_node_block_key", str, optional=True, default="node_hamil_blocks"),
        Argument("pred_edge_block_key", str, optional=True, default="edge_hamil_blocks"),
        Argument("target_node_block_key", str, optional=True, default="node_delta_hamil_blocks"),
        Argument("target_edge_block_key", str, optional=True, default="edge_delta_hamil_blocks"),
        Argument("target_node_shape_key", str, optional=True, default="node_delta_hamil_block_shape"),
        Argument("target_edge_shape_key", str, optional=True, default="edge_delta_hamil_block_shape"),
        Argument("optimization", str, optional=True, default="block_mae", doc="Supported: block_mae, block_l1_rmse, block_mae_mse, feature_compatible."),
        Argument("block_reduction", str, optional=True, default="global", doc="Supported: global or equal_onsite_hopping."),
        Argument("complex_reduction", str, optional=True, default="modulus", doc="Supported: modulus or real_imag."),
        Argument("log_feature_compatible", bool, optional=True, default=True),
        Argument("feature_log_no_grad", bool, optional=True, default=True),
        Argument("distributed_log_reduce", bool, optional=True, default=True),
        Argument("expose_component_sums", bool, optional=True, default=True),
        Argument("loss_weight", [int, float], optional=True, default=1.0, doc="Global multiplier applied only to the optimization loss. Set 10.0 for the QHFlow2 Hamiltonian objective; unlike LR scaling, this also changes gradient-clipping behavior."),
        Argument("eps", float, optional=True, default=1e-12),
    ]

    loss_args = Variant("method", [
        # Argument("hamil", dict, sub_fields=hamil),
        Argument("eigvals", dict, sub_fields=eigvals),
        Argument("skints", dict, sub_fields=skints),
        Argument("hamil_abs", dict, sub_fields=hamil),
        Argument("hamil_abs_element_avg", dict, sub_fields=hamil),
        Argument("hamil_abs_mae", dict, sub_fields=hamil),
        Argument("hamil_w_num_e", dict, sub_fields=property_aux),
        Argument("wa_loss", dict, sub_fields=wa_loss_aux),
        Argument("dip_loss", dict, sub_fields=property_aux),
        Argument("dip_loss_mae", dict, sub_fields=property_aux),
        Argument("hamil_blas", dict, sub_fields=hamil),
        Argument("hamil_wt", dict, sub_fields=hamil+wt),
        Argument("eig_ham", dict, sub_fields=hamil+eigvals+eig_ham),
        Argument("hamil_blockwise_nextham", dict, sub_fields=hamil_blockwise),
        Argument("hamil_block_abs", dict, sub_fields=hamil_blockwise),
    ], optional=False, doc=doc_method)



    args = [
        Argument("train", dict, optional=False, sub_fields=[], sub_variants=[loss_args], doc=doc_train),
        Argument("validation", dict, optional=True, sub_fields=[], sub_variants=[loss_args], doc=doc_validation),
        Argument("reference", dict, optional=True, sub_fields=[], sub_variants=[loss_args], doc=doc_reference),
        Argument("test", dict, optional=True, sub_fields=[], sub_variants=[loss_args], doc=doc_test),
    ]

    doc_loss_options = ""
    return Argument("loss_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_loss_options)


def _validate_p2_prior_full_h_contract(data):
    """Semantic checks for first-class P2/P23 absolute Full-H routes."""
    common = data.get("common_options", {})
    model = data.get("model_options", {})
    embedding_options = model.get("embedding", {})
    prediction_options = model.get("prediction", {})
    embedding_is_prior = embedding_options.get("method") == "lem_moe_v3_prior"
    prior_specs = {
        "p2": {
            "raw": "hamiltonian_p2",
            "node": "node_p2",
            "edge": "edge_p2",
            "node_blocks": "node_p2_blocks",
            "edge_blocks": "edge_p2_blocks",
            "label": "P2",
        },
        "p23": {
            "raw": "hamiltonian_p23",
            "node": "node_p23",
            "edge": "edge_p23",
            "node_blocks": "node_p23_blocks",
            "edge_blocks": "edge_p23_blocks",
            "label": "P23",
        },
    }
    embedding_prior_kind = str(
        embedding_options.get("prior_kind", "p2")
    ).strip().lower()
    if embedding_is_prior and embedding_prior_kind not in prior_specs:
        raise ValueError(
            "model_options.embedding.prior_kind must be 'p2' or 'p23'; "
            f"got {embedding_prior_kind!r}."
        )
    prior_spec = prior_specs.get(embedding_prior_kind, prior_specs["p2"])
    if embedding_is_prior:
        for option, expected in (
            ("prior_node_key", prior_spec["node"]),
            ("prior_edge_key", prior_spec["edge"]),
        ):
            actual = embedding_options.get(option)
            if actual != expected:
                raise ValueError(
                    f"model_options.embedding.prior_kind={embedding_prior_kind!r} "
                    f"requires {option}={expected!r}; got {actual!r}. Refusing "
                    "to mix P2 and P23 fields."
                )

    needs_prior_rme = bool(
        embedding_is_prior
        and embedding_options.get("use_prior_init", True)
        and (
            embedding_options.get("use_prior_node_init", True)
            or embedding_options.get("use_prior_edge_init", True)
        )
    )
    add_prior = bool(prediction_options.get("add_prior", False))
    add_h0 = bool(prediction_options.get("add_h0", False))
    if add_prior and not embedding_is_prior:
        raise ValueError(
            "prediction.add_prior=true requires method='lem_moe_v3_prior' so "
            "the RME conditioning and AO reconstruction cannot select different priors."
        )

    data_options_value = data.get("data_options", {})
    configured_splits = {
        split: data_options_value[split]
        for split in ("train", "validation", "reference", "test")
        if isinstance(data_options_value.get(split), dict)
    }
    loss_options_value = data.get("train_options", {}).get("loss_options", {})
    required_target_keys = {
        "target_node_block_key": "node_full_hamil_target_blocks",
        "target_edge_block_key": "edge_full_hamil_target_blocks",
        "target_node_shape_key": "node_full_hamil_target_block_shape",
        "target_edge_shape_key": "edge_full_hamil_target_block_shape",
    }

    def _require_absolute_target_keys(split, split_loss):
        for option, expected in required_target_keys.items():
            actual = split_loss.get(option)
            if actual != expected:
                raise ValueError(
                    f"train_options.loss_options.{split}.{option} must be "
                    f"{expected!r} for explicit absolute Full-H supervision; "
                    f"got {actual!r}."
                )

    def _is_sha256_hex(value):
        text = str(value).strip().lower()
        return len(text) == 64 and all(
            character in "0123456789abcdef" for character in text
        )

    # Dedicated Full-H supervision is a dataset/loss contract, not a property
    # of the P2 model alone.  Lock direct-Full-H arms to the same target fields.
    for split, split_options in configured_splits.items():
        if not bool(split_options.get("require_full_h_target", False)):
            continue
        expected_h0_source = str(
            split_options.get("expected_physical_h0_source_fingerprint", "")
        ).strip()
        if expected_h0_source and not _is_sha256_hex(expected_h0_source):
            raise ValueError(
                f"data_options.{split}.expected_physical_h0_source_fingerprint "
                "must be a 64-character SHA256 hex digest."
            )
        split_loss = loss_options_value.get(split)
        if isinstance(split_loss, dict):
            _require_absolute_target_keys(split, split_loss)

    if not embedding_is_prior and not add_prior:
        return

    if bool(common.get("has_soc", False)):
        raise ValueError(
            "lem_moe_v3_prior/add_prior is non-SOC only; set "
            "common_options.has_soc=false."
        )
    if add_prior and add_h0:
        raise ValueError("prediction.add_prior and prediction.add_h0 are mutually exclusive.")
    if (
        add_prior
        and prediction_options.get("method") == "e3tb"
        and not bool(prediction_options.get("blockwise_hamiltonian", False))
    ):
        raise ValueError(
            "e3tb prediction.add_prior=true requires blockwise_hamiltonian=true."
        )

    if add_prior:
        for option, expected in (
            ("prior_node_block_field", prior_spec["node_blocks"]),
            ("prior_edge_block_field", prior_spec["edge_blocks"]),
        ):
            actual = prediction_options.get(option)
            if actual != expected:
                raise ValueError(
                    f"prediction.add_prior=true with prior_kind="
                    f"{embedding_prior_kind!r} requires {option}={expected!r}; "
                    f"got {actual!r}."
                )
        prior_label = str(prediction_options.get("prior_label", "")).strip().upper()
        if prior_label != prior_spec["label"]:
            raise ValueError(
                f"prediction.prior_label must be {prior_spec['label']!r} for "
                f"prior_kind={embedding_prior_kind!r}; got {prior_label!r}."
            )

    for split, split_options in configured_splits.items():
        if (needs_prior_rme or add_prior) and not bool(split_options.get("get_P2", False)):
            raise ValueError(
                f"data_options.{split}.get_P2 must be true for the "
                f"{prior_spec['label']} prior route."
            )
        split_prior_kind = str(split_options.get("prior_kind", "p2")).strip().lower()
        if split_prior_kind != embedding_prior_kind:
            raise ValueError(
                f"data_options.{split}.prior_kind={split_prior_kind!r} does not "
                f"match embedding prior_kind={embedding_prior_kind!r}."
            )
        if split_options.get("p2_key", "hamiltonian_p2") != prior_spec["raw"]:
            raise ValueError(
                f"data_options.{split}.prior_kind={embedding_prior_kind!r} "
                f"requires p2_key={prior_spec['raw']!r}; got "
                f"{split_options.get('p2_key')!r}."
            )
        if not bool(split_options.get("get_Hamiltonian", False)):
            raise ValueError(
                f"data_options.{split}.get_Hamiltonian must be true for "
                "prior-conditioned Full-H supervision."
            )
        if bool(split_options.get("residual_hamiltonian", False)):
            raise ValueError(
                f"data_options.{split}.residual_hamiltonian must be false: "
                "both direct and prior-plus-correction heads are supervised "
                "against explicit absolute Full H."
            )
        if not bool(split_options.get("require_full_h_target", False)):
            raise ValueError(
                f"data_options.{split}.require_full_h_target must be true for "
                "prior-conditioned direct or residual Full-H training."
            )
        expected_source_fingerprint = str(
            split_options.get("expected_p2_source_fingerprint", "")
        ).strip()
        allow_unbound_source = bool(
            split_options.get("allow_unbound_prior_source_fingerprint", False)
        )
        if expected_source_fingerprint:
            if not _is_sha256_hex(expected_source_fingerprint):
                raise ValueError(
                    f"data_options.{split}.expected_p2_source_fingerprint must "
                    "be a non-empty 64-character SHA256 hex digest for "
                    "prior-conditioned Full-H training."
                )
        elif not allow_unbound_source:
            raise ValueError(
                f"data_options.{split}.expected_p2_source_fingerprint is "
                "required for production prior-conditioned Full-H training. "
                "Set allow_unbound_prior_source_fingerprint=true only for "
                "synthetic/dev configs."
            )
        require_blocks = bool(split_options.get("require_p2_blocks", False))
        if add_prior and not require_blocks:
            raise ValueError(
                f"data_options.{split}.require_p2_blocks must be true when "
                "prediction.add_prior=true."
            )
        if not add_prior and require_blocks:
            raise ValueError(
                f"data_options.{split}.require_p2_blocks must be false for the "
                "direct Full-H head; AO prior blocks are not consumed."
            )

    full_node_key = prediction_options.get(
        "full_output_node_field", "node_full_hamil_blocks"
    )
    full_edge_key = prediction_options.get(
        "full_output_edge_field", "edge_full_hamil_blocks"
    )
    for split, split_loss in loss_options_value.items():
        if not isinstance(split_loss, dict):
            continue
        if split_loss.get("method") not in {
            "hamil_blockwise_nextham",
            "hamil_block_abs",
        }:
            raise ValueError(
                f"train_options.loss_options.{split}.method must use a blockwise "
                "Hamiltonian loss for prior-conditioned Full-H training."
            )
        pred_node_key = split_loss.get("pred_node_block_key", "node_hamil_blocks")
        pred_edge_key = split_loss.get("pred_edge_block_key", "edge_hamil_blocks")
        expected_pred = (
            (full_node_key, full_edge_key)
            if add_prior
            else ("node_hamil_blocks", "edge_hamil_blocks")
        )
        if (pred_node_key, pred_edge_key) != expected_pred:
            mode = "reconstructed Full-H (prior-plus-correction)" if add_prior else "direct Full-H"
            raise ValueError(
                f"train_options.loss_options.{split} must read {mode} "
                f"fields {expected_pred[0]!r}/{expected_pred[1]!r}, got "
                f"{pred_node_key!r}/{pred_edge_key!r}."
            )
        _require_absolute_target_keys(split, split_loss)


# def normalize_restart(data):

#     co = common_options()
#     da = data_options()

#     base = Argument("base", dict, [co, da])
#     data = base.normalize_value(data)
#     # data = base.normalize_value(data, trim_pattern="_*")
#     base.check_value(data, strict=True)

#     # add check loss and use wannier:

#     # if data['data_options']['use_wannier']:
#     #     if not data['loss_options']['losstype'] .startswith("block"):
#     #         log.info(msg='\n Warning! set data_options use_wannier true, but the loss type is not block_l2! The the wannier TB will not be used when training!\n')

#     # if data['loss_options']['losstype'] .startswith("block"):
#     #     if not data['data_options']['use_wannier']:
#     #         log.error(msg="\n ERROR! for block loss type, must set data_options:use_wannier True\n")
#     #         raise ValueError

#     return data

# def normalize_init_model(data):

#     co = common_options()
#     da = data_options()
#     tr = train_options()

#     base = Argument("base", dict, [co, da, tr])
#     data = base.normalize_value(data)
#     # data = base.normalize_value(data, trim_pattern="_*")
#     base.check_value(data, strict=True)

#     # add check loss and use wannier:

#     # if data['data_options']['use_wannier']:
#     #     if not data['loss_options']['losstype'] .startswith("block"):
#     #         log.info(msg='\n Warning! set data_options use_wannier true, but the loss type is not block_l2! The the wannier TB will not be used when training!\n')

#     # if data['loss_options']['losstype'] .startswith("block"):
#     #     if not data['data_options']['use_wannier']:
#     #         log.error(msg="\n ERROR! for block loss type, must set data_options:use_wannier True\n")
#     #         raise ValueError

#     return data

def normalize_test(data):

    co = common_options()
    da = test_data_options()
    to = test_options()

    loss_opts = data.get("test_options", {}).get("loss_options", {})
    if isinstance(loss_opts, dict) and "test" in loss_opts and "train" not in loss_opts:
        loss_opts["train"] = loss_opts["test"]

    base = Argument("base", dict, [co, da, to])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data




def tbtrans_negf():
    doc_scf = ""
    doc_block_tridiagonal = ""
    doc_ele_T = ""
    doc_unit = ""
    doc_scf_options = ""
    doc_stru_options = ""
    doc_poisson_options = ""
    doc_sgf_solver = ""
    doc_espacing = ""
    doc_emin = ""
    doc_emax = ""
    doc_e_fermi = ""
    doc_eta_lead = ""
    doc_eta_device = ""
    doc_out_dos = ""
    doc_out_tc = ""
    doc_out_current = ""
    doc_out_current_nscf = ""
    doc_out_ldos = ""
    doc_out_density = ""
    doc_out_lcurrent = ""
    doc_density_options = ""
    doc_out_potential = ""

    return [
        Argument("scf", bool, optional=True, default=False, doc=doc_scf),
        Argument("block_tridiagonal", bool, optional=True, default=False, doc=doc_block_tridiagonal),
        Argument("ele_T", [float, int], optional=False, doc=doc_ele_T),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit),
        Argument("scf_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[scf_options()], doc=doc_scf_options),
        Argument("stru_options", dict, optional=False, sub_fields=stru_options(), doc=doc_stru_options),
        Argument("poisson_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[poisson_options()], doc=doc_poisson_options),
        Argument("sgf_solver", str, optional=True, default="Sancho-Rubio", doc=doc_sgf_solver),
        Argument("espacing", [int, float], optional=False, doc=doc_espacing),
        Argument("emin", [int, float], optional=False, doc=doc_emin),
        Argument("emax", [int, float], optional=False, doc=doc_emax),
        Argument("e_fermi", [int, float], optional=False, doc=doc_e_fermi),
        Argument("density_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[density_options()], doc=doc_density_options),
        Argument("eta_lead", [int, float], optional=True, default=1e-5, doc=doc_eta_lead),
        Argument("eta_device", [int, float], optional=True, default=0., doc=doc_eta_device),
        Argument("out_dos", bool, optional=True, default=False, doc=doc_out_dos),
        Argument("out_tc", bool, optional=True, default=False, doc=doc_out_tc),
        Argument("out_density", bool, optional=True, default=False, doc=doc_out_density),
        Argument("out_potential", bool, optional=True, default=False, doc=doc_out_potential),
        Argument("out_current", bool, optional=True, default=False, doc=doc_out_current),
        Argument("out_current_nscf", bool, optional=True, default=False, doc=doc_out_current_nscf),
        Argument("out_ldos", bool, optional=True, default=False, doc=doc_out_ldos),
        Argument("out_lcurrent", bool, optional=True, default=False, doc=doc_out_lcurrent)
    ]





def negf():
    doc_scf = ""
    doc_block_tridiagonal = ""
    doc_ele_T = ""
    doc_unit = ""
    doc_scf_options = ""
    doc_stru_options = ""
    doc_poisson_options = ""
    doc_sgf_solver = ""
    doc_espacing = ""
    doc_emin = ""
    doc_emax = ""
    doc_e_fermi = ""
    doc_eta_lead = ""
    doc_eta_device = ""
    doc_out_dos = ""
    doc_out_tc = ""
    doc_out_current = ""
    doc_out_current_nscf = ""
    doc_out_ldos = ""
    doc_out_density = ""
    doc_out_lcurrent = ""
    doc_density_options = ""
    doc_out_potential = ""

    return [
        Argument("scf", bool, optional=True, default=False, doc=doc_scf),
        Argument("block_tridiagonal", bool, optional=True, default=False, doc=doc_block_tridiagonal),
        Argument("ele_T", [float, int], optional=False, doc=doc_ele_T),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit),
        Argument("scf_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[scf_options()], doc=doc_scf_options),
        Argument("stru_options", dict, optional=False, sub_fields=stru_options(), doc=doc_stru_options),
        Argument("poisson_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[poisson_options()], doc=doc_poisson_options),
        Argument("sgf_solver", str, optional=True, default="Sancho-Rubio", doc=doc_sgf_solver),
        Argument("espacing", [int, float], optional=False, doc=doc_espacing),
        Argument("emin", [int, float], optional=False, doc=doc_emin),
        Argument("emax", [int, float], optional=False, doc=doc_emax),
        Argument("e_fermi", [int, float], optional=False, doc=doc_e_fermi),
        Argument("density_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[density_options()], doc=doc_density_options),
        Argument("eta_lead", [int, float], optional=True, default=1e-5, doc=doc_eta_lead),
        Argument("eta_device", [int, float], optional=True, default=0., doc=doc_eta_device),
        Argument("out_dos", bool, optional=True, default=False, doc=doc_out_dos),
        Argument("out_tc", bool, optional=True, default=False, doc=doc_out_tc),
        Argument("out_density", bool, optional=True, default=False, doc=doc_out_density),
        Argument("out_potential", bool, optional=True, default=False, doc=doc_out_potential),
        Argument("out_current", bool, optional=True, default=False, doc=doc_out_current),
        Argument("out_current_nscf", bool, optional=True, default=False, doc=doc_out_current_nscf),
        Argument("out_ldos", bool, optional=True, default=False, doc=doc_out_ldos),
        Argument("out_lcurrent", bool, optional=True, default=False, doc=doc_out_lcurrent)
    ]

def stru_options():
    doc_kmesh = ""
    doc_pbc = ""
    doc_device = ""
    doc_lead_L = ""
    doc_lead_R = ""
    doc_gamma_center=""
    doc_time_reversal_symmetry=""
    return [
        Argument("device", dict, optional=False, sub_fields=device(), doc=doc_device),
        Argument("lead_L", dict, optional=False, sub_fields=lead(), doc=doc_lead_L),
        Argument("lead_R", dict, optional=False, sub_fields=lead(), doc=doc_lead_R),
        Argument("kmesh", list, optional=True, default=[1,1,1], doc=doc_kmesh),
        Argument("pbc", list, optional=True, default=[False, False, False], doc=doc_pbc),
        Argument("gamma_center", list, optional=True, default=True, doc=doc_gamma_center),
        Argument("time_reversal_symmetry", list, optional=True, default=True, doc=doc_time_reversal_symmetry)
    ]

def device():
    doc_id=""
    doc_sort=""

    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("sort", bool, optional=True, default=True, doc=doc_sort)
    ]

def lead():
    doc_id=""
    doc_voltage=""

    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("voltage", [int, float], optional=False, doc=doc_voltage)
    ]

def scf_options():
    doc_mode = ""
    doc_PDIIS = ""

    return Variant("mode", [
        Argument("PDIIS", dict, PDIIS(), doc=doc_PDIIS)
        ], optional=True, default_tag="PDIIS", doc=doc_mode)

def PDIIS():
    doc_mixing_period = ""
    doc_step_size = ""
    doc_n_history = ""
    doc_abs_err = ""
    doc_rel_err = ""
    doc_max_iter = ""

    return [
        Argument("mixing_period", int, optional=True, default=3, doc=doc_mixing_period),
        Argument("step_size", [int, float], optional=True, default=0.05, doc=doc_step_size),
        Argument("n_history", int, optional=True, default=6, doc=doc_n_history),
        Argument("abs_err", [int, float], optional=True, default=1e-6, doc=doc_abs_err),
        Argument("rel_err", [int, float], optional=True, default=1e-4, doc=doc_rel_err),
        Argument("max_iter", int, optional=True, default=100, doc=doc_max_iter)
    ]

def poisson_options():
    doc_solver = ""
    doc_fmm = ""
    return Variant("solver", [
        Argument("fmm", dict, fmm(), doc=doc_fmm)
    ], optional=True, default_tag="fmm", doc=doc_solver)

def density_options():
    doc_method = ""
    doc_Ozaki = ""
    return Variant("method", [
        Argument("Ozaki", dict, Ozaki(), doc=doc_method)
    ], optional=True, default_tag="Ozaki", doc=doc_Ozaki)

def Ozaki():
    doc_M_cut = ""
    doc_R = ""
    doc_n_gauss = ""
    return [
        Argument("R", [int, float], optional=True, default=1e6, doc=doc_R),
        Argument("M_cut", int, optional=True, default=30, doc=doc_M_cut),
        Argument("n_gauss", int, optional=True, default=10, doc=doc_n_gauss),
    ]

def fmm():
    doc_err = ""

    return [
        Argument("err", [int, float], optional=True, default=1e-5, doc=doc_err)
    ]

def run_options():
    doc_task = "the task to run, includes: band, dos, pdos, FS2D, FS3D, ifermi"
    doc_structure = "the structure to run the task"
    doc_gui = "To use the GUI or not"
    doc_device = "The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: None. default None means to use the device seeting in the model ckpt file."
    doc_dtype = """The digital number's precison, choose among: 
                    Default: None,
                        - `float32`: indicating torch.float32
                        - `float64`: indicating torch.float64
                    default None means to use the device seeting in the model ckpt file.
                """
    doc_pbc = """The periodic boundary condition, choose among: 
                    Default: True,
                        - True: indicating the structure is periodic
                        - False: indicating the structure is not periodic
                        - list of bool: indicating the structure is periodic in x,y,z direction respectively.
                """

    args = [
        Argument("task_options", dict, sub_fields=[], optional=True, sub_variants=[task_options()], doc = doc_task),
        Argument("structure", [str,None], optional=True, default=None, doc = doc_structure),
        Argument("pbc", [None, bool, list], optional=True, doc=doc_pbc, default=None),
        Argument("use_gui", bool, optional=True, default=False, doc = doc_gui),
        Argument("device", [str,None], optional = True, default=None, doc = doc_device),
        Argument("dtype", [str,None], optional = True, default=None, doc = doc_dtype),
        AtomicData_options_sub()
    ]

    return Argument("run_op", dict, args)

def normalize_run(data):

    run_op = run_options()
    data = run_op.normalize_value(data)
    run_op.check_value(data, strict=True)

    return data

def task_options():
    doc_task = '''The string define the task DeePTB conduct, includes: 
                    - `band`: for band structure plotting. 
                    - `dos`: for density of states plotting.
                    - `pdos`: for projected density of states plotting.
                    - `FS2D`: for 2D fermi-surface plotting.
                    - `FS3D`: for 3D fermi-surface plotting.
                    - `write_sk`: for transcript the nnsk model to standard sk parameter table
                    - `ifermi`: for fermi surface plotting.
                    - `negf`: for non-equilibrium green function calculation.
                    - `tbtrans_negf`: for non-equilibrium green function calculation with tbtrans.
                '''
    write_block = []

    return Variant("task", [
            Argument("band", dict, band()),
            Argument("dos", dict, dos()),
            Argument("pdos", dict, pdos()),
            Argument("FS2D", dict, FS2D()),
            Argument("FS3D", dict, FS3D()),
            Argument("write_sk", dict, write_sk()),
            Argument("ifermi", dict, ifermi()),
            Argument("negf", dict, negf()),
            Argument("tbtrans_negf", dict, tbtrans_negf()),
            Argument("write_block", dict, write_block),
        ],optional=False, doc=doc_task)

def band():
    doc_kline_type ="""The different type to build kpath line mode.
                    - "abacus" : the abacus format 
                    - "vasp" : the vasp format
                    - "ase" : the ase format
                    """
    doc_kpath = "for abacus, this is list of list of float, for vasp it is a list[str] to specify the kpath."
    doc_klabels = "the labels for high symmetry kpoint"
    doc_emin="the min energy to show the band plot"
    doc_emax="the max energy to show the band plot"
    doc_E_fermi = "the fermi level used to plot band"
    doc_ref_band = "the reference band structure to be ploted together with dptb bands."
    doc_nel_atom = "the valence electron number of each type of atom."
    doc_high_sym_kpoints = "the high symmetry kpoints dict, e.g. {'G':[0,0,0],'K':[0.5,0.5,0]}, only used for kline_type is vasp"
    doc_num_in_line = "the number of kpoints in each line path, only used for kline_type is vasp."
    doc_override_overlap = "overlap file path to be input to override overlap matrix."
    doc_eig_solver = "the eigenvalue solver to be used."
    return [
        Argument("kline_type", str, optional=False, doc=doc_kline_type),
        Argument("kpath", [str,list], optional=False, doc=doc_kpath),
        Argument("high_sym_kpoints",dict,optional=True,default={},doc=doc_high_sym_kpoints),
        Argument("number_in_line", int, optional=True, default=None, doc=doc_num_in_line),
        Argument("klabels", list, optional=True, default=[''], doc=doc_klabels),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("emin", [float, int, None], optional=True, doc=doc_emin, default=None),
        Argument("emax", [float, int, None], optional=True, doc=doc_emax, default=None),
        Argument("nkpoints", int, optional=True, doc=doc_emax, default=0),
        Argument("ref_band", [str, None], optional=True, default=None, doc=doc_ref_band),
        Argument("nel_atom", [dict,None], optional=True, default=None, doc=doc_nel_atom),
        Argument("override_overlap", [str, None], optional=True, default=None, doc=doc_override_overlap),
        Argument("eig_solver", [str, None], optional=True, default=None, doc=doc_eig_solver)
    ]


def dos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""
    doc_E_fermi=""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("gamma_center", bool, optional=True, default=False, doc=doc_gamma_center)
    ]

def pdos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""
    doc_E_fermi=""
    doc_atom_index = ""
    doc_orbital_index = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("atom_index", list, optional=False, doc=doc_atom_index),
        Argument("orbital_index", list, optional=False, doc=doc_orbital_index),
        Argument("gamma_center", bool, optional=True, default=False, doc=doc_gamma_center)
    ]

def FS2D():
    doc_mesh_grid = ""
    doc_E0 = ""
    doc_sigma = ""
    doc_intpfactor = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("E0", int, optional=False, doc=doc_E0),
        Argument("intpfactor", int, optional=False, doc=doc_intpfactor)
    ]

def FS3D():
    doc_mesh_grid = ""
    doc_E0 = ""
    doc_sigma = ""
    doc_intpfactor = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("E0", int, optional=False, doc=doc_E0),
        Argument("intpfactor", int, optional=False, doc=doc_intpfactor)
    ]


def ifermi():
    doc_fermi = ""
    doc_prop = ""
    doc_mesh_grid = ""
    doc_mu = ""
    doc_sigma = ""
    doc_intpfactor = ""
    doc_wigner_seitz = ""
    doc_nworkers = ""
    doc_plot_type = "plot_type: Method used for plotting. Valid options are: matplotlib, plotly, mayavi, crystal_toolkit."
    doc_use_gui=""
    doc_plot_fs_bands = ""
    doc_fs_plane = ""
    doc_fs_distanc= ""
    doc_color_properties ="""color_properties: Whether to use the properties to color the Fermi surface.
                If the properties is a vector then the norm of the properties will be
                used. Note, this will only take effect if the Fermi surface has
                properties. If set to True, the viridis colormap will be used.
                Alternative colormaps can be selected by setting ``color_properties``
                to a matplotlib colormap name. This setting will override the ``colors``
                option. For vector properties, the arrows are colored according to the
                norm of the properties by default. If used in combination with the
                ``projection_axis`` option, the color will be determined by the dot
                product of the properties with the projection axis."""
    doc_fs_plot_options=""
    doc_projection_axis = """projection_axis: Projection axis that can be used to calculate the color of
                vector properties. If None, the norm of the properties will be used,
                otherwise the color will be determined by the dot product of the
                properties with the projection axis. Only has an effect when used with
                the ``vector_properties`` option."""

    doc_velocity = ""
    doc_colormap = ""
    doc_prop_plane = ""
    doc_prop_distance=""
    doc_prop_plot_options=""
    doc_hide_surface = """hide_surface: Whether to hide the Fermi surface. Only recommended in combination with the ``vector_properties`` option."""
    doc_hide_labels ="""hide_labels: Whether to show the high-symmetry k-point labels."""
    doc_hide_cell = """hide_cell: Whether to show the reciprocal cell boundary."""
    doc_vector_spacing="""vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                for resampling the Fermi surface to ensure that arrows are not too close
                together. Only has an effect when used with the ``vector_properties``
                option."""
    doc_azimuth="azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane."
    doc_elevation="The zenith angle of the viewpoint in degrees, i.e. the angle subtended by the position vector and the z-axis."
    doc_colors ="""The color specification for the iso-surfaces. Valid options are:
                - A single color to use for all Fermi surfaces, specified as a tuple of
                  rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                - A list of colors, specified as above.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                  colors are specified as above.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the default colors will be used.
                """

    """Defaults."""

    AZIMUTH = 45.0
    ELEVATION = 35.0
    VECTOR_SPACING = 0.2
    COLORMAP = "viridis"
    SYMPREC = 1e-3
    KTOL = 1e-5
    SCALE = 4


    plot_options=[
        Argument("colors", [str,dict,list,None], optional=True, default=None, doc=doc_colors),
        Argument("projection_axis", [list,None], optional=True, default=None, doc=doc_projection_axis),
        Argument("hide_surface", bool, optional=True, default=False, doc=doc_hide_surface),
        Argument("hide_labels", bool, optional=True, default=False, doc=doc_hide_labels),
        Argument("hide_cell", bool, optional=True, default=False, doc=doc_hide_cell),
        Argument("vector_spacing",float, optional=True, default=VECTOR_SPACING, doc=doc_vector_spacing),
        Argument("azimuth", float, optional=True, default=AZIMUTH, doc=doc_azimuth),
        Argument("elevation", float, optional=True, default=ELEVATION, doc=doc_elevation),
    ]


    plot_options_fs=[
        Argument("projection_axis", [list,None], optional=True, default=None, doc=doc_projection_axis)
    ]
    args_fermi = [
        Argument("mesh_grid", list, optional = False, default=[2,2,2], doc = doc_mesh_grid),
        Argument("mu", [float,int], optional = False, default=0.0, doc = doc_mu),
        Argument("sigma", float, optional = True, default=0.1, doc = doc_sigma),
        Argument("intpfactor", int, optional = False, default=1, doc = doc_intpfactor),
        Argument("wigner_seitz", bool, optional = True, default=True, doc = doc_wigner_seitz),
        Argument("nworkers", int, optional = True, default=-1, doc = doc_nworkers),
        Argument("plot_type", str, optional = True, default="plotly", doc = doc_plot_type),
        Argument("use_gui", bool, optional = True, default=False, doc = doc_use_gui),
        Argument("plot_fs_bands", bool, optional = True, default = False, doc = doc_plot_fs_bands),
        Argument("fs_plane", list, optional = True, default=[0,0,1], doc = doc_fs_plane),
        Argument("fs_distance", [int,float], optional = True, default=0, doc = doc_fs_distanc),
        Argument("plot_options", dict, optional=True, sub_fields=plot_options, sub_variants=[], default={}, doc=doc_fs_plot_options)
    ]


    args_prop = [
        Argument("velocity", bool, optional = True, default=False, doc = doc_velocity),
        Argument("color_properties", [str,bool], optional = True, default=False, doc = doc_color_properties),
        Argument("colormap", str, optional = True,default="viridis",doc = doc_colormap),
        Argument("prop_plane", list, optional = True, default=[0,0,1],doc = doc_prop_plane),
        Argument("prop_distance", [int,float], optional = True, default=0, doc = doc_prop_distance),
        Argument("plot_options", dict, optional = True, sub_fields=plot_options, sub_variants=[], default={}, doc = doc_prop_plot_options)
    ]

    fermiarg = Argument("fermisurface", dict, optional=False, sub_fields=args_fermi, sub_variants=[], default={}, doc=doc_fermi)
    prop = Argument("property", dict, optional=True, sub_fields=args_prop, sub_variants=[], default={}, doc=doc_prop)

    return [fermiarg, prop]

def write_sk():
    doc_thr = ""
    doc_format = ""

    return [
        Argument("format", str, optional=True, default="sktable",  doc=doc_format),
        Argument("thr", float, optional=True, default=1e-3, doc=doc_thr)
    ]


def host_normalize(data):

    co = common_options()
    mo = model_options()

    base = Argument("base", dict, [co, mo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=False)

    return data


def normalize_bandinfo(data):
    doc_band_min = ""
    doc_band_max = ""
    doc_emin = ""
    doc_emax = ""
    doc_gap_penalty = ""
    doc_fermi_band = ""
    doc_loss_gap_eta = ""
    doc_eout_weight=""
    doc_weight = ""
    doc_wannier_proj = ""
    doc_orb_wan = ""

    args = [
        Argument("band_min", int, optional=True, doc=doc_band_min, default=0),
        Argument("band_max", [int, None], optional=True, doc=doc_band_max, default=None),
        Argument("emin", [float, None], optional=True, doc=doc_emin,default=None),
        Argument("emax", [float, None], optional=True, doc=doc_emax,default=None),
        Argument("gap_penalty", bool, optional=True, doc=doc_gap_penalty, default=False),
        Argument("fermi_band", int, optional=True, doc=doc_fermi_band,default=0),
        Argument("loss_gap_eta", float, optional=True, doc=doc_loss_gap_eta, default=0.01),
        Argument("eout_weight", float, optional=True, doc=doc_eout_weight, default=0.00),
        Argument("weight", [int, float, list], optional=True, doc=doc_weight, default=1.),
        Argument("wannier_proj",dict, optional=True, doc=doc_wannier_proj, default={}),
        Argument("orb_wan",[dict, None], optional=True, doc=doc_orb_wan, default=None)
    ]
    bandinfo = Argument("bandinfo", dict, sub_fields=args)
    data = bandinfo.normalize_value(data)
    bandinfo.check_value(data, strict=True)

    return data

def bandinfo_sub():
    doc_band_min = """the minum band index for the training band window with respected to the correctly selected DFT bands.
                   `important`: before setting this tag you should make sure you have already  exclude all the irrelevant in your training data.
                                This logic for band_min and max is based on the simple fact the total number TB bands > the bands you care.   
                   """
    doc_band_max = "The maxmum band index for training band window"
    doc_emin = "the minmum energy window, 0 meand the min value of the band at index band_min"
    doc_emax = "the max energy window, emax value is respect to the min value of the band at index band_min"

    args = [
        Argument("band_min", int, optional=True, doc=doc_band_min, default=0),
        Argument("band_max", [int, None], optional=True, doc=doc_band_max, default=None),
        Argument("emin", [float, None], optional=True, doc=doc_emin,default=None),
        Argument("emax", [float, None], optional=True, doc=doc_emax,default=None),
    ]

    return Argument("bandinfo", dict, optional=True, sub_fields=args, sub_variants=[], doc="")

def AtomicData_options_sub():
    doc_r_max = "the cutoff value for bond considering in TB model."
    doc_er_max = "The cutoff value for environment for each site for env correction model. should set for nnsk+env correction model."
    doc_oer_max = "The cutoff value for onsite environment for nnsk model, for now only need to set in strain and NRL mode."
    doc_pbc = "The periodic condition for the structure, can bool or list of bool to specific x,y,z direction."

    args = [
        Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max, default=4.0),
        Argument("er_max", [float, int, dict], optional=True, doc=doc_er_max, default=None),
        Argument("oer_max", [float, int, dict], optional=True, doc=doc_oer_max,default=None)
    ]

    return Argument("AtomicData_options", dict, optional=True, sub_fields=args, sub_variants=[], doc="", default=None)

def set_info_options():
    doc_nframes = "Number of frames in this trajectory."
    doc_natoms = "Number of atoms in each frame."
    doc_pos_type = "Type of atomic position input. Can be frac / cart / ase."
    doc_pbc = "The periodic condition for the structure, can bool or list of bool to specific x,y,z direction."

    args = [
        Argument("nframes", int, optional=False, doc=doc_nframes),
        Argument("natoms", int, optional=True, default=-1, doc=doc_natoms),
        Argument("pos_type", str, optional=False, doc=doc_pos_type),
        Argument("pbc", [bool, list], optional=False, doc=doc_pbc),
        bandinfo_sub()
    ]

    return Argument("setinfo", dict, sub_fields=args)

def lmdbset_info_options():
    doc_r_max = "the cutoff value for bond considering in TB model."

    args = [
        Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max, default=4.0)
    ]
    return Argument("setinfo", dict, sub_fields=args)

def normalize_setinfo(data):

    setinfo = set_info_options()
    data = setinfo.normalize_value(data)
    setinfo.check_value(data, strict=True)

    return data

def normalize_lmdbsetinfo(data):

    setinfo = lmdbset_info_options()
    data = setinfo.normalize_value(data)
    setinfo.check_value(data, strict=True)

    return data


def format_cuts(rcut: Union[Dict[str, Number], Number], decay_w: Number, nbuffer: int) -> Union[Dict[str, Number], Number]:
    if not isinstance(decay_w, Number) or decay_w <= 0:
        raise ValueError("decay_w should be a positive number")

    buffer_addition = decay_w * nbuffer

    if isinstance(rcut, dict):
        return {key: value + buffer_addition for key, value in rcut.items()}
    elif isinstance(rcut, Number):
        return rcut + buffer_addition
    else:
        raise TypeError("rcut should be a dict or a number")

def get_cutoffs_from_model_options(model_options):
    """
    Extract cutoff values from the provided model options.

    This function retrieves the cutoff values `r_max`, `er_max`, and `oer_max` from the `model_options` 
    dictionary. It handles different model types such as `embedding`, `nnsk`, and `dftbsk`, ensuring 
    that the appropriate cutoff values are provided and valid.

    Parameters:
    model_options (dict): A dictionary containing model configuration options. It may include keys 
                          like `embedding`, `nnsk`, and `dftbsk` with their respective cutoff values.

    Returns:
    tuple: A tuple containing the cutoff values (`r_max`, `er_max`, `oer_max`).

    Raises:
    ValueError: If neither `r_max` nor `rc` is provided in `model_options` for embedding.
    AssertionError: If `r_max` is provided outside the `nnsk` or `dftbsk` context when those models are used.

    Logs:
    Error messages if required cutoff values are missing or incorrectly provided.
    """
    r_max, er_max, oer_max = None, None, None
    if model_options.get("embedding",None) is not None:
        # switch according to the embedding method
        embedding = model_options.get("embedding")
        if embedding["method"] == "se2":
            er_max = embedding["rc"]
        elif embedding["method"] in ["slem", "lem", "lem_moe", "lem_moe_topk", "lem_moe_v3", "lem_moe_v3_edge", "lem_moe_v3_h0", "lem_moe_v3_prior", "lem_moe_v3_edge_h0", "lem_non_linear", "lem_non_linear_h0", "lem_charge", "emoles", "emoles_openequi_norm", "emoles_openequi_norm_v2", "emoles_openequi_eqv3", "emoles_openequi_eqv3_ffn", "emoles_openequi_nodeffn", "emoles_openequi", "lem_cutoff", "lem_full_tp_oeq", "lem_moe_openequi", "lem_in_frame_moe", "lem_full_tp", "lem_in_frame_e3nn", "lem_in_frame_openequi", "lem_wo_ln", "lem_in_frame", "lem_in_frame_heavy", "lem_light_v2", "lem_light", "lem_moe_charge", "lem_frame", "lem_high_order", "lem_so2_local", "lem_so2_global", "lem_local", "lem_global", "lem_so2", "trinity"]:
            r_max = embedding["r_max"]
        else:
            log.error("The method of embedding have not been defined in get cutoff functions")
            raise NotImplementedError("The method of embedding have not been defined in get cutoff functions")

    if model_options.get("nnsk", None) is not None:
        assert r_max is None, "r_max should not be provided in outside the nnsk for training nnsk model."
        if model_options["nnsk"]["hopping"].get("rs",None) is not None:
            # 其他方法在模型公式中，已经包含了 +5w 的范围，所以这里为了保险额外加上3w 的范围; 
            # 对于两个特例，powerlaw 和 varTang96 的情况，为了和旧版存档的兼容, 模型公式的本身并没有 +5w 的范围，所以这里额外加上8w 的范围。
            if model_options["nnsk"]["hopping"]['method'] in ["powerlaw","varTang96"]:
                # r_max = model_options["nnsk"]["hopping"]["rs"] + 8 * model_options["nnsk"]["hopping"]["w"]
                r_max = format_cuts(model_options["nnsk"]["hopping"]["rs"], model_options["nnsk"]["hopping"]["w"], 8)
            else:
                # r_max = model_options["nnsk"]["hopping"]["rs"] + 3 * model_options["nnsk"]["hopping"]["w"]
                r_max = format_cuts(model_options["nnsk"]["hopping"]["rs"], model_options["nnsk"]["hopping"]["w"], 3)

        if model_options["nnsk"]["onsite"].get("rs",None) is not None:
            if  model_options["nnsk"]["onsite"]['method'] == "strain" and model_options["nnsk"]["hopping"]['method'] in ["powerlaw","varTang96"]:
                # oer_max = model_options["nnsk"]["onsite"]["rs"] + 8 * model_options["nnsk"]["onsite"]["w"]
                oer_max = format_cuts(model_options["nnsk"]["onsite"]["rs"], model_options["nnsk"]["onsite"]["w"], 8)
            else:
                # oer_max = model_options["nnsk"]["onsite"]["rs"] + 3 * model_options["nnsk"]["onsite"]["w"]
                oer_max = format_cuts(model_options["nnsk"]["onsite"]["rs"], model_options["nnsk"]["onsite"]["w"], 3)

    elif model_options.get("dftbsk", None) is not None:
        assert r_max is None, "r_max should not be provided orther than the dftbsk param section for training dftbsk model."
        r_max = model_options["dftbsk"].get("r_max")

    else:
        # not nnsk not dftbsk, must be only env or E3. the embedding should be provided.
        assert model_options.get("embedding",None) is not None

    return r_max, er_max, oer_max
def collect_cutoffs(jdata):
    """
    Collect cutoff values from the provided JSON data.

    This function extracts the cutoff values `r_max`, `er_max`, and `oer_max` from the `model_options` 
    in the provided JSON data. If the `nnsk` push model is used, it ensures that the necessary 
    cutoff values are provided in `data_options` and overrides the values from `model_options` 
    accordingly.

    Parameters:
    jdata (dict): A dictionary containing model and data options. It must include `model_options` 
                  and optionally `data_options` if `nnsk` push model is used.

    Returns:
    dict: A dictionary containing the cutoff options with keys `r_max`, `er_max`, and `oer_max`.

    Raises:
    AssertionError: If required keys are missing in `jdata` or if `r_max` is not provided when 
                    using the `nnsk` push model.

    Logs:
    Various informational messages about the cutoff values and their sources.
    """

    model_options = jdata["model_options"]
    r_max, er_max, oer_max = get_cutoffs_from_model_options(model_options)

    if model_options.get("nnsk", None) is not None:
        if model_options["nnsk"]["push"] and \
            abs(model_options["nnsk"]["push"]['rs_thr']) + \
            abs(model_options["nnsk"]["push"]['rc_thr']) + \
            abs(model_options["nnsk"]["push"]['w_thr']) > 1e-8:
            assert jdata.get("data_options",None) is not None, "data_options should be provided in jdata for nnsk push"
            assert jdata['data_options'].get("r_max") is not None, "r_max should be provided in data_options for nnsk push"
            log.info('YOU ARE USING NNSK PUSH MODEL, r_max will be used from data_options. Be careful! check the value in data options and model options. r_max or rs/rc !')
            r_max = jdata['data_options']['r_max']

            if model_options["nnsk"]["onsite"]["method"] in ["strain", "NRL"]:
                assert jdata['data_options'].get("oer_max") is not None, "oer_max should be provided in data_options for nnsk push with strain onsite mode"
                log.info('YOU ARE USING NNSK PUSH MODEL with `strain` onsite mode, oer_max will be used from data_options. Be careful! check the value in data options and model options. rs/rc !')
                oer_max = jdata['data_options']['oer_max']

            if jdata['data_options'].get("er_max") is not None:
                log.info("IN PUSH mode, the env correction should not be used. the er_max will not take effect.")
        else:
            if  jdata['data_options'].get("r_max") is not None:
                log.info("When not nnsk/push. the cutoffs will take from the model options: r_max  rs and rc values. this seting in data_options will be ignored.")

    assert r_max is not None
    cutoff_options = ({"r_max": r_max, "er_max": er_max, "oer_max": oer_max})

    log.info("-"*66)
    log.info('     {:<55}    '.format("Cutoff options:"))
    log.info('     {:<55}    '.format(" "*30))
    log.info('     {:<16} : {:<36}    '.format("r_max", f"{r_max}"))
    log.info('     {:<16} : {:<36}    '.format("er_max", f"{er_max}"))
    log.info('     {:<16} : {:<36}    '.format("oer_max", f"{oer_max}"))
    log.info("-"*66)

    return cutoff_options


def normalize(data):

    co = common_options()
    tr = train_options()
    da = data_options()
    mo = model_options()

    base = Argument("base", dict, [co, tr, da, mo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)
    _validate_p2_prior_full_h_contract(data)
    validate_flow_loss_contract(data)
    validate_block_ode_contract(data)

    # add check loss and use wannier:

    # if data['data_options']['use_wannier']:
    #     if not data['loss_options']['losstype'] .startswith("block"):
    #         log.info(msg='\n Warning! set data_options use_wannier true, but the loss type is not block_l2! The the wannier TB will not be used when training!\n')

    # if data['loss_options']['losstype'] .startswith("block"):
    #     if not data['data_options']['use_wannier']:
    #         log.error(msg="\n ERROR! for block loss type, must set data_options:use_wannier True\n")
    #         raise ValueError

    return data

def normalize_skf2nnsk(data):
    common_ops = [
        Argument("basis", [dict,str], optional=False, default='auto', doc="The basis set for the model, can be a dict or a string, default is 'auto'."),
        Argument("skdata",str, optional=False, doc="The path to the skf file."),
        Argument("device",str, optional=True, default='cpu', doc="The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: 'cpu'."),
        Argument("dtype",str, optional=True, default='float32', doc="The digital number's precison, choose among: 'float32', 'float64', Default: 'float32'."),
        Argument("seed", int, optional=True, default=3982377700, doc="The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`")
    ]

    model_ops = [
        Argument('method',str, optional=False, default='poly2pow', doc="The method for the hopping term, default is 'powerlaw'."),
        Argument('rs',[float,None,int], optional=True, default=None, doc="The rs value for the hopping term."),
        Argument('w', [float,int], optional=True, default=0.2, doc="The w value for the hopping term."),
        Argument('atomic_radius',[str,dict], optional=True, default='cov', doc="The atomic radius for the hopping term, default is 'cov'.")
    ]

    doc_lr_scheduler = "The learning rate scheduler tools settings, the lr scheduler is used to scales down the learning rate during the training process. Proper setting can make the training more stable and efficient. The supported lr schedular includes: `Exponential Decaying (exp)`, `Linear multiplication (linear)`"
    doc_optimizer = "\
        The optimizer setting for selecting the gradient optimizer of model training. Optimizer supported includes `Adam`, `AdamW`, `SGD` and `LBFGS` \n\n\
        For more information about these optmization algorithm, we refer to:\n\n\
        - `Adam`: [Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980)\n\n\
        - `AdamW`: [AdamW: Decoupled Weight Decay Regularization.](https://arxiv.org/abs/1711.05101)\n\n\
        - `SGD`: [Stochastic Gradient Descent.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)\n\n\
        - `LBFGS`: [On the limited memory BFGS method for large scale optimization.](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf) \n\n\
    "

    train_ops = [
        Argument('nstep', int, optional=False, doc="The number of steps for the training."),
        Argument('nsample', int, optional=True, default=256, doc="The number of steps for the training."),
        Argument('max_elmt_batch', int, optional=True, default=4, doc="The max number of elements in a batch."),
        Argument('dis_freq', int, optional=True, default=1, doc="The frequency of the display."),
        Argument('save_freq', int, optional=True, default=1, doc="The frequency of the save."),
        Argument("optimizer", dict, sub_fields=[], optional=True, default={}, sub_variants=[optimizer()], doc = doc_optimizer),
        Argument("lr_scheduler", dict, sub_fields=[], optional=True, default={}, sub_variants=[lr_scheduler()], doc = doc_lr_scheduler)
    ]
    co = Argument("common_options", dict, optional=False, sub_fields=common_ops, sub_variants=[], doc='The common options.')
    mo = Argument("model_options", dict, optional=False, sub_fields=model_ops, sub_variants=[], doc='The model options.')
    tr =  Argument("train_options", dict, sub_fields=train_ops, sub_variants=[], optional=False, doc='The training options.')

    base = Argument("base", dict, [co, mo, tr])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data
