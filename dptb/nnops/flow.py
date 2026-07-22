from __future__ import annotations

"""Conditional Flow Matching utilities for Hamiltonian training.

This module is intentionally lightweight and trainer-side.  It does not require a
new DeePTB model class: at every training step it replaces the H0 node/edge
fields by an interpolated Hamiltonian state H_t and trains the existing model to
predict the clean converged Hamiltonian.  This mirrors the residual-CFM training
used by QHFlow/QHFlow2, but is adapted to DeePTB/NextHAM-style physical H0
features.
"""

import copy
import logging
import math
import re
from numbers import Integral
from typing import Any, Dict, Optional, Tuple

import torch

from dptb.configuration import canonicalize_flow_options
from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    attach_prediction_block_tensors,
    block_mask_from_shapes,
    strict_reverse_edge_index,
    infer_block_shapes,
    is_soc_uureal_mapper,
    mapper_max_norb,
)
from dptb.nnops import prior_physical
from dptb.nnops.block_flow_codec import (
    BlockStateCodec,
    _FLOW_PROJECTED_STATE_TOKEN,
    project_block_state,
)
from dptb.nnops.flow_context import CFMContext, _to_torch_dtype
from dptb.nnops import prior_calibration
from dptb.nnops.flow_priors import (
    BasisOnsiteFamily,
    DFTBSKFamily,
    ExternalFamily,
    HaarDMFamily,
    OverlapHuckelFamily,
    PriorContext,
    SplitPriorFamily,
    build_prior_families,
    BASIS_ONSITE_NAMES,
    DFTBSK_NAMES,
    EXTERNAL_NAMES,
    HAAR_DM_NAMES,
    OVERLAP_HUCKEL_NAMES,
)
from dptb.nnops.layout import normalize_idp_mask_layout, project_uureal_to_like
from dptb.nnops.tied_irrep_gaussian_prior import (
    TIED_IRREP_CANONICAL_IRREPS,
    TIED_IRREP_GAUSSIAN_PRIOR,
    TIED_IRREP_LATENT_WIDTH,
    draw_standard_tied_irrep_latent,
    effective_tied_irrep_latent,
    fill_tied_irrep_rme,
    normalize_tied_irrep_mode,
    validate_tied_irrep_options,
)

log = logging.getLogger(__name__)


_BLOCK_ODE_OUTPUT_ONLY_KEYS = (
    _keys.NODE_PRED_HAMIL_BLOCKS_KEY,
    _keys.EDGE_PRED_HAMIL_BLOCKS_KEY,
    _keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
    _keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
)
_LEM_INPUT_SIDECAR_KEYS = (
    _keys.LEM_ACTIVE_EDGES_KEY,
    _keys.LEM_CUTOFF_COEFFS_KEY,
    _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY,
)
_MetricStats = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_WeightedMetricStats = Tuple[Tuple[_MetricStats, float], ...]
_MAX_TORCH_SEED = (1 << 64) - 1
_VALIDATION_SEED_STREAMS = {
    "prior": 0x9E3779B97F4A7C15,
    "time": 0xD1B54A32D192ED03,
}
# TA-2: per-(graph uid, node-vs-edge) prior substream constants.  A SEEDED prior
# draw derives one independent generator per graph keyed by its stable
# ``sample_uid`` and its component so a graph's epsilon is invariant to batch
# composition/order/sharding.  These extend the splitmix-style ``validation_seed``
# machinery: the conceptual purpose string is ``"prior:<component>:<uid>"``.
_PRIOR_COMPONENT_STREAMS = {
    "node": 0xA0761D6478BD642F,
    "edge": 0xE7037ED1A0B428DB,
}
_PRIOR_UID_MULTIPLIER = 0xD6E8FEB86659FD93


def _parse_strict_certification(value: Any) -> Tuple[str, int]:
    cadence = str(value).strip().lower()
    if cadence == "always":
        return cadence, 1
    if cadence == "first_batch":
        return cadence, 0
    match = re.fullmatch(r"every_n\(([1-9][0-9]*)\)", cadence)
    if match is not None:
        return "every_n", int(match.group(1))
    raise ValueError(
        "flow_options.strict_certification must be 'always', 'first_batch', "
        "or 'every_n(N)' with N >= 1."
    )


class HamiltonianCFM:
    """Trainer-side residual conditional flow matching helper.

    DeePTB's NextHAM-like branch already supports physical initial Hamiltonian
    features through ``node_h0`` and ``edge_h0``.  CFM is implemented by
    replacing these fields with an interpolated state

        H_t = H_base + ((1 - t) * eps + t * (H_ref - H_base)),

    then asking the original model to predict ``H_ref``.  The loss is the
    endpoint parameterization of the CFM velocity loss,

        ||(H_pred - H_t)/(1 - t) - (H_ref - H_t)/(1 - t)||^2,

    i.e. optionally weighted endpoint error ``||H_pred - H_ref||^2/(1-t)^2``.
    """

    # Pixel MeanFlow owns a separate component-wise objective and explicitly
    # opts out below.  CFM's true global element reduction has no unambiguous
    # component-multiplier interpretation, especially for l1_rmse, so keep it
    # unit-weight-only and route weighted components through equal_components.
    allow_nonunit_global_element_weights = False

    def __init__(
        self,
        options: Optional[Dict[str, Any]],
        *,
        idp: Any = None,
        dtype: Any = torch.float32,
        device: Any = torch.device("cpu"),
    ) -> None:
        options = canonicalize_flow_options(options)
        self.enabled = bool(options.get("enabled", False))
        self.options = options
        self.idp = idp
        self.dtype = _to_torch_dtype(dtype)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.allow_complex_prior_real_projection = bool(
            options.get("allow_complex_prior_real_projection", False)
        )

        # Keys.  The defaults match DeePTB's NextHAM/H0 branch.
        self.node_h0_key = str(options.get("node_h0_key", _keys.NODE_H0_KEY))
        self.edge_h0_key = str(options.get("edge_h0_key", _keys.EDGE_H0_KEY))
        self.node_target_key = str(options.get("node_target_key", _keys.NODE_FEATURES_KEY))
        self.edge_target_key = str(options.get("edge_target_key", _keys.EDGE_FEATURES_KEY))
        self.flow_time_key = str(options.get("flow_time_key", "flow_time"))
        self.output_space = str(options.get("output_space", "rme")).lower().replace("-", "_")
        if self.output_space in {"block", "ao", "ao_blocks"}:
            self.output_space = "ao_block"
        self.block_ode = bool(options.get("block_ode", False))
        if self.output_space in {"block_ode", "ao_blocks_ode"}:
            self.output_space = "ao_block_ode"
        if self.output_space in {"spatial_uureal_residual_block_ode", "uureal_residual_block_ode"}:
            self.output_space = "uureal_block_ode"
        self.uureal_block_ode = self.output_space == "uureal_block_ode"
        if self.uureal_block_ode:
            self.block_ode = True
        # Non-SOC direct-residual block ODE. No normalization aliases: the only
        # accepted spelling is the canonical name (hyphens already normalized to
        # underscores above), so a near-miss alias fails the whitelist below.
        self.residual_ao_block_ode = self.output_space == "residual_ao_block_ode"
        if self.residual_ao_block_ode:
            self.block_ode = True
        if self.output_space not in {"rme", "ao_block", "ao_block_ode", "uureal_block_ode", "residual_ao_block_ode"}:
            raise ValueError(
                "flow_options.output_space must be 'rme', 'ao_block', or "
                "one of the explicit ODE modes 'ao_block_ode'/'uureal_block_ode'/"
                "'residual_ao_block_ode', got "
                f"{self.output_space!r}."
            )
        if self.block_ode and self.output_space not in {"ao_block_ode", "uureal_block_ode", "residual_ao_block_ode"}:
            raise ValueError(
                "flow_options.block_ode=true is mutually exclusive with the frozen "
                "ao_block adapter; set output_space='ao_block_ode'."
            )
        if self.output_space in {"ao_block_ode", "uureal_block_ode", "residual_ao_block_ode"} and not self.block_ode:
            raise ValueError(
                "Block-space ODE is a distinct mode: set flow_options.block_ode=true "
                "together with output_space='ao_block_ode'."
            )
        if self.output_space in {"ao_block_ode", "uureal_block_ode", "residual_ao_block_ode"} and not self.enabled:
            raise ValueError(f"output_space={self.output_space!r} requires flow_options.enabled=true.")
        self.node_output_key = str(
            options.get("node_output_key", getattr(_keys, "NODE_PRED_HAMIL_BLOCKS_KEY", "node_hamil_blocks"))
        )
        self.edge_output_key = str(
            options.get("edge_output_key", getattr(_keys, "EDGE_PRED_HAMIL_BLOCKS_KEY", "edge_hamil_blocks"))
        )
        self.node_block_target_key = str(
            options.get(
                "node_block_target_key",
                getattr(_keys, "NODE_DELTA_HAMIL_BLOCKS_KEY", "node_delta_hamil_blocks"),
            )
        )
        self.edge_block_target_key = str(
            options.get(
                "edge_block_target_key",
                getattr(_keys, "EDGE_DELTA_HAMIL_BLOCKS_KEY", "edge_delta_hamil_blocks"),
            )
        )
        self.node_block_shape_key = str(
            options.get(
                "node_block_shape_key",
                getattr(_keys, "NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY", "node_delta_hamil_block_shape"),
            )
        )
        self.edge_block_shape_key = str(
            options.get(
                "edge_block_shape_key",
                getattr(_keys, "EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY", "edge_delta_hamil_block_shape"),
            )
        )
        # Physical B0 is block-authoritative.  Legacy node_h0/edge_h0 feature
        # tensors have existed in more than one convention (AO-product and
        # coupled RME), so block ODE must never infer its start state from them.
        self.node_h0_block_key = getattr(_keys, "NODE_H0_BLOCKS_KEY", "node_h0_blocks")
        self.edge_h0_block_key = getattr(_keys, "EDGE_H0_BLOCKS_KEY", "edge_h0_blocks")
        self.node_h0_block_shape_key = getattr(
            _keys, "NODE_H0_BLOCK_SHAPE_KEY", "node_h0_block_shape"
        )
        self.edge_h0_block_shape_key = getattr(
            _keys, "EDGE_H0_BLOCK_SHAPE_KEY", "edge_h0_block_shape"
        )
        self.target_semantics = str(options.get("target_semantics", "")).lower().replace("-", "_")
        self.prediction_add_h0 = bool(options.get("prediction_add_h0", False))
        self.time_conditioning_required = bool(options.get("time_conditioning_required", False))
        self.block_inverse_mode = str(options.get("block_inverse_mode", "strict")).lower()
        configured_block_atol = options.get("block_inverse_atol", None)
        self.block_inverse_atol = float(
            (1e-10 if self.dtype == torch.float64 else 2e-5)
            if configured_block_atol is None
            else configured_block_atol
        )
        if not math.isfinite(self.block_inverse_atol) or self.block_inverse_atol < 0:
            raise ValueError("flow_options.block_inverse_atol must be finite and non-negative.")
        # ``canonicalize_flow_options`` folds the legacy strict_h0 and
        # warn_missing_h0 booleans into this single policy.  Establish the
        # compatibility views before the block-ODE guards below so those guards
        # cannot accidentally observe a removed legacy key's default value.
        self.missing_h0_policy = str(
            options.get("missing_h0_policy", "error")
        ).lower()
        if self.missing_h0_policy not in {"error", "warn_zero", "zero"}:
            raise ValueError(
                "flow_options.missing_h0_policy must be 'error', 'warn_zero', or 'zero'."
            )
        self.strict_h0 = self.missing_h0_policy == "error"
        self.warn_missing_h0 = self.missing_h0_policy == "warn_zero"
        self.strict_certification = str(
            options.get("strict_certification", "always")
        ).strip().lower()
        (
            self._strict_certification_mode,
            self._strict_certification_period,
        ) = _parse_strict_certification(self.strict_certification)
        self._strict_certification_batches = 0

        # Residual CFM is the recommended mode for DeePTB: base = DFT/NextHAM H0.
        self.mode = str(options.get("mode", "residual")).lower()
        if self.mode not in {"residual", "full"}:
            raise ValueError(f"Unsupported flow_options.mode={self.mode!r}; use 'residual' or 'full'.")

        # In DeePTB, zero prior means the inference start state is exactly physical H0.
        # Gaussian is available for QHFlow-style noisy residual priors.
        # The TE aliases are feature-space approximations over masked
        # node_h0/edge_h0 rows. They do not materialize dense Tensor-Expansion
        # or Clebsch-Gordan products and should not be interpreted as strict
        # dense TE/CG priors.
        self.prior = str(options.get("prior", "zero")).lower().replace("-", "_")
        self._te_prior_names = {"te", "structured_te", "block_te", "te_like"}
        self._projected_te_prior_names = {"projected_te"}
        self._tied_irrep_gaussian_prior_names = {TIED_IRREP_GAUSSIAN_PRIOR}
        self._basis_prior_names = BASIS_ONSITE_NAMES
        self._overlap_huckel_prior_names = OVERLAP_HUCKEL_NAMES
        self._haar_dm_prior_names = HAAR_DM_NAMES
        self._external_prior_names = EXTERNAL_NAMES
        self._dftbsk_prior_names = DFTBSK_NAMES
        allowed_priors = {
            "zero",
            "gaussian",
            "residual_gaussian",
            *self._te_prior_names,
            *self._projected_te_prior_names,
            *self._tied_irrep_gaussian_prior_names,
            *self._basis_prior_names,
            *self._overlap_huckel_prior_names,
            *self._haar_dm_prior_names,
            *self._external_prior_names,
            *self._dftbsk_prior_names,
        }
        if self.prior not in allowed_priors:
            raise ValueError(
                f"Unsupported flow_options.prior={self.prior!r}; "
                "use 'zero', 'gaussian', 'residual_gaussian', 'te', "
                "'projected_te', 'tied_irrep_gaussian', "
                "'basis_onsite', 'overlap_huckel', 'haar_dm', 'dftbsk', 'external', "
                "'dftb', 'xtb', or 'physical'."
            )

        if self.block_ode:
            if self.idp is None:
                raise ValueError("Block-space ODE requires an OrbitalMapper idp.")
            if self.uureal_block_ode and not is_soc_uureal_mapper(self.idp):
                raise ValueError(
                    "uureal_block_ode requires has_soc=true, nextham_uureal_mask=true, "
                    "and full_soc_prediction=false; it may not masquerade as non-SOC."
                )
            if self.residual_ao_block_ode and is_soc_uureal_mapper(self.idp):
                # Reject the reduced-SOC uu_real mapper explicitly (and before the
                # generic non-uureal SOC NotImplementedError below) so a compact-uu
                # contract can never activate the non-SOC direct-residual mode.
                raise ValueError(
                    "residual_ao_block_ode requires a plain non-SOC mapper; it may "
                    "not masquerade as uu-real."
                )
            if not self.uureal_block_ode and bool(getattr(self.idp, "has_soc", False)):
                raise NotImplementedError("Block-space ODE v1 supports non-SOC mappers only.")
            if self.dtype not in {torch.float32, torch.float64}:
                raise TypeError(
                    "Block-space ODE v1 requires float32 or float64 so its strict "
                    "inverse tolerance has a certified dtype contract."
                )
            if self.mode != "residual":
                raise ValueError(
                    "Block-space ODE v1 requires mode='residual' so B0 is physical H0."
                )
            if self.uureal_block_ode and self.prior != "zero":
                raise ValueError("uureal_block_ode requires the exact zero residual prior.")
            if self.residual_ao_block_ode:
                allowed_residual_priors = {
                    "zero",
                    *self._projected_te_prior_names,
                    *self._tied_irrep_gaussian_prior_names,
                }
                if self.prior not in allowed_residual_priors:
                    raise ValueError(
                        "residual_ao_block_ode supports only prior='zero', "
                        "prior='projected_te', or "
                        "prior='tied_irrep_gaussian'; generic TE/Gaussian priors "
                        "do not own the projected block start-state contract."
                    )
            elif (
                not self.uureal_block_ode
                and self.prior not in {"zero", *self._projected_te_prior_names}
            ):
                raise ValueError(
                    "Block-space ODE supports only prior='zero' or the explicit "
                    "prior='projected_te'; generic TE/Gaussian priors do not own "
                    "the projected block start-state contract."
                )
            if self.target_semantics not in {"absolute_full_h", "residual_dh"}:
                raise ValueError(
                    "Block-space ODE requires explicit target_semantics="
                    "'absolute_full_h' or 'residual_dh'."
                )
            if self.uureal_block_ode and self.target_semantics != "residual_dh":
                raise ValueError("uureal_block_ode requires target_semantics='residual_dh'.")
            if self.residual_ao_block_ode and self.target_semantics != "residual_dh":
                raise ValueError("residual_ao_block_ode requires target_semantics='residual_dh'.")
            if self.residual_ao_block_ode and not bool(
                options.get("block_export_final_full_h", False)
            ):
                raise ValueError(
                    "residual_ao_block_ode assembles final full H exactly once "
                    "outside the ODE; set block_export_final_full_h=true."
                )
            if self.prediction_add_h0:
                raise ValueError(
                    "Block-space ODE requires prediction.add_h0=false; residual_dh "
                    "endpoints are adapted by BlockStateCodec exactly once."
                )
            if not self.time_conditioning_required:
                raise ValueError(
                    "Block-space ODE requires time_conditioning_required=true and "
                    "a fail-closed node+edge flow-time embedding."
                )
            if self.block_inverse_mode != "strict":
                raise ValueError("Block-space ODE production inverse must use mode='strict'.")
            if not self.strict_h0:
                raise ValueError(
                    "Block-space ODE requires strict_h0=true; physical H0 is the "
                    "mandatory B0 state and may not fall back to zeros."
                )
            maximum_atol = 1.0e-10 if self.dtype == torch.float64 else 2.0e-5
            if self.block_inverse_atol > maximum_atol:
                raise ValueError(
                    "Block-space ODE strict inverse tolerance exceeds the certified "
                    f"{self.dtype} maximum {maximum_atol:.6g}: "
                    f"got {self.block_inverse_atol:.6g}."
                )
            if self.target_semantics == "absolute_full_h":
                expected_block_keys = (
                    getattr(
                        _keys,
                        "NODE_FULL_HAMIL_TARGET_BLOCKS_KEY",
                        "node_full_hamil_target_blocks",
                    ),
                    getattr(
                        _keys,
                        "EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY",
                        "edge_full_hamil_target_blocks",
                    ),
                    getattr(
                        _keys,
                        "NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY",
                        "node_full_hamil_target_block_shape",
                    ),
                    getattr(
                        _keys,
                        "EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY",
                        "edge_full_hamil_target_block_shape",
                    ),
                )
            else:
                expected_block_keys = (
                    getattr(
                        _keys,
                        "NODE_DELTA_HAMIL_BLOCKS_KEY",
                        "node_delta_hamil_blocks",
                    ),
                    getattr(
                        _keys,
                        "EDGE_DELTA_HAMIL_BLOCKS_KEY",
                        "edge_delta_hamil_blocks",
                    ),
                    getattr(
                        _keys,
                        "NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY",
                        "node_delta_hamil_block_shape",
                    ),
                    getattr(
                        _keys,
                        "EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY",
                        "edge_delta_hamil_block_shape",
                    ),
                )
            configured_block_keys = (
                self.node_block_target_key,
                self.edge_block_target_key,
                self.node_block_shape_key,
                self.edge_block_shape_key,
            )
            if configured_block_keys != expected_block_keys:
                raise ValueError(
                    "Block-space ODE target keys do not match target_semantics="
                    f"{self.target_semantics!r}: expected {expected_block_keys}, "
                    f"got {configured_block_keys}."
                )

        raw_node_sigma = options.get("node_sigma", 1.0)
        raw_edge_sigma = options.get("edge_sigma", 1.0)
        raw_te_prior_sigma = options.get("te_prior_sigma", 1.0)
        raw_tied_irrep_sigma = options.get("tied_irrep_sigma", 1.0)
        self.node_sigma = float(raw_node_sigma)
        self.edge_sigma = float(raw_edge_sigma)
        self.residual_sigma_floor = float(options.get("residual_sigma_floor", 1.0e-6))
        self.te_prior_sigma = float(raw_te_prior_sigma)
        self.tied_irrep_sigma = float(raw_tied_irrep_sigma)
        self.tied_irrep_mode = normalize_tied_irrep_mode(
            options.get("tied_irrep_mode", "")
        )
        self.tied_irrep_irreps = str(
            options.get("tied_irrep_irreps", TIED_IRREP_CANONICAL_IRREPS)
        )
        self.tied_irrep_validation_seed = options.get(
            "tied_irrep_validation_seed", None
        )
        default_te_prior_mode = "block" if self.prior == "block_te" else "irrep"
        self.te_prior_mode = str(options.get("te_prior_mode", "auto")).lower().replace("-", "_")
        if self.te_prior_mode == "auto":
            self.te_prior_mode = default_te_prior_mode
        if self.te_prior_mode == "type":
            self.te_prior_mode = "typewise"
        if self.te_prior_mode not in {"irrep", "block", "typewise"}:
            raise ValueError("flow_options.te_prior_mode must be 'irrep', 'block', or 'typewise'.")
        self.te_prior_per_graph = bool(options.get("te_prior_per_graph", True))
        self.te_prior_validation_seed = options.get("te_prior_validation_seed", None)
        self.prior_validation_seed = self.te_prior_validation_seed
        if self.prior in self._projected_te_prior_names:
            if not self.block_ode:
                raise ValueError("prior='projected_te' is supported only by block_ode.")
            if self.te_prior_mode != "irrep":
                raise ValueError(
                    "projected_te block_ode requires te_prior_mode='irrep'; "
                    "typewise mode reads target residual scales and block mode is "
                    "outside the certified irrepwise prior contract."
                )
            scales = {
                "node_sigma": raw_node_sigma,
                "edge_sigma": raw_edge_sigma,
                "te_prior_sigma": raw_te_prior_sigma,
            }
            invalid = [
                name
                for name, value in scales.items()
                if isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ]
            if invalid:
                raise ValueError(
                    "projected_te block_ode requires finite positive scales; "
                    f"invalid options={invalid}."
                )
            effective_scales = {
                "node_sigma*te_prior_sigma": self.node_sigma * self.te_prior_sigma,
                "edge_sigma*te_prior_sigma": self.edge_sigma * self.te_prior_sigma,
            }
            # Working interval mirror of the argcheck gate: reject scales that a
            # cast-based representability check would accept but that collapse to
            # exact zero (subnormal-adjacent) or overflow (near dtype-max) once
            # the unbounded Gaussian radius multiplies in.
            working_interval = {
                torch.float32: (2.0 ** -100, (2.0 - 2.0 ** -23) * 2.0 ** 115),
                torch.float64: (2.0 ** -996, (2.0 - 2.0 ** -52) * 2.0 ** 1011),
            }
            work_min, work_max = working_interval[self.dtype]
            invalid_effective = []
            for name, value in effective_scales.items():
                magnitude = abs(float(value))
                if not math.isfinite(magnitude) or not work_min <= magnitude <= work_max:
                    invalid_effective.append(name)
            if invalid_effective:
                raise ValueError(
                    "projected_te block_ode effective scales must be finite and "
                    f"inside the safe {self.dtype} working interval "
                    f"[{work_min:.3g}, {work_max:.3g}]; invalid products="
                    f"{invalid_effective}."
                )
            if (
                isinstance(self.te_prior_validation_seed, bool)
                or not isinstance(self.te_prior_validation_seed, Integral)
                or self.te_prior_validation_seed < 0
                or self.te_prior_validation_seed > _MAX_TORCH_SEED
            ):
                raise ValueError(
                    "projected_te block_ode requires an explicit integer "
                    f"te_prior_validation_seed in [0, {_MAX_TORCH_SEED}]."
                )
        if self.prior in self._tied_irrep_gaussian_prior_names:
            if not self.residual_ao_block_ode:
                raise ValueError(
                    "prior='tied_irrep_gaussian' is supported only by "
                    "residual_ao_block_ode."
                )
            validate_tied_irrep_options(
                mode=options.get("tied_irrep_mode", ""),
                irreps=self.tied_irrep_irreps,
            )
            scales = {
                "node_sigma": raw_node_sigma,
                "edge_sigma": raw_edge_sigma,
                "tied_irrep_sigma": raw_tied_irrep_sigma,
            }
            invalid = [
                name
                for name, value in scales.items()
                if isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ]
            if invalid:
                raise ValueError(
                    "tied_irrep_gaussian requires finite positive scales; "
                    f"invalid options={invalid}."
                )
            effective_scales = {
                "node_sigma*tied_irrep_sigma": self.node_sigma
                * self.tied_irrep_sigma,
                "edge_sigma*tied_irrep_sigma": self.edge_sigma
                * self.tied_irrep_sigma,
            }
            working_interval = {
                torch.float32: (2.0 ** -100, (2.0 - 2.0 ** -23) * 2.0 ** 115),
                torch.float64: (2.0 ** -996, (2.0 - 2.0 ** -52) * 2.0 ** 1011),
            }
            work_min, work_max = working_interval[self.dtype]
            invalid_effective = []
            for name, value in effective_scales.items():
                magnitude = abs(float(value))
                if not math.isfinite(magnitude) or not work_min <= magnitude <= work_max:
                    invalid_effective.append(name)
            if invalid_effective:
                raise ValueError(
                    "tied_irrep_gaussian effective scales must be finite and "
                    f"inside the safe {self.dtype} working interval "
                    f"[{work_min:.3g}, {work_max:.3g}]; invalid products="
                    f"{invalid_effective}."
                )
            seed = self.tied_irrep_validation_seed
            if (
                isinstance(seed, bool)
                or not isinstance(seed, Integral)
                or seed < 0
                or seed > _MAX_TORCH_SEED
            ):
                raise ValueError(
                    "tied_irrep_gaussian requires an explicit integer "
                    f"tied_irrep_validation_seed in [0, {_MAX_TORCH_SEED}]."
                )
            self.prior_validation_seed = seed
            log.info(
                "tied_irrep_gaussian prior enabled: mode=%s irreps=%s "
                "effective_node_sigma=%.6g effective_edge_sigma=%.6g "
                "latent_component_variances={L0:3,L1:2,L2:1}",
                self.tied_irrep_mode,
                self.tied_irrep_irreps,
                self.node_sigma * self.tied_irrep_sigma,
                self.edge_sigma * self.tied_irrep_sigma,
            )

        # Physical prior families each own their option keys (parsed in
        # from_options).  Every family is built once so the external family can
        # fall back to the on-the-fly DFTB-SK guess without re-parsing options.
        self._families = build_prior_families(options, idp)
        self.prior_family = None
        for _cls in (
            BasisOnsiteFamily,
            OverlapHuckelFamily,
            HaarDMFamily,
            ExternalFamily,
            DFTBSKFamily,
        ):
            if self.prior in _cls.NAMES:
                self.prior_family = self._families[_cls]
                break

        # Optional per-label split: prior_node/prior_edge compose two absolute
        # families (e.g. the hybrid oracle basis_onsite + external-H0 hoppings).
        self.prior_node = str(options.get("prior_node", "") or "").lower().replace("-", "_")
        self.prior_edge = str(options.get("prior_edge", "") or "").lower().replace("-", "_")
        if bool(self.prior_node) != bool(self.prior_edge):
            raise ValueError(
                "flow_options.prior_node and prior_edge must be set together "
                "(one family per label); got "
                f"prior_node={self.prior_node!r}, prior_edge={self.prior_edge!r}."
            )
        if self.prior_node and self.prior_edge:
            if self.prior_family is None or self.prior in self._haar_dm_prior_names:
                raise ValueError(
                    "flow_options.prior_node/prior_edge require flow_options.prior to "
                    "be one of the physical absolute families (e.g. 'external' or "
                    "'overlap_huckel') so the split prior rides the physical prior "
                    f"path; got prior={self.prior!r}."
                )
            self.prior_family = SplitPriorFamily(
                node_family=SplitPriorFamily.resolve_side(self.prior_node, self._families),
                edge_family=SplitPriorFamily.resolve_side(self.prior_edge, self._families),
                node_name=self.prior_node,
                edge_name=self.prior_edge,
            )
        self._prior_ctx = PriorContext(self)
        self.prior_calibration_path = str(options.get("prior_calibration", "") or "")
        self._prior_calibration_artifact: Optional[Dict[str, Any]] = None
        self._prior_calibration_cache: Dict[
            Tuple[str, str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}
        self._huckel_pair_energy_table_cache: Dict[
            Tuple[str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}

        # physical_prior_fallback is cross-cutting (it governs which family the
        # 'physical' prior degrades to), so it stays on the trainer.
        self.physical_prior_fallback = str(
            options.get("physical_prior_fallback", "basis_onsite")
        ).lower().replace("-", "_")
        if self.physical_prior_fallback not in {"basis_onsite", "zero", "error"}:
            raise ValueError(
                "flow_options.physical_prior_fallback must be 'basis_onsite', 'zero', or 'error'."
            )
        self.last_haar_candidate_index = -1
        self._basis_onsite_table_cache: Dict[
            Tuple[str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}
        self._basis_onsite_type_mean_cache: Dict[
            Tuple[str, torch.dtype, int], Optional[torch.Tensor]
        ] = {}
        self.physical_prior_jitter_sigma = float(
            options.get("physical_prior_jitter_sigma", options.get("prior_jitter_sigma", 0.0))
        )
        self.physical_prior_jitter_reference_scale = bool(
            options.get("physical_prior_jitter_reference_scale", True)
        )
        self.physical_prior_jitter_edge_decay = float(
            options.get("physical_prior_jitter_edge_decay", 0.0)
        )

        # Time sampling.  QHFlow uses U(0,1); we expose a t0 mass so the network
        # explicitly sees the physical-H0 one-step inference point.
        self.time_sampling = str(options.get("time_sampling", "uniform")).lower()
        self.t_min = float(options.get("t_min", 0.0))
        self.t_max = float(options.get("t_max", 0.999))
        if self.uureal_block_ode:
            # The uu_real rollout ALWAYS starts at exactly t=0, D=0 -- the one
            # state inference is guaranteed to visit.  With D_t = t*D1 the model
            # can fit interior times via the D_t/t shortcut without ever
            # producing a trained answer at the boundary, so a t=0 training mass
            # is mandatory here.  Default 0.15: enough boundary gradient mass to
            # anchor the inference start without starving the interior schedule
            # (inside the recommended 0.1-0.25 band).  Explicit zero/negative is
            # a misconfiguration and fails closed.
            self.t0_probability = float(options.get("t0_probability", 0.15))
            if self.t0_probability <= 0.0:
                raise ValueError(
                    "uureal_block_ode requires 0 < t0_probability < 1: the rollout "
                    "starts at t=0, D=0, and that boundary state has no training "
                    "mass otherwise (recommended range 0.1-0.25)."
                )
        elif self.residual_ao_block_ode:
            # Same D_t = t*D1 boundary reasoning as uureal_block_ode: the direct
            # residual rollout always starts at exactly t=0, D=0, so a t=0 training
            # mass is mandatory here too. Default 0.15; explicit zero/negative is a
            # misconfiguration and fails closed.
            self.t0_probability = float(options.get("t0_probability", 0.15))
            if self.t0_probability <= 0.0:
                raise ValueError(
                    "residual_ao_block_ode requires 0 < t0_probability < 1: the rollout "
                    "starts at t=0, D=0, and that boundary state has no training "
                    "mass otherwise (recommended range 0.1-0.25)."
                )
        else:
            self.t0_probability = float(options.get("t0_probability", 0.0))
        # Universal upper bound (all modes).  t0_probability is the Bernoulli mass
        # of the exact-t=0 boundary injection in ``_sample_t`` (``torch.rand(...) <
        # p``); p >= 1 makes that comparison always True, sending EVERY sample to
        # t=0 and silently starving the interior [t_min, t_max] schedule, and a
        # non-finite p is never a valid probability.  The block-ODE modes require
        # p > 0 above, so their effective window is 0 < p < 1; the generic CFM
        # window is 0 <= p < 1.
        if not math.isfinite(self.t0_probability) or not (
            0.0 <= self.t0_probability < 1.0
        ):
            raise ValueError(
                "flow_options.t0_probability must be finite and satisfy "
                "0.0 <= t0_probability < 1.0 (recommended 0.1-0.25): a value >= 1 "
                "collapses every training time to the t=0 boundary and starves the "
                f"interior schedule; got {self.t0_probability!r}."
            )
        self.t_eps = float(options.get("t_eps", 1.0e-3))
        self.endpoint_weight_power = float(options.get("endpoint_weight_power", 0.0))
        self.endpoint_weight_cap = float(options.get("endpoint_weight_cap", 100.0))
        raw_validation_steps = options.get("validation_ode_steps", [1, 3])
        if self.block_ode and (
            not raw_validation_steps
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_validation_steps
            )
        ):
            raise ValueError(
                "Block-space ODE validation_ode_steps must contain integer steps drawn from [1, 3]."
            )
        validation_ode_steps = {
            int(v) for v in raw_validation_steps if int(v) > 0
        }
        # Euler-1 is the route-independent endpoint baseline. Additional steps
        # are diagnostics; configuration may not disable the common curve.
        validation_ode_steps.add(1)
        self.validation_ode_steps = tuple(sorted(validation_ode_steps))
        if self.output_space == "ao_block" and self.validation_ode_steps != (1,):
            raise ValueError(
                "flow_options.output_space='ao_block' is a cross-space one-step "
                "endpoint adapter and requires validation_ode_steps=[1]."
            )
        if self.block_ode and (
            not self.validation_ode_steps
            or not set(self.validation_ode_steps).issubset({1, 3})
        ):
            raise ValueError(
                "Block-space ODE v1 supports validation_ode_steps drawn from [1, 3]."
            )
        self.apply_to_reference = bool(options.get("apply_to_reference", False))
        self.validation_flow_metrics = frozenset(
            str(value).lower().replace("-", "_")
            for value in options.get(
                "validation_flow_metrics",
                ("random_t", "one_step", "trajectory"),
            )
        )
        self.log_validation_random_t_loss = "random_t" in self.validation_flow_metrics
        self.log_validation_t0_loss = "one_step" in self.validation_flow_metrics
        self.log_validation_flow_euler_loss = "trajectory" in self.validation_flow_metrics
        # Endpoint-compatible metrics and their legacy aliases are a stable
        # trainer contract, not optional logging features.
        self.log_train_compatible_loss = True
        self.log_validation_compatible_loss = True
        self.compatible_loss_to_legacy_keys = True
        # Loss and regularization.
        self.loss_type = str(options.get("loss_type", "mse")).lower()
        if self.loss_type not in {"mse", "l1_rmse"}:
            raise ValueError("flow_options.loss_type must be 'mse' or 'l1_rmse'.")
        self.node_weight = float(options.get("node_weight", 1.0))
        self.edge_weight = float(options.get("edge_weight", 1.0))
        self.router_z_loss_coef = float(options.get("z_loss_coef", 0.0))

        # Safety switches.
        self.overwrite_feature_keys = bool(options.get("overwrite_feature_keys", True))
        self.detach_interpolated_h0 = bool(options.get("detach_interpolated_h0", True))
        self.component_reduction = str(options.get("component_reduction", "global_elements")).lower()
        if self.component_reduction not in {"global_elements", "equal_components"}:
            raise ValueError(
                "flow_options.component_reduction must be 'global_elements' or 'equal_components'."
            )
        for name, value in (
            ("node_weight", self.node_weight),
            ("edge_weight", self.edge_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"flow_options.{name} must be finite and non-negative, got {value!r}."
                )
        if self.node_weight == 0.0 and self.edge_weight == 0.0:
            raise ValueError(
                "flow_options.node_weight and edge_weight may not both be zero."
            )
        if (
            self.component_reduction == "global_elements"
            and not self.allow_nonunit_global_element_weights
            and (self.node_weight != 1.0 or self.edge_weight != 1.0)
        ):
            raise ValueError(
                "flow_options.component_reduction='global_elements' requires "
                "node_weight=edge_weight=1. Use component_reduction="
                "'equal_components' for node/edge loss multipliers."
            )

        self.last_state: Dict[str, torch.Tensor] = {}
        self._te_irrep_slices_cache: Dict[int, Optional[Tuple[Tuple[int, int, int], ...]]] = {}
        self.block_codec = (
            BlockStateCodec(
                self.idp,
                dtype=self.dtype,
                device=self.device,
                inverse_mode=self.block_inverse_mode,
                atol=self.block_inverse_atol,
                target_semantics=self.target_semantics,
            )
            if self.block_ode and not self.uureal_block_ode
            else None
        )
        if self.enabled:
            log.info(
                "Hamiltonian CFM enabled: mode=%s prior=%s output_space=%s "
                "t=[%.3g, %.3g] t0_prob=%.3g loss=%s",
                self.mode,
                self.prior,
                self.output_space,
                self.t_min,
                self.t_max,
                self.t0_probability,
                self.loss_type,
            )

    # ------------------------------------------------------------------
    # Family-owned option accessors
    # ------------------------------------------------------------------
    # These options now live on the prior-family instances (single source of
    # truth).  The shared basis-onsite/Huckel/Haar helpers below and existing
    # callers still read them off the trainer, so expose read-only views.
    @property
    def basis_onsite_scale(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_scale

    @property
    def basis_onsite_missing_value(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_missing_value

    @property
    def basis_onsite_edge_value(self) -> float:
        return self._families[BasisOnsiteFamily].basis_onsite_edge_value

    @property
    def huckel_edge_energy_fallback(self) -> float:
        return self._families[OverlapHuckelFamily].huckel_edge_energy_fallback

    @property
    def huckel_strict_basis(self) -> bool:
        return self._families[OverlapHuckelFamily].huckel_strict_basis

    @property
    def haar_node_key(self) -> str:
        return self._families[HaarDMFamily].haar_node_key

    @property
    def haar_edge_key(self) -> str:
        return self._families[HaarDMFamily].haar_edge_key

    @property
    def haar_candidate_index(self) -> int:
        return self._families[HaarDMFamily].haar_candidate_index

    @property
    def haar_dm_strict(self) -> bool:
        return self._families[HaarDMFamily].haar_dm_strict

    # ------------------------------------------------------------------
    # Sampling / interpolation
    # ------------------------------------------------------------------
    def validation_seed(self, batch_index: int, purpose: str) -> int:
        """Derive a stable fixed-width validation substream seed."""
        if purpose not in _VALIDATION_SEED_STREAMS:
            raise ValueError(f"Unknown validation RNG purpose={purpose!r}.")
        if isinstance(batch_index, bool) or not isinstance(batch_index, Integral) or batch_index < 0:
            raise ValueError("validation batch_index must be a non-negative integer.")
        base = int(self.prior_validation_seed or 0) & _MAX_TORCH_SEED
        value = (
            base
            ^ _VALIDATION_SEED_STREAMS[purpose]
            ^ ((int(batch_index) + 1) * 0x94D049BB133111EB)
        ) & _MAX_TORCH_SEED
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & _MAX_TORCH_SEED
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & _MAX_TORCH_SEED
        return (value ^ (value >> 31)) & _MAX_TORCH_SEED

    def validation_prior_base_seed(self) -> int:
        """Batch-index-INDEPENDENT base seed for SEEDED validation prior draws.

        The per-(sample_uid, node|edge) substream (see
        :meth:`_prior_uid_substream_seed`) already gives every graph an epsilon
        that is invariant to batch composition, order, and sharding.  The prior
        base seed the validation caller threads into that substream must therefore
        NOT depend on the batch position -- otherwise a record's epsilon changes
        when it moves between validation batches (a smaller batch size, a
        resharded/reordered loader, an inserted record), re-introducing exactly
        the batch-composition dependence the substream was built to remove.

        :meth:`validation_seed` deliberately mixes ``batch_index`` into its output
        (so successive draws decorrelate); this canonicalises the prior stream at
        the fixed point ``batch_index == 0``, so the base seed is a pure function
        of the configured prior validation seed and the ``"prior"`` stream tag.
        The time stream (``validation_seed(..., "time")``) is a separate axis and
        is intentionally left batch-indexed.
        """
        return self.validation_seed(0, "prior")

    @staticmethod
    def _seeded_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= _MAX_TORCH_SEED:
            raise ValueError(f"RNG seed must be an integer in [0, {_MAX_TORCH_SEED}].")
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator

    def _prior_uid_substream_seed(
        self, base_seed: int, *, uid: int, component: str
    ) -> int:
        """Stable per-(graph uid, node/edge) substream seed for SEEDED draws.

        Extends the splitmix-style :meth:`validation_seed` machinery with the
        conceptual purpose string ``"prior:<component>:<uid>"``: the epsilon a
        graph receives is a deterministic function of (the base prior seed, the
        graph's stable ``sample_uid``, and whether this is the node or edge
        component), and is therefore independent of the graph's position in the
        batch, the batch size, and sharding.  ``base_seed`` is the initial seed
        of the caller's :class:`torch.Generator` (``prior_seed``), so a different
        ``prior_seed`` still redraws the whole batch.
        """
        if component not in _PRIOR_COMPONENT_STREAMS:
            raise ValueError(
                f"Unknown prior substream component={component!r}; expected 'node' or 'edge'."
            )
        value = (
            (int(base_seed) & _MAX_TORCH_SEED)
            ^ _VALIDATION_SEED_STREAMS["prior"]
            ^ _PRIOR_COMPONENT_STREAMS[component]
            ^ (((int(uid) & _MAX_TORCH_SEED) + 1) * _PRIOR_UID_MULTIPLIER)
        ) & _MAX_TORCH_SEED
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & _MAX_TORCH_SEED
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & _MAX_TORCH_SEED
        return (value ^ (value >> 31)) & _MAX_TORCH_SEED

    def _prior_substream_generators(
        self,
        data: Optional[AtomicDataDict.Type],
        *,
        base_seed: int,
        num_graphs: int,
        component: Optional[str],
        device: torch.device,
    ) -> list:
        """One independent :class:`torch.Generator` per graph, keyed by sample_uid.

        Fail-closed: a SEEDED prior draw must be reproducible per graph regardless
        of batch composition, which is impossible without a stable per-graph
        identity, so the batch MUST carry ``SAMPLE_UID_KEY`` (a graph-level long
        field, one value per graph after collation).
        """
        uids = None if data is None else data.get(_keys.SAMPLE_UID_KEY, None)
        if uids is None:
            raise ValueError(
                "A SEEDED stochastic block prior draw requires the per-graph record "
                f"identity `{_keys.SAMPLE_UID_KEY}` so a graph's epsilon is "
                "invariant to batch composition, order, and sharding; it is absent "
                "from this batch. Emit it from LMDBDataset (see "
                "AtomicDataDict.SAMPLE_UID_KEY) or, for a hand-built batch, attach a "
                "synthetic int64 per-graph uid tensor."
            )
        uids = torch.as_tensor(uids).detach().reshape(-1).to(dtype=torch.long)
        if uids.numel() != int(num_graphs):
            raise ValueError(
                f"`{_keys.SAMPLE_UID_KEY}` carries {int(uids.numel())} value(s) but "
                f"the batch has {int(num_graphs)} graph(s); it must hold exactly one "
                "stable id per graph."
            )
        uid_list = uids.tolist()
        return [
            self._seeded_generator(
                device,
                self._prior_uid_substream_seed(
                    base_seed, uid=int(u), component=component
                ),
            )
            for u in uid_list
        ]

    @staticmethod
    def _graph_row_slices(graph_index: torch.Tensor, num_graphs: int) -> list:
        """Ascending row positions belonging to each graph (batch-order index)."""
        return [
            (graph_index == g).nonzero(as_tuple=False).reshape(-1)
            for g in range(int(num_graphs))
        ]

    @staticmethod
    def _seeded_rows_by_graph(
        row_count: int,
        width: int,
        row_slices: list,
        generators: list,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fill ``[row_count, width]`` drawing each graph's rows from its own gen.

        Because a graph's rows are filled from its own uid-seeded generator in
        ascending position order, the same graph receives the same rows for-row
        regardless of how many other graphs share the batch or in which order.
        """
        out = torch.empty(int(row_count), int(width), device=device, dtype=dtype)
        if row_count == 0:
            return out
        for gen_g, rows_g in zip(generators, row_slices):
            n = int(rows_g.numel())
            if n == 0:
                continue
            out.index_copy_(
                0,
                rows_g.to(device=device),
                torch.randn(n, int(width), device=device, dtype=dtype, generator=gen_g),
            )
        return out

    def _seeded_radius_by_graph(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: torch.Tensor,
        row_slices: list,
        generators: list,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-uid analogue of :meth:`_te_radius` (one radius stream per graph)."""
        if self.te_prior_per_graph:
            radius_per_graph = torch.empty(
                len(generators), 1, device=device, dtype=dtype
            )
            for g, gen_g in enumerate(generators):
                radius_per_graph[g] = torch.randn(
                    1, 1, device=device, dtype=dtype, generator=gen_g
                )
            radius = radius_per_graph.index_select(0, graph_index.to(device=device))
        else:
            radius = torch.empty(int(row_count), 1, device=device, dtype=dtype)
            for gen_g, rows_g in zip(generators, row_slices):
                n = int(rows_g.numel())
                if n == 0:
                    continue
                radius.index_copy_(
                    0,
                    rows_g.to(device=device),
                    torch.randn(n, 1, device=device, dtype=dtype, generator=gen_g),
                )
        return radius * active_dim.sqrt()

    def _assert_stochastic_prior_draw_finite_and_scaled(
        self, prior_name: str, components
    ) -> None:
        """Runtime belt for any stochastic block prior draw before it is used.

        ``components`` is an iterable of ``(label, tensor, effective_scale)``.  A
        drawn component must be all-finite, and -- when its configured effective
        scale is nonzero and it has any elements -- must not be entirely zero (a
        silent all-masked / subnormal-underflow collapse of the stochastic
        bridge).  The error names the effective scale, dtype, and component label.
        """
        for label, tensor, effective_scale in components:
            if tensor is None:
                continue
            if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(
                    f"{prior_name} {label} draw contains NaN/Inf (effective scale="
                    f"{float(effective_scale):.6g}, dtype={tensor.dtype}, "
                    f"component={label})."
                )
            if (
                float(effective_scale) != 0.0
                and tensor.numel() > 0
                and not bool(tensor.detach().any().item())
            ):
                raise ValueError(
                    f"{prior_name} {label} draw is entirely zero despite a nonzero "
                    f"configured effective scale={float(effective_scale):.6g} "
                    f"(dtype={tensor.dtype}, component={label})."
                )

    def _assert_projected_te_draw_finite_and_scaled(self, components) -> None:
        """TA-3 runtime belt for any projected_te draw, before it is used."""
        self._assert_stochastic_prior_draw_finite_and_scaled(
            "projected_te", components
        )

    def _sample_t(
        self,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        lo = max(0.0, min(self.t_min, 1.0))
        hi = max(lo, min(self.t_max, 1.0 - self.t_eps))
        if self.time_sampling == "uniform":
            t = lo + (hi - lo) * torch.rand(
                num_graphs, device=device, dtype=dtype, generator=generator
            )
        elif self.time_sampling == "logit_normal":
            mean = float(self.options.get("time_logit_mean", -0.4))
            std = float(self.options.get("time_logit_std", 1.0))
            raw = torch.randn(
                num_graphs, device=device, dtype=dtype, generator=generator
            ) * std + mean
            t = torch.sigmoid(raw)
            t = lo + (hi - lo) * t
        else:
            raise ValueError(f"Unsupported flow_options.time_sampling={self.time_sampling!r}")
        t = t.clamp(min=lo, max=hi)
        # Inject exact-zero times AFTER the [t_min, t_max] clamp: t0_probability
        # exists precisely to train the t=0 boundary state, so a configured
        # t_min>0 must not re-clamp the injected zeros away (with t_min=0.5 and
        # t0_probability=1 every "zero" silently became 0.5 before this fix).
        if self.t0_probability > 0.0:
            use_t0 = torch.rand(
                num_graphs, device=device, generator=generator
            ) < self.t0_probability
            t = torch.where(use_t0, torch.zeros_like(t), t)
        return t

    @staticmethod
    def _num_graphs(data: AtomicDataDict.Type) -> int:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    @staticmethod
    def _normalize_t(
        t: torch.Tensor,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
        if t.numel() == 1:
            return t.expand(num_graphs)
        if t.numel() != num_graphs:
            raise ValueError(f"Expected one flow time per graph ({num_graphs}), got {t.numel()}.")
        return t

    @staticmethod
    def _expand_graph_times(
        data: AtomicDataDict.Type,
        t: torch.Tensor,
        *,
        node_count: Optional[int],
        edge_count: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None:
            if node_count is None:
                batch = torch.zeros(0, device=t.device, dtype=torch.long)
            else:
                batch = torch.zeros(node_count, device=t.device, dtype=torch.long)
        else:
            batch = batch.to(device=t.device, dtype=torch.long).reshape(-1)
        node_t = None if node_count is None else t.index_select(0, batch[:node_count])

        edge_t = None
        if edge_count is not None:
            edge_index = data.get(_keys.EDGE_INDEX_KEY, None)
            if edge_index is None:
                if t.numel() != 1:
                    raise KeyError("Per-graph edge flow time requires `edge_index`.")
                edge_t = t.expand(edge_count)
            else:
                edge_center = edge_index[0].to(device=t.device, dtype=torch.long).reshape(-1)
                edge_graph = batch.index_select(0, edge_center[:edge_count])
                edge_t = t.index_select(0, edge_graph)
        return node_t, edge_t

    def _base_like(self, data: AtomicDataDict.Type, target: torch.Tensor, h0_key: str, label: str) -> torch.Tensor:
        if self.mode == "full":
            return torch.zeros_like(target)

        base = data.get(h0_key, None)
        if base is None:
            if self.strict_h0:
                raise KeyError(
                    f"CFM residual mode requires `{h0_key}` for the {label} base; "
                    "set missing_h0_policy='zero' only for an explicit zero-base experiment."
                )
            if self.warn_missing_h0:
                log.warning(
                    "CFM residual mode did not find `%s`; falling back to zeros for %s base. "
                    "For NextHAM-style training, make sure the dataset emits node_h0/edge_h0.",
                    h0_key,
                    label,
                )
            base = torch.zeros_like(target)
        else:
            base = base.to(device=target.device, dtype=target.dtype)
            if base.shape != target.shape:
                if self.strict_h0:
                    raise ValueError(
                        f"CFM {label} base `{h0_key}` shape {tuple(base.shape)} "
                        f"!= target shape {tuple(target.shape)}."
                    )
                if self.warn_missing_h0:
                    log.warning(
                        "CFM %s base `%s` shape %s != target shape %s; using zeros.",
                        label,
                        h0_key,
                        tuple(base.shape),
                        tuple(target.shape),
                    )
                base = torch.zeros_like(target)
        return base

    @staticmethod
    def _align_bool_mask(mask: torch.Tensor, like: torch.Tensor, *, pad_value: bool = False) -> torch.Tensor:
        mask = mask.to(device=like.device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.reshape(1, 1)
        elif mask.ndim == 1:
            mask = mask.reshape(-1, 1)
        elif mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)

        fill = bool(pad_value)
        if mask.shape[0] < like.shape[0]:
            pad = torch.full(
                (like.shape[0] - mask.shape[0], mask.shape[1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > like.shape[0]:
            mask = mask[: like.shape[0]]

        if mask.shape[-1] == 1:
            while mask.ndim < like.ndim:
                mask = mask.unsqueeze(-1)
            return mask.expand_as(like)
        if mask.shape[-1] < like.shape[-1]:
            pad = torch.full(
                (mask.shape[0], like.shape[-1] - mask.shape[-1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > like.shape[-1]:
            mask = mask[:, : like.shape[-1]]
        while mask.ndim < like.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(like)

    def _project_raw_feature_table(self, table: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if table.ndim < 2 or table.shape[-1] == int(feature_dim) or self.idp is None:
            return table
        like = table.new_empty((table.shape[0], int(feature_dim)))
        table, _raw_mask = project_uureal_to_like(self.idp, table, like)
        return table

    def _coerce_prior_source(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
    ) -> torch.Tensor:
        if not torch.is_tensor(source):
            source = torch.as_tensor(source, device=like.device)
        if torch.is_complex(source):
            if not self.allow_complex_prior_real_projection:
                raise TypeError(
                    f"External {label or 'state'} prior `{key}` is complex. "
                    "For SOC, convert AO blocks through block_to_feature into the "
                    "real Re/Im RME layout. Setting "
                    "flow_options.allow_complex_prior_real_projection=true restores "
                    "the legacy, lossy .real projection only for an explicit ablation."
                )
            log.warning(
                "External %s prior `%s` is complex; applying the explicitly enabled "
                "legacy real-part projection and discarding imaginary channels.",
                label,
                key,
            )
            source = source.real
        source = source.to(device=like.device, dtype=like.dtype)
        if source.ndim == 1 and like.ndim >= 2:
            source = source.unsqueeze(0)
        source, _raw_mask = project_uureal_to_like(self.idp, source, like)
        if source.shape == like.shape:
            return source
        if source.numel() == like.numel():
            return source.reshape_as(like)
        raise ValueError(
            f"External {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
            f"does not match target shape {tuple(like.shape)}."
        )

    def _basis_onsite_energy(self, symbol: str, orbital: str) -> float:
        return prior_physical.basis_onsite_energy(
            symbol, orbital, missing=self.basis_onsite_missing_value
        )

    @staticmethod
    def _orbital_l(orbital: str) -> int:
        return prior_physical.orbital_l(orbital)

    def _basis_onsite_table(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        cache_key = (str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._basis_onsite_table_cache:
            return self._basis_onsite_table_cache[cache_key]

        idp = self.idp
        if idp is None:
            self._basis_onsite_table_cache[cache_key] = None
            return None
        table = prior_physical.basis_onsite_table(
            idp,
            device=like.device,
            dtype=like.dtype,
            scale=self.basis_onsite_scale,
            missing=self.basis_onsite_missing_value,
        )
        if table is None:
            self._basis_onsite_table_cache[cache_key] = None
            return None

        table, _raw_mask = project_uureal_to_like(self.idp, table, like)
        if table.ndim < 2 or table.shape[-1] != like.shape[-1]:
            self._basis_onsite_table_cache[cache_key] = None
            return None
        self._basis_onsite_table_cache[cache_key] = table
        return table

    def _basis_onsite_type_mean(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        cache_key = (str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._basis_onsite_type_mean_cache:
            return self._basis_onsite_type_mean_cache[cache_key]

        table = self._basis_onsite_table(like)
        if table is None:
            self._basis_onsite_type_mean_cache[cache_key] = None
            return None
        mean = prior_physical.basis_onsite_type_mean(
            table, fallback=self.huckel_edge_energy_fallback
        )
        self._basis_onsite_type_mean_cache[cache_key] = mean
        return mean

    def _huckel_pair_energy_table(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        """[num_bond_types, like_width] per-orbital-pair WH energies (cached)."""
        cache_key = (str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._huckel_pair_energy_table_cache:
            return self._huckel_pair_energy_table_cache[cache_key]
        table = None
        if self.idp is not None:
            table = prior_physical.huckel_pair_energy_table(
                self.idp,
                device=like.device,
                dtype=like.dtype,
                missing=self.basis_onsite_missing_value,
            )
            if table is not None:
                table, _raw_mask = project_uureal_to_like(self.idp, table, like)
                if table.ndim < 2 or table.shape[-1] != like.shape[-1]:
                    table = None
        self._huckel_pair_energy_table_cache[cache_key] = table
        return table

    def _calibration_table(self, like: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """A named table from the prior_calibration artifact, on like's device/dtype.

        Loads the artifact lazily (fail-closed: version/basis mismatch raises in
        :mod:`dptb.nnops.prior_calibration`), then projects the stored layout to
        ``like``'s width -- equal widths pass through; a wider raw/SOC layout is
        compressed via ``project_uureal_to_like``; anything else raises.
        """
        cache_key = (name, str(like.device), like.dtype, int(like.shape[-1]))
        if cache_key in self._prior_calibration_cache:
            return self._prior_calibration_cache[cache_key]
        table: Optional[torch.Tensor] = None
        if self.prior_calibration_path:
            if self._prior_calibration_artifact is None:
                self._prior_calibration_artifact = prior_calibration.load_calibration(
                    self.prior_calibration_path, idp=self.idp, strict=True
                )
            raw = self._prior_calibration_artifact.get(name)
            if raw is not None:
                table = raw.to(device=like.device, dtype=like.dtype)
                if table.ndim != 2:
                    raise ValueError(
                        f"prior_calibration `{name}` must be 2-D [num_types, rme_dim]; "
                        f"got shape {tuple(table.shape)}."
                    )
                if table.shape[-1] != like.shape[-1]:
                    projected, _raw_mask = project_uureal_to_like(self.idp, table, like)
                    if projected.ndim < 2 or projected.shape[-1] != like.shape[-1]:
                        raise ValueError(
                            f"prior_calibration `{name}` width {int(table.shape[-1])} does "
                            f"not match the feature width {int(like.shape[-1])} and cannot "
                            "be projected; regenerate the calibration in this layout."
                        )
                    table = projected
        self._prior_calibration_cache[cache_key] = table
        return table

    def _huckel_edge_energy_like(
        self,
        like: torch.Tensor,
        *,
        data: Optional[AtomicDataDict.Type],
    ) -> torch.Tensor:
        fallback = torch.full(
            (like.shape[0],),
            float(self.huckel_edge_energy_fallback),
            device=like.device,
            dtype=like.dtype,
        )
        type_mean = self._basis_onsite_type_mean(like)
        if type_mean is None:
            if self.huckel_strict_basis:
                raise ValueError(
                    "flow_options.prior='overlap_huckel' requires an OrbitalMapper-like idp "
                    "with basis_to_full_basis/orbpair_maps so edge energies can use basis onsite levels."
                )
            return fallback
        type_mean = type_mean.to(device=like.device, dtype=like.dtype)

        if data is None or _keys.EDGE_INDEX_KEY not in data or AtomicDataDict.ATOM_TYPE_KEY not in data:
            if self.huckel_strict_basis:
                raise KeyError(
                    "flow_options.prior='overlap_huckel' requires edge_index and atom_types "
                    "to map overlap rows to basis-derived endpoint energies."
                )
            valid = type_mean.abs() > 0
            if bool(valid.any().item()):
                fallback.fill_(float(type_mean[valid].mean().detach().item()))
            return fallback

        edge_index = data[_keys.EDGE_INDEX_KEY].to(device=like.device, dtype=torch.long)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].to(device=like.device, dtype=torch.long).reshape(-1)
        if edge_index.ndim != 2 or edge_index.shape[0] < 2 or atom_types.numel() == 0:
            if self.huckel_strict_basis:
                raise ValueError(
                    "flow_options.prior='overlap_huckel' expects edge_index with shape [2, n_edge] "
                    "and non-empty atom_types."
                )
            return fallback

        count = int(like.shape[0])
        energy = prior_physical.huckel_edge_energy(
            type_mean,
            edge_index,
            atom_types,
            count,
            fallback=float(self.huckel_edge_energy_fallback),
            strict=self.huckel_strict_basis,
        )
        return energy.reshape((count,) + (1,) * (like.ndim - 1))

    def _absolute_to_flow_prior(
        self,
        absolute_prior: torch.Tensor,
        like: torch.Tensor,
        *,
        base: Optional[torch.Tensor],
    ) -> torch.Tensor:
        absolute_prior = absolute_prior.to(device=like.device, dtype=like.dtype)
        if self.mode != "residual":
            return absolute_prior
        if base is None:
            base = torch.zeros_like(like)
        return absolute_prior - base.to(device=like.device, dtype=like.dtype)

    def _haar_candidate_axis(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
    ) -> Optional[Tuple[int, int]]:
        """Resolve ``(candidate_axis, candidate_count)`` for a Haar-DM source.

        Returns ``None`` when ``source`` already matches ``like`` (no candidate
        axis). Raises ValueError for any rank/shape that is neither the target
        shape nor a single-candidate-axis expansion of it. Only shapes are read,
        so ``source`` need not be on ``like``'s device/dtype.
        """
        if source.shape == like.shape:
            return None
        if source.ndim != like.ndim + 1:
            raise ValueError(
                f"Haar-DM {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
                f"must match target shape {tuple(like.shape)} or include one candidate axis."
            )
        if source.shape[0] == like.shape[0] and source.shape[-1] == like.shape[-1]:
            candidate_axis, candidate_count = 1, int(source.shape[1])
        elif source.shape[1] == like.shape[0] and source.shape[-1] == like.shape[-1]:
            candidate_axis, candidate_count = 0, int(source.shape[0])
        else:
            raise ValueError(
                f"Haar-DM {label or 'state'} prior `{key}` shape {tuple(source.shape)} "
                f"is incompatible with target shape {tuple(like.shape)}."
            )
        if candidate_count <= 0:
            raise ValueError(f"Haar-DM prior `{key}` has zero candidates.")
        return candidate_axis, candidate_count

    def _haar_candidate_count(
        self,
        source: Optional[torch.Tensor],
        like: Optional[torch.Tensor],
        *,
        key: str,
        label: Optional[str],
    ) -> Optional[int]:
        """Candidate-axis size for ``source`` against ``like`` (``None`` if absent)."""
        if source is None or like is None:
            return None
        if not torch.is_tensor(source):
            source = torch.as_tensor(source)
        axis_info = self._haar_candidate_axis(source, like, key=key, label=label)
        return None if axis_info is None else axis_info[1]

    def _resolve_haar_candidate_idx(
        self,
        data: Optional[AtomicDataDict.Type],
        *,
        node_like: Optional[torch.Tensor],
        edge_like: Optional[torch.Tensor],
    ) -> Optional[int]:
        """Draw one Haar-DM candidate index shared by the node and edge priors.

        The precomputed node/edge candidates are per-candidate coherent -- index
        ``i`` of ``haar_node_key`` and index ``i`` of ``haar_edge_key`` encode the
        same Haar density matrix ``D_haar`` (so ``DSD=spin*D`` and ``Tr(DS)=nelec``
        hold per candidate). Node and edge must therefore select the *same*
        candidate; drawing independently in the node and edge calls would break
        the density-matrix semantics whenever the candidate axis K>1. The index is
        threaded explicitly to both ``_prior_like`` calls rather than stashed on an
        instance attribute so concurrent node/edge preparation cannot desync.

        Returns ``None`` when the active prior is not Haar-DM or when neither
        source carries a candidate axis (2D sources are selected as-is
        downstream). ``haar_candidate_index >= 0`` is honored as a fixed override.
        """
        if self.prior not in self._haar_dm_prior_names or data is None:
            return None
        node_count = self._haar_candidate_count(
            data.get(self.haar_node_key, None),
            node_like,
            key=self.haar_node_key,
            label="node",
        )
        edge_count = self._haar_candidate_count(
            data.get(self.haar_edge_key, None),
            edge_like,
            key=self.haar_edge_key,
            label="edge",
        )
        if (
            node_count is not None
            and edge_count is not None
            and node_count != edge_count
        ):
            raise ValueError(
                f"Haar-DM node prior `{self.haar_node_key}` has {node_count} candidates "
                f"but edge prior `{self.haar_edge_key}` has {edge_count}; the node and "
                "edge candidate axes must match so the same Haar density matrix is "
                "selected for both."
            )
        count = node_count if node_count is not None else edge_count
        if count is None:
            return None
        if self.haar_candidate_index >= 0:
            if self.haar_candidate_index >= count:
                raise ValueError(
                    f"flow_options.haar_candidate_index={self.haar_candidate_index} "
                    f"is outside the Haar-DM candidate count {count}."
                )
            return int(self.haar_candidate_index)
        device = (
            node_like.device
            if node_like is not None
            else edge_like.device
            if edge_like is not None
            else torch.device("cpu")
        )
        return int(torch.randint(count, (), device=device).item())

    def _select_haar_candidate(
        self,
        source: torch.Tensor,
        like: torch.Tensor,
        *,
        key: str,
        label: Optional[str],
        candidate_idx: Optional[int] = None,
    ) -> torch.Tensor:
        source = torch.as_tensor(source, device=like.device, dtype=like.dtype)
        self.last_haar_candidate_index = -1
        axis_info = self._haar_candidate_axis(source, like, key=key, label=label)
        if axis_info is None:
            return source
        candidate_axis, candidate_count = axis_info
        if candidate_idx is not None:
            # Batch-resolved shared index (see _resolve_haar_candidate_idx).
            idx = int(candidate_idx)
            if idx < 0 or idx >= candidate_count:
                raise ValueError(
                    f"Haar-DM {label or 'state'} prior `{key}` candidate index {idx} "
                    f"is outside candidate count {candidate_count}."
                )
        elif self.haar_candidate_index >= 0:
            if self.haar_candidate_index >= candidate_count:
                raise ValueError(
                    f"flow_options.haar_candidate_index={self.haar_candidate_index} "
                    f"is outside `{key}` candidate count {candidate_count}."
                )
            idx = self.haar_candidate_index
        else:
            idx = int(torch.randint(candidate_count, (), device=like.device).item())
        self.last_haar_candidate_index = idx
        return source[:, idx, ...] if candidate_axis == 1 else source[idx, ...]

    def _physical_prior_jitter_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
    ) -> torch.Tensor:
        if self.physical_prior_jitter_sigma <= 0.0:
            return torch.zeros_like(like)
        mask = self._prior_mask(data, like, label).to(dtype=like.dtype)
        noise = torch.randn_like(like) * mask
        if self.physical_prior_jitter_reference_scale and like.ndim >= 2:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            row_rms = ((like.detach().square() * mask).sum(dim=-1, keepdim=True) / denom).sqrt()
            noise = noise * row_rms.clamp_min(float(self.residual_sigma_floor))
        if (
            label == "edge"
            and self.physical_prior_jitter_edge_decay > 0.0
            and data is not None
            and _keys.EDGE_LENGTH_KEY in data
        ):
            edge_lengths = data[_keys.EDGE_LENGTH_KEY].to(device=like.device, dtype=like.dtype)
            edge_lengths = edge_lengths.reshape(-1)
            if int(edge_lengths.shape[0]) == int(like.shape[0]):
                view_shape = (edge_lengths.shape[0],) + (1,) * (like.ndim - 1)
                noise = noise * torch.exp(
                    -edge_lengths.reshape(view_shape) / self.physical_prior_jitter_edge_decay
                )
        return noise * (float(sigma) * self.physical_prior_jitter_sigma)

    def _physical_prior_like(
        self,
        residual: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        base: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # basis_onsite / overlap_huckel / external / dftbsk all resolve through a
        # family instance; only the cross-family fallback chain and the
        # fail-closed errors stay here.
        ctx = self._prior_ctx
        absolute_prior = self.prior_family.absolute_prior_like(
            residual, data=data, label=label, ctx=ctx
        )

        # The external family may additionally try an on-the-fly DFTB-SK guess.
        if absolute_prior is None and self.prior in self._external_prior_names:
            absolute_prior = self._families[DFTBSKFamily].absolute_prior_like(
                residual, data=data, label=label, ctx=ctx
            )

        # Silent basis_onsite fallback is now reserved for prior='physical'
        # (governed by physical_prior_fallback); dftb/xtb/sk/nnsk fail closed.
        if (
            absolute_prior is None
            and self.prior == "physical"
            and self.physical_prior_fallback == "basis_onsite"
        ):
            absolute_prior = self._families[BasisOnsiteFamily].absolute_prior_like(
                residual, data=data, label=label, ctx=ctx
            )

        if absolute_prior is None:
            absolute_prior = self._physical_prior_absent(residual, data=data, label=label)

        prior = self._absolute_to_flow_prior(absolute_prior, residual, base=base)
        return prior + self._physical_prior_jitter_like(
            residual,
            sigma,
            data=data,
            label=label,
        )

    def _physical_prior_absent(
        self,
        residual: torch.Tensor,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
    ) -> torch.Tensor:
        """No family produced an absolute prior: fail closed or degrade to zeros."""
        if self.prior in self._basis_prior_names:
            raise ValueError(
                "flow_options.prior='basis_onsite' requires an OrbitalMapper-like idp "
                "with basis_to_full_basis/orbpair_maps and node atom_types."
            )
        if self.prior in self._dftbsk_prior_names:
            raise ValueError(
                "flow_options.prior='dftbsk' requires a successful on-the-fly "
                "DFTB-SK initialization; set prior_skdata/dftb_skdata to a "
                "Slater-Koster directory or formatted .pth."
            )
        if self.prior in self._overlap_huckel_prior_names:
            raise ValueError(
                "flow_options.prior='overlap_huckel' requires basis onsite levels "
                "and edge overlap features; set huckel_edge_overlap_key or precompute overlap."
            )
        # External family (external/dftb/xtb/physical/sk/nnsk/dftb_xtb).
        external = self._families[ExternalFamily]
        if (
            (self.prior == "physical" and self.physical_prior_fallback == "zero")
            or not external.external_prior_strict
        ):
            return torch.zeros_like(residual)
        keys = ", ".join(external.candidate_keys(label)[:8])
        raise KeyError(
            f"flow_options.prior={self.prior!r} did not find an external "
            f"{label or 'state'} prior. Tried keys: {keys}."
        )

    def _prior_mask(
        self,
        data: Optional[AtomicDataDict.Type],
        like: torch.Tensor,
        label: Optional[str],
    ) -> torch.Tensor:
        mask = torch.ones_like(like, dtype=torch.bool, device=like.device)
        if data is None or self.idp is None or like.ndim < 2:
            return mask

        if label == "node":
            type_key = AtomicDataDict.ATOM_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_nrme", None)
            expert_key = "expert_node_mask"
        elif label == "edge":
            type_key = AtomicDataDict.EDGE_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_erme", None)
            expert_key = "expert_edge_mask"
        else:
            return mask

        types = data.get(type_key, None)
        if types is not None and mask_table is not None:
            table = mask_table.to(device=like.device, dtype=torch.bool)
            if table.ndim == 0:
                table = table.reshape(1, 1)
            elif table.ndim == 1:
                table = table.reshape(-1, 1)
            elif table.ndim > 2:
                table = table.reshape(table.shape[0], -1)
            table = self._project_raw_feature_table(table, like.shape[-1])

            type_mask = torch.zeros(
                like.shape[0],
                table.shape[1],
                device=like.device,
                dtype=torch.bool,
            )
            if table.shape[0] > 0:
                row_types = types.to(device=like.device, dtype=torch.long).reshape(-1)
                take = min(int(row_types.numel()), int(like.shape[0]))
                if take > 0:
                    raw_types = row_types[:take]
                    valid = (raw_types >= 0) & (raw_types < table.shape[0])
                    valid_rows = torch.arange(take, device=like.device, dtype=torch.long)[valid]
                    if valid_rows.numel() > 0:
                        type_mask[valid_rows] = table.index_select(0, raw_types[valid])
            mask = mask & self._align_bool_mask(type_mask, like)

        expert_mask = data.get(expert_key, None)
        if expert_mask is not None:
            mask = mask & self._align_bool_mask(expert_mask, like)
        return mask

    def _row_graph_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> torch.Tensor:
        if count <= 0 or data is None:
            return torch.zeros(count, device=device, dtype=torch.long)

        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return torch.zeros(count, device=device, dtype=torch.long)
        batch = batch.to(device=device, dtype=torch.long).reshape(-1)

        if label == "node":
            if batch.numel() < count:
                batch = torch.cat([batch, batch.new_zeros(count - batch.numel())], dim=0)
            return batch[:count]

        if label == "edge":
            edge_index = data.get(_keys.EDGE_INDEX_KEY, None)
            if edge_index is None or edge_index.numel() == 0:
                return torch.zeros(count, device=device, dtype=torch.long)
            centers = edge_index[0].to(device=device, dtype=torch.long).reshape(-1)
            if centers.numel() < count:
                centers = torch.cat([centers, centers.new_zeros(count - centers.numel())], dim=0)
            centers = centers[:count].clamp(min=0, max=max(batch.numel() - 1, 0))
            return batch.index_select(0, centers)

        return torch.zeros(count, device=device, dtype=torch.long)

    def _row_type_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if count <= 0 or data is None:
            return None
        if label == "node":
            key = AtomicDataDict.ATOM_TYPE_KEY
        elif label == "edge":
            key = AtomicDataDict.EDGE_TYPE_KEY
        else:
            return None
        values = data.get(key, None)
        if values is None:
            return None
        values = values.to(device=device, dtype=torch.long).reshape(-1)
        if values.numel() < count:
            values = torch.cat([values, values.new_zeros(count - values.numel())], dim=0)
        return values[:count]

    def _te_irrep_slices(self, feature_dim: int) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        feature_dim = int(feature_dim)
        cache = getattr(self, "_te_irrep_slices_cache", None)
        if isinstance(cache, dict) and feature_dim in cache:
            return cache[feature_dim]

        def _remember(value):
            if isinstance(cache, dict):
                cache[feature_dim] = value
            return value

        if self.idp is None:
            return _remember(None)
        irreps = getattr(self.idp, "orbpair_irreps", None)
        if irreps is None:
            get_irreps = getattr(self.idp, "get_irreps", None)
            if not callable(get_irreps):
                return _remember(None)
            try:
                irreps = get_irreps()
            except Exception:
                return _remember(None)
            if irreps is None:
                return _remember(None)
        # Feature rows follow OrbitalMapper/orbpair_maps order. Sorting irreps
        # changes contiguous feature spans and breaks mask/typewise raw-slice priors.
        raw_irreps = irreps

        slices = []
        offset = 0
        try:
            for mul, ir in raw_irreps:
                degree = int(getattr(ir, "l", 0))
                width = int(getattr(ir, "dim", 2 * degree + 1))
                for _ in range(int(mul)):
                    slices.append((offset, offset + width, degree))
                    offset += width
        except Exception:
            return _remember(None)
        if offset != feature_dim:
            raw_mask = getattr(self.idp, "mask_uureal", None)
            if raw_mask is None:
                return _remember(None)
            raw_mask = raw_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            if raw_mask.numel() != offset:
                return _remember(None)
            if int(raw_mask.sum().item()) != feature_dim:
                return _remember(None)
            projected_slices = []
            compressed_offset = 0
            for start, end, degree in slices:
                kept = int(raw_mask[start:end].sum().item())
                if kept <= 0:
                    continue
                projected_slices.append((compressed_offset, compressed_offset + kept, degree))
                compressed_offset += kept
            if compressed_offset != feature_dim:
                return _remember(None)
            return _remember(tuple(projected_slices))
        return _remember(tuple(slices))

    def _te_radius(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.te_prior_per_graph and graph_index is not None and graph_index.numel() == row_count:
            if num_graphs is None:
                num_graphs = int(graph_index.max().detach().item()) + 1 if row_count > 0 else 1
            radius = torch.randn(
                num_graphs,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            ).index_select(0, graph_index)
        else:
            radius = torch.randn(
                row_count,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        return radius * active_dim.sqrt()

    def _te_radius_for_prior(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        optional = {"num_graphs": num_graphs}
        if generator is not None:
            optional["generator"] = generator
        while True:
            try:
                return self._te_radius(
                    row_count,
                    active_dim,
                    graph_index,
                    device=device,
                    dtype=dtype,
                    **optional,
                )
            except TypeError as exc:
                unsupported = next(
                    (name for name in optional if name in str(exc)), None
                )
                if unsupported is None:
                    raise
                optional.pop(unsupported)

    def _block_structured_prior_like(
        self,
        like: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        *,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        per_graph_generators: Optional[list] = None,
        row_slices: Optional[list] = None,
        graph_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The 1D unstructured fallback carries no per-row graph identity; it is
        # unreachable for the SEEDED projected_te contract (which always draws
        # rank>=2 RME rows in irrep mode), so it keeps the single-stream draw.
        if like.ndim < 2:
            return torch.randn(
                like.shape, device=like.device, dtype=like.dtype, generator=generator
            )
        mask_f = mask.to(device=like.device, dtype=like.dtype)
        seeded = per_graph_generators is not None
        if seeded:
            raw = self._seeded_rows_by_graph(
                like.shape[0],
                like.shape[-1],
                row_slices,
                per_graph_generators,
                device=like.device,
                dtype=like.dtype,
            ) * mask_f
        else:
            raw = torch.randn(
                like.shape, device=like.device, dtype=like.dtype, generator=generator
            ) * mask_f
        norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
        direction = raw / norm
        active_dim = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if graph_index is None:
            graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
        if seeded:
            radius = self._seeded_radius_by_graph(
                like.shape[0],
                active_dim,
                graph_index,
                row_slices,
                per_graph_generators,
                device=like.device,
                dtype=like.dtype,
            )
        else:
            radius = self._te_radius_for_prior(
                like.shape[0],
                active_dim,
                graph_index,
                device=like.device,
                dtype=like.dtype,
                num_graphs=num_graphs,
                generator=generator,
            )
        return direction * radius * mask_f

    def _apply_typewise_residual_scale(
        self,
        noise: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        slices: Tuple[Tuple[int, int, int], ...],
        *,
        reference_scale: bool,
    ) -> torch.Tensor:
        if self.te_prior_mode != "typewise" or not reference_scale:
            return noise
        type_index = self._row_type_index(data, noise.shape[0], label, noise.device)
        if type_index is None or noise.ndim < 2:
            return noise

        out = noise.clone()
        ref = reference.detach().to(device=noise.device, dtype=noise.dtype)
        mask_f = mask.to(device=noise.device, dtype=noise.dtype)
        _types, inverse = torch.unique(type_index, sorted=True, return_inverse=True)
        num_types = int(_types.numel())
        if num_types == 0:
            return out

        for start, end, _degree in slices:
            seg_mask = mask_f[:, start:end]
            row_count = seg_mask.sum(dim=-1)
            row_square_sum = (ref[:, start:end].square() * seg_mask).sum(dim=-1)

            type_count = torch.zeros(
                num_types, device=noise.device, dtype=noise.dtype
            ).scatter_add_(0, inverse, row_count)
            type_square_sum = torch.zeros_like(type_count).scatter_add_(
                0, inverse, row_square_sum
            )

            rms = torch.sqrt(type_square_sum / type_count.clamp_min(1.0))
            valid = (type_count > 0) & torch.isfinite(rms)
            scale_by_type = torch.where(
                valid,
                rms.clamp_min(self.residual_sigma_floor),
                torch.ones_like(rms),
            )
            out[:, start:end] = (
                out[:, start:end] * scale_by_type.index_select(0, inverse).unsqueeze(-1)
            )
        return out

    def _te_prior_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        reference_scale: bool = True,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if like.numel() == 0:
            return torch.zeros_like(like)

        mask = self._prior_mask(data, like, label)
        if self.te_prior_mode == "block":
            slices = None
        else:
            if like.ndim < 2:
                raise ValueError(
                    f"flow_options.te_prior_mode={self.te_prior_mode!r} requires "
                    "DeePTB RME feature rows with rank >= 2; use te_prior_mode='block' "
                    "for unstructured tensors."
                )
            slices = self._te_irrep_slices(like.shape[-1])
            if slices is None:
                raise ValueError(
                    f"flow_options.te_prior_mode={self.te_prior_mode!r} requires "
                    "idp.orbpair_irreps raw feature spans to match "
                    f"{label or 'unknown'} feature_dim={like.shape[-1]}; "
                    "use te_prior_mode='block' for whole-row structured noise."
                )

        # TA-2: when the draw is SEEDED, replace the single batch-order RNG stream
        # (whose per-graph output changes with batch composition/order/sharding)
        # with one independent generator per graph keyed by its stable sample_uid.
        # ``base_seed`` is the passed generator's initial seed, so node and edge
        # (distinct component tags) decouple while a different ``prior_seed`` still
        # redraws the whole batch.  Unseeded (training) draws keep the byte-
        # identical fast batch path below.
        seeded = generator is not None
        per_graph_generators = None
        row_slices = None
        graph_index = None
        if seeded:
            if num_graphs is None:
                num_graphs = self._num_graphs(data) if data is not None else 1
            per_graph_generators = self._prior_substream_generators(
                data,
                base_seed=int(generator.initial_seed()),
                num_graphs=num_graphs,
                component=label,
                device=like.device,
            )
            graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
            row_slices = self._graph_row_slices(graph_index, num_graphs)

        if like.ndim < 2 or slices is None:
            slices = ((0, like.shape[-1], -1),) if like.ndim >= 2 else ((0, like.numel(), -1),)
            noise = self._block_structured_prior_like(
                like,
                mask,
                data,
                label,
                num_graphs=num_graphs,
                generator=generator,
                per_graph_generators=per_graph_generators,
                row_slices=row_slices,
                graph_index=graph_index,
            )
        else:
            noise = torch.zeros_like(like)
            if graph_index is None:
                graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
            for start, end, _degree in slices:
                seg_mask = mask[:, start:end].to(device=like.device, dtype=like.dtype)
                if seeded:
                    raw = self._seeded_rows_by_graph(
                        like.shape[0],
                        end - start,
                        row_slices,
                        per_graph_generators,
                        device=like.device,
                        dtype=like.dtype,
                    )
                else:
                    raw = torch.randn(
                        like.shape[0],
                        end - start,
                        device=like.device,
                        dtype=like.dtype,
                        generator=generator,
                    )
                raw = raw * seg_mask
                norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
                direction = raw / norm
                active_dim = seg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                if seeded:
                    radius = self._seeded_radius_by_graph(
                        like.shape[0],
                        active_dim,
                        graph_index,
                        row_slices,
                        per_graph_generators,
                        device=like.device,
                        dtype=like.dtype,
                    )
                else:
                    radius = self._te_radius_for_prior(
                        like.shape[0],
                        active_dim,
                        graph_index,
                        device=like.device,
                        dtype=like.dtype,
                        num_graphs=num_graphs,
                        generator=generator,
                    )
                noise[:, start:end] = direction * radius * seg_mask

        noise = self._apply_typewise_residual_scale(
            noise,
            like,
            mask,
            data,
            label,
            slices,
            reference_scale=reference_scale,
        )
        return noise * (float(sigma) * self.te_prior_sigma)

    def _tied_irrep_gaussian_prior_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if like.numel() == 0:
            return torch.zeros_like(like)
        if like.ndim < 2:
            raise ValueError(
                "tied_irrep_gaussian requires DeePTB RME feature rows with rank >= 2."
            )
        validate_tied_irrep_options(
            mode=self.tied_irrep_mode,
            irreps=self.tied_irrep_irreps,
        )
        slices = self._te_irrep_slices(like.shape[-1])
        if slices is None:
            raise ValueError(
                "tied_irrep_gaussian requires idp.orbpair_irreps raw feature "
                f"spans to match {label or 'unknown'} feature_dim={like.shape[-1]}."
            )
        mask = self._prior_mask(data, like, label)

        seeded = generator is not None
        if seeded:
            if num_graphs is None:
                num_graphs = self._num_graphs(data) if data is not None else 1
            per_graph_generators = self._prior_substream_generators(
                data,
                base_seed=int(generator.initial_seed()),
                num_graphs=num_graphs,
                component=label,
                device=like.device,
            )
            graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
            row_slices = self._graph_row_slices(graph_index, num_graphs)
            standard_latent = self._seeded_rows_by_graph(
                like.shape[0],
                TIED_IRREP_LATENT_WIDTH,
                row_slices,
                per_graph_generators,
                device=like.device,
                dtype=like.dtype,
            )
        else:
            standard_latent = draw_standard_tied_irrep_latent(
                like.shape[0],
                device=like.device,
                dtype=like.dtype,
                generator=generator,
            )

        effective_latent = effective_tied_irrep_latent(standard_latent)
        return fill_tied_irrep_rme(
            like,
            slices,
            mask,
            effective_latent,
            sigma=float(sigma) * self.tied_irrep_sigma,
        )

    def _prior_like(
        self,
        residual: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        base: Optional[torch.Tensor] = None,
        reference_scale: bool = True,
        num_graphs: Optional[int] = None,
        candidate_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.prior == "zero":
            return torch.zeros_like(residual)
        if self.prior == "gaussian":
            return torch.randn_like(residual) * sigma
        if self.prior in self._te_prior_names:
            return self._te_prior_like(
                residual,
                sigma,
                data=data,
                label=label,
                reference_scale=reference_scale,
                num_graphs=num_graphs,
            )
        if self.prior in self._haar_dm_prior_names:
            absolute_prior = self.prior_family.absolute_prior_like(
                residual,
                data=data,
                label=label,
                ctx=self._prior_ctx,
                candidate_idx=candidate_idx,
            )
            if absolute_prior is None:
                return torch.zeros_like(residual)
            return self._absolute_to_flow_prior(absolute_prior, residual, base=base)
        if self.prior_family is not None:
            return self._physical_prior_like(
                residual,
                sigma,
                data=data,
                label=label,
                base=base,
            )
        # residual_gaussian: match global residual scale, useful as a rough TE/GOE proxy.
        scale = residual.detach().std().clamp_min(self.residual_sigma_floor)
        return torch.randn_like(residual) * scale * sigma

    @staticmethod
    def _require_real_finite_tensor(value: Any, *, label: str) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.is_complex():
            raise ValueError(f"{label} must be real for non-SOC block-space ODE.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{label} contains NaN or Inf.")
        return tensor

    @staticmethod
    def _block_primary_topology_keys() -> Tuple[str, ...]:
        """Primary graph metadata defining AO row identity and reverse pairs."""
        names = (
            ("EDGE_INDEX_KEY", "edge_index"),
            ("EDGE_CELL_SHIFT_KEY", "edge_cell_shift"),
            ("ATOMIC_NUMBERS_KEY", "atomic_numbers"),
            ("ATOM_TYPE_KEY", "atom_types"),
            ("POSITIONS_KEY", "pos"),
            ("BATCH_KEY", "batch"),
            ("PBC_KEY", "pbc"),
            ("EDGE_TYPE_KEY", "edge_type"),
            ("CELL_KEY", "cell"),
        )
        keys = [str(getattr(_keys, name, fallback)) for name, fallback in names]
        # Some legacy/raw dictionaries use the plural spelling even though the
        # canonical AtomicData key is singular.
        keys.append("edge_types")
        return tuple(dict.fromkeys(keys))

    @classmethod
    def _block_topology_keys(cls) -> Tuple[str, ...]:
        """Graph metadata a model output may never redefine during block ODE."""
        names = (
            # Derived geometry is topology-dependent too.  If it was absent on
            # entry, discard a model-returned value so the next step recomputes
            # it from the immutable primary graph instead of trusting stale data.
            ("EDGE_VECTORS_KEY", "edge_vectors"),
            ("EDGE_LENGTH_KEY", "edge_lengths"),
        )
        keys = list(cls._block_primary_topology_keys())
        keys.extend(str(getattr(_keys, name, fallback)) for name, fallback in names)
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _missing_keys(data: AtomicDataDict.Type, keys: Tuple[str, ...]) -> list[str]:
        return [key for key in keys if key not in data]

    @classmethod
    def _metadata_scalar(cls, value: Any) -> Any:
        if torch.is_tensor(value):
            values = value.detach().cpu().reshape(-1).tolist()
            if not values or any(item != values[0] for item in values[1:]):
                raise ValueError(
                    "uureal_block_ode batched metadata values must be nonempty and identical."
                )
            return values[0]
        if isinstance(value, (list, tuple)):
            values = [cls._metadata_scalar(item) for item in value]
            if not values or any(item != values[0] for item in values[1:]):
                raise ValueError(
                    "uureal_block_ode batched metadata values must be nonempty and identical."
                )
            return values[0]
        try:
            import numpy as np
        except Exception:
            return value
        array = np.asarray(value)
        values = array.reshape(-1).tolist()
        if not values or any(item != values[0] for item in values[1:]):
            raise ValueError(
                "uureal_block_ode batched metadata values must be nonempty and identical."
            )
        return values[0]

    def _require_uureal_block_contract(
        self,
        data: AtomicDataDict.Type,
        *,
        require_endpoint_labels: bool = True,
    ) -> None:
        keep = int(self.idp.reduced_matrix_element)
        required = (
            "blockwise_spatial_schema",
            "blockwise_target_mode",
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
            "soc_uureal_compact",
            "soc_uureal_full_rme",
            "soc_uureal_keep",
            self.node_h0_key,
            self.edge_h0_key,
        )
        if require_endpoint_labels:
            # Training/scoring needs the residual endpoint labels; label-free
            # inference (sampling) starts from D_0=0 with mapper-derived shapes
            # and never reads them, so it must not demand them.
            required = required + (
                self.node_block_target_key,
                self.edge_block_target_key,
                self.node_block_shape_key,
                self.edge_block_shape_key,
            )
        missing = self._missing_keys(data, required)
        if missing:
            raise KeyError(f"uureal_block_ode data contract missing keys={missing}.")
        expected = {
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
        for key, wanted in expected.items():
            actual = self._metadata_scalar(data[key])
            if actual != wanted:
                raise ValueError(
                    f"uureal_block_ode requires {key}={wanted!r}; got {actual!r}."
                )
        # A normal full-SOC->uu_real conversion records the ORIGINAL source width
        # (e.g. 5832 for a keep=729 compact target).  The contract is that the
        # *stored* tensors are keep-wide and keep matches the mapper; the recorded
        # source width must only be a valid integer >= keep (keep==source when the
        # source was already compact).  Forcing source_width==keep rejected every
        # genuine converter product.
        for key in (
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
        ):
            raw_actual = self._metadata_scalar(data[key])
            try:
                actual = int(raw_actual)
            except (TypeError, ValueError):
                raise ValueError(
                    f"uureal_block_ode requires integer {key}; got {raw_actual!r}."
                )
            if actual < keep:
                raise ValueError(
                    f"uureal_block_ode requires {key} >= keep={keep}; got {actual}."
                )
        for key in (self.node_h0_key, self.edge_h0_key):
            value = torch.as_tensor(data[key])
            if value.ndim != 2 or value.shape[-1] != keep:
                raise ValueError(
                    f"uureal_block_ode requires {key} width {keep}; got {tuple(value.shape)}."
                )
            if value.is_complex() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"uureal_block_ode requires finite real {key}.")

    @staticmethod
    def _attach_uureal_residual_state(data: AtomicDataDict.Type, state: BlockTensorResult) -> None:
        data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = state.node_blocks
        data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY] = state.edge_blocks
        data[_keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.node_shapes
        data[_keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.edge_shapes

    def _require_spatial_residual_block_contract(
        self,
        data: AtomicDataDict.Type,
        *,
        require_endpoint_labels: bool = True,
    ) -> None:
        """Fail-closed non-SOC direct-residual block contract.

        ``residual_ao_block_ode`` consumes plain non-SOC raw records: the physical
        H0 AO blocks (+ shapes) are always required, while the delta endpoint
        labels are required only for training/scoring.  Label-free sampling starts
        from D_0=0 with mapper-derived shapes and never reads them, so it must not
        demand them.  Mapper non-SOC-ness is a ctor check, not a per-batch one.
        """
        required = (
            self.node_h0_block_key,
            self.edge_h0_block_key,
            self.node_h0_block_shape_key,
            self.edge_h0_block_shape_key,
        )
        if require_endpoint_labels:
            required = required + (
                self.node_block_target_key,
                self.edge_block_target_key,
                self.node_block_shape_key,
                self.edge_block_shape_key,
            )
        missing = self._missing_keys(data, required)
        if missing:
            raise KeyError(f"residual_ao_block_ode data contract missing keys={missing}.")
        for key in (self.node_h0_block_key, self.edge_h0_block_key):
            self._require_real_finite_tensor(
                data[key], label=f"residual_ao_block_ode {key}"
            )
        if require_endpoint_labels:
            for key in (self.node_block_target_key, self.edge_block_target_key):
                self._require_real_finite_tensor(
                    data[key], label=f"residual_ao_block_ode {key}"
                )
        # A raw non-SOC record must never carry compact uu_real metadata: those
        # markers belong to uureal_block_ode and would signal a converter product
        # masquerading as a plain spatial record.
        forbidden_metadata = [
            key
            for key in (
                "soc_uureal_compact",
                "soc_uureal_full_rme",
                "soc_uureal_keep",
                "blockwise_spatial_schema",
                "blockwise_target_mode",
            )
            if key in data
        ]
        if forbidden_metadata:
            raise ValueError(
                "residual_ao_block_ode consumes raw non-SOC records; uu-real "
                f"compact metadata is forbidden (found {forbidden_metadata})."
            )
        # Nor the reduced-SOC delta state keys, which would silently redirect the
        # conditioning channel away from the spatial residual projector.
        forbidden_state = [
            key
            for key in (
                _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
                _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
                _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
                _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
            )
            if key in data
        ]
        if forbidden_state:
            raise ValueError(
                "residual_ao_block_ode consumes raw non-SOC records; uu-real "
                f"residual state keys are forbidden (found {forbidden_state})."
            )

    @staticmethod
    def _attach_spatial_residual_state(data: AtomicDataDict.Type, state: BlockTensorResult) -> None:
        data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.node_blocks
        data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.edge_blocks
        data[_keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.node_shapes
        data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.edge_shapes

    @staticmethod
    def _clone_sidecar_value(value: Any) -> Any:
        return value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    @classmethod
    def _snapshot_block_topology(cls, data: AtomicDataDict.Type) -> Dict[str, Any]:
        return {
            key: cls._clone_sidecar_value(data[key])
            for key in cls._block_topology_keys()
            if key in data
        }

    def _drop_block_authority_fields(self, data: AtomicDataDict.Type) -> None:
        """Keep certified endpoint/H0 block side channels outside model I/O."""
        for key in (
            self.node_block_target_key,
            self.edge_block_target_key,
            self.node_block_shape_key,
            self.edge_block_shape_key,
            self.node_h0_block_key,
            self.edge_h0_block_key,
            self.node_h0_block_shape_key,
            self.edge_h0_block_shape_key,
        ):
            data.pop(key, None)

    def _block_ode_output_only_keys(self) -> Tuple[str, ...]:
        """Model outputs that must never be recycled as the next step's input."""
        return tuple(
            dict.fromkeys(
                (
                    self.node_output_key,
                    self.edge_output_key,
                    *_BLOCK_ODE_OUTPUT_ONLY_KEYS,
                )
            )
        )

    def _require_fresh_block_ode_outputs(
        self,
        prediction: AtomicDataDict.Type,
        *,
        step: int,
    ) -> None:
        missing = self._missing_keys(
            prediction, (self.node_output_key, self.edge_output_key)
        )
        if missing:
            raise ValueError(
                f"Block-space ODE step {step} is missing fresh model output keys={missing}."
            )

    @classmethod
    def _require_matching_block_topology(
        cls,
        data_topology: Dict[str, Any],
        ref_topology: Dict[str, Any],
    ) -> None:
        for key in cls._block_primary_topology_keys():
            in_data = key in data_topology
            in_ref = key in ref_topology
            if in_data != in_ref:
                raise ValueError(
                    "Block-space ODE data/ref topology mismatch: "
                    f"key {key!r} is present in only one dictionary."
                )
            if not in_data:
                continue
            left = data_topology[key]
            right = ref_topology[key]
            if torch.is_tensor(left) or torch.is_tensor(right):
                try:
                    left_t = torch.as_tensor(left).detach().cpu()
                    right_t = torch.as_tensor(right).detach().cpu()
                except Exception as exc:
                    raise ValueError(
                        f"Block-space ODE cannot compare data/ref topology key {key!r}."
                    ) from exc
                equal = left_t.shape == right_t.shape and torch.equal(left_t, right_t)
            else:
                try:
                    equal = bool(left == right)
                except Exception:
                    equal = False
            if not equal:
                raise ValueError(
                    "Block-space ODE data/ref topology mismatch at "
                    f"key {key!r}; row-aligned H0/endpoint blocks cannot be mixed."
                )

    @classmethod
    def _restore_block_topology(
        cls,
        data: AtomicDataDict.Type,
        snapshot: Dict[str, Any],
        *,
        clone_values: bool = False,
    ) -> None:
        for key in cls._block_topology_keys():
            if key in snapshot:
                value = snapshot[key]
                data[key] = cls._clone_sidecar_value(value) if clone_values else value
            else:
                data.pop(key, None)

    @classmethod
    def _max_abs(cls, value: torch.Tensor, *, label: str = "residual") -> float:
        value = cls._require_real_finite_tensor(value, label=label)
        if value.numel() == 0:
            return 0.0
        return float(value.detach().abs().max().item())

    def _block_state_from_fields(
        self,
        field_data: AtomicDataDict.Type,
        topology_data: AtomicDataDict.Type,
        *,
        node_key: str,
        edge_key: str,
        node_shape_key: str,
        edge_shape_key: str,
        label: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[BlockTensorResult, float]:
        missing = self._missing_keys(
            field_data, (node_key, edge_key, node_shape_key, edge_shape_key)
        )
        if missing:
            raise KeyError(f"Block-space ODE requires {label} blocks and shapes; missing keys={missing}.")
        raw_node = self._require_real_finite_tensor(
            field_data[node_key], label=f"{label} node blocks"
        )
        raw_edge = self._require_real_finite_tensor(
            field_data[edge_key], label=f"{label} edge blocks"
        )
        raw = BlockTensorResult(
            node_blocks=raw_node.to(device=device, dtype=dtype),
            edge_blocks=raw_edge.to(device=device, dtype=dtype),
            node_shapes=torch.as_tensor(field_data[node_shape_key], device=device),
            edge_shapes=torch.as_tensor(field_data[edge_shape_key], device=device),
        )
        projected = project_block_state(topology_data, self.idp, raw)
        residual = max(
            self._max_abs(
                projected.node_blocks - raw.node_blocks,
                label=f"{label} node projection residual",
            ),
            self._max_abs(
                projected.edge_blocks - raw.edge_blocks,
                label=f"{label} edge projection residual",
            ),
        )
        if residual > self.block_inverse_atol:
            raise ValueError(
                f"Block-space ODE {label} violates onsite/reverse/padding constraints: "
                f"max residual={residual:.6g}, atol={self.block_inverse_atol:.6g}."
            )
        return projected, residual

    @staticmethod
    def _clone_block_state(state: BlockTensorResult) -> BlockTensorResult:
        return BlockTensorResult(
            state.node_blocks.clone(),
            state.edge_blocks.clone(),
            state.node_shapes.clone(),
            state.edge_shapes.clone(),
        )

    def _physical_h0_blocks(
        self,
        data: AtomicDataDict.Type,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[BlockTensorResult, float]:
        missing = self._missing_keys(
            data,
            (
                self.node_h0_block_key,
                self.edge_h0_block_key,
                self.node_h0_block_shape_key,
                self.edge_h0_block_shape_key,
            ),
        )
        if missing:
            raise KeyError(
                "Block-space ODE requires physical H0 blocks and shapes; "
                f"missing keys={missing}."
            )
        raw_node = self._require_real_finite_tensor(
            data[self.node_h0_block_key], label="physical H0 node blocks"
        )
        return self._block_state_from_fields(
            data,
            data,
            node_key=self.node_h0_block_key,
            edge_key=self.edge_h0_block_key,
            node_shape_key=self.node_h0_block_shape_key,
            edge_shape_key=self.edge_h0_block_shape_key,
            label="physical H0",
            device=raw_node.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _block_initial_state(
        self,
        data: AtomicDataDict.Type,
        h0_blocks: BlockTensorResult,
        *,
        prior_seed: Optional[int] = None,
        certify_image: bool = True,
    ) -> Tuple[BlockTensorResult, torch.Tensor, torch.Tensor]:
        """Draw one endpoint-independent start state and certify its codec image.

        Seeded-replay semantics (TA-1/TA-2): a fixed ``prior_seed`` reproduces the
        SAME start state bitwise only for an identical tensor layout, plus
        DISTRIBUTIONAL equivariance; the draw is keyed per graph by ``sample_uid``
        (batch-composition invariant) but is NOT pathwise equivariant under an
        input rotation/permutation.

        A-mode asymmetry (TA-1a): unlike ``residual_ao_block_ode``, this A-mode
        start state is the FULL state ``B0 = H0 + eps`` (not a pure residual D0),
        so an explicit transformable ``prior_state`` latent does NOT drop in
        trivially here (a caller-supplied full B0 would need its own H0-relative
        residual bookkeeping and a distinct verbatim-vs-projected contract).  The
        explicit-latent path is therefore offered only by the residual sampler;
        A-mode is left seeded-only by design.
        """
        assert self.block_codec is not None
        node_h0, edge_h0 = self.block_codec.blocks_to_rme(
            data,
            h0_blocks,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
        if self.prior == "zero":
            if prior_seed is not None:
                raise ValueError("block-space prior_seed requires prior='projected_te'.")
            return (
                self._clone_block_state(h0_blocks),
                torch.zeros_like(node_h0),
                torch.zeros_like(edge_h0),
            )

        if self.prior not in self._projected_te_prior_names:
            raise RuntimeError(f"Unsupported block-space prior {self.prior!r}.")

        def draw(
            generator: Optional[torch.Generator],
        ) -> Tuple[BlockTensorResult, torch.Tensor, torch.Tensor]:
            prior_data = {
                key: data[key]
                for key in (
                    AtomicDataDict.ATOM_TYPE_KEY,
                    AtomicDataDict.EDGE_TYPE_KEY,
                    _keys.BATCH_KEY,
                    _keys.EDGE_INDEX_KEY,
                    # TA-2: the stable per-graph id must ride the restricted prior
                    # substream data so a SEEDED draw can key per-uid generators.
                    _keys.SAMPLE_UID_KEY,
                )
                if key in data
            }
            num_graphs = self._num_graphs(prior_data)
            node_noise = self._te_prior_like(
                torch.zeros_like(node_h0),
                self.node_sigma,
                data=prior_data,
                label="node",
                reference_scale=False,
                num_graphs=num_graphs,
                generator=generator,
            )
            edge_noise = self._te_prior_like(
                torch.zeros_like(edge_h0),
                self.edge_sigma,
                data=prior_data,
                label="edge",
                reference_scale=False,
                num_graphs=num_graphs,
                generator=generator,
            )
            # TA-3: runtime belt on the projected_te draw before it is folded into
            # the start state -- all-finite, and not silently all-zero for a
            # component with a nonzero configured effective scale.
            self._assert_projected_te_draw_finite_and_scaled(
                (
                    ("node", node_noise, self.node_sigma * self.te_prior_sigma),
                    ("edge", edge_noise, self.edge_sigma * self.te_prior_sigma),
                )
            )
            noise_blocks = self.block_codec.rme_to_blocks(
                data, node_noise, edge_noise, project=False
            )
            start = project_block_state(
                data,
                self.idp,
                BlockTensorResult(
                    h0_blocks.node_blocks + noise_blocks.node_blocks,
                    h0_blocks.edge_blocks + noise_blocks.edge_blocks,
                    h0_blocks.node_shapes,
                    h0_blocks.edge_shapes,
                ),
            )
            node_start, edge_start = self.block_codec.blocks_to_rme(
                data,
                start,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            roundtrip = self.block_codec.rme_to_blocks(
                data, node_start, edge_start, project=True
            )
            residual = max(
                self._max_abs(roundtrip.node_blocks - start.node_blocks),
                self._max_abs(roundtrip.edge_blocks - start.edge_blocks),
            )
            if residual > self.block_inverse_atol:
                raise ValueError(
                    "Projected TE block start is outside the certified codec image: "
                    f"max residual={residual:.6g}, atol={self.block_inverse_atol:.6g}."
                )
            return start, node_start - node_h0, edge_start - edge_h0

        if prior_seed is None:
            return draw(None)
        generator = self._seeded_generator(h0_blocks.node_blocks.device, prior_seed)
        return draw(generator)

    def _residual_te_eps(
        self,
        data: AtomicDataDict.Type,
        node_like: torch.Tensor,
        edge_like: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
        certify_image: bool,
    ) -> BlockTensorResult:
        """Draw the projected_te residual epsilon in block space; NO H0 added.

        Mirrors the TE-noise draw of :meth:`_block_initial_state` exactly (same
        node/edge sigma semantics and the same seeded-generator path) but forms
        the PURE noise residual ``eps = project(rme_to_blocks(noise, project=False))``
        instead of A's ``H0 + noise`` start state: the direct-residual mode tracks
        ``D = H - H0``, so the t=0 boundary of D is the prior noise alone.

        ``eps`` is in the certified codec image by construction (the forward CG map
        is linear, ``rme_to_blocks(project=False)`` lands on that image, and
        :func:`project_block_state` is an image-preserving projection).  When
        certification is due the epsilon is certified the same way A certifies its
        noisy start -- a repack roundtrip residual bounded by ``block_inverse_atol``.

        Seeded-replay semantics (TA-1/TA-2): a fixed ``generator`` reproduces the
        SAME epsilon bitwise only for an identical tensor layout, and otherwise
        only DISTRIBUTIONAL equivariance -- sample(R.x, seed) is drawn from the
        rotation of sample(x, seed)'s law, not equal to it.  The draw is keyed
        per graph by ``sample_uid`` (batch-composition invariant) but is NOT
        pathwise equivariant under an input rotation/permutation; for a
        transformable latent, pass an explicit ``prior_state`` to the sampler and
        rotate/permute it alongside the input.
        """
        assert self.block_codec is not None
        prior_data = {
            key: data[key]
            for key in (
                AtomicDataDict.ATOM_TYPE_KEY,
                AtomicDataDict.EDGE_TYPE_KEY,
                _keys.BATCH_KEY,
                _keys.EDGE_INDEX_KEY,
                # TA-2: the stable per-graph id must ride the restricted prior
                # substream data so a SEEDED draw can key per-uid generators.
                _keys.SAMPLE_UID_KEY,
            )
            if key in data
        }
        num_graphs = self._num_graphs(prior_data)
        node_noise = self._te_prior_like(
            torch.zeros_like(node_like),
            self.node_sigma,
            data=prior_data,
            label="node",
            reference_scale=False,
            num_graphs=num_graphs,
            generator=generator,
        )
        edge_noise = self._te_prior_like(
            torch.zeros_like(edge_like),
            self.edge_sigma,
            data=prior_data,
            label="edge",
            reference_scale=False,
            num_graphs=num_graphs,
            generator=generator,
        )
        noise_blocks = self.block_codec.rme_to_blocks(
            data, node_noise, edge_noise, project=False
        )
        eps = project_block_state(data, self.idp, noise_blocks)
        # TA-3: runtime belt on the projected_te draw before it is used as D0 --
        # all-finite, and not silently all-zero for a component with a nonzero
        # configured effective scale.
        self._assert_projected_te_draw_finite_and_scaled(
            (
                ("node", eps.node_blocks, self.node_sigma * self.te_prior_sigma),
                ("edge", eps.edge_blocks, self.edge_sigma * self.te_prior_sigma),
            )
        )
        if certify_image:
            node_eps, edge_eps = self.block_codec.blocks_to_rme(
                data,
                eps,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            roundtrip = self.block_codec.rme_to_blocks(
                data, node_eps, edge_eps, project=True
            )
            residual = max(
                self._max_abs(roundtrip.node_blocks - eps.node_blocks),
                self._max_abs(roundtrip.edge_blocks - eps.edge_blocks),
            )
            if residual > self.block_inverse_atol:
                raise ValueError(
                    "Projected TE residual epsilon is outside the certified codec "
                    f"image: max residual={residual:.6g}, "
                    f"atol={self.block_inverse_atol:.6g}."
                )
        return eps

    def _residual_tied_irrep_gaussian_eps(
        self,
        data: AtomicDataDict.Type,
        node_like: torch.Tensor,
        edge_like: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
        certify_image: bool,
    ) -> BlockTensorResult:
        """Draw the multiplicity-tied total-L residual epsilon; NO H0 added."""
        assert self.block_codec is not None
        prior_data = {
            key: data[key]
            for key in (
                AtomicDataDict.ATOM_TYPE_KEY,
                AtomicDataDict.EDGE_TYPE_KEY,
                _keys.BATCH_KEY,
                _keys.EDGE_INDEX_KEY,
                _keys.SAMPLE_UID_KEY,
            )
            if key in data
        }
        num_graphs = self._num_graphs(prior_data)
        node_noise = self._tied_irrep_gaussian_prior_like(
            torch.zeros_like(node_like),
            self.node_sigma,
            data=prior_data,
            label="node",
            num_graphs=num_graphs,
            generator=generator,
        )
        edge_noise = self._tied_irrep_gaussian_prior_like(
            torch.zeros_like(edge_like),
            self.edge_sigma,
            data=prior_data,
            label="edge",
            num_graphs=num_graphs,
            generator=generator,
        )
        noise_blocks = self.block_codec.rme_to_blocks(
            data, node_noise, edge_noise, project=False
        )
        eps = project_block_state(data, self.idp, noise_blocks)
        self._assert_stochastic_prior_draw_finite_and_scaled(
            self.prior,
            (
                (
                    "node",
                    eps.node_blocks,
                    self.node_sigma * self.tied_irrep_sigma,
                ),
                (
                    "edge",
                    eps.edge_blocks,
                    self.edge_sigma * self.tied_irrep_sigma,
                ),
            ),
        )
        if certify_image:
            node_eps, edge_eps = self.block_codec.blocks_to_rme(
                data,
                eps,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            roundtrip = self.block_codec.rme_to_blocks(
                data, node_eps, edge_eps, project=True
            )
            residual = max(
                self._max_abs(roundtrip.node_blocks - eps.node_blocks),
                self._max_abs(roundtrip.edge_blocks - eps.edge_blocks),
            )
            if residual > self.block_inverse_atol:
                raise ValueError(
                    "tied_irrep_gaussian residual epsilon is outside the "
                    "certified codec image: max residual="
                    f"{residual:.6g}, atol={self.block_inverse_atol:.6g}."
                )
        return eps

    def _residual_stochastic_eps(
        self,
        data: AtomicDataDict.Type,
        node_like: torch.Tensor,
        edge_like: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
        certify_image: bool,
    ) -> BlockTensorResult:
        if self.prior in self._projected_te_prior_names:
            return self._residual_te_eps(
                data,
                node_like,
                edge_like,
                generator=generator,
                certify_image=certify_image,
            )
        if self.prior in self._tied_irrep_gaussian_prior_names:
            return self._residual_tied_irrep_gaussian_eps(
                data,
                node_like,
                edge_like,
                generator=generator,
                certify_image=certify_image,
            )
        raise RuntimeError(f"Unsupported residual stochastic prior {self.prior!r}.")

    def _strict_image_certification_due(self) -> bool:
        """Schedule only the pure repack/residual image-space self-check."""

        batch = self._strict_certification_batches
        if self._strict_certification_mode == "always":
            return True
        if self._strict_certification_mode == "first_batch":
            return batch == 0
        return batch % self._strict_certification_period == 0

    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        t: Optional[torch.Tensor] = None,
        prior_seed: Optional[int] = None,
        time_seed: Optional[int] = None,
    ) -> Tuple[AtomicDataDict.Type, AtomicDataDict.Type, CFMContext]:
        """Return a model-input dict with interpolated H_t written to H0 keys."""
        if not self.enabled:
            raise RuntimeError("HamiltonianCFM.prepare_batch called while disabled")

        data = data.copy()
        ref_data = ref_data.copy()
        if self.block_ode:
            # The model receives ``data`` while loss-time physical projection
            # uses ``ref_data``.  Break shallow tensor aliases up front so an
            # in-place model mutation cannot rewrite the reference topology.
            data_topology = self._snapshot_block_topology(data)
            ref_topology = self._snapshot_block_topology(ref_data)
            self._require_matching_block_topology(data_topology, ref_topology)
            self._restore_block_topology(data, data_topology)
            self._restore_block_topology(ref_data, ref_topology)

        node_target = ref_data.get(self.node_target_key, None)
        edge_target = ref_data.get(self.edge_target_key, None)
        if self.block_ode:
            if self.uureal_block_ode:
                self._require_uureal_block_contract(data)
                self._require_uureal_block_contract(ref_data)
            missing_endpoint_fields = self._missing_keys(
                ref_data,
                (
                    self.node_block_target_key,
                    self.edge_block_target_key,
                    self.node_block_shape_key,
                    self.edge_block_shape_key,
                ),
            )
            if missing_endpoint_fields:
                raise KeyError(
                    "Block-space ODE requires explicit endpoint blocks and shapes; "
                    f"missing keys={missing_endpoint_fields}."
                )
            if not self.uureal_block_ode:
                missing_h0_fields = self._missing_keys(
                    data,
                    (
                        self.node_h0_block_key,
                        self.edge_h0_block_key,
                        self.node_h0_block_shape_key,
                        self.edge_h0_block_shape_key,
                    ),
                )
                if missing_h0_fields:
                    raise KeyError(
                        "Block-space ODE requires physical H0 blocks and shapes; "
                        f"missing keys={missing_h0_fields}."
                    )
            like = torch.as_tensor(ref_data[self.node_block_target_key])
            device = like.device
            dtype = self.dtype
            node_count = int(torch.as_tensor(ref_data[self.node_block_target_key]).shape[0])
            edge_count = int(torch.as_tensor(ref_data[self.edge_block_target_key]).shape[0])
        else:
            if node_target is None and edge_target is None:
                raise KeyError(
                    "CFM requires node and/or edge Hamiltonian targets in ref_data; "
                    f"looked for `{self.node_target_key}` and `{self.edge_target_key}`."
                )
            like = node_target if node_target is not None else edge_target
            device = like.device
            dtype = like.dtype if torch.is_floating_point(like) else self.dtype
            node_count = None if node_target is None else node_target.shape[0]
            edge_count = None if edge_target is None else edge_target.shape[0]
        num_graphs = self._num_graphs(data)
        if t is None:
            time_generator = self._seeded_generator(device, time_seed)
            sample_kwargs = {
                "num_graphs": num_graphs,
                "device": device,
                "dtype": dtype,
            }
            if time_generator is not None:
                sample_kwargs["generator"] = time_generator
            t = self._sample_t(**sample_kwargs)
        else:
            t = self._normalize_t(t, num_graphs=num_graphs, device=device, dtype=dtype)
        node_t, edge_t = self._expand_graph_times(
            data,
            t,
            node_count=node_count,
            edge_count=edge_count,
        )

        if self.uureal_block_ode:
            endpoint, endpoint_projection_residual = self._block_state_from_fields(
                ref_data,
                ref_data,
                node_key=self.node_block_target_key,
                edge_key=self.edge_block_target_key,
                node_shape_key=self.node_block_shape_key,
                edge_shape_key=self.edge_block_shape_key,
                label="residual_dh endpoint",
                device=device,
                dtype=dtype,
            )
            node_alpha = node_t.reshape(
                (-1,) + (1,) * (endpoint.node_blocks.ndim - 1)
            )
            edge_alpha = edge_t.reshape(
                (-1,) + (1,) * (endpoint.edge_blocks.ndim - 1)
            )
            current = project_block_state(
                data,
                self.idp,
                BlockTensorResult(
                    node_alpha * endpoint.node_blocks,
                    edge_alpha * endpoint.edge_blocks,
                    endpoint.node_shapes,
                    endpoint.edge_shapes,
                ),
            )
            if self.detach_interpolated_h0:
                current = BlockTensorResult(
                    current.node_blocks.detach(),
                    current.edge_blocks.detach(),
                    current.node_shapes,
                    current.edge_shapes,
                )
            self._attach_uureal_residual_state(data, current)
            data[self.flow_time_key] = t.detach()
            ref_data[self.flow_time_key] = t.detach()
            ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
                endpoint_projection_residual, device=device, dtype=dtype
            )
            self._drop_block_authority_fields(data)
            return data, ref_data, CFMContext(
                t=t,
                node_t=node_t,
                edge_t=edge_t,
                node_base=torch.as_tensor(data[self.node_h0_key]),
                edge_base=torch.as_tensor(data[self.edge_h0_key]),
                node_target=None,
                edge_target=None,
                node_current=current.node_blocks,
                edge_current=current.edge_blocks,
                node_prior=torch.zeros_like(current.node_blocks),
                edge_prior=torch.zeros_like(current.edge_blocks),
                block_target_semantics="residual_dh",
            )

        if self.residual_ao_block_ode:
            assert self.block_codec is not None
            self._require_spatial_residual_block_contract(data)
            self._require_spatial_residual_block_contract(ref_data)
            certify_image = self._strict_image_certification_due()
            h0_blocks, h0_projection_residual = self._physical_h0_blocks(
                data, device=device, dtype=dtype
            )
            endpoint, endpoint_projection_residual = self._block_state_from_fields(
                ref_data,
                ref_data,
                node_key=self.node_block_target_key,
                edge_key=self.edge_block_target_key,
                node_shape_key=self.node_block_shape_key,
                edge_shape_key=self.edge_block_shape_key,
                label="residual_dh endpoint",
                device=device,
                dtype=dtype,
            )
            # Physical H0 RME is the CONSTANT conditioning channel: written to the
            # H0 keys exactly once and never interpolated.  The pure residual state
            # D travels in the spatial residual block keys instead.  This is the
            # deliberate divergence from the generic ao_block_ode branch below,
            # whose H0 keys carry the interpolated state RME.
            node_base, edge_base = self.block_codec.blocks_to_rme(
                data,
                h0_blocks,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            data[self.node_h0_key] = node_base
            data[self.edge_h0_key] = edge_base
            node_alpha = node_t.reshape((-1,) + (1,) * (endpoint.node_blocks.ndim - 1))
            edge_alpha = edge_t.reshape((-1,) + (1,) * (endpoint.edge_blocks.ndim - 1))
            if self.prior == "zero":
                # Zero prior: D_t = t * D1 exactly (no (1 - t) * prior term).  Legacy
                # feature-key overwrites are intentionally skipped for B: the H0 keys
                # already expose the physical conditioning channel honestly.
                D_t = project_block_state(
                    data,
                    self.idp,
                    BlockTensorResult(
                        node_alpha * endpoint.node_blocks,
                        edge_alpha * endpoint.edge_blocks,
                        endpoint.node_shapes,
                        endpoint.edge_shapes,
                    ),
                )
                node_prior_blocks = None
                edge_prior_blocks = None
            else:
                # Stochastic bridge: draw eps in the certified codec image
                # (NO H0 added), then D_t = project((1 - t) * eps + t * D1).
                # The (1 - t) * eps term removes the deterministic
                # D_t = t * D1 shortcut left by the zero prior.  prior_seed
                # selects the deterministic validation stream; training keeps
                # fresh rowwise noise.
                eps = self._residual_stochastic_eps(
                    data,
                    node_base,
                    edge_base,
                    generator=self._seeded_generator(device, prior_seed),
                    certify_image=certify_image,
                )
                D_t = project_block_state(
                    data,
                    self.idp,
                    BlockTensorResult(
                        (1.0 - node_alpha) * eps.node_blocks
                        + node_alpha * endpoint.node_blocks,
                        (1.0 - edge_alpha) * eps.edge_blocks
                        + edge_alpha * endpoint.edge_blocks,
                        endpoint.node_shapes,
                        endpoint.edge_shapes,
                    ),
                )
                node_prior_blocks = eps.node_blocks
                edge_prior_blocks = eps.edge_blocks
            if self.detach_interpolated_h0:
                D_t = BlockTensorResult(
                    D_t.node_blocks.detach(),
                    D_t.edge_blocks.detach(),
                    D_t.node_shapes,
                    D_t.edge_shapes,
                )
            self._attach_spatial_residual_state(data, D_t)
            data[self.flow_time_key] = t.detach()
            ref_data[self.flow_time_key] = t.detach()
            ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
                endpoint_projection_residual, device=device, dtype=dtype
            )
            ref_data["_block_ode_h0_projection_residual"] = torch.as_tensor(
                h0_projection_residual, device=device, dtype=dtype
            )
            self._drop_block_authority_fields(data)
            self._strict_certification_batches += 1
            return data, ref_data, CFMContext(
                t=t,
                node_t=node_t,
                edge_t=edge_t,
                node_base=node_base,
                edge_base=edge_base,
                node_target=None,
                edge_target=None,
                node_current=D_t.node_blocks,
                edge_current=D_t.edge_blocks,
                # Zero prior keeps the exact zeros_like telemetry; projected_te
                # exposes the drawn epsilon.  CFMContext.node_prior has no runtime
                # consumer for this mode (the block-ODE loss reads node_current /
                # node_target only), so this is telemetry-only either way.
                node_prior=(
                    torch.zeros_like(D_t.node_blocks)
                    if node_prior_blocks is None
                    else node_prior_blocks
                ),
                edge_prior=(
                    torch.zeros_like(D_t.edge_blocks)
                    if edge_prior_blocks is None
                    else edge_prior_blocks
                ),
                block_target_semantics="residual_dh",
            )

        if self.block_ode:
            assert self.block_codec is not None
            certify_image = self._strict_image_certification_due()
            h0_blocks, h0_projection_residual = self._physical_h0_blocks(
                data, device=device, dtype=dtype
            )
            endpoint, endpoint_projection_residual = self._block_state_from_fields(
                ref_data,
                ref_data,
                node_key=self.node_block_target_key,
                edge_key=self.edge_block_target_key,
                node_shape_key=self.node_block_shape_key,
                edge_shape_key=self.edge_block_shape_key,
                label=f"{self.target_semantics} endpoint",
                device=device,
                dtype=dtype,
            )
            # Coupled RME is generated only from certified blocks.  Legacy main
            # and H0 feature side channels are intentionally ignored because
            # older LMDBs store AO-product gathers under the same key names.
            node_base, edge_base = self.block_codec.blocks_to_rme(
                data,
                h0_blocks,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            node_target, edge_target = self.block_codec.blocks_to_rme(
                data,
                endpoint,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            block_start, node_prior, edge_prior = self._block_initial_state(
                data,
                h0_blocks,
                prior_seed=prior_seed,
                certify_image=certify_image,
            )

            full_endpoint = self.block_codec.endpoint_to_full(endpoint, h0_blocks)
            full_endpoint = project_block_state(data, self.idp, full_endpoint)
            node_alpha = node_t.reshape((-1,) + (1,) * (full_endpoint.node_blocks.ndim - 1))
            edge_alpha = edge_t.reshape((-1,) + (1,) * (full_endpoint.edge_blocks.ndim - 1))
            block_current = project_block_state(
                data,
                self.idp,
                BlockTensorResult(
                    node_blocks=(1.0 - node_alpha) * block_start.node_blocks
                    + node_alpha * full_endpoint.node_blocks,
                    edge_blocks=(1.0 - edge_alpha) * block_start.edge_blocks
                    + edge_alpha * full_endpoint.edge_blocks,
                    node_shapes=h0_blocks.node_shapes,
                    edge_shapes=h0_blocks.edge_shapes,
                ),
            )
            node_current, edge_current = self.block_codec.blocks_to_rme(
                data,
                block_current,
                certify_image=certify_image,
                _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
            )
            if self.detach_interpolated_h0:
                node_current = node_current.detach()
                edge_current = edge_current.detach()
            data[self.node_h0_key] = node_current
            data[self.edge_h0_key] = edge_current
            if self.overwrite_feature_keys:
                data[self.node_target_key] = node_current
                data[self.edge_target_key] = edge_current
            data[self.flow_time_key] = t.detach()
            ref_data[self.flow_time_key] = t.detach()
            ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
                endpoint_projection_residual, device=device, dtype=dtype
            )
            ref_data["_block_ode_h0_projection_residual"] = torch.as_tensor(
                h0_projection_residual, device=device, dtype=dtype
            )
            # Endpoint/H0 block side channels are flow authority, not model
            # inputs.  Removing them also breaks the shallow aliases created by
            # Trainer's input/reference copies, so an in-place model cannot
            # rewrite the labels used below by ``loss``.
            self._drop_block_authority_fields(data)
            # Advance only after all unconditional contract gates and any due
            # certification pass; a failed batch remains due on retry.
            self._strict_certification_batches += 1
            return data, ref_data, CFMContext(
                t=t,
                node_t=node_t,
                edge_t=edge_t,
                node_base=node_base,
                edge_base=edge_base,
                node_target=node_target,
                edge_target=edge_target,
                node_current=node_current,
                edge_current=edge_current,
                node_prior=node_prior,
                edge_prior=edge_prior,
                block_target_semantics=self.target_semantics,
            )

        node_base = edge_base = node_current = edge_current = None
        node_prior = edge_prior = None
        # One shared Haar-DM candidate for node+edge; None for every other prior.
        haar_candidate_idx = self._resolve_haar_candidate_idx(
            data, node_like=node_target, edge_like=edge_target
        )

        if node_target is not None:
            node_target = node_target.to(device=device, dtype=dtype)
            node_base = self._base_like(data, node_target, self.node_h0_key, "node")
            node_res = node_target - node_base
            node_prior = self._prior_like(
                node_res,
                self.node_sigma,
                data=data,
                label="node",
                base=node_base,
                num_graphs=num_graphs,
                candidate_idx=haar_candidate_idx,
            )
            node_t_view = node_t.reshape((-1,) + (1,) * (node_target.ndim - 1))
            node_current = node_base + (1.0 - node_t_view) * node_prior + node_t_view * node_res
            if self.detach_interpolated_h0:
                node_current = node_current.detach()
            data[self.node_h0_key] = node_current
            if self.overwrite_feature_keys:
                data[self.node_target_key] = node_current

        if edge_target is not None:
            edge_target = edge_target.to(device=device, dtype=dtype)
            edge_base = self._base_like(data, edge_target, self.edge_h0_key, "edge")
            edge_res = edge_target - edge_base
            edge_prior = self._prior_like(
                edge_res,
                self.edge_sigma,
                data=data,
                label="edge",
                base=edge_base,
                num_graphs=num_graphs,
                candidate_idx=haar_candidate_idx,
            )
            edge_t_view = edge_t.reshape((-1,) + (1,) * (edge_target.ndim - 1))
            edge_current = edge_base + (1.0 - edge_t_view) * edge_prior + edge_t_view * edge_res
            if self.detach_interpolated_h0:
                edge_current = edge_current.detach()
            data[self.edge_h0_key] = edge_current
            if self.overwrite_feature_keys:
                data[self.edge_target_key] = edge_current

        data[self.flow_time_key] = t.detach()
        ref_data[self.flow_time_key] = t.detach()

        return data, ref_data, CFMContext(
            t=t,
            node_t=node_t,
            edge_t=edge_t,
            node_base=node_base,
            edge_base=edge_base,
            node_target=node_target,
            edge_target=edge_target,
            node_current=node_current,
            edge_current=edge_current,
            node_prior=node_prior,
            edge_prior=edge_prior,
            block_target_semantics=self.target_semantics if self.block_ode else None,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _node_mask(self, data: AtomicDataDict.Type, pred: torch.Tensor) -> torch.Tensor:
        if self.idp is None or AtomicDataDict.ATOM_TYPE_KEY not in data:
            return torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        mask = self.idp.mask_to_nrme.to(device=atom_types.device)[atom_types]
        if "expert_node_mask" in data:
            mask = mask & data["expert_node_mask"].to(device=mask.device).unsqueeze(-1)
        return normalize_idp_mask_layout(self.idp, mask, pred, label="node idp mask")

    def _edge_mask(self, data: AtomicDataDict.Type, pred: torch.Tensor) -> torch.Tensor:
        if self.idp is None or AtomicDataDict.EDGE_TYPE_KEY not in data:
            return torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        edge_types = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
        mask = self.idp.mask_to_erme.to(device=edge_types.device)[edge_types]
        if "expert_edge_mask" in data:
            mask = mask & data["expert_edge_mask"].to(device=mask.device).unsqueeze(-1)
        return normalize_idp_mask_layout(self.idp, mask, pred, label="edge idp mask")

    def _project_loss_layout(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, _raw_mask = project_uureal_to_like(self.idp, pred, target)
        if pred.shape != target.shape:
            raise ValueError(
                "prediction layout does not match target layout; "
                "check nextham_uureal_mask/mask_uureal propagation."
            )
        mask = normalize_idp_mask_layout(self.idp, mask, pred, label="loss idp mask")
        return pred, mask

    @staticmethod
    def _active_block_mask(
        blocks: torch.Tensor,
        shapes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if blocks.ndim != 3:
            raise ValueError(
                "AO-block flow expects [n_item, max_norb, max_norb] tensors; "
                f"got shape={tuple(blocks.shape)}."
            )
        if shapes is None:
            return torch.ones_like(blocks, dtype=torch.bool)
        shapes = torch.as_tensor(shapes, device=blocks.device, dtype=torch.long)
        if shapes.ndim != 2 or shapes.shape != (blocks.shape[0], 2):
            raise ValueError(
                "AO-block shape metadata must be [n_item, 2]; "
                f"got shape={tuple(shapes.shape)} for blocks={tuple(blocks.shape)}."
            )
        rows = torch.arange(blocks.shape[-2], device=blocks.device)
        cols = torch.arange(blocks.shape[-1], device=blocks.device)
        return (rows[None, :, None] < shapes[:, 0, None, None]) & (
            cols[None, None, :] < shapes[:, 1, None, None]
        )

    def _block_endpoint_loss(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One-step endpoint loss for an H-B0 direct AO-block decoder.

        Interpolation and time conditioning remain in RME/H0 feature space, while
        the endpoint is supervised in AO-block space.  This is deliberately
        restricted to one-step validation: it is a cross-space endpoint adapter,
        not a same-coordinate multi-step ODE.
        """
        state: Dict[str, torch.Tensor] = {
            "train_flow_t": ctx.t.detach().mean(),
            "train_flow_weight": self._time_weight(ctx.t).detach().mean(),
        }
        component_stats = []

        for label, pred_key, target_key, shape_key, weight, item_t in (
            (
                "onsite",
                self.node_output_key,
                self.node_block_target_key,
                self.node_block_shape_key,
                self.node_weight,
                ctx.node_t,
            ),
            (
                "hopping",
                self.edge_output_key,
                self.edge_block_target_key,
                self.edge_block_shape_key,
                self.edge_weight,
                ctx.edge_t,
            ),
        ):
            pred = pred_data.get(pred_key, None)
            target = ref_data.get(target_key, None)
            if pred is None or target is None:
                continue
            target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
            if pred.shape != target.shape:
                raise ValueError(
                    f"AO-block flow {label} prediction/target mismatch: "
                    f"{pred_key}={tuple(pred.shape)} vs {target_key}={tuple(target.shape)}."
                )
            mask = self._active_block_mask(pred, ref_data.get(shape_key, None))
            diff = pred - target
            item_weights = None if item_t is None else self._time_weight(item_t)
            stats = self._legacy_metric_stats(
                diff, mask, self.loss_type, item_weights
            )
            component = stats[0]
            component_stats.append((stats, weight))
            state[f"train_flow_{label}_loss"] = component.detach()
            state[f"train_{label}_loss"] = component.detach()

        if not component_stats:
            raise KeyError(
                "AO-block CFM could not compute a loss because outputs/targets do "
                f"not contain {self.node_output_key}/{self.edge_output_key} and "
                f"{self.node_block_target_key}/{self.edge_block_target_key}."
            )
        total = self._reduce_legacy_component_stats(tuple(component_stats))
        return self._finalize_loss(total, state, pred_data)

    def _block_ode_endpoint_loss(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
        *,
        prediction_is_full_h: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fail-closed endpoint loss on independent physical block freedoms."""
        if ctx.block_target_semantics != self.target_semantics:
            raise ValueError(
                "Block-space ODE target semantics changed between prepare_batch and loss."
            )
        missing = [
            key
            for key in (
                self.node_output_key,
                self.edge_output_key,
                self.node_block_target_key,
                self.edge_block_target_key,
                self.node_block_shape_key,
                self.edge_block_shape_key,
            )
            if key not in (pred_data if key in {self.node_output_key, self.edge_output_key} else ref_data)
        ]
        if missing:
            raise KeyError(f"Block-space ODE endpoint loss is missing required keys={missing}.")

        node_pred = pred_data[self.node_output_key]
        edge_pred = pred_data[self.edge_output_key]
        node_target = torch.as_tensor(
            ref_data[self.node_block_target_key], device=node_pred.device, dtype=node_pred.dtype
        )
        edge_target = torch.as_tensor(
            ref_data[self.edge_block_target_key], device=edge_pred.device, dtype=edge_pred.dtype
        )
        if node_pred.shape != node_target.shape or edge_pred.shape != edge_target.shape:
            raise ValueError(
                "Block-space ODE prediction/target canvas mismatch: "
                f"node {tuple(node_pred.shape)} vs {tuple(node_target.shape)}, "
                f"edge {tuple(edge_pred.shape)} vs {tuple(edge_target.shape)}."
            )
        node_shapes = torch.as_tensor(ref_data[self.node_block_shape_key], device=node_pred.device)
        edge_shapes = torch.as_tensor(ref_data[self.edge_block_shape_key], device=edge_pred.device)
        pred_state = BlockTensorResult(node_pred, edge_pred, node_shapes, edge_shapes)
        target_state = BlockTensorResult(node_target, edge_target, node_shapes, edge_shapes)

        # Model outputs are predictions, not graph authority.  Pairing and
        # species shapes must come from the independently preserved reference
        # topology or a returned/stale PBC shift can redefine Hermitian mates.
        target_projected = project_block_state(ref_data, self.idp, target_state)
        target_residual = max(
            self._max_abs(target_projected.node_blocks - node_target),
            self._max_abs(target_projected.edge_blocks - edge_target),
        )
        if target_residual > self.block_inverse_atol:
            raise ValueError(
                "Block-space ODE target violates onsite/reverse/padding constraints: "
                f"max residual={target_residual:.6g}, atol={self.block_inverse_atol}."
            )

        if prediction_is_full_h and self.target_semantics == "residual_dh":
            if ctx.node_base is None or ctx.edge_base is None:
                raise ValueError("Residual block sample scoring requires both H0 RME components.")
            h0 = self.block_codec.rme_to_blocks(
                ref_data, ctx.node_base, ctx.edge_base, project=True
            )
            target_state = self.block_codec.endpoint_to_full(target_projected, h0)
            target_projected = project_block_state(ref_data, self.idp, target_state)

        pred_projected = project_block_state(ref_data, self.idp, pred_state)
        node_diff = pred_projected.node_blocks - target_projected.node_blocks
        edge_diff = pred_projected.edge_blocks - target_projected.edge_blocks

        node_valid = block_mask_from_shapes(
            pred_projected.node_shapes, tuple(node_diff.shape[-2:])
        )
        upper = torch.triu(
            torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool, device=node_diff.device)
        )
        node_mask = node_valid & upper.unsqueeze(0)

        edge_valid = block_mask_from_shapes(
            pred_projected.edge_shapes, tuple(edge_diff.shape[-2:])
        )
        rev = strict_reverse_edge_index(
            ref_data, device=edge_diff.device, idp=self.idp
        )
        rows = torch.arange(edge_diff.shape[0], device=edge_diff.device)
        canonical_rows = rows <= rev
        edge_mask = edge_valid & canonical_rows.view(-1, 1, 1)
        self_reverse = rows == rev
        if bool(self_reverse.any().item()):
            edge_mask[self_reverse] &= upper.unsqueeze(0)

        node_stats = self._metric_stats(
            node_diff, node_mask, self.loss_type, self._time_weight(ctx.node_t)
        )
        edge_stats = self._metric_stats(
            edge_diff, edge_mask, self.loss_type, self._time_weight(ctx.edge_t)
        )
        node_component = node_stats[0]
        edge_component = edge_stats[0]
        total = self._reduce_component_stats(
            ((node_stats, self.node_weight), (edge_stats, self.edge_weight))
        )

        # NOTE: the flow objective publishes its endpoint node/edge metric ONLY under
        # the flow namespace (train_flow_onsite_loss/train_flow_hopping_loss).  It does
        # NOT alias it into the bare train_onsite_loss/train_hopping_loss: those legacy
        # tags carry feature-compatible semantics and are written solely by the
        # compatible pass (Trainer._compatible_loss_state*, legacy_prefix), so the two
        # namespaces never mix under one tag.  (Previously the flow value was aliased
        # here; the compatible pass then overwrote it every step, but if that pass were
        # ever throttled/absent the flow value would leak under the compatible tag.)
        #
        # P1-1 (block endpoint population parity): `_compatible_clean_stats` below is
        # handed to Trainer._compatible_loss_state_from_flow_stats, which reduces it
        # through the *criterion's own* `compatible_loss_from_stats`
        # (HamilBlockwiseNexTHamLoss).  That criterion's `block_components` counts
        # EVERY shape-active directed AO entry -- no onsite lower-triangle drop, no
        # reverse-edge dedup (see blockwise_tensor.block_components /
        # block_mask_from_shapes).  `node_mask`/`edge_mask` above intentionally keep
        # only ONE independent physical freedom per Hermitian pair (onsite upper
        # triangle, one side of each reverse-edge pair) for the *training* reduction
        # -- a DIFFERENT population from the criterion's directed-full count.
        # Publishing the canonical sums under a "block" label the criterion treats as
        # directed-full silently understated L1/RMSE (population off by ~2x on
        # non-symmetric error).  Feed `_compatible_clean_stats` the directed-full
        # `node_valid`/`edge_valid` masks instead -- the same population
        # `train_compatible_directed_*` below already uses -- so the label and the
        # population it carries finally agree with the criterion's own definition.
        # This is an exact recount, not an approximation: Hermitian pairing is
        # bit-exact post-projection (project_block_state symmetrizes onsite as
        # 0.5*(X+X.T) and derives every reverse edge as the literal transpose of a
        # shared `averaged` tensor via index_copy_), so summing node_diff/edge_diff
        # over the full directed mask reproduces exactly what an analytic
        # canonical-sum doubling would give (diagonal/self-reverse counted once,
        # off-diagonal/reverse-pair counted twice) -- see
        # test_block_ode_compatible_stats_match_criterion_directed_population for the
        # locked numerical identity.
        state: Dict[str, torch.Tensor] = {
            "train_flow_t": ctx.t.detach().mean(),
            "train_flow_weight": self._time_weight(ctx.t).detach().mean(),
            "train_flow_onsite_loss": node_component.detach(),
            "train_flow_hopping_loss": edge_component.detach(),
            "block_ode_target_projection_residual": torch.as_tensor(
                target_residual, device=total.device, dtype=total.dtype
            ),
            "_compatible_clean_stats": {
                **self._compatible_clean_stats(node_diff, node_valid, "onsite"),
                **self._compatible_clean_stats(edge_diff, edge_valid, "hopping"),
            },
        }
        if self.uureal_block_ode:
            # Historical-comparison metric (review P1-5).  Two deliberately
            # coexisting reductions:
            #   * canonical (`train_flow_*_loss` above): each independent physical
            #     freedom counted ONCE -- onsite upper triangle plus one
            #     canonical edge per Hermitian (i,j,R)/(j,i,-R) pair;
            #   * compatible_directed (`train_compatible_directed_*`): every
            #     stored directed coordinate counted, i.e. all mapper-valid
            #     onsite entries and ALL directed edges -- the same population
            #     the historical SOC uu-real RME losses (H-B0/H-A1 runs)
            #     averaged over, and (as of P1-1) the same population now backing
            #     the `_compatible_clean_stats` payload above.  Counts differ
            #     (~2x from canonical) and values differ whenever the error is not
            #     Hermitian-symmetric, so curves plotted against historical runs
            #     must use the directed keys.
            directed_node_stats = self._metric_stats(
                node_diff, node_valid, self.loss_type, self._time_weight(ctx.node_t)
            )
            directed_edge_stats = self._metric_stats(
                edge_diff, edge_valid, self.loss_type, self._time_weight(ctx.edge_t)
            )
            directed_total = self._reduce_component_stats(
                (
                    (directed_node_stats, self.node_weight),
                    (directed_edge_stats, self.edge_weight),
                )
            )
            state["train_compatible_directed_onsite_loss"] = directed_node_stats[0].detach()
            state["train_compatible_directed_hopping_loss"] = directed_edge_stats[0].detach()
            state["train_compatible_directed_loss"] = directed_total.detach()
        return self._finalize_loss(total, state, pred_data)

    @staticmethod
    def _safe_exact_rmse(
        square_sum: torch.Tensor,
        safe_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return a finite-gradient RMSE that is exactly zero at zero error."""
        mean_square = square_sum / safe_count
        positive = mean_square > 0
        sqrt_input = torch.where(
            positive,
            mean_square,
            torch.ones_like(mean_square),
        )
        return torch.where(positive, torch.sqrt(sqrt_input), mean_square)

    @staticmethod
    def _metric_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str,
        weights: Optional[torch.Tensor] = None,
    ) -> _MetricStats:
        mask_f, weights_f, count = HamiltonianCFM._metric_inputs(
            diff, mask, weights
        )
        if loss_type == "mse":
            square_sum = (diff.square() * mask_f * weights_f).sum()
            metric = torch.where(
                count > 0,
                square_sum / count.clamp_min(1.0),
                square_sum * 0.0,
            )
            return metric, square_sum, square_sum, count
        abs_sum = (diff.abs() * mask_f * weights_f).sum()
        sq_sum = (diff.square() * mask_f * weights_f).sum()
        safe_count = count.clamp_min(1.0)
        metric = 0.5 * (
            abs_sum / safe_count
            + HamiltonianCFM._safe_exact_rmse(sq_sum, safe_count)
        )
        metric = torch.where(count > 0, metric, abs_sum * 0.0)
        return metric, abs_sum, sq_sum, count

    @staticmethod
    def _metric_inputs(
        diff: torch.Tensor,
        mask: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the shared mask, weight, and raw-count metric inputs."""
        mask_f = mask.to(device=diff.device, dtype=diff.dtype)
        count = mask_f.sum()
        if weights is None:
            weights_f = torch.ones_like(diff)
        else:
            weights_f = weights.to(device=diff.device, dtype=diff.dtype)
            weights_f = weights_f.reshape((-1,) + (1,) * (diff.ndim - 1))
            weights_f = weights_f.expand_as(diff)
        return mask_f, weights_f, count

    @staticmethod
    def _legacy_metric_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str,
        weights: Optional[torch.Tensor] = None,
    ) -> _MetricStats:
        """Frozen pre-block-ODE component metric from parent ``e7e5410``."""
        mask_f, weights_f, count = HamiltonianCFM._metric_inputs(
            diff, mask, weights
        )
        count = count.clamp_min(1.0)
        if loss_type == "mse":
            numerator = (diff.square() * mask_f * weights_f).sum()
            metric = numerator / count
        else:
            abs_sum = (diff.abs() * mask_f * weights_f).sum()
            square_sum = (diff.square() * mask_f * weights_f).sum()
            metric = 0.5 * (
                abs_sum / count + torch.sqrt(square_sum / count + 1.0e-12)
            )
            numerator = metric * count
        # Match the shared stats tuple shape. The legacy reducer consumes only
        # component metric, numerator, and clamped component count.
        return metric, numerator, numerator, count

    @staticmethod
    def _global_metric(
        primary_sum: torch.Tensor,
        square_sum: torch.Tensor,
        count: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        safe_count = count.clamp_min(1.0)
        if loss_type == "mse":
            metric = primary_sum / safe_count
        else:
            metric = 0.5 * (
                primary_sum / safe_count
                + HamiltonianCFM._safe_exact_rmse(square_sum, safe_count)
            )
        return torch.where(count > 0, metric, primary_sum * 0.0)

    def _reduce_component_stats(
        self,
        components: _WeightedMetricStats,
    ) -> torch.Tensor:
        """Apply the configured node/edge reduction to raw metric statistics.

        ``global_elements`` is a single reduction over all valid elements and
        therefore accepts unit component weights only. ``equal_components``
        keeps the historical sum of independently reduced component metrics,
        where node/edge weights are explicit outer loss multipliers. Empty
        components carry zero statistics and contribute zero in either mode.
        """
        if not components:
            raise ValueError("At least one loss component is required.")
        if self.component_reduction == "equal_components":
            return self._weighted_component_metric_sum(components)

        primary_sum = square_sum = count = None
        for stats, _weight in components:
            primary_sum = stats[1] if primary_sum is None else primary_sum + stats[1]
            square_sum = stats[2] if square_sum is None else square_sum + stats[2]
            count = stats[3] if count is None else count + stats[3]
        return self._global_metric(primary_sum, square_sum, count, self.loss_type)

    @staticmethod
    def _weighted_component_metric_sum(
        components: _WeightedMetricStats,
    ) -> torch.Tensor:
        total = None
        for stats, weight in components:
            weighted = float(weight) * stats[0]
            total = weighted if total is None else total + weighted
        return total

    def _reduce_legacy_component_stats(
        self,
        components: _WeightedMetricStats,
    ) -> torch.Tensor:
        """Reproduce the parent component-first reduction for legacy routes."""
        if not components:
            raise ValueError("At least one loss component is required.")
        if self.component_reduction == "equal_components":
            return self._weighted_component_metric_sum(components)

        numerator = count = None
        for stats, weight in components:
            weighted_numerator = float(weight) * stats[1]
            weighted_count = float(weight) * stats[3]
            numerator = (
                weighted_numerator
                if numerator is None
                else numerator + weighted_numerator
            )
            count = weighted_count if count is None else count + weighted_count
        return numerator / count.clamp_min(1.0)

    def _finalize_loss(
        self,
        total: torch.Tensor,
        state: Dict[str, torch.Tensor],
        pred_data: AtomicDataDict.Type,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Attach shared router diagnostics and publish the final train loss."""
        mean_max_prob = pred_data.get("mean_max_prob", None)
        if torch.is_tensor(mean_max_prob):
            state["mean_max_prob"] = mean_max_prob.detach()
            if self.router_z_loss_coef > 0.0:
                total = total + self.router_z_loss_coef * mean_max_prob
        expert_load_cv = pred_data.get("expert_load_cv", None)
        if torch.is_tensor(expert_load_cv):
            state["expert_load_cv"] = expert_load_cv.detach()
        state["train_flow_loss"] = total.detach()
        self.last_state = state
        return total, state

    @staticmethod
    def _compatible_clean_stats(
        diff: torch.Tensor,
        mask: torch.Tensor,
        component: str,
        *,
        metric_space: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect non-CFM HamilLossAbs reductions from an already aligned diff."""
        with torch.no_grad():
            diff = diff.detach()
            mask_f = mask.to(device=diff.device, dtype=diff.dtype)
            abs_sum = (diff.abs() * mask_f).sum()
            square_sum = (diff.square() * mask_f).sum()
            count = mask_f.sum()
        return {
            f"{component}_l1_sum": abs_sum.detach(),
            f"{component}_mse_sum": square_sum.detach(),
            f"{component}_count": count.detach(),
            "metric_space": metric_space
            or ("block" if diff.ndim >= 3 else "rme"),
        }

    @staticmethod
    def _target_metric_space(target_key: str, tensor: torch.Tensor) -> str:
        if "block" in str(target_key).lower():
            return "block"
        return "block" if tensor.ndim >= 3 else "rme"

    @staticmethod
    def _merge_compatible_clean_stats(
        state: Dict[str, Any], stats: Dict[str, Any]
    ) -> None:
        """Merge endpoint sums only when node/edge share one representation."""

        target = state.setdefault("_compatible_clean_stats", {})
        existing_space = target.get("metric_space")
        incoming_space = stats.get("metric_space")
        if (
            existing_space is not None
            and incoming_space is not None
            and str(existing_space) != str(incoming_space)
        ):
            raise ValueError(
                "Flow endpoint metric-space mismatch between onsite and hopping "
                f"targets: {existing_space!r} != {incoming_space!r}. Align "
                "node_target_key and edge_target_key to one block or RME route."
            )
        target.update(stats)

    def _time_weight(self, t: torch.Tensor) -> torch.Tensor:
        if self.endpoint_weight_power == 0.0:
            return torch.ones_like(t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        w = denom.pow(-self.endpoint_weight_power)
        if self.endpoint_weight_cap > 0:
            w = w.clamp_max(self.endpoint_weight_cap)
        return w

    def loss(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            raise RuntimeError("HamiltonianCFM.loss called while disabled")

        if self.output_space == "ao_block":
            return self._block_endpoint_loss(pred_data, ref_data, ctx)
        if self.block_ode:
            return self._block_ode_endpoint_loss(pred_data, ref_data, ctx)

        t_weight = self._time_weight(ctx.t).to(device=ctx.t.device, dtype=ctx.t.dtype)
        component_stats = []
        state: Dict[str, torch.Tensor] = {
            "train_flow_t": ctx.t.detach().mean(),
            "train_flow_weight": t_weight.detach().mean(),
        }

        if ctx.node_target is not None and self.node_target_key in pred_data:
            pred = pred_data[self.node_target_key]
            target = ref_data[self.node_target_key].to(device=pred.device, dtype=pred.dtype)
            mask = self._node_mask(pred_data, pred)
            pred, mask = self._project_loss_layout(pred, mask, target)
            node_diff = pred - target
            node_weights = self._time_weight(ctx.node_t)
            node_stats = self._legacy_metric_stats(
                node_diff, mask, self.loss_type, node_weights
            )
            node_loss = node_stats[0]
            if self.log_train_compatible_loss or self.log_validation_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        node_diff,
                        mask,
                        "onsite",
                        metric_space=self._target_metric_space(
                            self.node_target_key,
                            node_diff,
                        ),
                    )
                )
            component_stats.append((node_stats, self.node_weight))
            # Flow namespace only; the bare train_onsite_loss is compatible-only (see
            # the block-ODE endpoint loss note above).
            state["train_flow_onsite_loss"] = node_loss.detach()

        if ctx.edge_target is not None and self.edge_target_key in pred_data:
            pred = pred_data[self.edge_target_key]
            target = ref_data[self.edge_target_key].to(device=pred.device, dtype=pred.dtype)
            mask = self._edge_mask(pred_data, pred)
            pred, mask = self._project_loss_layout(pred, mask, target)
            edge_diff = pred - target
            edge_weights = self._time_weight(ctx.edge_t)
            edge_stats = self._legacy_metric_stats(
                edge_diff, mask, self.loss_type, edge_weights
            )
            edge_loss = edge_stats[0]
            if self.log_train_compatible_loss or self.log_validation_compatible_loss:
                self._merge_compatible_clean_stats(
                    state,
                    self._compatible_clean_stats(
                        edge_diff,
                        mask,
                        "hopping",
                        metric_space=self._target_metric_space(
                            self.edge_target_key,
                            edge_diff,
                        ),
                    )
                )
            component_stats.append((edge_stats, self.edge_weight))
            # Flow namespace only; the bare train_hopping_loss is compatible-only.
            state["train_flow_hopping_loss"] = edge_loss.detach()

        if not component_stats:
            raise KeyError(
                "CFM could not compute a loss because model outputs do not contain "
                f"`{self.node_target_key}` or `{self.edge_target_key}`."
            )
        total = self._reduce_legacy_component_stats(tuple(component_stats))
        return self._finalize_loss(total, state, pred_data)

    def loss_on_sample(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Score :meth:`sample` output without trusting model-owned metadata.

        Block-space sampling always returns physical Full-H blocks, whereas a
        residual-dH training endpoint is scored in its configured target space.
        Keep that distinction in the flow API instead of a writable data-dict
        flag.  Other CFM output spaces retain their ordinary loss semantics.
        """
        if self.block_ode:
            if self.uureal_block_ode:
                # The compact-uu rollout returns residual-dH blocks, not
                # physical Full-H; score it in its configured residual space.
                return self._block_ode_endpoint_loss(
                    pred_data,
                    ref_data,
                    ctx,
                    prediction_is_full_h=False,
                )
            return self.compatible_loss_on_sample(pred_data, ref_data, ctx)
        return self.loss(pred_data, ref_data, ctx)

    def compatible_loss_on_sample(
        self,
        pred_data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        ctx: CFMContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Score a rollout with endpoint-compatible reductions."""
        if not self.block_ode:
            raise RuntimeError("compatible_loss_on_sample requires block_ode.")
        if self.uureal_block_ode:
            # The compact-uu rollout returns residual-dH blocks, not physical
            # Full-H, and this mode intentionally has no exact_rme block codec
            # (block_codec=None).  Scoring it as Full-H both miscounts H0 as
            # prediction error and crashes on block_codec.rme_to_blocks
            # (MultiTrainer first validation with validation_ode_steps=[1] +
            # log_validation_flow_euler_loss=false).  The residual scorer already
            # applies the same endpoint-compatible reductions.
            return self._block_ode_endpoint_loss(
                pred_data,
                ref_data,
                ctx,
                prediction_is_full_h=False,
            )
        return self._block_ode_endpoint_loss(
            pred_data,
            ref_data,
            ctx,
            prediction_is_full_h=True,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sampling_base(
        self,
        data: AtomicDataDict.Type,
        h0_key: str,
        feature_key: str,
        label: str,
    ) -> Optional[torch.Tensor]:
        if self.mode == "full":
            template = data.get(h0_key, None)
            if template is None:
                template = data.get(feature_key, None)
            return None if template is None else torch.zeros_like(template)

        base = data.get(h0_key, None)
        if base is None:
            if self.strict_h0:
                raise KeyError(f"Flow sampling requires `{h0_key}` for the {label} start state.")
            feature = data.get(feature_key, None)
            return None if feature is None else torch.zeros_like(feature)
        return base

    def _sample_block_ode(
        self,
        model: torch.nn.Module,
        state: AtomicDataDict.Type,
        *,
        num_steps: int,
        num_graphs: int,
        prior_seed: Optional[int] = None,
    ) -> AtomicDataDict.Type:
        """Projected endpoint-blend rollout in full AO-block state space."""
        if self.block_codec is None:
            raise RuntimeError("Block-space ODE codec was not constructed.")
        topology_sidecar = self._snapshot_block_topology(state)
        self._restore_block_topology(state, topology_sidecar)
        h0_blocks, _ = self._physical_h0_blocks(state)
        accumulator_dtype = self.dtype
        h0_blocks = BlockTensorResult(
            h0_blocks.node_blocks.to(dtype=accumulator_dtype),
            h0_blocks.edge_blocks.to(dtype=accumulator_dtype),
            h0_blocks.node_shapes,
            h0_blocks.edge_shapes,
        )
        block_current, _, _ = self._block_initial_state(
            state, h0_blocks, prior_seed=prior_seed
        )
        self._drop_block_authority_fields(state)
        output_only_keys = self._block_ode_output_only_keys()
        # LemMoEV3H0 consumes and deletes these input-side accelerators.  Keep
        # them outside the mutable rollout state and inject an unchanged copy at
        # every model call so a precomputed active set cannot drift on step 2+.
        lem_sidecar = {
            key: state.pop(key)
            for key in _LEM_INPUT_SIDECAR_KEYS
            if key in state
        }
        times = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=block_current.node_blocks.device,
            dtype=accumulator_dtype,
        )

        for step in range(num_steps):
            cur_t = times[step]
            next_t = times[step + 1]
            if not bool((cur_t < 1.0).item()):
                raise RuntimeError("Block-space ODE must never evaluate the model at t=1.")
            denom = 1.0 - cur_t
            alpha = (next_t - cur_t) / denom
            if not bool(((alpha > 0.0) & (alpha <= 1.0)).item()):
                raise RuntimeError(
                    f"Invalid block-space ODE blend alpha={float(alpha.item())}."
                )

            node_rme, edge_rme = self.block_codec.blocks_to_rme(state, block_current)
            state[self.node_h0_key] = node_rme.clone()
            state[self.edge_h0_key] = edge_rme.clone()
            if self.overwrite_feature_keys:
                state[self.node_target_key] = node_rme.clone()
                state[self.edge_target_key] = edge_rme.clone()
            state[self.flow_time_key] = torch.full(
                (num_graphs,),
                float(cur_t.item()),
                device=block_current.node_blocks.device,
                dtype=accumulator_dtype,
            )
            # Supported DeePTB modules mutate the dictionary and may use
            # in-place tensor operations.  Give each ODE step owned tensor
            # storage so those writes cannot alias the persistent rollout state.
            model_input = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
                if key not in output_only_keys
            }
            self._restore_block_topology(
                model_input, topology_sidecar, clone_values=True
            )
            model_input.update(
                {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in lem_sidecar.items()
                }
            )
            prediction = model(model_input)
            self._require_fresh_block_ode_outputs(prediction, step=step + 1)
            merged = state.copy()
            merged.update(prediction)
            self._restore_block_topology(merged, topology_sidecar)
            for key in _LEM_INPUT_SIDECAR_KEYS:
                merged.pop(key, None)
            raw_node_endpoint = self._require_real_finite_tensor(
                prediction[self.node_output_key],
                label="block-space ODE node endpoint prediction",
            )
            raw_edge_endpoint = self._require_real_finite_tensor(
                prediction[self.edge_output_key],
                label="block-space ODE edge endpoint prediction",
            )
            endpoint = BlockTensorResult(
                raw_node_endpoint.to(
                    device=block_current.node_blocks.device, dtype=accumulator_dtype
                ),
                raw_edge_endpoint.to(
                    device=block_current.edge_blocks.device, dtype=accumulator_dtype
                ),
                block_current.node_shapes,
                block_current.edge_shapes,
            )
            full_endpoint = self.block_codec.endpoint_to_full(endpoint, h0_blocks)
            full_endpoint = project_block_state(merged, self.idp, full_endpoint)
            block_current = project_block_state(
                merged,
                self.idp,
                BlockTensorResult(
                    (1.0 - alpha) * block_current.node_blocks
                    + alpha * full_endpoint.node_blocks,
                    (1.0 - alpha) * block_current.edge_blocks
                    + alpha * full_endpoint.edge_blocks,
                    block_current.node_shapes,
                    block_current.edge_shapes,
                ),
            )
            state = merged

        node_final_rme, edge_final_rme = self.block_codec.blocks_to_rme(state, block_current)
        state[self.node_h0_key] = node_final_rme
        state[self.edge_h0_key] = edge_final_rme
        if self.overwrite_feature_keys:
            state[self.node_target_key] = node_final_rme
            state[self.edge_target_key] = edge_final_rme
        attach_prediction_block_tensors(
            state,
            block_current,
            node_key=self.node_output_key,
            edge_key=self.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        state[self.flow_time_key] = torch.ones(
            num_graphs, device=block_current.node_blocks.device, dtype=accumulator_dtype
        )
        return state

    def _sample_uureal_block_ode(
        self,
        model: torch.nn.Module,
        state: AtomicDataDict.Type,
        *,
        num_steps: int,
        num_graphs: int,
    ) -> AtomicDataDict.Type:
        """Roll out the compact-uu residual state D without H0 materialization."""
        self._require_uureal_block_contract(state, require_endpoint_labels=False)
        topology_sidecar = self._snapshot_block_topology(state)
        self._restore_block_topology(state, topology_sidecar)
        node_shapes, edge_shapes = infer_block_shapes(state, self.idp, device=self.device)
        canvas = mapper_max_norb(self.idp)
        node_count = int(node_shapes.shape[0])
        edge_count = int(edge_shapes.shape[0])
        current = BlockTensorResult(
            torch.zeros((node_count, canvas, canvas), dtype=self.dtype, device=self.device),
            torch.zeros((edge_count, canvas, canvas), dtype=self.dtype, device=self.device),
            node_shapes,
            edge_shapes,
        )
        self._drop_block_authority_fields(state)
        output_only_keys = self._block_ode_output_only_keys()
        lem_sidecar = {
            key: state.pop(key)
            for key in _LEM_INPUT_SIDECAR_KEYS
            if key in state
        }
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        for step in range(num_steps):
            cur_t, next_t = times[step], times[step + 1]
            alpha = (next_t - cur_t) / (1.0 - cur_t)
            self._attach_uureal_residual_state(state, current)
            state[self.flow_time_key] = torch.full(
                (num_graphs,), float(cur_t.item()), dtype=self.dtype, device=self.device
            )
            model_input = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
                if key not in output_only_keys
            }
            self._restore_block_topology(model_input, topology_sidecar, clone_values=True)
            model_input.update({
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in lem_sidecar.items()
            })
            prediction = model(model_input)
            self._require_fresh_block_ode_outputs(prediction, step=step + 1)
            merged = state.copy()
            merged.update(prediction)
            self._restore_block_topology(merged, topology_sidecar)
            for key in _LEM_INPUT_SIDECAR_KEYS:
                merged.pop(key, None)
            endpoint = BlockTensorResult(
                self._require_real_finite_tensor(
                    prediction[self.node_output_key],
                    label="uureal residual node endpoint",
                ).to(device=self.device, dtype=self.dtype),
                self._require_real_finite_tensor(
                    prediction[self.edge_output_key],
                    label="uureal residual edge endpoint",
                ).to(device=self.device, dtype=self.dtype),
                current.node_shapes,
                current.edge_shapes,
            )
            endpoint = project_block_state(merged, self.idp, endpoint)
            current = project_block_state(
                merged,
                self.idp,
                BlockTensorResult(
                    (1.0 - alpha) * current.node_blocks + alpha * endpoint.node_blocks,
                    (1.0 - alpha) * current.edge_blocks + alpha * endpoint.edge_blocks,
                    current.node_shapes,
                    current.edge_shapes,
                ),
            )
            state = merged

        self._attach_uureal_residual_state(state, current)
        attach_prediction_block_tensors(
            state,
            current,
            node_key=self.node_output_key,
            edge_key=self.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        state[self.flow_time_key] = torch.ones(
            num_graphs, device=self.device, dtype=self.dtype
        )
        return state

    def _prior_state_as_residual_D0(
        self,
        data: AtomicDataDict.Type,
        prior_state: Any,
        h0_blocks: BlockTensorResult,
    ) -> BlockTensorResult:
        """TA-1: validate a caller-supplied ``prior_state`` and return it as D0.

        ``prior_state`` is the explicit transformable latent: a
        :class:`BlockTensorResult` or a ``(node_blocks, edge_blocks)`` pair already
        in the physical-H0 canvas block layout.  It is validated exactly like the
        seeded draw -- real/finite, shape against the H0 canvas, and codec-image
        certification (both the onsite/reverse/padding projection residual and the
        strict CG repack roundtrip, each bounded by ``block_inverse_atol``) -- and
        returned VERBATIM so a caller can rotate/permute it alongside the input to
        obtain pathwise equivariance (a rotated/permuted valid latent stays in the
        codec image, so certification still passes).  The TA-3 all-zero belt is NOT
        applied: the latent is caller-authoritative, so a deliberate zero start is
        respected (finiteness is still enforced).
        """
        assert self.block_codec is not None
        if isinstance(prior_state, BlockTensorResult):
            node_blocks, edge_blocks = prior_state.node_blocks, prior_state.edge_blocks
        else:
            try:
                node_blocks, edge_blocks = prior_state
            except (TypeError, ValueError):
                raise ValueError(
                    "residual prior_state must be a BlockTensorResult or a "
                    "(node_blocks, edge_blocks) pair in the physical-H0 canvas "
                    "block layout."
                )
        node_blocks = self._require_real_finite_tensor(
            node_blocks, label="residual prior_state node blocks"
        ).to(device=h0_blocks.node_blocks.device, dtype=self.dtype)
        edge_blocks = self._require_real_finite_tensor(
            edge_blocks, label="residual prior_state edge blocks"
        ).to(device=h0_blocks.edge_blocks.device, dtype=self.dtype)
        if tuple(node_blocks.shape) != tuple(h0_blocks.node_blocks.shape):
            raise ValueError(
                f"residual prior_state node blocks shape {tuple(node_blocks.shape)} "
                f"!= H0 canvas node shape {tuple(h0_blocks.node_blocks.shape)}."
            )
        if tuple(edge_blocks.shape) != tuple(h0_blocks.edge_blocks.shape):
            raise ValueError(
                f"residual prior_state edge blocks shape {tuple(edge_blocks.shape)} "
                f"!= H0 canvas edge shape {tuple(h0_blocks.edge_blocks.shape)}."
            )
        state_blocks = BlockTensorResult(
            node_blocks, edge_blocks, h0_blocks.node_shapes, h0_blocks.edge_shapes
        )
        projected = project_block_state(data, self.idp, state_blocks)
        proj_residual = max(
            self._max_abs(
                projected.node_blocks - state_blocks.node_blocks,
                label="prior_state node projection residual",
            ),
            self._max_abs(
                projected.edge_blocks - state_blocks.edge_blocks,
                label="prior_state edge projection residual",
            ),
        )
        if proj_residual > self.block_inverse_atol:
            raise ValueError(
                "residual prior_state violates onsite/reverse/padding "
                "constraints (supply it in the certified H0 canvas layout): max "
                f"residual={proj_residual:.6g}, atol={self.block_inverse_atol:.6g}."
            )
        node_rme, edge_rme = self.block_codec.blocks_to_rme(
            data, state_blocks, certify_image=True
        )
        roundtrip = self.block_codec.rme_to_blocks(
            data, node_rme, edge_rme, project=True
        )
        image_residual = max(
            self._max_abs(
                roundtrip.node_blocks - state_blocks.node_blocks,
                label="prior_state node codec-image residual",
            ),
            self._max_abs(
                roundtrip.edge_blocks - state_blocks.edge_blocks,
                label="prior_state edge codec-image residual",
            ),
        )
        if image_residual > self.block_inverse_atol:
            raise ValueError(
                "residual prior_state is outside the certified codec image: "
                f"max residual={image_residual:.6g}, atol={self.block_inverse_atol:.6g}."
            )
        return state_blocks

    def _sample_residual_ao_block_ode(
        self,
        model: torch.nn.Module,
        state: AtomicDataDict.Type,
        *,
        num_steps: int,
        num_graphs: int,
        prior_seed: Optional[int] = None,
        prior_state: Any = None,
    ) -> AtomicDataDict.Type:
        """Roll out the non-SOC residual state D, then assemble H = H0 + D once."""
        if self.block_codec is None:
            raise RuntimeError("Block-space ODE codec was not constructed.")
        self._require_spatial_residual_block_contract(
            state, require_endpoint_labels=False
        )
        topology_sidecar = self._snapshot_block_topology(state)
        self._restore_block_topology(state, topology_sidecar)
        h0_blocks, _ = self._physical_h0_blocks(state)
        h0_blocks = BlockTensorResult(
            h0_blocks.node_blocks.to(dtype=self.dtype),
            h0_blocks.edge_blocks.to(dtype=self.dtype),
            h0_blocks.node_shapes,
            h0_blocks.edge_shapes,
        )
        # Physical H0 RME is derived ONCE from the certified H0 blocks and written
        # to the H0 keys as the constant conditioning channel (contract-(2) rollout
        # lock): the model sees the SAME physical H0 RME at every step, while only
        # the residual state D advances (carried in the spatial residual keys).
        node_base, edge_base = self.block_codec.blocks_to_rme(
            state, h0_blocks, certify_image=True
        )
        if self.prior == "zero":
            # Zero prior: D0 = 0 exactly (byte-identical to v1).  A prior_seed and
            # an explicit prior_state are both meaningless here and rejected,
            # mirroring _block_initial_state's zero-prior symmetry (the top-level
            # sample() guard rejects them before dispatch; this is defense in depth
            # for direct callers).
            if prior_seed is not None:
                raise ValueError(
                    "residual_ao_block_ode with prior='zero' uses an exact zero "
                    "prior and rejects prior_seed."
                )
            if prior_state is not None:
                raise ValueError(
                    "residual_ao_block_ode with prior='zero' uses an exact zero "
                    "prior and rejects prior_state."
                )
            current = BlockTensorResult(
                torch.zeros_like(h0_blocks.node_blocks),
                torch.zeros_like(h0_blocks.edge_blocks),
                h0_blocks.node_shapes,
                h0_blocks.edge_shapes,
            )
        elif prior_state is not None:
            # TA-1 explicit latent: D0 = prior_state verbatim (validated/certified).
            # Mutually exclusive with prior_seed -- the caller supplies the state
            # rather than seeding a draw of it.
            if prior_seed is not None:
                raise ValueError(
                    "residual_ao_block_ode prior_state and prior_seed are mutually "
                    "exclusive: supply an explicit latent OR seed a draw, not both."
                )
            current = self._prior_state_as_residual_D0(state, prior_state, h0_blocks)
        else:
            # Stochastic residual prior: D0 = eps, drawn deterministically for a
            # given seed and certified in the codec image, matching the training
            # boundary state D_0 = project(eps) exactly.
            current = self._residual_stochastic_eps(
                state,
                node_base,
                edge_base,
                generator=self._seeded_generator(node_base.device, prior_seed),
                certify_image=True,
            )
        self._drop_block_authority_fields(state)
        state[self.node_h0_key] = node_base
        state[self.edge_h0_key] = edge_base
        output_only_keys = self._block_ode_output_only_keys()
        lem_sidecar = {
            key: state.pop(key)
            for key in _LEM_INPUT_SIDECAR_KEYS
            if key in state
        }
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        for step in range(num_steps):
            cur_t, next_t = times[step], times[step + 1]
            alpha = (next_t - cur_t) / (1.0 - cur_t)
            self._attach_spatial_residual_state(state, current)
            # Re-assert the constant channel every step: a model that echoed a
            # mutated H0 key back through ``merged`` must not drift contract (2).
            state[self.node_h0_key] = node_base
            state[self.edge_h0_key] = edge_base
            state[self.flow_time_key] = torch.full(
                (num_graphs,), float(cur_t.item()), dtype=self.dtype, device=self.device
            )
            model_input = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
                if key not in output_only_keys
            }
            self._restore_block_topology(model_input, topology_sidecar, clone_values=True)
            model_input.update({
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in lem_sidecar.items()
            })
            prediction = model(model_input)
            self._require_fresh_block_ode_outputs(prediction, step=step + 1)
            merged = state.copy()
            merged.update(prediction)
            self._restore_block_topology(merged, topology_sidecar)
            for key in _LEM_INPUT_SIDECAR_KEYS:
                merged.pop(key, None)
            endpoint = BlockTensorResult(
                self._require_real_finite_tensor(
                    prediction[self.node_output_key],
                    label="spatial residual node endpoint",
                ).to(device=self.device, dtype=self.dtype),
                self._require_real_finite_tensor(
                    prediction[self.edge_output_key],
                    label="spatial residual edge endpoint",
                ).to(device=self.device, dtype=self.dtype),
                current.node_shapes,
                current.edge_shapes,
            )
            endpoint = project_block_state(merged, self.idp, endpoint)
            current = project_block_state(
                merged,
                self.idp,
                BlockTensorResult(
                    (1.0 - alpha) * current.node_blocks + alpha * endpoint.node_blocks,
                    (1.0 - alpha) * current.edge_blocks + alpha * endpoint.edge_blocks,
                    current.node_shapes,
                    current.edge_shapes,
                ),
            )
            state = merged

        # Assemble the full Hamiltonian H = H0 + D exactly ONCE, outside the ODE.
        self._attach_spatial_residual_state(state, current)
        full = project_block_state(
            state,
            self.idp,
            BlockTensorResult(
                h0_blocks.node_blocks + current.node_blocks,
                h0_blocks.edge_blocks + current.edge_blocks,
                h0_blocks.node_shapes,
                h0_blocks.edge_shapes,
            ),
        )
        attach_prediction_block_tensors(
            state,
            full,
            node_key=self.node_output_key,
            edge_key=self.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        # H0 keys stay PHYSICAL H0 RME (deliberate divergence from the generic
        # ao_block_ode sampler, which overwrites them with the final state RME).
        state[self.node_h0_key] = node_base
        state[self.edge_h0_key] = edge_base
        state[self.flow_time_key] = torch.ones(
            num_graphs, device=self.device, dtype=self.dtype
        )
        return state

    def sample(
        self,
        model: torch.nn.Module,
        data: AtomicDataDict.Type,
        *,
        num_steps: int,
        prior_seed: Optional[int] = None,
        prior_state: Any = None,
    ) -> AtomicDataDict.Type:
        """Euler-integrate the endpoint-parameterized flow from the configured prior.

        ``prior_state`` (TA-1) is the optional explicit transformable latent for
        ``residual_ao_block_ode`` under stochastic residual priors: a
        BlockTensorResult or ``(node_blocks, edge_blocks)`` pair in the physical-H0
        canvas block layout, used as D0 verbatim so a caller can rotate/permute it
        alongside the input for pathwise equivariance.  It is mutually exclusive
        with ``prior_seed`` and illegal under a zero prior or in any other mode.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        state = data.copy()
        num_graphs = self._num_graphs(state)
        if self.uureal_block_ode:
            if prior_seed is not None:
                raise ValueError("uureal_block_ode uses an exact zero prior and rejects prior_seed.")
            if prior_state is not None:
                raise ValueError("uureal_block_ode uses an exact zero prior and rejects prior_state.")
            return self._sample_uureal_block_ode(
                model, state, num_steps=num_steps, num_graphs=num_graphs
            )
        if self.residual_ao_block_ode:
            # TA-4: fail BEFORE dispatch under the zero prior (exception ordering).
            # prior='zero' owns an exact zero D0 and rejects both a seed and an
            # explicit latent; stochastic priors consume prior_seed as the
            # deterministic D0=eps draw, or prior_state as the verbatim D0.
            if self.prior == "zero":
                if prior_seed is not None:
                    raise ValueError(
                        "residual_ao_block_ode uses an exact zero prior and rejects prior_seed."
                    )
                if prior_state is not None:
                    raise ValueError(
                        "residual_ao_block_ode uses an exact zero prior and rejects prior_state."
                    )
            return self._sample_residual_ao_block_ode(
                model,
                state,
                num_steps=num_steps,
                num_graphs=num_graphs,
                prior_seed=prior_seed,
                prior_state=prior_state,
            )
        if self.block_ode:
            # A-mode ao_block_ode draws its full start state B0=H0+eps and does not
            # accept an explicit prior_state (see the asymmetry note on
            # _block_initial_state); reject it loudly rather than silently ignore.
            if prior_state is not None:
                raise ValueError(
                    "prior_state is supported only by residual_ao_block_ode under "
                    "a stochastic residual prior; the A-mode ao_block_ode sampler "
                    "draws its start state and does not accept an explicit latent."
                )
            return self._sample_block_ode(
                model,
                state,
                num_steps=num_steps,
                num_graphs=num_graphs,
                prior_seed=prior_seed,
            )
        if prior_seed is not None:
            raise ValueError("prior_seed is supported only by block_ode.")
        if prior_state is not None:
            raise ValueError("prior_state is supported only by block_ode.")
        node_current = self._sampling_base(state, self.node_h0_key, self.node_target_key, "node")
        edge_current = self._sampling_base(state, self.edge_h0_key, self.edge_target_key, "edge")
        if node_current is None and edge_current is None:
            raise KeyError("Flow sampling requires node and/or edge Hamiltonian start features.")

        if self.prior != "zero":
            # One shared Haar-DM candidate for node+edge; None for every other prior.
            haar_candidate_idx = self._resolve_haar_candidate_idx(
                state, node_like=node_current, edge_like=edge_current
            )
            if node_current is not None:
                node_current = node_current + self._prior_like(
                    node_current,
                    self.node_sigma,
                    data=state,
                    label="node",
                    base=node_current,
                    reference_scale=False,
                    candidate_idx=haar_candidate_idx,
                )
            if edge_current is not None:
                edge_current = edge_current + self._prior_like(
                    edge_current,
                    self.edge_sigma,
                    data=state,
                    label="edge",
                    base=edge_current,
                    reference_scale=False,
                    candidate_idx=haar_candidate_idx,
                )

        like = node_current if node_current is not None else edge_current
        if self.output_space == "ao_block":
            if num_steps != 1:
                raise ValueError(
                    "AO-block CFM is a cross-space endpoint adapter and supports "
                    "only num_steps=1."
                )
            if node_current is not None:
                state[self.node_h0_key] = node_current
                if self.overwrite_feature_keys:
                    state[self.node_target_key] = node_current
            if edge_current is not None:
                state[self.edge_h0_key] = edge_current
                if self.overwrite_feature_keys:
                    state[self.edge_target_key] = edge_current
            state[self.flow_time_key] = torch.zeros(
                num_graphs, device=like.device, dtype=like.dtype
            )
            prediction = model(state)
            prediction[self.flow_time_key] = torch.ones(
                num_graphs, device=like.device, dtype=like.dtype
            )
            return prediction

        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            cur_t = float(step) * dt
            graph_t = torch.full((num_graphs,), cur_t, device=like.device, dtype=like.dtype)
            carry_keys = (
                "expert_edge_mask",
                "expert_node_mask",
                "expert_idx",
            )
            carried = {key: state[key] for key in carry_keys if key in state}
            if node_current is not None:
                state[self.node_h0_key] = node_current
                if self.overwrite_feature_keys:
                    state[self.node_target_key] = node_current
            if edge_current is not None:
                state[self.edge_h0_key] = edge_current
                if self.overwrite_feature_keys:
                    state[self.edge_target_key] = edge_current
            state[self.flow_time_key] = graph_t
            prediction = model(state)
            denom = max(1.0 - cur_t, self.t_eps)
            if node_current is not None:
                endpoint = prediction[self.node_target_key]
                endpoint, _raw_mask = project_uureal_to_like(self.idp, endpoint, node_current)
                node_current = node_current + dt * (endpoint - node_current) / denom
            if edge_current is not None:
                endpoint = prediction[self.edge_target_key]
                endpoint, _raw_mask = project_uureal_to_like(self.idp, endpoint, edge_current)
                edge_current = edge_current + dt * (endpoint - edge_current) / denom
            state = prediction.copy()
            state.update(carried)

        if node_current is not None:
            state[self.node_h0_key] = node_current
            state[self.node_target_key] = node_current
        if edge_current is not None:
            state[self.edge_h0_key] = edge_current
            state[self.edge_target_key] = edge_current
        state[self.flow_time_key] = torch.ones(num_graphs, device=like.device, dtype=like.dtype)
        return state


def build_hamiltonian_flow(
    options: Optional[Dict[str, Any]],
    *,
    idp: Any = None,
    dtype: Any = torch.float32,
    device: Any = torch.device("cpu"),
) -> HamiltonianCFM:
    options = canonicalize_flow_options(options)
    objective = str(options.get("objective", "cfm")).lower()
    block_ode_requested = bool(options.get("block_ode", False)) or str(
        options.get("output_space", "")
    ).lower().replace("-", "_") in {
        "ao_block_ode",
        "block_ode",
        "ao_blocks_ode",
        "uureal_block_ode",
        "spatial_uureal_residual_block_ode",
        "uureal_residual_block_ode",
        "residual_ao_block_ode",
    }
    if block_ode_requested and objective not in {"cfm", "flow_matching", "flow"}:
        raise ValueError("Block-space ODE v1 supports only the CFM endpoint objective.")
    if objective in {"pixel_meanflow", "pixel_mean_flow", "pmf", "meanflow", "mean_flow"}:
        from dptb.nnops.flow_meanflow import HamiltonianPixelMeanFlow

        return HamiltonianPixelMeanFlow(options, idp=idp, dtype=dtype, device=device)
    return HamiltonianCFM(options, idp=idp, dtype=dtype, device=device)


def assert_flow_h0_keys_reach_model(flow: Any, model: Any) -> None:
    """Fail closed when an enabled flow's H0 write-keys never reach the model.

    An enabled flow overwrites ``data[flow_options.node_h0_key]`` /
    ``data[flow_options.edge_h0_key]`` with the interpolated flow state ``x_t``
    precisely so the model's H0-init embedding consumes it.  The embedding reads
    its own ``embedding.h0_node_key`` / ``embedding.h0_edge_key`` slots, though.
    When the two disagree (e.g. the flow writes ``node_physical_h0`` but the
    embedding still reads the stored ``node_h0``), ``x_t`` never reaches the
    network: the base/prior difference is silently dropped, training looks
    healthy, and every flow ``mode`` collapses to the same stored-H0 run.  This
    guard turns that silent P0 deactivation into a build-time error.

    The check is a no-op unless the flow is enabled and the model actually
    exposes an H0-init consumer (a module carrying string ``h0_node_key`` /
    ``h0_edge_key`` attributes, i.e. ``H0InitLayer``).  Standard configs where
    the flow and embedding keys already agree (the shared ``node_h0`` /
    ``edge_h0`` defaults) pass untouched.
    """
    if flow is None or not getattr(flow, "enabled", False):
        return
    flow_node_key = getattr(flow, "node_h0_key", None)
    flow_edge_key = getattr(flow, "edge_h0_key", None)
    if flow_node_key is None and flow_edge_key is None:
        return

    modules_iter = getattr(model, "modules", None)
    if not callable(modules_iter):
        return

    modules = list(modules_iter())
    saw_h0_init = False
    mismatches = set()
    for module in modules:
        emb_node_key = getattr(module, "h0_node_key", None)
        emb_edge_key = getattr(module, "h0_edge_key", None)
        if not (isinstance(emb_node_key, str) and isinstance(emb_edge_key, str)):
            continue
        saw_h0_init = True
        if flow_node_key is not None and emb_node_key != flow_node_key:
            mismatches.add(("node", flow_node_key, emb_node_key))
        if flow_edge_key is not None and emb_edge_key != flow_edge_key:
            mismatches.add(("edge", flow_edge_key, emb_edge_key))

    if saw_h0_init and mismatches:
        detail = "; ".join(
            f"{side}: train_options.flow_options.{side}_h0_key={fk!r} but "
            f"model_options.embedding.h0_{side}_key={ek!r}"
            for side, fk, ek in sorted(mismatches)
        )
        raise ValueError(
            "Flow prior is silently deactivated: the interpolated H0 state is "
            "written to keys the model's H0-init embedding never reads, so the flow "
            "base/prior difference cannot reach the network. "
            f"{detail}. Point model_options.embedding.h0_node_key/h0_edge_key at the "
            "same keys as train_options.flow_options.node_h0_key/edge_h0_key "
            "(e.g. node_physical_h0 / edge_physical_h0), or align the flow keys to "
            "the embedding keys. See configs/physical_h0_flow_overlay.yaml."
        )

    if not getattr(flow, "block_ode", False):
        return
    if not saw_h0_init:
        raise ValueError("Block-space ODE requires an H0InitLayer consuming node+edge RME.")

    from dptb.nn.deeptb import NNENV
    from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0

    block_embeddings = [
        module for module in modules if isinstance(module, LemMoEV3H0)
    ]
    if len(block_embeddings) != 1:
        raise ValueError(
            "Block-space ODE requires exactly one real LemMoEV3H0 embedding; "
            f"found {len(block_embeddings)}."
        )
    block_embedding = block_embeddings[0]
    if (
        getattr(block_embedding, "output_route_name", None) != "h_b0"
        or not bool(getattr(block_embedding, "use_block_native_output", False))
        or not bool(
            getattr(block_embedding, "supports_full_block_edge_coverage", False)
        )
        or not bool(
            getattr(block_embedding, "require_full_block_edge_coverage", False)
        )
    ):
        raise ValueError(
            "Block-space ODE requires the real H-B0 block-native embedding with "
            "require_full_block_edge_coverage=true."
        )
    owners = [
        module
        for module in modules
        if isinstance(module, NNENV)
        and getattr(module, "embedding", None) is block_embedding
    ]
    if (
        len(owners) != 1
        or getattr(owners[0], "method", None) != "block_native"
        or not bool(getattr(owners[0], "blockwise_hamiltonian", False))
    ):
        raise ValueError(
            "Block-space ODE requires one NNENV owner with prediction.method="
            "'block_native' and prediction.blockwise_hamiltonian=true."
        )

    if bool(getattr(owners[0], "block_native_add_h0", False)) or any(
        bool(getattr(module, "add_h0", False)) for module in modules
    ):
        raise ValueError(
            "Block-space ODE requires model prediction.add_h0=false; the codec owns "
            "the single residual endpoint add-back."
        )

    conditioners = []
    for module in modules:
        if not bool(getattr(module, "use_flow_time_embedding", False)):
            continue
        conditioner = getattr(module, "flow_time_conditioner", None)
        if conditioner is not None:
            conditioners.append((module, conditioner))
    valid_conditioner = False
    for module, conditioner in conditioners:
        keys = tuple(getattr(conditioner, "flow_time_keys", ()) or ())
        key_matches = (
            flow.flow_time_key in keys
            if keys
            else getattr(conditioner, "flow_time_key", None) == flow.flow_time_key
        )
        if (
            key_matches
            and bool(getattr(module, "flow_time_condition_edges", False))
            and not bool(getattr(conditioner, "allow_missing_time", True))
        ):
            valid_conditioner = True
            break
    if not valid_conditioner:
        raise ValueError(
            "Block-space ODE requires explicit node+edge flow-time conditioning "
            f"on key {flow.flow_time_key!r} with flow_time_allow_missing=false."
        )

    merge_modes = [
        getattr(module, "merge_mode", None)
        for module in modules
        if isinstance(getattr(module, "h0_node_key", None), str)
    ]
    if not merge_modes or any(mode != "replace" for mode in merge_modes):
        raise ValueError("Block-space ODE requires H0InitLayer h0_merge_mode='replace'.")


def assert_model_in_loss_endpoint_metric_space(
    flow: Any,
    criterion: Any,
    *,
    criterion_name: str = "train",
) -> None:
    """Fail before training when MeanFlow and its endpoint criterion disagree.

    Model-in-loss objectives build their compatible endpoint reductions directly
    from ``flow.node_target_key`` / ``flow.edge_target_key``.  There is no model
    output dictionary left for :class:`Trainer` to re-run a block criterion on,
    so an RME-target MeanFlow cannot satisfy a block-native endpoint reducer.
    Converting representations in this hot path is intentionally not an
    implicit fallback: block-to-RME slice reductions can dominate a block-native
    loss forward on GPU.

    The check is deliberately limited to criteria that explicitly declare
    ``endpoint_metric_space='block'`` and flows with ``model_in_loss=True``.
    Other criteria keep their existing behavior.  At configuration time a
    target key containing ``block`` is the flow's explicit block-space contract,
    matching :meth:`HamiltonianCFM._target_metric_space` at runtime.
    """
    if (
        flow is None
        or not getattr(flow, "enabled", False)
        or not getattr(flow, "model_in_loss", False)
        or criterion is None
    ):
        return

    metric_space = str(getattr(criterion, "endpoint_metric_space", "rme")).lower()
    if metric_space != "block":
        return

    node_key = str(getattr(flow, "node_target_key", ""))
    edge_key = str(getattr(flow, "edge_target_key", ""))
    non_block_targets = []
    if "block" not in node_key.lower():
        non_block_targets.append(f"node_target_key={node_key!r}")
    if "block" not in edge_key.lower():
        non_block_targets.append(f"edge_target_key={edge_key!r}")
    if not non_block_targets:
        return

    raise ValueError(
        "Pixel MeanFlow endpoint metric-space mismatch at trainer initialization: "
        f"the {criterion_name} criterion declares endpoint_metric_space='block', "
        "but the configured MeanFlow target keys are not all block-space "
        f"({', '.join(non_block_targets)}). MeanFlow computes endpoint statistics "
        "directly in its target representation; Trainer will not perform an "
        "implicit online block-to-RME or RME-to-block conversion. Configure "
        "flow_options.node_target_key/edge_target_key as block-space endpoint "
        "keys that the selected model route emits under the same names and align "
        "the block criterion's prediction/target route, or use a matching "
        "RME feature-space flow and endpoint criterion."
    )


def configure_jvp_friendly_backends(flow_options: Optional[Dict[str, Any]]) -> bool:
    """Best-effort process-level prep for the pixel-meanflow jvp backend.

    TorchScript-compiled e3nn modules (SphericalHarmonics first) reject the
    storageless dual tensors that torch.func.jvp propagates, so e3nn's
    jit_mode must be 'eager' before any model is built. No-op unless the
    resolved flow objective is meanflow-family with du_dt_backend=jvp.
    Returns True if e3nn was switched.
    """
    options = canonicalize_flow_options(flow_options)
    if not options.get("enabled", False):
        return False
    objective = str(options.get("objective", "cfm")).lower()
    if objective not in {"pixel_meanflow", "pixel_mean_flow", "pmf", "meanflow", "mean_flow"}:
        return False
    mf = dict(options.get("meanflow", {}) or {})
    backend = str(mf.get("du_dt_backend", "finite_difference")).lower().replace("-", "_")
    if backend != "jvp":
        return False
    try:
        import e3nn

        e3nn.set_optimization_defaults(jit_mode="eager")
    except Exception as exc:
        log.warning(
            "pixel_meanflow.du_dt_backend=jvp requested but e3nn could not be "
            "switched to eager jit_mode (%s); the jvp call will likely fall "
            "back to finite_difference at the first scripted module.",
            exc,
        )
        return False
    log.info(
        "pixel_meanflow.du_dt_backend=jvp: switched e3nn jit_mode to 'eager' so "
        "TorchScript-compiled e3nn modules accept forward-mode dual tensors."
    )
    return True


def resolve_flow_log_fields(flow: Optional[HamiltonianCFM]) -> Tuple[list, bool]:
    """Scalar log fields implied by the *effective* flow flags.

    Entry points used to re-derive the field list from raw top-level
    flow_options keys, which drifts from the resolved flow object: pixel
    meanflow reads ``meanflow.*`` overrides, never computes the raw-batch
    train_compatible loss, and emits ``validation_flow_one_step_loss`` instead
    of CFM's ``validation_flow_t0_loss``/``validation_flow_euler_N_loss``. A
    registered field the run never updates is seeded as 0.0 and the terminal
    Logger then prints a constant zero that reads like a perfect model.

    Returns ``(flow_log_fields, register_legacy_validation)``. The second
    element says whether the legacy ``validation_onsite_loss`` /
    ``validation_hopping_loss`` keys will actually be produced: for flow runs
    that is the endpoint-compatible euler-1 pass; with flow disabled the plain
    validation criterion fills them.
    """
    if flow is None or not getattr(flow, "enabled", False):
        return [], True

    model_in_loss = bool(getattr(flow, "model_in_loss", False))
    ode_steps = [int(n) for n in getattr(flow, "validation_ode_steps", ()) or ()]
    fields = [
        "train_flow_loss",
        "train_flow_onsite_loss",
        "train_flow_hopping_loss",
        "train_flow_t",
        "train_flow_weight",
    ]
    if model_in_loss:
        fields.extend(
            [
                "train_flow_r",
                "train_flow_h",
                "train_flow_objective_finite_difference",
                "train_flow_objective_semigroup",
                # canary scalars: a silent jvp->finite_difference fallback is
                # invisible in production without these in the terminal/TB log.
                "train_flow_du_dt_backend_jvp",
                "train_flow_explicit_model_calls",
            ]
        )
        if getattr(flow, "meanflow_objective", "finite_difference") in {"semigroup", "hybrid"}:
            fields.extend(
                [
                    "train_flow_semigroup_split_t",
                    "train_flow_semigroup_weight",
                    "train_flow_semigroup_endpoint_weight",
                    "train_flow_semigroup_explicit_model_calls",
                    "train_flow_onsite_semigroup_loss",
                    "train_flow_hopping_semigroup_loss",
                ]
            )
    if getattr(flow, "log_validation_random_t_loss", True):
        fields.append("validation_flow_random_t_loss")
    if getattr(flow, "log_validation_t0_loss", True):
        fields.append(
            "validation_flow_one_step_loss" if model_in_loss else "validation_flow_t0_loss"
        )
    if not model_in_loss and getattr(flow, "log_validation_flow_euler_loss", True):
        for num_steps in ode_steps:
            fields.append(f"validation_flow_euler_{num_steps}_loss")
    # Euler-1 compatible metrics are exposed through the canonical non-CFM
    # train/validation fields.  Block ODE additionally keeps the explicit
    # Euler-1 aliases for continuity with its certified endpoint telemetry;
    # ordinary CFM/MeanFlow avoids the duplicate TensorBoard curves.
    log_validation_compatible = True
    if log_validation_compatible:
        for num_steps in ode_steps:
            if num_steps == 1 and not bool(getattr(flow, "block_ode", False)):
                continue
            fields.extend(
                [
                    f"validation_compatible_euler_{num_steps}_loss",
                    f"validation_compatible_euler_{num_steps}_onsite_loss",
                    f"validation_compatible_euler_{num_steps}_hopping_loss",
                ]
            )
    register_legacy = bool(
        log_validation_compatible
        and getattr(flow, "compatible_loss_to_legacy_keys", True)
        and 1 in ode_steps
    )
    return fields, register_legacy


def __getattr__(name: str) -> Any:
    """Lazy backward-compatible export without a circular-import side effect."""

    if name == "HamiltonianPixelMeanFlow":
        from dptb.nnops.flow_meanflow import HamiltonianPixelMeanFlow

        return HamiltonianPixelMeanFlow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
