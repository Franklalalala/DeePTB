"""Upper-bound guard for ``flow_options.t0_probability`` (0.0 <= p < 1.0).

``t0_probability`` is the Bernoulli mass of the exact-``t=0`` boundary injection
in :meth:`HamiltonianCFM._sample_t` (``torch.rand(...) < p``).  Because
``torch.rand`` draws from ``[0, 1)``, a probability ``p >= 1`` makes that
comparison *always* True: every sample in the batch is forced to ``t=0`` and the
interior ``[t_min, t_max]`` schedule silently starves.  Before this guard the
value only had a lower bound (``> 0`` for the block-ODE modes, ``>= 0`` for the
generic CFM), so a misconfigured ``p >= 1`` was accepted at both the flow ctor
and ``validate_block_ode_contract``.

This suite pins the closed upper bound at BOTH layers for both block-ODE modes
(``uureal_block_ode`` / ``residual_ao_block_ode``) and for the generic CFM.  It
reuses the frozen mode suites' flow builders / argcheck config so the option
dicts stay in lock-step with production; it never edits those files.

NOTE: hardening this bound deliberately narrows the previously-frozen block-ODE
gate -- ``p == 1`` was always a misconfiguration (full boundary collapse), so it
now fails closed like any other out-of-range probability.
"""
from __future__ import annotations

import copy

import pytest

from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import validate_block_ode_contract

# Reuse the frozen mode suites' builders (default "prepend" import mode makes the
# sibling test modules importable by bare name).  Do NOT edit those files.
from test_uureal_block_ode import (  # noqa: E402
    _flow as _uureal_flow,
    _mapper as _uureal_mapper,
)
from test_residual_ao_block_ode import (  # noqa: E402
    _b_flow as _residual_flow,
    _load_b_config,
    _mapper as _residual_mapper,
    _mutate,
)


# p >= 1 (and non-finite p) collapse ``torch.rand(...) < p`` to all-True, sending
# every training sample to the t=0 boundary.  These must fail closed everywhere.
OUT_OF_RANGE = [1.0, 1.0000001, 1.5, 2.0, 1e9]

# Interior probabilities that must keep validating (0 < p < 1, incl. the 0.999
# edge just below the open upper bound and the recommended 0.15 default band).
INTERIOR = [0.15, 0.999]


def _uureal_config(**flow_overrides):
    """The uureal_block_ode contract config (mirrors test_uureal_block_ode.py)."""
    config = {
        "common_options": {
            "dtype": "float32", "has_soc": True,
            "nextham_uureal_mask": True, "full_soc_prediction": False,
        },
        "train_options": {
            "flow_options": {
                "enabled": True, "mode": "residual", "prior": "zero",
                "output_space": "uureal_block_ode", "block_ode": True,
                "state_space": "residual_ao_block", "target_semantics": "residual_dh",
                "block_input_adapter": "direct_cg",
                "h0_condition_space": "compact_uureal_rme",
                "prediction_add_h0": False, "time_conditioning_required": True,
                "node_block_target_key": "node_delta_hamil_blocks",
                "edge_block_target_key": "edge_delta_hamil_blocks",
                "node_block_shape_key": "node_delta_hamil_block_shape",
                "edge_block_shape_key": "edge_delta_hamil_block_shape",
                "validation_ode_steps": [1, 3],
            }
        },
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0", "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_uureal_residual_block_input": True,
                "use_flow_time_embedding": True, "flow_time_condition_edges": True,
                "flow_time_allow_missing": False, "h0_merge_mode": "replace",
                "use_h0_node_init": True, "use_h0_edge_init": True,
            },
            "prediction": {
                "method": "block_native", "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True, "add_h0": False,
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset", "get_Hamiltonian": True, "get_H0": True,
                "residual_hamiltonian": False, "require_full_h_target": False,
                "require_residual_h_target": False, "require_uureal_block_ode": True,
            }
        },
    }
    config["train_options"]["flow_options"].update(flow_overrides)
    return config


def _residual_config(t0_probability):
    """The residual_ao_block_ode B-arm yaml with t0_probability overridden."""
    return _mutate(
        _load_b_config(),
        ("train_options", "flow_options", "t0_probability"),
        t0_probability,
    )


# ---------------------------------------------------------------------------
# (1) flow ctor rejects p >= 1 for BOTH block-ODE modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", OUT_OF_RANGE)
def test_uureal_ctor_rejects_t0_probability_ge_one(p):
    mapper = _uureal_mapper()
    with pytest.raises(ValueError, match="t0_probability"):
        _uureal_flow(mapper, t0_probability=p)


@pytest.mark.parametrize("p", OUT_OF_RANGE)
def test_residual_ctor_rejects_t0_probability_ge_one(p):
    mapper = _residual_mapper()
    with pytest.raises(ValueError, match="t0_probability"):
        _residual_flow(mapper, t0_probability=p)


# ---------------------------------------------------------------------------
# (2) validate_block_ode_contract rejects p >= 1 for BOTH modes' full configs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", OUT_OF_RANGE)
def test_uureal_argcheck_rejects_t0_probability_ge_one(p):
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(_uureal_config(t0_probability=p))


@pytest.mark.parametrize("p", OUT_OF_RANGE)
def test_residual_argcheck_rejects_t0_probability_ge_one(p):
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(_residual_config(p))


# ---------------------------------------------------------------------------
# (3) interior probabilities still accepted at BOTH layers for both modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", INTERIOR)
def test_uureal_accepts_interior_probability_both_layers(p):
    mapper = _uureal_mapper()
    assert _uureal_flow(mapper, t0_probability=p).t0_probability == pytest.approx(p)
    assert validate_block_ode_contract(_uureal_config(t0_probability=p)) is None


@pytest.mark.parametrize("p", INTERIOR)
def test_residual_accepts_interior_probability_both_layers(p):
    mapper = _residual_mapper()
    assert _residual_flow(mapper, t0_probability=p).t0_probability == pytest.approx(p)
    assert validate_block_ode_contract(_residual_config(p)) is None


# ---------------------------------------------------------------------------
# (4) generic (non-block) CFM: explicit p >= 1 now rejected; frozen p=0 default OK
# ---------------------------------------------------------------------------
def _generic_flow(**overrides):
    """A minimal generic (non-block, output_space='rme') CFM."""
    options = {"enabled": True, "prior": "zero"}
    options.update(overrides)
    return HamiltonianCFM(options)


def test_generic_cfm_rejects_explicit_t0_probability_ge_one():
    with pytest.raises(ValueError, match="t0_probability"):
        _generic_flow(t0_probability=1.5)


def test_generic_cfm_accepts_zero_default():
    # Omitted -> frozen 0.0 default; explicit 0.0 also fine (generic window 0 <= p < 1).
    assert _generic_flow().t0_probability == pytest.approx(0.0)
    assert _generic_flow(t0_probability=0.0).t0_probability == pytest.approx(0.0)


# ===========================================================================
# H9: projected_te effective-scale working-interval gates (P3 argcheck + flow
# ctor).  The GATE rejects effective scales node_sigma*te_prior_sigma /
# edge_sigma*te_prior_sigma that a bare representability check would accept but
# that collapse to exact zero (subnormal-adjacent) or overflow (near dtype-max)
# once the unbounded Gaussian radius multiplies in.  Both layers are pinned here.
# ===========================================================================
import math as _math_h9

import torch as _torch_h9

from test_residual_ao_block_ode import (  # noqa: E402
    _b_te_flow as _te_flow,
    _load_te_config,
)

_FMAX_FP32 = float(_torch_h9.finfo(_torch_h9.float32).max)
_FMAX_FP64 = float(_torch_h9.finfo(_torch_h9.float64).max)

# fp32 effective scales OUTSIDE the safe working interval [2**-100, fmax/2**12]:
#   2**-149 (smallest subnormal) and 2**-120 sit below the min; fmax/2 is above
#   the max headroom bound (fmax/2**12).
FP32_OUT_OF_INTERVAL = [2.0 ** -149, 2.0 ** -120, _FMAX_FP32 / 2.0]
# fp32 effective scales safely inside the working interval.
FP32_IN_INTERVAL = [1.0, 1e-6, 1e6]
_SCALE_MATCH = "working interval|effective scale"


def _te_config_scale(te_prior_sigma, *, dtype="float32"):
    """The te-arm yaml with an overridden te_prior_sigma (effective scale ==
    node_sigma*te_prior_sigma == te_prior_sigma, since node/edge sigma == 1.0)."""
    cfg = _mutate(
        _load_te_config(),
        ("train_options", "flow_options", "te_prior_sigma"),
        float(te_prior_sigma),
    )
    if dtype != "float32":
        cfg = _mutate(cfg, ("common_options", "dtype"), dtype)
    return cfg


# ---- flow ctor ------------------------------------------------------------
@pytest.mark.parametrize("scale", FP32_OUT_OF_INTERVAL)
def test_h9_flow_ctor_rejects_fp32_effective_scale_outside_interval(scale):
    mapper = _residual_mapper()
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        _te_flow(mapper, dtype=_torch_h9.float32, te_prior_sigma=float(scale))


@pytest.mark.parametrize("scale", FP32_IN_INTERVAL)
def test_h9_flow_ctor_accepts_fp32_effective_scale_inside_interval(scale):
    mapper = _residual_mapper()
    flow = _te_flow(mapper, dtype=_torch_h9.float32, te_prior_sigma=float(scale))
    assert flow.te_prior_sigma == pytest.approx(scale)


# ---- argcheck -------------------------------------------------------------
@pytest.mark.parametrize("scale", FP32_OUT_OF_INTERVAL)
def test_h9_argcheck_rejects_fp32_effective_scale_outside_interval(scale):
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        validate_block_ode_contract(_te_config_scale(scale))


@pytest.mark.parametrize("scale", FP32_IN_INTERVAL)
def test_h9_argcheck_accepts_fp32_effective_scale_inside_interval(scale):
    assert validate_block_ode_contract(_te_config_scale(scale)) is None


# ---- fp64 spot-check (both layers) ---------------------------------------
def test_h9_fp64_working_interval_spot_check_both_layers():
    """fp64 uses the wider interval [2**-996, fmax/2**12]: a 2**-1040 subnormal-
    adjacent scale is rejected at BOTH layers, while 1.0 is accepted at both."""
    mapper = _residual_mapper()
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        _te_flow(mapper, dtype=_torch_h9.float64, te_prior_sigma=2.0 ** -1040)
    assert _te_flow(mapper, dtype=_torch_h9.float64, te_prior_sigma=1.0).te_prior_sigma == pytest.approx(1.0)

    # argcheck fp64: the effective-scale gate fires before the atol cap, so a
    # below-floor scale is rejected for the working-interval reason even though the
    # te yaml's fp32 atol would separately be too loose for fp64.
    with pytest.raises(ValueError, match=_SCALE_MATCH):
        validate_block_ode_contract(_te_config_scale(2.0 ** -1040, dtype="float64"))
