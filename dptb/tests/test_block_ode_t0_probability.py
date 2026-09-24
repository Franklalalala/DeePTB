"""``t0_probability`` window (the exact-``t=0`` boundary-injection mass): the closed upper bound
and (for ``uureal_block_ode``/``residual_ao_block_ode``) the positive lower bound, at both the
flow-constructor and ``validate_block_ode_contract`` layers; the omitted-key default of 0.15
surviving the real ``normalize()`` ordering; and the runtime injection bypassing ``t_min``.

``torch.rand(...)`` draws from ``[0, 1)``, so ``t0_probability`` must stay in a closed-below,
open-above window: ``p >= 1`` (or non-finite) forces every sample to ``t=0`` and starves the
interior ``[t_min, t_max]`` schedule.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from dptb.configuration import canonicalize_flow_options
from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import flow_options, validate_block_ode_contract
from dptb.tests.block_ode_fixtures import (
    _b_flow,
    _load_b_config,
    _mapper,
    _mutate,
    _uureal_config,
    _uureal_flow,
    _uureal_mapper,
    _valid_contract,
)

_FLOW_SCHEMA = flow_options()

# p >= 1 (and non-finite p) collapse `torch.rand(...) < p` to all-True, sending every sample to
# the t=0 boundary; NaN/Inf are promised rejected by the docstring's `math.isfinite` gate.
OUT_OF_RANGE = [1.0, 1.0000001, math.nextafter(1.0, 2.0), float("nan"), float("inf")]
# 0 < p < 1, incl. the edge just below the open upper bound and the recommended default band.
INTERIOR = [0.15, 0.999]


def _residual_config(t0_probability):
    return _mutate(_load_b_config(), ("train_options", "flow_options", "t0_probability"), t0_probability)


_MODES = {
    "uureal": (_uureal_mapper, _uureal_flow, lambda p: _uureal_config(t0_probability=p)),
    "residual": (_mapper, _b_flow, _residual_config),
}


@pytest.mark.parametrize("layer", ["ctor", "argcheck"])
@pytest.mark.parametrize("mode", sorted(_MODES))
@pytest.mark.parametrize("p", OUT_OF_RANGE)
def test_t0_probability_window_rejects_out_of_range_at_both_layers(mode, layer, p):
    mapper_fn, flow_fn, config_fn = _MODES[mode]
    with pytest.raises(ValueError, match="t0_probability"):
        if layer == "ctor":
            flow_fn(mapper_fn(), t0_probability=p)
        else:
            validate_block_ode_contract(config_fn(p))


@pytest.mark.parametrize("layer", ["ctor", "argcheck"])
@pytest.mark.parametrize("mode", sorted(_MODES))
@pytest.mark.parametrize("p", INTERIOR)
def test_t0_probability_window_accepts_interior_at_both_layers(mode, layer, p):
    mapper_fn, flow_fn, config_fn = _MODES[mode]
    if layer == "ctor":
        assert flow_fn(mapper_fn(), t0_probability=p).t0_probability == pytest.approx(p)
    else:
        assert validate_block_ode_contract(config_fn(p)) is None


def _generic_flow(**overrides):
    """A minimal generic (non-block, output_space='rme') CFM."""
    return HamiltonianCFM({"enabled": True, "prior": "zero", **overrides})


def test_generic_cfm_rejects_explicit_t0_probability_ge_one():
    with pytest.raises(ValueError, match="t0_probability"):
        _generic_flow(t0_probability=1.5)


def test_generic_cfm_accepts_zero_default():
    # Omitted -> frozen 0.0 default (generic window is [0, 1), no positive lower bound);
    # explicit 0.0 is likewise fine.
    assert _generic_flow().t0_probability == pytest.approx(0.0)
    assert _generic_flow(t0_probability=0.0).t0_probability == pytest.approx(0.0)


@pytest.mark.parametrize("mode,mapper_fn,flow_fn,dtype", [
    ("uureal", _uureal_mapper, _uureal_flow, torch.float32),
    ("residual", _mapper, _b_flow, torch.float64),
], ids=["uureal", "residual"])
def test_t0_injection_bypasses_t_min_clamp(mode, mapper_fn, flow_fn, dtype):
    """t0_probability exists to train the t=0, D=0 inference boundary: injected zeros must not be
    re-clamped to t_min, for both block-ODE modes."""
    assert mode in ("uureal", "residual")
    mapper = mapper_fn()
    # p=1 is rejected at construction (full boundary collapse is a misconfiguration); force full
    # injection AFTER construction to keep exercising the runtime bypass mechanism itself.
    flow = flow_fn(mapper, t_min=0.5, t0_probability=0.15)
    flow.t0_probability = 1.0
    t = flow._sample_t(num_graphs=64, device=torch.device("cpu"), dtype=dtype)
    assert torch.equal(t, torch.zeros_like(t))

    # Partial injection: zeros survive AND every non-zero sample honours t_min.
    flow = flow_fn(mapper, t_min=0.5, t0_probability=0.5)
    generator = torch.Generator().manual_seed(0)
    t = flow._sample_t(num_graphs=512, device=torch.device("cpu"), dtype=dtype, generator=generator)
    zero = t == 0.0
    assert bool(zero.any()) and bool((~zero).any())
    assert bool((t[~zero] >= 0.5).all())


# ---------------------------------------------------------------------------
# Omitted-key default (0.15 for uureal_block_ode/residual_ao_block_ode) must
# survive the REAL normalize() ordering: canonicalize_flow_options runs before
# dargs' schema normalize_value, which otherwise turns "omitted" and "explicit
# 0.0" into the same 0.0 -- making the omission-default path unreachable.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "output_space",
    [
        "uureal_block_ode", "spatial_uureal_residual_block_ode", "uureal_residual_block_ode",
        "residual_ao_block_ode", "residual-ao-block-ode", "RESIDUAL_AO_BLOCK_ODE",
    ],
)
def test_canonicalize_injects_0_15_for_omitted_t0_probability(output_space):
    out = canonicalize_flow_options({"output_space": output_space, "block_ode": True})
    assert out["t0_probability"] == pytest.approx(0.15)


def _through_real_normalize_order(flow_options_raw):
    """canonicalize_flow_options -> dargs normalize_value: the exact ordering
    dptb.utils.argcheck.normalize() applies to train_options.flow_options."""
    return _FLOW_SCHEMA.normalize_value(canonicalize_flow_options(flow_options_raw))


def _config_with_normalized_flow(config):
    config = copy.deepcopy(config)
    config["train_options"]["flow_options"] = _through_real_normalize_order(config["train_options"]["flow_options"])
    return config


def test_uureal_omitted_t0_probability_survives_real_normalize_order():
    cfg = _uureal_config()
    assert "t0_probability" not in cfg["train_options"]["flow_options"]
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == pytest.approx(0.15)
    assert validate_block_ode_contract(normalized) is None


def test_residual_omitted_t0_probability_survives_real_normalize_order():
    cfg = _load_b_config()
    del cfg["train_options"]["flow_options"]["t0_probability"]
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == pytest.approx(0.15)
    assert validate_block_ode_contract(normalized) is None


def test_generic_ao_block_ode_omitted_t0_probability_keeps_dargs_zero_default():
    """The plain (non-uureal, non-residual) block_ode route never required a positive
    t0_probability, and still gets dargs' schema-wide 0.0 default."""
    cfg = _valid_contract()
    assert "t0_probability" not in cfg["train_options"]["flow_options"]
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == 0.0
    assert validate_block_ode_contract(normalized) is None


def test_non_block_ode_omitted_t0_probability_keeps_dargs_zero_default():
    raw = {"enabled": True, "prior": "zero", "output_space": "rme"}
    normalized = _through_real_normalize_order(raw)
    assert normalized["t0_probability"] == 0.0


@pytest.mark.parametrize("explicit", [0.0, -0.1])
def test_uureal_explicit_nonpositive_t0_probability_still_rejected(explicit):
    cfg = _uureal_config(t0_probability=explicit)
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == explicit
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(normalized)


@pytest.mark.parametrize("explicit", [0.0, -0.1])
def test_residual_explicit_nonpositive_t0_probability_still_rejected(explicit):
    cfg = _mutate(_load_b_config(), ("train_options", "flow_options", "t0_probability"), explicit)
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == explicit
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(normalized)


@pytest.mark.parametrize("explicit", [0.1, 0.15, 0.25, 0.999])
def test_uureal_explicit_positive_t0_probability_unchanged(explicit):
    cfg = _uureal_config(t0_probability=explicit)
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == pytest.approx(explicit)
    assert validate_block_ode_contract(normalized) is None
