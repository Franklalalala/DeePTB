# SPDX-License-Identifier: LGPL-3.0-or-later
"""P2-1 fix: block_ode's positive-t0_probability requirement (uureal_block_ode /
residual_ao_block_ode) must not make the documented "omit it and the runtime
default 0.15 applies" behavior unreachable.

Before this fix: dargs' schema-wide t0_probability default is 0.0
(dptb.utils.argcheck's flow_options Argument), and normalize() runs dargs'
``base.normalize_value(data)`` immediately after
``dptb.configuration.canonicalize_training_config`` but BEFORE
``validate_block_ode_contract``.  So by the time
``validate_block_ode_contract`` inspected the config in the real pipeline,
"omitted" and "explicit 0.0" had already become indistinguishable -- both
read back as 0.0 -- and ``validate_block_ode_contract`` (correctly) rejected
0.0.  That made the "omitting t0_probability lets the runtime default 0.15
apply" comments in ``dptb.nnops.flow`` and
``validate_block_ode_contract`` itself unreachable in the actual
``normalize()`` pipeline, even though:
  * the identical omitted-key dict validated fine when
    ``validate_block_ode_contract`` was called directly in existing tests
    (its own internal ``canonicalize_training_config`` pre-pass never
    actually observes dargs' schema-default injection, so those tests never
    exercised the bug either way), and
  * ``HamiltonianCFM``'s own ``options.get("t0_probability", 0.15)``
    constructor fallback works fine in that same direct-construction context
    (``test_uureal_block_ode.py``/``test_residual_ao_block_ode.py`` already
    pin exactly that).

Fix (swimlane task notes, Option A -- provenance IS recoverable):
``dptb.configuration.canonicalize_flow_options`` now injects
``t0_probability=0.15`` for uureal_block_ode/residual_ao_block_ode configs
that omit the key, before dargs ever sees the dict (canonicalize always runs
before dargs' ``normalize_value`` in ``normalize()``).  An explicitly
configured value -- including an explicit 0.0 -- is left completely untouched
and still fails the positive-probability contract, exactly as documented.
The generic (non-uureal, non-residual) ao_block_ode route and non-block-ode
flows are unaffected: they keep dargs' schema-wide 0.0 default.

This suite pins the chosen behavior against the REAL normalize() ordering
(canonicalize -> dargs schema normalize -> validate_block_ode_contract), not
just against validate_block_ode_contract in isolation -- section 2 below
proves the fix actually closes the gap the bug lived in, and would fail on
the pre-fix code (verified by hand against the unmodified module during
development; see the swimlane handoff notes).
"""
from __future__ import annotations

import copy

import pytest

from dptb.configuration import canonicalize_flow_options
from dptb.utils.argcheck import flow_options, validate_block_ode_contract

# Reuse the frozen block-ODE contract fixtures from sibling test modules
# (default "prepend" import mode makes them importable by bare name).  Do NOT
# edit those files.
from test_block_ode_redteam import _valid_contract  # noqa: E402 -- generic ao_block_ode
from test_residual_ao_block_ode import _load_b_config, _mutate  # noqa: E402
from test_t0_probability_bounds import _uureal_config  # noqa: E402


_FLOW_SCHEMA = flow_options()


def _through_real_normalize_order(flow_options_raw):
    """canonicalize_flow_options -> dargs normalize_value: exactly the
    ordering dptb.utils.argcheck.normalize() applies to
    train_options.flow_options (canonicalize_training_config(data) runs
    before base.normalize_value(data))."""
    return _FLOW_SCHEMA.normalize_value(canonicalize_flow_options(flow_options_raw))


def _config_with_normalized_flow(config):
    config = copy.deepcopy(config)
    config["train_options"]["flow_options"] = _through_real_normalize_order(
        config["train_options"]["flow_options"]
    )
    return config


# ---------------------------------------------------------------------------
# 1. Unit coverage of the canonicalizer itself (fast, no dargs involved).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "output_space",
    [
        "uureal_block_ode",
        "spatial_uureal_residual_block_ode",
        "uureal_residual_block_ode",
        "residual_ao_block_ode",
        "residual-ao-block-ode",
        "RESIDUAL_AO_BLOCK_ODE",
    ],
)
def test_canonicalize_injects_0_15_for_omitted_t0_probability(output_space):
    out = canonicalize_flow_options({"output_space": output_space, "block_ode": True})
    assert out["t0_probability"] == pytest.approx(0.15)


@pytest.mark.parametrize("output_space", ["ao_block_ode", "block_ode", "ao_blocks_ode"])
def test_canonicalize_does_not_inject_for_generic_block_ode(output_space):
    out = canonicalize_flow_options({"output_space": output_space, "block_ode": True})
    assert "t0_probability" not in out


@pytest.mark.parametrize("options", [{"output_space": "rme"}, {"output_space": "ao_block"}, {}])
def test_canonicalize_does_not_inject_for_non_block_ode(options):
    out = canonicalize_flow_options(options)
    assert "t0_probability" not in out


@pytest.mark.parametrize("explicit", [0.0, -0.1, 1.0, 0.2])
def test_canonicalize_never_overrides_an_explicit_value(explicit):
    out = canonicalize_flow_options(
        {"output_space": "uureal_block_ode", "block_ode": True, "t0_probability": explicit}
    )
    assert out["t0_probability"] == explicit


# ---------------------------------------------------------------------------
# 2. The actual bug repro: real normalize() ordering (canonicalize -> dargs).
#    These are the load-bearing regression tests for this fix.
# ---------------------------------------------------------------------------
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
    """The plain (non-uureal, non-residual) block_ode route is unaffected: it
    never required a positive t0_probability, and still gets dargs' 0.0."""
    cfg = _valid_contract()
    assert "t0_probability" not in cfg["train_options"]["flow_options"]
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == 0.0
    assert validate_block_ode_contract(normalized) is None


def test_non_block_ode_omitted_t0_probability_keeps_dargs_zero_default():
    raw = {"enabled": True, "prior": "zero", "output_space": "rme"}
    normalized = _through_real_normalize_order(raw)
    assert normalized["t0_probability"] == 0.0


# ---------------------------------------------------------------------------
# 3. Explicit non-positive values still fail closed through the SAME real
#    normalize() ordering (Option A promise: omission != explicit zero).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Explicit positive values still pass through the real normalize() order
#    unchanged (sanity: the canonicalizer only ever fills in an absent key).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("explicit", [0.1, 0.15, 0.25, 0.999])
def test_uureal_explicit_positive_t0_probability_unchanged(explicit):
    cfg = _uureal_config(t0_probability=explicit)
    normalized = _config_with_normalized_flow(cfg)
    assert normalized["train_options"]["flow_options"]["t0_probability"] == pytest.approx(explicit)
    assert validate_block_ode_contract(normalized) is None
