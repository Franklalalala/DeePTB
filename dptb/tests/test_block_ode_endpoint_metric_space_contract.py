# SPDX-License-Identifier: LGPL-3.0-or-later
"""P1-2 merge-blocker fix: block_ode + a blockwise criterion that would log a
non-'block' endpoint metric space (log_feature_compatible=true, or
optimization in {feature, feature_compatible, compat}) must fail closed at
configuration time, not crash at the first validation step.

Before this fix, block_ode + log_feature_compatible=true was schema-legal
(dargs never sees train_options.loss_options and train_options.flow_options
together) and log_feature_compatible was even documented as a way to opt into
cross-representation comparison "on a separate validation loss"
(docs/advanced/pixel_meanflow.md) -- yet the combination always crashed once
training reached "one-step" validation:
HamilBlockwiseNexTHamLoss.endpoint_metric_space would be 'rme' (see
dptb.nnops.blockwise_metric_space.endpoint_metric_space_for_options, the
single source of truth shared by the loss module and this validator),
block-ODE flows only ever publish 'block'-space endpoint statistics
(dptb.nnops.flow), Trainer explicitly refuses the direct-criterion recompute
fallback for block_ode (the ``and not block_ode`` guards in
dptb.nnops.trainer._loss_on_batch / the validation loop), and the
'block'/'rme' mismatch left ``compatible_state`` as ``None`` going into the
one-step endpoint-triplet assertion (``_require_endpoint_triplet``).

This suite pins:
  1. dptb.nnops.blockwise_metric_space.endpoint_metric_space_for_options (the
     shared pure function) matches the real HamilBlockwiseNexTHamLoss class
     attribute for every (log_feature_compatible, optimization) combination,
     so the validator's rejection reflects genuine runtime behavior rather
     than a second, possibly-drifted copy of the ternary.
  2. Every loss configuration that would produce a non-'block' endpoint
     (log_feature_compatible=true on train and/or validation; every
     feature-space optimization alias; both loss method names,
     hamil_blockwise_nextham and its hamil_block_abs registration alias) is
     rejected by validate_block_ode_contract with an explanatory message --
     for both the generic ao_block_ode route and residual_ao_block_ode.
  3. Every block-space optimization (the ones actually meant for block_ode)
     stays legal, and the check does not misfire when loss_options is absent
     entirely or configures an unrelated (non-blockwise) loss method.
  4. The identical loss configuration remains completely legal when block_ode
     is NOT active -- this is a cross-tree contract keyed on block_ode, not a
     blanket ban on log_feature_compatible/feature-space optimizations.
"""
from __future__ import annotations

import copy

import pytest

from dptb.nnops.blockwise_metric_space import (
    FEATURE_ENDPOINT_OPTIMIZATION_MODES,
    endpoint_metric_space_for_options,
)
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.utils.argcheck import validate_block_ode_contract

# Reuse the frozen block-ODE contract fixtures from sibling test modules
# (default "prepend" import mode makes them importable by bare name, the same
# pattern dptb/tests/test_t0_probability_bounds.py already relies on).  Do NOT
# edit those files.
from test_block_ode_redteam import _valid_contract  # noqa: E402 -- generic ao_block_ode
from test_residual_ao_block_ode import _load_b_config  # noqa: E402 -- residual_ao_block_ode


BLOCK_SPACE_OPTIMIZATIONS = ["block_mae", "block_l1_rmse", "block_mae_mse"]
FEATURE_SPACE_OPTIMIZATIONS = sorted(FEATURE_ENDPOINT_OPTIMIZATION_MODES)
BLOCKWISE_LOSS_METHODS = ["hamil_blockwise_nextham", "hamil_block_abs"]

MATCH = "endpoint_metric_space"


def _with_loss(config, *, split="train", method="hamil_blockwise_nextham", **loss_fields):
    """Attach a minimal train_options.loss_options.<split> block."""
    config = copy.deepcopy(config)
    loss_options = config["train_options"].setdefault("loss_options", {})
    loss_options[split] = {"method": method, **loss_fields}
    return config


# ---------------------------------------------------------------------------
# 1. Pure function unit coverage + cross-check against the real loss class.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("log_feature_compatible", [False, True])
@pytest.mark.parametrize("optimization", BLOCK_SPACE_OPTIMIZATIONS + FEATURE_SPACE_OPTIMIZATIONS)
def test_pure_function_matches_real_loss_class(optimization, log_feature_compatible):
    expected = endpoint_metric_space_for_options(
        log_feature_compatible=log_feature_compatible, optimization=optimization
    )
    loss = HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"},
        optimization=optimization,
        log_feature_compatible=log_feature_compatible,
    )
    assert loss.endpoint_metric_space == expected


@pytest.mark.parametrize("optimization", FEATURE_SPACE_OPTIMIZATIONS)
def test_pure_function_feature_optimizations_are_rme_even_without_log_flag(optimization):
    assert (
        endpoint_metric_space_for_options(log_feature_compatible=False, optimization=optimization)
        == "rme"
    )


@pytest.mark.parametrize("optimization", BLOCK_SPACE_OPTIMIZATIONS)
def test_pure_function_block_optimizations_stay_block(optimization):
    assert (
        endpoint_metric_space_for_options(log_feature_compatible=False, optimization=optimization)
        == "block"
    )


@pytest.mark.parametrize("optimization", BLOCK_SPACE_OPTIMIZATIONS)
def test_pure_function_log_feature_compatible_forces_rme_regardless_of_optimization(optimization):
    assert (
        endpoint_metric_space_for_options(log_feature_compatible=True, optimization=optimization)
        == "rme"
    )


# ---------------------------------------------------------------------------
# 2. block_ode + log_feature_compatible=true -> configuration-time ValueError.
#    Covers both block-ode variants (generic ao_block_ode, residual_ao_block_ode),
#    both loss method spellings, and both the train and validation splits (the
#    real crash is specifically in "one-step" validation, but train's endpoint
#    contract is equally broken and must fail closed the same way).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", BLOCKWISE_LOSS_METHODS)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_generic_block_ode_rejects_log_feature_compatible(split, method):
    config = _with_loss(_valid_contract(), split=split, method=method, log_feature_compatible=True)
    with pytest.raises(ValueError, match=MATCH):
        validate_block_ode_contract(config)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_residual_block_ode_rejects_log_feature_compatible(split):
    config = _with_loss(_load_b_config(), split=split, log_feature_compatible=True)
    with pytest.raises(ValueError, match=MATCH):
        validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# 3. block_ode + optimization in {feature, feature_compatible, compat} -> the
#    same configuration-time ValueError, WITHOUT log_feature_compatible set
#    (isolating the optimization-only trigger from the log-flag trigger).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("optimization", FEATURE_SPACE_OPTIMIZATIONS)
def test_generic_block_ode_rejects_feature_space_optimization(optimization):
    config = _with_loss(_valid_contract(), optimization=optimization)
    with pytest.raises(ValueError, match=MATCH):
        validate_block_ode_contract(config)


@pytest.mark.parametrize("optimization", FEATURE_SPACE_OPTIMIZATIONS)
def test_residual_block_ode_rejects_feature_space_optimization(optimization):
    config = _with_loss(_load_b_config(), optimization=optimization)
    with pytest.raises(ValueError, match=MATCH):
        validate_block_ode_contract(config)


# ---------------------------------------------------------------------------
# 4. block_ode + block-space optimization (log_feature_compatible=false) ->
#    stays legal.  Also: the check must not misfire when loss_options is
#    absent, or configures an unrelated (non-blockwise) loss method.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("optimization", BLOCK_SPACE_OPTIMIZATIONS)
def test_generic_block_ode_accepts_block_space_optimization(optimization):
    config = _with_loss(_valid_contract(), optimization=optimization, log_feature_compatible=False)
    assert validate_block_ode_contract(config) is None


@pytest.mark.parametrize("optimization", BLOCK_SPACE_OPTIMIZATIONS)
def test_residual_block_ode_accepts_block_space_optimization(optimization):
    config = _with_loss(_load_b_config(), optimization=optimization, log_feature_compatible=False)
    assert validate_block_ode_contract(config) is None


def test_generic_block_ode_accepts_missing_loss_options_entirely():
    """The check must not require loss_options to be configured at all --
    many other tests in this suite validate flow_options in isolation."""
    assert validate_block_ode_contract(_valid_contract()) is None


def test_generic_block_ode_ignores_non_blockwise_loss_methods():
    config = _with_loss(_valid_contract(), method="hamil_abs", log_feature_compatible=True)
    # hamil_abs has no endpoint_metric_space concept; this block-ODE-specific
    # check must not fire for it (dargs and the rest of the loss schema own
    # that method's own field legality, not this cross-tree check).
    assert validate_block_ode_contract(config) is None


# ---------------------------------------------------------------------------
# 5. Not misfiring when block_ode is inactive: log_feature_compatible=true
#    (and every feature-space optimization) remains completely legal for a
#    non-block-ode flow -- this is a cross-tree contract keyed on block_ode
#    being requested, not a blanket ban on the loss fields themselves.
# ---------------------------------------------------------------------------
def _non_block_ode_config(**loss_fields):
    return {
        "train_options": {
            "flow_options": {"enabled": True, "prior": "zero", "output_space": "rme"},
            "loss_options": {"train": {"method": "hamil_blockwise_nextham", **loss_fields}},
        },
    }


def test_non_block_ode_flow_still_allows_log_feature_compatible():
    config = _non_block_ode_config(log_feature_compatible=True)
    assert validate_block_ode_contract(config) is None


@pytest.mark.parametrize("optimization", FEATURE_SPACE_OPTIMIZATIONS)
def test_non_block_ode_flow_still_allows_feature_space_optimization(optimization):
    config = _non_block_ode_config(optimization=optimization)
    assert validate_block_ode_contract(config) is None


def test_no_flow_options_at_all_still_allows_log_feature_compatible():
    """No train_options.flow_options -> block_ode cannot be requested."""
    config = {
        "train_options": {
            "loss_options": {
                "train": {"method": "hamil_blockwise_nextham", "log_feature_compatible": True}
            },
        },
    }
    assert validate_block_ode_contract(config) is None
