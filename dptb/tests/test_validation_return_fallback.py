# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression for Trainer.validation()'s fail-closed return selection.

FINDING A (pre-existing): ``Trainer.validation()`` initializes ``loss = 0`` and,
when the endpoint-compatible pass never wrote the legacy ``validation_loss`` key,
returned that accumulated 0.0.  This is reachable with the LEGAL config
``validation_ode_steps=[3]`` together with ``log_validation_random_t_loss`` /
``log_validation_t0_loss`` / ``log_validation_flow_euler_loss`` all false: only the
``num_steps == 1`` pass writes ``validation_loss``, so with steps == [3] the state
carries only ``validation_compatible_euler_3_loss`` and any scheduler /
best-checkpoint consumer then sees a perfect 0.0.

FIX A extracts the return-tail selection into ``Trainer._resolve_validation_return``
which fails closed: prefer the legacy ``validation_loss`` (byte-identical to the
historical behavior), else the SMALLEST-n ``validation_compatible_euler_{n}_loss``,
else the accumulated ``loss``.  These tests drive that helper directly with the
reviewer's minimal-state repro shape (mirroring test_trainer_reference_batches.py's
``Trainer.__new__`` stub style, without editing that file).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dptb.nnops.trainer import Trainer
from dptb.nnops.multi_trainer import MultiTrainer


def _trainer_with_state(state):
    trainer = Trainer.__new__(Trainer)
    trainer._last_flow_validation_state = state
    return trainer


# ---------------------------------------------------------------------------
# (1) The dead combo: only validation_compatible_euler_3_loss present (no legacy
#     key) must return that scalar (7.0), NOT the accumulated 0.0.
# ---------------------------------------------------------------------------
def test_euler_compatible_loss_used_when_legacy_absent():
    trainer = _trainer_with_state({"validation_compatible_euler_3_loss": torch.tensor(7.0)})
    accumulated = torch.tensor(0.0)  # the dead-0 that the base bug returned

    result = trainer._resolve_validation_return(accumulated)

    assert torch.is_tensor(result)
    assert result.item() == pytest.approx(7.0)
    assert result is not accumulated


# ---------------------------------------------------------------------------
# (2) Byte-identical when the legacy key exists: return it even if euler keys are
#     also present.
# ---------------------------------------------------------------------------
def test_legacy_validation_loss_takes_precedence():
    legacy = torch.tensor(5.0)
    trainer = _trainer_with_state(
        {
            "validation_loss": legacy,
            "validation_compatible_euler_1_loss": torch.tensor(2.0),
            "validation_compatible_euler_3_loss": torch.tensor(7.0),
        }
    )

    result = trainer._resolve_validation_return(torch.tensor(0.0))

    assert result is legacy  # exact object identity -> byte-identical to base


# ---------------------------------------------------------------------------
# (3) Neither legacy nor any euler-compatible key -> fall through to the
#     accumulated loss (unchanged object).
# ---------------------------------------------------------------------------
def test_falls_through_to_accumulated_loss_when_no_compatible_key():
    accumulated = torch.tensor(3.3)
    trainer = _trainer_with_state({"validation_flow_random_t_loss": torch.tensor(9.0)})

    result = trainer._resolve_validation_return(accumulated)

    assert result is accumulated


def test_empty_or_missing_state_returns_accumulated_loss():
    accumulated = torch.tensor(1.25)
    # empty dict
    assert _trainer_with_state({})._resolve_validation_return(accumulated) is accumulated
    # attribute entirely missing (getattr fallback)
    bare = Trainer.__new__(Trainer)
    assert bare._resolve_validation_return(accumulated) is accumulated


# ---------------------------------------------------------------------------
# (4) Multiple euler keys -> the SMALLEST n wins (1-step endpoint is the closest
#     no-CFM/CFM-comparable scalar).
# ---------------------------------------------------------------------------
def test_smallest_num_steps_euler_key_is_selected():
    trainer = _trainer_with_state(
        {
            "validation_compatible_euler_5_loss": torch.tensor(50.0),
            "validation_compatible_euler_2_loss": torch.tensor(20.0),
            "validation_compatible_euler_10_loss": torch.tensor(100.0),
        }
    )

    result = trainer._resolve_validation_return(torch.tensor(0.0))

    # numeric ordering (2 < 5 < 10), not lexical ("10" < "2" < "5")
    assert result.item() == pytest.approx(20.0)


def test_non_numeric_euler_suffix_is_ignored():
    # Only well-formed validation_compatible_euler_<int>_loss keys are eligible.
    trainer = _trainer_with_state(
        {
            "validation_compatible_euler_x_loss": torch.tensor(99.0),
            "validation_compatible_euler_4_loss": torch.tensor(4.0),
        }
    )

    result = trainer._resolve_validation_return(torch.tensor(0.0))

    assert result.item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# (5) MultiTrainer decision (documentation-as-test): MultiTrainer.validation()
#     does NOT route its return through this helper -- it returns the accumulated
#     ``total_loss`` where each per-batch ``loss_i`` is already selected inline via
#     ``state.get("validation_loss", state.get("validation_compatible_euler_{n}_loss"))``
#     (multi_trainer.py, the flow-euler validation branch), i.e. the Fix-A
#     preference is already inlined there and only degrades to a scalar 0 when the
#     euler pack has no active nodes (a genuinely empty batch), never for the
#     legal validation_ode_steps=[3] + log-flags-false config on real data.  The
#     helper is still inherited, so it stays available and correct for the
#     subclass if that path is ever refactored to use it.
# ---------------------------------------------------------------------------
def test_multitrainer_inherits_resolver_and_matches_trainer_semantics():
    assert MultiTrainer._resolve_validation_return is Trainer._resolve_validation_return

    mt = MultiTrainer.__new__(MultiTrainer)
    mt._last_flow_validation_state = {"validation_compatible_euler_3_loss": torch.tensor(7.0)}
    assert mt._resolve_validation_return(torch.tensor(0.0)).item() == pytest.approx(7.0)
