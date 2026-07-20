# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for HamilBlockwiseNexTHamLoss.log_feature_compatible_interval.

The interval throttles ONLY the logging-only feature-compatible onsite/hopping
metric (the host-sync-heavy path: cpu()/tolist(), Python species/pair grouping,
two collective all-reduces).  It never touches the optimization/gradient loss.

Design choice for non-firing steps: the feature side-effect attributes are set to
None exactly as they are when log_feature_compatible=false (last_feature_compat_loss,
last_onsite_loss, last_hopping_loss, last_feature_count) and the feature_* raw
component sums are simply absent from last_component_stats.  Every trainer/multi_trainer
consumer reads these through getattr(..., default)/dict.get(...) (trainer.py:401-402
via _loss_component_state only inserts when not None; multi_trainer.py:2003-2004 via
_as_scalar_tensor(..., default=0.0) which maps None->default), so absence/None can
never KeyError a consumer and no carry-forward is needed.
"""

from __future__ import annotations

import pytest
import torch

from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss


BASIS = {"H": "1s", "O": "1s1p"}
# The feature-compatible count is the mapper's reduced-matrix-element feature
# slice count (RME-derived), which is deterministic but distinct from the block
# active-entry count.  Tests capture it dynamically from a reference criterion
# rather than hardcoding it, so they assert gating behavior + cross-criterion
# byte-identity instead of a brittle magic number.


def _data() -> dict:
    """Fresh synthetic block payload (mirrors test_blockwise_qhflow_lr)."""
    max_norb = 4  # 1s1p union
    pred_node = torch.zeros(2, max_norb, max_norb)
    target_node = torch.zeros(2, max_norb, max_norb)
    target_node[0, 0, 0] = 0.5  # H onsite 1x1
    target_node[1, :4, :4] = 0.25  # O onsite 4x4
    pred_edge = torch.zeros(2, max_norb, max_norb)
    target_edge = torch.zeros(2, max_norb, max_norb)
    target_edge[0, :1, :4] = 1.0  # H->O 1x4
    target_edge[1, :4, :1] = -1.0  # O->H 4x1
    return {
        "node_hamil_blocks": pred_node,
        "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]),
        "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node,
        "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }


def _criterion(**kwargs) -> HamilBlockwiseNexTHamLoss:
    opts = dict(basis=BASIS, optimization="block_mae", block_reduction="global")
    opts.update(kwargs)
    return HamilBlockwiseNexTHamLoss(**opts)


def _feature_state(crit: HamilBlockwiseNexTHamLoss):
    return (
        crit.last_feature_compat_loss,
        crit.last_onsite_loss,
        crit.last_hopping_loss,
        crit.last_feature_count,
    )


def _feature_absent(crit: HamilBlockwiseNexTHamLoss) -> bool:
    compat, onsite, hopping, count = _feature_state(crit)
    return compat is None and onsite is None and hopping is None and count is None


def _feature_present(crit: HamilBlockwiseNexTHamLoss) -> bool:
    return all(v is not None for v in _feature_state(crit))


# ---------------------------------------------------------------------------
# (a) default interval == 1: present every call, byte-identical to an explicit
#     log_feature_compatible=true reference on the same inputs.
# ---------------------------------------------------------------------------
def test_default_interval_is_one_and_matches_explicit_reference():
    default = _criterion()
    assert default.log_feature_compatible_interval == 1

    reference = _criterion(log_feature_compatible=True)
    reference(_data())
    ref_compat, ref_onsite, ref_hopping, ref_count = _feature_state(reference)

    # Reference anchors: a real, non-trivial metric was computed.
    assert ref_count.item() > 0.0
    assert ref_count.item() == pytest.approx(round(ref_count.item()))  # integral count
    assert torch.isfinite(ref_compat) and ref_compat.item() > 0.0
    assert torch.isfinite(ref_onsite) and ref_onsite.item() > 0.0
    assert torch.isfinite(ref_hopping) and ref_hopping.item() > 0.0

    # Default criterion (interval=1) fires on EVERY call, byte-identical values.
    for _ in range(3):
        default(_data())
        assert _feature_present(default)
        compat, onsite, hopping, count = _feature_state(default)
        assert torch.equal(compat, ref_compat)
        assert torch.equal(onsite, ref_onsite)
        assert torch.equal(hopping, ref_hopping)
        assert torch.equal(count, ref_count)


# ---------------------------------------------------------------------------
# (b) interval == 3: fires on calls 1 and 4 (call_index 0 and 3); absent on 2,3.
# ---------------------------------------------------------------------------
def test_interval_three_fires_on_calls_one_and_four():
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)

    fired = []
    for _ in range(4):
        crit(_data())
        fired.append(_feature_present(crit))
        # Block-level logging state is NEVER gated -- present on every call.
        assert crit.last_block_loss is not None
        assert crit.last_block_onsite_loss is not None
        assert crit.last_block_hopping_loss is not None

    assert fired == [True, False, False, True]

    # Re-run to confirm the counter keeps advancing: call 5 (index 4) no, call 6
    # (index 5) no, call 7 (index 6) fires again.
    more = []
    for _ in range(3):
        crit(_data())
        more.append(_feature_present(crit))
    assert more == [False, False, True]


def test_non_firing_step_clears_feature_state_and_stats_like_disabled():
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)
    crit(_data())  # call 1 fires
    assert _feature_present(crit)
    assert "feature_onsite_abs_sum" in crit.last_component_stats
    assert "feature_total_count" in crit.last_component_stats

    crit(_data())  # call 2 does not fire
    assert _feature_absent(crit)
    # Feature raw component sums are simply absent (same as disabled path);
    # block_* raw sums remain present.
    assert "feature_onsite_abs_sum" not in crit.last_component_stats
    assert "feature_hopping_abs_sum" not in crit.last_component_stats
    assert "feature_total_count" not in crit.last_component_stats
    assert "block_onsite_abs_sum" in crit.last_component_stats


# ---------------------------------------------------------------------------
# (c) firing values under interval=3 equal an interval=1 criterion (same inputs).
# ---------------------------------------------------------------------------
def test_firing_values_identical_across_intervals():
    ref = _criterion(log_feature_compatible=True, log_feature_compatible_interval=1)
    ref(_data())
    ref_state = _feature_state(ref)

    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)
    firing_calls = (0, 3)  # call_index values that fire
    for call_index in range(4):
        crit(_data())
        if call_index in firing_calls:
            for got, expected in zip(_feature_state(crit), ref_state):
                assert torch.equal(got, expected)
        else:
            assert _feature_absent(crit)


# ---------------------------------------------------------------------------
# (d) invalid interval -> fail-closed ValueError (0, negative, bool, non-int).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -5, True, False, 2.5])
def test_invalid_interval_raises(bad):
    with pytest.raises(ValueError, match="log_feature_compatible_interval"):
        _criterion(log_feature_compatible_interval=bad)


# ---------------------------------------------------------------------------
# (e) optimization-mode feature requirement is NOT gated by the interval.
# ---------------------------------------------------------------------------
def test_optimization_feature_mode_not_gated_by_interval():
    crit = _criterion(
        optimization="feature_compatible",
        log_feature_compatible=False,  # logging trigger off ...
        log_feature_compatible_interval=5,  # ... and a large cadence
    )
    # Every call must still compute features because the optimization loss needs
    # them; the interval only ever gates the logging trigger.
    counts = []
    for _ in range(6):
        loss = crit(_data())
        assert torch.isfinite(loss)
        assert _feature_present(crit)
        counts.append(crit.last_feature_count.item())
    assert all(c > 0.0 for c in counts)
    assert len(set(counts)) == 1  # feature count stable across calls (never gated)


# ---------------------------------------------------------------------------
# (f) argcheck accepts the new key (both variants) and defaults to 1.
# ---------------------------------------------------------------------------
def test_argcheck_accepts_interval_and_defaults_to_one():
    from dptb.utils.argcheck import train_options

    def _cfg(method, loss_train):
        return {
            "num_epoch": 1,
            "batch_size": 1,
            "optimizer": {"type": "AdamW", "lr": 1e-3},
            "lr_scheduler": {"type": "rop"},
            "loss_options": {"train": {"method": method, **loss_train}},
        }

    # explicit value, hamil_blockwise_nextham
    cfg = _cfg("hamil_blockwise_nextham", {"log_feature_compatible_interval": 25})
    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    assert normalized["loss_options"]["train"]["log_feature_compatible_interval"] == 25

    # default when omitted
    cfg_default = _cfg("hamil_blockwise_nextham", {})
    normalized_default = train_options().normalize_value(cfg_default)
    train_options().check_value(normalized_default, strict=True)
    assert normalized_default["loss_options"]["train"]["log_feature_compatible_interval"] == 1

    # sibling variant hamil_block_abs shares the same schema
    cfg_abs = _cfg("hamil_block_abs", {"log_feature_compatible_interval": 7})
    normalized_abs = train_options().normalize_value(cfg_abs)
    train_options().check_value(normalized_abs, strict=True)
    assert normalized_abs["loss_options"]["train"]["log_feature_compatible_interval"] == 7


# ---------------------------------------------------------------------------
# (g) log_feature_compatible=false still skips the feature path entirely
#     (frozen behavior; the interval counter has no effect).
# ---------------------------------------------------------------------------
def test_log_feature_compatible_false_skips_feature_path():
    crit = _criterion(log_feature_compatible=False)  # default interval == 1
    for _ in range(3):
        loss = crit(_data())
        assert torch.isfinite(loss)
        assert _feature_absent(crit)
        assert "feature_onsite_abs_sum" not in crit.last_component_stats
        # optimization/block loss is untouched and always present
        assert crit.last_block_loss is not None
        assert crit.last_opt_loss is not None
