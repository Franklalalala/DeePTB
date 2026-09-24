"""HamilBlockwiseNexTHamLoss: optimization modes, endpoint metric space, and the
log_feature_compatible_interval throttle on the logging-only feature metric.

The interval throttles ONLY the logging-only feature-compatible onsite/hopping
metric (the host-sync-heavy path: cpu()/tolist(), Python species/pair grouping,
two collective all-reduces); it never touches the optimization/gradient loss.
When feature-compatible logging is enabled but throttled, all feature side
effects are None and feature raw sums are absent from last_component_stats.
"""
from __future__ import annotations

import pytest
import torch

from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.nnops.loss import Loss
from dptb.nnops.trainer import Trainer

BASIS = {"H": "1s", "O": "1s1p"}


def _data() -> dict:
    """Fresh H-O block payload: zero predictions against nonzero target blocks."""
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
        "node_hamil_blocks": pred_node, "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]), "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node, "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }


def _criterion(**kwargs) -> HamilBlockwiseNexTHamLoss:
    opts = dict(basis=BASIS, optimization="block_mae", block_reduction="global")
    opts.update(kwargs)
    return HamilBlockwiseNexTHamLoss(**opts)


def _feature_state(crit: HamilBlockwiseNexTHamLoss):
    return (crit.last_feature_compat_loss, crit.last_onsite_loss, crit.last_hopping_loss, crit.last_feature_count)


def _feature_absent(crit: HamilBlockwiseNexTHamLoss) -> bool:
    return all(v is None for v in _feature_state(crit))


def _feature_present(crit: HamilBlockwiseNexTHamLoss) -> bool:
    return all(v is not None for v in _feature_state(crit))


# ---------------------------------------------------------------------------
# registration and optimization modes
# ---------------------------------------------------------------------------
def test_hamil_blockwise_nextham_and_hamil_block_abs_construct_through_loss_registry():
    for method in ("hamil_blockwise_nextham", "hamil_block_abs"):
        loss_fn = Loss(method=method, basis=BASIS, optimization="block_mae", block_reduction="global")
        assert isinstance(loss_fn, HamilBlockwiseNexTHamLoss)


def test_blockwise_default_endpoint_stays_in_block_space_without_rme_walk(monkeypatch):
    import dptb.nnops.blockwise_nextham_loss as blockwise_mod

    monkeypatch.setattr(
        blockwise_mod, "feature_components_from_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RME-compatible slice walk must be opt-in")
        ),
    )
    loss_fn = blockwise_mod.HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"}, optimization="block_l1_rmse",
        block_reduction="equal_onsite_hopping", loss_weight=10.0,
    )
    data = {
        "node_hamil_blocks": torch.zeros(2, 1, 1), "edge_hamil_blocks": torch.zeros(2, 1, 1),
        "node_delta_hamil_blocks": torch.tensor([[[1.0]], [[3.0]]]),
        "edge_delta_hamil_blocks": torch.tensor([[[2.0]], [[4.0]]]),
        "node_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "edge_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "atomic_numbers": torch.ones(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }

    opt_loss = loss_fn(data, data)
    onsite = 0.5 * (2.0 + 5.0 ** 0.5)
    hopping = 0.5 * (3.0 + 10.0 ** 0.5)
    endpoint = 0.5 * (onsite + hopping)

    assert opt_loss.item() == pytest.approx(10.0 * endpoint)
    assert loss_fn.last_endpoint_loss.item() == pytest.approx(endpoint)
    assert loss_fn.last_onsite_loss.item() == pytest.approx(onsite)
    assert loss_fn.last_hopping_loss.item() == pytest.approx(hopping)
    assert loss_fn.last_endpoint_metric_space == "block"
    assert loss_fn.last_feature_compat_loss is None

    reduced = loss_fn.compatible_loss_from_stats(
        onsite_l1_sum=loss_fn.last_onsite_l1_sum, onsite_mse_sum=loss_fn.last_onsite_mse_sum,
        onsite_count=loss_fn.last_onsite_count, hopping_l1_sum=loss_fn.last_hopping_l1_sum,
        hopping_mse_sum=loss_fn.last_hopping_mse_sum, hopping_count=loss_fn.last_hopping_count,
    )
    assert tuple(value.item() for value in reduced) == pytest.approx((endpoint, onsite, hopping))


def test_feature_optimization_keeps_components_differentiable_with_distributed_log_reduce(monkeypatch):
    import dptb.nnops.blockwise_nextham_loss as blockwise_mod

    monkeypatch.setattr(
        blockwise_mod, "maybe_all_reduce_components",
        lambda comp: (_ for _ in ()).throw(
            AssertionError("feature optimization components must keep gradients")
        ),
    )
    loss_fn = blockwise_mod.HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"}, optimization="feature", distributed_log_reduce=True,
    )
    pred_node = torch.zeros(2, 1, 1, requires_grad=True)
    pred_edge = torch.zeros(2, 1, 1, requires_grad=True)
    data = {
        "node_hamil_blocks": pred_node, "edge_hamil_blocks": pred_edge,
        "node_delta_hamil_blocks": torch.tensor([[[1.0]], [[3.0]]]),
        "edge_delta_hamil_blocks": torch.tensor([[[2.0]], [[4.0]]]),
        "node_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "edge_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "atomic_numbers": torch.ones(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }

    loss = loss_fn(data, data)
    loss.backward()

    assert pred_node.grad is not None
    assert pred_edge.grad is not None
    assert pred_node.grad.abs().sum().item() > 0.0
    assert pred_edge.grad.abs().sum().item() > 0.0
    assert loss_fn.last_endpoint_metric_space == "rme"


def test_hamil_blockwise_mae_mse_optimization_mode():
    loss_fn = _criterion(optimization="block_mae_mse", log_feature_compatible=False)
    data = _data()
    out = loss_fn(data, data)
    # Active entries: node 1 + 16, edge 4 + 4 = 25 total.
    abs_sum = 0.5 + 16 * 0.25 + 4 * 1.0 + 4 * 1.0
    sq_sum = 0.25 + 16 * 0.0625 + 4 * 1.0 + 4 * 1.0
    expected = abs_sum / 25.0 + sq_sum / 25.0
    assert torch.isfinite(out)
    assert abs(out.item() - expected) < 1e-6
    endpoint_onsite = 0.5 * (4.5 / 17.0 + (1.25 / 17.0 + loss_fn.eps) ** 0.5)
    endpoint_hopping = 0.5 * (1.0 + (1.0 + loss_fn.eps) ** 0.5)
    assert loss_fn.last_endpoint_loss.item() == pytest.approx(0.5 * (endpoint_onsite + endpoint_hopping))
    assert loss_fn.last_endpoint_loss.item() != pytest.approx(out.item())
    assert loss_fn.last_endpoint_metric_space == "block"

    weighted_loss_fn = _criterion(optimization="block_mae_mse", log_feature_compatible=False, loss_weight=10.0)
    weighted = weighted_loss_fn(data, data)
    assert abs(weighted.item() - 10.0 * expected) < 1e-6
    assert abs(weighted_loss_fn.last_block_loss.item() - expected) < 1e-6
    assert abs(weighted_loss_fn.last_opt_loss.item() - 10.0 * expected) < 1e-6


def test_blockwise_stats_reducer_matches_feature_endpoint_triplet():
    loss_fn = _criterion(basis={"H": "1s"}, log_feature_compatible=True)
    total, onsite, hopping = loss_fn.compatible_loss_from_stats(
        onsite_l1_sum=torch.tensor(4.0), onsite_mse_sum=torch.tensor(10.0), onsite_count=torch.tensor(2.0),
        hopping_l1_sum=torch.tensor(3.0), hopping_mse_sum=torch.tensor(9.0), hopping_count=torch.tensor(3.0),
    )

    expected_onsite = 0.5 * (2.0 + (5.0 + loss_fn.eps) ** 0.5)
    expected_hopping = 0.5 * (1.0 + (3.0 + loss_fn.eps) ** 0.5)
    assert onsite.item() == pytest.approx(expected_onsite)
    assert hopping.item() == pytest.approx(expected_hopping)
    assert total.item() == pytest.approx(0.5 * (expected_onsite + expected_hopping))


# ---------------------------------------------------------------------------
# log_feature_compatible_interval: gates only the logging-only feature metric
# ---------------------------------------------------------------------------
def test_feature_compatible_default_and_explicit_false_both_skip_the_feature_path():
    """log_feature_compatible defaults to False; explicit False is the same frozen
    path. Either way, every call leaves the feature metric absent while onsite/
    hopping mirror the block endpoint and the optimization/block loss stay present."""
    reference = _criterion(log_feature_compatible=True)
    reference(_data())
    ref_compat, ref_onsite, ref_hopping, ref_count = _feature_state(reference)
    # Reference anchor: a real, non-trivial metric was computed when enabled.
    assert ref_count.item() > 0.0 and ref_count.item() == pytest.approx(round(ref_count.item()))
    assert torch.isfinite(ref_compat) and ref_compat.item() > 0.0
    assert torch.isfinite(ref_onsite) and ref_onsite.item() > 0.0
    assert torch.isfinite(ref_hopping) and ref_hopping.item() > 0.0

    for crit in (_criterion(), _criterion(log_feature_compatible=False)):
        assert crit.log_feature_compatible_interval == 1
        assert crit.log_feature_compatible is False
        for _ in range(3):
            loss = crit(_data())
            assert torch.isfinite(loss)
            # Only the feature-specific fields are absent; onsite/hopping keep
            # mirroring the block endpoint (checked below), unlike a throttled
            # interval step where the whole triplet goes sparse (see the
            # interval-throttle test below).
            assert crit.last_feature_compat_loss is None
            assert crit.last_feature_count is None
            assert torch.equal(crit.last_onsite_loss, crit.last_block_onsite_loss)
            assert torch.equal(crit.last_hopping_loss, crit.last_block_hopping_loss)
            assert "feature_onsite_abs_sum" not in crit.last_component_stats
            assert crit.last_block_loss is not None
            assert crit.last_opt_loss is not None


def test_interval_throttle_fires_on_the_grid_and_matches_an_untouched_reference():
    """interval=3 fires on calls 1, 4, 7 (indices 0, 3, 6); non-firing calls clear
    the feature state and stats exactly like the disabled path, and firing calls
    are byte-identical to an interval=1 criterion on the same inputs."""
    ref = _criterion(log_feature_compatible=True, log_feature_compatible_interval=1)
    ref(_data())
    ref_state = _feature_state(ref)

    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=3)
    fired = []
    for _ in range(7):
        crit(_data())
        # Block-level logging state is NEVER gated -- present on every call.
        assert crit.last_block_loss is not None
        assert crit.last_block_onsite_loss is not None
        assert crit.last_block_hopping_loss is not None
        present = _feature_present(crit)
        fired.append(present)
        if present:
            for got, expected in zip(_feature_state(crit), ref_state):
                assert torch.equal(got, expected)
            assert "feature_onsite_abs_sum" in crit.last_component_stats
            assert "feature_total_count" in crit.last_component_stats
        else:
            assert _feature_absent(crit)
            assert "feature_onsite_abs_sum" not in crit.last_component_stats
            assert "feature_hopping_abs_sum" not in crit.last_component_stats
            assert "feature_total_count" not in crit.last_component_stats

    assert fired == [True, False, False, True, False, False, True]


def test_non_firing_step_is_an_explicit_sparse_endpoint_triplet():
    crit = _criterion(log_feature_compatible=True, log_feature_compatible_interval=2)
    crit(_data())  # firing call
    optimization_loss = crit(_data())  # throttled call

    state = Trainer._endpoint_loss_state(crit, optimization_loss, prefix="train")
    assert state["train_loss"] is None
    assert state["train_onsite_loss"] is None
    assert state["train_hopping_loss"] is None
    torch.testing.assert_close(state["train_loss_opt"], optimization_loss.detach())
    # The fail-closed API still sees a criterion that implemented all three
    # endpoint fields; per-key accumulators then omit this sparse sample.
    Trainer._require_endpoint_triplet(state, prefix="train", route="cadence regression")
    assert "block_onsite_abs_sum" in crit.last_component_stats


@pytest.mark.parametrize("bad", [0, -1, -5, True, False, 2.5])
def test_invalid_interval_raises(bad):
    with pytest.raises(ValueError, match="log_feature_compatible_interval"):
        _criterion(log_feature_compatible_interval=bad)


def test_optimization_feature_mode_not_gated_by_interval():
    crit = _criterion(
        optimization="feature_compatible", log_feature_compatible=False, log_feature_compatible_interval=5,
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
