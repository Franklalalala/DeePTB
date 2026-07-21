"""Golden tests for :class:`dptb.nnops.metric_reducer.MetricReducer`.

These pin the *numerical* behaviour of the pure pack-reduction functions that
were lifted out of ``MultiTrainer`` so a future edit cannot silently drift the
displayed loss/onsite/hopping arithmetic. They exercise every branch:

* ``compatible_state_from_pack`` — stats path, RMS fallback, active weighted
  fallback, onsite-boost, z-loss, and the not-supported / empty short-circuits.
* ``component_state_from_pack`` — compatible path + active fallback + empty.
* ``stitched_loss_reduce`` — stats path, RMS fallback, and the no-data ``None``.
* ``display_state_from_packs`` — the full reduced-pack -> display-dict assembly.
* ``safe_mean`` / ``maybe_call_or_value`` scalar helpers.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.metric_pack import MetricPack, DynamicBatchStat, ExpertDisplayMetric
from dptb.nnops.metric_reducer import MetricReducer


DTYPE = torch.float32
DEVICE = torch.device("cpu")


def _pack(**fields) -> torch.Tensor:
    return MetricPack(**fields).to_tensor(dtype=DTYPE, device=DEVICE)


def _db_pack(**fields) -> torch.Tensor:
    return DynamicBatchStat(**fields).to_tensor(dtype=DTYPE, device=DEVICE)


class _StatsLoss:
    """Loss double exposing ``compatible_loss_from_stats`` (the reduce path)."""

    onsite_boost = False
    z_loss_coef = 0.0

    def __init__(self):
        self.calls = 0
        self.last_global_step = "unset"
        self.last_z_loss = "unset"

    def compatible_loss_from_stats(
        self,
        *,
        onsite_l1_sum,
        onsite_mse_sum,
        onsite_count,
        hopping_l1_sum,
        hopping_mse_sum,
        hopping_count,
        z_loss=None,
        global_step=None,
    ):
        self.calls += 1
        self.last_global_step = global_step
        self.last_z_loss = z_loss
        onsite = 0.5 * (
            onsite_l1_sum / onsite_count.clamp_min(1.0)
            + torch.sqrt(onsite_mse_sum / onsite_count.clamp_min(1.0) + 1e-12)
        )
        hopping = 0.5 * (
            hopping_l1_sum / hopping_count.clamp_min(1.0)
            + torch.sqrt(hopping_mse_sum / hopping_count.clamp_min(1.0) + 1e-12)
        )
        return 0.5 * (onsite + hopping), onsite, hopping


def _plain_loss(**overrides):
    """Loss double *without* ``compatible_loss_from_stats`` -> RMS fallback."""
    base = dict(onsite_boost=False, z_loss_coef=0.0, _current_onsite_weight=None)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# scalar helpers
# --------------------------------------------------------------------------

def test_safe_mean_divides_and_clamps():
    out = MetricReducer.safe_mean(torch.tensor(6.0), torch.tensor(3.0), dtype=DTYPE, device=DEVICE)
    assert out.item() == pytest.approx(2.0)
    # count clamps to 1.0 -> returns the sum
    clamped = MetricReducer.safe_mean(torch.tensor(5.0), torch.tensor(0.0), dtype=DTYPE, device=DEVICE)
    assert clamped.item() == pytest.approx(5.0)


def test_safe_mean_none_returns_zero():
    assert MetricReducer.safe_mean(None, torch.tensor(3.0), dtype=DTYPE, device=DEVICE).item() == 0.0
    assert MetricReducer.safe_mean(torch.tensor(3.0), None, dtype=DTYPE, device=DEVICE).item() == 0.0


def test_maybe_call_or_value():
    assert MetricReducer.maybe_call_or_value(None, default=1.5) == 1.5
    assert MetricReducer.maybe_call_or_value(2.0) == 2.0
    assert MetricReducer.maybe_call_or_value(lambda: 3.0) == 3.0
    # callable that raises falls back to default
    def _boom():
        raise RuntimeError
    assert MetricReducer.maybe_call_or_value(_boom, default=7.0) == 7.0


# --------------------------------------------------------------------------
# compatible_state_from_pack
# --------------------------------------------------------------------------

def test_compatible_state_not_supported_returns_none():
    assert MetricReducer.compatible_state_from_pack(
        _pack(onsite_cnt_sum=1.0),
        loss_module=_plain_loss(),
        supports_triplet=False,
        dtype=DTYPE,
        device=DEVICE,
    ) is None


def test_compatible_state_rms_fallback_golden():
    pack = _pack(
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
    )
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True,
        dtype=DTYPE, device=DEVICE, prefix="train",
    )
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(16.0 / 2.0))   # 1 + sqrt(2)
    hopping = 0.5 * (3.0 / 3.0 + math.sqrt(36.0 / 3.0))  # 0.5 + sqrt(3)
    total = 0.5 * (onsite + hopping)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite, rel=1e-6)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping, rel=1e-6)
    assert state["train_loss"].item() == pytest.approx(total, rel=1e-6)


def test_compatible_state_onsite_boost_and_zloss():
    pack = _pack(
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
        z_sum=6.0, z_cnt=3.0,
    )
    loss = _plain_loss(onsite_boost=True, _current_onsite_weight=2.0, z_loss_coef=0.5)
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=loss, supports_triplet=True, dtype=DTYPE, device=DEVICE,
    )
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(16.0 / 2.0))
    hopping = 0.5 * (3.0 / 3.0 + math.sqrt(36.0 / 3.0))
    total = 2.0 * onsite + hopping + 0.5 * (6.0 / 3.0)
    assert state["train_loss"].item() == pytest.approx(total, rel=1e-6)


def test_compatible_state_active_weighted_fallback():
    # No l1/mse counts -> uses active-node/edge weighted means.
    pack = _pack(
        onsite_weighted_sum=10.0, active_nodes_sum=4.0,
        hopping_weighted_sum=6.0, active_edges_sum=2.0,
    )
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE,
    )
    assert state["train_onsite_loss"].item() == pytest.approx(2.5)
    assert state["train_hopping_loss"].item() == pytest.approx(3.0)
    assert state["train_loss"].item() == pytest.approx(0.5 * (2.5 + 3.0))


def test_compatible_state_empty_pack_returns_none():
    assert MetricReducer.compatible_state_from_pack(
        _pack(), loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE,
    ) is None


def test_compatible_state_stats_path_forwards_global_step_and_zloss():
    pack = _pack(
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
        z_sum=8.0, z_cnt=4.0,
    )
    loss = _StatsLoss()
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=loss, supports_triplet=True, dtype=DTYPE, device=DEVICE,
        prefix="val", global_step=42,
    )
    assert loss.calls == 1
    assert loss.last_global_step == 42
    assert loss.last_z_loss.item() == pytest.approx(2.0)  # z_sum / z_cnt
    assert set(state) == {"val_loss", "val_onsite_loss", "val_hopping_loss"}


# --------------------------------------------------------------------------
# component_state_from_pack
# --------------------------------------------------------------------------

def test_component_state_empty_when_not_supported():
    assert MetricReducer.component_state_from_pack(
        _pack(onsite_cnt_sum=1.0), loss_module=_plain_loss(), supports_triplet=False,
        dtype=DTYPE, device=DEVICE, prefix="validation",
    ) == {}


def test_component_state_uses_compatible_when_available():
    pack = _pack(
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
        # conflicting active means that must be IGNORED in favour of stats
        onsite_weighted_sum=999.0, active_nodes_sum=1.0,
        hopping_weighted_sum=999.0, active_edges_sum=1.0,
    )
    out = MetricReducer.component_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True,
        dtype=DTYPE, device=DEVICE, prefix="validation",
    )
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(16.0 / 2.0))
    hopping = 0.5 * (3.0 / 3.0 + math.sqrt(36.0 / 3.0))
    assert out["validation_onsite_loss"].item() == pytest.approx(onsite, rel=1e-6)
    assert out["validation_hopping_loss"].item() == pytest.approx(hopping, rel=1e-6)


def test_component_state_active_fallback_when_compatible_none():
    # Empty stats + zero active -> compatible returns None, then component falls
    # back to weighted/active (with clamp), so an all-zero pack yields zeros.
    pack = _pack(onsite_weighted_sum=8.0, active_nodes_sum=0.0,
                 hopping_weighted_sum=4.0, active_edges_sum=0.0)
    out = MetricReducer.component_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True,
        dtype=DTYPE, device=DEVICE, prefix="train",
    )
    # active counts clamp to 1.0
    assert out["train_onsite_loss"].item() == pytest.approx(8.0)
    assert out["train_hopping_loss"].item() == pytest.approx(4.0)


def test_component_state_omits_cadence_skipped_zero_over_zero_pack():
    out = MetricReducer.component_state_from_pack(
        _pack(loss_opt_sum=3.0, step_count=1.0),
        loss_module=_plain_loss(),
        supports_triplet=True,
        dtype=DTYPE,
        device=DEVICE,
        prefix="train",
    )
    assert out == {}


# --------------------------------------------------------------------------
# stitched_loss_reduce
# --------------------------------------------------------------------------

def test_stitched_loss_none_when_no_stats():
    assert MetricReducer.stitched_loss_reduce(
        [None, {"loss": 1.0}], loss_module=_plain_loss(),
        dtype=DTYPE, device=DEVICE, global_step=0,
    ) is None


def test_stitched_loss_rms_fallback_golden():
    payload = {
        "onsite_l1_sum": torch.tensor(4.0), "onsite_mse_sum": torch.tensor(16.0), "onsite_cnt": torch.tensor(2.0),
        "hopping_l1_sum": torch.tensor(3.0), "hopping_mse_sum": torch.tensor(36.0), "hopping_cnt": torch.tensor(3.0),
        "z_values": [],
    }
    out = MetricReducer.stitched_loss_reduce(
        [payload], loss_module=_plain_loss(), dtype=DTYPE, device=DEVICE, global_step=0,
    )
    onsite = 0.5 * (2.0 + math.sqrt(8.0))
    hopping = 0.5 * (1.0 + math.sqrt(12.0))
    total = 0.5 * (onsite + hopping)
    assert out.item() == pytest.approx(total, rel=1e-6)


def test_stitched_loss_aggregates_across_payloads_stats_path():
    def mk(scale):
        return {
            "onsite_l1_sum": torch.tensor(scale), "onsite_mse_sum": torch.tensor(scale),
            "onsite_cnt": torch.tensor(1.0),
            "hopping_l1_sum": torch.tensor(scale), "hopping_mse_sum": torch.tensor(scale),
            "hopping_cnt": torch.tensor(1.0), "z_values": [],
        }
    loss = _StatsLoss()
    out = MetricReducer.stitched_loss_reduce(
        [mk(1.0), mk(3.0)], loss_module=loss, dtype=DTYPE, device=DEVICE, global_step=5,
    )
    # sums aggregate: l1=4, mse=4, cnt=2 both components
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(4.0 / 2.0 + 1e-12))
    hopping = onsite
    assert out.item() == pytest.approx(0.5 * (onsite + hopping), rel=1e-6)
    assert loss.last_global_step == 5


# --------------------------------------------------------------------------
# display_state_from_packs
# --------------------------------------------------------------------------

def test_display_state_from_packs_golden():
    reduced_pack = _pack(
        loss_opt_sum=5.0, step_count=1.0, grad_norm_sum=2.0,
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
    )
    reduced_db = _db_pack()
    gathered = [ExpertDisplayMetric(
        expert_onsite=torch.tensor(1.0), expert_hopping=torch.tensor(2.0),
        grad_norm=torch.tensor(0.0), lr=torch.tensor(0.1),
        active_nodes=torch.tensor(7.0), active_edges=torch.tensor(9.0),
    ).to_tensor(dtype=DTYPE, device=DEVICE)]

    state = MetricReducer.display_state_from_packs(
        reduced_pack, reduced_db, gathered,
        total_steps=1.0, num_experts=1, rank_to_expert_idx=lambda r: 0,
        train_loss_module=_plain_loss(), supports_triplet=True,
        dtype=DTYPE, device=DEVICE, time_idx=2,
    )
    onsite = 0.5 * (2.0 + math.sqrt(8.0))
    hopping = 0.5 * (1.0 + math.sqrt(12.0))
    total = 0.5 * (onsite + hopping)
    assert state["field"] == "iteration"
    assert state["window_steps"] == 1
    assert state["train_loss"].item() == pytest.approx(total, rel=1e-6)
    assert state["train_onsite_loss"] == pytest.approx(onsite, rel=1e-6)
    assert state["train_hopping_loss"] == pytest.approx(hopping, rel=1e-6)
    assert state["train_loss_opt"].item() == pytest.approx(5.0)
    assert state["lr"] == pytest.approx(0.1)
    assert state["total_grad_norm"] == pytest.approx(2.0)
    assert state["expert_0_onsite"] == pytest.approx(1.0)
    assert state["expert_0_hopping"] == pytest.approx(2.0)
    assert state["expert_0_active_nodes"] == pytest.approx(7.0)
    assert state["expert_0_active_edges"] == pytest.approx(9.0)
    # no dynamic-batch stats present
    assert "batch_cost" not in state


def test_display_state_includes_dynamic_batch_and_oom():
    reduced_pack = _pack(loss_opt_sum=2.0, step_count=1.0)
    reduced_db = _db_pack(
        num_graphs_sum=8.0, cost_sum=100.0, num_nodes_sum=20.0,
        num_edges_sum=40.0, max_item_cost_sum=30.0, step_count=1.0,
        oom_skipped_count=3.0,
    )
    gathered = [ExpertDisplayMetric(
        expert_onsite=torch.tensor(0.0), expert_hopping=torch.tensor(0.0),
        grad_norm=torch.tensor(0.0), lr=torch.tensor(0.2),
        active_nodes=torch.tensor(1.0), active_edges=torch.tensor(1.0),
    ).to_tensor(dtype=DTYPE, device=DEVICE)]
    state = MetricReducer.display_state_from_packs(
        reduced_pack, reduced_db, gathered,
        total_steps=1.0, num_experts=1, rank_to_expert_idx=lambda r: 0,
        train_loss_module=_plain_loss(), supports_triplet=False,
        dtype=DTYPE, device=DEVICE, time_idx=1,
    )
    assert state["batch_num_graphs"] == pytest.approx(8.0)
    assert state["batch_cost"] == pytest.approx(100.0)
    assert state["batch_num_nodes"] == pytest.approx(20.0)
    assert state["batch_num_edges"] == pytest.approx(40.0)
    assert state["batch_max_item_cost"] == pytest.approx(30.0)
    assert state["dynamic_batch_oom_skipped_iters"] == 3
    # supports_triplet False -> no per-expert onsite/hopping tags
    assert "expert_0_onsite" not in state
    assert "expert_0_lr" in state


def test_display_state_pools_sparse_expert_metrics_by_fired_count():
    reduced_pack = _pack(loss_opt_sum=2.0, step_count=2.0)
    reduced_db = _db_pack()
    gathered = [
        ExpertDisplayMetric(
            expert_onsite=10.0, expert_hopping=8.0, grad_norm=0.0,
            lr=0.1, active_nodes=4.0, active_edges=6.0,
        ).to_tensor(dtype=DTYPE, device=DEVICE),
        ExpertDisplayMetric(
            expert_onsite=1.0, expert_hopping=2.0, grad_norm=0.0,
            lr=0.1, active_nodes=4.0, active_edges=6.0,
        ).to_tensor(dtype=DTYPE, device=DEVICE),
    ]
    fired = [torch.tensor([1.0, 0.0]), torch.tensor([9.0, 4.0])]

    state = MetricReducer.display_state_from_packs(
        reduced_pack,
        reduced_db,
        gathered,
        total_steps=2.0,
        num_experts=1,
        rank_to_expert_idx=lambda _rank: 0,
        train_loss_module=_plain_loss(),
        supports_triplet=True,
        dtype=DTYPE,
        device=DEVICE,
        time_idx=2,
        gathered_fired_counts=fired,
    )

    assert state["expert_0_onsite"] == pytest.approx(1.9)
    # Rank 0 did not fire hopping, so only rank 1 contributes.
    assert state["expert_0_hopping"] == pytest.approx(2.0)
    assert ExpertDisplayMetric.LENGTH == 6
