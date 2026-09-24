"""Typed metric packs (MetricPack/DynamicBatchStat/ExpertDisplayMetric) and the
pure pack-reduction functions in MetricReducer that turn them into display and
compatible-loss state.

Round-trip and slot-placement tests guard the wire layout MultiTrainer ships
through all_reduce/all_gather; the MetricReducer tests pin the numerical
behaviour of every reduction branch (stats path, RMS fallback, active-weighted
fallback, onsite-boost, z-loss, and the not-supported/empty short-circuits).
"""
import math
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.metric_pack import DynamicBatchStat, ExpertDisplayMetric, MetricPack
from dptb.nnops.metric_reducer import MetricReducer

DTYPE = torch.float32
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# MetricPack / DynamicBatchStat / ExpertDisplayMetric: wire round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MetricPack, DynamicBatchStat, ExpertDisplayMetric])
def test_to_tensor_places_each_field_in_its_slot(cls):
    names = [f.name for f in dataclass_fields(cls)]
    indices = {name: cls.index(name) for name in names}
    # every slot is covered exactly once
    assert sorted(indices.values()) == list(range(cls.LENGTH))

    for name, slot in indices.items():
        vec = cls(**{name: 3.5}).to_tensor(dtype=torch.float64, device="cpu")
        assert vec.shape == (cls.LENGTH,)
        assert float(vec[slot].item()) == 3.5, (name, slot)
        for other in range(cls.LENGTH):
            if other != slot:
                assert float(vec[other].item()) == 0.0, (name, other)


@pytest.mark.parametrize("cls", [MetricPack, DynamicBatchStat, ExpertDisplayMetric])
def test_round_trip_identity(cls):
    # Distinct nonzero value per slot so a transposition would be caught.
    values = torch.arange(1, cls.LENGTH + 1, dtype=torch.float64)
    wrapped = cls.from_tensor(values)
    for name in (f.name for f in dataclass_fields(cls)):
        assert float(getattr(wrapped, name).item()) == float(values[cls.index(name)].item()), name
    # to_tensor(from_tensor(x)) == x byte-for-byte.
    rebuilt = wrapped.to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(rebuilt, values), cls.__name__


def test_from_tensor_returns_views_not_copies():
    t = torch.zeros(MetricPack.LENGTH, dtype=torch.float64)
    mp = MetricPack.from_tensor(t)
    t[MetricPack.index("loss_opt_sum")] = 9.0
    assert float(mp.loss_opt_sum.item()) == 9.0


def test_all_none_pack_serializes_to_zeros():
    for cls in (MetricPack, DynamicBatchStat, ExpertDisplayMetric):
        vec = cls().to_tensor(dtype=torch.float32, device="cpu")
        assert torch.equal(vec, torch.zeros(cls.LENGTH, dtype=torch.float32)), cls.__name__


def test_to_tensor_honors_dtype_and_device():
    vec = MetricPack(step_count=1.0).to_tensor(dtype=torch.float64, device="cpu")
    assert vec.dtype == torch.float64
    assert vec.device.type == "cpu"


# ---------------------------------------------------------------------------
# MetricReducer: pack -> compatible/component/display state
# ---------------------------------------------------------------------------
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

    def compatible_loss_from_stats(self, *, onsite_l1_sum, onsite_mse_sum, onsite_count,
                                   hopping_l1_sum, hopping_mse_sum, hopping_count,
                                   z_loss=None, global_step=None):
        self.calls += 1
        self.last_global_step = global_step
        self.last_z_loss = z_loss
        onsite = 0.5 * (onsite_l1_sum / onsite_count.clamp_min(1.0)
                        + torch.sqrt(onsite_mse_sum / onsite_count.clamp_min(1.0) + 1e-12))
        hopping = 0.5 * (hopping_l1_sum / hopping_count.clamp_min(1.0)
                         + torch.sqrt(hopping_mse_sum / hopping_count.clamp_min(1.0) + 1e-12))
        return 0.5 * (onsite + hopping), onsite, hopping


def _plain_loss(**overrides):
    """Loss double *without* ``compatible_loss_from_stats`` -> RMS fallback."""
    base = dict(onsite_boost=False, z_loss_coef=0.0, _current_onsite_weight=None)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# scalar helpers
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("total", "count", "expected"),
    [(6.0, 3.0, 2.0), (5.0, 0.0, 5.0), (None, 3.0, 0.0), (3.0, None, 0.0)],
    ids=["divides", "zero_count_clamps_to_one", "none_total", "none_count"],
)
def test_safe_mean(total, count, expected):
    total_t = None if total is None else torch.tensor(total)
    count_t = None if count is None else torch.tensor(count)
    out = MetricReducer.safe_mean(total_t, count_t, dtype=DTYPE, device=DEVICE)
    assert out.item() == pytest.approx(expected)


def test_maybe_call_or_value():
    assert MetricReducer.maybe_call_or_value(None, default=1.5) == 1.5
    assert MetricReducer.maybe_call_or_value(2.0) == 2.0
    assert MetricReducer.maybe_call_or_value(lambda: 3.0) == 3.0

    def _boom():
        raise RuntimeError

    assert MetricReducer.maybe_call_or_value(_boom, default=7.0) == 7.0


# --------------------------------------------------------------------------
# compatible_state_from_pack
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pack_kwargs", "supports_triplet"),
    [({"onsite_cnt_sum": 1.0}, False), ({}, True)],
    ids=["not_supported", "empty_pack"],
)
def test_compatible_state_returns_none_when_not_supported_or_empty(pack_kwargs, supports_triplet):
    assert MetricReducer.compatible_state_from_pack(
        _pack(**pack_kwargs), loss_module=_plain_loss(), supports_triplet=supports_triplet,
        dtype=DTYPE, device=DEVICE,
    ) is None


def test_compatible_state_rms_fallback_golden():
    pack = _pack(onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
                hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0)
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE, prefix="train",
    )
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(16.0 / 2.0))   # 1 + sqrt(2)
    hopping = 0.5 * (3.0 / 3.0 + math.sqrt(36.0 / 3.0))  # 0.5 + sqrt(3)
    total = 0.5 * (onsite + hopping)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite, rel=1e-6)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping, rel=1e-6)
    assert state["train_loss"].item() == pytest.approx(total, rel=1e-6)


def test_compatible_state_onsite_boost_and_zloss():
    pack = _pack(onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
                hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0, z_sum=6.0, z_cnt=3.0)
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
    pack = _pack(onsite_weighted_sum=10.0, active_nodes_sum=4.0, hopping_weighted_sum=6.0, active_edges_sum=2.0)
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE,
    )
    assert state["train_onsite_loss"].item() == pytest.approx(2.5)
    assert state["train_hopping_loss"].item() == pytest.approx(3.0)
    assert state["train_loss"].item() == pytest.approx(0.5 * (2.5 + 3.0))


def test_compatible_state_stats_path_forwards_global_step_and_zloss():
    pack = _pack(onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
                hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0, z_sum=8.0, z_cnt=4.0)
    loss = _StatsLoss()
    state = MetricReducer.compatible_state_from_pack(
        pack, loss_module=loss, supports_triplet=True, dtype=DTYPE, device=DEVICE, prefix="val", global_step=42,
    )
    assert loss.calls == 1
    assert loss.last_global_step == 42
    assert loss.last_z_loss.item() == pytest.approx(2.0)  # z_sum / z_cnt
    assert set(state) == {"val_loss", "val_onsite_loss", "val_hopping_loss"}


# --------------------------------------------------------------------------
# component_state_from_pack
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pack_kwargs", "supports_triplet"),
    [({"onsite_cnt_sum": 1.0}, False), ({"loss_opt_sum": 3.0, "step_count": 1.0}, True)],
    ids=["not_supported", "cadence_skipped_zero_over_zero_pack"],
)
def test_component_state_returns_empty_when_not_supported_or_cadence_skipped(pack_kwargs, supports_triplet):
    assert MetricReducer.component_state_from_pack(
        _pack(**pack_kwargs), loss_module=_plain_loss(), supports_triplet=supports_triplet,
        dtype=DTYPE, device=DEVICE, prefix="validation",
    ) == {}


def test_component_state_uses_compatible_when_available():
    pack = _pack(
        onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
        hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0,
        # conflicting active means that must be IGNORED in favour of stats
        onsite_weighted_sum=999.0, active_nodes_sum=1.0, hopping_weighted_sum=999.0, active_edges_sum=1.0,
    )
    out = MetricReducer.component_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE, prefix="validation",
    )
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(16.0 / 2.0))
    hopping = 0.5 * (3.0 / 3.0 + math.sqrt(36.0 / 3.0))
    assert out["validation_onsite_loss"].item() == pytest.approx(onsite, rel=1e-6)
    assert out["validation_hopping_loss"].item() == pytest.approx(hopping, rel=1e-6)


def test_component_state_active_fallback_when_compatible_none():
    # Empty stats + zero active -> compatible returns None, then component falls
    # back to weighted/active (with clamp), so an all-zero pack yields zeros.
    pack = _pack(onsite_weighted_sum=8.0, active_nodes_sum=0.0, hopping_weighted_sum=4.0, active_edges_sum=0.0)
    out = MetricReducer.component_state_from_pack(
        pack, loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE, prefix="train",
    )
    # active counts clamp to 1.0
    assert out["train_onsite_loss"].item() == pytest.approx(8.0)
    assert out["train_hopping_loss"].item() == pytest.approx(4.0)


# --------------------------------------------------------------------------
# stitched_loss_reduce
# --------------------------------------------------------------------------
def test_stitched_loss_none_when_no_stats():
    assert MetricReducer.stitched_loss_reduce(
        [None, {"loss": 1.0}], loss_module=_plain_loss(), dtype=DTYPE, device=DEVICE, global_step=0,
    ) is None


def test_stitched_loss_rms_fallback_golden():
    payload = {
        "onsite_l1_sum": torch.tensor(4.0), "onsite_mse_sum": torch.tensor(16.0), "onsite_cnt": torch.tensor(2.0),
        "hopping_l1_sum": torch.tensor(3.0), "hopping_mse_sum": torch.tensor(36.0), "hopping_cnt": torch.tensor(3.0),
        "z_values": [],
    }
    out = MetricReducer.stitched_loss_reduce([payload], loss_module=_plain_loss(), dtype=DTYPE, device=DEVICE, global_step=0)
    onsite = 0.5 * (2.0 + math.sqrt(8.0))
    hopping = 0.5 * (1.0 + math.sqrt(12.0))
    total = 0.5 * (onsite + hopping)
    assert out.item() == pytest.approx(total, rel=1e-6)


def test_stitched_loss_aggregates_across_payloads_stats_path():
    def mk(scale):
        return {
            "onsite_l1_sum": torch.tensor(scale), "onsite_mse_sum": torch.tensor(scale), "onsite_cnt": torch.tensor(1.0),
            "hopping_l1_sum": torch.tensor(scale), "hopping_mse_sum": torch.tensor(scale), "hopping_cnt": torch.tensor(1.0),
            "z_values": [],
        }

    loss = _StatsLoss()
    out = MetricReducer.stitched_loss_reduce([mk(1.0), mk(3.0)], loss_module=loss, dtype=DTYPE, device=DEVICE, global_step=5)
    # sums aggregate: l1=4, mse=4, cnt=2 both components
    onsite = 0.5 * (4.0 / 2.0 + math.sqrt(4.0 / 2.0 + 1e-12))
    hopping = onsite
    assert out.item() == pytest.approx(0.5 * (onsite + hopping), rel=1e-6)
    assert loss.last_global_step == 5


# --------------------------------------------------------------------------
# display_state_from_packs
# --------------------------------------------------------------------------
def test_display_state_from_packs_golden():
    reduced_pack = _pack(loss_opt_sum=5.0, step_count=1.0, grad_norm_sum=2.0,
                         onsite_l1_sum=4.0, onsite_mse_sum=16.0, onsite_cnt_sum=2.0,
                         hopping_l1_sum=3.0, hopping_mse_sum=36.0, hopping_cnt_sum=3.0)
    reduced_db = _db_pack()
    gathered = [ExpertDisplayMetric(
        expert_onsite=torch.tensor(1.0), expert_hopping=torch.tensor(2.0),
        grad_norm=torch.tensor(0.0), lr=torch.tensor(0.1),
        active_nodes=torch.tensor(7.0), active_edges=torch.tensor(9.0),
    ).to_tensor(dtype=DTYPE, device=DEVICE)]

    state = MetricReducer.display_state_from_packs(
        reduced_pack, reduced_db, gathered, total_steps=1.0, num_experts=1, rank_to_expert_idx=lambda r: 0,
        train_loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE, time_idx=2,
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
    reduced_db = _db_pack(num_graphs_sum=8.0, cost_sum=100.0, num_nodes_sum=20.0,
                          num_edges_sum=40.0, max_item_cost_sum=30.0, step_count=1.0, oom_skipped_count=3.0)
    gathered = [ExpertDisplayMetric(
        expert_onsite=torch.tensor(0.0), expert_hopping=torch.tensor(0.0),
        grad_norm=torch.tensor(0.0), lr=torch.tensor(0.2),
        active_nodes=torch.tensor(1.0), active_edges=torch.tensor(1.0),
    ).to_tensor(dtype=DTYPE, device=DEVICE)]
    state = MetricReducer.display_state_from_packs(
        reduced_pack, reduced_db, gathered, total_steps=1.0, num_experts=1, rank_to_expert_idx=lambda r: 0,
        train_loss_module=_plain_loss(), supports_triplet=False, dtype=DTYPE, device=DEVICE, time_idx=1,
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
        ExpertDisplayMetric(expert_onsite=10.0, expert_hopping=8.0, grad_norm=0.0,
                            lr=0.1, active_nodes=4.0, active_edges=6.0).to_tensor(dtype=DTYPE, device=DEVICE),
        ExpertDisplayMetric(expert_onsite=1.0, expert_hopping=2.0, grad_norm=0.0,
                            lr=0.1, active_nodes=4.0, active_edges=6.0).to_tensor(dtype=DTYPE, device=DEVICE),
    ]
    fired = [torch.tensor([1.0, 0.0]), torch.tensor([9.0, 4.0])]

    state = MetricReducer.display_state_from_packs(
        reduced_pack, reduced_db, gathered, total_steps=2.0, num_experts=1, rank_to_expert_idx=lambda _rank: 0,
        train_loss_module=_plain_loss(), supports_triplet=True, dtype=DTYPE, device=DEVICE, time_idx=2,
        gathered_fired_counts=fired,
    )

    assert state["expert_0_onsite"] == pytest.approx(1.9)
    # Rank 0 did not fire hopping, so only rank 1 contributes.
    assert state["expert_0_hopping"] == pytest.approx(2.0)
